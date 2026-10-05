"""Episode-transactional teacher and DAgger collection with deterministic recovery."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..util import atomic_json, digest_file, digest_json, seed_for
from .evaluation import episode_metrics, validate_qualification
from .hover import FlyBodyInterface
from .imitation import source_identity
from .teacher import FlightTeacher
from .trajectories import TrajectoryCache, write_cache


def actor_identity(actor):
    if actor is None:
        return None
    weights = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        weights.update(name.encode())
        weights.update(str((array.dtype, array.shape)).encode())
        weights.update(array.tobytes())
    return digest_json(
        {
            "parameters": actor.parameter_report,
            "weights": weights.hexdigest(),
            "substrate": actor.substrate.report if actor.substrate is not None else None,
            "rate": dataclasses.asdict(actor.substrate.config)
            if actor.substrate is not None
            else None,
            "ports": actor.ports.fingerprint if actor.ports is not None else None,
        }
    )


@torch.no_grad()
def _collect(
    config,
    output,
    *,
    split,
    episodes,
    seed,
    teacher_path,
    qualification,
    actor,
    beta,
    interactions,
    smoke,
    resume,
):
    if not 0 <= beta <= 1 or episodes < 1:
        raise ValueError("Invalid collection counts or mixing coefficient")
    if actor is not None and split != "train":
        raise ValueError("DAgger may only create training data")
    if interactions is not None and interactions < 1:
        raise ValueError("Positive interaction cap required")
    if actor is None:
        torch.set_num_threads(1)
    teacher, body = FlightTeacher(teacher_path), FlyBodyInterface(config)
    try:
        if not smoke:
            validate_qualification(qualification, body, teacher)
        metadata = {
            "task": "hover",
            "split": split,
            "seed": seed,
            "body_config": dataclasses.asdict(config),
            "body_fingerprint": body.fingerprint,
            "teacher_fingerprint": teacher.fingerprint,
            "evidence": "native_hover_smoke" if smoke else "native_hover_teacher_data",
            "collection": "teacher" if actor is None else "dagger",
            "beta": beta,
            "teacher_queries_per_interaction": 1,
            "observation_schema": body.obs_schema,
            "action_names": body.action_names,
            "qualification_sha256": digest_file(qualification) if qualification else None,
        }
        identity = digest_json(
            {
                "metadata": metadata,
                "episodes": episodes,
                "interactions": interactions,
                "actor": actor_identity(actor),
                "imitation_source": source_identity(),
                "collector_sha256": digest_file(__file__),
                "teacher_and_metrics_source": {
                    name: digest_file(Path(__file__).parent / name)
                    for name in ("teacher.py", "evaluation.py")
                },
            }
        )
        progress_path = output / "collection.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if not resume or progress["identity"] != identity:
                raise ValueError("Collection exists or its code, policy, body, or budget changed")
        elif resume:
            raise FileNotFoundError("No collection journal to resume")
        else:
            progress = {
                "identity": identity,
                "episodes": 0,
                "interactions": 0,
                "inflight": None,
                "recovery_attempts": 0,
                "possible_replayed_interactions_upper_bound": 0,
            }
            atomic_json(progress_path, progress)
        if (output / "manifest.json").exists():
            result = TrajectoryCache(output)
            if result.manifest["metadata"].get("collection_identity") != identity:
                raise ValueError("Completed collection identity changed")
            return result

        rows = []
        # Adopt complete episode transactions, including one committed just before a crash.
        for child in sorted((output / "episodes").glob("episode-*")):
            if not (child / "manifest.json").exists():
                continue
            cache = TrajectoryCache(child)
            index = len(rows)
            if child.name != f"episode-{index:06d}" or cache.manifest["metadata"] != {
                "collection_identity": identity,
                "episode": index,
            }:
                raise ValueError("Episode journal is out of order or changed identity")
            for row in cache.manifest["trajectories"]:
                rows.append(dict(row, file=str(child.relative_to(output) / row["file"])))
        consumed = sum(r["steps"] for r in rows)
        if len(rows) < progress["episodes"] or consumed < progress["interactions"]:
            raise ValueError("Committed episode data is missing")
        if progress["inflight"] is not None and progress["inflight"]["episode"] == len(rows):
            progress["recovery_attempts"] += 1
            progress["possible_replayed_interactions_upper_bound"] += progress["inflight"]["cap"]
            partial = output / "episodes" / f"episode-{len(rows):06d}"
            if partial.exists():
                abandoned = (
                    output
                    / "abandoned"
                    / f"episode-{len(rows):06d}-attempt-{progress['recovery_attempts']:04d}"
                )
                abandoned.parent.mkdir(parents=True, exist_ok=True)
                partial.rename(abandoned)
        progress.update(episodes=len(rows), interactions=consumed, inflight=None)
        atomic_json(progress_path, progress)
        context = actor.context() if actor is not None else None
        device = next(actor.parameters()).device if actor is not None else "cpu"
        while (interactions is not None and consumed < interactions) or (
            interactions is None and len(rows) < episodes
        ):
            index = len(rows)
            cap = (
                min(config.horizon, interactions - consumed)
                if interactions is not None
                else config.horizon
            )
            progress["inflight"] = {"episode": index, "cap": cap}
            atomic_json(progress_path, progress)
            case_seed = seed_for(seed, f"{split}-episode:{index}")
            rng = np.random.default_rng(seed_for(seed, f"{split}-mixing:{index}"))
            observation = body.reset(case_seed, split)
            state = actor.reset(1) if actor is not None else None
            obs, labels, actions, rewards, terminals, truncations = (
                [observation],
                [],
                [],
                [],
                [],
                [],
            )
            for step in range(cap):
                expert = teacher.action(body)
                if actor is None:
                    executed = expert
                else:
                    predicted, state = actor(
                        torch.as_tensor(observation, device=device)[None], state, context
                    )
                    executed = expert if rng.random() < beta else predicted[0].cpu().numpy()
                observation, reward, terminal, truncated, info = body.step(executed)
                obs.append(observation)
                labels.append(expert)
                actions.append(executed)
                rewards.append(reward)
                terminals.append(terminal)
                truncations.append(truncated or step == cap - 1)
                if terminal or truncated:
                    break
            child = output / "episodes" / f"episode-{index:06d}"
            cache = write_cache(
                child,
                [
                    {
                        "observations": obs,
                        "actions": labels,
                        "executed_actions": actions,
                        "rewards": rewards,
                        "terminated": terminals,
                        "truncated": truncations,
                        "scenario": body.case,
                        "metrics": episode_metrics(body, info),
                    }
                ],
                {"collection_identity": identity, "episode": index},
            )
            row = cache.manifest["trajectories"][0]
            rows.append(dict(row, file=str(child.relative_to(output) / row["file"])))
            consumed += row["steps"]
            progress.update(episodes=len(rows), interactions=consumed, inflight=None)
            atomic_json(progress_path, progress)
        if interactions is not None and consumed != interactions:
            raise ValueError("Collection missed its exact interaction budget")
        metadata.update(
            collection_identity=identity,
            recovery={
                key: progress[key]
                for key in ("recovery_attempts", "possible_replayed_interactions_upper_bound")
            },
            experience_definition="Committed teacher-labeled transitions; interrupted uncommitted episodes replay deterministically and extra simulator work is bounded separately",
        )
        manifest = {
            "schema": "connectome-trajectories-v1",
            "metadata": metadata,
            "observation_dim": body.obs_dim,
            "action_dim": body.action_dim,
            "trajectories": rows,
            "interactions": consumed,
        }
        manifest["fingerprint"] = digest_json(manifest)
        atomic_json(output / "manifest.json", manifest)
        return TrajectoryCache(output)
    finally:
        body.close()


def collect_resumable(
    config,
    output,
    *,
    split="train",
    episodes=32,
    seed=1000,
    teacher_path=None,
    qualification=None,
    actor=None,
    beta=0.0,
    interactions=None,
    smoke=False,
    resume=False,
):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Collection already exists; use --resume with identical settings")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker owns this trajectory collection") from exc
        return _collect(
            config,
            output,
            split=split,
            episodes=episodes,
            seed=seed,
            teacher_path=teacher_path,
            qualification=qualification,
            actor=actor,
            beta=beta,
            interactions=interactions,
            smoke=smoke,
            resume=resume,
        )
