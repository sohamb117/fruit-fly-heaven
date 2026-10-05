"""Closed-loop native evaluation and teacher-labeled trajectory collection."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from ..util import atomic_json, digest_file, seed_for
from .hover import FlyBodyInterface, HoverConfig
from .teacher import FlightTeacher
from .trajectories import write_cache


def episode_metrics(body, info):
    result = dict(info)
    for key in ("position_error", "orientation_error", "control_magnitude"):
        result[f"mean_{key}"] = float(np.mean([m[key] for m in body.metric_history]))
    result["steps"] = body.steps
    return result


def summarize(episodes):
    keys = (
        "success",
        "score",
        "survival",
        "mean_position_error",
        "mean_orientation_error",
        "mean_control_magnitude",
        "numerical_failure",
    )
    return {key: float(np.mean([r[key] for r in episodes])) for key in keys} | {
        "episodes": episodes,
        "evaluation_interactions": sum(r["steps"] for r in episodes),
    }


@torch.no_grad()
def evaluate_controller(
    body, *, seeds, split="validation", actor=None, teacher=None, intervention="none"
):
    if actor is not None and teacher is not None:
        raise ValueError("Student evaluation never queries or executes a teacher")
    episodes = []
    context = actor.context() if actor is not None else None
    device = next(actor.parameters()).device if actor is not None else "cpu"
    for seed in seeds:
        observation = body.reset(int(seed), split)
        state = actor.reset(1) if actor is not None else None
        while True:
            if actor is not None:
                command, state = actor(
                    torch.as_tensor(observation, device=device)[None], state, context, intervention
                )
                action = command[0].cpu().numpy()
            elif teacher is not None:
                action = teacher.action(body)
            else:
                action = np.zeros(body.action_dim, dtype=np.float32)
            observation, _, terminated, truncated, info = body.step(action)
            if terminated or truncated:
                episodes.append(episode_metrics(body, info) | {"seed": int(seed), "split": split})
                break
    return summarize(episodes)


def qualify_teacher(
    config: HoverConfig, output, episodes=8, seed=910000, teacher_path=None, smoke=False
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Teacher qualification reports are immutable")
    if not smoke and (episodes < 8 or config.horizon * config.control_dt < 0.5):
        raise ValueError("Scientific qualification needs >=8 episodes lasting >=0.5 seconds")
    torch.set_num_threads(1)
    teacher = FlightTeacher(teacher_path)
    conversion = json.loads((teacher.path / "conversion-check.json").read_text())
    if conversion["status"] != "passed" or conversion["teacher_fingerprint"] != teacher.fingerprint:
        raise ValueError("Teacher conversion has not passed verification")
    body = FlyBodyInterface(config)
    try:
        seeds = [seed_for(seed, f"teacher-qualification:{i}") for i in range(episodes)]
        trained = evaluate_controller(body, seeds=seeds, teacher=teacher)
        zero = evaluate_controller(body, seeds=seeds)
        qualified = not smoke and trained["success"] >= 0.875 and zero["success"] < 0.5
        report = {
            "status": "qualified" if qualified else "not_qualified",
            "qualified": qualified,
            "evidence": "native_hover_smoke" if smoke else "native_hover_teacher_qualification",
            "body_config": dataclasses.asdict(config),
            "body_fingerprint": body.fingerprint,
            "teacher_fingerprint": teacher.fingerprint,
            "teacher": trained,
            "wpg_zero_modulation": zero,
            "thresholds": {"teacher_success": 0.875, "zero_modulation_success_strictly_below": 0.5},
            "source_sha256": digest_file(__file__),
        }
        atomic_json(output, report)
        return report
    finally:
        body.close()


def validate_qualification(path, body, teacher):
    report = json.loads(Path(path).read_text())
    if (
        not report.get("qualified")
        or report.get("evidence") != "native_hover_teacher_qualification"
    ):
        raise ValueError("Hover teacher has not qualified; scientific data/training are gated")
    if (
        report["body_fingerprint"] != body.fingerprint
        or report["teacher_fingerprint"] != teacher.fingerprint
    ):
        raise ValueError("Teacher qualification does not match this body or teacher")
    return report


@torch.no_grad()
def collect_cache(
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
):
    if not 0 <= beta <= 1 or episodes < 1:
        raise ValueError("Invalid collection counts or mixing coefficient")
    if actor is not None and split != "train":
        raise ValueError("DAgger may only create training data")
    torch.set_num_threads(1)
    teacher, body = FlightTeacher(teacher_path), FlyBodyInterface(config)
    try:
        if not smoke:
            validate_qualification(qualification, body, teacher)
        if interactions is not None and interactions < 1:
            raise ValueError("Positive interaction cap required")
        context = actor.context() if actor is not None else None
        device = next(actor.parameters()).device if actor is not None else "cpu"
        rng = np.random.default_rng(seed_for(seed, "dagger-mixing"))
        consumed = 0

        def trajectories():
            nonlocal consumed
            episode = 0
            while (interactions is not None and consumed < interactions) or (
                interactions is None and episode < episodes
            ):
                case_seed = seed_for(seed, f"{split}-episode:{episode}")
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
                while True:
                    expert = teacher.action(body)
                    if actor is None:
                        executed = expert
                    else:
                        predicted, state = actor(
                            torch.as_tensor(observation, device=device)[None], state, context
                        )
                        executed = expert if rng.random() < beta else predicted[0].cpu().numpy()
                    observation, reward, terminal, truncated, info = body.step(executed)
                    consumed += 1
                    budget_end = interactions is not None and consumed == interactions
                    obs.append(observation)
                    labels.append(expert)
                    actions.append(executed)
                    rewards.append(reward)
                    terminals.append(terminal)
                    truncations.append(truncated or budget_end)
                    if terminal or truncated or budget_end:
                        break
                yield {
                    "observations": obs,
                    "actions": labels,
                    "executed_actions": actions,
                    "rewards": rewards,
                    "terminated": terminals,
                    "truncated": truncations,
                    "scenario": body.case,
                    "metrics": episode_metrics(body, info),
                }
                episode += 1

        return write_cache(
            output,
            trajectories(),
            {
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
            },
        )
    finally:
        body.close()
