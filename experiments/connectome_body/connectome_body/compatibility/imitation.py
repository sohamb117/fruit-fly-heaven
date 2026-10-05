"""Shared expert trajectories and resumable sequence BC for the paper Controller.

Experts consume precisely the student's current observation; their private state
never enters the dataset. Rollouts, manifests, and optimization stages are sealed.
"""

from __future__ import annotations

import fcntl
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..adaptation.teacher import FlightTeacher
from ..util import (
    atomic_json,
    atomic_torch_save,
    digest_file,
    digest_json,
    restore_rng,
    rng_state,
    seed_everything,
    seed_for,
)
from .bodies import make_body
from .controller import Controller
from .evaluation import evaluate
from .runtime import hardware_profile
from .training import controller_identity, source_identity


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 5
    batch_size: int = 2
    sequence_length: int = 32
    learning_rate: float = 0.001
    max_grad_norm: float = 0.5
    standardize_actions: bool = False
    initialize_action_mean: bool = False

    def validate(self):
        for name in ("standardize_actions", "initialize_action_mean"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("epochs", "batch_size", "sequence_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Positive integer {name} required")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Positive finite {name} required")


class ObservationExpert:
    """External reference controller; action() never receives a simulator object."""

    def __init__(self, body, worm_frequency=0.4, worm_amplitude=0.8):
        self.name = body.study_spec.name if hasattr(body, "study_spec") else body.spec.name
        self.dt = body.control_dt
        self.frequency, self.amplitude = worm_frequency, worm_amplitude
        self.phase = 0.0
        if self.name == "fly":
            self.policy = FlightTeacher()
            self.fields, offset = {}, 0
            for field in body.obs_schema:
                size = int(np.prod(field["shape"]))
                self.fields[field["name"]] = (slice(offset, offset + size), field["scale"])
                offset += size
            self.low, self.high = body.low.copy(), body.high.copy()
            self.identity = {
                "kind": "published_external_flybody_mlp",
                "policy": self.policy.fingerprint,
                "input": "student observation, including clipping; fixed inverse unit scaling only",
            }
        elif self.name == "worm":
            self.identity = {
                "kind": "external_worm_wave_with_heading_feedback",
                "frequency": self.frequency,
                "amplitude": self.amplitude,
                "heading_feedback_gain": -0.3,
                "input": "student egocentric target direction; private oscillator state",
            }
        else:
            raise ValueError("No qualified observation-only expert for this body")
        self.identity["implementation_sha256"] = digest_file(__file__)

    def reset(self, seed):
        self.phase = 0.0

    @torch.no_grad()
    def action(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if self.name == "fly":
            native_obs = np.concatenate(
                [observation[self.fields[k][0]] * self.fields[k][1] for k in sorted(self.fields)]
            )
            native = self.policy(torch.from_numpy(native_obs)[None])[0].numpy()
            return np.clip(2 * (native - self.low) / (self.high - self.low) - 1, -1, 1).astype(
                np.float32
            )
        self.phase += 2 * np.pi * self.frequency * self.dt
        drive = self.amplitude * np.sin(self.phase - 2 * np.pi * np.linspace(0, 1, 24))
        drive[:8] += np.clip(-0.3 * observation[147], -0.3, 0.3)
        return (2 * np.clip(np.r_[np.maximum(drive, 0), np.maximum(-drive, 0)], 0, 1) - 1).astype(
            np.float32
        )


def load_demonstrations(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != "compatibility-demonstrations-v1" or manifest.get(
        "fingerprint"
    ) != digest_json({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("Expert dataset manifest seal mismatch")
    if digest_file(path / "trajectories.npz") != manifest["arrays_sha256"]:
        raise ValueError("Expert dataset arrays changed")
    with np.load(path / "trajectories.npz", allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    for split in ("train", "validation"):
        x, a, mask = (arrays[f"{split}_{key}"] for key in ("obs", "actions", "mask"))
        if (
            x.ndim != 3
            or a.ndim != 3
            or x.shape[:2] != a.shape[:2]
            or mask.shape != x.shape[:2]
            or mask.dtype != bool
        ):
            raise ValueError("Malformed expert sequence shapes")
        if x.shape[-1] != manifest["observation_dim"] or a.shape[-1] != manifest["action_dim"]:
            raise ValueError("Expert interface dimensions changed")
        if not np.isfinite(x).all() or not np.isfinite(a).all() or np.abs(a).max() > 1:
            raise ValueError("Nonfinite or unnormalized expert data")
        if not mask[:, 0].all() or np.any(np.diff(mask.astype(int), axis=1) > 0):
            raise ValueError("Demonstrations must start at reset with contiguous valid prefixes")
    if set(manifest["seeds"]["train"]) & set(manifest["seeds"]["validation"]):
        raise ValueError("Expert data splits overlap")
    return manifest, arrays


def build_demonstrations(
    body_spec,
    output,
    *,
    train_episodes=16,
    validation_episodes=4,
    seed=0,
    min_success=0.75,
    purpose="experiment",
    expert_options=None,
):
    # Tiny per-step expert MLPs become much slower with a large BLAS thread pool.
    # Dataset generation runs in its own process, separate from student workers.
    torch.set_num_threads(1)
    if purpose not in ("experiment", "smoke") or not 0 <= min_success <= 1:
        raise ValueError("Invalid expert qualification")
    if min(train_episodes, validation_episodes) < 1:
        raise ValueError("Need separate training and validation trajectories")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        body = make_body(body_spec)
        try:
            expert = ObservationExpert(body, **(expert_options or {}))
            request = {
                "body_fingerprint": body.fingerprint,
                "teacher": expert.identity,
                "train_episodes": train_episodes,
                "validation_episodes": validation_episodes,
                "seed": seed,
                "min_success": min_success,
                "purpose": purpose,
            }
            if (output / "manifest.json").exists():
                saved, _ = load_demonstrations(output)
                if saved["request"] != request:
                    raise ValueError("Immutable dataset request changed")
                return saved
            started = time.monotonic()
            arrays = {}
            records = {}
            seeds = {}
            for split, count in [("train", train_episodes), ("validation", validation_episodes)]:
                x = np.zeros((count, body_spec.horizon, body.obs_dim), np.float32)
                a = np.zeros((count, body_spec.horizon, body.action_dim), np.float32)
                mask = np.zeros((count, body_spec.horizon), bool)
                rows = []
                seeds[split] = []
                for ep in range(count):
                    s = seed_for(seed, f"expert-{body_spec.name}-{body_spec.task}-{split}-{ep}")
                    seeds[split].append(s)
                    o = body.reset(s, split)
                    expert.reset(s)
                    for t in range(body_spec.horizon):
                        action = expert.action(o)
                        x[ep, t] = o
                        a[ep, t] = action
                        mask[ep, t] = True
                        o, _, term, trunc, info = body.step(action)
                        if term or trunc:
                            break
                    rows.append({"seed": s, **info})
                arrays.update({f"{split}_obs": x, f"{split}_actions": a, f"{split}_mask": mask})
                records[split] = rows
            tmp = output / "trajectories.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            tmp.replace(output / "trajectories.npz")
            success = float(np.mean([x["success"] for x in records["validation"]]))
            qualified = success >= min_success and not any(
                x["numerical_failure"] for rows in records.values() for x in rows
            )
            manifest = {
                "schema": "compatibility-demonstrations-v1",
                "request": request,
                "body_fingerprint": body.fingerprint,
                "body": asdict(body_spec),
                "observation_dim": body.obs_dim,
                "action_dim": body.action_dim,
                "observation_schema": body.obs_schema,
                "action_names": body.action_names,
                "teacher": expert.identity,
                "seeds": seeds,
                "episodes": records,
                "qualified": qualified,
                "qualification_success_rate": success,
                "teacher_environment_interactions": sum(
                    x["steps"] for rows in records.values() for x in rows
                ),
                "unique_train_samples": int(arrays["train_mask"].sum()),
                "generation_wall_seconds": time.monotonic() - started,
                "arrays_sha256": digest_file(output / "trajectories.npz"),
            }
            manifest["fingerprint"] = digest_json(manifest)
            atomic_json(output / "manifest.json", manifest)
            return manifest
        finally:
            body.close()


@torch.no_grad()
def action_error(actor, arrays, split, batch_size):
    device = next(actor.parameters()).device
    was = actor.training
    actor.eval()
    total = count = 0
    try:
        context = actor.context()
        for i in range(0, len(arrays[f"{split}_obs"]), batch_size):
            x = torch.as_tensor(arrays[f"{split}_obs"][i : i + batch_size], device=device)
            a = torch.as_tensor(arrays[f"{split}_actions"][i : i + batch_size], device=device)
            mask = torch.as_tensor(arrays[f"{split}_mask"][i : i + batch_size], device=device)
            state = actor.reset(len(x))
            for t in range(x.shape[1]):
                if not mask[:, t].any():
                    break
                pred, state = actor(x[:, t], state, context)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("Nonfinite BC validation")
                total += float(((pred - a[:, t]) ** 2)[mask[:, t]].sum())
                count += int(mask[:, t].sum()) * a.shape[-1]
        return total / max(1, count)
    finally:
        actor.train(was)


def training_statistics(arrays):
    """Fit fixed preprocessing on valid TRAINING samples, never validation/test."""
    mask = arrays["train_mask"]
    obs = arrays["train_obs"][mask].astype(np.float64)
    actions = arrays["train_actions"][mask].astype(np.float64)
    if not len(obs):
        raise ValueError("No valid training samples for normalization")
    return {
        "observation_mean": obs.mean(0).astype(np.float32),
        "observation_scale": np.maximum(obs.std(0), 0.01).astype(np.float32),
        "action_mean": actions.mean(0).astype(np.float32),
        "action_scale": np.maximum(actions.std(0), 0.01).astype(np.float32),
    }


def train_bc(spec, bc, dataset, output, *, resume=False, max_seconds=None, stop_after_updates=None):
    spec.validate()
    bc.validate()
    if max_seconds is not None and (max_seconds <= 0 or not math.isfinite(max_seconds)):
        raise ValueError("Positive finite time budget required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, arrays = load_demonstrations(dataset)
        if not data["qualified"] and spec.purpose != "smoke":
            raise ValueError("Expert failed task qualification")
        body = make_body(spec.body)
        try:
            if body.fingerprint != data["body_fingerprint"]:
                raise ValueError("Expert/student body mismatch")
            seed_everything(seed_for(spec.train_seed, "paper-actor"), spec.threads)
            actor = Controller(body.obs_dim, body.action_dim, spec.controller).to(spec.device)
            stats = training_statistics(arrays)
            if spec.controller.normalize_observations:
                for name in ("observation_mean", "observation_scale"):
                    getattr(actor, name).copy_(torch.as_tensor(stats[name], device=spec.device))
            if spec.controller.normalize_actions:
                action_mean = torch.as_tensor(stats["action_mean"], device=spec.device)
                actor.action_center.copy_(torch.atanh(action_mean.clamp(-0.999, 0.999)))
                valid_actions = arrays["train_actions"][arrays["train_mask"]].astype(np.float64)
                raw_scale = np.arctanh(valid_actions.clip(-0.999, 0.999)).std(0)
                actor.action_scale.copy_(
                    torch.as_tensor(np.maximum(raw_scale, 0.01), device=spec.device)
                )
            if bc.initialize_action_mean:
                layer = list(actor.decoder.modules())[-1]
                if not isinstance(layer, torch.nn.Linear) or not layer.bias.requires_grad:
                    raise ValueError(
                        "Mean-action initialization requires a trainable linear decoder"
                    )
                with torch.no_grad():
                    layer.weight.zero_()
                    if spec.controller.normalize_actions:
                        layer.bias.zero_()
                    else:
                        layer.bias.copy_(
                            torch.atanh(
                                torch.as_tensor(stats["action_mean"], device=spec.device).clamp(
                                    -0.999, 0.999
                                )
                            )
                        )
            action_scale = torch.as_tensor(
                stats["action_scale"] if bc.standardize_actions else np.ones(body.action_dim),
                dtype=torch.float32,
                device=spec.device,
            )
            constant_mse = float(
                np.mean(
                    (arrays["validation_actions"][arrays["validation_mask"]] - stats["action_mean"])
                    ** 2
                )
            )
            if (
                spec.metadata.get("expected_graph_fingerprint", actor.graph_fingerprint)
                != actor.graph_fingerprint
            ):
                raise ValueError("BC graph changed since planning")
            ci = controller_identity(actor)
            manifest = {
                "schema": "compatibility-imitation-run-v1",
                "config": spec.to_dict(),
                "bc": asdict(bc),
                "dataset_fingerprint": data["fingerprint"],
                "controller_identity": ci,
                "body_fingerprint": body.fingerprint,
                "action_names": body.action_names,
                "code_fingerprint": source_identity(),
                "hardware": hardware_profile(spec.device, spec.threads),
                "parameters": actor.parameter_report,
                "expert_data": {
                    k: data[k]
                    for k in (
                        "unique_train_samples",
                        "teacher_environment_interactions",
                        "generation_wall_seconds",
                    )
                },
                "state_protocol": "Reset at trajectory start; carry detached state across contiguous TBPTT chunks; checkpoint chunk cursor and state",
                "preprocessing": {
                    "fit_split": "train",
                    "statistics": {k: v.tolist() for k, v in stats.items()},
                    "observation_normalization": spec.controller.normalize_observations,
                    "action_parameterization": spec.controller.normalize_actions,
                    "standardized_action_loss": bc.standardize_actions,
                    "constant_action_validation_mse": constant_mse,
                },
                "selection": "minimum held-out expert action MSE; no test selection",
            }
            identity = digest_json(manifest)
            manifest["identity"] = identity
            if (output / "manifest.json").exists() and not resume:
                raise FileExistsError("BC run exists; resume explicitly")
            optimizer = torch.optim.Adam(
                [p for p in actor.parameters() if p.requires_grad], lr=bc.learning_rate
            )
            cursor = {
                "epoch": 0,
                "batch": 0,
                "time": 0,
                "updates": 0,
                "exposures": 0,
                "wall_seconds": 0.0,
                "optimization_seconds": 0.0,
                "curve": [],
                "best_error": None,
                "best_actor": None,
                "state": None,
            }
            if resume:
                saved = torch.load(
                    output / "latest.pt", map_location=spec.device, weights_only=False
                )
                if saved["identity"] != identity:
                    raise ValueError("BC resume identity mismatch")
                actor.load_state_dict(saved["actor"])
                optimizer.load_state_dict(saved["optimizer"])
                cursor = saved["cursor"]
                restore_rng(saved["rng"])
                if (output / "result.json").exists():
                    return json.loads((output / "result.json").read_text())
            else:
                atomic_json(output / "manifest.json", manifest)
            started = time.monotonic()
            prior = cursor["wall_seconds"]
            n, length = arrays["train_obs"].shape[:2]

            def save():
                if spec.device == "cuda":
                    torch.cuda.synchronize()
                cursor["wall_seconds"] = prior + time.monotonic() - started
                atomic_torch_save(
                    output / "latest.pt",
                    {
                        "schema": "compatibility-imitation-checkpoint-v1",
                        "identity": identity,
                        "actor": actor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "cursor": cursor,
                        "rng": rng_state(),
                    },
                )
                if cursor["best_actor"] is not None:
                    atomic_torch_save(
                        output / "best.pt",
                        {
                            "schema": "compatibility-imitation-policy-v1",
                            "identity": identity,
                            "actor": cursor["best_actor"],
                            "controller_identity": ci,
                            "expert_exposures": cursor["exposures"],
                        },
                    )
                atomic_json(output / "learning_curve.json", cursor["curve"])

            def validate():
                error = action_error(actor, arrays, "validation", bc.batch_size)
                closed = evaluate(
                    actor, body, spec.train_seed, spec.training.eval_episodes, "validation"
                )
                row = {
                    **closed,
                    "expert_action_mse": error,
                    "constant_action_validation_mse": constant_mse,
                    "expert_action_mse_over_constant": error / max(constant_mse, 1e-12),
                    "expert_exposures": cursor["exposures"],
                    "optimizer_steps": cursor["updates"],
                    "optimization_seconds": cursor["optimization_seconds"],
                    "wall_seconds": prior + time.monotonic() - started,
                    "evaluation_wall_seconds": closed["wall_seconds"],
                }
                cursor["curve"].append(row)
                if cursor["best_error"] is None or error < cursor["best_error"]:
                    cursor["best_error"] = error
                    cursor["best_actor"] = {
                        k: v.detach().cpu().clone() for k, v in actor.state_dict().items()
                    }

            if not cursor["curve"]:
                validate()
                save()
            while cursor["epoch"] < bc.epochs:
                order = np.random.default_rng(
                    seed_for(spec.train_seed, f"bc-epoch-{cursor['epoch']}")
                ).permutation(n)
                ids = order[cursor["batch"] : cursor["batch"] + bc.batch_size]
                x = torch.as_tensor(arrays["train_obs"][ids], device=spec.device)
                y = torch.as_tensor(arrays["train_actions"][ids], device=spec.device)
                mask = torch.as_tensor(arrays["train_mask"][ids], device=spec.device)
                start = cursor["time"]
                stop = min(length, start + bc.sequence_length)
                state = actor.reset(len(ids)) if start == 0 else cursor["state"].to(spec.device)
                if spec.device == "cuda":
                    torch.cuda.synchronize()
                tick = time.monotonic()
                context = actor.context()
                pred = []
                for t in range(start, stop):
                    a, state = actor(x[:, t], state, context)
                    pred.append(a)
                prediction = torch.stack(pred, 1)
                valid = mask[:, start:stop]
                if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
                    raise FloatingPointError("Nonfinite BC state/action")
                if valid.any():
                    loss = torch.mean(
                        ((prediction[valid] - y[:, start:stop][valid]) / action_scale) ** 2
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), bc.max_grad_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                    cursor["updates"] += 1
                    cursor["exposures"] += int(valid.sum())
                if spec.device == "cuda":
                    torch.cuda.synchronize()
                cursor["optimization_seconds"] += time.monotonic() - tick
                cursor["state"] = state.detach().cpu()
                cursor["time"] = stop
                if stop == length:
                    cursor["batch"] += len(ids)
                    cursor["time"] = 0
                    cursor["state"] = None
                    if cursor["batch"] >= n:
                        cursor["epoch"] += 1
                        cursor["batch"] = 0
                        validate()
                save()
                print(
                    json.dumps(
                        {
                            "stage": "bc",
                            "updates": cursor["updates"],
                            "epoch": cursor["epoch"],
                            "expert_exposures": cursor["exposures"],
                        }
                    ),
                    flush=True,
                )
                if cursor["epoch"] < bc.epochs and (
                    (max_seconds is not None and time.monotonic() - started >= max_seconds)
                    or (stop_after_updates and cursor["updates"] >= stop_after_updates)
                ):
                    return {
                        "status": "paused_at_checkpoint",
                        "identity": identity,
                        "expert_exposures": cursor["exposures"],
                    }
            actor.load_state_dict(cursor["best_actor"])
            closed = {
                s: evaluate(actor, body, spec.train_seed, spec.training.test_episodes, s)
                for s in ("test", "ood")
            }
            result = {
                "schema": "compatibility-imitation-result-v1",
                "status": "complete",
                "identity": identity,
                "manifest_sha256": digest_json(manifest),
                "expert_action_validation_mse": cursor["best_error"],
                "expert_exposures": cursor["exposures"],
                "optimizer_steps": cursor["updates"],
                "unique_expert_samples": data["unique_train_samples"],
                "teacher_environment_interactions": data["teacher_environment_interactions"],
                "optimization_seconds": cursor["optimization_seconds"],
                "wall_seconds": prior + time.monotonic() - started,
                "selected_checkpoint_evaluation": closed,
                "selected_checkpoint_sha256": digest_file(output / "best.pt"),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if spec.device == "cuda"
                else None,
            }
            atomic_json(output / "result.json", result)
            return result
        finally:
            body.close()
