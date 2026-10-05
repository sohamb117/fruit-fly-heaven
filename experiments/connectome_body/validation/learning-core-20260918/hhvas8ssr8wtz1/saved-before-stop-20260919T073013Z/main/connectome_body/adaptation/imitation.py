"""Paired offline imitation with bounded BPTT, accounting, and exact update resumption."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..graphs import Graph
from ..util import (
    atomic_json,
    atomic_torch_save,
    digest_file,
    digest_json,
    restore_rng,
    rng_state,
    seed_everything,
)
from . import PROTOCOL
from .models import GenericAdapter
from .substrates import RateConfig
from .trajectories import TrajectoryCache, assert_disjoint_scenarios, assert_paired_schema


@dataclass(frozen=True)
class Optimization:
    updates: int = 2000
    batch_size: int = 4
    sequence_length: int = 32
    burn_in: int = 64
    learning_rate: float = 0.001
    eval_every: int = 100
    checkpoint_every: int = 25
    max_grad_norm: float = 1.0

    def validate(self):
        if (
            min(
                self.updates,
                self.batch_size,
                self.sequence_length,
                self.eval_every,
                self.checkpoint_every,
            )
            < 1
            or self.burn_in < 0
        ):
            raise ValueError("Invalid imitation counts")
        if min(self.learning_rate, self.max_grad_norm) <= 0:
            raise ValueError("Invalid optimizer scale")


@dataclass(frozen=True)
class AdapterSpec:
    graph: str | None = None
    variant: str = "real"
    budget: int = 5000
    channels: int = 16
    support: int = 256
    seed: int = 0
    port_seed: int = 0
    device: str = "cpu"
    threads: int = 1
    rate: RateConfig = field(default_factory=RateConfig)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["rate"] = RateConfig(**value.get("rate", {}))
        result = cls(**value)
        result.rate.validate()
        if (
            result.device not in ("cpu", "cuda", "mps")
            or min(result.budget, result.channels, result.support, result.threads) < 1
        ):
            raise ValueError("Invalid actor resource configuration")
        return result


def source_identity():
    root = Path(__file__).parent
    names = (
        "__init__.py",
        "features.py",
        "imitation.py",
        "models.py",
        "nulls.py",
        "ports.py",
        "substrates.py",
        "trajectories.py",
    )
    files = {name: digest_file(root / name) for name in names}
    for name in ("graphs.py", "policy.py", "util.py", "body.py", "metal_sparse.py"):
        files[f"../{name}"] = digest_file(root.parent / name)
    return digest_json(
        {
            "files": files,
            "python": platform.python_version(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scipy", "numba")
            },
        }
    )


def build_adapter(spec: AdapterSpec, obs_dim, action_dim):
    if spec.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    if spec.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS requested but unavailable; no fallback")
    seed_everything(spec.seed, spec.threads)
    graph = None if spec.variant in ("adapter_only", "trainable_gru") else Graph.load(spec.graph)
    actor = GenericAdapter(
        obs_dim,
        action_dim,
        spec.budget,
        spec.channels,
        spec.support,
        graph,
        spec.variant,
        spec.rate,
        spec.port_seed,
        backend=spec.device,
    ).to(spec.device)
    return actor, graph


@torch.no_grad()
def evaluate_cache(actor, cache, device, batch_size=4):
    if batch_size < 1:
        raise ValueError("Positive evaluation batch size required")
    actor.eval()
    context = actor.context()
    accumulator_dtype = torch.float32 if device == "mps" else torch.float64
    sums = torch.zeros(3, dtype=accumulator_dtype, device=device)
    for start in range(0, len(cache), batch_size):
        items = [cache.load(idx) for idx in range(start, min(start + batch_size, len(cache)))]
        length = max(len(item["actions"]) for item in items)
        observations = np.zeros((length, len(items), actor.obs_dim), dtype=np.float32)
        targets = np.zeros((length, len(items), actor.action_dim), dtype=np.float32)
        masks = np.zeros((length, len(items)), dtype=np.float32)
        for index, item in enumerate(items):
            count = len(item["actions"])
            observations[:count, index] = item["observations"][:-1]
            targets[:count, index] = item["actions"]
            masks[:count, index] = item["mask"]
        observations, targets, masks = [
            torch.as_tensor(v, device=device) for v in (observations, targets, masks)
        ]
        state = actor.reset(len(items))
        for observation, target, mask in zip(observations, targets, masks):
            action, state = actor(observation, state, context)
            sums[0] += ((action - target).square().mean(-1).to(accumulator_dtype) * mask).sum()
            sums[1] += (
                ((action >= 0) == (target >= 0)).float().mean(-1).to(accumulator_dtype) * mask
            ).sum()
            sums[2] += mask.sum()
    squared, correct, denominator = sums.cpu().tolist()
    return {
        "mse": squared / denominator,
        "sign_accuracy": correct / denominator,
        "labeled_steps": int(denominator),
    }


def _sample_batch(caches, rng, opt):
    entries = [(cache, index) for cache in caches for index in range(len(cache))]
    selected = []
    for _ in range(opt.batch_size):
        cache, index = entries[int(rng.integers(len(entries)))]
        item = cache.load(index)
        target = int(rng.choice(np.flatnonzero(item["mask"])))
        start = target // opt.sequence_length * opt.sequence_length
        end = min(start + opt.sequence_length, len(item["actions"]))
        burn = max(0, start - opt.burn_in)
        selected.append((item, burn, start, end))
    return selected


def train_offline(
    spec,
    opt,
    train_caches,
    validation_cache,
    output,
    *,
    resume=False,
    initial_checkpoint=None,
    stop_after=None,
):
    opt.validate()
    output = Path(output)
    caches = [TrajectoryCache(path) for path in train_caches]
    validation = TrajectoryCache(validation_cache)
    assert_paired_schema(caches + [validation])
    for cache in caches:
        if cache.manifest["metadata"]["split"] != "train":
            raise ValueError("Only training trajectories may enter the optimizer")
        assert_disjoint_scenarios(cache, validation)
    if validation.manifest["metadata"]["split"] != "validation":
        raise ValueError("Checkpoint selection requires the validation split")
    if output.exists() and not resume:
        raise FileExistsError(f"Run exists: {output}")
    if resume and not (output / "latest.pt").exists():
        raise FileNotFoundError("No complete checkpoint to resume")
    output.mkdir(parents=True, exist_ok=True)
    actor, graph = build_adapter(
        spec, caches[0].manifest["observation_dim"], caches[0].manifest["action_dim"]
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=opt.learning_rate)
    manifest = {
        "protocol": PROTOCOL,
        "source_identity": source_identity(),
        "adapter": dataclasses.asdict(spec),
        "optimization": dataclasses.asdict(opt),
        "parameter_report": actor.parameter_report,
        "graph_fingerprint": graph.fingerprint if graph else None,
        "training_caches": [c.fingerprint for c in caches],
        "validation_cache": validation.fingerprint,
        "initial_checkpoint": digest_file(initial_checkpoint) if initial_checkpoint else None,
        "data_metadata": caches[0].manifest["metadata"],
    }
    manifest["identity"] = digest_json(manifest)
    rng = np.random.default_rng(spec.seed)
    updates, presentations, burn_presentations, elapsed = 0, 0, 0, 0.0
    curve, losses = [], []
    best, best_step, best_actor = float("inf"), 0, None
    if initial_checkpoint and not resume:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if initial["parameter_report"] != actor.parameter_report:
            raise ValueError("Initial controller architecture differs")
        parent = json.loads((Path(initial_checkpoint).parent / "manifest.json").read_text())
        if initial["identity"] != parent["identity"] or any(
            parent[key] != manifest[key]
            for key in ("source_identity", "adapter", "graph_fingerprint")
        ):
            raise ValueError("Initial controller code, settings, or graph differs")
        for key in ("body_fingerprint", "teacher_fingerprint", "task"):
            if parent["data_metadata"].get(key) != manifest["data_metadata"].get(key):
                raise ValueError("Initial controller belongs to a different task or teacher")
        actor.load_state_dict(initial["actor"])
    if resume:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != manifest["identity"]:
            raise ValueError("Resume identity changed (code, data, graph, settings, or versions)")
        actor.load_state_dict(saved["actor"])
        optimizer.load_state_dict(saved["optimizer"])
        updates, presentations = saved["updates"], saved["sample_presentations"]
        burn_presentations, elapsed = saved["burn_presentations"], saved["elapsed_seconds"]
        curve, losses = saved["curve"], saved["losses"]
        best, best_step, best_actor = saved["best"], saved["best_step"], saved["best_actor"]
        rng.bit_generator.state = saved["sample_rng"]
        restore_rng(saved["rng"])
    atomic_json(output / "manifest.json", manifest)
    started = time.monotonic()

    def evaluate():
        nonlocal best, best_step, best_actor
        metrics = evaluate_cache(actor, validation, spec.device, opt.batch_size)
        curve.append({"updates": updates, "sample_presentations": presentations, **metrics})
        if metrics["mse"] < best:
            best, best_step = metrics["mse"], updates
            best_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}

    def save():
        value = {
            "identity": manifest["identity"],
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(),
            "sample_rng": rng.bit_generator.state,
            "updates": updates,
            "sample_presentations": presentations,
            "burn_presentations": burn_presentations,
            "elapsed_seconds": elapsed + time.monotonic() - started,
            "curve": curve,
            "losses": losses,
            "best": best,
            "best_step": best_step,
            "best_actor": best_actor,
            "parameter_report": actor.parameter_report,
        }
        atomic_torch_save(output / "latest.pt", value)
        atomic_torch_save(
            output / "best.pt",
            {
                "actor": best_actor,
                "identity": manifest["identity"],
                "updates": best_step,
                "parameter_report": actor.parameter_report,
            },
        )
        atomic_json(output / "learning_curve.json", curve)
        atomic_json(output / "losses.json", losses)

    if not curve:
        evaluate()
        save()
    while updates < opt.updates:
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        context = actor.context()
        loss, count = next(actor.parameters()).new_zeros(()), 0
        batch = _sample_batch(caches, rng, opt)
        # Group windows have different lengths; each episode has its own reset.
        # Sparse operations are vectorized across the batch using left-padded burn-in.
        burns = max(start - burn for _, burn, start, _ in batch)
        state = actor.reset(len(batch))
        with torch.no_grad():
            for offset in range(burns):
                observations, active = [], []
                for item, burn, start, _ in batch:
                    t = start - burns + offset
                    valid = t >= burn
                    observations.append(
                        item["observations"][t] if valid else np.zeros(actor.obs_dim, np.float32)
                    )
                    active.append(valid)
                _, candidate = actor(
                    torch.as_tensor(np.stack(observations), device=spec.device), state, context
                )
                state = torch.where(
                    torch.as_tensor(active, device=spec.device)[:, None], candidate, state
                )
                burn_presentations += sum(active)
        for offset in range(max(end - start for _, _, start, end in batch)):
            observations, targets, masks = [], [], []
            for item, _, start, end in batch:
                t = start + offset
                valid = t < end
                observations.append(
                    item["observations"][t] if valid else np.zeros(actor.obs_dim, np.float32)
                )
                targets.append(
                    item["actions"][t] if valid else np.zeros(actor.action_dim, np.float32)
                )
                masks.append(float(item["mask"][t]) if valid else 0.0)
                presentations += int(valid)
            predicted, state = actor(
                torch.as_tensor(np.stack(observations), device=spec.device), state, context
            )
            target = torch.as_tensor(np.stack(targets), device=spec.device)
            mask = torch.as_tensor(masks, device=spec.device)
            loss = loss + ((predicted - target).square().mean(-1) * mask).sum()
            count += sum(masks)
        loss = loss / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite imitation loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(actor.parameters(), opt.max_grad_norm)
        if not torch.isfinite(gradient):
            raise FloatingPointError("Nonfinite imitation gradient")
        optimizer.step()
        updates += 1
        losses.append(
            {"updates": updates, "mse": float(loss.detach()), "gradient_norm": float(gradient)}
        )
        if updates % opt.eval_every == 0 or updates == opt.updates:
            evaluate()
        if updates % opt.checkpoint_every == 0 or updates == opt.updates:
            save()
        if stop_after is not None and updates >= stop_after and updates < opt.updates:
            save()
            return {"status": "paused", "updates": updates}
    save()  # Also repairs derived files after a final-budget crash.
    result = {
        "status": "complete",
        "identity": manifest["identity"],
        "updates": updates,
        "selected_update": best_step,
        "validation_mse": best,
        "sample_presentations": presentations,
        "burn_presentations": burn_presentations,
        "unique_training_interactions": sum(c.manifest["interactions"] for c in caches),
        "new_environment_interactions": 0,
        "elapsed_seconds": elapsed + time.monotonic() - started,
        "parameter_report": actor.parameter_report,
    }
    atomic_json(output / "result.json", result)
    return result


def load_actor(run, checkpoint="best.pt"):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if digest_json({k: v for k, v in manifest.items() if k != "identity"}) != manifest["identity"]:
        raise ValueError("Saved actor manifest changed")
    if manifest["source_identity"] != source_identity():
        raise ValueError(
            "Evaluation source differs from training; use the archived experiment code"
        )
    spec = AdapterSpec.from_dict(manifest["adapter"])
    report = manifest["parameter_report"]
    saved = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
    if saved["identity"] != manifest["identity"]:
        raise ValueError("Checkpoint identity differs from run")
    # Recover dimensions from encoder/decoder checkpoint tensors without guessing.
    if spec.variant == "trainable_gru":
        obs_dim = saved["actor"]["gru.weight_ih"].shape[1]
        action_dim = saved["actor"]["decoder.bias"].shape[0]
    else:
        obs_dim = saved["actor"]["encoder.0.weight"].shape[1]
        action_dim = saved["actor"]["decoder.2.bias"].shape[0]
    actor, graph = build_adapter(spec, obs_dim, action_dim)
    if (graph.fingerprint if graph else None) != manifest["graph_fingerprint"]:
        raise ValueError("Saved actor graph differs from the current graph")
    if actor.parameter_report != report:
        raise ValueError("Actor capacity accounting changed")
    actor.load_state_dict(saved["actor"])
    actor.eval()
    return actor, spec, manifest
