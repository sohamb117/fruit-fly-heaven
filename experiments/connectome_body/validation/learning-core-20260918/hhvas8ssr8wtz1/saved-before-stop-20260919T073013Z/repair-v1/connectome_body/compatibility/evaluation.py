"""Paired held-out episodes and causal interventions on persistent neural activity."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..util import digest_file, seed_for

INTERVENTIONS = (
    "none",
    "zero",
    "temporal_mean",
    "time_shuffle",
    "neuron_permutation",
    "reset",
    "targeted_lesion",
    "remove_edges",
    "random_graph",
)


@dataclass(frozen=True)
class InterventionSpec:
    kind: str = "none"
    seed: int = 0
    history: int = 256
    lesion_file: str | None = None
    mean_file: str | None = None

    def validate(self):
        if self.kind not in INTERVENTIONS or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Unknown intervention or invalid intervention seed")
        if type(self.history) is not int or self.history < 2:
            raise ValueError("Time shuffling needs a history of at least two states")
        if (self.kind == "targeted_lesion") != bool(self.lesion_file):
            raise ValueError("Targeted lesions require a pinned neuronal selection")
        if (self.kind == "temporal_mean") != bool(self.mean_file):
            raise ValueError("Temporal mean intervention requires development-only calibration")


class StateIntervention:
    def __init__(self, controller, spec, policy_identity=None, body_fingerprint=None):
        spec.validate()
        if spec.kind != "none" and not controller.state_dim:
            raise ValueError("Substrate interventions cannot be applied to an adapter-only policy")
        self.spec, self.controller = spec, controller
        self.device = next(controller.parameters()).device
        self.report = asdict(spec)
        self.lesion_mask, self.mean = None, None
        if spec.kind == "targeted_lesion":
            if controller.topology is None:
                raise ValueError("Anatomical lesions require a connectome")
            selection = json.loads(Path(spec.lesion_file).read_text())
            if selection.get(
                "graph_fingerprint"
            ) != controller.graph_fingerprint or not selection.get("provenance"):
                raise ValueError(
                    "Lesion selection lacks the graph identity or anatomical provenance"
                )
            lookup = {str(node): i for i, node in enumerate(controller.topology.node_ids)}
            ids = selection["node_ids"]
            if not ids or len(ids) != len(set(ids)) or any(str(node) not in lookup for node in ids):
                raise ValueError("Invalid lesion neuron IDs")
            self.lesion_mask = torch.zeros(
                controller.state_dim, dtype=torch.bool, device=self.device
            )
            self.lesion_mask[[lookup[str(node)] for node in ids]] = True
            self.report.update(neurons=len(ids), selection_sha256=digest_file(spec.lesion_file))
        if spec.kind == "temporal_mean":
            with np.load(spec.mean_file, allow_pickle=False) as record:
                metadata = json.loads(str(record["metadata"]))
                mean = record["mean"].copy()
            if (
                metadata.get("split") not in ("train", "validation")
                or not policy_identity
                or metadata.get("policy_identity") != policy_identity
                or metadata.get("body_fingerprint") != body_fingerprint
            ):
                raise ValueError("Activity means must come from this policy's development episodes")
            self.mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32)
            if (
                self.mean.shape != (controller.state_dim,)
                or not torch.isfinite(self.mean).all()
                or metadata.get("mean_sha256") != hashlib.sha256(mean.tobytes()).hexdigest()
            ):
                raise ValueError("Invalid calibrated neural activity mean")
            self.report.update(calibration=metadata, calibration_sha256=digest_file(spec.mean_file))
        if spec.kind == "remove_edges" and controller.topology is None:
            raise ValueError("Edge removal requires a graph substrate")
        if spec.kind == "random_graph":
            raise ValueError("Use the acute graph replacement evaluator for random_graph")
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(seed_for(self.spec.seed, "neural-intervention"))
        self.history = deque(maxlen=self.spec.history)
        self.permutation = torch.as_tensor(
            self.rng.permutation(self.controller.state_dim), device=self.device
        )

    def transform(self, state):
        if self.spec.kind == "zero":
            return torch.zeros_like(state)
        if self.spec.kind == "temporal_mean":
            return self.mean.expand_as(state)
        if self.spec.kind == "neuron_permutation":
            return state[:, self.permutation]
        if self.spec.kind == "time_shuffle":
            # Causal delayed-state null: sample only past states, never future
            # activity from the held-out trajectory. The first step is intact.
            result = (
                self.history[int(self.rng.integers(len(self.history)))] if self.history else state
            )
            self.history.append(state.detach().clone())
            return result
        return state

    def action(self, observation, state, context):
        if self.spec.kind == "reset":
            state = torch.zeros_like(state)
        return self.controller(
            observation,
            state,
            context,
            state_transform=self.transform,
            lesion_mask=self.lesion_mask,
            remove_edges=self.spec.kind == "remove_edges",
        )


@torch.no_grad()
def evaluate(
    controller,
    body,
    seed,
    episodes,
    split="validation",
    *,
    intervention=None,
    policy_identity=None,
    episode_seeds=None,
    state_observer=None,
):
    if type(episodes) is not int or episodes < 1:
        raise ValueError("Evaluation requires at least one episode")
    if episode_seeds is not None and len(episode_seeds) != episodes:
        raise ValueError("Explicit evaluation seeds must cover every episode")
    intervention = StateIntervention(
        controller, intervention or InterventionSpec(), policy_identity, body.fingerprint
    )
    device = next(controller.parameters()).device
    started, records = time.monotonic(), []
    training = controller.training
    controller.eval()
    try:
        context = controller.context()
        for episode in range(episodes):
            episode_seed = (
                episode_seeds[episode]
                if episode_seeds is not None
                else seed_for(seed, f"paper-evaluation-{split}-{episode}")
            )
            obs = body.reset(episode_seed, split)
            state = controller.reset(1)
            intervention.reset()
            for _ in range(body.config.horizon):
                action, state = intervention.action(
                    torch.as_tensor(obs, device=device).unsqueeze(0), state, context
                )
                if not torch.isfinite(action).all() or not torch.isfinite(state).all():
                    raise FloatingPointError(
                        "Nonfinite evaluated controller; do not drop failed seeds"
                    )
                if state_observer is not None:
                    state_observer(state)
                obs, _, terminated, truncated, info = body.step(action[0].cpu().numpy())
                if terminated or truncated:
                    break
            records.append({"seed": int(episode_seed), **info})
    finally:
        controller.train(training)
    return {
        "split": split,
        "intervention": intervention.report,
        "episodes": records,
        "success_rate": float(np.mean([row["success"] for row in records])),
        "mean_score": float(np.mean([row["score"] for row in records])),
        "mean_return": float(np.mean([row["episode_return"] for row in records])),
        "numerical_failure_rate": float(np.mean([row["numerical_failure"] for row in records])),
        "interactions": sum(row["steps"] for row in records),
        "wall_seconds": time.monotonic() - started,
    }
