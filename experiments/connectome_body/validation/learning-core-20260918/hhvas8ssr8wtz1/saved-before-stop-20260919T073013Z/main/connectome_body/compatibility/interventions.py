"""Persisted causal evaluations; development-only activity calibration and acute rewiring."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..graphs import Graph, transform_graph
from ..util import atomic_json, digest_file, digest_json
from .dynamics import SparseRateCore
from .evaluation import InterventionSpec, evaluate
from .training import policy_digest, source_identity


@contextmanager
def acute_random_graph(controller, seed):
    """Replace wiring while keeping trained interfaces, neuron IDs, and weight multiset."""
    original = controller.core
    if not isinstance(original, SparseRateCore):
        raise ValueError("Acute matched-graph replacement requires a connectome")
    topology = controller.topology
    magnitudes = (
        (
            original.initial_magnitudes
            if original.raw_magnitudes is None
            else F.softplus(original.raw_magnitudes).clamp_min(1e-12)
        )
        .detach()
        .cpu()
        .numpy()
    )
    signs = original.source_signs.detach().cpu().numpy()
    graph = Graph(
        topology.node_ids,
        original.src.cpu().numpy(),
        original.dst.cpu().numpy(),
        magnitudes,
        signs,
        {"fingerprint": controller.graph_fingerprint},
    )
    src, dst, weights, report = transform_graph(graph, "matched_random", signs, seed)
    replacement = SparseRateCore(graph.n, src, dst, weights, signs, original.config).to(
        original.source_signs.device
    )
    controller.core = replacement
    try:
        yield {
            "kind": "random_graph",
            "seed": seed,
            "null": report,
            "trained_interfaces_retained": True,
            "weight_magnitudes_retained": True,
            "normalization_recomputed": True,
            "without_retraining": True,
        }
    finally:
        controller.core = original


@torch.no_grad()
def calibrate_temporal_mean(controller, body, output, *, seed=0, episodes=10, split="validation"):
    if split not in ("train", "validation"):
        raise ValueError("Activity calibration cannot use held-out test or OOD episodes")
    if not controller.state_dim:
        raise ValueError("An adapter-only controller has no temporal neural activity")
    if type(episodes) is not int or episodes < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Calibration needs a positive episode count and nonnegative integer seed")
    request = {
        "schema": "neural-activity-mean-v1",
        "policy_identity": policy_digest(controller),
        "body_fingerprint": body.fingerprint,
        "code_fingerprint": source_identity(),
        "split": split,
        "seed": seed,
        "episodes": episodes,
    }
    output = Path(output)
    if output.exists():
        # A worker can stop after the atomic NPZ commit but before its job
        # result is committed. Reuse only this exact validated calibration.
        with np.load(output, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            mean = saved["mean"].copy()
        if any(metadata.get(key) != value for key, value in request.items()):
            raise ValueError("Existing immutable activity calibration belongs to another request")
        if (
            mean.shape != (controller.state_dim,)
            or not np.isfinite(mean).all()
            or metadata.get("neural_states", 0) < 1
            or metadata.get("mean_sha256") != hashlib.sha256(mean.tobytes()).hexdigest()
        ):
            raise ValueError("Existing activity calibration is invalid or changed")
        return {**metadata, "sha256": digest_file(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    total = torch.zeros(
        controller.state_dim, dtype=torch.float64, device=next(controller.parameters()).device
    )
    count = 0

    def observe(state):
        nonlocal count
        total.add_(state.sum(0, dtype=torch.float64))
        count += len(state)

    result = evaluate(controller, body, seed, episodes, split, state_observer=observe)
    mean = (total / count).float().cpu().numpy()
    metadata = {
        **request,
        "neural_states": count,
        "environment_interactions": result["interactions"],
        "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                mean=mean,
                metadata=np.array(json.dumps(metadata, sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {**metadata, "sha256": digest_file(output)}


def evaluate_intervention(
    controller, body, spec: InterventionSpec, *, seed=0, episodes=100, split="test", output=None
):
    spec.validate()
    identity = policy_digest(controller)
    declared = asdict(spec)
    for key in ("mean_file", "lesion_file"):
        if declared[key]:
            declared[key] = digest_file(declared[key])
    evaluation_identity = digest_json(
        {
            "policy": identity,
            "body": body.fingerprint,
            "intervention": declared,
            "seed": seed,
            "episodes": episodes,
            "split": split,
            "code_fingerprint": source_identity(),
        }
    )
    if output is not None and Path(output).exists():
        existing = json.loads(Path(output).read_text())
        if existing.get("evaluation_identity") != evaluation_identity or existing.get(
            "fingerprint"
        ) != digest_json({key: value for key, value in existing.items() if key != "fingerprint"}):
            raise ValueError("Existing causal result belongs to another policy or evaluation")
        return existing
    if spec.kind == "random_graph":
        with acute_random_graph(controller, spec.seed) as report:
            result = evaluate(controller, body, seed, episodes, split)
            result["intervention"] = report
    else:
        result = evaluate(
            controller, body, seed, episodes, split, intervention=spec, policy_identity=identity
        )
    if policy_digest(controller) != identity:
        raise RuntimeError("Post-training intervention mutated the trained policy")
    result.update(
        schema="compatibility-causal-evaluation-v1",
        evaluation_identity=evaluation_identity,
        policy_identity=identity,
        body_fingerprint=body.fingerprint,
        code_fingerprint=source_identity(),
    )
    result["fingerprint"] = digest_json(result)
    if output is not None:
        atomic_json(output, result)
    return result
