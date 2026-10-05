"""Measured sparse/plastic transitions and empirical conventional-RNN compute controls."""

from __future__ import annotations

import gc
import json
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ..policy import Critic
from ..util import atomic_json, seed_everything
from .adapters import count_parameters
from .config import SubstrateConfig
from .controller import Controller
from .run_config import RunSpec
from .runtime import hardware_profile
from .training import controller_identity, source_identity


def profile_controller(spec: RunSpec, observation_dim, action_dim, *, repeats=5, warmup=2):
    if min(repeats, warmup) < 1:
        raise ValueError("Warmup and repeated synchronized measurements are required")
    spec.validate()
    hardware = hardware_profile(spec.device, spec.threads)
    code = source_identity()
    seed_everything(spec.train_seed, spec.threads)
    device = torch.device(spec.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    controller = Controller(observation_dim, action_dim, spec.controller).to(device)
    critic = Critic(observation_dim, spec.training.critic_width).to(device)
    params = [p for p in controller.parameters() if p.requires_grad] + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=spec.training.learning_rate, eps=1e-5)
    batch, length, burn = (
        spec.training.num_envs,
        spec.training.sequence_length,
        spec.training.burn_in,
    )
    # Identical seeded observations/actions for every substrate. This is a
    # compute benchmark, not an embodied learning result or imitation dataset.
    generator = torch.Generator(device="cpu").manual_seed(1029)
    observations = torch.randn(length + burn, batch, observation_dim, generator=generator).to(
        device
    )
    targets = torch.randn(length, batch, action_dim, generator=generator).to(device)
    sync()
    setup_seconds = time.perf_counter() - started
    timings = []
    try:
        for index in range(warmup + repeats):
            if device.type == "cuda" and index == warmup:
                torch.cuda.reset_peak_memory_stats(device)
            sync()
            begin = time.perf_counter()
            with torch.no_grad():
                state, context = controller.reset(batch), controller.context()
                for observation in observations[:length]:
                    _, state = controller(observation, state, context)
            sync()
            collected = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            state = controller.reset(batch)
            with torch.no_grad():
                context = controller.context()
                for observation in observations[:burn]:
                    _, state = controller(observation, state, context)
            context = controller.context()
            loss = torch.zeros((), device=device)
            for t in range(length):
                action, state = controller(observations[t + burn], state, context, squash=False)
                value = critic(observations[t + burn])
                loss = (
                    loss
                    + (action - targets[t]).square().mean() / length
                    + value.square().mean() / length
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, spec.training.max_grad_norm)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite benchmark gradients")
            optimizer.step()
            sync()
            optimized = time.perf_counter()
            if index >= warmup:
                collection = collected - begin
                optimization = optimized - collected
                timings.append(
                    {
                        "collection_seconds": collection,
                        "optimization_seconds": optimization,
                        "policy_compute_seconds_per_interaction": (
                            collection + spec.training.epochs * optimization
                        )
                        / (batch * length),
                    }
                )
        samples = np.array([row["policy_compute_seconds_per_interaction"] for row in timings])
        record = {
            "schema": "compatibility-controller-benchmark-v1",
            "status": "measured",
            "hardware": hardware,
            "code_fingerprint": code,
            "controller_identity": controller_identity(controller),
            "graph_fingerprint": controller.graph_fingerprint,
            "controller": spec.controller.to_dict(),
            "parameters": controller.parameter_report,
            "critic_parameters": count_parameters(critic),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "learning": asdict(spec.training),
            "setup_seconds": setup_seconds,
            "warmup": warmup,
            "repeats": repeats,
            "batch": batch,
            "sequence_length": length,
            "burn_in": burn,
            "timings": timings,
            "median_policy_compute_seconds_per_interaction": float(np.median(samples)),
            "timing_coefficient_of_variation": float(samples.std() / samples.mean()),
            "forward_transitions_per_second": batch
            * length
            / float(np.median([x["collection_seconds"] for x in timings])),
            "optimized_transitions_per_second": batch
            * length
            / float(np.median([x["optimization_seconds"] for x in timings])),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "process_lifetime_max_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if hardware["platform"] == "Darwin" else 1024)
            ),
            "cpu_memory_note": "Process lifetime high-water mark; use a fresh process for independently comparable CPU peaks",
            "scope": "policy collection plus repeated TBPTT/Adam, including ports, edge normalization and a common critic; native physics excluded",
            "gradient_objective": "finite synthetic regression surrogate with the same recurrent autograd structure; not a learning result",
        }
    finally:
        del optimizer, controller, critic, params
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


def calibrate_rnn_match(
    condition, body_record, output, *, hidden_sizes, tolerance=0.2, repeats=5, warmup=2
):
    if (
        condition["status"] != "ready"
        or condition["config"]["controller"]["substrate"]["kind"] != "connectome"
    ):
        raise ValueError("Compute calibration requires a ready biological reference")
    if (
        not hidden_sizes
        or any(type(size) is not int or size < 1 for size in hidden_sizes)
        or not 0 < tolerance < 1
    ):
        raise ValueError("Declare positive recurrent widths and a relative matching tolerance")
    path = Path(output)
    record = {
        "schema": "compatibility-compute-matches-v1",
        "code_fingerprint": source_identity(),
        "matches": {},
    }
    if path.exists():
        record = json.loads(path.read_text())
        if (
            record.get("schema") != "compatibility-compute-matches-v1"
            or record.get("code_fingerprint") != source_identity()
        ):
            raise ValueError("Compute calibration artifact belongs to different code")
    if condition["id"] in record["matches"]:
        raise FileExistsError(
            "Reference already calibrated; keep the previous measurements and use another output"
        )
    source = RunSpec.from_dict(condition["config"])
    obs, act = body_record["observation_dim"], body_record["action_dim"]
    reference = profile_controller(source, obs, act, repeats=repeats, warmup=warmup)
    measurement = {
        "status": "in_progress",
        "reference": reference,
        "candidates": [],
        "tolerance": tolerance,
    }
    record["matches"][condition["id"]] = measurement
    atomic_json(path, record)
    # Compute matching holds the adapter family/bottleneck and optimizer fixed;
    # the number of recurrent parameters is free and separately reported.
    for hidden in sorted(set(hidden_sizes)):
        config = replace(
            source.controller,
            substrate=SubstrateConfig(
                kind="rnn",
                plasticity="joint",
                hidden_size=hidden,
                dynamics=source.controller.substrate.dynamics,
                seed=source.controller.substrate.seed,
            ),
        )
        try:
            result = profile_controller(
                replace(source, controller=config), obs, act, repeats=repeats, warmup=warmup
            )
            result["hidden_size"] = hidden
            result["relative_error"] = abs(
                result["median_policy_compute_seconds_per_interaction"]
                / reference["median_policy_compute_seconds_per_interaction"]
                - 1
            )
        except (RuntimeError, FloatingPointError) as exc:
            result = {"status": "failed", "hidden_size": hidden, "reason": str(exc)}
            gc.collect()
            if source.device == "cuda":
                torch.cuda.empty_cache()
        measurement["candidates"].append(result)
        atomic_json(path, record)
    measured = [item for item in measurement["candidates"] if item["status"] == "measured"]
    best = min(measured, key=lambda item: item["relative_error"]) if measured else None
    stable = (
        best is not None
        and max(
            reference["timing_coefficient_of_variation"], best["timing_coefficient_of_variation"]
        )
        <= tolerance
    )
    measurement.update(
        status="matched" if stable and best["relative_error"] <= tolerance else "no_match",
        device=source.device,
        hardware=reference["hardware"],
        adapter_budget=source.controller.adapter.budget,
        hidden_size=best["hidden_size"] if best else None,
        relative_error=best["relative_error"] if best else None,
        timing_stable=stable,
    )
    atomic_json(path, record)
    return measurement
