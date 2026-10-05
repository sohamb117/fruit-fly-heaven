"""Seeded, causal temporal diagnostics with explicit target masks and provenance."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..util import atomic_json, digest_file, digest_json, seed_for

TEMPORAL_TASKS = (
    "linear_memory",
    "impulse_recall",
    "match_sample",
    "temporal_xor",
    "parity",
    "running_sum",
    "leaky_integration",
    "multi_timescale",
    "narma10",
    "narma20",
    "narma30",
    "mackey_glass",
    "lorenz",
    "frequency",
    "phase",
    "rhythm",
    "missing_step",
    "change_point",
    "context_integration",
    "selective_memory",
    "temporal_order",
    "variable_delay",
    "denoising",
    "system_identification",
    "inverse_dynamics",
)
CORE_TEMPORAL = (
    "linear_memory",
    "temporal_xor",
    "narma10",
    "multi_timescale",
    "frequency",
    "context_integration",
    "mackey_glass",
    "change_point",
)
CLASSIFICATION = {
    "match_sample",
    "temporal_xor",
    "parity",
    "frequency",
    "phase",
    "change_point",
    "temporal_order",
    "variable_delay",
}


@dataclass(frozen=True)
class TemporalSpec:
    task: str = "linear_memory"
    length: int = 512
    train_sequences: int = 64
    validation_sequences: int = 16
    test_sequences: int = 32
    seed: int = 0
    delay: int = 8
    forecast: int = 10
    warmup: int = 64
    delays: tuple = (1, 2, 4, 8, 16, 32)
    timescales: tuple = (2, 8, 32)
    noise: float = 0.15

    def validate(self):
        if self.task not in TEMPORAL_TASKS:
            raise ValueError("Unknown temporal task")
        for name in (
            "length",
            "train_sequences",
            "validation_sequences",
            "test_sequences",
            "delay",
            "forecast",
            "warmup",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Positive integer {name} required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative dataset seed required")
        for name in ("delays", "timescales"):
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or any(type(x) is not int or x < 1 for x in values)
            ):
                raise ValueError(f"Distinct positive {name} required")
        if self.length < self.warmup + max(
            2 * self.delay + 8, max(self.delays), max(self.timescales), 32
        ):
            raise ValueError("Sequence too short for the declared memory and warmup")
        if not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError("Finite nonnegative noise required")


def _lorenz(rng, count):
    state = rng.normal(0, 1, 3) + [1, 1, 20]

    def f(s):
        x, y, z = s
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 * z / 3])

    values = []
    for i in range(1000 + count * 5):
        dt = 0.01
        a = f(state)
        b = f(state + dt * a / 2)
        c = f(state + dt * b / 2)
        d = f(state + dt * c)
        state += dt * (a + 2 * b + 2 * c + d) / 6
        if i >= 1000 and (i - 1000) % 5 == 0:
            values.append(state.copy())
    return np.array(values) / 20


def _mackey(rng, count):
    # Euler dt=.1, beta=.2, gamma=.1, n=10, delay=17; sample interval=1.
    lag, burn, dt = 170, 2000, 0.1
    x = np.full(lag + burn + count * 10 + 1, 1.2)
    x[:lag] += rng.normal(0, 0.05, lag)
    for t in range(lag, len(x) - 1):
        delayed = x[t - lag]
        x[t + 1] = x[t] + dt * (0.2 * delayed / (1 + delayed**10) - 0.1 * x[t])
    return x[lag + burn : lag + burn + count * 10 : 10, None]


def sequence(spec, split, index):
    spec.validate()
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown temporal split")
    rng = np.random.default_rng(seed_for(spec.seed, f"temporal-{spec.task}-{split}-{index}"))
    n, task, d = spec.length, spec.task, spec.delay
    x = rng.uniform(-1, 1, (n, 1))
    y, mask = np.zeros((n, 1)), np.ones((n, 1), bool)
    metadata = {
        "task": task,
        "split": split,
        "sequence": index,
        "classification": task in CLASSIFICATION,
        "baseline": "train_mean",
    }
    baseline = None
    if task == "linear_memory":
        y = np.zeros((n, len(spec.delays)))
        for k, delay in enumerate(spec.delays):
            y[delay:, k] = x[:-delay, 0]
        mask = np.ones_like(y, bool)
        mask[: max(spec.delays)] = False
        metadata["output_delays"] = list(spec.delays)
    elif task in ("temporal_xor", "parity"):
        x = rng.integers(0, 2, (n, 1)).astype(float)
        if task == "temporal_xor":
            y[d:, 0] = np.logical_xor(x[d:, 0], x[:-d, 0])
        else:
            for t in range(d, n):
                y[t, 0] = int(x[t - d + 1 : t + 1].sum()) % 2
        mask[:d] = False
    elif task in ("running_sum", "leaky_integration", "multi_timescale", "context_integration"):
        taus = list(spec.timescales) if task == "multi_timescale" else [d]
        y = np.zeros((n, len(taus)))
        if task == "context_integration":
            x = np.c_[
                rng.uniform(-1, 1, (n, 2)), np.repeat(rng.integers(0, 2, (n + 15) // 16), 16)[:n]
            ]
            u = x[np.arange(n), x[:, 2].astype(int)]
        else:
            u = x[:, 0]
        if task == "running_sum":
            y[:, 0] = np.convolve(u, np.ones(d) / np.sqrt(d), mode="full")[:n]
        else:
            for k, tau in enumerate(taus):
                alpha = np.exp(-1 / tau)
                for t in range(n):
                    y[t, k] = (alpha * y[t - 1, k] if t else 0) + (1 - alpha) * u[t]
        mask = np.ones_like(y, bool)
        metadata["timescales"] = taus
    elif task.startswith("narma"):
        order = int(task[5:])
        u = rng.uniform(0, 0.5, n + 200)
        output = np.zeros_like(u)
        # NARMA10 standard; 20/30 are explicitly order-scaled feedback variants.
        coefficients = [0.3, 0.5 / order, 1.5, 0.1]
        for t in range(order, len(u) - 1):
            output[t + 1] = (
                0.3 * output[t]
                + (0.5 / order) * output[t] * output[t - order + 1 : t + 1].sum()
                + 1.5 * u[t - order + 1] * u[t]
                + 0.1
            )
            if not np.isfinite(output[t + 1]) or abs(output[t + 1]) > 1e6:
                raise FloatingPointError("NARMA diverged; no clipping or replacement")
        x, y = u[199 : 199 + n, None], output[200 : 200 + n, None]
        metadata.update(
            order=order,
            coefficients=coefficients,
            convention="y[t+1] from u[t]; beta=0.5/order; no target feedback",
        )
    elif task in ("mackey_glass", "lorenz"):
        values = (_mackey if task == "mackey_glass" else _lorenz)(rng, n + spec.forecast)
        x, y = values[:n], values[spec.forecast :]
        mask = np.ones_like(y, bool)
        baseline = x.copy()
        metadata.update(forecast_horizon=spec.forecast, baseline="persistence")
    elif task in ("frequency", "phase", "rhythm", "missing_step", "denoising"):
        t = np.arange(n)
        label = int(rng.integers(2))
        freq = rng.uniform(0.015, 0.035) if label == 0 else rng.uniform(0.07, 0.10)
        phase = rng.uniform(-np.pi, np.pi)
        amplitude = rng.uniform(0.5, 1.5)
        signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
        if task == "frequency":
            x, y = signal[:, None], np.full((n, 1), label)
            mask[: max(64, spec.warmup)] = False
        elif task == "phase":
            reference = np.sin(2 * np.pi * 0.04 * t + phase)
            delta = np.pi / 2 if label else -np.pi / 2
            x = np.c_[amplitude * np.sin(2 * np.pi * 0.04 * t + phase + delta), reference]
            y[:] = label
        elif task == "rhythm":
            cutoff = n // 2
            x, y = signal[:, None].copy(), signal[:, None]
            x[cutoff:] = 0
            mask[:cutoff] = False
            metadata["cue_ends"] = cutoff
        elif task == "missing_step":
            missing = rng.random(n) < 0.2
            x = np.c_[signal * (~missing), ~missing]
            y[:, 0], mask[:, 0] = signal, missing
        else:
            x = (signal + rng.normal(0, spec.noise, n))[:, None]
            y, baseline = signal[:, None], x.copy()
            metadata["baseline"] = "noisy_observation"
    elif task == "change_point":
        # Unannounced change, independently sampled sign/mean; target stays on
        # for four samples. Current amplitude alone does not identify a change.
        mean = rng.uniform(-0.7, 0.7)
        x[:] = 0
        events = []
        next_event = int(rng.integers(32, 65))
        for t in range(n):
            if t == next_event:
                jump = float(rng.choice([-1, 1]) * rng.uniform(0.35, 0.7))
                # Reflect at the boundary rather than labeling a clipped,
                # physically unchanged mean as a detectable event.
                mean = mean + jump if abs(mean + jump) <= 1.5 else mean - jump
                events.append(t)
                y[t : min(n, t + 4)] = 1
                next_event += int(rng.integers(32, 65))
            x[t, 0] = mean + rng.normal(0, max(0.1, spec.noise))
        metadata["events"] = events
    elif task in ("system_identification", "inverse_dynamics"):
        a, b = (
            (rng.uniform(0.2, 0.9), rng.uniform(0.5, 1.5))
            if task == "system_identification"
            else (0.7, 0.5)
        )
        u, state = rng.uniform(-1, 1, n), np.zeros(n)
        for t in range(1, n):
            state[t] = a * state[t - 1] + b * u[t - 1]
        if task == "system_identification":
            x, y = np.c_[u, state], np.tile([a, b], (n, 1))
            mask = np.ones_like(y, bool)
        else:
            x, y = state[:, None], np.r_[0, u[:-1]][:, None]
    else:
        # Independent, explicitly cued trials within a sequence. Only query/recall
        # windows are scored, except event timing which needs negative windows.
        block = 2 * d + 8
        x = np.zeros((n, 3))
        mask[:] = False
        for start in range(0, n - block + 1, block):
            sample = rng.uniform(-1, 1)
            query = start + d + 3
            x[start, :2] = [sample, 1]
            if task == "impulse_recall":
                x[start, 0] = sample
                query = start + d
                y[query, 0] = sample
                mask[max(start, query - 1) : query + 2] = True
            elif task == "match_sample":
                label = int(rng.integers(2))
                sample = rng.choice([-1.0, 1.0])
                x[start, 0] = sample
                x[query] = [sample if label else -sample, 0, 1]
                y[query] = label
                mask[query] = True
            elif task == "selective_memory":
                x[start + 1 : query, 0] = rng.uniform(-1, 1, query - start - 1)
                x[query, 2], y[query, 0], mask[query] = 1, sample, True
            elif task == "temporal_order":
                x[start : query + 1] = 0
                first = int(rng.integers(2))
                x[start + 1, first] = 1
                x[start + d, 1 - first] = 1
                x[query, 2], y[query, 0], mask[query] = 1, first, True
            elif task == "variable_delay":
                wait = int(rng.integers(2, 2 * d + 1))
                x[start] = [1, wait / (2 * d), 0]
                y[start + wait] = 1
                mask[start : start + block] = True
            else:
                raise ValueError(f"Unimplemented task {task}")
    mask[: spec.warmup] = False
    if not mask.any() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid diagnostic data/mask")
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "mask": mask,
        "baseline": None if baseline is None else baseline.astype(np.float32),
        "metadata": metadata,
    }


def build_dataset(spec, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_dataset(spec, output)


def _build_dataset(spec, output):
    spec.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if digest_json(manifest["spec"]) != digest_json(asdict(spec)):
            raise ValueError("Immutable dataset already contains a different task specification")
        load_dataset(output)
        return manifest
    arrays, metadata = {}, {}
    for split in ("train", "validation", "test"):
        rows = [sequence(spec, split, i) for i in range(getattr(spec, f"{split}_sequences"))]
        for key in ("x", "y", "mask"):
            arrays[f"{split}_{key}"] = np.stack([r[key] for r in rows])
        if rows[0]["baseline"] is not None:
            arrays[f"{split}_baseline"] = np.stack([r["baseline"] for r in rows])
        metadata[split] = [r["metadata"] for r in rows]
    train_y, train_mask = arrays["train_y"], arrays["train_mask"]
    mean = np.array([train_y[..., k][train_mask[..., k]].mean() for k in range(train_y.shape[-1])])
    std = np.array([train_y[..., k][train_mask[..., k]].std() for k in range(train_y.shape[-1])])
    arrays["target_mean"], arrays["target_std"] = mean, np.maximum(std, 1e-4)
    path = output / "trajectories.npz"
    temporary = output / "trajectories.pending.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    manifest = {
        "schema": "temporal-dataset-v1",
        "spec": json.loads(json.dumps(asdict(spec))),
        "classification": spec.task in CLASSIFICATION,
        "metadata": metadata,
        "observation_dim": arrays["train_x"].shape[-1],
        "action_dim": train_y.shape[-1],
        "trajectory_sha256": digest_file(path),
        "split_hashes": {
            s: hashlib.sha256(arrays[f"{s}_x"].tobytes() + arrays[f"{s}_y"].tobytes()).hexdigest()
            for s in metadata
        },
        "scaling": "target mean/std from scored training targets only; inputs in declared physical units",
    }
    manifest["fingerprint"] = digest_json(manifest)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("schema") != "temporal-dataset-v1"
        or digest_file(path / "trajectories.npz") != manifest["trajectory_sha256"]
    ):
        raise ValueError("Diagnostic dataset checksum/schema mismatch")
    expected = manifest.pop("fingerprint")
    if digest_json(manifest) != expected:
        raise ValueError("Diagnostic manifest checksum mismatch")
    manifest["fingerprint"] = expected
    with np.load(path / "trajectories.npz", allow_pickle=False) as loaded:
        arrays = {k: loaded[k].copy() for k in loaded.files}
    return manifest, arrays


def diagnostic_metrics(prediction, target, mask, *, classification=False, task=None, baseline=None):
    if (
        prediction.shape != target.shape
        or mask.shape != target.shape
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Invalid diagnostic prediction")
    result, channels = {}, []
    for k in range(target.shape[-1]):
        valid = mask[..., k]
        y, p = target[..., k][valid], prediction[..., k][valid]
        if classification:
            guess = p >= 0.5
            positives, negatives = y >= 0.5, y < 0.5
            recall = float(guess[positives].mean()) if positives.any() else None
            specificity = float((~guess[negatives]).mean()) if negatives.any() else None
            channels.append(
                {
                    "accuracy": float((guess == positives).mean()),
                    "balanced_accuracy": (recall + specificity) / 2
                    if recall is not None and specificity is not None
                    else None,
                    "precision": float(positives[guess].mean()) if guess.any() else 0.0,
                    "recall": recall,
                    "false_positive_rate": 1 - specificity if specificity is not None else None,
                }
            )
        else:
            mse, variance = float(np.mean((p - y) ** 2)), float(np.var(y))
            corr = (
                float(np.corrcoef(p, y)[0, 1] ** 2)
                if np.std(p) > 1e-10 and variance > 1e-10
                else 0.0
            )
            channels.append(
                {
                    "mse": mse,
                    "nmse": mse / max(variance, 1e-10),
                    "r2": 1 - mse / max(variance, 1e-10),
                    "squared_correlation": corr,
                }
            )
    if classification:
        values = [r["balanced_accuracy"] for r in channels if r["balanced_accuracy"] is not None]
        result["score"] = (
            float(np.mean(values)) if values else float(np.mean([r["accuracy"] for r in channels]))
        )
        if task in ("change_point", "variable_delay"):
            delays = []
            for y, p, m in zip(target, prediction, mask, strict=True):
                events = np.flatnonzero((y[:, 0] > 0.5) & np.r_[True, y[:-1, 0] <= 0.5] & m[:, 0])
                for event in events:
                    after = np.flatnonzero(
                        (p[event : event + 16, 0] >= 0.5) & m[event : event + 16, 0]
                    )
                    delays.append(int(after[0]) if len(after) else None)
            result.update(detection_delays=delays, undetected_events=sum(x is None for x in delays))
    else:
        result["score"] = float(np.mean([r["r2"] for r in channels]))
        if task == "impulse_recall":
            timing, amplitude, missed = [], [], 0
            for y, p, m in zip(target, prediction, mask, strict=True):
                events = np.flatnonzero((np.abs(y[:, 0]) > 1e-8) & m[:, 0])
                for event in events:
                    left, right = max(0, event - 4), min(len(y), event + 5)
                    peak = left + int(np.argmax(np.abs(p[left:right, 0])))
                    amplitude.append(float((p[event, 0] - y[event, 0]) ** 2))
                    if abs(p[peak, 0]) < 0.2 * abs(y[event, 0]):
                        missed += 1
                    else:
                        timing.append(abs(peak - int(event)))
            result.update(
                recall_timing_mae_samples=float(np.mean(timing)) if timing else None,
                recall_amplitude_mse=float(np.mean(amplitude)) if amplitude else None,
                missed_pulses=missed,
            )
        if task == "rhythm":
            phases, frequency_errors = [], []
            for y, p, m in zip(target, prediction, mask, strict=True):
                actual, expected = p[m[:, 0], 0], y[m[:, 0], 0]
                spectrum_y = np.fft.rfft(expected - expected.mean())
                spectrum_p = np.fft.rfft(actual - actual.mean())
                if len(spectrum_y) > 1:
                    peak = 1 + int(np.argmax(np.abs(spectrum_y[1:])))
                    if abs(spectrum_p[peak]) > 1e-8:
                        phases.append(float(abs(np.angle(spectrum_p[peak] / spectrum_y[peak]))))
                    peak_p = 1 + int(np.argmax(np.abs(spectrum_p[1:])))
                    frequency_errors.append(abs(peak_p - peak) / len(actual))
            result.update(
                continuation_phase_error_radians=float(np.mean(phases)) if phases else None,
                continuation_frequency_error_cycles_per_sample=float(np.mean(frequency_errors))
                if frequency_errors
                else None,
            )
        if task == "linear_memory":
            result["memory_capacity"] = sum(r["squared_correlation"] for r in channels)
            result["short_memory"] = float(np.mean([r["r2"] for r in channels[:3]]))
            result["long_memory"] = (
                float(np.mean([r["r2"] for r in channels[3:]])) if len(channels) > 3 else None
            )
    if baseline is not None and not classification:
        result["baseline_nmse"] = [
            float(
                np.mean((baseline[..., k][mask[..., k]] - target[..., k][mask[..., k]]) ** 2)
                / max(np.var(target[..., k][mask[..., k]]), 1e-10)
            )
            for k in range(target.shape[-1])
        ]
    return {**result, "channels": channels, "scored_targets": int(mask.sum())}
