"""Resumable supervised TBPTT using exactly the embodied Controller interface."""

from __future__ import annotations

import fcntl
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..util import (
    atomic_json,
    atomic_torch_save,
    digest_json,
    restore_rng,
    rng_state,
    seed_everything,
    seed_for,
)
from .config import ControllerConfig
from .controller import Controller
from .temporal import diagnostic_metrics, load_dataset
from .training import controller_identity, source_identity


@dataclass(frozen=True)
class DiagnosticRun:
    dataset: str
    controller: ControllerConfig
    epochs: int = 20
    batch_size: int = 4
    sequence_length: int = 64
    learning_rate: float = 0.001
    seed: int = 0
    device: str = "cpu"
    threads: int = 1
    success_threshold: float = 0.8
    threshold_confirmations: int = 3
    primary_linear_readout: bool = True
    metadata: dict = field(default_factory=dict)

    def validate(self):
        self.controller.validate()
        for k in ("epochs", "batch_size", "sequence_length", "threads", "threshold_confirmations"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(f"Positive integer {k} required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative training seed required")
        if (
            self.device not in ("cpu", "cuda")
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Invalid training device/rate")
        if not 0 < self.success_threshold <= 1:
            raise ValueError("Invalid diagnostic success threshold")
        if self.primary_linear_readout and self.controller.adapter.family not in (
            "linear_linear",
            "mlp_linear",
        ):
            raise ValueError("Primary diagnostics require a linear readout")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["controller"] = ControllerConfig.from_dict(value["controller"])
        result = cls(**value)
        result.validate()
        return result


@torch.no_grad()
def evaluate_diagnostic(controller, arrays, manifest, split, batch_size, intervention="none"):
    device = next(controller.parameters()).device
    was_training = controller.training
    controller.eval()
    predictions = []
    try:
        context = controller.context()
        x = arrays[f"{split}_x"]
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], device=device)
            state, outputs = controller.reset(len(batch)), []
            for t in range(batch.shape[1]):
                if intervention == "reset":
                    state = torch.zeros_like(state)
                logits, state = controller(
                    batch[:, t],
                    state,
                    context,
                    squash=False,
                    remove_edges=intervention == "remove_edges",
                )
                if not torch.isfinite(logits).all() or not torch.isfinite(state).all():
                    raise FloatingPointError("Nonfinite temporal controller evaluation")
                outputs.append(logits.cpu().numpy())
            predictions.append(np.stack(outputs, axis=1))
        values = np.concatenate(predictions)
        if manifest["classification"]:
            values = 1 / (1 + np.exp(-np.clip(values, -80, 80)))
        else:
            values = values * arrays["target_std"] + arrays["target_mean"]
        return diagnostic_metrics(
            values,
            arrays[f"{split}_y"],
            arrays[f"{split}_mask"],
            classification=manifest["classification"],
            task=manifest["spec"]["task"],
            baseline=arrays.get(f"{split}_baseline"),
        )
    finally:
        controller.train(was_training)


def train_diagnostic(config, output, *, resume=False, max_seconds=None, stop_after_batches=None):
    config.validate()
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Positive wall-time allowance required")
    if stop_after_batches is not None and (
        type(stop_after_batches) is not int or stop_after_batches < 1
    ):
        raise ValueError("Positive batch limit required")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, arrays = load_dataset(config.dataset)
        identity = digest_json(
            {
                "config": asdict(config),
                "dataset": manifest["fingerprint"],
                "code": source_identity(),
                "torch": torch.__version__,
            }
        )
        if (output / "manifest.json").exists() and not resume:
            raise FileExistsError("Diagnostic run exists; use --resume")
        seed_everything(seed_for(config.seed, "diagnostic-training"), config.threads)
        controller = Controller(
            manifest["observation_dim"], manifest["action_dim"], config.controller
        ).to(config.device)
        identity = digest_json({"run": identity, "controller": controller_identity(controller)})
        expected = config.metadata.get("expected_graph_fingerprint")
        if expected and controller.graph_fingerprint != expected:
            raise ValueError("Diagnostic graph changed since planning")
        optimizer = torch.optim.Adam(
            [p for p in controller.parameters() if p.requires_grad], lr=config.learning_rate
        )
        n, length = arrays["train_x"].shape[:2]
        run_manifest = {
            "schema": "diagnostic-run-v1",
            "identity": identity,
            "config": asdict(config),
            "dataset_fingerprint": manifest["fingerprint"],
            "data_spec": manifest["spec"],
            "parameters": controller.parameter_report,
            "graph_fingerprint": controller.graph_fingerprint,
            "code_fingerprint": source_identity(),
            "experience_unit": "input timesteps; exposures count repeated dataset passes separately",
            "bptt": "One preceding chunk is replayed with gradients before each scored chunk; loss only on scored chunk",
        }
        cursor = {
            "epoch": 0,
            "batch_start": 0,
            "batches": 0,
            "optimizer_steps": 0,
            "exposures": 0,
            "replayed_timesteps": 0,
            "unique_timesteps": 0,
            "evaluation_timesteps": 0,
            "wall_seconds": 0.0,
            "curve": [],
            "best_score": -float("inf"),
            "best_exposures": 0,
            "best_actor": None,
        }
        if resume:
            saved = torch.load(output / "latest.pt", weights_only=False, map_location=config.device)
            if saved["identity"] != identity:
                raise ValueError("Diagnostic resume identity mismatch")
            controller.load_state_dict(saved["actor"])
            optimizer.load_state_dict(saved["optimizer"])
            cursor = saved["cursor"]
            restore_rng(saved["rng"])
            if (output / "result.json").exists():
                result = json.loads((output / "result.json").read_text())
                if result["identity"] != identity:
                    raise ValueError("Diagnostic result identity mismatch")
                return result
        else:
            atomic_json(output / "manifest.json", run_manifest)
        started = time.monotonic()
        elapsed_before = cursor["wall_seconds"]

        def save():
            cursor["wall_seconds"] = elapsed_before + time.monotonic() - started
            atomic_torch_save(
                output / "latest.pt",
                {
                    "identity": identity,
                    "actor": controller.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cursor": cursor,
                    "rng": rng_state(),
                },
            )
            atomic_json(output / "learning_curve.json", cursor["curve"])
            if cursor["best_actor"] is not None:
                atomic_torch_save(
                    output / "best.pt",
                    {
                        "identity": identity,
                        "actor": cursor["best_actor"],
                        "exposures": cursor["best_exposures"],
                    },
                )

        def validate():
            score = evaluate_diagnostic(
                controller, arrays, manifest, "validation", config.batch_size
            )
            cursor["evaluation_timesteps"] += len(arrays["validation_x"]) * length
            cursor["curve"].append(
                {
                    "unique_timesteps": cursor["unique_timesteps"],
                    "exposures": cursor["exposures"],
                    "optimizer_steps": cursor["optimizer_steps"],
                    **score,
                }
            )
            if score["score"] > cursor["best_score"]:
                cursor["best_score"], cursor["best_exposures"] = score["score"], cursor["exposures"]
                cursor["best_actor"] = {
                    k: v.detach().cpu().clone() for k, v in controller.state_dict().items()
                }

        if not cursor["curve"]:
            validate()
            save()
        try:
            while cursor["epoch"] < config.epochs:
                order = np.random.default_rng(
                    seed_for(config.seed, f"diagnostic-epoch-{cursor['epoch']}")
                ).permutation(n)
                while cursor["batch_start"] < n:
                    ids = order[cursor["batch_start"] : cursor["batch_start"] + config.batch_size]
                    x = torch.as_tensor(arrays["train_x"][ids], device=config.device)
                    y = arrays["train_y"][ids]
                    if not manifest["classification"]:
                        y = (y - arrays["target_mean"]) / arrays["target_std"]
                    y = torch.as_tensor(y, device=config.device)
                    mask = torch.as_tensor(arrays["train_mask"][ids], device=config.device)
                    state = controller.reset(len(ids))
                    previous_start_state = None
                    for start in range(0, length, config.sequence_length):
                        stop = min(length, start + config.sequence_length)
                        context, outputs = controller.context(), []
                        if previous_start_state is not None:
                            state = previous_start_state
                            for t in range(max(0, start - config.sequence_length), start):
                                _, state = controller(x[:, t], state, context, squash=False)
                                cursor["replayed_timesteps"] += len(ids)
                        current_start_state = state.detach()
                        for t in range(start, stop):
                            logits, state = controller(x[:, t], state, context, squash=False)
                            outputs.append(logits)
                        prediction = torch.stack(outputs, dim=1)
                        valid = mask[:, start:stop]
                        if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
                            raise FloatingPointError("Nonfinite diagnostic trajectory")
                        if valid.any():
                            if manifest["classification"]:
                                loss = F.binary_cross_entropy_with_logits(
                                    prediction[valid], y[:, start:stop][valid]
                                )
                            else:
                                loss = F.mse_loss(prediction[valid], y[:, start:stop][valid])
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Nonfinite diagnostic loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                controller.parameters(), 1.0, error_if_nonfinite=True
                            )
                            optimizer.step()
                            cursor["optimizer_steps"] += 1
                        state = state.detach()
                        previous_start_state = current_start_state
                    cursor["batch_start"] += len(ids)
                    cursor["batches"] += 1
                    cursor["exposures"] += len(ids) * length
                    cursor["unique_timesteps"] = min(
                        n * length, cursor["unique_timesteps"] + len(ids) * length
                    )
                    if cursor["batch_start"] >= n:
                        cursor["epoch"] += 1
                        cursor["batch_start"] = 0
                        validate()
                        save()
                        break
                    save()
                    if (max_seconds is not None and time.monotonic() - started >= max_seconds) or (
                        stop_after_batches is not None and cursor["batches"] >= stop_after_batches
                    ):
                        return {
                            "status": "paused_at_checkpoint",
                            "identity": identity,
                            "exposures": cursor["exposures"],
                        }
                if cursor["epoch"] < config.epochs and (
                    (max_seconds is not None and time.monotonic() - started >= max_seconds)
                    or (stop_after_batches is not None and cursor["batches"] >= stop_after_batches)
                ):
                    return {
                        "status": "paused_at_checkpoint",
                        "identity": identity,
                        "exposures": cursor["exposures"],
                    }
            controller.load_state_dict(cursor["best_actor"])
            test = evaluate_diagnostic(controller, arrays, manifest, "test", config.batch_size)
            lesions = {}
            if controller.state_dim:
                lesions["reset"] = evaluate_diagnostic(
                    controller, arrays, manifest, "test", config.batch_size, "reset"
                )
            if controller.topology is not None:
                lesions["remove_edges"] = evaluate_diagnostic(
                    controller, arrays, manifest, "test", config.batch_size, "remove_edges"
                )
            # Evaluation never changes model selection. Result is reconstructed
            # from the terminal checkpoint after an interrupted final evaluation.
            threshold = None
            for i in range(len(cursor["curve"]) - config.threshold_confirmations + 1):
                rows = cursor["curve"][i : i + config.threshold_confirmations]
                if all(r["score"] >= config.success_threshold for r in rows):
                    threshold = rows[-1]["exposures"]
                    break
            curve = cursor["curve"]
            auc = float(
                np.trapz([r["score"] for r in curve], [r["exposures"] for r in curve])
                / max(1, cursor["exposures"])
            )
            result = {
                "schema": "diagnostic-result-v1",
                "status": "complete",
                "identity": identity,
                "parameters": controller.parameter_report,
                "task": manifest["spec"]["task"],
                "unique_training_timesteps": cursor["unique_timesteps"],
                "optimization_exposures": cursor["exposures"],
                "replayed_timesteps": cursor["replayed_timesteps"],
                "optimizer_steps": cursor["optimizer_steps"],
                "selected_exposures": cursor["best_exposures"],
                "exposures_to_threshold": threshold,
                "right_censored": threshold is None,
                "validation_auc": auc,
                "test": test,
                "causal_interventions": lesions,
                "wall_seconds": elapsed_before + time.monotonic() - started,
                "evaluation_timesteps": cursor["evaluation_timesteps"]
                + (1 + len(lesions)) * len(arrays["test_x"]) * length,
            }
            atomic_json(output / "result.json", result)
            return result
        except BaseException as exc:
            atomic_json(
                output / "failure.json",
                {
                    "identity": identity,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "resume": "latest committed complete sequence batch",
                },
            )
            raise
