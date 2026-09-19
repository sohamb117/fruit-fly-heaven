"""Bounded, checkpoint-preserving CUDA qualification on a single RunPod GPU.

This harness does not change the experiment's controller or optimization code.
The caller must stop the RunPod resource separately after retrieving artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def checkpoint_summary(path):
    import torch

    path = Path(path).resolve()
    payload = path.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    required = {"actor", "optimizer", "rng", "sample_rng", "updates", "identity"}
    if not required.issubset(saved):
        raise ValueError("Checkpoint is missing resumable training state")
    finite = all(torch.isfinite(value).all().item() for value in saved["actor"].values())
    if not finite:
        raise ValueError("Checkpoint contains nonfinite actor state")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "updates": saved["updates"],
        "identity": saved["identity"],
        "optimizer_entries": len(saved["optimizer"]["state"]),
        "rng_saved": True,
        "finite_actor": finite,
        "elapsed_seconds": saved["elapsed_seconds"],
        "last_validation": saved["curve"][-1] if saved["curve"] else None,
    }


def preflight(config, output, expected_gpu, expected_torch, body_reference=None):
    import importlib.metadata

    import torch

    from connectome_body.adaptation.configuration import hover_config_from_dict
    from connectome_body.adaptation.evaluation import evaluate_controller, validate_qualification
    from connectome_body.adaptation.hover import FlyBodyInterface
    from connectome_body.adaptation.imitation import source_identity
    from connectome_body.adaptation.teacher import FlightTeacher

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This test requires exactly one CUDA GPU")
    gpu = torch.cuda.get_device_name(0)
    if expected_gpu.casefold() not in gpu.casefold():
        raise RuntimeError(f"Expected the authorized {expected_gpu}, got {gpu}")
    if torch.__version__.split("+")[0] != expected_torch:
        raise RuntimeError(f"Expected PyTorch {expected_torch}, got {torch.__version__}")
    torch.set_num_threads(1)
    if body_reference is None:
        body = FlyBodyInterface(hover_config_from_dict(config["body"]))
        native_identity = None
    else:
        from runpod_body_equivalence import construct_body

        body, native_identity = construct_body(hover_config_from_dict(config["body"]))
    teacher = FlightTeacher()
    try:
        equivalence = {"status": "identical_native_identity"}
        try:
            validate_qualification(config["qualification"], body, teacher)
        except ValueError:
            qualification = read_json(config["qualification"])
            if (
                body_reference is None
                or not qualification.get("qualified")
                or qualification.get("evidence") != "native_hover_teacher_qualification"
                or qualification["teacher_fingerprint"] != teacher.fingerprint
            ):
                raise
            from runpod_body_equivalence import validate_reference

            equivalence = validate_reference(
                native_identity,
                body.task.root_entity.mjcf_model.to_xml_string(),
                qualification["body_fingerprint"],
                body_reference,
            )
        started = time.monotonic()
        teacher_rollout = evaluate_controller(body, seeds=[910001, 910002], teacher=teacher)
        seconds = time.monotonic() - started
        report = {
            "status": "passed",
            "gpu": gpu,
            "expected_gpu": expected_gpu,
            "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "expected_torch": expected_torch,
            "cuda_runtime": torch.version.cuda,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "numba", "mujoco", "dm-control")
            },
            "source_identity": source_identity(),
            "body_fingerprint": body.fingerprint,
            "body_equivalence": equivalence,
            "teacher_fingerprint": teacher.fingerprint,
            "teacher_rollout_seconds": seconds,
            "teacher_body_steps_per_second": teacher_rollout["evaluation_interactions"] / seconds,
            "teacher_rollout": teacher_rollout,
        }
        write_json(output / "preflight.json", report)
    finally:
        body.close()


def rollout(config, output):
    import torch

    from connectome_body.adaptation.configuration import hover_config_from_dict
    from connectome_body.adaptation.evaluation import evaluate_controller
    from connectome_body.adaptation.hover import FlyBodyInterface
    from connectome_body.adaptation.imitation import load_actor

    torch.set_num_threads(1)
    stage = output / "experiment/stage-00"
    actor, _, _ = load_actor(stage, "best.pt")
    body = FlyBodyInterface(hover_config_from_dict(config["body"]))
    try:
        if body.fingerprint != read_json(output / "preflight.json")["body_fingerprint"]:
            raise ValueError("Rollout body differs from the validated native body")
        torch.cuda.synchronize()
        started = time.monotonic()
        metrics = evaluate_controller(body, seeds=[920001, 920002, 920003], actor=actor)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        write_json(
            output / "student-rollout.json",
            {
                "evidence": "engineering_validation_rollout_not_headline_evaluation",
                "wall_seconds": seconds,
                "body_steps_per_second": metrics["evaluation_interactions"] / seconds,
                "metrics": metrics,
                "checkpoint_sha256": hashlib.sha256((stage / "best.pt").read_bytes()).hexdigest(),
            },
        )
    finally:
        body.close()


def stop_child(child):
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait(timeout=10)
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def supervise(
    config_path,
    output,
    deadline,
    expected_gpu,
    expected_torch,
    training_mode="pipeline",
    body_reference=None,
):
    config = read_json(config_path)
    output.mkdir(parents=True, exist_ok=True)
    events = []
    child = None
    log_handle = None

    def event(phase, **details):
        row = {"phase": phase, "time_unix": time.time(), **details}
        events.append(row)
        write_json(output / "events.json", events)
        write_json(output / "status.json", row)
        print(json.dumps(row), flush=True)

    def launch(name, command):
        nonlocal child, log_handle
        if log_handle is not None:
            log_handle.close()
        log_handle = (output / f"{name}.log").open("a")
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        event(name, pid=child.pid)

    def wait_phase(limit):
        while child.poll() is None and time.time() < limit:
            time.sleep(1)
        if child.poll() is None:
            stop_child(child)
            raise TimeoutError("Phase reached its allocated deadline")
        if child.returncode != 0:
            raise RuntimeError(f"Phase failed with exit code {child.returncode}")

    try:
        launch(
            "preflight",
            [
                sys.executable,
                __file__,
                "preflight",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--expected-gpu",
                expected_gpu,
                "--expected-torch",
                expected_torch,
            ]
            + (["--body-reference", str(body_reference)] if body_reference else []),
        )
        wait_phase(min(time.time() + 180, deadline - 240))
        launch(
            "benchmark",
            [
                sys.executable,
                "-m",
                "connectome_body.adaptation.benchmark",
                "--graph",
                config["adapter"]["graph"],
                "--device",
                "cuda",
                "--output",
                str(output / "banc-cuda-benchmark.json"),
            ],
        )
        wait_phase(min(time.time() + 240, deadline - 240))
        command = [
            sys.executable,
            "-m",
            "connectome_body.adaptation.cli",
            "offline" if training_mode == "offline" else "run",
            "--config",
            str(config_path),
            "--output",
            str(
                output / "experiment/stage-00"
                if training_mode == "offline"
                else output / "experiment"
            ),
        ]
        launch("training", command)
        resumed = False
        before = None
        while child.poll() is None and time.time() < deadline - 150:
            losses = read_json(output / "experiment/stage-00/losses.json", [])
            updates = losses[-1]["updates"] if losses else 0
            write_json(
                output / "status.json",
                {
                    "phase": "training",
                    "time_unix": time.time(),
                    "checkpointed_updates_stage00": updates,
                    "resume_exercised": resumed,
                },
            )
            if updates >= 25 and not resumed:
                stop_child(child)
                before = checkpoint_summary(output / "experiment/stage-00/latest.pt")
                write_json(
                    output / "resume-proof.json", {"status": "resume_requested", "before": before}
                )
                launch("training-resumed", command + ["--resume"])
                resumed = True
            if before is not None and updates > before["updates"]:
                write_json(
                    output / "resume-proof.json",
                    {
                        "status": "passed",
                        "before": before,
                        "checkpointed_updates_after_resume": updates,
                    },
                )
                before = None
            time.sleep(2)
        exit_code = child.poll()
        stop_child(child)
        if exit_code not in (None, 0):
            raise RuntimeError(f"Training failed with exit code {exit_code}")
        checkpoints = [
            checkpoint_summary(p) for p in sorted((output / "experiment").glob("stage-*/latest.pt"))
        ]
        write_json(output / "checkpoint-audit.json", checkpoints)
        if not checkpoints:
            raise RuntimeError("No resumable checkpoint was produced")
        if time.time() < deadline - 90:
            launch(
                "student-rollout",
                [
                    sys.executable,
                    __file__,
                    "rollout",
                    "--config",
                    str(config_path),
                    "--output",
                    str(output),
                ],
            )
            wait_phase(deadline - 30)
        event(
            "complete",
            checkpoints=checkpoints,
            training_mode=training_mode,
            offline_complete=(output / "experiment/stage-00/result.json").exists(),
            experiment_complete=(output / "experiment/result.json").exists(),
        )
    except Exception as error:
        event("error", error=repr(error))
        raise
    finally:
        stop_child(child)
        if log_handle is not None:
            log_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preflight", "rollout", "supervise"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--expected-gpu", default="RTX 4090")
    parser.add_argument("--expected-torch", default="2.8.0")
    parser.add_argument("--training-mode", choices=("pipeline", "offline"), default="pipeline")
    parser.add_argument("--body-reference", type=Path)
    args = parser.parse_args()
    if args.phase == "supervise":
        if args.deadline_unix is None:
            parser.error("supervise requires --deadline-unix")
        supervise(
            args.config,
            args.output,
            args.deadline_unix,
            args.expected_gpu,
            args.expected_torch,
            args.training_mode,
            args.body_reference,
        )
    elif args.phase == "preflight":
        preflight(
            read_json(args.config),
            args.output,
            args.expected_gpu,
            args.expected_torch,
            args.body_reference,
        )
    else:
        rollout(read_json(args.config), args.output)


if __name__ == "__main__":
    main()
