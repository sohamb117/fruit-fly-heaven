#!/usr/bin/env python3
"""Bounded runtime check on an existing machine; never provision a resource."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def probe(device, threads, expected_gpu):
    import numpy as np
    import torch

    from connectome_body.compatibility.dynamics import SparseRateCore
    from connectome_body.compatibility.runtime import hardware_profile
    from connectome_body.compatibility.training import source_identity

    torch.set_num_threads(threads)
    torch.manual_seed(19)
    hardware = hardware_profile(device, threads)
    if expected_gpu and expected_gpu.casefold() not in hardware["device_name"].casefold():
        raise RuntimeError("The allocated GPU differs from --expected-gpu")
    src = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    dst = np.array([1, 2, 0, 3, 0, 3, 1, 2])
    checks = []
    for plastic in (False, True):
        core = SparseRateCore(
            4, src, dst, np.arange(1, 9), np.array([1, -1, 1, -1]), plastic=plastic
        ).to(device)
        initial = torch.randn(2, 4, device=device, requires_grad=True)
        state = initial
        neural_input = torch.arange(8, device=device).reshape(2, 4) / 10
        for _ in range(3):
            state = core(state, neural_input)
        state.square().mean().backward()
        gradients = [initial.grad]
        if plastic:
            gradients.append(core.raw_magnitudes.grad)
        if not torch.isfinite(state).all() or any(
            gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient != 0)
            for gradient in gradients
        ):
            raise RuntimeError("Sparse recurrent state/edge gradient check failed")
        if device == "cuda":
            torch.cuda.synchronize()
        checks.append({"plastic_edges": plastic, "finite_nonzero_gradients": True})
    return {
        "hardware": hardware,
        "code_fingerprint": source_identity(),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("torch", "numpy", "scipy", "mujoco", "pyarrow")
        },
        "cpp17_compiler": shutil.which("clang++") or shutil.which("g++"),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--expected-gpu")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.timeout <= 0 or args.timeout > 60:
        parser.error("Use positive threads and a timeout of at most 60 seconds")
    if args.probe:
        try:
            result = {"status": "passed", **probe(args.device, args.threads, args.expected_gpu)}
        except Exception as exc:
            result = {"status": "failed", "exception": type(exc).__name__, "message": str(exc)}
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "passed" else 1
    if not args.output:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("Preserve existing evidence; select a new output path")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe",
        "--device",
        args.device,
        "--threads",
        str(args.threads),
    ]
    if args.expected_gpu:
        command.extend(["--expected-gpu", args.expected_gpu])
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        result = json.loads(completed.stdout)
        if completed.returncode and result.get("status") == "passed":
            raise ValueError("Probe process failed despite a passing report")
    except subprocess.TimeoutExpired:
        result = {
            "status": "failed",
            "exception": "TimeoutExpired",
            "message": "Runtime initialization or sparse gradient probe exceeded the time limit",
        }
    except (OSError, ValueError) as exc:
        result = {
            "status": "failed",
            "exception": type(exc).__name__,
            "message": "Runtime probe did not return a valid report",
        }
    report = {
        "schema": "connectome-paper-runtime-preflight-v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "requested_device": args.device,
        "timeout_seconds": args.timeout,
        "scope": "Tiny native sparse forward/state-gradient/edge-gradient check; no throughput or learning claim",
        "cloud_resources_created": False,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
