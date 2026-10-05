"""Bound two development workers inside the existing paid window; never stop the pod."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2100)
    parser.add_argument("--variants", nargs="+", default=["adapter_only", "gru"])
    parser.add_argument("--tau", type=float)
    parser.add_argument("--name", default="controls")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "request.json").exists():
        raise FileExistsError("Select a new window name")
    scheduler = args.old_root / "runs/main-parallel-20260918"
    original = json.loads((scheduler / "request.json").read_text())
    deadline = min(time.time() + args.seconds, original["checkpoint_deadline"])
    if deadline - time.time() < 120:
        raise RuntimeError("No time left in original authorized window")
    control_path = scheduler / "control.json"
    control = json.loads(control_path.read_text())
    previous = control["max_workers"]
    atomic(
        out / "request.json",
        dict(
            started=time.time(),
            deadline=deadline,
            original_deadline=original["deadline"],
            previous_max_workers=previous,
            variants=args.variants,
        ),
    )
    jobs, stopping = [], False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        control["max_workers"] = 0
        atomic(control_path, control)
        env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
        env.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MUJOCO_GL="disable",
            MPLBACKEND="Agg",
            MPLCONFIGDIR="/workspace/.mplconfig",
            CONNECTOME_PREPARATION_REQUIRE_CACHE="1",
            PYTHONUNBUFFERED="1",
        )
        for variant in args.variants:
            log = (out / f"{variant}.log").open("a")
            command = [
                sys.executable,
                "-u",
                "scripts/qualify_training_repairs.py",
                "--parent-plan",
                str(args.old_root / "runs/learning-core-cuda-v2/plan.json"),
                "--output",
                str(out / variant),
                "--variant",
                variant,
                "--max-seconds",
                str(max(1, deadline - time.time() - 120)),
            ]
            if args.tau is not None:
                command += ["--tau", str(args.tau)]
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            jobs.append(dict(variant=variant, process=process, log=log))
        while not stopping and time.time() < deadline:
            rows = [
                dict(variant=j["variant"], pid=j["process"].pid, returncode=j["process"].poll())
                for j in jobs
            ]
            atomic(out / "status.json", dict(time=time.time(), deadline=deadline, jobs=rows))
            if all(r["returncode"] is not None for r in rows):
                break
            time.sleep(5)
    finally:
        for job in jobs:
            process = job["process"]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            job["log"].close()
        current = json.loads(control_path.read_text())
        if current["max_workers"] == 0:
            current["max_workers"] = previous
            atomic(control_path, current)
        atomic(
            out / "complete.json",
            dict(
                time=time.time(),
                admission_limit=current["max_workers"],
                jobs=[dict(variant=j["variant"], returncode=j["process"].poll()) for j in jobs],
            ),
        )


if __name__ == "__main__":
    main()
