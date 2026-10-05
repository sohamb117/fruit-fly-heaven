"""Bring two existing frozen-plan controls forward within the original GPU deadline.

Existing workers continue. New scheduler admissions pause only while these two
bounded workers run. The previous admission limit is restored on exit. No source,
scientific budget, dataset, checkpoint identity, or provider resource is changed.
"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    scheduler = ROOT / "runs/main-parallel-20260918"
    output = ROOT / "runs/priority-controls-20260918"
    output.mkdir(exist_ok=True)
    lock = (output / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "request.json").exists():
        raise FileExistsError("Priority window already started; inspect its status")
    request = json.loads((scheduler / "request.json").read_text())
    deadline = min(time.time() + 1800, request["checkpoint_deadline"])
    seconds = min(1200, deadline - time.time() - 30)
    if seconds < 60:
        raise RuntimeError("Original authorized window has insufficient time")
    control_path = scheduler / "control.json"
    control = json.loads(control_path.read_text())
    previous = control["max_workers"]
    atomic(
        output / "request.json",
        {
            "started_at": time.time(),
            "hard_deadline": deadline,
            "original_deadline": request["deadline"],
            "soft_seconds": seconds,
            "previous_max_workers": previous,
            "note": "Existing six workers continue",
        },
    )
    jobs, exiting = [], False

    def stop(signum, frame):
        nonlocal exiting
        exiting = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
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
        for name, condition, execution in [
            ("matched_gru_hover", "7ef6e0e2e6ebd0a7a54bd888", "4d0dbec884577da83120f530-bc_ppo"),
            ("adapter_only_hover", "237270e26693494d8383e2b7", "e2ecf7db63b53727fab0bcea-bc_ppo"),
        ]:
            log = (output / f"{name}.log").open("a")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "connectome_body.compatibility.learning_study",
                    "worker",
                    "--plan",
                    "runs/learning-core-cuda-v2/plan.json",
                    "--qualification",
                    "runs/learning-qualification-cuda-v2/qualification.json",
                    "--experiment",
                    "cross_body_core",
                    "--condition",
                    condition,
                    "--max-runs",
                    "1",
                    "--max-seconds",
                    str(seconds),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            jobs.append({"name": name, "execution": execution, "process": process, "log": log})
        while not exiting and time.time() < deadline:
            records = [
                {
                    "name": j["name"],
                    "execution": j["execution"],
                    "pid": j["process"].pid,
                    "returncode": j["process"].poll(),
                }
                for j in jobs
            ]
            atomic(
                output / "status.json", {"time": time.time(), "jobs": records, "deadline": deadline}
            )
            if all(j["process"].poll() is not None for j in jobs):
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
            output / "complete.json",
            {
                "time": time.time(),
                "admission_limit": current["max_workers"],
                "jobs": [
                    {
                        "name": j["name"],
                        "execution": j["execution"],
                        "returncode": j["process"].poll(),
                    }
                    for j in jobs
                ],
                "checkpoints_retained": True,
            },
        )


if __name__ == "__main__":
    main()
