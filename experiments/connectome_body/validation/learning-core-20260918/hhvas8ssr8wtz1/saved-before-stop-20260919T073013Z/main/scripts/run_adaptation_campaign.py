"""Run a validated adaptation plan sequentially, with durable local status.

Each cell uses the existing worker in a fresh process. Restarting this script
resumes unfinished cells from their checkpoints and skips verified results.
It does not change the plan, optimization, data, or evaluation procedure.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def progress(row):
    if row is None:
        return None
    output = Path(row["output"])
    stages = []
    for stage in sorted(output.glob("stage-[0-9][0-9]")):
        losses = read(stage / "losses.json", [])
        validation = read(stage / "learning_curve.json", [])
        result = read(stage / "result.json")
        stages.append(
            {
                "stage": stage.name,
                "completed": result is not None,
                "checkpointed_updates": losses[-1]["updates"] if losses else 0,
                "last_training": losses[-1] if losses else None,
                "last_validation": validation[-1] if validation else None,
                "checkpoint": str(stage / "latest.pt") if (stage / "latest.pt").exists() else None,
            }
        )
    return {
        "run": row["name"],
        "output": str(output),
        "stages": stages,
        "closed_loop_frontier": read(output / "frontier.json", []),
        "selection": read(output / "selection.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("Poll interval must be positive")
    project = Path(__file__).resolve().parents[1]
    plan_path = args.plan.resolve()
    plan = read(plan_path)
    if plan is None or plan["readiness"] != "ready":
        raise RuntimeError("A ready, validated plan is required")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be on PATH")
    root = plan_path.parent
    status_path = root / "campaign-status.json"
    log_root = root / "logs"
    log_root.mkdir(exist_ok=True)
    status = {
        "state": "starting",
        "supervisor_pid": os.getpid(),
        "started_utc": now(),
        "plan": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "planned_runs": len(plan["runs"]),
        "completed_runs": 0,
        "attempts": [],
    }
    worker = None
    keep_awake = None
    stop_requested = False

    def stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        if worker is not None and worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def update(state=None, row=None):
        if state is not None:
            status["state"] = state
        status["updated_utc"] = now()
        status["completed_runs"] = sum(
            (Path(item["output"]) / "result.json").exists() for item in plan["runs"]
        )
        status["current"] = progress(row)
        status["free_disk_bytes"] = shutil.disk_usage(root).free
        write(status_path, status)

    command_prefix = [
        uv,
        "run",
        "--frozen",
        "--offline",
        "--cache-dir",
        "/private/tmp/connectome-body-uv",
        "python",
        "-u",
        "-m",
        "connectome_body.adaptation.cli",
    ]
    with (root / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read(status_path)
        if prior is not None:
            with (root / "campaign-history.jsonl").open("a") as history:
                history.write(json.dumps(prior, sort_keys=True) + "\n")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if sys.platform == "darwin":
            keep_awake = subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            status["idle_sleep_inhibitor_pid"] = keep_awake.pid
        update()
        try:
            while not stop_requested:
                pending = [
                    row
                    for row in plan["runs"]
                    if not (Path(row["output"]) / "result.json").exists()
                ]
                row = pending[0] if pending else None
                label = row["name"] if row else "verify-completed"
                command = command_prefix + [
                    "worker",
                    "--plan",
                    str(plan_path),
                    "--max-runs",
                    "1",
                ]
                log_path = log_root / f"{label}.log"
                attempt = {"run": label, "started_utc": now(), "log": str(log_path)}
                status["attempts"].append(attempt)
                print(json.dumps({"event": "starting", **attempt}), flush=True)
                started = time.monotonic()
                with log_path.open("a") as log:
                    log.write(f"\nCampaign attempt started {attempt['started_utc']}\n")
                    log.flush()
                    worker = subprocess.Popen(
                        command,
                        cwd=project,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    status["worker_pid"] = worker.pid
                    update("running", row)
                    while worker.poll() is None:
                        try:
                            worker.wait(timeout=args.poll_seconds)
                        except subprocess.TimeoutExpired:
                            update(row=row)
                    attempt["elapsed_seconds"] = time.monotonic() - started
                    attempt["finished_utc"] = now()
                    attempt["returncode"] = worker.returncode
                status["worker_pid"] = None
                print(json.dumps({"event": "finished", **attempt}), flush=True)
                if stop_requested:
                    update("stopped", row)
                    return
                if worker.returncode:
                    update("failed", row)
                    raise RuntimeError(f"Worker failed; inspect {log_path}")
                update("running", row)
                if row is None:
                    break
            if stop_requested:
                update("stopped")
                return
            analysis = root.with_name(root.name + "-analysis")
            update("analyzing")
            with (log_root / "analysis.log").open("a") as log:
                worker = subprocess.Popen(
                    command_prefix + ["analyze", "--runs", str(root), "--output", str(analysis)],
                    cwd=project,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                status["worker_pid"] = worker.pid
                update()
                worker.wait()
                status["worker_pid"] = None
                if worker.returncode:
                    raise RuntimeError(f"Analysis failed; inspect {log_root / 'analysis.log'}")
            status["analysis"] = str(analysis)
            update("complete")
            print(json.dumps({"event": "complete", "analysis": str(analysis)}), flush=True)
        except BaseException as error:
            if worker is not None and worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                worker.wait()
            status["worker_pid"] = None
            status["error"] = repr(error)
            update("stopped" if stop_requested else "failed")
            raise
        finally:
            if keep_awake is not None:
                keep_awake.terminate()
                keep_awake.wait()


if __name__ == "__main__":
    main()
