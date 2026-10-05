"""Bounded concurrent execution of an already qualified, frozen learning plan.

Changes scheduling only: each child uses the existing source-pinned worker,
unchanged observations, data, optimizer, parameter budgets and training budgets.
This script bounds workloads; stopping provider billing is a separate operation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def unique_rows(plan):
    seen, rows = set(), []
    for row in plan["conditions"]:
        if row["regime"] != "bc_ppo" or row["experiment"] != "cross_body_core":
            continue
        if row["execution_id"] not in seen:
            seen.add(row["execution_id"])
            rows.append(row)
    pairs = {
        ("banc", "hover"): 0,
        ("malecns", "hover"): 1,
        ("celegans", "worm_locomotion"): 2,
        ("banc", "worm_locomotion"): 3,
        ("malecns", "worm_locomotion"): 4,
        ("celegans", "hover"): 5,
    }
    variants = {"real": 0, "degree_rewired": 1, "gru": 2, "rnn": 3, "adapter_only": 4}
    return sorted(
        rows,
        key=lambda r: (
            r["seed"],
            r["plasticity"] == "joint",
            variants[r["variant"]],
            pairs[r["source"], r["task"]],
        ),
    )


def reservation(row, graph_info):
    """Conservative initial reservations; telemetry controls additional admission."""
    cfg = row["config"]["controller"]["substrate"]
    if cfg["kind"] == "connectome":
        edges = graph_info[row["source"]]["m"]
        large = edges > 1_000_000
        if large:
            gpu = 3.0 if row["plasticity"] == "adapters" else 6.5
            ram = 2.0 + edges * 160 / 2**30
        else:
            gpu, ram = 0.8, 1.5
    else:
        large = False
        params = row["parameters"]["total_trainable_parameters"]
        gpu = 1.0 + params * 32 / 2**30
        ram = 1.5 + params * 24 / 2**30
    return {"gpu_gib": gpu, "ram_gib": ram, "large_graph": large}


def fits(candidate, active, control, available_ram):
    reservations = [r["reservation"] for r in active]
    return (
        len(active) < control["max_workers"]
        and sum(r["large_graph"] for r in reservations) + candidate["large_graph"]
        <= control["max_large_graphs"]
        and sum(r["gpu_gib"] for r in reservations) + candidate["gpu_gib"]
        <= control["gpu_budget_gib"]
        and sum(r["ram_gib"] for r in reservations) + candidate["ram_gib"]
        <= control["ram_budget_gib"]
        and candidate["ram_gib"] + 2 <= available_ram
    )


def memory_available_gib():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    maximum = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if maximum.exists() and current.exists() and maximum.read_text().strip() != "max":
        available = min(available, int(maximum.read_text()) - int(current.read_text()))
    return available / 2**30


def gpu_usage():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    util, used, total = map(float, result.stdout.strip().splitlines()[0].split(","))
    return {"utilization_percent": util, "used_gib": used / 1024, "total_gib": total / 1024}


def stop_children(active):
    for item in active.values():
        if item["process"].poll() is None:
            os.killpg(item["process"].pid, signal.SIGTERM)
    until = time.monotonic() + 15
    for item in active.values():
        try:
            item["process"].wait(timeout=max(0.1, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(item["process"].pid, signal.SIGKILL)
            item["process"].wait()
        item["log"].close()


def run(args):
    from connectome_body.compatibility.learning_study import read_plan
    from connectome_body.util import atomic_json, digest_json

    plan_path, out = Path(args.plan).resolve(), Path(args.output).resolve()
    plan = read_plan(plan_path)
    q = json.loads(Path(args.qualification).read_text())
    if (
        not q.get("passed")
        or q["source"] != plan["source"]
        or q["plan_fingerprint"] != plan["fingerprint"]
        or q["fingerprint"] != digest_json({k: v for k, v in q.items() if k != "fingerprint"})
    ):
        raise ValueError("Passing qualification must match this exact source and plan")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "scheduler.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request_path = out / "request.json"
    if request_path.exists():
        raise FileExistsError("Select a new scheduling window; existing run checkpoints are reused")
    now = time.time()
    deadline = now + args.seconds
    soft_deadline = deadline - args.checkpoint_grace
    rows = unique_rows(plan)
    atomic_json(
        request_path,
        {
            "plan": plan["fingerprint"],
            "source": plan["source"],
            "started_at": now,
            "deadline": deadline,
            "checkpoint_deadline": soft_deadline,
            "unique_pipelines": len(rows),
            "qualification": str(Path(args.qualification).resolve()),
            "note": "Scheduling window only; each pipeline retains its fixed scientific budget",
        },
    )
    control_path = out / "control.json"
    if not control_path.exists():
        atomic_json(
            control_path,
            {
                "max_workers": 4,
                "max_large_graphs": 2,
                "gpu_budget_gib": 21.0,
                "ram_budget_gib": 24.0,
                "ready_seeds": [0],
            },
        )
    active, finished, failed = {}, {}, {}
    quitting = False

    def request_stop(signum, frame):
        nonlocal quitting
        quitting = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        while not quitting and time.time() < deadline:
            # Detect a source replacement before admitting further work.
            read_plan(plan_path)
            control = json.loads(control_path.read_text())
            for eid, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                del active[eid]
                record = {
                    "condition": item["row"]["id"],
                    "returncode": code,
                    "ended_at": time.time(),
                }
                result = plan_path.parent / "runs" / eid / "result.json"
                if code == 0 and result.exists():
                    finished[eid] = record
                elif code == 0:
                    record["status"] = "paused_at_checkpoint"
                    finished[eid] = record
                else:
                    failed[eid] = record
            gpu = gpu_usage()
            admission_used_gib = gpu["used_gib"]
            free_ram = memory_available_gib()
            if time.time() < soft_deadline:
                for row in rows:
                    eid = row["execution_id"]
                    if eid in active or eid in finished or eid in failed:
                        continue
                    if row["seed"] not in control["ready_seeds"]:
                        continue
                    result = plan_path.parent / "runs" / eid / "result.json"
                    if result.exists():
                        finished[eid] = {"status": "previously_complete"}
                        continue
                    data = Path(plan["datasets"][row["task"]]["path"]) / "manifest.json"
                    if not data.exists() or not json.loads(data.read_text()).get("qualified"):
                        continue
                    reserve = reservation(row, plan["graphs"])
                    if not fits(reserve, list(active.values()), control, free_ram):
                        continue
                    # Allow for transient allocator peaks and external processes.
                    if admission_used_gib + reserve["gpu_gib"] > gpu["total_gib"] - 1.5:
                        continue
                    log = (out / f"{eid}.log").open("a")
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "connectome_body.compatibility.learning_study",
                        "worker",
                        "--plan",
                        str(plan_path),
                        "--qualification",
                        args.qualification,
                        "--experiment",
                        "cross_body_core",
                        "--condition",
                        row["id"],
                        "--max-runs",
                        "1",
                        "--max-seconds",
                        str(max(1, soft_deadline - time.time())),
                    ]
                    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
                    env.update(CONNECTOME_PREPARATION_REQUIRE_CACHE="1", PYTHONUNBUFFERED="1")
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[eid] = {
                        "process": process,
                        "log": log,
                        "row": row,
                        "reservation": reserve,
                        "started_at": time.time(),
                    }
                    free_ram -= reserve["ram_gib"]
                    admission_used_gib += reserve["gpu_gib"]
                    print(
                        json.dumps(
                            {
                                "launched": eid,
                                "pid": process.pid,
                                "source": row["source"],
                                "task": row["task"],
                                "variant": row["variant"],
                                "plasticity": row["plasticity"],
                                "seed": row["seed"],
                            }
                        ),
                        flush=True,
                    )
            status = {
                "time": time.time(),
                "deadline": deadline,
                "gpu": gpu,
                "available_ram_gib": memory_available_gib(),
                "control": control,
                "active": [
                    {
                        "execution_id": k,
                        "pid": v["process"].pid,
                        "source": v["row"]["source"],
                        "task": v["row"]["task"],
                        "variant": v["row"]["variant"],
                        "plasticity": v["row"]["plasticity"],
                        "seed": v["row"]["seed"],
                        "reservation": v["reservation"],
                    }
                    for k, v in active.items()
                ],
                "finished": finished,
                "failed": failed,
                "total": len(rows),
            }
            atomic_json(out / "status.json", status)
            if len(finished) + len(failed) == len(rows) and not active:
                break
            if not active and time.time() >= soft_deadline:
                break
            time.sleep(args.poll)
    finally:
        stop_children(active)
        atomic_json(
            out / "window-complete.json",
            {
                "ended_at": time.time(),
                "deadline": deadline,
                "finished": finished,
                "failed": failed,
                "interrupted": quitting,
                "checkpoints_retained": True,
            },
        )
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--qualification", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--checkpoint-grace", type=float, default=600)
    p.add_argument("--poll", type=float, default=10)
    args = p.parse_args()
    if not 0 < args.checkpoint_grace < args.seconds <= 86400 or args.poll <= 0:
        p.error("Use a positive checkpoint grace below a window of at most 24 hours")
    run(args)


if __name__ == "__main__":
    main()
