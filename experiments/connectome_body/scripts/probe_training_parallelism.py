"""Bounded end-to-end training throughput probe; production resumes in finally.

All optimizers/checkpoints here are disposable performance artifacts. The original
controller code, datasets, checkpoint, and optimizer are left unchanged.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runpod_short_test import read_json, stop_child, write_json  # noqa: E402


def worker(args):
    import torch

    from connectome_body.adaptation import imitation
    from connectome_body.adaptation.imitation import AdapterSpec, Optimization
    from connectome_body.adaptation.trajectories import TrajectoryCache

    config = read_json(args.config)
    output = args.output.resolve()
    case = output.parent
    output.mkdir(parents=True)
    original_load = TrajectoryCache.load
    load_seconds = 0.0
    load_calls = 0

    def measured_load(cache, index):
        nonlocal load_seconds, load_calls
        start = time.perf_counter()
        value = original_load(cache, index)
        load_seconds += time.perf_counter() - start
        load_calls += 1
        return value

    TrajectoryCache.load = (
        functools.lru_cache(maxsize=None)(measured_load) if args.ram else measured_load
    )
    sample = imitation._sample_batch
    sample_seconds, started = 0.0, None

    def timed_sample(caches, rng, opt):
        nonlocal sample_seconds, started
        if started is None:
            if args.ram:
                for cache in caches:
                    for index in range(len(cache)):
                        cache.load(index)
            write_json(output / "ready.json", {"pid": os.getpid()})
            while not (case / "go").exists():
                time.sleep(0.02)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
        before = time.perf_counter()
        batch = sample(caches, rng, opt)
        sample_seconds += time.perf_counter() - before
        return batch

    imitation._sample_batch = timed_sample
    opt = Optimization(
        **(
            config["offline"]
            | {
                "batch_size": args.batch,
                "updates": 100000,
                "checkpoint_every": 100000,
                "eval_every": 100000,
            }
        )
    )
    imitation.train_offline(
        AdapterSpec.from_dict(config["adapter"]),
        opt,
        config["training_caches"],
        args.mini_validation,
        output / "training",
        initial_checkpoint=args.parent_checkpoint,
        stop_after=args.updates,
    )
    torch.cuda.synchronize()
    ended = time.time()
    saved = torch.load(output / "training/latest.pt", map_location="cpu", weights_only=False)
    write_json(
        output / "result.json",
        {
            "started": started,
            "ended": ended,
            "active_seconds": ended - started,
            "updates": saved["updates"],
            "sample_presentations": saved["sample_presentations"],
            "burn_presentations": saved["burn_presentations"],
            "sample_seconds": sample_seconds,
            "load_calls": load_calls,
            "load_seconds_including_preload": load_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "ram_cache": args.ram,
            "batch_size": args.batch,
            "finite_actor": all(torch.isfinite(v).all().item() for v in saved["actor"].values()),
        },
    )


def supervise(args):
    import numpy as np

    from connectome_body.adaptation.trajectories import TrajectoryCache, write_cache
    from connectome_body.util import digest_file

    config = read_json(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paused, children, logs, rows = [], [], [], []

    def interrupt(_signal, _frame):
        raise InterruptedError("Probe interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    production = args.production.resolve()
    latest = production / "training/latest.pt"
    # Pause the supervisor before reading its child, so it cannot launch another.
    os.killpg(args.supervisor_pgid, signal.SIGSTOP)
    paused.append(args.supervisor_pgid)
    try:
        child = read_json(production / "status.json").get("pid")
        if child:
            try:
                os.killpg(child, signal.SIGSTOP)
                paused.append(child)
            except ProcessLookupError:
                pass
        time.sleep(0.2)
        parent = output / "parent"
        parent.mkdir()
        shutil.copy2(latest, parent / "latest.pt")
        shutil.copy2(latest.parent / "manifest.json", parent / "manifest.json")
        before_hash = digest_file(latest)
        write_json(output / "pause.json", {"groups": paused, "checkpoint_sha256": before_hash})
        validation = TrajectoryCache(config["data"]["validation"])
        mini = output / "mini-validation"
        trajectories = []
        for index in range(4):
            item = validation.load(index)
            length = min(64, len(item["actions"]))
            trajectories.append(
                {
                    "observations": item["observations"][: length + 1],
                    "actions": item["actions"][:length],
                    "mask": item["mask"][:length],
                    "truncated": np.arange(length) == length - 1,
                    "scenario": validation.manifest["trajectories"][index]["scenario"],
                }
            )
        write_cache(
            mini,
            trajectories,
            validation.manifest["metadata"]
            | {"purpose": "throughput_probe_only_not_model_selection"},
        )
        cases = [
            (1, 64, False),
            (1, 64, True),
            (1, 128, True),
            (1, 256, True),
            (2, 64, True),
            (2, 128, True),
            (4, 64, True),
        ]
        overall_deadline = time.time() + args.max_seconds
        for count, batch, ram in cases:
            if time.time() + 45 > overall_deadline:
                break
            name = f"c{count}-b{batch}-{'ram' if ram else 'disk'}"
            case = output / name
            case.mkdir()
            children, logs = [], []
            status = "passed"
            try:
                for index in range(count):
                    path = case / f"worker-{index}"
                    log = (case / f"worker-{index}.log").open("w")
                    logs.append(log)
                    command = [
                        sys.executable,
                        __file__,
                        "worker",
                        "--config",
                        str(args.config),
                        "--output",
                        str(path),
                        "--batch",
                        str(batch),
                        "--updates",
                        str(args.updates),
                        "--mini-validation",
                        str(mini),
                        "--parent-checkpoint",
                        str(parent / "latest.pt"),
                    ]
                    if ram:
                        command.append("--ram")
                    children.append(
                        subprocess.Popen(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    )
                deadline = min(time.time() + 150, overall_deadline)
                while len(list(case.glob("worker-*/ready.json"))) != count:
                    if any(p.poll() is not None for p in children) or time.time() >= deadline:
                        raise RuntimeError("Worker failed or timed out during preparation")
                    time.sleep(0.2)
                (case / "go").touch()
                while any(p.poll() is None for p in children) and time.time() < deadline:
                    time.sleep(0.2)
                if any(p.poll() is None or p.returncode != 0 for p in children):
                    raise RuntimeError("Worker failed or timed out during training")
                results = [read_json(case / f"worker-{i}/result.json") for i in range(count)]
                seconds = max(r["ended"] for r in results) - min(r["started"] for r in results)
                row = {
                    "case": name,
                    "workers": count,
                    "batch_size": batch,
                    "ram": ram,
                    "status": status,
                    "wall_seconds": seconds,
                    "aggregate_updates_per_second": sum(r["updates"] for r in results) / seconds,
                    "aggregate_timesteps_per_second": sum(
                        r["sample_presentations"] for r in results
                    )
                    / seconds,
                    "sampling_fraction": sum(r["sample_seconds"] for r in results)
                    / sum(r["active_seconds"] for r in results),
                    "worker_results": results,
                }
            except RuntimeError as error:
                row = {"case": name, "status": "failed", "error": str(error)}
            finally:
                for process in children:
                    stop_child(process)
                for log in logs:
                    log.close()
            rows.append(row)
            write_json(output / "results.json", rows)
            print(json.dumps({k: v for k, v in row.items() if k != "worker_results"}), flush=True)
        if digest_file(latest) != before_hash:
            raise ValueError("Production checkpoint unexpectedly changed while paused")
    finally:
        for process in children:
            stop_child(process)
        for group in reversed(paused):
            try:
                os.killpg(group, signal.SIGCONT)
            except ProcessLookupError:
                pass
        write_json(output / "resumed.json", {"groups": paused, "time_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "supervise"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--ram", action="store_true")
    parser.add_argument("--mini-validation", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--production", type=Path)
    parser.add_argument("--supervisor-pgid", type=int)
    parser.add_argument("--max-seconds", type=int, default=480)
    args = parser.parse_args()
    (worker if args.mode == "worker" else supervise)(args)


if __name__ == "__main__":
    main()
