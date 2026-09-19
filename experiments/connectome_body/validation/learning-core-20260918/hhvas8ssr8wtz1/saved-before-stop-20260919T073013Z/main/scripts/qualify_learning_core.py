"""Bounded actual-CUDA qualification; never launch the full scientific budget."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cases(plan):
    """Cover every source/body before slower topology and plasticity ablations."""
    native = [("banc", "hover"), ("malecns", "hover"), ("celegans", "worm_locomotion")]
    pairings = [
        ("banc", "hover"),
        ("celegans", "worm_locomotion"),
        ("malecns", "hover"),
        ("banc", "worm_locomotion"),
        ("celegans", "hover"),
        ("malecns", "worm_locomotion"),
    ]
    requests = [(source, task, "real", "adapters") for source, task in pairings]
    requests += [("banc", "hover", kind, "adapters") for kind in ("gru", "rnn", "adapter_only")]
    for topology, plasticity in [
        ("real", "joint"),
        ("degree_rewired", "adapters"),
        ("degree_rewired", "joint"),
    ]:
        requests += [(source, task, topology, plasticity) for source, task in native]
    selected = []
    for source, task, topology, plasticity in requests:
        matches = [
            r
            for r in plan["conditions"]
            if r["experiment"] == "cross_body_core"
            and r["regime"] == "bc_ppo"
            and r["source"] == source
            and r["task"] == task
            and r["variant"] == topology
            and r["plasticity"] == plasticity
            and r["seed"] == 0
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Qualification condition missing or ambiguous: {(source, task, topology, plasticity)}"
            )
        selected.append(matches[0])
    return selected


def one_case(plan_path, output, index):
    from dataclasses import replace

    import torch

    from connectome_body.compatibility.imitation import BCConfig, build_demonstrations, train_bc
    from connectome_body.compatibility.learning_study import read_plan
    from connectome_body.compatibility.run_config import RunSpec
    from connectome_body.compatibility.runtime import hardware_profile
    from connectome_body.compatibility.training import train
    from connectome_body.util import atomic_json, digest_file

    plan = read_plan(plan_path)
    row = cases(plan)[index]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Qualification requires the requested RTX 4090")
    torch.cuda.reset_peak_memory_stats()
    spec = RunSpec.from_dict(row["config"])
    spec = replace(
        spec,
        body=replace(spec.body, horizon=64),
        purpose="smoke",
        device="cuda",
        metadata={k: v for k, v in spec.metadata.items() if k != "expected_body_fingerprint"},
        training=replace(
            spec.training,
            interactions=16,
            num_envs=1,
            rollout_steps=8,
            sequence_length=8,
            burn_in=8,
            epochs=1,
            eval_every=16,
            eval_episodes=1,
            test_episodes=1,
        ),
    )
    dataset = output.parent / "datasets" / row["task"]
    build_demonstrations(
        spec.body, dataset, train_episodes=2, validation_episodes=1, min_success=0, purpose="smoke"
    )
    bc = BCConfig(epochs=1, batch_size=2, sequence_length=32)
    started = time.monotonic()
    stage_seconds = {}

    def stage(name, function, *args, **kwargs):
        tick = time.monotonic()
        print(json.dumps({"stage_started": name}), flush=True)
        result = function(*args, **kwargs)
        torch.cuda.synchronize()
        stage_seconds[name] = time.monotonic() - tick
        print(
            json.dumps({"stage_completed": name, "wall_seconds": stage_seconds[name]}), flush=True
        )
        return result

    if not (output / "bc/latest.pt").exists():
        paused = stage(
            "bc_first_update", train_bc, spec, bc, dataset, output / "bc", stop_after_updates=1
        )
        if paused["status"] != "paused_at_checkpoint":
            raise RuntimeError("BC did not exercise checkpoint interruption")
    completed = stage("bc_resume", train_bc, spec, bc, dataset, output / "bc", resume=True)
    initial = digest_file(output / "bc/best.pt")
    spec = replace(spec, initial_checkpoint=str((output / "bc/best.pt").resolve()))
    if not (output / "ppo/latest.pt").exists():
        p = stage("ppo_first_update", train, spec, output / "ppo", stop_after_updates=1)
        if p["status"] != "paused_at_checkpoint":
            raise RuntimeError("PPO did not exercise checkpoint interruption")
    r = stage("ppo_resume", train, spec, output / "ppo", resume=True)
    saved = torch.load(output / "ppo/latest.pt", map_location="cpu", weights_only=False)
    finite = all(torch.isfinite(v).all().item() for v in saved["actor"].values())
    ok = (
        finite
        and r["training_interactions"] == 16
        and r["pretraining"]["checkpoint_sha256"] == initial
        and completed["status"] == "complete"
    )
    record = {
        "passed": ok,
        "plan_fingerprint": plan["fingerprint"],
        "stage_seconds": stage_seconds,
        "source": plan["source"],
        "condition": row["id"],
        "substrate": row["source"],
        "variant": row["variant"],
        "plasticity": row["plasticity"],
        "body": spec.body.name,
        "hardware": hardware_profile("cuda", spec.threads),
        "wall_seconds": time.monotonic() - started,
        "bc_updates": completed["optimizer_steps"],
        "ppo_updates": r["rollout_updates"],
        "ppo_interactions_per_second": 16 / r["training_wall_seconds"],
        "cuda_peak_allocated_bytes": max(
            completed["cuda_peak_allocated_bytes"],
            r["cuda_peak_allocated_bytes"],
            torch.cuda.max_memory_allocated(),
        ),
        "finite_actor": finite,
        "bc_checkpoint_sha256": initial,
        "ppo_checkpoint_sha256": digest_file(output / "ppo/latest.pt"),
    }
    atomic_json(output / "qualification-case.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-seconds", type=float, default=1200)
    p.add_argument("--case-max-seconds", type=float, default=180)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--case", type=int)
    a = p.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in (a.max_seconds, a.case_max_seconds)):
        p.error("Deadlines must be positive and finite")
    os.environ["CONNECTOME_PREPARATION_REQUIRE_CACHE"] = "1"
    os.environ["CONNECTOME_PREPARATION_LOG"] = "1"
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.case is not None:
        print(json.dumps(one_case(a.plan, out, a.case)), flush=True)
        return
    from connectome_body.compatibility.learning_study import read_plan
    from connectome_body.util import atomic_json, digest_file, digest_json

    plan = read_plan(a.plan)
    request = {
        "plan": plan["fingerprint"],
        "source": plan["source"],
        "qualifier": digest_file(__file__),
    }
    request_path = out / "request.json"
    if request_path.exists():
        if not a.resume or json.loads(request_path.read_text()) != request:
            raise ValueError(
                "Existing qualification requires --resume and the same plan/source/script"
            )
    else:
        if a.resume or list(out.glob("case-*")):
            raise ValueError("No matching resumable qualification request")
        atomic_json(request_path, request)
    started = time.monotonic()
    records = []
    if a.max_seconds <= 0:
        raise ValueError("Positive qualification deadline required")
    for i, condition in enumerate(cases(plan)):
        dest = out / f"case-{i:02d}"
        dest.mkdir(exist_ok=True)
        remaining = a.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        result = dest / "qualification-case.json"
        if a.resume and result.exists() and json.loads(result.read_text()).get("passed"):
            prior = json.loads(result.read_text())
            valid = (
                prior.get("passed")
                and prior.get("condition") == condition["id"]
                and prior.get("source") == plan["source"]
                and prior.get("plan_fingerprint") == plan["fingerprint"]
            )
            if not valid:
                raise ValueError("Qualification case identity mismatch")
            for path, key in (
                ("bc/best.pt", "bc_checkpoint_sha256"),
                ("ppo/latest.pt", "ppo_checkpoint_sha256"),
            ):
                if digest_file(dest / path) != prior[key]:
                    raise ValueError("Completed qualification checkpoint changed")
            records.append(prior)
            continue
        timed_out = False
        with (dest / "worker.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--plan",
                    str(Path(a.plan).resolve()),
                    "--output",
                    str(dest),
                    "--case",
                    str(i),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=min(remaining, a.case_max_seconds))
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                timed_out = True
                code = child.returncode
        row = (
            json.loads(result.read_text())
            if code == 0 and result.exists()
            else {
                "case": i,
                "condition": condition["id"],
                "passed": False,
                "reason": (
                    "case_deadline" if remaining > a.case_max_seconds else "qualification_deadline"
                )
                if timed_out
                else "execution_error",
                "returncode": code,
            }
        )
        records.append(row)
        atomic_json(
            out / "progress.json",
            {"cases": records, "completed": len(records), "planned": len(cases(plan))},
        )
        print(
            json.dumps(
                {"case": i, "passed": row["passed"], "wall_seconds": time.monotonic() - started}
            ),
            flush=True,
        )
        # Record failures and continue to other bodies while the global budget remains.
    result = {
        "schema": "learning-gpu-qualification-v1",
        "source": plan["source"],
        "plan_fingerprint": plan["fingerprint"],
        "passed": len(records) == len(cases(plan)) and all(r["passed"] for r in records),
        "cases": records,
        "wall_seconds": time.monotonic() - started,
        "scope": "Full graph CUDA BC and PPO checkpoint/resume on short smoke episodes; no scientific performance claim",
        "hardware": next((r["hardware"] for r in records if "hardware" in r), None),
    }
    result["fingerprint"] = digest_json(result)
    atomic_json(out / "qualification.json", result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
