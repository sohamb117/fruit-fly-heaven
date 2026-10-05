"""Isolated BC-to-PPO development run; never counted as a frozen-plan result.

Uses the existing shared expert data and parameter budgets. Each request records
its parent plan, full changed protocol and code identity, and resumes only itself.
Run under a bounded external supervisor as evaluations can outlast a soft limit.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from connectome_body.compatibility.imitation import BCConfig, train_bc
    from connectome_body.compatibility.run_config import RunSpec
    from connectome_body.compatibility.training import source_identity, train
    from connectome_body.util import atomic_json, digest_file, digest_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["adapter_only", "gru", "real", "degree_rewired"], required=True
    )
    parser.add_argument("--source", default="banc")
    parser.add_argument("--task", default="hover")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--ppo-interactions", type=int, default=40000)
    args = parser.parse_args()
    plan = json.loads(args.parent_plan.read_text())
    if plan["fingerprint"] != digest_json({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Parent plan seal changed")
    row = next(
        r
        for r in plan["conditions"]
        if r["source"] == args.source
        and r["task"] == args.task
        and r["variant"] == args.variant
        and r["plasticity"] == "adapters"
        and r["seed"] == 0
        and r["regime"] == "bc_ppo"
        and r["experiment"] == "cross_body_core"
    )
    spec = RunSpec.from_dict(row["config"])
    dt = spec.controller.substrate.dynamics.control_dt
    dynamics = spec.controller.substrate.dynamics
    if args.tau is not None:
        dynamics = replace(dynamics, tau_seconds=args.tau)
    training = replace(
        spec.training,
        learning_rate=0.0001,
        target_kl=0.01,
        exploration_log_std=-3.0,
        eval_episodes=4,
        test_episodes=4,
        interactions=args.ppo_interactions,
        eval_every=10000,
    )
    if args.task == "hover":
        training = training.physical_horizons(
            dt,
            discount_seconds=0.2,
            advantage_seconds=0.05,
            sequence_seconds=0.0256,
            burn_in_seconds=0.0256,
            rollout_seconds=0.1024,
        )
    spec = replace(
        spec,
        controller=replace(
            spec.controller,
            normalize_observations=True,
            substrate=replace(spec.controller.substrate, dynamics=dynamics),
        ),
        training=training,
        metadata={**spec.metadata, "development_only": True, "parent_plan": plan["fingerprint"]},
    )
    bc = BCConfig(
        epochs=args.epochs,
        batch_size=4,
        sequence_length=128,
        standardize_actions=True,
        initialize_action_mean=True,
    )
    spec.validate()
    bc.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    request = dict(
        parent_plan_sha256=digest_file(args.parent_plan),
        parent_condition=row["id"],
        source=source_identity(),
        config=spec.to_dict(),
        bc=asdict(bc),
        dataset=plan["datasets"][args.task]["path"],
        evidence="development qualification; not a paired scientific trial",
    )
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Development request changed; use a new output version")
    atomic_json(request_path, request)
    started = time.monotonic()
    output = args.output / "bc"
    result = train_bc(
        spec,
        bc,
        request["dataset"],
        output,
        resume=(output / "latest.pt").exists(),
        max_seconds=args.max_seconds,
    )
    atomic_json(args.output / "stage.json", {"stage": "bc", **result})
    remaining = args.max_seconds - (time.monotonic() - started)
    if result["status"] == "complete" and remaining > 30:
        output = args.output / "ppo"
        result = train(
            replace(spec, initial_checkpoint=str((args.output / "bc/best.pt").resolve())),
            output,
            resume=(output / "latest.pt").exists(),
            max_seconds=remaining,
        )
        atomic_json(args.output / "stage.json", {"stage": "ppo", **result})


if __name__ == "__main__":
    main()
