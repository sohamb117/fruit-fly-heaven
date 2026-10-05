"""Paired zero-shot severity and acute state/edge tests for trained drone policies."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from ..util import atomic_json, digest_file
from .battery import execution_key, read_battery
from .bodies import make_body
from .controller import Controller
from .evaluation import InterventionSpec, evaluate
from .run_config import RunSpec
from .training import policy_digest


def evaluate_drone_condition(plan_path, condition, output, *, episodes=20):
    if type(episodes) is not int or episodes < 1:
        raise ValueError("Positive evaluation episode count required")
    plan = read_battery(plan_path)
    row = next((c for c in plan["conditions"] if c["id"] == condition), None)
    if row is None or row["suite"] != "drone":
        raise ValueError("Select a drone condition from this plan")
    directory = Path(plan_path).resolve().parent / "runs" / row["execution_id"]
    if not (directory / "result.json").exists():
        raise ValueError(
            "Complete training and checkpoint selection before held-out intervention tests"
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        execution_key(manifest["config"]) != row["execution_id"]
        or manifest["code_fingerprint"] != plan["code_fingerprint"]
    ):
        raise ValueError("Trained policy does not match this plan")
    config = RunSpec.from_dict(row["config"])
    body = make_body(config.body)
    controller = Controller(body.obs_dim, body.action_dim, config.controller).to(config.device)
    saved = torch.load(directory / "best.pt", map_location=config.device, weights_only=False)
    if saved["identity"] != manifest["identity"]:
        raise ValueError("Checkpoint identity mismatch")
    controller.load_state_dict(saved["actor"])
    body.close()
    policy = policy_digest(controller)
    records = []
    original = json.loads((Path(plan_path).resolve().parent / "config.json").read_text())
    cases = [("trained_task", config.body)]
    for task, settings in original.get("severity", {}).items():
        # All stress tasks use hover commands except failure's explicit landing
        # objective; the original task is always retained as its own reference.
        for parameters in settings:
            cases.append(
                (
                    f"{task}:{json.dumps(parameters, sort_keys=True)}",
                    replace(
                        config.body, task=task, parameters={**config.body.parameters, **parameters}
                    ),
                )
            )
    for label, spec in cases:
        env = make_body(spec)
        try:
            record = evaluate(
                controller, env, config.train_seed, episodes, "test", policy_identity=policy
            )
            records.append(
                {
                    "case": label,
                    "parameters": spec.parameters,
                    "task": spec.task,
                    "evaluation": record,
                }
            )
        finally:
            env.close()
    interventions = []
    if controller.state_dim:
        interventions.append("reset")
    if controller.topology is not None:
        interventions.append("remove_edges")
    for intervention in interventions:
        env = make_body(config.body)
        try:
            records.append(
                {
                    "case": f"acute:{intervention}",
                    "task": config.body.task,
                    "evaluation": evaluate(
                        controller,
                        env,
                        config.train_seed,
                        episodes,
                        "test",
                        intervention=InterventionSpec(kind=intervention),
                        policy_identity=policy,
                    ),
                }
            )
        finally:
            env.close()
    result = {
        "schema": "drone-zero-shot-evaluation-v1",
        "plan_fingerprint": plan["fingerprint"],
        "condition": condition,
        "checkpoint_sha256": digest_file(directory / "best.pt"),
        "policy_digest": policy,
        "training_updates": 0,
        "paired_seeds": True,
        "records": records,
        "scope": "Fixed selected policy; exogenous task changes and severity tests explicitly labeled",
    }
    atomic_json(output, result)
    return result
