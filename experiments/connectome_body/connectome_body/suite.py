"""Explicit run matrices, data readiness, and independent worker shards."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from .config import RunConfig
from .data import source_registry
from .graphs import Graph
from .policy import BASELINES
from .train import code_fingerprint, train
from .util import atomic_json, digest_json


def make_plan(spec: dict, graph_root: Path, output: Path, datasets=None):
    output = Path(output).resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("Choose a new plan directory; plans are immutable")
    chosen = list(datasets) if datasets is not None else spec["datasets"]
    missing, graphs = {}, {}
    registry = source_registry()["datasets"]
    for name in chosen:
        path = (graph_root / name).resolve()
        if not (path / "manifest.json").exists():
            entry = registry.get(name, {})
            missing[name] = entry.get("reason", f"Prepare the graph at {path}")
        else:
            graph = Graph.load(path)
            graphs[name] = {
                "path": str(path),
                "fingerprint": graph.fingerprint,
                "summary": graph.manifest["summary"],
            }
    rows = []
    cases = spec.get("dynamics_cases", [{"name": "primary", "settings": {}}])
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Dynamics cases must have distinct names")
    for case, name, task, substrate, budget, seed, draw in itertools.product(
        cases,
        chosen,
        spec["tasks"],
        spec["substrates"],
        spec["adapter_budgets"],
        spec["train_seeds"],
        spec["substrate_seeds"],
    ):
        value = json.loads(json.dumps(spec.get("base", {})))
        value.update(
            graph=str((graph_root / name).resolve()),
            substrate=substrate,
            adapter_budget=budget,
            train_seed=seed,
            substrate_seed=draw,
        )
        value.setdefault("body", {})["task"] = task
        value.setdefault("dynamics", {}).update(case["settings"])
        config = RunConfig.from_dict(value)
        run_id = f"{name}-{task}-{substrate}-p{budget}-s{seed}-g{draw}"
        if "dynamics_cases" in spec:
            run_id = f"{case['name']}-{run_id}"
        rows.append(
            {
                "id": run_id,
                "dataset": name,
                "ready": name not in missing,
                "config": config.to_dict(),
            }
        )
    # Brain-free baselines have no graph/interface randomness. Do not duplicate
    # identical baseline runs once per dataset or port draw (pseudoreplication).
    for case, task, substrate, budget, seed in itertools.product(
        cases,
        spec["tasks"],
        spec.get("baselines", BASELINES),
        spec["adapter_budgets"],
        spec["train_seeds"],
    ):
        if substrate not in BASELINES:
            raise ValueError(f"Unknown brain-free baseline {substrate}")
        value = json.loads(json.dumps(spec.get("base", {})))
        value.update(
            graph=None,
            substrate=substrate,
            adapter_budget=budget,
            train_seed=seed,
            substrate_seed=0,
        )
        value.setdefault("body", {})["task"] = task
        value.setdefault("dynamics", {}).update(case["settings"])
        config = RunConfig.from_dict(value)
        run_id = f"baseline-{task}-{substrate}-p{budget}-s{seed}"
        if "dynamics_cases" in spec:
            run_id = f"{case['name']}-{run_id}"
        rows.append(
            {"id": run_id, "dataset": "baseline", "ready": True, "config": config.to_dict()}
        )
    plan = {
        "schema": "connectome-body-plan-v1",
        "code_fingerprint": code_fingerprint(),
        "study": spec,
        "requested_datasets": chosen,
        "explicit_dataset_override": datasets is not None,
        "missing_datasets": missing,
        "graphs": graphs,
        "runs": rows,
        "planned_runs": len(rows),
        "ready_runs": sum(row["ready"] for row in rows),
        "training_interactions": sum(row["config"]["ppo"]["interactions"] for row in rows),
        "status": "data_required" if missing else "ready",
        "evaluation_experience": "Additional, logged separately; early falls change evaluation episode lengths",
        "resource_note": "Profile full graphs with preflight/train-smoke before launching; a plan is not a completed experiment",
    }
    plan["fingerprint"] = digest_json(plan)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan)
    for row in rows:
        atomic_json(output / "configs" / f"{row['id']}.json", row["config"])
    return plan


def run_plan(path: Path, shard_index=0, shard_count=1, max_runs=None, resume=True):
    plan = json.loads(path.read_text())
    expected = plan.pop("fingerprint")
    if digest_json(plan) != expected:
        raise ValueError("Plan was edited; generate a new one")
    if plan["code_fingerprint"] != code_fingerprint():
        raise ValueError("Code changed after planning; generate a new plan")
    if plan["missing_datasets"]:
        raise ValueError(
            f"Plan has missing datasets: {plan['missing_datasets']}. "
            "Prepare them or explicitly generate a dataset-limited plan."
        )
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid worker shard")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    results, completed = [], 0
    for index, row in enumerate(plan["runs"]):
        if index % shard_count != shard_index:
            continue
        config = RunConfig.from_dict(row["config"])
        if config.graph:
            actual = Graph.load(config.graph).fingerprint
            if actual != plan["graphs"][row["dataset"]]["fingerprint"]:
                raise ValueError(f"Graph changed after planning: {row['dataset']}")
        output = path.parent / "runs" / row["id"]
        if (output / "result.json").is_file():
            result = json.loads((output / "result.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            if result.get("status") != "complete" or result["identity"] != manifest["identity"]:
                raise ValueError(f"Invalid completed artifact at {output}")
            completed += 1
            continue
        do_resume = resume and (output / "latest.pt").exists()
        results.append(train(config, output, resume=do_resume))
        if max_runs is not None and len(results) >= max_runs:
            break
    return {
        "processed_runs": len(results),
        "already_completed_runs": completed,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
