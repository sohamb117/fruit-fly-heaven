"""Bounded core and 120-cell training-regime experiment, before the full paper grid."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..graphs import Graph
from ..util import atomic_json, digest_file, digest_json
from .bodies import BodySpec, make_body
from .config import AdapterConfig, ControllerConfig, DynamicsConfig, SubstrateConfig
from .design import estimate_parameters
from .imitation import BCConfig, build_demonstrations, train_bc
from .run_config import LearningConfig, RunSpec
from .temporal import TemporalSpec, build_dataset, sequence
from .temporal_training import DiagnosticRun, train_diagnostic
from .training import source_identity, train


def compile_plan(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema") != "compatibility-learning-study-v1":
        raise ValueError("Unknown learning-study schema")
    if cfg["seeds"] != [0, 1, 2, 3, 4]:
        raise ValueError("The core requires five paired seeds")
    if cfg["adapter"]["family"] != "mlp_linear":
        raise ValueError("Freeze one stateless MLP/linear family")
    if (output / "plan.json").exists():
        raise FileExistsError("Keep the frozen plan; choose a new version")
    root = (config_path.parent / cfg.get("project_root", "../..")).resolve()
    graphs = {}
    for label, item in cfg["graphs"].items():
        graph = Graph.load(root / item["path"])
        if graph.fingerprint != item["fingerprint"] or graph.manifest["provenance"].get(
            "is_synthetic"
        ):
            raise ValueError(f"Unqualified biological graph {label}")
        graphs[label] = {
            "path": str((root / item["path"]).resolve()),
            "fingerprint": graph.fingerprint,
            "n": graph.n,
            "m": graph.m,
            "native_body": item["native_body"],
        }
    tasks = {}
    datasets = {}
    conditions = []
    for name, values in cfg["tasks"].items():
        values = copy.deepcopy(values)
        if values.get("model_manifest"):
            values["model_manifest"] = str((root / values["model_manifest"]).resolve())
        spec = BodySpec(**values)
        body = make_body(spec)
        try:
            if body.evidence == "software_fixture_only":
                raise ValueError("Primary core requires actual bodies")
            if spec.name == "worm":
                from .qualification import body_interface_identity

                qpath = root / cfg["worm_qualification"]
                q = json.loads(qpath.read_text())
                if (
                    not q["passed"]
                    or q["body_interface_identity"] != body_interface_identity("worm")
                    or q["model_manifest_sha256"] != digest_file(spec.model_manifest)
                ):
                    raise ValueError("Worm qualification is stale or failed")
            tasks[name] = {
                "body": asdict(spec),
                "obs": body.obs_dim,
                "actions": body.action_dim,
                "dt": body.control_dt,
                "body_fingerprint": body.fingerprint,
            }
            path = str((root / cfg["dataset_root"] / f"{name}-{body.fingerprint[:12]}").resolve())
            datasets[name] = {"path": path, "body": asdict(spec), "settings": cfg["demonstrations"]}
        finally:
            body.close()
    temporal = TemporalSpec(**cfg["temporal_data"])
    probe = sequence(temporal, "train", 0)
    temporal_path = str(
        (root / cfg["dataset_root"] / f"temporal-{digest_json(asdict(temporal))[:12]}").resolve()
    )
    tasks["linear_memory"] = {
        "obs": probe["x"].shape[-1],
        "actions": probe["y"].shape[-1],
        "dt": 0.02,
        "dataset": temporal_path,
    }
    learning = LearningConfig(**cfg["ppo"])
    bc = BCConfig(**cfg["bc"])
    learning.validate()
    bc.validate()
    adapter = AdapterConfig(**cfg["adapter"])

    def controller_for(task, label, variant, plasticity, seed):
        task = tasks[task]
        g = graphs[label]
        dyn = DynamicsConfig(**{**cfg["dynamics"], "control_dt": task["dt"]})
        base = ControllerConfig(
            adapter,
            SubstrateConfig(
                graph=g["path"],
                plasticity=plasticity,
                seed=seed,
                dynamics=dyn,
                preserve_strengths=True,
                swap_attempts_per_edge=cfg["swap_attempts_per_edge"],
            ),
            seed,
        )
        target = estimate_parameters(
            base, task["obs"], task["actions"], neurons=g["n"], edges=g["m"]
        )["total_trainable_parameters"]
        if variant in ("real", "degree_rewired"):
            result = replace(base, substrate=replace(base.substrate, topology=variant))
        elif variant == "adapter_only":
            result = ControllerConfig(
                adapter, SubstrateConfig(kind="adapter_only", seed=seed, dynamics=dyn), seed
            )
        else:
            result = ControllerConfig(
                replace(adapter, budget=max(256, min(adapter.budget, target // 2))),
                SubstrateConfig(
                    kind=variant, plasticity="joint", seed=seed, dynamics=dyn, total_budget=target
                ),
                seed,
            )
        parameters = estimate_parameters(
            result, task["obs"], task["actions"], neurons=g["n"], edges=g["m"]
        )
        return result, parameters, target

    def add(experiment, task, label, variant, plasticity, seed, regime):
        controller, parameters, target = controller_for(task, label, variant, plasticity, seed)
        meta = (
            {"expected_graph_fingerprint": graphs[label]["fingerprint"]}
            if variant in ("real", "degree_rewired")
            else {}
        )
        if task != "linear_memory":
            meta["expected_body_fingerprint"] = tasks[task]["body_fingerprint"]
        if task == "linear_memory":
            run = asdict(
                DiagnosticRun(
                    dataset=temporal_path,
                    controller=controller,
                    seed=seed,
                    device=cfg["device"],
                    threads=cfg["threads"],
                    **cfg["temporal_training"],
                    metadata=meta,
                )
            )
        else:
            run = RunSpec(
                body=BodySpec(**tasks[task]["body"]),
                controller=controller,
                training=learning,
                train_seed=seed,
                device=cfg["device"],
                threads=cfg["threads"],
                metadata=meta,
            ).to_dict()
        # Common controls, BC initialization and task data are actually shared, not duplicated.
        policy_id = digest_json(run)[:24]
        stage = "temporal" if task == "linear_memory" else regime
        row = {
            "experiment": experiment,
            "task": task,
            "source": label,
            "variant": variant,
            "plasticity": plasticity,
            "seed": seed,
            "regime": regime,
            "config": run,
            "policy_id": policy_id,
            "execution_id": f"{policy_id}-{stage}",
            "parameters": parameters,
            "native_pair": graphs[label]["native_body"] == tasks[task].get("body", {}).get("name"),
            "matched_total_target": target if variant in ("gru", "rnn") else None,
            "match_relative_error": (target - parameters["total_trainable_parameters"]) / target
            if variant in ("gru", "rnn")
            else None,
        }
        row["id"] = digest_json(
            {k: v for k, v in row.items() if k not in ("config", "parameters")}
        )[:24]
        conditions.append(row)

    for task in cfg["regime_tasks"]:
        for seed in cfg["seeds"]:
            for variant in ("real", "degree_rewired", "gru", "adapter_only"):
                for regime in ("bc", "ppo", "bc_ppo"):
                    add("training_regime", task, "banc", variant, "adapters", seed, regime)
    for task in [*cfg["core_tasks"], "linear_memory"]:
        for label in cfg["graphs"]:
            for seed in cfg["seeds"]:
                for plasticity in ("adapters", "joint"):
                    for variant in ("real", "degree_rewired", "gru", "rnn"):
                        add(
                            "cross_body_core",
                            task,
                            label,
                            variant,
                            plasticity,
                            seed,
                            "temporal" if task == "linear_memory" else "bc_ppo",
                        )
                add(
                    "cross_body_core",
                    task,
                    label,
                    "adapter_only",
                    "adapters",
                    seed,
                    "temporal" if task == "linear_memory" else "bc_ppo",
                )
    plan = {
        "schema": "compatibility-learning-plan-v1",
        "source": source_identity(),
        "goal_sha256": digest_file(root / "../GOAL.md"),
        "config": cfg,
        "tasks": tasks,
        "datasets": datasets,
        "temporal": {"path": temporal_path, "spec": asdict(temporal)},
        "graphs": graphs,
        "conditions": conditions,
        "counts": {
            "training_regime_cells": sum(x["experiment"] == "training_regime" for x in conditions),
            "core_comparison_cells": sum(x["experiment"] == "cross_body_core" for x in conditions),
            "unique_executions": len({x["execution_id"] for x in conditions}),
        },
        "qualification_required": True,
        "dagger": "deferred; not counted among essential regimes",
    }
    plan["fingerprint"] = digest_json(plan)
    atomic_json(output / "plan.json", plan)
    return plan


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "compatibility-learning-plan-v1" or plan.get(
        "fingerprint"
    ) != digest_json({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Learning plan seal mismatch")
    if plan["source"] != source_identity():
        raise ValueError("Source changed; compile a new frozen plan")
    return plan


def prepare_data(plan, tasks=None):
    records = {}
    for name, item in plan["datasets"].items():
        if tasks and name not in tasks:
            continue
        records[name] = build_demonstrations(
            BodySpec(**item["body"]), item["path"], **item["settings"]
        )
    records["temporal"] = build_dataset(
        TemporalSpec(**plan["temporal"]["spec"]), plan["temporal"]["path"]
    )
    return records


def execute_row(plan_path, row, *, max_seconds):
    plan = read_plan(plan_path)
    root = Path(plan_path).resolve().parent / "runs"
    root.mkdir(exist_ok=True)
    started = time.monotonic()
    output = root / row["execution_id"]
    if row["regime"] == "temporal":
        return train_diagnostic(
            DiagnosticRun.from_dict(row["config"]),
            output,
            resume=(output / "latest.pt").exists(),
            max_seconds=max_seconds,
        )
    spec = RunSpec.from_dict(row["config"])
    bc_dir = root / f"{row['policy_id']}-bc"
    if row["regime"] in ("bc", "bc_ppo"):
        data = plan["datasets"][row["task"]]["path"]
        if not (bc_dir / "result.json").exists():
            result = train_bc(
                spec,
                BCConfig(**plan["config"]["bc"]),
                data,
                bc_dir,
                resume=(bc_dir / "latest.pt").exists(),
                max_seconds=max_seconds,
            )
            if result["status"] != "complete" or time.monotonic() - started >= max_seconds:
                return result
        if row["regime"] == "bc":
            return json.loads((bc_dir / "result.json").read_text())
        spec = replace(spec, initial_checkpoint=str(bc_dir / "best.pt"))
    result = train(
        spec,
        output,
        resume=(output / "latest.pt").exists(),
        max_seconds=max(0.001, max_seconds - (time.monotonic() - started)),
    )
    return result


def run_worker(
    plan_path, *, max_seconds, max_runs, experiment=None, condition=None, qualification=None
):
    if max_seconds <= 0 or max_runs < 1:
        raise ValueError("Explicit bounded worker required")
    plan = read_plan(plan_path)
    if not qualification:
        raise ValueError("A passing source- and plan-pinned GPU qualification is required")
    q = json.loads(Path(qualification).read_text())
    if (
        not q.get("passed")
        or q.get("source") != plan["source"]
        or q.get("plan_fingerprint") != plan["fingerprint"]
        or q.get("fingerprint") != digest_json({k: v for k, v in q.items() if k != "fingerprint"})
    ):
        raise ValueError("Qualification is failed or belongs to a different plan/source")
    from .runtime import hardware_profile

    if q.get("hardware") != hardware_profile(plan["config"]["device"], plan["config"]["threads"]):
        raise ValueError("GPU qualification belongs to different hardware or precision")
    start = time.monotonic()
    results = []
    seen = set()
    for row in plan["conditions"]:
        if row["regime"] not in plan["config"].get(
            "launch_regimes", ["bc", "ppo", "bc_ppo", "temporal"]
        ):
            continue
        if (experiment and row["experiment"] != experiment) or (
            condition and row["id"] != condition
        ):
            continue
        if row["execution_id"] in seen:
            continue
        seen.add(row["execution_id"])
        if (
            Path(plan_path).resolve().parent / "runs" / row["execution_id"] / "result.json"
        ).exists():
            continue
        if len(results) >= max_runs or time.monotonic() - start >= max_seconds:
            break
        result = execute_row(plan_path, row, max_seconds=max_seconds - (time.monotonic() - start))
        results.append({"condition": row["id"], "result": result})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--config", default="configs/paper/learning_core.json")
    a.add_argument("--output", required=True)
    a = sub.add_parser("data")
    a.add_argument("--plan", required=True)
    a.add_argument("--task", action="append")
    a = sub.add_parser("worker")
    a.add_argument("--plan", required=True)
    a.add_argument("--max-seconds", type=float, required=True)
    a.add_argument("--max-runs", type=int, required=True)
    a.add_argument("--experiment", choices=["training_regime", "cross_body_core"])
    a.add_argument("--condition")
    a.add_argument("--qualification", required=True)
    a = sub.add_parser("report")
    a.add_argument("--plan", required=True)
    a.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "plan":
        result = compile_plan(args.config, args.output)["counts"]
    elif args.command == "data":
        result = {
            k: {x: v.get(x) for x in ("fingerprint", "qualified", "qualification_success_rate")}
            for k, v in prepare_data(read_plan(args.plan), args.task).items()
        }
    elif args.command == "report":
        from .learning_analysis import report_learning

        result = report_learning(args.plan, args.output)
    else:
        result = run_worker(
            args.plan,
            max_seconds=args.max_seconds,
            max_runs=args.max_runs,
            experiment=args.experiment,
            condition=args.condition,
            qualification=args.qualification,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
