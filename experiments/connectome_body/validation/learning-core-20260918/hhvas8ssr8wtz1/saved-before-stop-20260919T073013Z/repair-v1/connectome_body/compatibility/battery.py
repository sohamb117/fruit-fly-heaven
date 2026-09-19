"""Config-driven drone/temporal expansion; planning never launches compute."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from ..graphs import Graph
from ..util import atomic_json, digest_file, digest_json
from .bodies import BodySpec
from .config import AdapterConfig, ControllerConfig, DynamicsConfig, SubstrateConfig
from .design import estimate_parameters
from .drone import CORE_DRONE, DRONE_TASKS, DroneBody, qualify_drone
from .run_config import LearningConfig, RunSpec
from .temporal import CORE_TEMPORAL, TEMPORAL_TASKS, TemporalSpec, build_dataset, sequence
from .temporal_training import DiagnosticRun, train_diagnostic
from .training import source_identity, train


def execution_config(config):
    value = copy.deepcopy(config)
    value["metadata"] = {
        k: v for k, v in value.get("metadata", {}).items() if k.startswith("expected_")
    }
    return value


def execution_key(config):
    """Comparison anchors do not create additional independent training runs."""
    return digest_json(execution_config(config))[:24]


def compile_battery(config_path, output, *, core=False):
    path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(path.read_text())
    if config.get("schema") != "substrate-battery-v1":
        raise ValueError("Unsupported battery schema")
    root = (path.parent / config.get("project_root", "../..")).resolve()
    goal = (root / "../GOAL.md").resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("Frozen battery plan exists; choose a new output version")
    required = {"real", "degree_rewired", "community_rewired", "random"}
    if not required <= set(config["topologies"]):
        raise ValueError("Battery must retain real, degree/community nulls and matched random")
    if any(type(s) is not int or s < 0 for s in config["seeds"]) or not config["seeds"]:
        raise ValueError("Paired training seeds required")
    inputs, graphs = {}, {}
    for name, item in config["graphs"].items():
        declaration = copy.deepcopy(item)
        graph_path = (root / item["path"]).resolve()
        prerequisites = []
        if not (graph_path / "manifest.json").exists():
            graph = None
            prerequisites.append("graph_missing")
        else:
            graph = Graph.load(graph_path)
            if graph.fingerprint != item["fingerprint"]:
                raise ValueError(f"Changed graph {name}")
        qualification = None
        if item.get("qualification"):
            qpath = root / item["qualification"]
            qualification = json.loads(qpath.read_text())
            if qualification.get("graph_fingerprint") != item["fingerprint"]:
                raise ValueError("Qualification belongs to another graph")
            expected = qualification.get("fingerprint")
            content = {k: v for k, v in qualification.items() if k != "fingerprint"}
            if expected != digest_json(content):
                raise ValueError("Graph qualification checksum mismatch")
            if not qualification.get("eligible_for_primary"):
                prerequisites.append("primary_graph_qualification")
        community = (root / item["communities"]).resolve() if item.get("communities") else None
        if community and community.exists():
            record = json.loads(community.read_text())
            if record.get("graph_fingerprint") != item["fingerprint"]:
                raise ValueError("Community partition belongs to another graph")
        else:
            community = None
        inputs[name] = {
            "declaration": declaration,
            "graph_path": str(graph_path),
            "neurons": graph.n if graph else None,
            "edges": graph.m if graph else None,
            "qualification": qualification,
            "community_path": str(community) if community else None,
            "community_sha256": digest_file(community) if community else None,
            "prerequisites": prerequisites,
        }
        graphs[name] = inputs[name]
        del graph
    qualification = qualify_drone()
    if not qualification["passed"]:
        raise ValueError("Engineered drone failed mechanical qualification")
    tasks = []
    for task in CORE_DRONE if core else DRONE_TASKS:
        parameters = dict(config.get("drone_parameters", {}))
        for setting in config.get("severity", {}).get(task, [{}]):
            spec = BodySpec(
                name="drone",
                task=task,
                horizon=config["drone_horizon"],
                parameters={**parameters, **setting},
            )
            body = DroneBody(spec)
            tasks.append(
                {
                    "suite": "drone",
                    "task": task,
                    "setting": setting,
                    "body": spec,
                    "obs": body.obs_dim,
                    "outputs": body.action_dim,
                    "dt": body.control_dt,
                    "body_fingerprint": body.fingerprint,
                }
            )
    datasets = {}
    for task in CORE_TEMPORAL if core else TEMPORAL_TASKS:
        spec = TemporalSpec(task=task, **config["temporal_data"])
        probe = sequence(spec, "train", 0)
        key = f"{task}-{digest_json(asdict(spec))[:12]}"
        directory = (root / config["dataset_root"] / key).resolve()
        datasets[key] = {"path": str(directory), "spec": asdict(spec)}
        tasks.append(
            {
                "suite": "temporal",
                "task": task,
                "setting": {},
                "dataset": str(directory),
                "obs": probe["x"].shape[1],
                "outputs": probe["y"].shape[1],
                "dt": parameters.get("dt", 0.02),
            }
        )
    conditions = []

    def add(task, label, variant, plasticity, budget, seed, controller, graph_info, target=None):
        prereqs = (
            list(graph_info["prerequisites"]) if controller.substrate.kind == "connectome" else []
        )
        if variant == "community_rewired" and not graph_info["community_path"]:
            prereqs.append("community_partition")
        report, error = None, None
        try:
            report = estimate_parameters(
                controller,
                task["obs"],
                task["outputs"],
                neurons=graph_info["neurons"],
                edges=graph_info["edges"],
            )
        except ValueError as exc:
            error = str(exc)
        metadata = {
            "source_substrate": label,
            "group": graph_info["declaration"].get("source_group", label),
            "variant": variant,
            "anchor_plasticity": plasticity,
            "requested_budget": budget,
            "suite": task["suite"],
            "task": task["task"],
            "setting": task["setting"],
            "matched_total_target": target,
            "match_difference": report["total_trainable_parameters"] - target
            if report and target
            else None,
        }
        if controller.substrate.kind == "connectome":
            metadata["expected_graph_fingerprint"] = graph_info["declaration"]["fingerprint"]
            if variant == "community_rewired":
                metadata["expected_community_sha256"] = graph_info["community_sha256"]
        if task["suite"] == "drone":
            metadata["expected_body_fingerprint"] = task["body_fingerprint"]
            run = RunSpec(
                body=task["body"],
                controller=controller,
                training=LearningConfig(**config["drone_training"]),
                train_seed=seed,
                device=config["device"],
                threads=config.get("threads", 1),
                metadata=metadata,
            ).to_dict()
        else:
            run = asdict(
                DiagnosticRun(
                    dataset=task["dataset"],
                    controller=controller,
                    seed=seed,
                    device=config["device"],
                    threads=config.get("threads", 1),
                    metadata=metadata,
                    **config["temporal_training"],
                )
            )
        row = {
            "id": digest_json(run)[:24],
            "execution_id": execution_key(run),
            "suite": task["suite"],
            "task": task["task"],
            "substrate": label,
            "variant": variant,
            "plasticity": plasticity,
            "budget": budget,
            "seed": seed,
            "setting": task["setting"],
            "status": "infeasible" if error else "blocked_prerequisite" if prereqs else "ready",
            "prerequisites": prereqs,
            "error": error,
            "parameters": report,
            "config": run,
        }
        conditions.append(row)
        return report

    for task in tasks:
        for label, graph_info in graphs.items():
            for budget in config["capacities"]:
                adapter = AdapterConfig(**{**config["adapter"], "budget": budget})
                dynamics = DynamicsConfig(**{**config["dynamics"], "control_dt": task["dt"]})
                for seed in config["seeds"]:
                    for plasticity in config["plasticity"]:
                        base = ControllerConfig(
                            adapter=adapter,
                            seed=seed,
                            substrate=SubstrateConfig(
                                graph=graph_info["graph_path"],
                                plasticity=plasticity,
                                dynamics=dynamics,
                                seed=seed,
                                preserve_strengths=config.get("preserve_strengths", True),
                                swap_attempts_per_edge=config.get("swap_attempts_per_edge", 10),
                            ),
                        )
                        real = None
                        for topology in config["topologies"]:
                            community = (
                                graph_info["community_path"]
                                if topology == "community_rewired"
                                else None
                            )
                            # Missing partition remains a prerequisite and is never replaced.
                            if topology == "community_rewired" and not community:
                                community = str(root / "data/missing-community.json")
                            controller = replace(
                                base,
                                substrate=replace(
                                    base.substrate, topology=topology, community_partition=community
                                ),
                            )
                            report = add(
                                task,
                                label,
                                topology,
                                plasticity,
                                budget,
                                seed,
                                controller,
                                graph_info,
                            )
                            if topology == "real":
                                real = report
                        # Brain-free and conventional controls use the same data,
                        # task and optimizer. Dense learned cores match TOTAL active
                        # capacity, while no-brain retains the interface ceiling.
                        target = real["total_trainable_parameters"] if real else budget
                        for kind in ("adapter_only", "rnn", "gru"):
                            recurrent = kind != "adapter_only"
                            sub = SubstrateConfig(
                                kind=kind,
                                plasticity="joint" if recurrent else "adapters",
                                dynamics=dynamics,
                                seed=seed,
                                total_budget=target if recurrent else None,
                            )
                            control_adapter = (
                                replace(adapter, budget=max(128, min(budget, target // 3)))
                                if recurrent
                                else adapter
                            )
                            controller = ControllerConfig(
                                adapter=control_adapter, substrate=sub, seed=seed
                            )
                            add(
                                task,
                                label,
                                kind,
                                plasticity,
                                budget,
                                seed,
                                controller,
                                graph_info,
                                target if recurrent else None,
                            )
    # Identical controls anchored to multiple biological graphs remain explicit
    # paired references. Reports flag duplicate configurations, never count them
    # as independent substrates in predictive analyses.
    plan = {
        "schema": "substrate-battery-plan-v1",
        "source_config": str(path),
        "source_config_sha256": digest_file(path),
        "config_fingerprint": digest_json(config),
        "code_fingerprint": source_identity(),
        "goal_sha256": digest_file(goal),
        "core": core,
        "inputs": inputs,
        "drone_qualification": qualification,
        "datasets": datasets,
        "conditions": conditions,
        "summary": {
            "conditions": len(conditions),
            "unique_training_runs": len({r["execution_id"] for r in conditions}),
            "statuses": dict(Counter(r["status"] for r in conditions)),
            "drone_tasks": len(CORE_DRONE if core else DRONE_TASKS),
            "temporal_tasks": len(CORE_TEMPORAL if core else TEMPORAL_TASKS),
        },
        "comparison_note": "No native-drone biological diagonal; predictive CV groups whole source reconstructions including nulls",
    }
    plan["fingerprint"] = digest_json(plan)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "plan.json", plan)
    (output / "GOAL.md").write_text(goal.read_text())
    atomic_json(output / "config.json", config)
    return plan


def read_battery(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    content = {k: v for k, v in plan.items() if k != "fingerprint"}
    if (
        plan.get("schema") != "substrate-battery-plan-v1"
        or digest_json(content) != plan["fingerprint"]
    ):
        raise ValueError("Battery plan checksum/schema mismatch")
    if plan["code_fingerprint"] != source_identity():
        raise ValueError("Battery code changed; freeze a new plan version")
    if (
        digest_json(json.loads((path.parent / "config.json").read_text()))
        != plan["config_fingerprint"]
    ):
        raise ValueError("Frozen battery configuration changed")
    for item in plan["inputs"].values():
        p = item["community_path"]
        if p and digest_file(p) != item["community_sha256"]:
            raise ValueError("Community inputs changed")
    return plan


def prepare_datasets(plan):
    return {
        key: build_dataset(TemporalSpec(**row["spec"]), row["path"])["fingerprint"]
        for key, row in plan["datasets"].items()
    }


def run_battery(plan_path, *, max_runs, max_seconds, suite=None, tasks=None, condition=None):
    if max_runs < 1 or max_seconds <= 0:
        raise ValueError("Explicit positive run count and wall-time cap required")
    import time

    plan = read_battery(plan_path)
    root = Path(plan_path).resolve().parent / "runs"
    started, results = time.monotonic(), []
    for row in plan["conditions"]:
        if (
            row["status"] != "ready"
            or (suite and row["suite"] != suite)
            or (tasks and row["task"] not in tasks)
            or (condition and row["id"] != condition)
        ):
            continue
        directory = root / row["execution_id"]
        if (directory / "result.json").exists():
            continue
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0 or len(results) >= max_runs:
            break
        resume = (directory / "latest.pt").exists()
        run_config = execution_config(row["config"])
        if row["suite"] == "drone":
            result = train(
                RunSpec.from_dict(run_config), directory, resume=resume, max_seconds=remaining
            )
        else:
            cfg = DiagnosticRun.from_dict(run_config)
            dataset = next(x for x in plan["datasets"].values() if x["path"] == cfg.dataset)
            build_dataset(TemporalSpec(**dataset["spec"]), cfg.dataset)
            result = train_diagnostic(cfg, directory, resume=resume, max_seconds=remaining)
        results.append({"condition": row["id"], "result": result})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Drone, temporal and mechanistic benchmark expansion"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", default="configs/paper/battery.json")
    plan.add_argument("--output", required=True)
    plan.add_argument("--core", action="store_true")
    data = sub.add_parser("datasets")
    data.add_argument("--plan", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--max-runs", type=int, required=True)
    worker.add_argument("--max-seconds", type=float, required=True)
    worker.add_argument("--suite", choices=("drone", "temporal"))
    worker.add_argument("--task", action="append")
    worker.add_argument("--condition")
    report = sub.add_parser("report")
    report.add_argument("--plan", required=True)
    report.add_argument("--output", required=True)
    qualification = sub.add_parser("qualify-drone")
    qualification.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--plan", required=True)
    evaluation.add_argument("--condition", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    if args.command == "plan":
        result = compile_battery(args.config, args.output, core=args.core)["summary"]
    elif args.command == "datasets":
        result = prepare_datasets(read_battery(args.plan))
    elif args.command == "worker":
        result = run_battery(
            args.plan,
            max_runs=args.max_runs,
            max_seconds=args.max_seconds,
            suite=args.suite,
            tasks=args.task,
            condition=args.condition,
        )
    elif args.command == "qualify-drone":
        result = qualify_drone()
        atomic_json(args.output, result)
    elif args.command == "evaluate":
        from .battery_evaluation import evaluate_drone_condition

        result = evaluate_drone_condition(
            args.plan, args.condition, args.output, episodes=args.episodes
        )
    else:
        from .battery_analysis import report_battery

        result = report_battery(args.plan, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
