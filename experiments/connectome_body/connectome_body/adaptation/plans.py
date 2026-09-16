"""Configuration-driven matrices, immutable methodology, and launch-time gates."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

from ..body import PROJECT
from ..graphs import Graph
from ..util import atomic_json, digest_file, digest_json
from .configuration import hover_config_from_dict
from .imitation import AdapterSpec, Optimization, source_identity
from .pipeline import pipeline_identity, run_experiment
from .trajectories import TrajectoryCache, assert_disjoint_scenarios, assert_paired_schema

BASELINES = ("adapter_only", "trainable_gru")
CONDITIONS = ("real", "degree_shuffled", "matched_random", "no_edges", *BASELINES)


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT / path).resolve()


def method_spec(study):
    """Scientific method, independent of which held-out datasets/seeds are requested."""
    body = hover_config_from_dict(study.get("body", {}))
    adapter = dict(study.get("adapter", {}))
    adapter["rate"] = dict(adapter.get("rate", {}), control_dt=body.control_dt)
    adapter = dataclasses.asdict(AdapterSpec.from_dict(adapter))
    for key in ("graph", "variant", "budget", "seed", "port_seed"):
        adapter.pop(key)
    optimizers = {}
    for name in ("offline", "dagger_optimization"):
        opt = Optimization(**study[name])
        opt.validate()
        optimizers[name] = dataclasses.asdict(opt)
    budgets = study["online_budgets"]
    if not budgets or budgets[0] != 0 or any(a >= b for a, b in zip(budgets, budgets[1:])):
        raise ValueError("Experience grid must start at zero and increase")
    if (
        not 0 <= study.get("dagger_beta", 0) <= 1
        or not 0 < study.get("success_threshold", 0.8) <= 1
    ):
        raise ValueError("Invalid mixing or success threshold")
    for name in ("datasets", "conditions", "budgets", "seeds"):
        if not study[name] or len(set(study[name])) != len(study[name]):
            raise ValueError(f"{name} must be unique and nonempty")
    if any(v not in CONDITIONS for v in study["conditions"]) or min(study["budgets"]) < 1:
        raise ValueError("Unknown condition or invalid capacity")
    if min(study.get("validation_episodes", 8), study.get("test_episodes", 50)) < 1:
        raise ValueError("Positive evaluation counts required")
    return {
        "adapter": adapter,
        "body": dataclasses.asdict(body),
        **optimizers,
        "budgets": sorted(study["budgets"]),
        "conditions": sorted(study["conditions"]),
        "online_budgets": budgets,
        "dagger_beta": study.get("dagger_beta", 0.0),
        "validation_episodes": study.get("validation_episodes", 8),
        "test_episodes": study.get("test_episodes", 50),
        "success_threshold": study.get("success_threshold", 0.8),
        "interventions": study.get("interventions", True),
    }


def file_record(path):
    return {"path": str(path), "sha256": digest_file(path)} if path.exists() else None


def verify_record(record):
    if record is not None and digest_file(record["path"]) != record["sha256"]:
        raise ValueError(f"Pinned prerequisite changed: {record['path']}")


def verify_mvp_decision(path):
    from .analysis import adaptation_records, mvp_decision

    report = json.loads(Path(path).read_text())
    if not report.get("go") or report.get("schema") != "connectome-mvp-decision-v1":
        raise ValueError("A complete scientific MVP decision is required")
    if report.get("source_identities") != [pipeline_identity()]:
        raise ValueError("The MVP used a different implementation")
    records = []
    for row in report["run_sources"]:
        result_path = Path(row["path"]) / "result.json"
        if digest_file(result_path) != row["result_sha256"]:
            raise ValueError("An MVP result changed after the decision")
        records.extend(adaptation_records(row["path"]))
    if not mvp_decision(records)["go"]:
        raise ValueError("The saved MVP results do not satisfy the advancement rule")
    return report


def _verify_freeze(record, method):
    frozen = json.loads(Path(record["path"]).read_text())
    if (
        digest_json({k: v for k, v in frozen.items() if k != "fingerprint"})
        != frozen["fingerprint"]
    ):
        raise ValueError("Method freeze changed")
    if frozen["source_identity"] != pipeline_identity() or frozen["method"] != method:
        raise ValueError("Held-out method differs from the frozen BANC method")


def make_plan(
    study, output, *, graph_root=None, cache_root=None, qualification=None, gpu_benchmark=None
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Plans are immutable; choose a new destination")
    method = method_spec(study)
    graph_root = project_path(graph_root or "data/graphs")
    cache_root = project_path(cache_root or "data/hover-v1")
    qualification = project_path(qualification or "validation/hover-teacher-qualification.json")
    benchmark_device = method["adapter"]["device"]
    gpu_benchmark = project_path(
        gpu_benchmark
        or study.get(
            "feasibility_benchmark", f"validation/brief-banc-{benchmark_device}-benchmark.json"
        )
    )
    missing, graph_records, caches, data_records = [], {}, {}, {}
    needs_graph = any(v not in BASELINES for v in study["conditions"])
    for dataset in study["datasets"] if needs_graph else []:
        path = graph_root / dataset
        if not (path / "manifest.json").exists():
            missing.append(f"graph:{dataset}")
        else:
            graph = Graph.load(path)
            if graph.manifest["provenance"].get("is_synthetic", False):
                raise ValueError("Scientific matrices cannot substitute synthetic graph fixtures")
            graph_records[dataset] = {"fingerprint": graph.fingerprint, "path": str(path)}
    for split in ("train", "validation", "test"):
        path = cache_root / split
        if not (path / "manifest.json").exists():
            missing.append(f"teacher_cache:{split}")
        else:
            cache = caches[split] = TrajectoryCache(path)
            metadata = cache.manifest["metadata"]
            if (
                metadata["split"] != split
                or metadata["body_config"] != method["body"]
                or metadata["evidence"] != "native_hover_teacher_data"
            ):
                raise ValueError(
                    "Teacher cache split, body configuration, or evidence is incompatible"
                )
            data_records[split] = {
                "path": str(path),
                "fingerprint": cache.fingerprint,
                "interactions": cache.manifest["interactions"],
            }
    if caches:
        assert_paired_schema(list(caches.values()))
        for first, second in itertools.combinations(caches.values(), 2):
            assert_disjoint_scenarios(first, second)
    qualification_record = file_record(qualification)
    if qualification_record is None:
        missing.append("qualified_hover_teacher")
    else:
        qualified = json.loads(qualification.read_text())
        if (
            not qualified.get("qualified")
            or qualified.get("evidence") != "native_hover_teacher_qualification"
        ):
            missing.append("qualified_hover_teacher")
        elif qualified["body_config"] != method["body"]:
            raise ValueError("Teacher was qualified with a different hover task")
        elif caches and any(
            qualified[key] != next(iter(caches.values())).manifest["metadata"][key]
            for key in ("body_fingerprint", "teacher_fingerprint")
        ):
            raise ValueError("Teacher qualification and trajectory cache differ")
    gpu_record = file_record(gpu_benchmark)
    if study.get("require_gpu_benchmark", True):
        if gpu_record is None:
            missing.append(f"full_banc_{benchmark_device}_benchmark")
        else:
            benchmark = json.loads(gpu_benchmark.read_text())
            if benchmark.get("device") != benchmark_device or benchmark.get("status") != "complete":
                missing.append(f"completed_{benchmark_device}_benchmark")
            if (
                benchmark.get("schema") == "connectome-structural-benchmark-v1"
                and benchmark.get("source_identity") != source_identity()
            ):
                missing.append("benchmark_implementation_identity")
            if (
                "banc" in graph_records
                and benchmark.get("graph_fingerprint") != graph_records["banc"]["fingerprint"]
            ):
                missing.append("benchmark_graph_identity")
            if not benchmark.get("rows") or any(r["status"] != "passed" for r in benchmark["rows"]):
                missing.append("passed_benchmark_rows")
    decision_record, freeze_record = None, None
    if study.get("requires_mvp_go"):
        decision_path = project_path(
            study.get("mvp_decision", "runs/brief-mvp-analysis/decision.json")
        )
        decision_record = file_record(decision_path)
        if decision_record is None or not json.loads(decision_path.read_text()).get("go", False):
            missing.append("mvp_go_decision")
        else:
            verify_mvp_decision(decision_path)
    if study.get("requires_method_freeze"):
        freeze_record = file_record(
            project_path(study.get("method_freeze", "runs/brief-method-freeze.json"))
        )
        if freeze_record is None:
            missing.append("frozen_method_before_heldout_connectomes")
        else:
            _verify_freeze(freeze_record, method)
    reuse = {}
    if study.get("reuse_plan"):
        previous = project_path(study["reuse_plan"])
        if previous.exists():
            old = json.loads(previous.read_text())
            if old["source_identity"] != pipeline_identity():
                raise ValueError("Cannot reuse runs from a different implementation")
            for row in old["runs"]:
                if digest_file(row["config"]) != row["config_sha256"]:
                    raise ValueError("Reuse-plan configuration changed")
                reuse[digest_json(json.loads(Path(row["config"]).read_text()))] = row["output"]
    output.mkdir(parents=True)
    rows = []
    for variant in study["conditions"]:
        datasets = ["baseline"] if variant in BASELINES else study["datasets"]
        for dataset, budget, seed in itertools.product(datasets, study["budgets"], study["seeds"]):
            spec = dict(
                method["adapter"],
                variant=variant,
                budget=budget,
                seed=seed,
                port_seed=seed,
                graph=None if dataset == "baseline" else str(graph_root / dataset),
            )
            config = {
                key: value for key, value in method.items() if key not in ("budgets", "conditions")
            }
            config.update(
                dataset=dataset,
                adapter=dataclasses.asdict(AdapterSpec.from_dict(spec)),
                data={s: str(cache_root / s) for s in ("train", "validation", "test")},
                qualification=str(qualification),
            )
            name = f"{dataset}-{variant}-p{budget}-s{seed}"
            path = output / "configs" / f"{name}.json"
            atomic_json(path, config)
            reused = reuse.get(digest_json(config))
            rows.append(
                {
                    "name": name,
                    "config": str(path.resolve()),
                    "config_sha256": digest_file(path),
                    "output": reused or str((output / "runs" / name).resolve()),
                    "reused_from_prior_plan": bool(reused),
                }
            )
    plan = {
        "schema": "connectome-hover-plan-v1",
        "source_identity": pipeline_identity(),
        "study": study,
        "method": method,
        "runs": rows,
        "planned_runs": len(rows),
        "graphs": graph_records,
        "data": data_records,
        "readiness": "ready" if not missing else "gated",
        "missing": missing,
        "reused_runs": sum(r["reused_from_prior_plan"] for r in rows),
        "online_training_interactions": len(rows) * study["online_budgets"][-1],
        "shared_teacher_interactions": data_records.get("train", {}).get("interactions"),
        "qualification": qualification_record,
        "gpu_benchmark": gpu_record,
        "mvp_decision": decision_record,
        "method_freeze": freeze_record,
        "accounting": "Teacher dataset collected once and shared; per-run exposure and offline presentations reported separately; evaluation adds interactions",
    }
    plan["fingerprint"] = digest_json(plan)
    atomic_json(output / "plan.json", plan)
    return plan


def run_plan(path, shard_index=0, shard_count=1, max_runs=None):
    if not 0 <= shard_index < shard_count or (max_runs is not None and max_runs < 1):
        raise ValueError("Invalid worker shard/count")
    plan = json.loads(Path(path).read_text())
    if digest_json({k: v for k, v in plan.items() if k != "fingerprint"}) != plan["fingerprint"]:
        raise ValueError("Plan changed")
    if plan["readiness"] != "ready":
        raise RuntimeError("Plan is gated: " + ", ".join(plan["missing"]))
    if plan["source_identity"] != pipeline_identity():
        raise ValueError("Plan code changed; generate a new versioned plan")
    for key in ("qualification", "gpu_benchmark", "mvp_decision", "method_freeze"):
        verify_record(plan[key])
    if plan["method_freeze"]:
        _verify_freeze(plan["method_freeze"], plan["method"])
    if plan["mvp_decision"]:
        verify_mvp_decision(plan["mvp_decision"]["path"])
    for row in plan["graphs"].values():
        if Graph.load(row["path"]).fingerprint != row["fingerprint"]:
            raise ValueError("Graph changed after planning")
    for row in plan["data"].values():
        if TrajectoryCache(row["path"]).fingerprint != row["fingerprint"]:
            raise ValueError("Teacher data changed after planning")
    completed, skipped = [], 0
    for idx, row in enumerate(plan["runs"]):
        if idx % shard_count != shard_index:
            continue
        if digest_file(row["config"]) != row["config_sha256"]:
            raise ValueError("Run config changed after planning")
        destination = Path(row["output"])
        config = json.loads(Path(row["config"]).read_text())
        if (destination / "result.json").exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            result = json.loads((destination / "result.json").read_text())
            if (
                manifest["config"] != config
                or manifest["source_identity"] != plan["source_identity"]
                or result["identity"] != manifest["identity"]
            ):
                raise ValueError("Completed run no longer matches this plan")
            skipped += 1
            continue
        result = run_experiment(
            config, destination, resume=(destination / "manifest.json").exists()
        )
        completed.append({"run": row["name"], "status": result["status"]})
        if max_runs is not None and len(completed) >= max_runs:
            break
    return {"completed": completed, "skipped_completed": skipped}


def freeze_method(study, decision, output):
    verify_mvp_decision(decision)
    if study["datasets"] != ["banc"]:
        raise ValueError("Freeze the BANC method after a recorded MVP go decision")
    record = {
        "method": method_spec(study),
        "source_identity": pipeline_identity(),
        "mvp_decision_sha256": digest_file(decision),
        "development_datasets": ["banc"],
    }
    record["fingerprint"] = digest_json(record)
    if Path(output).exists():
        raise FileExistsError("Method freezes are immutable")
    atomic_json(output, record)
    return record
