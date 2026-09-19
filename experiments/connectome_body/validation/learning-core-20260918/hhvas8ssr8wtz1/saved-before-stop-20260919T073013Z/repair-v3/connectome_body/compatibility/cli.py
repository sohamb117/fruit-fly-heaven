"""Independent command line for the paper; does not alter the legacy hover runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from ..graphs import Graph
from ..util import atomic_json
from .analysis import analyze
from .anatomy import prepare_anatomy, prepare_task_subgraphs
from .benchmark import calibrate_rnn_match, profile_controller
from .bodies import BodySpec, make_body
from .datasets import prepare_celegans
from .fish1 import Fish1ExportSpec, export_fish1, import_fish1
from .fish1_circuit import prepare_fish1_hmi
from .fish_model import FishModelConfig, prepare_fish_body
from .paper_analysis import analyze_study
from .predictors import body_features, graph_features
from .qualification import qualify_body
from .run_config import RunSpec
from .runner import read_plan, run_conditions, run_jobs
from .selection import select_conditions
from .study import compile_study
from .subgraphs import prepare_subgraph_controls
from .topology import write_communities
from .training import source_identity, train
from .worm_native import prepare_worm_body


def parser():
    root = argparse.ArgumentParser(
        description="The ten-experiment connectome-body compatibility study"
    )
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Validate inputs and compile all ten experiments; launches no training"
    )
    plan.add_argument("--study", default="configs/paper/study.json")
    plan.add_argument("--output", required=True)
    status = commands.add_parser(
        "status", help="Inventory artifacts without starting or restarting a process"
    )
    status.add_argument("--plan", required=True)
    selection = commands.add_parser(
        "select", help="Review a bounded subset and its full experience budget"
    )
    selection.add_argument("--plan", required=True)
    selection.add_argument("--filters", required=True)
    selection.add_argument("--output", required=True)
    worker = commands.add_parser(
        "worker", help="Execute a bounded number of ready conditions on this machine"
    )
    worker.add_argument("--plan", required=True)
    worker.add_argument("--max-runs", type=int, required=True)
    worker.add_argument("--experiment", choices=list(map(str, range(1, 11))))
    worker.add_argument("--condition", action="append")
    worker.add_argument("--selection", help="Plan-pinned JSON produced by select")
    worker.add_argument(
        "--max-seconds",
        type=float,
        help="Pause at a checkpoint after this allowance; checked at rollout boundaries",
    )
    single = commands.add_parser(
        "train", help="Execute or exactly resume one explicit run configuration"
    )
    single.add_argument("--config", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--resume", action="store_true")
    single.add_argument("--max-seconds", type=float)
    jobs = commands.add_parser(
        "jobs", help="Run post-training causal, transfer and prediction jobs"
    )
    jobs.add_argument("--plan", required=True)
    jobs.add_argument("--max-jobs", type=int, required=True)
    jobs.add_argument("--experiment", choices=list(map(str, range(1, 11))))
    jobs.add_argument("--job", action="append")
    report = commands.add_parser(
        "report", help="All ten experiment reports, including missing and unmeasured evidence"
    )
    report.add_argument("--plan", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--no-figures", action="store_true")
    analysis = commands.add_parser("analyze", help="Analyze terminal runs and adaptation frontiers")
    analysis.add_argument("--root", required=True)
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--include-smoke", action="store_true")
    analysis.add_argument("--no-figures", action="store_true")
    benchmark = commands.add_parser(
        "benchmark", help="Sparse/plastic forward, backward, optimizer and memory scaling"
    )
    benchmark.add_argument("--plan", required=True)
    benchmark.add_argument("--condition", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    benchmark.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32])
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--warmup", type=int, default=2)
    match = commands.add_parser(
        "compute-match", help="Measure conventional RNN widths against one biological condition"
    )
    match.add_argument("--plan", required=True)
    match.add_argument("--condition", required=True)
    match.add_argument("--output", required=True)
    match.add_argument("--hidden-sizes", type=int, nargs="+", required=True)
    match.add_argument("--tolerance", type=float, default=0.2)
    match.add_argument("--repeats", type=int, default=5)
    match.add_argument("--warmup", type=int, default=2)
    communities = commands.add_parser(
        "communities", help="Prepare a reproducible community partition and report degeneracy"
    )
    communities.add_argument("--graph", required=True)
    communities.add_argument("--output", required=True)
    communities.add_argument("--seed", type=int, default=0)
    communities.add_argument("--iterations", type=int, default=50)
    features = commands.add_parser(
        "graph-features", help="Measure graph structure for held-out-pair prediction"
    )
    features.add_argument("--graph", required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--communities")
    features.add_argument("--anatomy")
    features.add_argument("--landmarks", type=int, default=32)
    features.add_argument("--motif-samples", type=int, default=5000)
    features.add_argument("--seed", type=int, default=0)
    body = commands.add_parser("body-features", help="Measure the actual executed body and task")
    body.add_argument("--body", required=True, help="JSON file containing a BodySpec")
    body.add_argument("--output", required=True)
    subgraphs = commands.add_parser(
        "subgraphs", help="Prepare task-relevant and disjoint equal-size controls"
    )
    subgraphs.add_argument("--graph", required=True)
    subgraphs.add_argument("--relevant", required=True)
    subgraphs.add_argument("--irrelevant")
    subgraphs.add_argument("--output", required=True)
    subgraphs.add_argument("--seed", type=int, default=0)
    anatomy = commands.add_parser(
        "prepare-anatomy",
        help="Pin privileged sensory/motor oracles and functional module selections",
    )
    anatomy.add_argument("--kind", choices=["banc", "malecns", "celegans"], required=True)
    anatomy.add_argument("--graph", required=True)
    anatomy.add_argument("--annotations", required=True)
    anatomy.add_argument("--output", required=True)
    anatomy.add_argument("--channels", type=int, nargs="+", default=[16])
    anatomy.add_argument("--subgraphs-output")
    anatomy.add_argument("--max-neurons", type=int, default=2048)
    anatomy.add_argument("--seed", type=int, default=0)
    worm = commands.add_parser(
        "prepare-celegans", help="Import the pinned corrected Cook 2019 neuronal adjacency"
    )
    worm.add_argument("--raw", default="data/raw/celegans-cook2019-corrected2020")
    worm.add_argument("--output", default="data/graphs/celegans-cook2019-chemical")
    worm.add_argument(
        "--connections", choices=["chemical", "electrical", "combined"], default="chemical"
    )
    worm_body = commands.add_parser(
        "prepare-worm-body",
        help="Download pinned published mechanics and build the isolated muscle interface",
    )
    worm_body.add_argument("--output", default="data/paper-bodies/worm")
    fish_body = commands.add_parser(
        "prepare-fish-body",
        help="Prepare the pinned simZFish-derived 3D extension with declared assumptions",
    )
    fish_body.add_argument("--output", default="data/paper-bodies/fish")
    fish_body.add_argument(
        "--config", help="JSON FishModelConfig; use a new output for each variant"
    )
    qualification = commands.add_parser(
        "qualify-body", help="Record native body actuation checks without training"
    )
    qualification.add_argument("--body", choices=["worm", "fish"], required=True)
    qualification.add_argument("--manifest", required=True)
    qualification.add_argument("--output", required=True)
    qualification.add_argument("--seed", type=int, default=0)
    export = commands.add_parser(
        "export-fish1",
        help="Resume an authenticated fixed-materialization export; never remap to current roots",
    )
    export.add_argument("--version", type=int, required=True)
    export.add_argument(
        "--coverage", required=True, help="Document proofreading and completeness limitations"
    )
    export.add_argument(
        "--membership",
        choices=["single_soma_roots", "molecularly_annotated_single_soma_roots"],
        default="single_soma_roots",
    )
    export.add_argument("--segmentation-table", default="fish1_v250915")
    export.add_argument("--page-size", type=int, default=50000)
    export.add_argument(
        "--request-page-size",
        type=int,
        help="Tune future transfer pages (up to 500000) while resuming the original export specification",
    )
    export.add_argument("--output", required=True)
    fish = commands.add_parser(
        "import-fish1", help="Verify exported tables and prepare a lossless sparse Fish1 graph"
    )
    fish.add_argument("--export", required=True)
    fish.add_argument("--output", default="data/graphs/fish1")
    hmi = commands.add_parser(
        "prepare-fish1-hmi", help="Prepare a separate published HMI circuit with binary adjacency"
    )
    hmi.add_argument("--raw", default="data/raw/fish1-paper-circuits")
    hmi.add_argument("--output", default="data/graphs/fish1-hmi-binary")
    return root


def _reference(args):
    plan, _ = read_plan(args.plan)
    condition = next((row for row in plan["conditions"] if row["id"] == args.condition), None)
    if condition is None or condition["status"] != "ready":
        raise ValueError("Specify a ready condition ID from the plan")
    source = RunSpec.from_dict(condition["config"])
    body = plan["inputs"]["bodies"][f"{source.body.name}:{source.body.task}"]
    return condition, source, body


def dispatch(args):
    if args.command == "plan":
        plan = compile_study(args.study, args.output)
        return {
            "plan": str(Path(args.output).resolve() / "plan.json"),
            "fingerprint": plan["fingerprint"],
            "unique_conditions": len(plan["conditions"]),
            "status_counts": dict(Counter(row["status"] for row in plan["conditions"])),
            "post_training_jobs": len(plan["jobs"]),
            "coverage": plan["coverage"],
        }
    if args.command == "status":
        plan, root = read_plan(args.plan, verify_code=False)
        counts = Counter()
        for condition in plan["conditions"]:
            folder = root / "runs" / condition["id"]
            state = condition["status"]
            for name in ("latest.pt", "failure.json", "result.json"):
                if (folder / name).exists():
                    state = {
                        "latest.pt": "checkpoint_available",
                        "failure.json": "failure_record",
                        "result.json": "result_available",
                    }[name]
            counts[state] += 1
        return {
            "plan": plan["fingerprint"],
            "training_artifacts": dict(counts),
            "post_training_results": sum(
                (root / "jobs" / job["id"] / "result.json").is_file() for job in plan["jobs"]
            ),
            "code_unchanged": plan["code_fingerprint"] == source_identity(),
            "note": "Artifact inventory does not infer that a worker is currently live",
        }
    if args.command == "worker":
        result = run_conditions(
            args.plan,
            max_runs=args.max_runs,
            experiment=args.experiment,
            condition_ids=args.condition,
            max_seconds=args.max_seconds,
            selection_path=args.selection,
        )
        return {
            "results": result["results"],
            "skipped_counts": dict(Counter(row["reason"] for row in result["skipped"])),
        }
    if args.command == "select":
        result = select_conditions(args.plan, args.filters, args.output)
        return {key: value for key, value in result.items() if key not in ("rows", "condition_ids")}
    if args.command == "train":
        result = train(
            RunSpec.load(args.config), args.output, resume=args.resume, max_seconds=args.max_seconds
        )
        return {
            key: result[key]
            for key in ("status", "identity", "training_interactions", "optimizer_steps")
        }
    if args.command == "jobs":
        result = run_jobs(
            args.plan, max_jobs=args.max_jobs, experiment=args.experiment, job_ids=args.job
        )
        return {
            "completed": result["completed"],
            "attempted": result["attempted"],
            "pending_count": len(result["pending"]),
            "pending_examples": result["pending"][:5],
        }
    if args.command in ("report", "analyze"):
        if args.command == "report":
            result = analyze_study(args.plan, args.output, figures=not args.no_figures)
            return {
                "report": str(Path(args.output).resolve() / "paper-report.json"),
                "terminal_training_runs": result["terminal_training_runs"],
            }
        result = analyze(
            args.root, args.output, include_smoke=args.include_smoke, figures=not args.no_figures
        )
        return {
            "report": str(Path(args.output).resolve() / "analysis.json"),
            "terminal_runs": result["terminal_runs"],
        }
    if args.command == "compute-match":
        condition, _, body = _reference(args)
        measured = calibrate_rnn_match(
            condition,
            body,
            args.output,
            hidden_sizes=args.hidden_sizes,
            tolerance=args.tolerance,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        return {
            key: measured[key] for key in ("status", "hidden_size", "relative_error", "hardware")
        }
    if args.command == "benchmark":
        condition, source, body = _reference(args)
        if Path(args.output).exists():
            raise FileExistsError("Benchmark outputs are immutable; use another filename")
        if min(args.batches + args.lengths) < 1:
            raise ValueError("Positive batches and BPTT lengths are required")
        record = {
            "schema": "compatibility-benchmark-sweep-v1",
            "reference": condition["id"],
            "code_fingerprint": source_identity(),
            "cases": [],
            "scientific_training_result": False,
        }
        atomic_json(args.output, record)
        for batch in args.batches:
            for length in args.lengths:
                # This finite kernel benchmark does not consume a body's
                # training budget. Its synthetic counts merely validate shapes.
                learning = replace(
                    source.training,
                    num_envs=batch,
                    sequence_length=length,
                    interactions=batch * length,
                    eval_every=batch * length,
                )
                try:
                    result = profile_controller(
                        replace(source, training=learning),
                        body["observation_dim"],
                        body["action_dim"],
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    result = {
                        "status": "failed",
                        "batch": batch,
                        "sequence_length": length,
                        "reason": str(exc),
                    }
                record["cases"].append(result)
                atomic_json(args.output, record)
        return {
            "output": str(Path(args.output).resolve()),
            "cases": [
                {
                    key: row.get(key)
                    for key in (
                        "status",
                        "batch",
                        "sequence_length",
                        "forward_transitions_per_second",
                        "optimized_transitions_per_second",
                        "peak_cuda_allocated_bytes",
                        "reason",
                    )
                }
                for row in record["cases"]
            ],
        }
    if args.command == "communities":
        if Path(args.output).exists():
            raise FileExistsError("Keep the pinned community partition and choose a new path")
        result = write_communities(
            Graph.load(args.graph), args.output, seed=args.seed, iterations=args.iterations
        )
        return {
            key: result[key]
            for key in ("communities", "modularity_undirected", "degenerate", "fingerprint")
        }
    if args.command == "graph-features":
        result = graph_features(
            Graph.load(args.graph),
            community_file=args.communities,
            anatomy_file=args.anatomy,
            landmarks=args.landmarks,
            motif_samples=args.motif_samples,
            seed=args.seed,
        )
        atomic_json(args.output, result)
        return {"output": str(Path(args.output).resolve()), "features": result["features"]}
    if args.command == "body-features":
        body = make_body(BodySpec(**json.loads(Path(args.body).read_text())))
        try:
            result = body_features(body)
        finally:
            body.close()
        atomic_json(args.output, result)
        return result
    if args.command == "subgraphs":
        return prepare_subgraph_controls(
            Graph.load(args.graph),
            args.relevant,
            args.output,
            irrelevant_file=args.irrelevant,
            seed=args.seed,
        )
    if args.command == "prepare-anatomy":
        graph = Graph.load(args.graph)
        result = prepare_anatomy(
            graph, args.kind, args.annotations, args.output, channels=tuple(args.channels)
        )
        if args.subgraphs_output:
            selected = prepare_task_subgraphs(
                graph,
                args.kind,
                args.annotations,
                args.subgraphs_output,
                max_neurons=args.max_neurons,
                seed=args.seed,
            )
            result["subgraphs"] = {
                key: {
                    field: value.get(field)
                    for field in ("status", "selected_neurons", "missing_or_infeasible")
                }
                for key, value in selected.items()
            }
        return result
    if args.command == "prepare-celegans":
        if (Path(args.output) / "manifest.json").exists():
            graph = Graph.load(args.output)
            if (
                graph.manifest["provenance"].get("dataset") != "Cook2019Herm"
                or graph.manifest["provenance"].get("connections") != args.connections
            ):
                raise ValueError(
                    "Existing graph uses different biological data or connection modalities"
                )
        else:
            graph = prepare_celegans(args.raw, args.output, connections=args.connections)
        return {
            "graph": str(Path(args.output).resolve()),
            "fingerprint": graph.fingerprint,
            **graph.manifest["summary"],
        }
    if args.command == "prepare-worm-body":
        return prepare_worm_body(args.output)
    if args.command == "prepare-fish-body":
        config = (
            FishModelConfig(**json.loads(Path(args.config).read_text())) if args.config else None
        )
        return prepare_fish_body(args.output, config)
    if args.command == "qualify-body":
        return qualify_body(args.body, args.manifest, args.output, seed=args.seed)
    if args.command == "export-fish1":
        try:
            from caveclient import CAVEclient, set_session_defaults
            from requests.exceptions import ConnectionError as RequestsConnectionError
            from requests.exceptions import HTTPError, Timeout
        except ImportError as exc:
            raise RuntimeError(
                "Fish1 export needs optional CAVE dependencies; use uv run --no-sync "
                "--with-requirements scripts/fish1-export-requirements.txt. "
                "Core training dependencies are unchanged."
            ) from exc
        spec = Fish1ExportSpec(
            args.version,
            args.coverage,
            membership_mode=args.membership,
            segmentation_table=args.segmentation_table,
            page_size=args.page_size,
        )
        spec.validate()
        set_session_defaults(
            max_retries=5,
            backoff_factor=1,
            backoff_max=30,
            status_forcelist=(429, 500, 502, 503, 504),
        )

        def progress(page):
            print(json.dumps({"event": "fish1_export_page", **page}), file=sys.stderr, flush=True)

        for attempt in range(6):
            try:
                client = CAVEclient(
                    datastack_name=spec.datastack, server_address=spec.server_url, max_retries=5
                )
                result = export_fish1(
                    client,
                    spec,
                    args.output,
                    on_progress=progress,
                    request_page_size=args.request_page_size,
                )
                break
            except (HTTPError, RequestsConnectionError, Timeout) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 5 or (status is not None and status not in (429, 500, 502, 503, 504)):
                    raise
                print(
                    json.dumps(
                        {
                            "event": "fish1_service_retry",
                            "attempt": attempt + 1,
                            "http_status": status,
                            "saved_pages_retained": True,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(30, 2**attempt))
        return {
            "output": str(Path(args.output).resolve()),
            "materialization": result["materialization_version"],
            "tables": result["tables"],
        }
    if args.command == "import-fish1":

        def import_progress(event):
            print(
                json.dumps({"event": "fish1_import_progress", **event}), file=sys.stderr, flush=True
            )

        graph = import_fish1(args.export, args.output, on_progress=import_progress)
        return {
            "fingerprint": graph.fingerprint,
            "graph": str(Path(args.output).resolve()),
            **graph.manifest["summary"],
        }
    if args.command == "prepare-fish1-hmi":
        graph = prepare_fish1_hmi(args.raw, args.output)
        return {"fingerprint": graph.fingerprint, **graph.manifest["summary"]}
    raise AssertionError(args.command)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "exception": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
