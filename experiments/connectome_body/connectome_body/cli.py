"""Command line entry points; scientific runs never silently use toy substitutes."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .body import PROJECT
from .config import BodyConfig, PPOConfig, RunConfig
from .data import bootstrap_flybody, import_edges, prepare_banc, prepare_malecns, source_registry
from .graphs import Graph, make_fixture, matched_subgraph
from .policy import BASELINES


def load_config(path):
    config = RunConfig.from_dict(json.loads(Path(path).read_text()))
    if config.graph:
        config = dataclasses.replace(config, graph=str(Path(config.graph).resolve()))
    if config.body.source:
        config = dataclasses.replace(
            config,
            body=dataclasses.replace(config.body, source=str(Path(config.body.source).resolve())),
        )
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "data-status", help="Show source availability, versions, and unresolved datasets"
    )
    bootstrap = commands.add_parser(
        "bootstrap", help="Fetch and verify the pinned stock FlyBody source"
    )
    bootstrap.add_argument("--destination", type=Path)
    prepare = commands.add_parser("prepare", help="Prepare a pinned BANC or MaleCNS graph")
    prepare.add_argument("dataset", choices=["banc", "malecns"])
    prepare.add_argument("--raw", type=Path)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--download", action="store_true")
    generic = commands.add_parser("import-edges", help="Import any documented neuron/synapse graph")
    generic.add_argument("--edges", type=Path, required=True)
    generic.add_argument("--nodes", type=Path, required=True)
    generic.add_argument("--provenance", type=Path, required=True)
    generic.add_argument("--output", type=Path, required=True)
    generic.add_argument("--pre", default="pre")
    generic.add_argument("--post", default="post")
    generic.add_argument("--weight", default="weight")
    generic.add_argument("--node-id", default="id")
    subset = commands.add_parser("matched-subgraph", help="Create an explicit size/density control")
    subset.add_argument("--graph", type=Path, required=True)
    subset.add_argument("--neurons", type=int, required=True)
    subset.add_argument("--edges", type=int, required=True)
    subset.add_argument("--seed", type=int, required=True)
    subset.add_argument("--output", type=Path, required=True)
    fixture = commands.add_parser("fixture", help="Create a graph explicitly marked synthetic")
    fixture.add_argument("--output", type=Path, default=PROJECT / "data/graphs/fixture")
    fixture.add_argument("--neurons", type=int, default=64)
    fixture.add_argument("--seed", type=int, default=0)
    preflight = commands.add_parser(
        "preflight", help="Run actual graph/body forward and backward passes"
    )
    preflight.add_argument("--config", type=Path)
    preflight.add_argument("--graph", type=Path)
    preflight.add_argument("--backend", choices=["flybody", "fixture"], default="flybody")
    preflight.add_argument("--task", choices=["balance", "walk", "reach"], default="balance")
    preflight.add_argument(
        "--substrate",
        choices=["real", "degree_shuffled", "matched_random", "no_edges", *BASELINES],
        default="real",
    )
    preflight.add_argument("--budget", type=int, default=5000)
    preflight.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    preflight.add_argument("--steps", type=int, default=8)
    preflight.add_argument("--output", type=Path, required=True)
    training = commands.add_parser(
        "train", help="Train one configured actor and evaluate held-out tasks"
    )
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--stop-after-updates", type=int)
    smoke = commands.add_parser(
        "train-smoke", help="Short native FlyBody PPO check; never a performance result"
    )
    smoke.add_argument("--graph", type=Path, default=PROJECT / "data/graphs/banc")
    smoke.add_argument(
        "--substrate",
        choices=["real", "degree_shuffled", "matched_random", "no_edges", *BASELINES],
        default="real",
    )
    smoke.add_argument("--task", choices=["balance", "walk", "reach"], default="balance")
    smoke.add_argument("--budget", type=int, default=5000)
    smoke.add_argument("--interactions", type=int, default=32)
    smoke.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--resume", action="store_true")
    plan = commands.add_parser(
        "plan", help="Write an immutable experiment matrix and data readiness report"
    )
    plan.add_argument("--study", type=Path, default=PROJECT / "configs/study.json")
    plan.add_argument("--graph-root", type=Path, default=PROJECT / "data/graphs")
    plan.add_argument("--datasets", nargs="+")
    plan.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser(
        "worker", help="Run a deterministic independent shard of a ready plan"
    )
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, default=0)
    worker.add_argument("--shard-count", type=int, default=1)
    worker.add_argument("--max-runs", type=int)
    report = commands.add_parser(
        "analyze", help="Produce tables, paired effects, and PNG/PDF figures"
    )
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--include-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "data-status":
        result = source_registry()
    elif args.command == "bootstrap":
        result = {"flybody_content_sha256": bootstrap_flybody(args.destination)}
    elif args.command == "prepare":
        raw = args.raw or PROJECT / "data/raw" / args.dataset
        output = args.output or PROJECT / "data/graphs" / args.dataset
        function = prepare_banc if args.dataset == "banc" else prepare_malecns
        result = function(raw, output, args.download).manifest
    elif args.command == "import-edges":
        result = import_edges(
            args.edges,
            args.nodes,
            args.provenance,
            args.output,
            args.pre,
            args.post,
            args.weight,
            args.node_id,
        ).manifest
    elif args.command == "fixture":
        result = make_fixture(args.output, args.neurons, args.seed).manifest
    elif args.command == "matched-subgraph":
        result = matched_subgraph(
            Graph.load(args.graph), args.output, args.neurons, args.edges, args.seed
        ).manifest
    elif args.command == "preflight":
        from .preflight import preflight as run_preflight

        config = (
            load_config(args.config)
            if args.config
            else RunConfig(
                graph=str(args.graph.resolve()) if args.graph else None,
                substrate=args.substrate,
                adapter_budget=args.budget,
                device=args.device,
                body=BodyConfig(backend=args.backend, task=args.task),
            )
        )
        result = run_preflight(config, args.output, args.steps)
    elif args.command in ("train", "train-smoke"):
        from .train import train

        if args.command == "train":
            config = load_config(args.config)
        else:
            config = RunConfig(
                graph=str(args.graph.resolve()) if args.substrate not in BASELINES else None,
                substrate=args.substrate,
                adapter_budget=args.budget,
                device=args.device,
                purpose="smoke",
                body=BodyConfig(task=args.task, horizon=32),
                ppo=PPOConfig(
                    interactions=args.interactions,
                    num_envs=1,
                    rollout_steps=8,
                    sequence_length=4,
                    burn_in=4,
                    epochs=1,
                    eval_every=16,
                    eval_episodes=1,
                    test_episodes=1,
                    threshold_confirmations=2,
                ),
            )
        result = train(config, args.output, args.resume, getattr(args, "stop_after_updates", None))
    elif args.command == "plan":
        from .suite import make_plan

        result = make_plan(
            json.loads(args.study.read_text()), args.graph_root, args.output, args.datasets
        )
        # Full matrix is on disk; keep console output small and reviewable.
        result = {
            key: value for key, value in result.items() if key not in ("runs", "study", "graphs")
        }
    elif args.command == "worker":
        from .suite import run_plan

        result = run_plan(args.plan, args.shard_index, args.shard_count, args.max_runs)
    else:
        from .analyze import analyze

        result = analyze(args.runs, args.output, args.include_validation)
        result = {
            key: value for key, value in result.items() if key not in ("groups", "paired_contrasts")
        }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
