"""Hover adaptation experiments; separate from the earlier ground/PPO protocol."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from ..body import PROJECT
from .configuration import hover_config_from_dict


def read_json(path):
    return json.loads(Path(path).read_text())


def body_config(path):
    value = read_json(path) if path else {}
    return hover_config_from_dict(value.get("body", value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("assets", help="Verify or fetch the pinned public flight assets")
    assets.add_argument("--output", type=Path)
    assets.add_argument("--download", action="store_true")
    temporal = commands.add_parser("temporal-data", help="Balanced memory sanity trajectories")
    temporal.add_argument(
        "--task", choices=["memory", "delayed_xor", "oscillator"], default="memory"
    )
    temporal.add_argument("--episodes", type=int, default=128)
    temporal.add_argument("--length", type=int, default=24)
    temporal.add_argument("--split", choices=["train", "validation", "test"], default="train")
    temporal.add_argument("--seed", type=int, default=0)
    temporal.add_argument("--output", type=Path, required=True)
    qualification = commands.add_parser(
        "teacher-check", help="Native hover teacher and WPG-only check"
    )
    qualification.add_argument("--config", type=Path)
    qualification.add_argument("--episodes", type=int, default=8)
    qualification.add_argument("--seed", type=int, default=910000)
    qualification.add_argument("--teacher-path", type=Path)
    qualification.add_argument("--smoke", action="store_true")
    qualification.add_argument("--output", type=Path, required=True)
    collection = commands.add_parser(
        "collect", help="Collect immutable shared teacher trajectories"
    )
    collection.add_argument("--config", type=Path)
    collection.add_argument("--split", choices=["train", "validation", "test"], required=True)
    collection.add_argument("--episodes", type=int, default=32)
    collection.add_argument("--seed", type=int, default=1000)
    collection.add_argument("--teacher-path", type=Path)
    collection.add_argument(
        "--qualification",
        type=Path,
        default=PROJECT / "validation/hover-teacher-qualification.json",
    )
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--smoke", action="store_true")
    collection.add_argument("--resume", action="store_true")
    offline = commands.add_parser("offline", help="Imitation only; no simulator interactions")
    offline.add_argument("--config", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--resume", action="store_true")
    offline.add_argument("--stop-after-updates", type=int)
    experiment = commands.add_parser("run", help="One offline/DAgger/evaluation experiment")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--resume", action="store_true")
    plan = commands.add_parser("plan", help="Write an immutable matrix, including readiness gates")
    plan.add_argument("--study", type=Path, default=PROJECT / "configs/brief/mvp.json")
    plan.add_argument("--graph-root", type=Path)
    plan.add_argument("--cache-root", type=Path)
    plan.add_argument("--qualification", type=Path)
    plan.add_argument("--gpu-benchmark", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser(
        "worker", help="Run or resume one deterministic shard of a ready plan"
    )
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, default=0)
    worker.add_argument("--shard-count", type=int, default=1)
    worker.add_argument("--max-runs", type=int)
    analysis = commands.add_parser(
        "analyze", help="Held-out frontiers, paired controls, and figures"
    )
    analysis.add_argument("--runs", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--include-validation", action="store_true")
    freeze = commands.add_parser(
        "freeze", help="Lock methodology before exposing held-out connectomes"
    )
    freeze.add_argument("--study", type=Path, required=True)
    freeze.add_argument("--decision", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "assets":
        from .assets import prepare_assets

        result = prepare_assets(args.output, args.download)
    elif args.command == "temporal-data":
        from .temporal import temporal_cache

        result = temporal_cache(
            args.output, args.task, args.episodes, args.length, args.seed, args.split
        ).manifest
    elif args.command == "teacher-check":
        from .evaluation import qualify_teacher

        report = qualify_teacher(
            body_config(args.config),
            args.output,
            args.episodes,
            args.seed,
            args.teacher_path,
            args.smoke,
        )
        result = {
            "status": report["status"],
            "teacher_success": report["teacher"]["success"],
            "wpg_only_success": report["wpg_zero_modulation"]["success"],
            "report": str(args.output),
        }
    elif args.command == "collect":
        from .collection import collect_resumable

        cache = collect_resumable(
            body_config(args.config),
            args.output,
            split=args.split,
            episodes=args.episodes,
            seed=args.seed,
            teacher_path=args.teacher_path,
            qualification=args.qualification,
            smoke=args.smoke,
            resume=args.resume,
        )
        result = {
            "fingerprint": cache.fingerprint,
            "interactions": cache.manifest["interactions"],
            "path": str(cache.path),
        }
    elif args.command == "offline":
        from .imitation import AdapterSpec, Optimization, train_offline

        config = read_json(args.config)
        train = config["data"]["train"]
        result = train_offline(
            AdapterSpec.from_dict(config["adapter"]),
            Optimization(**config["offline"]),
            train if isinstance(train, list) else [train],
            config["data"]["validation"],
            args.output,
            resume=args.resume,
            stop_after=args.stop_after_updates,
        )
    elif args.command == "run":
        from .pipeline import run_experiment

        config = read_json(args.config)
        config["body"] = dataclasses.asdict(hover_config_from_dict(config.get("body", {})))
        report = run_experiment(config, args.output, args.resume)
        result = {
            "status": report["status"],
            "evidence": report["evidence"],
            "test_success": report["test"]["success"],
            "result": str(args.output / "result.json"),
        }
    elif args.command == "plan":
        from .plans import make_plan

        report = make_plan(
            read_json(args.study),
            args.output,
            graph_root=args.graph_root,
            cache_root=args.cache_root,
            qualification=args.qualification,
            gpu_benchmark=args.gpu_benchmark,
        )
        result = {
            key: report[key]
            for key in ("planned_runs", "readiness", "missing", "online_training_interactions")
        }
        result["plan"] = str(args.output / "plan.json")
    elif args.command == "worker":
        from .plans import run_plan

        result = run_plan(args.plan, args.shard_index, args.shard_count, args.max_runs)
    elif args.command == "analyze":
        from .analysis import analyze

        report = analyze(args.runs, args.output, args.include_validation)
        result = {
            "completed_runs": report["completed_runs"],
            "report": str(args.output / "report.md"),
        }
    else:
        from .plans import freeze_method

        result = freeze_method(read_json(args.study), args.decision, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
