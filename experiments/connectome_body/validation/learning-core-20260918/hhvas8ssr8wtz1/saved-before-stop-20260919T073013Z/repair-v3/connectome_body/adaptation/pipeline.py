"""Offline imitation -> budgeted DAgger -> frozen held-out evaluation."""

from __future__ import annotations

import dataclasses
import fcntl
import json
from pathlib import Path

from ..graphs import Graph
from ..util import atomic_json, digest_file, digest_json, seed_for
from . import PROTOCOL
from .collection import collect_resumable
from .evaluation import evaluate_controller, validate_qualification
from .hover import FlyBodyInterface, HoverConfig
from .imitation import AdapterSpec, Optimization, load_actor, source_identity, train_offline
from .models import INTERVENTIONS
from .teacher import FlightTeacher
from .trajectories import TrajectoryCache, assert_disjoint_scenarios, assert_paired_schema


def pipeline_identity():
    root = Path(__file__).parent
    return digest_json(
        {
            "imitation": source_identity(),
            "pipeline": {
                name: digest_file(root / name)
                for name in (
                    "pipeline.py",
                    "collection.py",
                    "evaluation.py",
                    "hover.py",
                    "teacher.py",
                    "assets.py",
                )
            },
        }
    )


def _run_experiment(config, output, resume):
    spec = AdapterSpec.from_dict(config["adapter"])
    body_config = HoverConfig(**config.get("body", {}))
    if abs(spec.rate.control_dt - body_config.control_dt) > 1e-12:
        raise ValueError("Neural and body policy clocks differ")
    budgets = config.get("online_budgets", [0, 5000, 20000])
    if not budgets or budgets[0] != 0 or any(a >= b for a, b in zip(budgets, budgets[1:])):
        raise ValueError("Online budgets must start at zero and strictly increase")
    smoke = bool(config.get("smoke", False))
    if not smoke and spec.graph is not None:
        graph = Graph.load(spec.graph)
        if graph.manifest["provenance"].get("is_synthetic", False):
            raise ValueError("Synthetic graph fixtures are restricted to explicit smoke runs")
    train = TrajectoryCache(config["data"]["train"])
    validation = TrajectoryCache(config["data"]["validation"])
    heldout = TrajectoryCache(config["data"]["test"])
    assert_paired_schema([train, validation, heldout])
    assert_disjoint_scenarios(train, validation)
    assert_disjoint_scenarios(train, heldout)
    assert_disjoint_scenarios(validation, heldout)
    if [c.manifest["metadata"]["split"] for c in (train, validation, heldout)] != [
        "train",
        "validation",
        "test",
    ]:
        raise ValueError("Train/validation/test split labels differ from their roles")
    if not smoke and any(
        c.manifest["metadata"]["evidence"] != "native_hover_teacher_data"
        for c in (train, validation, heldout)
    ):
        raise ValueError("Scientific runs may not use smoke or synthetic teacher caches")
    teacher = FlightTeacher(config.get("teacher_path"))
    body = FlyBodyInterface(body_config)
    try:
        if train.manifest["metadata"]["body_fingerprint"] != body.fingerprint:
            raise ValueError("Cache body differs from closed-loop evaluation body")
        if train.manifest["metadata"]["teacher_fingerprint"] != teacher.fingerprint:
            raise ValueError("Cache teacher differs from DAgger oracle")
        if not smoke:
            validate_qualification(config["qualification"], body, teacher)
        manifest = {
            "protocol": PROTOCOL,
            "source_identity": pipeline_identity(),
            "config": config,
            "data": {
                "train": train.fingerprint,
                "validation": validation.fingerprint,
                "test": heldout.fingerprint,
            },
            "body_fingerprint": body.fingerprint,
            "teacher_fingerprint": teacher.fingerprint,
            "teacher_pretraining_cost": "Published pretrained teacher; external pretraining cost not measured here",
            "evidence": "native_hover_smoke" if smoke else "native_hover_adaptation_experiment",
        }
        manifest["identity"] = digest_json(manifest)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing experiment identity differs or --resume was omitted")
        elif resume:
            raise FileNotFoundError("No experiment manifest to resume")
        atomic_json(manifest_path, manifest)
        if (output / "result.json").exists():
            return json.loads((output / "result.json").read_text())
        data_paths = [train.path]
        stages = []
        previous_checkpoint = None
        previous_actor = None
        evaluation_count = int(config.get("validation_episodes", 8))
        test_count = int(config.get("test_episodes", 50))
        if min(evaluation_count, test_count) < 1:
            raise ValueError("Positive evaluation episode counts required")
        # These cohorts pair every condition and capacity within a training seed.
        validation_seeds = [
            seed_for(spec.seed, f"closed-validation:{i}") for i in range(evaluation_count)
        ]
        test_seeds = [seed_for(spec.seed, f"closed-test:{i}") for i in range(test_count)]
        for stage, budget in enumerate(budgets):
            stage_path = output / f"stage-{stage:02d}"
            if stage:
                cache_path = output / f"dagger-{stage:02d}"
                added = collect_resumable(
                    body_config,
                    cache_path,
                    actor=previous_actor,
                    qualification=config.get("qualification"),
                    teacher_path=config.get("teacher_path"),
                    seed=seed_for(spec.seed, f"dagger:{stage}"),
                    interactions=budget - budgets[stage - 1],
                    beta=float(config.get("dagger_beta", 0)),
                    smoke=smoke,
                    resume=cache_path.exists(),
                )
                if added.manifest["interactions"] != budget - budgets[stage - 1]:
                    raise ValueError(
                        "DAgger cache exceeds or misses the exact experience increment"
                    )
                data_paths.append(cache_path)
            opt = Optimization(**config["offline" if stage == 0 else "dagger_optimization"])
            if not (stage_path / "result.json").exists():
                train_offline(
                    spec,
                    opt,
                    data_paths,
                    validation.path,
                    stage_path,
                    resume=(stage_path / "latest.pt").exists(),
                    initial_checkpoint=previous_checkpoint,
                )
            actor, _, stage_manifest = load_actor(stage_path)
            if stage_manifest["adapter"] != dataclasses.asdict(spec) or stage_manifest[
                "optimization"
            ] != dataclasses.asdict(opt):
                raise ValueError("Stage checkpoint does not match the experiment configuration")
            if stage_manifest["training_caches"] != [
                TrajectoryCache(path).fingerprint for path in data_paths
            ]:
                raise ValueError("Stage checkpoint was trained on a different dataset prefix")
            previous_actor, previous_checkpoint = actor, stage_path / "best.pt"
            stage_eval = output / f"stage-{stage:02d}-evaluation.json"
            if stage_eval.exists():
                row = json.loads(stage_eval.read_text())
                if (
                    row["stage_identity"] != stage_manifest["identity"]
                    or row["online_interactions"] != budget
                ):
                    raise ValueError(
                        "Saved stage evaluation has a different actor or experience budget"
                    )
            else:
                metrics = evaluate_controller(body, seeds=validation_seeds, actor=actor)
                row = {
                    "stage": stage,
                    "online_interactions": budget,
                    "common_teacher_interactions": train.manifest["interactions"],
                    "total_training_experience": train.manifest["interactions"] + budget,
                    "dagger_teacher_queries": budget,
                    "validation": metrics,
                    "stage_identity": stage_manifest["identity"],
                }
                atomic_json(stage_eval, row)
            stages.append(row)
            atomic_json(output / "frontier.json", stages)
        # Final test is accessed once, after stage/checkpoint selection on validation only.
        selected = max(
            range(len(stages)),
            key=lambda i: (stages[i]["validation"]["success"], stages[i]["validation"]["score"]),
        )
        prefix_selections = [
            max(
                range(index + 1),
                key=lambda i: (
                    stages[i]["validation"]["success"],
                    stages[i]["validation"]["score"],
                ),
            )
            for index in range(len(stages))
        ]
        atomic_json(
            output / "selection.json",
            {
                "selected_stage": selected,
                "prefix_selections": prefix_selections,
                "basis": "Validation only; recorded before all held-out controller evaluations",
            },
        )
        actor, _, actor_manifest = load_actor(output / f"stage-{selected:02d}")
        test = evaluate_controller(body, seeds=test_seeds, split="test", actor=actor)
        # Freeze every prefix selection before opening the test set. This gives
        # held-out J(P,N) while preventing test performance from choosing N or weights.
        test_by_selection = {selected: test}
        test_frontier = []
        for index, stage_row in enumerate(stages):
            prefix_selected = prefix_selections[index]
            if prefix_selected not in test_by_selection:
                prefix_actor, _, _ = load_actor(output / f"stage-{prefix_selected:02d}")
                test_by_selection[prefix_selected] = evaluate_controller(
                    body, seeds=test_seeds, split="test", actor=prefix_actor
                )
                del prefix_actor
            test_frontier.append(
                {
                    "online_interactions": stage_row["online_interactions"],
                    "total_training_experience": stage_row["total_training_experience"],
                    "selected_stage": prefix_selected,
                    "test": test_by_selection[prefix_selected],
                }
            )
        ood = evaluate_controller(body, seeds=test_seeds, split="ood", actor=actor)
        interventions = {}
        if config.get("interventions", True):
            names = (
                INTERVENTIONS[1:]
                if actor.substrate is not None
                else (
                    ("reset", "state_shuffle", "silence") if actor.gru is not None else ("silence",)
                )
            )
            for name in names:
                interventions[name] = evaluate_controller(
                    body, seeds=test_seeds, split="test", actor=actor, intervention=name
                )
        tau = float(config.get("success_threshold", 0.8))
        crossing = next((r for r in stages if r["validation"]["success"] >= tau), None)
        result = {
            "status": "complete",
            "identity": manifest["identity"],
            "evidence": manifest["evidence"],
            "parameters": actor.parameter_report,
            "graph": actor.substrate.report if actor.substrate else None,
            "frontier": stages,
            "selected_stage": selected,
            "experience": {
                "event": crossing is not None,
                "online_interactions": crossing["online_interactions"] if crossing else budgets[-1],
                "total_training_experience": crossing["total_training_experience"]
                if crossing
                else train.manifest["interactions"] + budgets[-1],
                "censor_online_interactions": budgets[-1],
                "success_threshold": tau,
                "definition": "first tested experience grid point above validation threshold; failures right-censored",
            },
            "test": test,
            "test_frontier": test_frontier,
            "ood": ood,
            "interventions": interventions,
            "selected_actor_identity": actor_manifest["identity"],
            "common_teacher_interactions": train.manifest["interactions"],
            "online_interactions": budgets[-1],
            "recovery": {
                "possible_replayed_training_interactions_upper_bound": sum(
                    TrajectoryCache(path)
                    .manifest["metadata"]
                    .get("recovery", {})
                    .get("possible_replayed_interactions_upper_bound", 0)
                    for path in data_paths[1:]
                ),
                "definition": "Training experience counts committed labeled transitions; interrupted episodes may replay, with extra simulator work reported separately",
            },
            "evaluation_interactions": sum(
                r["validation"]["evaluation_interactions"] for r in stages
            )
            + sum(r["evaluation_interactions"] for r in test_by_selection.values())
            + ood["evaluation_interactions"]
            + sum(r["evaluation_interactions"] for r in interventions.values()),
        }
        atomic_json(output / "result.json", result)
        return result
    finally:
        body.close()


def run_experiment(config, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This experiment already has an active worker") from exc
        try:
            return _run_experiment(config, output, resume)
        except Exception as exc:
            atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
