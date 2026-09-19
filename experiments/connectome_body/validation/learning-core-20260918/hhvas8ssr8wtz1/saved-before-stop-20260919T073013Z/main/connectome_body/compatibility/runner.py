"""Local/allocated-machine execution of an immutable plan; no cloud provisioning."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from ..graphs import Graph
from ..util import atomic_json, digest_file, digest_json
from .analysis import collect_runs
from .bodies import BodySpec, make_body
from .controller import Controller
from .evaluation import InterventionSpec
from .interventions import calibrate_temporal_mean, evaluate_intervention
from .predictors import graph_features, write_prediction
from .robustness import PerturbationSpec, make_perturbed_body
from .run_config import RunSpec
from .training import controller_identity, policy_digest, source_identity, train


def read_plan(path, *, verify_code=True):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema") != "compatibility-study-plan-v1" or value.get(
        "fingerprint"
    ) != digest_json({key: item for key, item in value.items() if key != "fingerprint"}):
        raise ValueError("Study plan schema or fingerprint mismatch")
    if verify_code and value["code_fingerprint"] != source_identity():
        raise ValueError("Source changed since planning; preserve this plan and compile a new one")
    return value, path.parent


@contextmanager
def claim(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / "worker.lock").open("a")
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BlockingIOError("Another worker has claimed this condition") from exc
        yield
    finally:
        stream.close()


def completed_training(directory, condition, *, allow_failed=False):
    directory = Path(directory)
    failure = directory / "failure.json"
    if allow_failed and failure.is_file():
        record = json.loads(failure.read_text())
        if record.get("status") == "failed":
            manifest = json.loads((directory / "manifest.json").read_text())
            if (
                manifest.get("config") != condition["config"]
                or record.get("identity") != manifest.get("identity")
                or record.get("manifest_sha256") != digest_json(manifest)
                or record.get("exception") != "FloatingPointError"
                or record.get("fingerprint")
                != digest_json(
                    {key: value for key, value in record.items() if key != "fingerprint"}
                )
            ):
                raise ValueError("Numerical failure configuration, identity or seal mismatch")
            return True
    if not (directory / "result.json").is_file():
        return False
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if manifest.get("config") != condition["config"]:
        raise ValueError("Completed run configuration differs from the planned condition")
    if result.get("identity") != manifest.get("identity") or result.get(
        "manifest_sha256"
    ) != digest_json(manifest):
        raise ValueError("Completed run identity or manifest seal mismatch")
    return result.get("status") == "complete"


def run_conditions(
    plan_path,
    *,
    max_runs,
    experiment=None,
    condition_ids=None,
    max_seconds=None,
    selection_path=None,
):
    if type(max_runs) is not int or max_runs < 1:
        raise ValueError("An explicit positive max_runs is required")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("Checkpoint time allowance must be positive")
    plan, root = read_plan(plan_path)
    if selection_path:
        from .selection import selection_ids

        if condition_ids:
            raise ValueError("Choose a selection or explicit condition IDs, not both")
        condition_ids = selection_ids(plan, selection_path)
    available = {item["id"] for item in plan["conditions"]}
    if condition_ids and not set(condition_ids) <= available:
        raise ValueError("Unknown condition identifier")
    results, skipped = [], []
    for condition in plan["conditions"]:
        identity = condition["id"]
        if experiment is not None and str(experiment) not in condition["experiments"]:
            continue
        if condition_ids and identity not in condition_ids:
            continue
        if condition["status"] != "ready":
            skipped.append({"condition": identity, "reason": condition["status"]})
            continue
        directory = root / "runs" / identity
        try:
            with claim(directory):
                if completed_training(directory, condition):
                    skipped.append({"condition": identity, "reason": "already_complete"})
                    continue
                if completed_training(directory, condition, allow_failed=True):
                    skipped.append({"condition": identity, "reason": "terminal_failure_retained"})
                    continue
                config = RunSpec.from_dict(condition["config"])
                resume = (directory / "latest.pt").is_file()
                try:
                    result = train(config, directory, resume=resume, max_seconds=max_seconds)
                except Exception as exc:
                    # Startup problems are not manufactured zero-performance
                    # observations. A training numerical failure has its own
                    # explicit failure.json written by the learner.
                    diagnostic = {
                        "condition": identity,
                        "status": "execution_error",
                        "exception": type(exc).__name__,
                        "message": str(exc),
                    }
                    atomic_json(directory / "execution_error.json", diagnostic)
                    results.append(diagnostic)
                    break
                results.append(
                    {
                        "condition": identity,
                        "status": result["status"],
                        "training_interactions": result["training_interactions"],
                        "resumed": resume,
                    }
                )
                if result["status"] == "paused_at_checkpoint":
                    break
        except BlockingIOError:
            skipped.append({"condition": identity, "reason": "live_worker"})
            continue
        if len(results) >= max_runs:
            break
    report = {
        "schema": "compatibility-worker-report-v1",
        "plan_fingerprint": plan["fingerprint"],
        "results": results,
        "skipped": skipped,
        "cloud_resources_created": False,
    }
    return report


@contextmanager
def selected_policy(
    root, condition, *, body_spec=None, perturbation=None, expected_body_fingerprint=None
):
    directory = Path(root) / "runs" / condition["id"]
    if not completed_training(directory, condition):
        raise FileNotFoundError("A completed parent run and selected checkpoint are required")
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    config = RunSpec.from_dict(condition["config"])
    target = body_spec or config.body
    body = make_body(target)
    try:
        expected = expected_body_fingerprint or (
            config.metadata.get("expected_body_fingerprint") if body_spec is None else None
        )
        if expected and body.fingerprint != expected:
            raise ValueError("Evaluation body changed since planning")
        if perturbation is not None:
            body.close()
            body = make_perturbed_body(target, perturbation)
        if (
            body.obs_dim != manifest["observation_dim"]
            or body.action_names != manifest["action_names"]
        ):
            raise ValueError("Transfer must preserve observation dimensions and actuator semantics")
        if digest_json(body.obs_schema) != digest_json(manifest["observation_schema"]):
            raise ValueError("Related-task transfer must preserve named observation semantics")
        if target.name != config.body.name:
            raise ValueError("Related-task transfer keeps the same physical body")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; evaluation never silently changes devices")
        controller = Controller(body.obs_dim, body.action_dim, config.controller).to(config.device)
        if controller_identity(controller) != manifest["controller_identity"]:
            raise ValueError("Selected checkpoint controller or graph identity mismatch")
        path = directory / "best.pt"
        checksum = digest_file(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("identity") != manifest["identity"] or checksum != digest_file(path):
            raise ValueError("Checkpoint identity changed or belongs to another run")
        controller.load_state_dict(saved["actor"])
        if policy_digest(controller) != result["selected_policy_digest"]:
            raise ValueError("Selected policy does not match the finalized result")
        yield (
            controller,
            body,
            config,
            {
                "checkpoint_sha256": checksum,
                "source_identity": manifest["identity"],
                "source_training_interactions": result["training_interactions"],
            },
        )
    finally:
        body.close()


def _execute_job(plan, root, job):
    output = root / "jobs" / job["id"]
    parameters = job["parameters"]
    conditions = {item["id"]: item for item in plan["conditions"]}
    if job["kind"] == "predict_pairs":
        collected = collect_runs(root / "runs", include_smoke=plan["study"]["purpose"] == "smoke")
        ids = {str(root / "runs" / identity) for identity in job["dependencies"]}
        rows = [row for row in collected["rows"] if row["run"] in ids]
        graphs = {}
        for name in sorted({row["connectome"] for row in rows}):
            declaration = plan["inputs"]["graphs"][name]
            graph = Graph.load(declaration["path"])
            if graph.fingerprint != declaration["fingerprint"]:
                raise ValueError("Prediction graph differs from the preregistered substrate")
            inputs = declaration.get("descriptor_inputs", {})
            # Optional anatomical descriptors remain explicitly unknown; never
            # create them from neuron-name/ID heuristics.
            paths = {}
            for argument, field in (("community_file", "communities"), ("anatomy_file", "anatomy")):
                item = inputs.get(field, {})
                path, expected = item.get("path"), item.get("sha256")
                if expected and (not Path(path).is_file() or digest_file(path) != expected):
                    raise ValueError("Structural predictor input changed after preregistration")
                # Unknown at planning remains unknown even if annotations are
                # subsequently added. A new input requires a new study plan.
                paths[argument] = path if expected else None
            graphs[name] = graph_features(graph, **paths)
        bodies = {
            key: value["features"]
            for key, value in plan["inputs"]["bodies"].items()
            if "features" in value
        }
        report = write_prediction(
            rows, graphs, bodies, output / "prediction.json", outcome=parameters["outcome"]
        )
        return {
            "prediction_file": str(output / "prediction.json"),
            "pairs": report["pairs"],
            "structural_mse_reduction": report["structural_mse_reduction"],
        }
    condition = conditions[job["reference"]]
    body_spec = BodySpec(**parameters["body"]) if "body" in parameters else None
    perturbation = (
        PerturbationSpec(**parameters["perturbation"]) if "perturbation" in parameters else None
    )
    target_spec = body_spec or RunSpec.from_dict(condition["config"]).body
    expected_body = plan["inputs"]["bodies"][f"{target_spec.name}:{target_spec.task}"].get(
        "fingerprint"
    )
    with selected_policy(
        root,
        condition,
        body_spec=body_spec,
        perturbation=perturbation,
        expected_body_fingerprint=expected_body,
    ) as (
        controller,
        body,
        config,
        provenance,
    ):
        if job["kind"] == "finetune":
            next_config = replace(
                config,
                body=body_spec or config.body,
                training=replace(config.training, interactions=parameters["interactions"]),
                initial_checkpoint=str(root / "runs" / condition["id"] / "best.pt"),
                perturbation=perturbation,
                metadata={
                    **config.metadata,
                    "experiments": ["9"],
                    "transfer_parent": condition["id"],
                    "expected_body_fingerprint": body.fingerprint,
                },
            )
            # Do not retain a second full plastic connectome on the GPU while
            # Trainer constructs the fine-tuning controller.
            del controller
        elif job["kind"] == "calibrate_mean":
            path = output / "activity_mean.npz"
            report = calibrate_temporal_mean(
                controller,
                body,
                path,
                seed=config.train_seed,
                episodes=parameters["episodes"],
                split=parameters["split"],
            )
            return {**provenance, "activity_mean": str(path), "calibration": report}
        else:
            spec = {
                key: value
                for key, value in parameters.items()
                if key in InterventionSpec.__dataclass_fields__
            }
            spec.setdefault("kind", "none")
            if spec["kind"] == "temporal_mean":
                mean_jobs = [
                    identity for identity in job["dependencies"] if identity != job["reference"]
                ]
                if len(mean_jobs) != 1:
                    raise ValueError(
                        "Temporal-mean intervention needs exactly one development calibration"
                    )
                spec["mean_file"] = str(root / "jobs" / mean_jobs[0] / "activity_mean.npz")
            measured = evaluate_intervention(
                controller,
                body,
                InterventionSpec(**spec),
                seed=config.train_seed,
                episodes=config.training.test_episodes,
                output=output / "evaluation.json",
            )
            return {
                **provenance,
                "evaluation": measured,
                "perturbation": parameters.get("perturbation"),
                "target_body": parameters.get("body"),
                "retrained": False,
            }
    # selected_policy's context has now released the parent controller/body.
    return {
        **provenance,
        "fine_tune": train(
            next_config, output / "training", resume=(output / "training/latest.pt").is_file()
        ),
    }


def run_jobs(plan_path, *, max_jobs, experiment=None, job_ids=None):
    if type(max_jobs) is not int or max_jobs < 1:
        raise ValueError("An explicit positive max_jobs is required")
    plan, root = read_plan(plan_path)
    conditions = {item["id"]: item for item in plan["conditions"]}
    jobs = {item["id"]: item for item in plan["jobs"]}
    if job_ids and not set(job_ids) <= jobs.keys():
        raise ValueError("Unknown post-training job identifier")
    completed, pending, attempted = [], [], 0
    for job in plan["jobs"]:
        if attempted >= max_jobs:
            break
        if experiment is not None and str(experiment) not in job["experiments"]:
            continue
        if job_ids and job["id"] not in job_ids:
            continue
        directory = root / "jobs" / job["id"]
        invalid_inputs = [
            path
            for path, expected in job.get("input_files", {}).items()
            if expected is None or not Path(path).is_file() or digest_file(path) != expected
        ]
        if invalid_inputs:
            pending.append(
                {
                    "job": job["id"],
                    "reason": "missing_or_changed_preregistered_input",
                    "files": invalid_inputs,
                }
            )
            continue
        waiting = []
        for dependency in job["dependencies"]:
            if dependency in conditions:
                if not completed_training(
                    root / "runs" / dependency,
                    conditions[dependency],
                    allow_failed=job["kind"] == "predict_pairs",
                ):
                    waiting.append(dependency)
            elif not (root / "jobs" / dependency / "result.json").is_file():
                waiting.append(dependency)
        if waiting:
            pending.append(
                {"job": job["id"], "reason": "unfinished_dependencies", "dependencies": waiting}
            )
            continue
        try:
            with claim(directory):
                request = {"plan_fingerprint": plan["fingerprint"], "job": job}
                identity = digest_json(request)
                if (directory / "result.json").exists():
                    saved = json.loads((directory / "result.json").read_text())
                    if saved.get("request_identity") != identity or saved.get(
                        "fingerprint"
                    ) != digest_json(
                        {key: value for key, value in saved.items() if key != "fingerprint"}
                    ):
                        raise ValueError("Post-training result belongs to another request")
                    continue
                atomic_json(directory / "request.json", {**request, "identity": identity})
                attempted += 1
                try:
                    result = _execute_job(plan, root, job)
                except (OSError, ValueError, RuntimeError) as exc:
                    diagnostic = {
                        "job": job["id"],
                        "reason": str(exc),
                        "exception": type(exc).__name__,
                    }
                    atomic_json(directory / "error.json", diagnostic)
                    pending.append(diagnostic)
                    continue
                record = {
                    "schema": "compatibility-job-result-v1",
                    "status": "complete",
                    "request_identity": identity,
                    "job": job,
                    "result": result,
                }
                record["fingerprint"] = digest_json(record)
                atomic_json(directory / "result.json", record)
                (directory / "error.json").unlink(missing_ok=True)
                completed.append({"job": job["id"], "kind": job["kind"]})
        except BlockingIOError:
            pending.append({"job": job["id"], "reason": "live_worker"})
        if len(completed) >= max_jobs:
            break
    return {
        "completed": completed,
        "pending": pending,
        "attempted": attempted,
        "plan_fingerprint": plan["fingerprint"],
    }
