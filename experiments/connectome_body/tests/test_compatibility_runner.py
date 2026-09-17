import json

import pytest
from test_compatibility_study import small_study

from connectome_body.compatibility.runner import read_plan, run_conditions, run_jobs
from connectome_body.compatibility.study import compile_study
from connectome_body.util import atomic_json


def test_compiled_condition_runs_once_and_causal_and_transfer_jobs_use_saved_policy(
    tmp_path, compatibility_manifest_factory
):
    source = small_study(tmp_path, compatibility_manifest_factory)
    root = tmp_path / "plan"
    plan = compile_study(source, root)
    parent = next(
        row
        for row in plan["conditions"]
        if row["status"] == "ready"
        and row["config"]["metadata"]["connectome"] == "fixture"
        and row["config"]["body"]["task"] == "locomotion"
        and row["config"]["controller"]["adapter"]["family"] == "mlp_mlp"
        and row["config"]["controller"]["adapter"]["budget"] == 1000
        and row["config"]["controller"]["substrate"]["topology"] == "real"
        and row["config"]["controller"]["substrate"]["plasticity"] == "adapters"
    )
    report = run_conditions(root / "plan.json", max_runs=1, condition_ids=[parent["id"]])
    assert report["results"][0]["status"] == "complete"
    again = run_conditions(root / "plan.json", max_runs=1, condition_ids=[parent["id"]])
    assert not again["results"] and again["skipped"][0]["reason"] == "already_complete"
    selected = [
        job["id"]
        for job in plan["jobs"]
        if job["reference"] == parent["id"]
        and (
            (
                job["kind"] == "intervention"
                and job["parameters"]["kind"] in ("zero", "temporal_mean")
            )
            or job["kind"] in ("calibrate_mean", "task_transfer", "finetune")
            or (
                job["kind"] == "robustness"
                and job["parameters"]["perturbation"]["name"] == "sensor_noise"
            )
        )
    ]
    assert len(selected) >= 6
    evaluated = run_jobs(root / "plan.json", max_jobs=20, job_ids=selected)
    assert not evaluated["pending"], evaluated["pending"]
    assert len(evaluated["completed"]) == len(selected)
    for item in evaluated["completed"]:
        result = json.loads((root / "jobs" / item["job"] / "result.json").read_text())
        assert result["result"]["source_training_interactions"] == 16
        if item["kind"] == "finetune":
            assert result["result"]["fine_tune"]["training_interactions"] == 8
    data = json.loads((root / "plan.json").read_text())
    data["conditions"][0]["config"]["train_seed"] = 999
    atomic_json(root / "plan.json", data)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_plan(root / "plan.json")
