import json

import pytest

from connectome_body.compatibility.run_config import RunSpec
from connectome_body.compatibility.study import compile_study, load_study
from connectome_body.graphs import make_fixture
from connectome_body.util import atomic_json, digest_json


def small_study(tmp_path, compatibility_manifest_factory):
    graph = make_fixture(tmp_path / "graph", 32, 3)
    original = json.loads(
        (__import__("pathlib").Path(__file__).parents[1] / "configs/paper/study.json").read_text()
    )
    original.update(
        project_root=str(tmp_path),
        purpose="smoke",
        device="cpu",
        seeds=[0],
        capacities=[600, 1000],
        graphs={
            "fixture": {"path": "graph", "fingerprint": graph.fingerprint, "native_body": "worm"}
        },
    )
    manifest = str(compatibility_manifest_factory())
    original["bodies"] = {
        f"worm:{task}": {
            "control_dt": 0.01,
            "spec": {"name": "worm", "task": task, "horizon": 8, "model_manifest": manifest},
        }
        for task in ("locomotion", "steering")
    }
    original["adapter"].update(budget=1000, channels=4, support=4)
    original["training"].update(
        interactions=16,
        num_envs=2,
        rollout_steps=4,
        sequence_length=2,
        burn_in=2,
        epochs=1,
        eval_every=8,
        eval_episodes=1,
        test_episodes=1,
        threshold_confirmations=2,
        critic_width=8,
    )
    original["experiments"]["2"] = {
        "ceiling": 1000,
        "sweeps": [
            {"factor": "width", "family": "mlp_mlp", "values": [2, 8]},
            {"factor": "rank", "family": "low_rank", "values": [1, 4]},
        ],
    }
    for experiment in ("5", "7", "8", "9"):
        original["experiments"][experiment]["capacities"] = [1000]
    original["experiments"]["9"]["fine_tune_interactions"] = 8
    path = tmp_path / "study.json"
    atomic_json(path, original)
    return path


def test_all_ten_experiments_compile_without_hiding_missing_prerequisites(
    tmp_path, compatibility_manifest_factory
):
    path = small_study(tmp_path, compatibility_manifest_factory)
    plan = compile_study(path, tmp_path / "plan")
    assert set(plan["coverage"]) == {str(i) for i in range(1, 11)}
    assert all(
        row["unique_training_conditions"] or row["post_training_jobs"]
        for row in plan["coverage"].values()
    )
    ids = [row["id"] for row in plan["conditions"]]
    assert len(set(ids)) == len(ids)
    statuses = {row["status"] for row in plan["conditions"]}
    assert {"ready", "blocked_prerequisite", "infeasible_capacity"} <= statuses
    assert any(
        any(issue["input"] == "compute_match" for issue in row["issues"])
        for row in plan["conditions"]
    )
    assert any(
        any(issue["input"] == "anatomical_mapping" for issue in row["issues"])
        for row in plan["conditions"]
    )
    linear = [
        row
        for row in plan["conditions"]
        if row["config"]["controller"]["adapter"]["family"] == "linear_linear"
        and row["config"]["controller"]["substrate"]["kind"] == "connectome"
        and row["config"]["controller"]["substrate"]["topology"] == "real"
        and row["config"]["controller"]["substrate"]["plasticity"] == "adapters"
    ]
    assert all(row["requested_adapter_budgets"] == [600, 1000] for row in linear)
    for row in plan["conditions"]:
        if row["status"] == "ready":
            RunSpec.load(tmp_path / "plan" / "conditions" / f"{row['id']}.json")
            assert row["parameters"]["total_trainable_parameters"] > 0
    kinds = {job["kind"] for job in plan["jobs"]}
    assert {
        "intervention",
        "calibrate_mean",
        "robustness",
        "task_transfer",
        "finetune",
        "predict_pairs",
    } <= kinds
    assert compile_study(path, tmp_path / "plan")["fingerprint"] == plan["fingerprint"]


def test_primary_study_cannot_silently_shrink_to_a_single_body(
    tmp_path, compatibility_manifest_factory
):
    path = small_study(tmp_path, compatibility_manifest_factory)
    data = json.loads(path.read_text())
    data["purpose"] = "experiment"
    atomic_json(path, data)
    with pytest.raises(ValueError, match="full three-body"):
        load_study(path)


def test_present_graph_with_failed_scientific_qualification_remains_blocked(
    tmp_path, compatibility_manifest_factory
):
    path = small_study(tmp_path, compatibility_manifest_factory)
    data = json.loads(path.read_text())
    review = {
        "schema": "connectome-use-qualification-v1",
        "graph_fingerprint": data["graphs"]["fixture"]["fingerprint"],
        "eligible_for_primary": False,
        "scope": "primary connectome-body comparison",
        "reason": "Severe fragmentation confounds a primary compatibility inference",
    }
    review["fingerprint"] = digest_json(review)
    atomic_json(tmp_path / "review.json", review)
    data["graphs"]["fixture"]["qualification"] = "review.json"
    atomic_json(path, data)
    plan = compile_study(path, tmp_path / "plan")
    graph = plan["inputs"]["graphs"]["fixture"]
    assert graph["summary"]["neurons"] == 32
    assert not graph["qualification"]["record"]["eligible_for_primary"]
    rows = [
        row for row in plan["conditions"] if row["config"]["metadata"]["connectome"] == "fixture"
    ]
    assert rows and all(row["status"] == "blocked_prerequisite" for row in rows)
    assert all(
        any(issue["input"] == "graph_qualification:fixture" for issue in row["issues"])
        for row in rows
    )
