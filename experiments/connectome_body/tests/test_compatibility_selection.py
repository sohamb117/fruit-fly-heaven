import json

import pytest
from test_compatibility_study import small_study

from connectome_body.compatibility.runner import run_conditions, run_jobs
from connectome_body.compatibility.selection import select_conditions, selection_ids
from connectome_body.compatibility.study import compile_study
from connectome_body.util import atomic_json


def test_selection_is_plan_pinned_budgeted_and_bounds_worker_scope(
    tmp_path, compatibility_manifest_factory
):
    study = small_study(tmp_path, compatibility_manifest_factory)
    root = tmp_path / "plan"
    plan = compile_study(study, root)
    filters = tmp_path / "filters.json"
    atomic_json(
        filters,
        {
            "schema": "compatibility-selection-filter-v1",
            "filters": {
                "kind": ["adapter_only"],
                "family": ["mlp_mlp"],
                "task": ["locomotion"],
                "budget": [1000],
                "training_interactions": [16],
            },
        },
    )
    target = root / "selection.json"
    selected = select_conditions(root / "plan.json", filters, target)
    assert selected["statuses"] == {"ready": 1}
    assert selected["ready_training_interactions"] == 16
    assert selected["ready_evaluation_interactions_upper_bound"] == 48
    result = run_conditions(root / "plan.json", max_runs=1, selection_path=target)
    assert result["results"][0]["condition"] in selected["condition_ids"]
    assert len(result["results"]) == 1
    selected["condition_ids"] = []
    atomic_json(target, selected)
    with pytest.raises(ValueError, match="changed"):
        selection_ids(plan, target)


def test_predictor_inputs_cannot_be_changed_after_plan_compilation(
    tmp_path, compatibility_manifest_factory
):
    study = small_study(tmp_path, compatibility_manifest_factory)
    value = json.loads(study.read_text())
    annotation = tmp_path / "anatomy.json"
    atomic_json(
        annotation,
        {
            "graph_fingerprint": value["graphs"]["fixture"]["fingerprint"],
            "sensory_ids": [],
            "motor_ids": [],
            "provenance": {"source": "test fixture"},
        },
    )
    value["graphs"]["fixture"]["anatomy"] = str(annotation)
    atomic_json(study, value)
    root = tmp_path / "plan"
    plan = compile_study(study, root)
    jobs = [row for row in plan["jobs"] if row["kind"] == "predict_pairs"]
    assert all(str(annotation) in row["input_files"] for row in jobs)
    annotation.write_text("changed after seeing performance\n")
    result = run_jobs(root / "plan.json", max_jobs=1, job_ids=[jobs[0]["id"]])
    assert result["pending"][0]["reason"] == "missing_or_changed_preregistered_input"


@pytest.mark.parametrize("family", ("mlp_mlp", "mlp_linear", "linear_linear"))
def test_biological_capacity_selection_keeps_its_total_budget_matched_rnn_controls(
    tmp_path, compatibility_manifest_factory, family
):
    study = small_study(tmp_path, compatibility_manifest_factory)
    root = tmp_path / "plan"
    plan = compile_study(study, root)
    filters = tmp_path / "filters.json"
    atomic_json(
        filters,
        {
            "schema": "compatibility-selection-filter-v1",
            "include_matched_controls": True,
            "filters": {
                "kind": ["connectome"],
                "topology": ["real"],
                "family": [family],
                "task": ["locomotion"],
                "budget": [1000],
                "plasticity": ["adapters"],
                "width": [None],
                "sign_mode": ["random_dale"],
                "subgraph_role": [None],
                "training_interactions": [16],
            },
        },
    )
    selected = select_conditions(root / "plan.json", filters, root / "selection.json")
    assert len(selected["direct_condition_ids"]) == 1
    if family == "linear_linear":
        selected_reference = next(
            row for row in selected["rows"] if row["selection_reason"] == "filter"
        )
        assert selected_reference["budget"] == 600
        assert selected_reference["requested_adapter_budgets"] == [600, 1000]
    direct = set(selected["direct_condition_ids"])
    controls = [row for row in selected["rows"] if row["selection_reason"] == "matched_control"]
    assert {"rnn", "gru", "adapter_only"} <= {row["kind"] for row in controls}
    assert any(row["kind"] == "rnn" and row["budget"] < 1000 for row in controls)
    assert all(set(row["matched_reference_ids"]) <= direct for row in controls)
    if family == "mlp_mlp":
        assert any(row["status"] == "blocked_prerequisite" for row in controls)
    for row in plan["conditions"]:
        linked = any(match["reference"] in direct for match in row.get("matches", []))
        assert (row["id"] in selected["condition_ids"]) == (row["id"] in direct or linked)
