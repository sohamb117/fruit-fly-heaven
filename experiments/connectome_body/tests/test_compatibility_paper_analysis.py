import json

import pytest
from test_compatibility_analysis import row
from test_compatibility_study import small_study

from connectome_body.compatibility.paper_analysis import (
    OUTCOMES,
    analyze_study,
    interaction_test,
    topology_adjusted_compatibility,
)
from connectome_body.compatibility.runner import run_conditions
from connectome_body.compatibility.study import compile_study


@pytest.mark.parametrize("biological_match_effect", [0.0, 0.15])
def test_topology_adjusted_match_effect_removes_native_advantage_shared_with_nulls(
    biological_match_effect,
):
    records = []
    for seed in range(3):
        for ci, connectome in enumerate(("fly_c", "worm_c", "fish_c")):
            for bi, body in enumerate(("fly", "worm", "fish")):
                for topology in ("real", "degree_rewired"):
                    record = row(connectome, body, seed, topology=topology)
                    matched = record["native_body"] == body
                    score = 0.2 + 0.04 * ci + 0.03 * bi + 0.1 * matched + 0.002 * seed
                    if topology == "real":
                        score += 0.05 + biological_match_effect * matched
                    record.update({outcome: score for outcome in OUTCOMES})
                    records.append(record)
    result = topology_adjusted_compatibility(records)
    assert result["paired_rows"] == 27 and result["unmatched_real_rows"] == 0
    for regression in result["outcomes"].values():
        assert regression["status"] == "estimated"
        assert regression["matched_pair_delta"] == pytest.approx(biological_match_effect)
    unpaired = topology_adjusted_compatibility(records[:-1])
    assert unpaired["paired_rows"] == 26 and unpaired["unmatched_real_rows"] == 1


def test_three_way_interaction_has_additional_rank_and_keeps_training_seeds_as_clusters():
    rows = []
    for seed in range(5):
        for c in range(2):
            for b in range(2):
                for family in range(2):
                    record = row("fly_c" if c else "worm_c", "fly" if b else "worm", seed)
                    record.update(
                        family=f"f{family}",
                        body_task=record["body"] + ":task",
                        score_auc=0.3
                        + 0.04 * c
                        + 0.03 * b
                        + 0.05 * family
                        + 0.2 * c * b * family
                        + 0.002 * seed,
                    )
                    rows.append(record)
    report = interaction_test(rows, ("family", "connectome", "body_task"))
    assert report["status"] == "estimated"
    assert report["full_rank"] > report["reduced_rank"]
    assert report["weighted_rss_reduction"] > 0
    assert report["training_seeds"] == 5 and report["sign_flip_draws"] == 32
    missing = [
        r
        for r in rows
        if not (r["family"] == "f1" and r["connectome"] == "fly_c" and r["body"] == "fly")
    ]
    assert (
        interaction_test(missing, ("family", "connectome", "body_task"))["status"] == "unidentified"
    )


def test_ten_experiment_report_retains_unmeasured_evidence_and_no_brain_pairings(
    tmp_path, compatibility_manifest_factory
):
    study = small_study(tmp_path, compatibility_manifest_factory)
    root = tmp_path / "plan"
    plan = compile_study(study, root)
    no_brain = next(
        item
        for item in plan["conditions"]
        if item["status"] == "ready"
        and item["config"]["controller"]["substrate"]["kind"] == "adapter_only"
        and "3" in item["experiments"]
    )
    assert no_brain["matches"]
    for item in plan["conditions"]:
        keys = [(x["reference"], x["dimension"]) for x in item["matches"]]
        assert len(set(keys)) == len(keys)
    run_conditions(root / "plan.json", max_runs=1, condition_ids=[no_brain["id"]])
    report = analyze_study(root / "plan.json", tmp_path / "report", figures=False)
    assert report["terminal_training_runs"] == 1
    assert set(report["experiments"]) == {str(i) for i in range(1, 11)}
    assert report["experiments"]["6"]["score_auc"]["status"] == "unidentified"
    assert report["experiments"]["3"]["missing_pairs"]
    saved = json.loads((tmp_path / "report/paper-report.json").read_text())
    assert saved["purpose"] == "smoke" and saved["pending_jobs"]
