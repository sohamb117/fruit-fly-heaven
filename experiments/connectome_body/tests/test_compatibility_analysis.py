import json
from dataclasses import replace

import numpy as np
import pytest
from test_compatibility_training import short_spec

from connectome_body.compatibility.analysis import (
    _mean_interval,
    analyze,
    capacity_frontiers,
    compatibility_regression,
    curve_metrics,
    paired_contrast,
    pareto_interfaces,
)
from connectome_body.compatibility.training import train


def row(connectome="fly_c", body="fly", seed=0, capacity=5000, topology="real", score=0.6):
    native = {"fly_c": "fly", "worm_c": "worm", "fish_c": "fish"}[connectome]
    curve = [
        {
            "training_interactions": n,
            "mean_score": s,
            "success_rate": s,
            "optimizer_steps": n // 10,
            "training_wall_seconds": n / 100,
        }
        for n, s in [(0, 0.1), (100, score), (200, score)]
    ]
    return {
        "connectome": connectome,
        "body": body,
        "native_body": native,
        "task": "control",
        "train_seed": seed,
        "substrate_seed": 0,
        "kind": "connectome",
        "topology": topology,
        "family": "mlp_mlp",
        "initialization": "biological",
        "sign_mode": "random_dale",
        "plasticity": "adapters",
        "channels": 4,
        "experience_budget": 200,
        "adapter_parameters": capacity,
        "actor_parameters": capacity,
        "declared_budget": capacity,
        "width": 20,
        "rank": None,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "training_failed": False,
        "final_score": score,
        "success": score,
        "selected_test_success": score,
        "score_auc": score,
        **curve_metrics(curve, 200, 0.8, 2),
    }


def test_curve_metrics_preserve_censoring_and_separate_gradient_and_environment_steps():
    successful = row(score=0.9)
    assert successful["experience"]["first_crossing_step"] == 100
    assert successful["experience"]["certification_step"] == 200
    assert successful["optimizer_steps_to_threshold"] == 20
    assert successful["compute_seconds_to_threshold"] == 2
    failure = row(score=0.1)
    assert not failure["experience"]["event"] and failure["restricted_experience"] == 200
    assert failure["optimizer_steps_to_threshold"] is None
    with pytest.raises(ValueError, match="exact budget"):
        curve_metrics(
            [{"training_interactions": 0, "mean_score": 0.0, "success_rate": 0.0}], 100, 0.8, 2
        )


def test_seed_bootstrap_does_not_count_repeated_nulls_as_new_seeds():
    summary = _mean_interval([0.0, 0.0, 1.0], [0, 0, 1])
    assert summary["mean"] == 0.5 and summary["training_seeds"] == 2
    assert _mean_interval([0.0, 1.0], [0, 0])["ci95"] is None


def test_paired_comparison_and_unmatched_records_are_explicit():
    rows = [
        row(seed=s, topology=t, score=0.8 if t == "real" else 0.2)
        for s in range(3)
        for t in ("real", "degree_rewired")
    ]
    rows.append(row(seed=4))
    result = paired_contrast(
        rows,
        {"topology": "real"},
        {"topology": "degree_rewired"},
        keys=("connectome", "body", "train_seed"),
    )
    assert result["training_seeds"] == 3 and result["unmatched_left"] == 1
    assert result["mean"] > 0 and len(result["pairs"]) == 3
    assert all(pair["actor_parameter_difference"] == 0 for pair in result["pairs"])


def test_capacity_frontier_uses_actual_parameters_without_assuming_monotonicity():
    rows = [row(capacity=p, score=s) for p, s in [(1000, 0.2), (2000, 0.9), (5000, 0.1)]]
    frontier = capacity_frontiers(rows)[0]
    assert frontier["minimum_adapter_parameters"] == 2000
    assert frontier["tested_capacities"] == [1000, 2000, 5000]
    assert not frontier["monotonicity_assumed"]
    failed = capacity_frontiers([row(capacity=1000, score=0.2)])[0]
    assert failed["capacity_censored"] and failed["minimum_adapter_parameters"] is None
    frontier = pareto_interfaces(rows)[0]["interfaces"]
    assert not next(item for item in frontier if item["adapter_parameters"] == 5000)["pareto"]


def test_crossed_regression_recovers_match_effect_beyond_body_and_connectome_main_effects():
    rows = []
    for seed in range(3):
        for ci, c in enumerate(("fly_c", "worm_c", "fish_c")):
            for bi, b in enumerate(("fly", "worm", "fish")):
                example = row(c, b, seed)
                example["final_score"] = (
                    0.2
                    + 0.04 * ci
                    + 0.03 * bi
                    + 0.17 * (example["native_body"] == b)
                    + 0.001 * seed
                )
                rows.append(example)
    result = compatibility_regression(rows, outcome="final_score", replicates=30)
    assert result["status"] == "estimated"
    np.testing.assert_allclose(result["matched_pair_delta"], 0.17, atol=1e-10)
    assert result["training_seed_clusters"] == 3
    assert (
        compatibility_regression(rows[:-1], outcome="final_score", replicates=10)["status"]
        == "estimated"
    )
    incomplete = [r for r in rows if (r["connectome"], r["body"]) != ("fish_c", "worm")]
    assert compatibility_regression(incomplete)["status"] == "unidentified"
    assert (
        compatibility_regression([r for r in rows if r["body"] == "fly"])["status"]
        == "unidentified"
    )


def test_saved_runs_analyze_end_to_end_exclude_smoke_and_detect_changed_manifests(
    tmp_path, compatibility_manifest_factory
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, "adapter_only")
    spec = replace(spec, metadata={"connectome": "adapter_only"})
    directory = tmp_path / "run"
    train(spec, directory)
    with pytest.raises(ValueError, match="No terminal"):
        analyze(directory, tmp_path / "excluded", figures=False)
    report = analyze(directory, tmp_path / "report", include_smoke=True)
    assert report["terminal_runs"] == 1 and report["figures"]
    assert list((tmp_path / "report").glob("*.pdf"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["train_seed"] = 9
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        analyze(directory, tmp_path / "bad", include_smoke=True, figures=False)
