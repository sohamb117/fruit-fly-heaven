import copy

import numpy as np
import pytest

from connectome_body.compatibility.bodies import BodySpec, make_body
from connectome_body.compatibility.predictors import (
    _ridge_fit,
    _ridge_predict,
    body_features,
    graph_features,
    predict_held_out_pairs,
    predicted_capacity_frontier,
)
from connectome_body.graphs import save_graph
from connectome_body.util import atomic_json, digest_json


def test_graph_measurements_match_a_known_cycle_and_preserve_unknown_anatomy(tmp_path):
    graph = save_graph(
        tmp_path / "cycle",
        ["a", "b", "c", "isolated"],
        [0, 1, 2],
        [1, 2, 0],
        [1, 2, 3],
        provenance={"is_synthetic": True},
    )
    first = graph_features(graph, landmarks=4, motif_samples=200)
    assert first == graph_features(graph, landmarks=4, motif_samples=200)
    values = first["features"]
    assert values["chain_cyclic_closure"] == 1 and values["chain_feedforward_closure"] == 0
    assert values["chain_motif_1000"] == 1
    assert values["flow_hierarchy"] == 0 and values["strong_components"] == 2
    assert values["reachable_pair_fraction"] == 0.5 and values["reachable_path_mean"] == 1.5
    assert values["bilateral_edge_agreement"] is None and values["directed_modularity"] is None
    anatomy = tmp_path / "anatomy.json"
    atomic_json(
        anatomy,
        {
            "graph_fingerprint": graph.fingerprint,
            "provenance": "test",
            "sensory_ids": ["a"],
            "motor_ids": ["c"],
        },
    )
    annotated = graph_features(graph, anatomy_file=anatomy)["features"]
    assert annotated["sensory_motor_path_mean"] == 2
    assert annotated["sensory_motor_reachable_fraction"] == 1
    atomic_json(anatomy, {"graph_fingerprint": "changed", "provenance": "test"})
    with pytest.raises(ValueError, match="graph identity"):
        graph_features(graph, anatomy_file=anatomy)


def test_body_descriptors_are_derived_from_the_executed_model(compatibility_manifest_factory):
    body = make_body(
        BodySpec(
            "worm", "steering", horizon=8, model_manifest=str(compatibility_manifest_factory())
        )
    )
    try:
        result = body_features(body)
        assert result["body_fingerprint"] == body.fingerprint
        assert result["features"]["action_dimension"] == 1
        assert result["features"]["articulated_hinges"] == 1
        assert result["features"]["control_dt_seconds"] == 0.01
        assert result["evidence"] == "software_fixture_only"
    finally:
        body.close()


def _records():
    graphs, bodies, rows = {}, {}, []

    def seal(record):
        return {**record, "fingerprint": digest_json(record)}

    for ci in range(3):
        name = f"c{ci}"
        graphs[name] = seal(
            {
                "schema": "connectome-structural-features-v1",
                "graph_fingerprint": name,
                "features": {"degree_cv": ci, "missing": None},
            }
        )
    for bi in range(2):
        name = f"b{bi}"
        bodies[f"{name}:task"] = seal(
            {
                "schema": "body-structural-features-v1",
                "body_fingerprint": name,
                "features": {"damping": bi},
            }
        )
        for ci in range(3):
            for seed in range(2):
                for p in (100, 200):
                    rows.append(
                        {
                            "connectome": f"c{ci}",
                            "body": name,
                            "task": "task",
                            "graph_fingerprint": f"c{ci}",
                            "body_fingerprint": name,
                            "kind": "connectome",
                            "topology": "real",
                            "family": "mlp_mlp",
                            "plasticity": "adapters",
                            "train_seed": seed,
                            "adapter_parameters": p,
                            "actor_parameters": p,
                            "experience_budget": 1000,
                            "channels": 4,
                            "state_dimension": 100,
                            "observation_dim": 8,
                            "action_dim": 2,
                            "success": 0.2 + 0.12 * ci * bi + p / 1000,
                        }
                    )
    return rows, graphs, bodies


def test_nested_prediction_holds_out_all_replicates_and_uses_identical_model_folds():
    rows, graphs, bodies = _records()
    report = predict_held_out_pairs(rows, graphs, bodies, alphas=(0.01, 1))
    baseline, structural = report["models"].values()
    for a, b in zip(baseline["folds"], structural["folds"], strict=True):
        assert a["test_rows"] == 4
        assert a["test_pair"] not in a["train_pairs"]
        assert a["inner_validation_pairs"] == b["inner_validation_pairs"]
        assert all(a["test_pair"] not in group for group in a["inner_validation_pairs"])
    # Changing every label of the held-out pair cannot alter its fitted
    # predictions or regularization selection; none of those labels is seen.
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["connectome"] == "c0" and row["body"] == "b0":
            row["success"] = 0.99
    other = predict_held_out_pairs(changed, graphs, bodies, alphas=(0.01, 1))
    for a, b in zip(report["predictions"], other["predictions"], strict=True):
        if a["pair"] == ["c0", "b0"]:
            assert a["structural_interactions"] == b["structural_interactions"]
            assert a["main_effect_baseline"] == b["main_effect_baseline"]
    frontier = predicted_capacity_frontier(report)
    assert all(not row["extrapolation"] for row in frontier)
    assert all(row["minimum_predicted_adapter_parameters"] in (None, 100, 200) for row in frontier)
    graphs["c0"]["features"]["degree_cv"] = 99
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        predict_held_out_pairs(rows, graphs, bodies)


def test_imputation_and_scaling_use_training_data_only():
    model = _ridge_fit(np.array([[1.0, np.nan], [3.0, 4.0]]), np.array([0.0, 1.0]), ["a", "b"], 0.1)
    np.testing.assert_array_equal(model["medians"], [2.0, 4.0])
    prediction = _ridge_predict(model, np.array([[10000.0, np.nan]]))
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(model["medians"], [2.0, 4.0])
