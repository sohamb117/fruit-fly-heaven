import copy

import pytest
from test_compatibility_analysis import row

from connectome_body.compatibility.analysis import _figures
from connectome_body.compatibility.figures import _paired_capacity_records, paper_figures


def figure_records():
    values = []
    for seed in range(3):
        for capacity in (5000, 20000):
            for connectome, score in (("fly_c", 0.7), ("worm_c", 0.5)):
                for topology in ("real", "degree_rewired"):
                    record = row(
                        connectome=connectome,
                        seed=seed,
                        capacity=capacity,
                        topology=topology,
                        score=score,
                    )
                    record.update(
                        selected_test_score=score - (0.2 if topology == "degree_rewired" else 0),
                        purpose="smoke",
                        evidence="software_fixture_only",
                    )
                    values.append(record)
    return values


def test_paired_plot_data_retain_real_null_and_connectome_contrasts():
    values = figure_records()
    topology = _paired_capacity_records(values)
    assert len(topology) == 4
    assert all(row["estimate"]["mean"] == pytest.approx(0.2) for row in topology)
    real = [r for r in values if r["topology"] == "real"]
    cross = _paired_capacity_records(real, cross_connectome=True)
    assert len(cross) == 2 and all(r["estimate"]["training_seeds"] == 3 for r in cross)


def test_standalone_paper_figures_render_from_explicit_software_fixture(tmp_path):
    values = figure_records()
    estimate = {"mean": 0.2, "ci95": [0.1, 0.3], "training_seeds": 3}
    reports = {
        "3": {
            "comparisons": [
                {
                    "connectome": "fixture",
                    "body": "fly",
                    "task": "control",
                    "family": "mlp_mlp",
                    "plasticity": "adapters",
                    "matching_dimension": "total_trainable_parameters",
                    "adapter_parameters": 5000,
                    "control_kind": "rnn",
                    "control_topology": "real",
                    "outcomes": {"score_auc": estimate},
                }
            ]
        },
        "6": {
            "score_auc": {
                "status": "estimated",
                "residual_by_pair": [
                    {"connectome": "fixture", "body": "fly", "residual": 0.2},
                    {"connectome": "fixture", "body": "worm", "residual": -0.2},
                ],
            }
        },
        "8": {
            "paired_intervention_losses": [
                {
                    "condition": [
                        "zero",
                        "fixture",
                        "fly",
                        "control",
                        "mlp_mlp",
                        "adapters",
                        5000,
                        "real",
                    ],
                    "score_loss": estimate,
                }
            ]
        },
        "9": {
            "paired_robustness_losses": [
                {
                    "condition": [
                        "sensor_noise",
                        "fixture",
                        "fly",
                        "control",
                        "mlp_mlp",
                        "adapters",
                        5000,
                        "real",
                    ],
                    "score_loss": estimate,
                }
            ]
        },
    }
    artifacts = paper_figures(
        values, [r for r in values if r["topology"] == "real"], reports, tmp_path, smoke=True
    )
    assert {a["kind"] for a in artifacts} == {
        "topology-advantage",
        "substrate-gap",
        "matched-controller-advantage",
        "causal-lesions",
        "robustness",
        "compatibility_residual_matrix",
    }
    for artifact in artifacts:
        assert (tmp_path / artifact["png"]).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / artifact["pdf"]).read_bytes().startswith(b"%PDF")
    # Different recurrent sizes may not be silently pooled at one adapter size.
    recurrent = copy.deepcopy(values[0])
    recurrent["kind"] = "rnn"
    assert _figures([recurrent], tmp_path) == []
