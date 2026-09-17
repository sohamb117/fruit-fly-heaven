import json
from pathlib import Path

import numpy as np
import pytest

from connectome_body.compatibility.battery import compile_battery, read_battery, run_battery
from connectome_body.compatibility.battery_analysis import (
    ASSOCIATIONS,
    capability_cells,
    held_out_prediction,
    report_battery,
    specificity,
)
from connectome_body.compatibility.topology import write_communities
from connectome_body.graphs import make_fixture
from connectome_body.util import atomic_json, digest_json


def fixture_plan(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "GOAL.md").write_text("Test-only benchmark specification")
    graph = make_fixture(root / "graph", 32, 3)
    write_communities(graph, root / "communities.json", seed=0, iterations=3)
    config = json.loads(Path("configs/paper/battery.json").read_text())
    config.update(
        project_root=".",
        device="cpu",
        graphs={
            "fixture": {
                "path": "graph",
                "fingerprint": graph.fingerprint,
                "communities": "communities.json",
            }
        },
        seeds=[0],
        capacities=[500],
        plasticity=["adapters"],
        severity={},
        drone_horizon=8,
        adapter={"family": "mlp_linear", "channels": 2, "support": 4},
        temporal_data={
            "length": 80,
            "warmup": 16,
            "delay": 2,
            "train_sequences": 2,
            "validation_sequences": 2,
            "test_sequences": 2,
            "seed": 0,
        },
        temporal_training={
            "epochs": 1,
            "batch_size": 2,
            "sequence_length": 20,
            "learning_rate": 0.001,
        },
        drone_training={
            "interactions": 8,
            "num_envs": 1,
            "rollout_steps": 8,
            "sequence_length": 4,
            "burn_in": 2,
            "epochs": 1,
            "eval_every": 8,
            "eval_episodes": 1,
            "test_episodes": 1,
        },
    )
    atomic_json(root / "config.json", config)
    return root, compile_battery(root / "config.json", root / "plan", core=True)


def test_compiler_keeps_core_controls_and_parameter_matching(tmp_path):
    root, plan = fixture_plan(tmp_path)
    assert len(plan["conditions"]) == 15 * 7
    assert set(c["variant"] for c in plan["conditions"]) == {
        "real",
        "degree_rewired",
        "community_rewired",
        "random",
        "adapter_only",
        "rnn",
        "gru",
    }
    assert plan["summary"]["statuses"] == {"ready": 105}
    for row in plan["conditions"]:
        metadata = row["config"]["metadata"]
        if row["variant"] in ("rnn", "gru"):
            assert (
                row["parameters"]["total_trainable_parameters"] <= metadata["matched_total_target"]
            )
        if row["variant"] == "adapter_only":
            assert row["parameters"]["state_dimension"] == 0
    result = report_battery(root / "plan/plan.json", root / "report")
    assert result["measured"] == 0 and result["prediction"] == "unidentifiable"
    saved = json.loads((root / "plan/plan.json").read_text())
    saved["summary"]["conditions"] = 1
    atomic_json(root / "plan/plan.json", saved)
    with pytest.raises(ValueError, match="checksum"):
        read_battery(root / "plan/plan.json")


def test_fish_qualification_cannot_be_erased_by_presence_of_files(tmp_path):
    root, plan = fixture_plan(tmp_path)
    config = json.loads((root / "config.json").read_text())
    q = {
        "schema": "connectome-use-qualification-v1",
        "graph_fingerprint": config["graphs"]["fixture"]["fingerprint"],
        "eligible_for_primary": False,
        "scope": "primary whole connectome",
        "reason": "Fragmented reconstruction",
    }
    q["fingerprint"] = digest_json(q)
    atomic_json(root / "qualification.json", q)
    config["graphs"]["fixture"]["qualification"] = "qualification.json"
    atomic_json(root / "blocked.json", config)
    plan = compile_battery(root / "blocked.json", root / "blocked-plan", core=True)
    biological = [
        r
        for r in plan["conditions"]
        if r["variant"] in ("real", "degree_rewired", "community_rewired", "random")
    ]
    assert all(r["status"] == "blocked_prerequisite" for r in biological)


def test_battery_worker_runs_selected_checkpointed_temporal_condition(tmp_path):
    root, plan = fixture_plan(tmp_path)
    row = next(
        c
        for c in plan["conditions"]
        if c["suite"] == "temporal"
        and c["task"] == "temporal_xor"
        and c["variant"] == "adapter_only"
    )
    result = run_battery(root / "plan/plan.json", max_runs=1, max_seconds=60, condition=row["id"])
    assert len(result) == 1 and result[0]["result"]["status"] == "complete"
    report = report_battery(root / "plan/plan.json", root / "report")
    assert report["measured"] == 1
    assert (
        run_battery(root / "plan/plan.json", max_runs=1, max_seconds=60, condition=row["id"]) == []
    )


def cells_fixture():
    rng = np.random.default_rng(4)
    cells = []
    for group in range(5):
        for variant in ("real", "degree_rewired"):
            memory = group / 4
            diagnostic = {k: float(rng.uniform()) for k in ASSOCIATIONS}
            diagnostic["long_memory"] = memory
            cells.append(
                {
                    "key": [str(group), variant, 5000, "adapters"],
                    "group": str(group),
                    "diagnostic": diagnostic,
                    "neurons": 100 + group,
                    "edges": 1000 + group,
                    "capacity": 5000,
                    "plasticity": "adapters",
                    "drone": {
                        "attitude": 0.5 - memory * 0.15,
                        "hover": 0.5 - memory * 0.15,
                        "sensor_delay": 0.5 + memory * 0.2,
                        "sensor_dropout": 0.5 + memory * 0.2,
                    },
                }
            )
    return cells


def test_prediction_holds_source_and_all_nulls_out_together():
    cells = cells_fixture()
    result = held_out_prediction(cells)
    assert result["status"] == "measured_exploratory"
    for model in result["models"].values():
        for fold in model["folds"]:
            assert fold["held_out"] not in fold["training_groups"]
            assert len(fold["training_groups"]) == 4
    assert len(result["source_groups"]) == 5
    assert held_out_prediction(cells[:6])["status"] == "unidentifiable"
    assert np.isfinite(result["mse_improvement"])
    association = next(x for x in specificity(cells) if x["capability"] == "long_memory")
    assert association["correlation"] == pytest.approx(1)
    assert association["p_holm"] >= association["p_two_sided"]


def test_seed_replicates_are_aggregated_not_counted_as_substrates():
    rows = []
    for seed in range(5):
        for suite, task in [("temporal", "linear_memory"), ("drone", "hover")]:
            rows.append(
                {
                    "source": "banc",
                    "group": "banc",
                    "variant": "real",
                    "budget": 5000,
                    "plasticity": "adapters",
                    "suite": suite,
                    "task": task,
                    "score": seed / 10,
                    "metrics": {"short_memory": 0.3, "long_memory": 0.2},
                    "neurons": 100,
                    "edges": 1000,
                }
            )
    cells = capability_cells(rows)
    assert len(cells) == 1 and cells[0]["drone"]["hover"] == pytest.approx(0.2)
    assert held_out_prediction(cells)["status"] == "unidentifiable"


def test_drone_worker_zero_shot_and_acute_evaluation(tmp_path):
    from connectome_body.compatibility.battery_evaluation import evaluate_drone_condition

    root, plan = fixture_plan(tmp_path)
    row = next(
        c
        for c in plan["conditions"]
        if c["suite"] == "drone" and c["task"] == "hover" and c["variant"] == "gru"
    )
    results = run_battery(root / "plan/plan.json", max_runs=1, max_seconds=60, condition=row["id"])
    assert results[0]["result"]["status"] == "complete"
    result = evaluate_drone_condition(
        root / "plan/plan.json", row["id"], root / "evaluation.json", episodes=1
    )
    assert result["training_updates"] == 0
    assert [r["case"] for r in result["records"]] == ["trained_task", "acute:reset"]
    assert (
        result["records"][0]["evaluation"]["episodes"][0]["seed"]
        == result["records"][1]["evaluation"]["episodes"][0]["seed"]
    )


def test_frontier_censoring_and_seed_paired_control_effects():
    from connectome_body.compatibility.battery_analysis import frontiers, paired_effects

    rows = []
    for seed in range(3):
        for budget in (5000, 20000):
            for variant in ("real", "random"):
                rows.append(
                    {
                        "suite": "drone",
                        "task": "hover",
                        "source": "banc",
                        "variant": variant,
                        "plasticity": "adapters",
                        "seed": seed,
                        "budget": budget,
                        "setting": {},
                        "exposures": 1000,
                        "actual_adapter": budget - 20,
                        "actual_trainable": budget - 20,
                        "threshold_score": 0.9 if budget == 20000 and variant == "real" else 0.5,
                        "score": 0.7 if variant == "real" else 0.4,
                        "threshold": None,
                    }
                )
    frontier = frontiers(rows)
    assert all(r["right_censored"] for r in frontier if r["variant"] == "random")
    assert all(
        r["minimum_observed_successful_adapter"] == 19980
        for r in frontier
        if r["variant"] == "real"
    )
    effects = paired_effects(rows)
    assert all(r["mean"] == pytest.approx(0.3) and r["training_seeds"] == 3 for r in effects)
