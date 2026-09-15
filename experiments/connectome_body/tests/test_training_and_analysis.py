import dataclasses
import json

import numpy as np
import pytest
import torch

from connectome_body.analyze import (
    analyze,
    capacity_frontier,
    cluster_interval,
    cross_connectome_contrasts,
    survival,
)
from connectome_body.config import BodyConfig, DynamicsConfig, PPOConfig, RunConfig
from connectome_body.data import import_edges
from connectome_body.graphs import make_fixture
from connectome_body.suite import make_plan, run_plan
from connectome_body.train import Trainer, advantages, threshold_experience, train


def short_config(graph, substrate="real"):
    return RunConfig(
        graph=str(graph),
        substrate=substrate,
        adapter_budget=1024,
        body=BodyConfig(backend="fixture", horizon=8),
        dynamics=DynamicsConfig(channels=4),
        ppo=PPOConfig(
            interactions=32,
            num_envs=2,
            rollout_steps=4,
            sequence_length=2,
            burn_in=2,
            epochs=2,
            eval_every=16,
            eval_episodes=1,
            test_episodes=2,
            threshold_confirmations=2,
        ),
    )


def test_gae_bootstraps_timeouts_not_terminal_or_next_episode():
    reward = torch.tensor([[1.0], [100.0]])
    values = torch.zeros_like(reward)
    next_values = torch.tensor([[10.0], [0.0]])
    done = torch.tensor([[True], [True]])
    term = torch.tensor([[False], [True]])
    adv, _ = advantages(reward, values, next_values, term, done, 0.9, 0.95)
    assert adv[0].item() == 10  # 1 + 0.9*10; excludes second episode's 100.
    term[0] = True
    adv, _ = advantages(reward, values, next_values, term, done, 0.9, 0.95)
    assert adv[0].item() == 1


def test_threshold_requires_consecutive_evaluations_and_reports_censoring():
    curve = [
        {"training_interactions": i * 20, "success_rate": r}
        for i, r in enumerate([0.8, 0.9, 0.7, 0.8, 0.9, 0.9])
    ]
    result = threshold_experience(curve, 0.8, 3, 100)
    assert result["event"] and result["first_crossing_step"] == 60
    assert result["certification_step"] == 100
    result = threshold_experience(curve[:-1], 0.8, 3, 80)
    assert not result["event"] and result["certification_step"] is None
    assert result["censor_step"] == 80


def test_survival_keeps_failures_in_experience_estimate():
    result = survival([10, 100], [True, False], 100)
    assert result["restricted_mean_interactions"] == 55
    assert result["censored"] == 1
    assert survival([100, 100], [False, False], 100)["restricted_mean_interactions"] == 100
    # Two repeated port draws do not outweigh one independent training seed.
    weighted = survival([10, 10, 100], [True, True, False], 100, weights=[0.5, 0.5, 1])
    assert weighted["restricted_mean_interactions"] == 55
    assert weighted["reached_fraction"] == 0.5


def test_bootstrap_does_not_treat_port_draws_as_training_seeds():
    result = cluster_interval([0.0, 0.0, 1.0], [0, 0, 1])
    assert result["mean"] == 0.5
    assert result["training_seeds"] == 2
    assert cluster_interval([0.0, 1.0], [0, 0])["ci95"] is None


def test_resume_exact_and_analysis_end_to_end(tmp_path):
    make_fixture(tmp_path / "graph", 32)
    config = short_config(tmp_path / "graph")
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    train(config, continuous)
    stopped = train(config, resumed, stop_after_updates=2)
    assert stopped["status"] == "paused_at_checkpoint"
    train(config, resumed, resume=True)
    a = torch.load(continuous / "latest.pt", weights_only=False)
    b = torch.load(resumed / "latest.pt", weights_only=False)
    assert a["interactions"] == b["interactions"] == 32
    for key in a["actor"]:
        torch.testing.assert_close(a["actor"][key], b["actor"][key], rtol=0, atol=0)
    np.testing.assert_array_equal(a["obs"], b["obs"])
    baseline_dir = tmp_path / "baseline"
    train(dataclasses.replace(config, substrate="adapter_only", graph=None), baseline_dir)
    report = analyze(tmp_path, tmp_path / "analysis", include_fixtures=True)
    assert report["completed_runs"] == 2  # The exact resumed copy is not another seed.
    assert (tmp_path / "analysis/report.md").exists()
    assert list((tmp_path / "analysis").glob("capacity-*.png"))
    with pytest.raises(ValueError, match="No completed"):
        analyze(tmp_path, tmp_path / "invalid-report")


def test_resume_rejects_parameter_budget_drift(tmp_path):
    make_fixture(tmp_path / "graph", 32)
    config = short_config(tmp_path / "graph", "adapter_gru")
    train(config, tmp_path / "run", stop_after_updates=1)
    with pytest.raises(ValueError, match="mismatch"):
        Trainer(dataclasses.replace(config, adapter_budget=1500), tmp_path / "run", resume=True)


def test_resume_repairs_logs_if_last_checkpoint_was_committed_before_json(tmp_path):
    config = dataclasses.replace(short_config(None, "adapter_only"), graph=None)
    output = tmp_path / "run"
    train(config, output)
    original = json.loads((output / "learning_curve.json").read_text())
    (output / "result.json").unlink()  # Final evaluation had not been committed.
    (output / "learning_curve.json").write_text("[]")
    (output / "updates.json").write_text("[]")
    result = train(config, output, resume=True)
    assert result["training_interactions"] == 32
    assert json.loads((output / "learning_curve.json").read_text()) == original
    assert len(json.loads((output / "updates.json").read_text())) > 0
    assert analyze(output, tmp_path / "report", include_fixtures=True)["completed_runs"] == 1


@pytest.mark.parametrize(
    "substrate", ["degree_shuffled", "matched_random", "no_edges", "adapter_gru", "trainable_gru"]
)
def test_every_condition_has_a_working_learning_loop(tmp_path, substrate):
    make_fixture(tmp_path / "graph", 32)
    config = short_config(tmp_path / "graph", substrate)
    result = train(config, tmp_path / "run")
    assert result["status"] == "complete" and result["training_interactions"] == 32
    assert result["actor_parameters"] <= 1024
    assert result["evidence"] == "software_fixture_only"


def test_generic_import_preserves_ids_and_rejects_missing_nodes(tmp_path):
    ids = [str(2**63 + i) for i in range(4)]
    nodes, edges, provenance = (tmp_path / n for n in ("nodes.csv", "edges.csv", "provenance.json"))
    nodes.write_text("id,sign\n" + "\n".join(f"{i},1" for i in ids))
    edges.write_text(f"pre,post,weight\n{ids[0]},{ids[1]},3\n{ids[1]},{ids[2]},2\n")
    provenance.write_text(
        json.dumps(
            dict(
                dataset="example",
                species="test species",
                release="v1",
                coverage="specified subset",
                resolution="neuron_synapse",
                citation="https://example.org/paper",
                license="CC0",
            )
        )
    )
    graph = import_edges(edges, nodes, provenance, tmp_path / "g")
    assert list(graph.node_ids) == ids
    edges.write_text(f"pre,post,weight\n{ids[0]},missing,3\n")
    with pytest.raises(ValueError, match="endpoints"):
        import_edges(edges, nodes, provenance, tmp_path / "bad")


def test_missing_datasets_are_explicit_and_baselines_not_duplicated(tmp_path):
    make_fixture(tmp_path / "graphs/a", 32)
    spec = {
        "datasets": ["a", "missing"],
        "tasks": ["balance"],
        "substrates": ["real"],
        "baselines": ["adapter_only"],
        "adapter_budgets": [1024],
        "train_seeds": [0],
        "substrate_seeds": [0, 1],
        "base": {"body": {"backend": "fixture"}, "dynamics": {"channels": 4}},
    }
    plan = make_plan(spec, tmp_path / "graphs", tmp_path / "plan")
    assert plan["planned_runs"] == 5  # Four graph runs + ONE baseline.
    assert plan["status"] == "data_required"
    with pytest.raises(ValueError, match="missing datasets"):
        run_plan(tmp_path / "plan/plan.json")
    spec["dynamics_cases"] = [
        {"name": "slow", "settings": {"tau_seconds": 0.05}},
        {"name": "fast", "settings": {"tau_seconds": 0.01}},
    ]
    sweep = make_plan(spec, tmp_path / "graphs", tmp_path / "sweep", datasets=["a"])
    assert sweep["planned_runs"] == 6 and sweep["status"] == "ready"
    assert len({r["id"] for r in sweep["runs"]}) == 6
    assert {r["config"]["dynamics"]["tau_seconds"] for r in sweep["runs"]} == {0.05, 0.01}


def test_capacity_frontier_does_not_invent_success_for_censored_sizes():
    base = {
        "protocol": "example",
        "task": "balance",
        "dataset": "example",
        "substrate": "real",
        "success_threshold": 0.8,
        "test_success": {"mean": 0.9},
    }
    points = [
        {**base, "actor_parameters": 4800, "experience": {"reached_fraction": 0.2}},
        {**base, "actor_parameters": 49000, "experience": {"reached_fraction": 1.0}},
    ]
    assert capacity_frontier(points)[0]["smallest_observed_successful_actor"] == 49000
    assert capacity_frontier(points[:1])[0]["smallest_observed_successful_actor"] is None


def test_cross_connectome_experience_keeps_censored_runs():
    base = {
        "protocol": "same",
        "task": "balance",
        "budget": 5000,
        "substrate": "real",
        "test_success": 0.9,
        "ood_success": 0.6,
        "learning_auc": 0.5,
        "train_seed": 0,
    }
    rows = [
        {
            **base,
            "dataset": "A",
            "experience": {"event": True, "certification_step": 10, "censor_step": 100},
        },
        {
            **base,
            "dataset": "B",
            "experience": {"event": False, "certification_step": None, "censor_step": 100},
        },
    ]
    result = cross_connectome_contrasts(rows)[0]
    assert result["metrics"]["capped_experience"]["mean"] == -90
    assert result["metrics"]["capped_experience"]["ci95"] is None


def test_worker_continues_past_completed_runs_and_rejects_code_drift(tmp_path, monkeypatch):
    spec = {
        "datasets": [],
        "tasks": ["balance"],
        "substrates": [],
        "baselines": ["adapter_only"],
        "adapter_budgets": [1024],
        "train_seeds": [0, 1],
        "substrate_seeds": [0],
        "base": short_config(None, "adapter_only").to_dict(),
    }
    make_plan(spec, tmp_path / "graphs", tmp_path / "plan")
    plan_path = tmp_path / "plan/plan.json"
    assert run_plan(plan_path, max_runs=1)["processed_runs"] == 1
    result = run_plan(plan_path, max_runs=1)
    assert result["processed_runs"] == result["already_completed_runs"] == 1
    monkeypatch.setattr("connectome_body.suite.code_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="Code changed"):
        run_plan(plan_path)
