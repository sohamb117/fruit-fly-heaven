import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from connectome_body.compatibility.bodies import BodySpec
from connectome_body.compatibility.config import (
    PLASTICITY_REGIMES,
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.run_config import LearningConfig, RunSpec
from connectome_body.compatibility.training import Trainer, train
from connectome_body.graphs import make_fixture


def test_execution_errors_are_not_manufactured_scientific_failures(
    tmp_path, compatibility_manifest_factory, monkeypatch
):
    from connectome_body.compatibility.analysis import collect_runs
    from connectome_body.compatibility.runner import completed_training

    spec = short_spec(tmp_path, compatibility_manifest_factory)
    for name, exception in (
        ("io", OSError("disk unavailable")),
        ("numerical", FloatingPointError("nan")),
    ):

        def fail(*args, **kwargs):
            raise exception

        monkeypatch.setattr(Trainer, "run", fail)
        output = tmp_path / name
        with pytest.raises(type(exception)):
            train(spec, output)
        report = collect_runs(output, include_smoke=True)
        manifest = json.loads((output / "manifest.json").read_text())
        condition = {"config": manifest["config"]}
        if name == "io":
            assert not report["rows"] and len(report["pending"]) == 1
            assert not completed_training(output, condition, allow_failed=True)
        else:
            assert len(report["rows"]) == 1 and report["rows"][0]["training_failed"]
            assert completed_training(output, condition, allow_failed=True)
            assert not completed_training(output, condition)


def short_spec(tmp_path, compatibility_manifest_factory, kind="connectome", regime="adapters"):
    make_fixture(tmp_path / "graph", 32, 1)
    dynamics = DynamicsConfig(control_dt=0.01, tau_seconds=0.02)
    substrate = SubstrateConfig(
        kind=kind,
        graph=str(tmp_path / "graph") if kind == "connectome" else None,
        dynamics=dynamics,
        plasticity="joint" if kind in ("rnn", "gru") else regime,
        total_budget=600 if kind in ("rnn", "gru") else None,
    )
    return RunSpec(
        body=BodySpec(
            "worm", "steering", horizon=8, model_manifest=str(compatibility_manifest_factory())
        ),
        controller=ControllerConfig(AdapterConfig(budget=600, channels=4, support=4), substrate),
        training=LearningConfig(
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
            critic_width=8,
        ),
        purpose="smoke",
    )


@pytest.mark.parametrize("regime", PLASTICITY_REGIMES)
def test_every_plasticity_regime_trains_and_resumes_exactly(
    tmp_path, compatibility_manifest_factory, regime
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, regime=regime)
    complete, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    result = train(spec, complete)
    paused = train(spec, interrupted, stop_after_updates=2)
    assert paused["status"] == "paused_at_checkpoint" and paused["training_interactions"] == 16
    train(spec, interrupted, resume=True)
    a, b = [torch.load(path / "latest.pt", weights_only=False) for path in (complete, interrupted)]
    assert a["interactions"] == b["interactions"] == 32
    assert a["optimizer_steps"] == b["optimizer_steps"] == 16
    assert a["optimized_decisions"] == b["optimized_decisions"] == 64
    for group in ("actor", "critic"):
        for name in a[group]:
            torch.testing.assert_close(a[group][name], b[group][name], rtol=0, atol=0)
    torch.testing.assert_close(a["state"], b["state"], rtol=0, atol=0)
    np.testing.assert_array_equal(a["obs"], b["obs"])
    assert result["evidence"] == "software_fixture_only"
    assert result["evaluation_interactions"] > 0
    assert result["total_environment_interactions"] == 32 + result["evaluation_interactions"]
    assert [row["training_interactions"] for row in a["curve"]] == [0, 16, 32]
    manifest = json.loads((complete / "manifest.json").read_text())
    assert (
        manifest["parameters"]["total_trainable_parameters"] + manifest["critic_parameters"]
        == manifest["total_optimization_parameters"]
    )
    assert manifest["exploration"]["kind"] == "fixed_pre_tanh_gaussian"


@pytest.mark.parametrize("kind", ("adapter_only", "rnn", "gru"))
def test_mandatory_artificial_controls_use_the_same_learning_loop(
    tmp_path, compatibility_manifest_factory, kind
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, kind)
    result = train(spec, tmp_path / "run")
    assert result["status"] == "complete" and result["training_interactions"] == 32
    assert result["parameters"]["total_trainable_parameters"] <= 600
    assert (result["parameters"]["state_dimension"] == 0) == (kind == "adapter_only")


def test_recovery_repairs_derived_files_and_rejects_drift(tmp_path, compatibility_manifest_factory):
    spec = short_spec(tmp_path, compatibility_manifest_factory)
    output = tmp_path / "run"
    train(spec, output)
    (output / "result.json").unlink()
    (output / "learning_curve.json").write_text("[]")
    (output / "best.pt").unlink()
    train(spec, output, resume=True)
    assert len(json.loads((output / "learning_curve.json").read_text())) == 3
    assert (output / "best.pt").exists()
    with pytest.raises(ValueError, match="mismatch"):
        Trainer(
            replace(spec, training=replace(spec.training, learning_rate=0.01)), output, resume=True
        )
    with pytest.raises(FileExistsError):
        Trainer(spec, output)


def test_transfer_counts_new_experience_and_resets_optimizer(
    tmp_path, compatibility_manifest_factory
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, "gru")
    parent = tmp_path / "parent"
    train(spec, parent)
    transfer = replace(
        spec, body=replace(spec.body, task="locomotion"), initial_checkpoint=str(parent / "best.pt")
    )
    child = Trainer(transfer, tmp_path / "child")
    try:
        assert child.interactions == child.optimizer_steps == 0
        assert len(child.optimizer.state) == 0
        source = torch.load(parent / "best.pt", weights_only=False)
        for name, tensor in child.actor.state_dict().items():
            torch.testing.assert_close(tensor, source["actor"][name], rtol=0, atol=0)
        assert (
            child.manifest["transfer"]["source_training_interactions"]
            == source["training_interactions"]
        )
    finally:
        child.close()


def test_exact_budget_and_body_time_grid_are_enforced(tmp_path, compatibility_manifest_factory):
    spec = short_spec(tmp_path, compatibility_manifest_factory)
    with pytest.raises(ValueError, match="divide"):
        replace(spec.training, interactions=31).validate()
    bad = replace(
        spec,
        controller=replace(
            spec.controller,
            substrate=replace(spec.controller.substrate, dynamics=DynamicsConfig(control_dt=0.2)),
        ),
    )
    with pytest.raises(ValueError, match="control_dt"):
        Trainer(bad, tmp_path / "bad")


@pytest.mark.flybody
def test_native_hover_runs_through_the_paper_training_pipeline(
    tmp_path, compatibility_manifest_factory
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, "adapter_only")
    spec = replace(
        spec,
        body=BodySpec(task="hover", horizon=8),
        controller=replace(
            spec.controller,
            adapter=AdapterConfig(budget=5000, channels=16),
            substrate=replace(
                spec.controller.substrate, dynamics=DynamicsConfig(control_dt=0.0002)
            ),
        ),
        training=replace(spec.training, interactions=8, num_envs=1, eval_every=8),
    )
    result = train(spec, tmp_path / "fly")
    assert result["status"] == "complete" and result["training_interactions"] == 8
    assert result["evidence"] == "native_flybody"
