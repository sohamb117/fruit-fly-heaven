"""Regression journeys for train-only scaling, time horizons and PPO drift guards."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_compatibility_imitation import Expert
from test_compatibility_training import short_spec

from connectome_body.compatibility import imitation as il
from connectome_body.compatibility.run_config import LearningConfig
from connectome_body.compatibility.training import Trainer, train


def test_statistics_ignore_padding_and_every_held_out_value():
    x = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    a = np.linspace(-1, 1, 16, dtype=np.float32).reshape(2, 4, 2)
    mask = np.array([[True, True, False, False], [True, True, True, False]])
    arrays = dict(
        train_obs=x,
        train_actions=a,
        train_mask=mask,
        validation_obs=x.copy(),
        validation_actions=a.copy(),
    )
    expected = il.training_statistics(arrays)
    arrays["train_obs"][~mask] = 1e9
    arrays["train_actions"][~mask] = -1e9
    arrays["validation_obs"][:] = -1e8
    arrays["validation_actions"][:] = 1e8
    actual = il.training_statistics(arrays)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_allclose(actual["observation_mean"], x[mask].mean(0))


@pytest.mark.parametrize("kind", ["adapter_only", "gru", "connectome"])
def test_normalized_bc_resume_transfer_and_guarded_ppo(
    tmp_path, compatibility_manifest_factory, monkeypatch, kind
):
    monkeypatch.setattr(il, "ObservationExpert", Expert)
    spec = short_spec(tmp_path, compatibility_manifest_factory, kind)
    spec = replace(
        spec,
        controller=replace(spec.controller, normalize_observations=True, normalize_actions=True),
        training=replace(spec.training, target_kl=0.01),
    )
    dataset = tmp_path / "data"
    il.build_demonstrations(
        spec.body, dataset, train_episodes=2, validation_episodes=1, min_success=0, purpose="smoke"
    )
    bc = il.BCConfig(
        epochs=2, sequence_length=2, standardize_actions=True, initialize_action_mean=True
    )
    full, resumed = tmp_path / "bc-full", tmp_path / "bc-resumed"
    il.train_bc(spec, bc, dataset, full)
    il.train_bc(spec, bc, dataset, resumed, stop_after_updates=1)
    il.train_bc(spec, bc, dataset, resumed, resume=True)
    a, b = [torch.load(p / "latest.pt", weights_only=False) for p in (full, resumed)]
    for key in a["actor"]:
        torch.testing.assert_close(a["actor"][key], b["actor"][key], rtol=0, atol=0)
    stats = il.training_statistics(il.load_demonstrations(dataset)[1])
    np.testing.assert_array_equal(a["actor"]["observation_mean"].numpy(), stats["observation_mean"])
    ppo = replace(spec, initial_checkpoint=str(full / "best.pt"))
    result = train(ppo, tmp_path / "ppo", stop_after_updates=1)
    assert result["status"] == "paused_at_checkpoint"
    result = train(ppo, tmp_path / "ppo", resume=True)
    assert result["training_interactions"] == spec.training.interactions
    saved = torch.load(tmp_path / "ppo/latest.pt", weights_only=False)
    for key in ("observation_mean", "observation_scale", "action_center", "action_scale"):
        torch.testing.assert_close(saved["actor"][key], a["actor"][key], rtol=0, atol=0)
    assert all(math.isfinite(x["max_conditional_kl"]) for x in saved["update_log"])


def test_kl_guard_stops_before_updating_a_drifted_policy(tmp_path, compatibility_manifest_factory):
    spec = short_spec(tmp_path, compatibility_manifest_factory, "gru")
    spec = replace(spec, training=replace(spec.training, target_kl=1e-8))
    with_trainer = Trainer(spec, tmp_path / "ppo")
    try:
        batch, snapshots = with_trainer.collect()
        original = copy.deepcopy(with_trainer.actor.state_dict())
        # Exact Gaussian KL catches drift even if sampled log-ratios look benign.
        batch["mean"] += 1
        result = with_trainer.optimize(batch, snapshots)
        assert result["kl_early_stopped"] and result["minibatch_optimizer_steps"] == 0
        assert result["max_conditional_kl"] > 1
        assert with_trainer.optimizer_steps == 0
        for name, value in with_trainer.actor.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        batch["mean"] -= 1
        result = with_trainer.optimize(batch, snapshots)
        assert result["minibatch_optimizer_steps"] == 1
        assert result["kl_early_stopped"]
    finally:
        with_trainer.close()


def test_physical_time_horizons_have_declared_discount_and_credit_times():
    settings = dict(
        discount_seconds=0.2,
        advantage_seconds=0.05,
        sequence_seconds=0.0256,
        burn_in_seconds=0.0256,
        rollout_seconds=0.1024,
    )
    fly = LearningConfig().physical_horizons(0.0002, **settings)
    assert (fly.sequence_length, fly.burn_in, fly.rollout_steps) == (128, 128, 512)
    for dt in (0.0002, 0.01):
        cfg = LearningConfig().physical_horizons(dt, **settings)
        assert -dt / math.log(cfg.gamma) == pytest.approx(0.2)
        assert -dt / math.log(cfg.gamma * cfg.gae_lambda) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        LearningConfig().physical_horizons(0.0002, **{**settings, "advantage_seconds": 1})
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(LearningConfig(), target_kl=invalid).validate()
