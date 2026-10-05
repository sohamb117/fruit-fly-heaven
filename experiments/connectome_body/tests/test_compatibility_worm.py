import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from connectome_body.compatibility.bodies import BodySpec, make_body
from connectome_body.compatibility.config import (
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.predictors import body_features
from connectome_body.compatibility.robustness import PerturbationSpec, make_perturbed_body
from connectome_body.compatibility.run_config import LearningConfig, RunSpec
from connectome_body.compatibility.training import train
from connectome_body.compatibility.worm_native import WormEngine
from connectome_body.graphs import make_fixture


@pytest.fixture(scope="module")
def worm_manifest():
    path = Path(__file__).resolve().parents[1] / "data/paper-bodies/worm/manifest.json"
    if not path.is_file():
        pytest.skip("Run python -m connectome_body.compatibility prepare-worm-body first")
    return str(path)


@pytest.mark.parametrize("task", ("locomotion", "steering", "posture"))
@pytest.mark.parametrize("shift", (False, True))
def test_published_worm_tasks_replay_all_mechanical_and_sensor_state(worm_manifest, task, shift):
    spec = BodySpec("worm", task, horizon=32, model_manifest=worm_manifest)
    env = (
        make_perturbed_body(
            spec,
            PerturbationSpec(
                name="shift",
                sensor_delay=2,
                sensor_noise=0.03,
                friction_scale=1.5,
                actuator_scale=0.7,
                failed_actuators=(3,),
            ),
        )
        if shift
        else make_body(spec)
    )
    try:
        first = env.reset(37, "test")
        action = np.random.default_rng(9).uniform(-0.9, 0.9, 48).astype(np.float32)
        env.step(action)
        snapshot = env.state_dict()
        expected = env.step(action)
        env.load_state_dict(snapshot)
        actual = env.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        np.testing.assert_array_equal(env.reset(37, "test"), first)
        assert env.obs_dim == 173 and env.action_dim == 48
        assert sum(row["size"] for row in env.obs_schema) == env.obs_dim
        assert np.isfinite(actual[0]).all()
        assert env.evidence == "published_worm_mechanics"
        features = body_features(env)
        assert features["features"]["actuator_state_dimension"] == 48
        assert features["features"]["fluid_density_native_units"] is None
    finally:
        env.close()


def test_native_boundary_rejects_short_arrays_nonfinite_values_and_changed_source(
    worm_manifest, tmp_path
):
    engine = WormEngine(worm_manifest)
    try:
        for activation in (np.zeros(47), np.full(48, np.nan), np.full(48, 2)):
            with pytest.raises(ValueError, match="48 bounded"):
                engine.step(activation, np.ones(48), 0.001, 1)
        with pytest.raises(ValueError, match="48 bounded"):
            engine.step(np.zeros(48), np.ones(48), float("nan"), 1)
    finally:
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.reset()
    copied = tmp_path / "copy"
    shutil.copytree(Path(worm_manifest).parent / "source", copied / "source")
    manifest = json.loads(Path(worm_manifest).read_text())
    manifest["files"]["source/WormBody.cpp"] = "0" * 64
    (copied / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="published worm mechanics"):
        WormEngine(copied / "manifest.json")


def test_traveling_muscle_waves_move_the_actual_body_and_failure_removes_propulsion(worm_manifest):
    engine = WormEngine(worm_manifest)
    displacement = {}
    try:
        for name, direction, capacity in (("forward", 1, 1), ("backward", -1, 1), ("failed", 1, 0)):
            engine.reset()
            origin = engine.rods[:, :2].mean(0)
            for t in range(500):
                wave = np.sin(2 * np.pi * (direction * 0.5 * t * 0.01 - np.linspace(0, 1, 24)))
                assert engine.step(
                    np.r_[np.maximum(wave, 0), np.maximum(-wave, 0)],
                    np.full(48, capacity),
                    0.001,
                    10,
                )
            displacement[name] = -(engine.rods[:, :2].mean(0) - origin)[0] / 0.001
        assert displacement["forward"] > 0.5
        assert displacement["backward"] < -0.5
        assert abs(displacement["failed"]) < 1e-12
    finally:
        engine.close()


def test_worm_robustness_uses_actual_target_default_and_rejects_inapplicable_physics(worm_manifest):
    spec = BodySpec("worm", "locomotion", model_manifest=worm_manifest)
    body = make_perturbed_body(spec, PerturbationSpec(name="faster", target_speed_scale=1.5))
    try:
        body.reset(1)
        assert body.target_speed == pytest.approx(0.3)
    finally:
        body.close()
    with pytest.raises(ValueError, match="no inertia"):
        make_perturbed_body(spec, PerturbationSpec(name="inertia", inertia_scale=2))


@pytest.mark.parametrize("regime", ("adapters", "joint"))
def test_native_worm_ppo_resume_preserves_exact_future_learning(worm_manifest, tmp_path, regime):
    make_fixture(tmp_path / "graph", 40, 4)
    spec = RunSpec(
        body=BodySpec("worm", "steering", horizon=8, model_manifest=worm_manifest),
        controller=ControllerConfig(
            AdapterConfig(budget=2500, channels=4, support=4),
            SubstrateConfig(
                graph=str(tmp_path / "graph"),
                plasticity=regime,
                dynamics=DynamicsConfig(control_dt=0.01, tau_seconds=0.02),
            ),
        ),
        training=LearningConfig(
            interactions=16,
            num_envs=2,
            rollout_steps=4,
            sequence_length=2,
            burn_in=2,
            epochs=1,
            eval_every=8,
            eval_episodes=1,
            test_episodes=1,
            critic_width=8,
        ),
        purpose="smoke",
    )
    train(spec, tmp_path / "whole")
    paused = train(spec, tmp_path / "resumed", stop_after_updates=1)
    assert paused["status"] == "paused_at_checkpoint"
    train(spec, tmp_path / "resumed", resume=True)
    a, b = [
        torch.load(tmp_path / name / "latest.pt", weights_only=False)
        for name in ("whole", "resumed")
    ]
    for field in ("actor", "critic"):
        for key in a[field]:
            torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
    np.testing.assert_array_equal(a["obs"], b["obs"])
    # Related tasks preserve the exact same sensory/motor semantics.
    posture, steering = make_body(replace(spec.body, task="posture")), make_body(spec.body)
    try:
        assert posture.obs_schema == steering.obs_schema
    finally:
        posture.close()
        steering.close()
