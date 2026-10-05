
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
from connectome_body.compatibility.drone import DRONE_TASKS, DroneBody, qualify_drone
from connectome_body.compatibility.run_config import LearningConfig, RunSpec
from connectome_body.compatibility.training import train


@pytest.mark.parametrize("task", DRONE_TASKS)
def test_each_drone_task_rolls_out_and_restores_exactly(task):
    body = make_body(BodySpec(name="drone", task=task, horizon=40))
    obs = body.reset(31, "test")
    assert obs.shape == (body.obs_dim,) and np.isfinite(obs).all()
    for _ in range(4):
        body.step(np.zeros(6))
    saved = body.state_dict()
    first = body.step(np.linspace(-0.5, 0.1, 6))
    body.load_state_dict(saved)
    second = body.step(np.linspace(-0.5, 0.1, 6))
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert body.control_dt == 0.02
    for _ in range(35):
        _, _, terminated, truncated, info = body.step(np.full(6, -0.346))
        if terminated or truncated:
            assert "position_rmse" in info and "angular_rmse" in info
            assert np.isfinite(info["score"])
            break
    else:
        raise AssertionError("Episode must end")


def test_drone_trim_single_rotor_feasibility_and_freefall():
    report = qualify_drone()
    assert report["passed"]
    body = DroneBody(BodySpec(name="drone", task="hover", horizon=100))
    body.rotation = np.eye(3)
    body.position[:] = [0, 0, 5]
    body.velocity[:] = 0
    body.omega[:] = 0
    body.thrust[:] = 0
    body._physics(np.zeros(6))
    assert body.velocity[2] == pytest.approx(-9.81 * 0.02, abs=0.001)
    np.testing.assert_allclose(body.rotation.T @ body.rotation, np.eye(3), atol=1e-12)


def test_sensor_delay_and_dropout_have_no_clean_state_bypass():
    body = DroneBody(
        BodySpec(name="drone", task="sensor_delay", horizon=100, parameters={"delay": 3})
    )
    observed = body.observation().copy()
    body.position += 20
    # Reading twice is pure; even range/relative goals cannot expose current state.
    np.testing.assert_array_equal(observed, body.observation())
    body._capture_sensor()
    np.testing.assert_array_equal(observed, body.observation())
    drop = DroneBody(BodySpec(name="drone", task="sensor_dropout", horizon=100))
    drop.steps = drop.event_step
    drop._capture_sensor()
    obs = drop.observation()
    assert not obs[45]  # sensor_valid, schema width 52
    assert not obs[:18].any()
    drop.position += 5
    drop.rotation[:] = 4
    np.testing.assert_array_equal(obs, drop.observation())


def test_split_seeds_and_color_cues_not_direct_goal_leakage():
    body = DroneBody(BodySpec(name="drone", task="color_navigation", horizon=100))
    a = body.reset(4, "train")
    b = body.reset(4, "test")
    assert not np.array_equal(a, b)
    assert not b[18:21].any()
    assert b[40:42].sum() == 1


def test_real_drone_ppo_exact_checkpoint_resume(tmp_path):
    spec = RunSpec(
        body=BodySpec(name="drone", task="hover", horizon=8),
        controller=ControllerConfig(
            adapter=AdapterConfig(family="mlp_linear", budget=600, channels=4),
            substrate=SubstrateConfig(
                kind="gru",
                plasticity="joint",
                total_budget=1000,
                dynamics=DynamicsConfig(control_dt=0.02),
            ),
        ),
        training=LearningConfig(
            interactions=32,
            num_envs=1,
            rollout_steps=8,
            sequence_length=4,
            burn_in=2,
            epochs=1,
            eval_every=16,
            eval_episodes=1,
            test_episodes=1,
        ),
        threads=1,
    )
    complete = train(spec, tmp_path / "complete")
    paused = train(spec, tmp_path / "resume", stop_after_updates=1)
    assert paused["status"] == "paused_at_checkpoint"
    resumed = train(spec, tmp_path / "resume", resume=True)
    assert (
        complete["selected_checkpoint_evaluation"]["test"]["episodes"]
        == resumed["selected_checkpoint_evaluation"]["test"]["episodes"]
    )
    a = torch.load(tmp_path / "complete/latest.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/latest.pt", weights_only=False)
    for group in ("actor", "critic"):
        for k in a[group]:
            torch.testing.assert_close(a[group][k], b[group][k], rtol=0, atol=0)
    assert complete["evidence"] == "engineered_se3_hexrotor_v1"
