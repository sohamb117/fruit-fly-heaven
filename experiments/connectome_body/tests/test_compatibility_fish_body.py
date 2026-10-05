import json
from pathlib import Path

import numpy as np
import pytest
import torch

from connectome_body.compatibility.bodies import BodySpec, ManifestBody, make_body
from connectome_body.compatibility.config import (
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.robustness import PerturbationSpec, make_perturbed_body
from connectome_body.compatibility.run_config import LearningConfig, RunSpec
from connectome_body.compatibility.training import train
from connectome_body.graphs import make_fixture


@pytest.fixture(scope="module")
def fish_manifest():
    path = Path(__file__).resolve().parents[1] / "data/paper-bodies/fish/manifest.json"
    if not path.is_file():
        pytest.skip("Prepare the pinned fish body before running native integration tests")
    return str(path)


@pytest.mark.parametrize("task", ("swimming", "heading", "depth"))
@pytest.mark.parametrize("shift", (False, True))
def test_fish_replay_preserves_physics_velocity_measurement_and_sensor_history(
    fish_manifest, task, shift
):
    spec = BodySpec("fish", task, horizon=32, model_manifest=fish_manifest)
    body = (
        make_perturbed_body(
            spec,
            PerturbationSpec(
                name="shift",
                sensor_noise=0.02,
                sensor_delay=2,
                mass_scale=1.1,
                inertia_scale=1.1,
                viscosity_scale=1.2,
                actuator_scale=0.7,
                failed_actuators=(3,),
            ),
        )
        if shift
        else make_body(spec)
    )
    try:
        first = body.reset(31, "test")
        action = np.random.default_rng(9).uniform(-0.3, 0.3, body.action_dim)
        body.step(action)
        state = body.state_dict()
        expected = body.step(action)
        body.load_state_dict(state)
        actual = body.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        np.testing.assert_array_equal(body.reset(31, "test"), first)
        assert body.obs_dim == 34 and body.action_dim == 10
        assert sum(field["size"] for field in body.obs_schema) == body.obs_dim
        if task == "depth":
            assert abs(body.target_depth - body.origin[2] / body.length) >= 0.5
        elif task == "heading":
            assert np.arccos(np.clip(body.target_direction[0], -1, 1)) >= np.pi / 6
    finally:
        body.close()


def test_inertial_frame_is_not_mistaken_for_anatomical_body_frame(fish_manifest):
    body = make_body(BodySpec("fish", "swimming", model_manifest=fish_manifest))
    try:
        body.reset(0)
        # The published head's principal inertial axes are not anatomical axes.
        body.data.qvel[:] = 0
        body.data.qvel[:3] = [0.1, 0.2, 0.3]
        body.mujoco.mj_forward(body.model, body.data)
        rotation, velocity = ManifestBody._kinematics(body)
        np.testing.assert_allclose(rotation @ velocity[3:], [0.1, 0.2, 0.3], atol=1e-12)
        before = body.position().copy()
        body.step(np.zeros(body.action_dim))
        rotation, velocity = body._kinematics()
        np.testing.assert_allclose(
            rotation @ velocity[3:], (body.position() - before) / body.control_dt
        )
    finally:
        body.close()


def test_source_inertias_and_the_small_explicit_repair_are_auditable(fish_manifest):
    data = json.loads((Path(fish_manifest).parent / "morphology.json").read_text())
    corrected = [r for r in data["records"] if r["inertia_feasibility_repair"]]
    assert len(corrected) == 1 and corrected[0]["index"] == 1
    repair = corrected[0]["inertia_feasibility_repair"]
    assert 0 < repair["relative_frobenius_change"] < 0.001
    for row in data["records"]:
        xx, yy, zz, xy, xz, yz = row["inertia"]
        moments = np.linalg.eigvalsh([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        assert moments.min() > 0 and moments[2] <= moments[:2].sum() * (1 + 1e-12)


@pytest.mark.parametrize("regime", ("adapters", "joint"))
def test_native_fish_training_resume_preserves_future_learning(fish_manifest, tmp_path, regime):
    make_fixture(tmp_path / "graph", 40, 4)
    spec = RunSpec(
        body=BodySpec("fish", "heading", horizon=8, model_manifest=fish_manifest),
        controller=ControllerConfig(
            AdapterConfig(budget=1500, channels=4, support=4),
            SubstrateConfig(
                graph=str(tmp_path / "graph"),
                plasticity=regime,
                dynamics=DynamicsConfig(control_dt=0.005, tau_seconds=0.02),
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
    train(spec, tmp_path / "resume", stop_after_updates=1)
    train(spec, tmp_path / "resume", resume=True)
    a, b = [
        torch.load(tmp_path / name / "latest.pt", weights_only=False)
        for name in ("whole", "resume")
    ]
    for group in ("actor", "critic"):
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], rtol=0, atol=0)
    np.testing.assert_array_equal(a["obs"], b["obs"])
