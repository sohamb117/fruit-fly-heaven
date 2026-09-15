import os
from pathlib import Path

import numpy as np
import pytest

from connectome_body.body import PROJECT, FlyBodyEnv
from connectome_body.config import BodyConfig


def source_path():
    path = Path(os.environ.get("FLYBODY_SOURCE", str(PROJECT / "references/flybody")))
    if not (path / "flybody/fruitfly/assets/fruitfly.xml").is_file():
        pytest.skip("Run cbbench bootstrap for native FlyBody integration tests")
    return str(path)


@pytest.mark.flybody
@pytest.mark.parametrize("task", ["balance", "walk", "reach"])
def test_actual_flybody_state_restore_and_action_contract(task):
    config = BodyConfig(task=task, source=source_path(), horizon=20)
    env = FlyBodyEnv(config)
    try:
        obs = env.reset(42)
        assert obs.ndim == 1 and np.isfinite(obs).all()
        assert env.action_dim > 50  # Stock full walking actuator interface.
        action = np.random.default_rng(1).uniform(-0.1, 0.1, env.action_dim)
        env.step(action)
        state = env.state_dict()
        expected = env.step(action)
        env.load_state_dict(state)
        actual = env.step(action)
        np.testing.assert_allclose(expected[0], actual[0], atol=1e-6, rtol=1e-6)
        assert expected[1:] == actual[1:]
        np.testing.assert_array_equal(env.reset(42), obs)
    finally:
        env.close()


@pytest.mark.flybody
def test_tasks_share_morphology_observation_and_action_dimensions():
    dimensions = []
    for task in ("balance", "walk", "reach"):
        env = FlyBodyEnv(BodyConfig(task=task, source=source_path()))
        try:
            dimensions.append((env.obs_dim, env.action_dim, env.action_names, env.obs_schema))
        finally:
            env.close()
    assert dimensions[0] == dimensions[1] == dimensions[2]
