from dataclasses import replace

import numpy as np
import pytest

from connectome_body.compatibility.bodies import BODY_TASKS, BodySpec, make_body


def assert_snapshot_replay(env):
    initial = env.reset(26)
    assert initial.shape == (env.obs_dim,) and np.isfinite(initial).all()
    action = np.random.default_rng(9).uniform(-0.1, 0.1, env.action_dim).astype(np.float32)
    env.step(action)
    saved = env.state_dict()
    expected = env.step(action)
    env.load_state_dict(saved)
    actual = env.step(action)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    assert actual[1:] == expected[1:]
    assert np.isfinite(actual[0]).all() and 0 <= actual[1] <= 1
    assert "success" in actual[-1] and "score" in actual[-1]
    np.testing.assert_array_equal(env.reset(26), initial)


@pytest.mark.flybody
@pytest.mark.parametrize("task", BODY_TASKS["fly"])
def test_native_fly_tasks_have_replayable_state(task):
    env = make_body(BodySpec(task=task, horizon=32))
    try:
        assert_snapshot_replay(env)
        assert env.evidence == "native_flybody"
        assert len(env.action_names) == env.action_dim
        if task == "limb_control":
            assert sum(size for _, size in env.obs_schema) == env.obs_dim
    finally:
        env.close()


@pytest.mark.parametrize("body,task", [(b, t) for b in ("worm", "fish") for t in BODY_TASKS[b]])
def test_manifest_tasks_and_physics_state_replay(compatibility_manifest_factory, body, task):
    manifest = compatibility_manifest_factory(body)
    env = make_body(BodySpec(body, task, horizon=32, model_manifest=str(manifest)))
    try:
        assert_snapshot_replay(env)
        assert env.evidence == "software_fixture_only"
        assert sum(field["size"] for field in env.obs_schema) == env.obs_dim
        clone = make_body(env.spec)
        try:
            assert clone.fingerprint == env.fingerprint
        finally:
            clone.close()
    finally:
        env.close()


def test_body_asset_change_and_invalid_configuration_are_rejected(compatibility_manifest_factory):
    path = compatibility_manifest_factory()
    spec = BodySpec("worm", "locomotion", model_manifest=str(path))
    with pytest.raises(ValueError, match="nonnegative"):
        make_body(replace(spec, parameters={"initial_noise": -0.1}))
    xml = path.parent / "model.xml"
    xml.write_text(xml.read_text().replace('gear="0.0001"', 'gear="0.0002"'))
    with pytest.raises(ValueError, match="asset"):
        make_body(spec)
    with pytest.raises(ValueError, match="manifest"):
        BodySpec("fish", "swimming").validate()
    with pytest.raises(ValueError, match="Task"):
        BodySpec("worm", "hover", model_manifest=str(path)).validate()
