import json
import shutil
from dataclasses import replace

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
from connectome_body.compatibility.controller import Controller
from connectome_body.compatibility.evaluation import InterventionSpec, StateIntervention
from connectome_body.compatibility.interventions import (
    acute_random_graph,
    calibrate_temporal_mean,
    evaluate_intervention,
)
from connectome_body.compatibility.robustness import PerturbationSpec, make_perturbed_body
from connectome_body.compatibility.training import policy_digest
from connectome_body.graphs import make_fixture
from connectome_body.util import atomic_json


@pytest.fixture
def model_and_body(tmp_path, compatibility_manifest_factory):
    graph = make_fixture(tmp_path / "graph", 32, 6)
    body = make_body(
        BodySpec(
            "worm", "steering", horizon=12, model_manifest=str(compatibility_manifest_factory())
        )
    )
    config = ControllerConfig(
        AdapterConfig(budget=600, channels=4, support=8),
        SubstrateConfig(
            graph=str(tmp_path / "graph"),
            dynamics=DynamicsConfig(control_dt=0.01, tau_seconds=0.02),
        ),
    )
    model = Controller(body.obs_dim, body.action_dim, config, graph)
    yield model, body
    body.close()


@pytest.mark.parametrize(
    "kind", ("zero", "time_shuffle", "neuron_permutation", "reset", "remove_edges", "random_graph")
)
def test_causal_evaluations_are_paired_and_do_not_change_weights(model_and_body, tmp_path, kind):
    model, body = model_and_body
    before = policy_digest(model)
    clean = evaluate_intervention(model, body, InterventionSpec(), episodes=2)
    output = tmp_path / f"{kind}.json"
    altered = evaluate_intervention(
        model, body, InterventionSpec(kind=kind), episodes=2, output=output
    )
    assert policy_digest(model) == before
    assert [r["seed"] for r in clean["episodes"]] == [r["seed"] for r in altered["episodes"]]
    assert altered["intervention"]["kind"] == kind
    assert (
        evaluate_intervention(model, body, InterventionSpec(kind=kind), episodes=2, output=output)
        == altered
    )


def test_zero_reset_permutation_and_causal_time_history_have_intended_semantics(model_and_body):
    model, _ = model_and_body
    state = torch.arange(model.state_dim, dtype=torch.float32).reshape(1, -1)
    assert not StateIntervention(model, InterventionSpec(kind="zero")).transform(state).any()
    permutation = StateIntervention(model, InterventionSpec(kind="neuron_permutation"))
    assert torch.equal(permutation.transform(state), state[:, permutation.permutation])
    history = StateIntervention(model, InterventionSpec(kind="time_shuffle", history=2))
    torch.testing.assert_close(history.transform(state), state)
    torch.testing.assert_close(history.transform(state + 10), state)
    assert len(history.history) == 2
    reset = StateIntervention(model, InterventionSpec(kind="reset"))
    observation, context = torch.ones(1, model.obs_dim), model.context()
    a, actual = reset.action(observation, state, context)
    b, expected = model(observation, model.reset(1), context)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(actual, expected)


def test_mean_calibration_rejects_test_data_and_other_policies(model_and_body, tmp_path):
    model, body = model_and_body
    path = tmp_path / "mean.npz"
    with pytest.raises(ValueError, match="held-out"):
        calibrate_temporal_mean(model, body, path, episodes=1, split="test")
    record = calibrate_temporal_mean(model, body, path, episodes=2)
    assert record["neural_states"] > 0
    # Resume after calibration commits but before a job-result commit.
    assert calibrate_temporal_mean(model, body, path, episodes=2) == record
    with pytest.raises(ValueError, match="another request"):
        calibrate_temporal_mean(model, body, path, episodes=3)
    spec = InterventionSpec(kind="temporal_mean", mean_file=str(path))
    result = evaluate_intervention(model, body, spec, episodes=1)
    assert result["intervention"]["calibration"]["split"] == "validation"
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="development"):
        evaluate_intervention(model, body, spec, episodes=1)


def test_targeted_lesions_are_checked_against_graph_identity(model_and_body, tmp_path):
    model, body = model_and_body
    path = tmp_path / "lesion.json"
    atomic_json(
        path,
        {
            "graph_fingerprint": model.graph_fingerprint,
            "provenance": {"test": "explicit fixture lesion"},
            "node_ids": [str(node) for node in model.topology.node_ids[:8]],
        },
    )
    spec = InterventionSpec(kind="targeted_lesion", lesion_file=str(path))
    perturbation = StateIntervention(model, spec)
    _, state = perturbation.action(
        torch.ones(1, model.obs_dim), torch.ones(1, model.state_dim), model.context()
    )
    assert not state[:, :8].any()
    assert evaluate_intervention(model, body, spec, episodes=1)["intervention"]["neurons"] == 8
    data = json.loads(path.read_text())
    data["graph_fingerprint"] = "incorrect"
    atomic_json(path, data)
    with pytest.raises(ValueError, match="identity"):
        StateIntervention(model, spec)


def test_acute_rewiring_preserves_ports_and_recovers_after_exception(model_and_body):
    model, _ = model_and_body
    core, ports, before = model.core, model.ports, policy_digest(model)
    with pytest.raises(RuntimeError, match="injected"):
        with acute_random_graph(model, 23):
            assert model.core is not core and model.ports is ports
            assert model.core.src.numel() == core.src.numel()
            torch.testing.assert_close(
                model.core.initial_magnitudes.sort().values, core.initial_magnitudes.sort().values
            )
            raise RuntimeError("injected evaluator failure")
    assert model.core is core and policy_digest(model) == before


def test_sensor_delay_noise_mass_and_failure_are_real_and_replayable(
    compatibility_manifest_factory,
):
    base = BodySpec(
        "worm", "steering", horizon=20, model_manifest=str(compatibility_manifest_factory())
    )
    nominal = make_body(base)
    altered = make_perturbed_body(
        base,
        PerturbationSpec(
            sensor_noise=0.05,
            sensor_delay=2,
            mass_scale=1.5,
            inertia_scale=1.5,
            failed_actuators=(0,),
        ),
    )
    try:
        np.testing.assert_allclose(altered.model.body_mass, nominal.model.body_mass * 1.5)
        assert altered.model.actuator_gainprm[0, 0] == 0
        initial = altered.reset(24)
        following = altered.step(np.ones(1))[0]
        np.testing.assert_array_equal(initial, following)  # Two-decision delay.
        assert np.all(altered.data.actuator_force == 0)
        saved = altered.state_dict()
        expected = altered.step(np.ones(1))
        altered.load_state_dict(saved)
        actual = altered.step(np.ones(1))
        np.testing.assert_array_equal(expected[0], actual[0])
        assert expected[1:] == actual[1:]
    finally:
        nominal.close()
        altered.close()


def test_portable_body_identity_and_nonexistent_physics_rejection(
    tmp_path, compatibility_manifest_factory
):
    path = compatibility_manifest_factory()
    base = BodySpec("worm", "locomotion", model_manifest=str(path))
    target = tmp_path / "relocated"
    shutil.copytree(path.parent, target)
    a, b = make_body(base), make_body(replace(base, model_manifest=str(target / "manifest.json")))
    try:
        assert a.fingerprint == b.fingerprint
    finally:
        a.close()
        b.close()
    with pytest.raises(ValueError, match="fluid density"):
        make_perturbed_body(base, PerturbationSpec(fluid_density_scale=1.3))
    with pytest.raises(ValueError, match="ground plane"):
        make_perturbed_body(base, PerturbationSpec(terrain_slope=0.2))
