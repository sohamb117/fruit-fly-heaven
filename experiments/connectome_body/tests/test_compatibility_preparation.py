from dataclasses import replace

import pytest
import torch

from connectome_body.compatibility import preparation as prep
from connectome_body.compatibility.config import (
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.controller import Controller
from connectome_body.graphs import make_fixture


@pytest.mark.parametrize("topology", ["real", "degree_rewired"])
@pytest.mark.parametrize("plasticity", ["adapters", "joint"])
def test_cache_preserves_actions_gradients_and_initial_weights(
    tmp_path, monkeypatch, topology, plasticity
):
    path = tmp_path / "graph"
    make_fixture(path, 64, 3)
    config = ControllerConfig(
        AdapterConfig(budget=1200, channels=4, support=8),
        SubstrateConfig(
            graph=str(path),
            topology=topology,
            plasticity=plasticity,
            dynamics=DynamicsConfig(control_dt=0.01, tau_seconds=0.02),
        ),
    )
    monkeypatch.setenv("CONNECTOME_PREPARATION_CACHE", "off")
    uncached = Controller(7, 2, config)
    monkeypatch.setenv("CONNECTOME_PREPARATION_CACHE", str(tmp_path / "cache"))
    cold = Controller(7, 2, config)
    assert not cold.preparation_report["hit"]

    def unexpected(*args, **kwargs):
        raise AssertionError("A cache hit must not regenerate topology/features")

    monkeypatch.setattr(prep, "_build", unexpected)
    warm = Controller(7, 2, config)
    assert warm.preparation_report["hit"]
    observations = torch.randn(4, 2, 7)
    actions, gradients = [], []
    for model in (uncached, cold, warm):
        state, context = model.reset(2), model.context()
        outputs = []
        for observation in observations:
            action, state = model(observation, state, context)
            outputs.append(action)
        result = torch.stack(outputs)
        result.square().sum().backward()
        actions.append(result)
        gradients.append({k: v.grad for k, v in model.named_parameters() if v.requires_grad})
    for i in (1, 2):
        torch.testing.assert_close(actions[0], actions[i], rtol=0, atol=0)
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[i][name], rtol=0, atol=0)
    # Training one model never mutates the cached arrays or the next model.
    with torch.no_grad():
        for parameter in warm.parameters():
            parameter.add_(1)
    fresh = Controller(7, 2, config)
    for name, value in cold.state_dict().items():
        torch.testing.assert_close(value, fresh.state_dict()[name], rtol=0, atol=0)


def test_cache_reuse_boundaries_and_corruption(tmp_path, monkeypatch):
    path = tmp_path / "graph"
    graph = make_fixture(path, 64, 3)
    config = SubstrateConfig(graph=str(path))
    monkeypatch.setenv("CONNECTOME_PREPARATION_CACHE", str(tmp_path / "cache"))
    a = prep.prepare_substrate(graph, config)
    b = prep.prepare_substrate(
        graph,
        replace(config, plasticity="joint", dynamics=replace(config.dynamics, control_dt=0.01)),
    )
    assert a.cache_report["key"] == b.cache_report["key"] and b.cache_report["hit"]
    for other in (
        replace(config, seed=1),
        replace(config, topology="degree_rewired"),
        replace(config, preserve_strengths=True),
        replace(config, dynamics=replace(config.dynamics, inhibitory_fraction=0.3)),
    ):
        assert prep.preparation_key(graph, other) != prep.preparation_key(graph, config)
    target = tmp_path / "cache" / a.cache_report["key"] / "features.npy"
    with target.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        prep.prepare_substrate(graph, config)


def test_cache_required_before_paid_work(tmp_path, monkeypatch):
    path = tmp_path / "graph"
    graph = make_fixture(path, 64, 3)
    monkeypatch.setenv("CONNECTOME_PREPARATION_REQUIRE_CACHE", "1")
    with pytest.raises(FileNotFoundError, match="on CPU"):
        prep.prepare_substrate(graph, SubstrateConfig(graph=str(path)))
