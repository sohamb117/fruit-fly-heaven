"""Sparse routing geometry, gradients, and controller checkpoint integration."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from connectome_body.adaptation.ports import StructuralPorts
from connectome_body.compatibility.config import AdapterConfig, ControllerConfig, SubstrateConfig
from connectome_body.compatibility.controller import Controller
from connectome_body.graphs import make_fixture


@pytest.mark.parametrize("n,support", [(67, 7), (300, 256), (1024, 8)])
def test_balanced_ports_reject_common_mode_without_shortcuts_and_retain_gradients(n, support):
    features = np.random.default_rng(3).normal(size=(n, 6)).astype(np.float32)
    torch.manual_seed(4)
    ports = StructuralPorts(features, channels=4, support=support, routing="balanced_disjoint")
    mapping = ports.compile()
    matrices = []
    for nodes, weights in (
        (mapping.input_nodes, mapping.input_weights),
        (mapping.output_nodes, mapping.output_weights),
    ):
        assert nodes.unique().numel() == nodes.numel()
        assert nodes.shape[1] <= support
        matrix = torch.zeros(4, n).scatter(1, nodes, weights)
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(4))
        torch.testing.assert_close(matrix.sum(-1), torch.zeros(4), atol=2e-6, rtol=0)
        matrices.append(matrix)
    assert ports.overlap(mapping) == 0
    torch.testing.assert_close(matrices[0] @ matrices[1].T, torch.zeros(4, 4))
    torch.testing.assert_close(
        ports.read(torch.ones(2, n), mapping), torch.zeros(2, 4), atol=2e-6, rtol=0
    )

    # This sparse ring is the only path from input to output neurons. Compare
    # several recurrent steps to the independent dense projection expression.
    latent = torch.randn(2, 4, requires_grad=True)
    state, reference = torch.zeros(2, n), torch.zeros(2, n)
    for _ in range(5):
        state = torch.tanh(0.7 * state.roll(1, -1) + ports.inject(latent, mapping))
        reference = torch.tanh(0.7 * reference.roll(1, -1) + latent @ matrices[0])
    read = ports.read(state, mapping)
    torch.testing.assert_close(read, reference @ matrices[1].T)
    read.square().sum().backward()
    for parameter in (latent, ports.input_queries, ports.output_queries):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0
    # With no recurrent propagation, there is no immediate encoder-to-decoder path.
    torch.testing.assert_close(
        ports.read(ports.inject(latent, mapping), mapping), torch.zeros(2, 4)
    )


def test_routing_is_checkpointed_and_keeps_controller_capacity_and_causal_memory(tmp_path):
    graph = make_fixture(tmp_path / "graph", 128, 3)
    config = ControllerConfig(
        AdapterConfig(
            family="mlp_linear",
            budget=2000,
            channels=4,
            support=12,
            port_routing="balanced_disjoint",
        ),
        SubstrateConfig(graph="supplied-in-test"),
    )
    actor = Controller(7, 2, config, graph)
    legacy = Controller(
        7, 2, replace(config, adapter=replace(config.adapter, port_routing="positive")), graph
    )
    assert actor.trainable_parameters() == legacy.trainable_parameters()
    assert actor.ports.fingerprint != legacy.ports.fingerprint
    observations = torch.randn(6, 2, 7, requires_grad=True)
    state, context = actor.reset(2), actor.context()
    for obs in observations:
        action, state = actor(obs, state, context)
    action.square().sum().backward()
    assert observations.grad[0].abs().max() > 0
    assert actor.ports.input_queries.grad.abs().max() > 0
    assert actor.ports.output_queries.grad.abs().max() > 0
    checkpoint = tmp_path / "actor.pt"
    torch.save({"config": config.to_dict(), "actor": actor.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    resumed = Controller(7, 2, ControllerConfig.from_dict(saved["config"]), graph)
    resumed.load_state_dict(saved["actor"])
    state, context = resumed.reset(2), resumed.context()
    for obs in observations.detach():
        restored, state = resumed(obs, state, context)
    torch.testing.assert_close(action, restored, rtol=0, atol=0)


def test_legacy_default_is_identical_and_invalid_routing_fails_explicitly():
    features = np.random.default_rng(1).normal(size=(64, 6)).astype(np.float32)
    torch.manual_seed(8)
    implicit = StructuralPorts(features, channels=4, support=8)
    torch.manual_seed(8)
    explicit = StructuralPorts(features, channels=4, support=8, routing="positive")
    for key, value in vars(implicit.compile()).items():
        torch.testing.assert_close(value, getattr(explicit.compile(), key), rtol=0, atol=0)
    assert implicit.fingerprint == explicit.fingerprint
    with pytest.raises(ValueError, match=">=4K"):
        StructuralPorts(features[:15], channels=4, support=2, routing="balanced_disjoint")
    with pytest.raises(ValueError, match="support"):
        AdapterConfig(port_routing="balanced_disjoint", support=1).validate()
    with pytest.raises(ValueError, match="Unknown"):
        AdapterConfig(port_routing="typo").validate()
