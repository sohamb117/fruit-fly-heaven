from dataclasses import replace

import numpy as np
import pytest
import torch

from connectome_body.compatibility.adapters import SparseLinear, allocate_maps
from connectome_body.compatibility.config import (
    ADAPTER_FAMILIES,
    PLASTICITY_REGIMES,
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.controller import Controller
from connectome_body.compatibility.dynamics import SparseRateCore
from connectome_body.graphs import make_fixture
from connectome_body.util import atomic_json


@pytest.fixture
def graph(tmp_path):
    return make_fixture(tmp_path / "graph", 64, 3)


def configuration(graph, **changes):
    substrate = SubstrateConfig(
        graph="supplied-in-test", dynamics=DynamicsConfig(control_dt=0.01, tau_seconds=0.02)
    )
    config = ControllerConfig(AdapterConfig(budget=1200, channels=4, support=8), substrate)
    return replace(config, **changes)


@pytest.mark.parametrize("family", ADAPTER_FAMILIES)
def test_each_adapter_family_is_stateless_and_obeys_capacity(graph, tmp_path, family):
    mapping = None
    if family == "anatomical":
        mapping = str(tmp_path / "oracle.json")
        atomic_json(
            mapping,
            {
                "schema": "anatomical-ports-v1",
                "graph_fingerprint": graph.fingerprint,
                "provenance": {"fixture": "declared test-only anatomy"},
                "input_groups": [[str(node)] for node in graph.node_ids[:4]],
                "output_groups": [[str(node)] for node in graph.node_ids[-4:]],
            },
        )
    config = configuration(graph)
    config = replace(
        config, adapter=replace(config.adapter, family=family, anatomical_mapping=mapping)
    )
    model = Controller(7, 2, config, graph)
    assert model.trainable_parameters() <= config.adapter.budget
    assert model.adapter_report["allocated"] == model.trainable_parameters()
    assert not any(
        isinstance(m, (torch.nn.RNNBase, torch.nn.RNNCellBase))
        for m in (list(model.encoder.modules()) + list(model.decoder.modules()))
    )
    state, context = model.reset(2), model.context()
    for _ in range(5):
        actions, state = model(torch.randn(2, 7), state, context)
    actions.square().sum().backward()
    assert torch.isfinite(actions).all() and actions.abs().max() <= 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    assert not list(model.core.parameters())


@pytest.mark.parametrize("regime", PLASTICITY_REGIMES)
def test_optimizer_changes_exactly_the_requested_parameter_groups(graph, regime):
    config = configuration(graph)
    config = replace(config, substrate=replace(config.substrate, plasticity=regime))
    model = Controller(7, 2, config, graph)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.02)
    state, context = model.reset(3), model.context()
    generator = torch.Generator().manual_seed(18)
    for _ in range(6):
        actions, state = model(torch.randn(3, 7, generator=generator), state, context)
    (actions - 0.4).square().mean().backward()
    optimizer.step()
    changed = set()
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter, before[name]):
            changed.add(name.split(".")[0])
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            assert parameter.grad is None
    expected = {
        "adapters": {"encoder", "decoder", "ports"},
        "substrate": {"core"},
        "encoder": {"encoder", "ports"},
        "decoder": {"decoder", "ports"},
        "joint": {"encoder", "decoder", "ports", "core"},
    }[regime]
    assert changed == expected
    assert model.ports.input_queries.requires_grad == (regime in ("adapters", "encoder", "joint"))
    assert model.ports.output_queries.requires_grad == (regime in ("adapters", "decoder", "joint"))
    if model.core.raw_magnitudes is not None:
        values = model.core.edge_values().detach()
        assert values.numel() == graph.m
        assert torch.equal(values.sign(), model.core.source_signs[model.core.src])


@pytest.mark.parametrize("kind", ["rnn", "gru"])
@pytest.mark.parametrize("family", ["linear_linear", "mlp_mlp", "low_rank", "sparse_structured"])
def test_learned_recurrence_counts_toward_the_total_budget(kind, family):
    config = ControllerConfig(
        AdapterConfig(family=family, budget=1200, channels=4),
        SubstrateConfig(kind=kind, plasticity="joint", total_budget=1200),
    )
    model = Controller(7, 2, config)
    assert model.trainable_parameters() <= 1200
    assert model.parameter_report["substrate_trainable"] > 0
    assert model.parameter_report["readout_trainable"] > 0
    assert model.state_dim > 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.core.weight_hh.detach().clone()
    state = model.reset(2)
    for _ in range(3):
        action, state = model(torch.ones(2, 7), state)
    action.square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model.core.weight_hh)


def test_no_brain_has_no_persistent_state_and_needs_no_graph():
    config = ControllerConfig(AdapterConfig(budget=5000), SubstrateConfig(kind="adapter_only"))
    model = Controller(104, 12, config)
    assert model.state_dim == 0 and model.ports is None and model.core is None
    observation = torch.randn(3, 104)
    a, state = model(observation, model.reset(3))
    model(torch.randn(3, 104), state)
    b, _ = model(observation, state)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_linear_maps_do_not_pad_unused_budget_with_irrelevant_parameters():
    small = allocate_maps(7, 2, AdapterConfig(family="linear_linear", channels=4, budget=100))[2]
    large = allocate_maps(7, 2, AdapterConfig(family="linear_linear", channels=4, budget=10000))[2]
    assert small["allocated"] == large["allocated"] == 42
    assert large["unused_budget"] == 9958
    with pytest.raises(ValueError, match="ceiling"):
        allocate_maps(7, 2, AdapterConfig(family="linear_linear", channels=4, budget=20))


def test_sparse_map_optimizes_only_declared_coefficients():
    layer = SparseLinear(10, 4, 0.2, 3)
    assert layer.weight.numel() == 8 and layer.bias.numel() == 4
    x = torch.randn(3, 10, requires_grad=True)
    dense = torch.zeros(4, 10).scatter(1, layer.columns, layer.weight)
    torch.testing.assert_close(layer(x), x @ dense.T + layer.bias)


def test_sparse_weight_and_state_gradients_match_dense_reference():
    src = np.array([0, 1, 2, 3, 0, 2])
    dst = np.array([1, 2, 3, 0, 2, 0])
    core = SparseRateCore(
        4,
        src,
        dst,
        np.arange(1, 7),
        np.array([1, -1, 1, -1]),
        DynamicsConfig(gradient_edge_chunk=2),
        plastic=True,
    ).double()
    state = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    values = core.edge_values()
    sparse_output = core.multiply(state, values)
    dense_matrix = torch.zeros(4, 4, dtype=torch.float64).index_put((core.dst, core.src), values)
    dense_output = state @ dense_matrix.T
    torch.testing.assert_close(sparse_output, dense_output, rtol=1e-12, atol=1e-12)
    sparse_grad = torch.autograd.grad(
        sparse_output.square().sum(), (state, core.raw_magnitudes), retain_graph=True
    )
    dense_grad = torch.autograd.grad(dense_output.square().sum(), (state, core.raw_magnitudes))
    for actual, expected in zip(sparse_grad, dense_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert torch.autograd.gradcheck(
        lambda s, v: core.multiply(s, v), (state, values.detach().requires_grad_())
    )


def test_plastic_weights_cannot_create_edges_or_escape_normalization(graph):
    core = SparseRateCore(
        graph.n, graph.src, graph.dst, graph.weight, np.ones(graph.n), plastic=True
    )
    with torch.no_grad():
        core.raw_magnitudes.uniform_(-20, 20)
    values = core.edge_values()
    matrix = torch.zeros(graph.n, graph.n).index_put((core.dst, core.src), values)
    assert torch.count_nonzero(matrix) == graph.m
    assert torch.linalg.matrix_norm(matrix, 2) <= 1 + 1e-6
    # Immutable sparse matrices are reconstructed from the pinned graph/config;
    # only the permitted-edge trainable state belongs in a model checkpoint.
    assert set(core.state_dict()) == {"raw_magnitudes"}


def test_node_lesion_cuts_input_and_state_at_every_neural_substep(graph):
    core = SparseRateCore(graph.n, graph.src, graph.dst, graph.weight, np.ones(graph.n))
    mask = torch.zeros(graph.n, dtype=torch.bool)
    mask[:20] = True
    state = core(torch.ones(2, graph.n), torch.ones(2, graph.n), lesion_mask=mask)
    assert not state[:, mask].any()
    assert state[:, ~mask].any()


def test_adapter_capacity_is_independent_of_graph_size(graph, tmp_path):
    larger = make_fixture(tmp_path / "larger", 128, 2)
    config = configuration(graph)
    a = Controller(7, 2, config, graph)
    b = Controller(7, 2, config, larger)
    assert a.adapter_report == b.adapter_report
    assert a.trainable_parameters() == b.trainable_parameters()


def test_substrate_only_training_is_invalid_without_a_substrate():
    with pytest.raises(ValueError, match="no substrate"):
        SubstrateConfig(kind="adapter_only", plasticity="substrate").validate()
