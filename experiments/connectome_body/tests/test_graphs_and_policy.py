import dataclasses
import hashlib

import numpy as np
import pytest
import torch

from connectome_body.config import BodyConfig, DynamicsConfig, RunConfig
from connectome_body.graphs import (
    Graph,
    _random_edges,
    _swap_edges,
    make_fixture,
    matched_subgraph,
    neuron_signs,
    normalized_matrix,
    save_graph,
    transform_graph,
)
from connectome_body.policy import BASELINES, Actor, HashPorts, _FixedSparseMultiply


@pytest.fixture
def graph(tmp_path):
    return make_fixture(tmp_path / "graph", 64)


def config(graph, **kwargs):
    return RunConfig(
        graph="unused",
        body=BodyConfig(backend="fixture"),
        dynamics=DynamicsConfig(channels=8),
        adapter_budget=2048,
        **kwargs,
    )


def test_lossless_large_ids_and_integrity(graph, tmp_path):
    assert int(graph.node_ids[0]) >= 2**63
    assert len(set(graph.node_ids)) == graph.n
    path = tmp_path / "graph/src.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="changed"):
        Graph.load(tmp_path / "graph")


def test_duplicate_aggregation_and_no_autapses(tmp_path):
    graph = save_graph(
        tmp_path / "g",
        ["a", "b", "c", "d"],
        np.array([0, 0, 1, 2]),
        np.array([1, 1, 1, 3]),
        [2, 3, 7, 11],
        provenance={"dataset": "test"},
    )
    assert graph.m == 2
    assert sorted(graph.weight) == [5, 11]
    assert graph.manifest["summary"]["removed_self_edges"] == 1
    with pytest.raises(ValueError, match="floating"):
        save_graph(
            tmp_path / "bad",
            np.arange(4, dtype=float),
            np.array([0]),
            np.array([1]),
            [1],
            provenance={},
        )


def test_degree_null_preserves_signed_degrees_and_weights(graph):
    signs = neuron_signs(graph, "random_dale", 0.25, 3)
    src, dst, weight, report = transform_graph(graph, "degree_shuffled", signs, 3)
    assert report["accepted_swaps"] > 0
    assert report["changed_target_fraction"] > 0.5
    assert np.array_equal(src, graph.src)
    assert np.array_equal(weight, graph.weight)
    for sign in [-1, 1]:
        keep = signs[src] == sign
        assert np.array_equal(
            np.bincount(dst[keep], minlength=graph.n),
            np.bincount(graph.dst[keep], minlength=graph.n),
        )
    assert np.all(src != dst)
    assert len(set(zip(src, dst, strict=True))) == graph.m
    again = transform_graph(graph, "degree_shuffled", signs, 3)[1]
    assert np.array_equal(dst, again)


def test_matched_random_is_matched_without_fake_degree_claim(graph):
    signs = neuron_signs(graph, "random_dale", 0.2, 0)
    src, dst, weight, report = transform_graph(graph, "matched_random", signs, 0)
    assert len(src) == graph.m
    assert np.all(src != dst)
    assert len(set(zip(src, dst, strict=True))) == graph.m
    assert np.array_equal(np.sort(weight), np.sort(graph.weight))
    assert "degree_sequence" in report["does_not_preserve"]


def test_fast_nulls_match_seeded_reference_before_hash_table_optimization():
    n = 64
    src = np.repeat(np.arange(n, dtype=np.uint32), 4)
    dst = ((src.reshape(n, 4) + np.array([1, 2, 5, 7])) % n).ravel().astype(np.uint32)
    signs = np.where(np.arange(n) % 5 == 0, -1, 1).astype(np.int8)
    dst, accepted = _swap_edges(src, dst, signs, n, len(src) * 10, 7)
    assert accepted == 1511
    assert (
        hashlib.sha256(dst.tobytes()).hexdigest()
        == "86b5d7715863ec3f2a95f32b15f606ac53ea9cac926cae044deb2812d9386015"
    )
    pre, post = _random_edges(n, 256, 7)
    assert (
        hashlib.sha256(pre.tobytes() + post.tobytes()).hexdigest()
        == "9e7ed788a28b86ecdb8e9991a3a2576b10e73d6acd6a9e1a1bd24ae3fa6a42c1"
    )


def test_normalization_is_contractive_and_oriented(graph):
    signs = neuron_signs(graph, "random_dale", 0.2, 0)
    matrix = normalized_matrix(graph.n, graph.src, graph.dst, graph.weight, signs).toarray()
    assert np.linalg.norm(matrix, 2) <= 1.00001
    edge = 3
    assert np.sign(matrix[graph.dst[edge], graph.src[edge]]) == signs[graph.src[edge]]


def test_ports_have_no_direct_observation_shortcut():
    ports = HashPorts(64, 8, 0)
    z = torch.randn(3, 8)
    assert torch.count_nonzero(ports.read(ports.inject(z))) == 0
    assert torch.count_nonzero(ports.read(ports.inject(z), permute=True)) == 0
    assert not set(ports.input_nodes.tolist()) & set(ports.output_nodes.tolist())


def test_parameter_count_independent_of_neuron_count(graph, tmp_path):
    other = make_fixture(tmp_path / "other", 128)
    first = Actor(12, 3, config(graph), graph)
    second = Actor(12, 3, config(other), other)
    assert first.num_parameters == second.num_parameters <= 2048
    assert not list(first.brain.parameters())
    for kind in BASELINES:
        baseline = Actor(12, 3, config(graph, substrate=kind))
        assert baseline.num_parameters <= 2048
    mlp = Actor(12, 3, config(graph, substrate="adapter_only"))
    assert first.num_parameters == mlp.num_parameters


def test_encoder_learns_through_frozen_edges(graph):
    torch.manual_seed(2)
    actor = Actor(12, 3, config(graph), graph)
    initial = actor.brain.matrix.values().clone()
    state = actor.initial_state(2)
    for _ in range(5):
        mean, _, state = actor(torch.randn(2, 12), state)
    mean.square().sum().backward()
    assert actor.encoder[0].weight.grad.norm() > 0
    assert actor.decoder[-1].weight.grad.norm() > 0
    assert not actor.brain.matrix.requires_grad
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    optimizer.step()
    assert torch.equal(initial, actor.brain.matrix.values())


def test_episode_reset_cuts_gradients_to_previous_episode(graph):
    actor = Actor(12, 3, config(graph), graph)
    before = torch.randn(1, 12, requires_grad=True)
    after = torch.randn(1, 12, requires_grad=True)
    _, _, state = actor(before, actor.initial_state(1))
    mean, _, _ = actor(after, state, torch.tensor([True]))
    mean.square().sum().backward()
    assert before.grad is None or torch.count_nonzero(before.grad) == 0
    assert after.grad.norm() > 0


def test_fixed_sparse_backward_matches_dense_reference(graph):
    actor = Actor(12, 3, config(graph), graph)
    state = torch.randn(2, graph.n, requires_grad=True)
    dense_state = state.detach().clone().requires_grad_(True)
    sparse_value = _FixedSparseMultiply.apply(state, actor.brain.matrix, actor.brain.transpose)
    dense_value = dense_state @ actor.brain.matrix.to_dense().T
    sparse_value.sin().sum().backward()
    dense_value.sin().sum().backward()
    torch.testing.assert_close(sparse_value, dense_value, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state.grad, dense_state.grad, atol=1e-6, rtol=1e-5)


def test_size_control_retains_only_original_edges_and_has_distinct_identity(graph, tmp_path):
    subset = matched_subgraph(graph, tmp_path / "subset", 48, 80, 1)
    again = matched_subgraph(graph, tmp_path / "again", 48, 80, 1)
    assert subset.n == 48 and subset.m == 80
    assert subset.fingerprint == again.fingerprint != graph.fingerprint
    original = {
        (graph.node_ids[s], graph.node_ids[d]): w
        for s, d, w in zip(graph.src, graph.dst, graph.weight, strict=True)
    }
    for s, d, w in zip(subset.src, subset.dst, subset.weight, strict=True):
        assert original[subset.node_ids[s], subset.node_ids[d]] == w
    assert subset.manifest["provenance"]["is_synthetic"]
    with pytest.raises(ValueError, match="fewer"):
        matched_subgraph(graph, tmp_path / "impossible", 4, 200, 1)


def test_no_edges_and_silence_remove_observation_dependence(graph):
    actor = Actor(12, 3, config(graph, substrate="no_edges"), graph)
    mean, _, _ = actor(torch.randn(2, 12), actor.initial_state(2))
    assert torch.equal(mean[0], mean[1])
    real = Actor(12, 3, config(graph), graph)
    mean, _, _ = real(torch.randn(2, 12), real.initial_state(2), intervention="silence")
    assert torch.equal(mean[0], mean[1])


def test_tanh_log_probability_is_finite_at_extreme_actions():
    raw = torch.tensor([[100.0, -100.0, 0.0]], requires_grad=True)
    logp = Actor.log_probability(torch.zeros_like(raw), torch.zeros_like(raw), raw)
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_gru_fits_realistic_large_observation_budget(graph):
    c = dataclasses.replace(config(graph), adapter_budget=5000, substrate="trainable_gru")
    actor = Actor(300, 59, c)
    assert actor.num_parameters <= 5000
    mean, _, state = actor(torch.zeros(2, 300), actor.initial_state(2))
    assert mean.shape == (2, 59) and state.shape[1] > 0
