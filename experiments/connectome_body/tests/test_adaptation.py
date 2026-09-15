from dataclasses import replace

import numpy as np
import pytest
import torch

from connectome_body.adaptation.features import structural_features
from connectome_body.adaptation.imitation import (
    AdapterSpec,
    Optimization,
    evaluate_cache,
    load_actor,
    train_offline,
)
from connectome_body.adaptation.models import GenericAdapter
from connectome_body.adaptation.nulls import weighted_degree_null
from connectome_body.adaptation.substrates import RateConfig
from connectome_body.adaptation.temporal import temporal_cache
from connectome_body.adaptation.trajectories import TrajectoryCache, assert_disjoint_scenarios
from connectome_body.graphs import make_fixture, neuron_signs
from connectome_body.util import seed_everything


@pytest.fixture
def graph(tmp_path):
    return make_fixture(tmp_path / "graph", 64, 0)


def test_structural_features_ignore_neuron_names_and_order(graph):
    features, _ = structural_features(graph.n, graph.src, graph.dst, graph.weight)
    order = np.random.default_rng(11).permutation(graph.n)
    remap = np.argsort(order)
    permuted, _ = structural_features(graph.n, remap[graph.src], remap[graph.dst], graph.weight)
    np.testing.assert_allclose(permuted, features[order], atol=2e-6)


def test_weighted_null_preserves_signed_strengths(graph):
    signs = neuron_signs(graph, "random_dale", 0.2, 3)
    src, dst, weight, report = weighted_degree_null(graph, signs, 3)
    assert report["accepted_swaps"] > 0
    assert np.all(src != dst)
    assert len(np.unique(src.astype(np.int64) * graph.n + dst)) == graph.m
    for values in (None, weight, signs[src], weight * signs[src]):
        np.testing.assert_allclose(
            np.bincount(dst, weights=values, minlength=graph.n),
            np.bincount(graph.dst, weights=values, minlength=graph.n),
        )
    np.testing.assert_array_equal(dst, weighted_degree_null(graph, signs, 3)[1])


def test_queries_are_trainable_disjoint_and_budgeted(graph):
    seed_everything(2)
    model = GenericAdapter(3, 1, 5000, 4, 8, graph)
    context = model.context()
    assert model.ports.overlap(context) == 0
    assert model.trainable_parameters() <= 5000
    assert model.parameter_report["port_queries"] == 2 * 4 * 12
    assert not any(
        isinstance(m, (torch.nn.GRU, torch.nn.GRUCell, torch.nn.LSTM)) for m in model.modules()
    )
    state = model.reset(2)
    for _ in range(6):
        action, state = model(torch.randn(2, 3), state, context)
    action.square().sum().backward()
    assert model.ports.input_queries.grad.norm() > 0
    assert model.ports.output_queries.grad.norm() > 0
    assert list(model.substrate.parameters()) == []
    assert set(model.ports.output_permutation[model.ports.output_pool].tolist()) == set(
        model.ports.output_pool.tolist()
    )


def test_capacity_independent_of_neuron_count(graph, tmp_path):
    larger = make_fixture(tmp_path / "larger", 128, 0)
    small = GenericAdapter(3, 1, 5000, 4, 8, graph)
    large = GenericAdapter(3, 1, 5000, 4, 8, larger)
    assert small.trainable_parameters() == large.trainable_parameters()


def test_lesion_removes_input_output_path(graph):
    model = GenericAdapter(3, 1, 5000, 4, 8, graph)
    context = model.context()
    state = model.reset(2)
    for _ in range(10):
        action, state = model(torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]]), state, context, "lesion")
    torch.testing.assert_close(action[0], action[1], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["adapter_only", "trainable_gru"])
def test_no_brain_controls_fit_total_budget(variant):
    model = GenericAdapter(104, 12, 5000, variant=variant)
    assert model.trainable_parameters() <= 5000
    assert model.substrate is None
    if variant == "adapter_only":
        assert model.state_dim == 0


def test_temporal_queries_cannot_be_solved_without_memory(tmp_path):
    for task in ("memory", "delayed_xor"):
        cache = temporal_cache(tmp_path / task, task, episodes=16, length=16)
        observations, targets = [], []
        for idx in range(len(cache)):
            item = cache.load(idx)
            mask = item["mask"].astype(bool)
            observations.extend(item["observations"][:-1][mask])
            targets.extend(item["actions"][mask])
        np.testing.assert_array_equal(np.unique(observations, axis=0), [[0, 0, 1]])
        assert float(np.mean(targets)) == 0


def test_cache_integrity_and_split_leakage(tmp_path):
    cache = temporal_cache(tmp_path / "cache", episodes=8)
    with pytest.raises(ValueError, match="same cache"):
        assert_disjoint_scenarios(cache, cache)
    path = cache.path / cache.manifest["trajectories"][0]["file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        TrajectoryCache(cache.path)


def test_imitation_resume_matches_uninterrupted(tmp_path):
    train = temporal_cache(tmp_path / "train", episodes=8, length=8)
    val = temporal_cache(tmp_path / "val", episodes=8, length=8, split="validation")
    spec = AdapterSpec(variant="trainable_gru", budget=600, channels=4)
    opt = Optimization(
        updates=4, batch_size=2, sequence_length=4, burn_in=4, eval_every=2, checkpoint_every=1
    )
    train_offline(spec, opt, [train.path], val.path, tmp_path / "full")
    train_offline(spec, opt, [train.path], val.path, tmp_path / "resume", stop_after=2)
    train_offline(spec, opt, [train.path], val.path, tmp_path / "resume", resume=True)
    a = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/latest.pt", weights_only=False)
    for name in a["actor"]:
        torch.testing.assert_close(a["actor"][name], b["actor"][name], rtol=0, atol=0)
    assert a["curve"] == b["curve"]
    model, _, _ = load_actor(tmp_path / "resume")
    assert np.isfinite(evaluate_cache(model, val, "cpu")["mse"])
    with pytest.raises(ValueError, match="identity changed"):
        train_offline(
            replace(spec, seed=8), opt, [train.path], val.path, tmp_path / "resume", resume=True
        )


def test_full_graph_gradient_shape_at_10000_nodes(tmp_path):
    graph = make_fixture(tmp_path / "large", 10000, 0)
    model = GenericAdapter(3, 1, 1000, 4, 8, graph, rate=RateConfig(control_dt=0.01))
    context, state = model.context(), model.reset(1)
    for _ in range(8):
        prediction, state = model(torch.ones(1, 3), state, context)
    prediction.square().mean().backward()
    assert torch.isfinite(model.encoder[0].weight.grad).all()
    assert model.encoder[0].weight.grad.norm() > 0


def test_saved_actor_rejects_replaced_graph_with_same_dimensions(tmp_path):
    graph_path = tmp_path / "graph"
    make_fixture(graph_path, 64, 0)
    train = temporal_cache(tmp_path / "train", episodes=4, length=8)
    val = temporal_cache(tmp_path / "val", episodes=4, length=8, split="validation")
    spec = AdapterSpec(graph=str(graph_path), budget=1000, channels=4, support=8)
    opt = Optimization(
        updates=1, batch_size=1, sequence_length=8, burn_in=0, eval_every=1, checkpoint_every=1
    )
    train_offline(spec, opt, [train.path], val.path, tmp_path / "actor")
    graph_path.rename(tmp_path / "original_graph")
    make_fixture(graph_path, 64, 19)
    with pytest.raises(ValueError, match="graph differs"):
        load_actor(tmp_path / "actor")
