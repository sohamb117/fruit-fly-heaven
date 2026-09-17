from dataclasses import replace

import numpy as np
import pytest

from connectome_body.compatibility.config import DynamicsConfig, SubstrateConfig
from connectome_body.compatibility.topology import (
    initialize_weights,
    prepare_topology,
    write_communities,
)
from connectome_body.graphs import make_fixture, save_graph
from connectome_body.util import atomic_json


@pytest.fixture
def graph(tmp_path):
    return make_fixture(tmp_path / "graph", 64, 8)


def test_degree_null_preserves_degrees_and_signs_but_changes_wiring(graph):
    config = SubstrateConfig(graph="fixture", topology="degree_rewired", seed=5)
    a, b = prepare_topology(graph, config), prepare_topology(graph, config)
    np.testing.assert_array_equal(a.src, b.src)
    np.testing.assert_array_equal(a.dst, b.dst)
    for endpoint in ("src", "dst"):
        np.testing.assert_array_equal(
            np.bincount(getattr(a, endpoint), minlength=graph.n),
            np.bincount(getattr(graph, endpoint), minlength=graph.n),
        )
    np.testing.assert_array_equal(
        np.bincount(a.dst, weights=a.signs[a.src], minlength=graph.n),
        np.bincount(graph.dst, weights=a.signs[graph.src], minlength=graph.n),
    )
    assert a.report["accepted_swaps"] > 0
    assert a.report["changed_target_fraction"] > 0


def test_strict_degree_null_preserves_both_weighted_degrees_and_signed_input_strength(graph):
    config = SubstrateConfig(
        graph="fixture", topology="degree_rewired", seed=5, preserve_strengths=True
    )
    null = prepare_topology(graph, config)
    for endpoint in ("src", "dst"):
        np.testing.assert_array_equal(
            np.bincount(getattr(null, endpoint), weights=null.weight, minlength=graph.n),
            np.bincount(getattr(graph, endpoint), weights=graph.weight, minlength=graph.n),
        )
    np.testing.assert_array_equal(
        np.bincount(null.dst, weights=null.weight * null.signs[null.src], minlength=graph.n),
        np.bincount(graph.dst, weights=graph.weight * null.signs[graph.src], minlength=graph.n),
    )
    assert null.report["changed_target_fraction"] > 0


def test_community_null_preserves_per_neuron_block_degrees(graph, tmp_path):
    labels = np.arange(graph.n) % 3
    partition = tmp_path / "communities.json"
    atomic_json(partition, {"graph_fingerprint": graph.fingerprint, "labels": labels.tolist()})
    t = prepare_topology(
        graph,
        SubstrateConfig(
            graph="fixture",
            topology="community_rewired",
            community_partition=str(partition),
            seed=5,
        ),
    )
    for label in range(3):
        before = labels[graph.dst] == label
        after = labels[t.dst] == label
        np.testing.assert_array_equal(
            np.bincount(graph.src[before], minlength=graph.n),
            np.bincount(t.src[after], minlength=graph.n),
        )
        before = labels[graph.src] == label
        after = labels[t.src] == label
        np.testing.assert_array_equal(
            np.bincount(graph.dst[before], minlength=graph.n),
            np.bincount(t.dst[after], minlength=graph.n),
        )
    assert t.report["partition_sha256"]


@pytest.mark.parametrize("variant", ["direction_shuffled", "random"])
def test_coarser_nulls_preserve_nodes_edges_and_weight_multiset(graph, variant):
    t = prepare_topology(graph, SubstrateConfig(graph="fixture", topology=variant, seed=6))
    assert t.n == graph.n and t.m == graph.m
    assert len(np.unique(t.src.astype(np.int64) * t.n + t.dst)) == graph.m
    np.testing.assert_array_equal(np.sort(t.weight), np.sort(graph.weight))
    if variant == "direction_shuffled":
        original = np.minimum(graph.src, graph.dst).astype(np.int64) * graph.n + np.maximum(
            graph.src, graph.dst
        )
        shuffled = np.minimum(t.src, t.dst).astype(np.int64) * t.n + np.maximum(t.src, t.dst)
        np.testing.assert_array_equal(np.sort(original), np.sort(shuffled))
        assert t.report["reversed_edges"] > 0


def test_weight_initialization_changes_only_weights_not_ports_or_topology(graph):
    config = SubstrateConfig(graph="fixture")
    t = prepare_topology(graph, config)
    original = t.weight.copy()
    native, native_signs, _ = initialize_weights(t, config)
    shuffled, signs, _ = initialize_weights(
        t, replace(config, initialization="shuffled_magnitudes")
    )
    np.testing.assert_array_equal(np.sort(native), np.sort(shuffled))
    np.testing.assert_array_equal(signs, native_signs)
    np.testing.assert_array_equal(original, t.weight)
    assert not np.array_equal(native, shuffled)
    for mode in ("random_magnitudes", "random_signed"):
        c = replace(config, initialization=mode)
        a, signs_a, _ = initialize_weights(t, c)
        b, signs_b, _ = initialize_weights(t, c)
        assert (a > 0).all() and np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(signs_a, signs_b)


def test_unknown_physiology_cannot_be_labeled_fully_annotated(graph):
    # make_fixture supplies synthetic labels, but an explicitly unknown copy
    # must be rejected by the fully biological-sign condition.
    from dataclasses import replace as replace_graph

    unknown = replace_graph(graph, signs=np.zeros(graph.n, dtype=np.int8))
    with pytest.raises(ValueError, match="unknown signs"):
        prepare_topology(
            unknown,
            SubstrateConfig(
                graph="fixture",
                dynamics=DynamicsConfig(sign_mode="annotated"),
            ),
        )


def test_random_source_signs_do_not_confound_identity_with_global_ei_ratio(graph):
    annotated = replace(graph, signs=np.r_[np.full(48, -1), np.ones(16)].astype(np.int8))
    config = SubstrateConfig(graph="fixture", dynamics=DynamicsConfig(sign_mode="available"))
    topology = prepare_topology(annotated, config)
    _, signs, _ = initialize_weights(topology, replace(config, initialization="random_signed"))
    assert np.count_nonzero(signs < 0) == 48
    assert not np.array_equal(signs, topology.signs)


def test_community_discovery_is_reproducible_and_reports_degeneracy(graph, tmp_path):
    a = write_communities(graph, tmp_path / "a.json", seed=1)
    b = write_communities(graph, tmp_path / "b.json", seed=1)
    assert a == b
    assert len(a["labels"]) == graph.n
    assert -1 <= a["modularity_undirected"] <= 1
    assert a["degenerate"] == (a["active_communities"] < 2)


def test_isolates_cannot_fake_multiple_communities_and_louvain_recovers_dense_blocks(tmp_path):
    edges = [
        (a, b)
        for start in (0, 8)
        for a in range(start, start + 8)
        for b in range(start, start + 8)
        if a != b
    ]
    edges += [(7, 8), (8, 7)]
    src, dst = np.array(edges).T
    g = save_graph(
        tmp_path / "blocks",
        np.arange(20),
        src,
        dst,
        np.ones(len(src)),
        provenance={"is_synthetic": True},
    )
    p = write_communities(g, tmp_path / "partition.json")
    assert p["active_communities"] == 2
    assert p["isolated_neuron_communities"] == 4
    assert p["modularity_undirected"] > 0.45
    labels = np.r_[np.zeros(16, dtype=int), np.arange(1, 5)]
    atomic_json(
        tmp_path / "fake.json", {"graph_fingerprint": g.fingerprint, "labels": labels.tolist()}
    )
    with pytest.raises(ValueError, match="single community"):
        prepare_topology(
            g,
            SubstrateConfig(
                graph="fixture",
                topology="community_rewired",
                community_partition=str(tmp_path / "fake.json"),
            ),
        )
