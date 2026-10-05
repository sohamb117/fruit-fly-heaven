"""Auditable topology controls and initialization, independent of controller learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy import sparse

from ..adaptation.nulls import _weighted_swaps
from ..graphs import Graph, neuron_signs, transform_graph
from ..util import atomic_json, digest_file, digest_json, seed_for
from .config import SubstrateConfig


@dataclass
class Topology:
    node_ids: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    signs: np.ndarray
    report: dict

    @property
    def n(self):
        return len(self.node_ids)

    @property
    def m(self):
        return len(self.src)


def _partition(path, graph):
    value = json.loads(Path(path).read_text())
    if value.get("graph_fingerprint") != graph.fingerprint:
        raise ValueError("Community partition belongs to another graph")
    labels = np.asarray(value["labels"])
    if labels.shape != (graph.n,) or labels.dtype.kind not in "iu":
        raise ValueError("Community labels must be one integer per canonical neuron")
    _, labels = np.unique(labels, return_inverse=True)
    active = np.unique(np.r_[graph.src, graph.dst])
    if len(np.unique(labels[active])) < 2:
        raise ValueError("A single community cannot test community-specific organization")
    return labels, digest_file(path)


def _swaps(graph, signs, config):
    # Grouping both endpoint communities preserves each neuron's in/out degree
    # to every community, a stronger constraint than the global block counts.
    grouping = [signs[graph.src]]
    preserved = ["N", "M", "in_degree", "out_degree", "signed_in_degree", "global_weights"]
    partition_digest = None
    if config.topology == "community_rewired":
        labels, partition_digest = _partition(config.community_partition, graph)
        grouping += [labels[graph.src], labels[graph.dst]]
        preserved += [
            "community_membership",
            "per_neuron_community_degrees",
            "community_edge_counts",
        ]
    if config.preserve_strengths:
        grouping.append(graph.weight)
        preserved += ["in_strength", "out_strength", "signed_in_strength"]
    else:
        preserved += ["out_strength", "per_source_weight_multiset"]
    order = np.lexsort(tuple(grouping))
    boundary = np.zeros(graph.m, dtype=bool)
    boundary[0] = True
    for group in grouping:
        boundary[1:] |= group[order[1:]] != group[order[:-1]]
    first = np.flatnonzero(boundary)
    last = np.r_[first[1:], graph.m]
    starts, ends = np.repeat(first, last - first), np.repeat(last, last - first)
    dst, accepted = _weighted_swaps(
        np.asarray(graph.src),
        np.array(graph.dst),
        order,
        starts,
        ends,
        graph.n,
        config.swap_attempts_per_edge * graph.m,
        seed_for(config.seed, config.topology),
    )
    if accepted == 0:
        raise ValueError("No admissible rewiring swaps; this null is unidentifiable on this graph")
    return (
        graph.src.copy(),
        dst,
        graph.weight.copy(),
        {
            "algorithm": "directed double-edge swaps within declared sign/weight/community strata",
            "accepted_swaps": int(accepted),
            "attempts": config.swap_attempts_per_edge * graph.m,
            "changed_target_fraction": float(np.mean(dst != graph.dst)),
            "partition_sha256": partition_digest,
            "preserves": preserved,
            "limitation": "Finite swap chain; uniform sampling and adequate mixing are not assumed",
        },
    )


def prepare_topology(graph: Graph, config: SubstrateConfig):
    config.validate()
    mode = config.dynamics.sign_mode
    if mode == "annotated" and np.any(graph.signs == 0):
        raise ValueError("Fully annotated signs requested but some neurons have unknown signs")
    signs = neuron_signs(graph, "random_dale", config.dynamics.inhibitory_fraction, config.seed)
    if mode in ("annotated", "available"):
        signs = np.where(graph.signs == 0, signs, graph.signs).astype(np.int8)
    report = {
        "parent_graph": graph.fingerprint,
        "topology": config.topology,
        "seed": config.seed,
        "sign_mode": mode,
        "annotated_sign_fraction": float(np.mean(graph.signs != 0)),
        "is_synthetic": bool(
            graph.manifest["provenance"].get("is_synthetic", False)
            or graph.manifest["provenance"].get("coverage") == "synthetic"
        ),
    }
    if config.topology in ("degree_rewired", "community_rewired"):
        src, dst, weight, details = _swaps(graph, signs, config)
        report.update(details)
    elif config.topology == "random":
        src, dst, weight, details = transform_graph(graph, "matched_random", signs, config.seed)
        report.update(details)
        report["topology"] = config.topology
    elif config.topology == "direction_shuffled":
        src, dst, weight = graph.src.copy(), graph.dst.copy(), graph.weight.copy()
        undirected = np.minimum(src, dst).astype(np.int64) * graph.n + np.maximum(src, dst)
        _, group, counts = np.unique(undirected, return_inverse=True, return_counts=True)
        rng = np.random.default_rng(seed_for(config.seed, "direction-shuffle"))
        # Reciprocal pairs already occupy both directions. Flipping only single
        # edges preserves the entire undirected graph and cannot create duplicates.
        flip = (counts[group] == 1) & (rng.random(graph.m) < 0.5)
        src[flip], dst[flip] = dst[flip], src[flip]
        report.update(
            reversed_edges=int(flip.sum()),
            preserves=["N", "M", "undirected_graph", "reciprocal_pairs", "global_weights"],
        )
    else:
        src, dst, weight = graph.src.copy(), graph.dst.copy(), graph.weight.copy()
    order = np.lexsort((src, dst))
    src, dst, weight = src[order], dst[order], weight[order]
    if np.any(src == dst) or len(np.unique(src.astype(np.int64) * graph.n + dst)) != graph.m:
        raise AssertionError("Topology control changed edge count or introduced duplicate/autapse")
    return Topology(graph.node_ids.copy(), src, dst, weight, signs, report)


def initialize_weights(topology: Topology, config: SubstrateConfig):
    """Alter permitted-edge initialization without altering topology or port features."""
    rng = np.random.default_rng(seed_for(config.seed, "weight-initialization"))
    weight = topology.weight.astype(np.float64).copy()
    if config.dynamics.weight_transform == "log1p":
        weight = np.log1p(weight)
    signs = topology.signs.copy()
    if config.initialization == "shuffled_magnitudes":
        weight = rng.permutation(weight)
    elif config.initialization in ("random_magnitudes", "random_signed"):
        logs = np.log(weight)
        weight = rng.lognormal(float(logs.mean()), max(float(logs.std()), 0.1), size=topology.m)
    if config.initialization == "random_signed":
        signs = np.ones(topology.n, dtype=np.int8)
        inhibitory_count = int(np.count_nonzero(topology.signs < 0))
        signs[rng.permutation(topology.n)[:inhibitory_count]] = -1
    if not np.isfinite(weight).all() or np.any(weight <= 0):
        raise FloatingPointError("Nonfinite or nonpositive permitted-edge initialization")
    report = {
        "initialization": config.initialization,
        "transform": config.dynamics.weight_transform,
        "sign_mode": "random_dale"
        if config.initialization == "random_signed"
        else config.dynamics.sign_mode,
        "annotated_sign_fraction": float(topology.report["annotated_sign_fraction"]),
        "inhibitory_fraction": float(np.mean(signs < 0)),
        "random_magnitude_distribution": "lognormal matched to transformed log-weight mean/std",
        "random_sign_rule": "permuted source identities with the original source E/I count retained",
        "ports": "features fixed from condition topology before weight initialization",
    }
    return weight.astype(np.float32), signs, report


@njit(cache=True)
def _modularity_moves(indptr, indices, weights, degree, steps, seed, resolution):
    """Weighted Louvain local moves; self-loops stay with their supernode."""
    np.random.seed(seed)
    count = len(degree)
    labels = np.arange(count)
    volume = degree.copy()
    total = degree.sum()
    scores, touched = np.zeros(count), np.empty(count, np.int64)
    for iteration in range(steps):
        changes = 0
        for node in np.random.permutation(count):
            if degree[node] == 0:
                continue
            old, length = labels[node], 0
            for edge in range(indptr[node], indptr[node + 1]):
                neighbor = indices[edge]
                if neighbor == node:
                    continue
                label = labels[neighbor]
                if scores[label] == 0:
                    touched[length] = label
                    length += 1
                scores[label] += weights[edge]
            volume[old] -= degree[node]
            best = old
            best_score = scores[old] - resolution * degree[node] * volume[old] / total
            for index in range(length):
                label = touched[index]
                score = scores[label] - resolution * degree[node] * volume[label] / total
                if score > best_score + 1e-12:
                    best, best_score = label, score
            volume[best] += degree[node]
            labels[node] = best
            changes += best != old
            for index in range(length):
                scores[touched[index]] = 0.0
        if changes == 0:
            break
    return labels, iteration + 1


def _louvain(adjacency, iterations, seed, resolution=1.0, max_levels=20):
    membership = np.arange(adjacency.shape[0])
    levels = []
    for level in range(max_levels):
        degree = np.asarray(adjacency.sum(1)).ravel().astype(np.float64)
        labels, used = _modularity_moves(
            adjacency.indptr,
            adjacency.indices,
            adjacency.data,
            degree,
            iterations,
            seed_for(seed, f"louvain-level-{level}"),
            resolution,
        )
        unique, labels = np.unique(labels, return_inverse=True)
        membership = labels[membership]
        levels.append({"nodes": adjacency.shape[0], "communities": len(unique), "iterations": used})
        if len(unique) == adjacency.shape[0]:
            break
        edges = adjacency.tocoo(copy=False)
        adjacency = sparse.csr_matrix(
            (edges.data, (labels[edges.row], labels[edges.col])), shape=(len(unique), len(unique))
        )
        adjacency.sum_duplicates()
    return membership, levels


def write_communities(graph, output, seed=0, iterations=50):
    if iterations < 1 or seed < 0:
        raise ValueError("Invalid community discovery budget")
    if Path(output).exists():
        raise FileExistsError("Community partitions are immutable")
    adjacency = sparse.csr_matrix((graph.weight, (graph.src, graph.dst)), shape=(graph.n, graph.n))
    symmetric = (adjacency + adjacency.T).tocsr()
    labels, levels = _louvain(symmetric, iterations, seed)
    _, labels = np.unique(labels, return_inverse=True)
    degree = np.asarray(symmetric.sum(1)).ravel()
    internal = float(graph.weight[labels[graph.src] == labels[graph.dst]].sum(dtype=np.float64))
    volume = np.bincount(labels, weights=degree)
    modularity = internal / graph.weight.sum(dtype=np.float64) - float(
        np.square(volume / degree.sum()).sum()
    )
    active_communities = int(np.count_nonzero(volume))
    result = {
        "schema": "connectome-community-partition-v1",
        "graph_fingerprint": graph.fingerprint,
        "algorithm": "seeded multilevel weighted Louvain on A+A.T",
        "resolution": 1.0,
        "seed": seed,
        "iteration_budget": iterations,
        "iterations": sum(level["iterations"] for level in levels),
        "levels": levels,
        "communities": len(volume),
        "active_communities": active_communities,
        "isolated_neuron_communities": int(np.count_nonzero(volume == 0)),
        "largest_community_degree_fraction": float(volume.max() / degree.sum()),
        "modularity_undirected": modularity,
        "degenerate": active_communities < 2,
        "labels": labels.tolist(),
    }
    result["fingerprint"] = digest_json(result)
    atomic_json(output, result)
    return result
