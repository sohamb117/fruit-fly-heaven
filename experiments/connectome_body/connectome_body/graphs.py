"""Directed sparse graphs, lossless IDs, and auditable topology nulls.

An edge is always src -> dst. Raw positive weights are retained separately from
the fixed dynamical normalization. No anatomical labels are used to make ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .util import atomic_json, digest_file, digest_json, seed_for


@dataclass
class Graph:
    node_ids: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    signs: np.ndarray
    manifest: dict

    @property
    def n(self):
        return len(self.node_ids)

    @property
    def m(self):
        return len(self.src)

    @property
    def fingerprint(self):
        return self.manifest["fingerprint"]

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> "Graph":
        import json

        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema") != "connectome-body-graph-v1":
            raise ValueError("Unsupported graph schema")
        fingerprint = manifest.pop("fingerprint")
        if digest_json(manifest) != fingerprint:
            raise ValueError("Graph manifest fingerprint mismatch")
        manifest["fingerprint"] = fingerprint
        arrays = {}
        for name, expected in manifest["files"].items():
            if verify and digest_file(path / name) != expected:
                raise ValueError(f"Graph file changed: {name}")
            arrays[name.removesuffix(".npy")] = np.load(
                path / name, mmap_mode="r", allow_pickle=False
            )
        return cls(**arrays, manifest=manifest)


def save_graph(
    path: str | Path, node_ids, src, dst, weight, *, signs=None, provenance: dict
) -> Graph:
    path = Path(path)
    if (path / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite a prepared graph: {path}")
    node_ids = np.asarray(node_ids)
    if node_ids.dtype.kind == "f":
        raise ValueError("Neuron IDs must never pass through floating point")
    node_ids = node_ids.astype(str)
    n = len(node_ids)
    if n < 4 or len(np.unique(node_ids)) != n:
        raise ValueError("Need at least four distinct neurons")
    src, dst = np.asarray(src), np.asarray(dst)
    weight = np.asarray(weight, dtype=np.float32)
    if src.dtype.kind not in "iu" or dst.dtype.kind not in "iu":
        raise ValueError("Edge indices must be integers")
    if not (src.shape == dst.shape == weight.shape) or src.ndim != 1 or not len(src):
        raise ValueError("Invalid or empty edge arrays")
    if min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n:
        raise ValueError("Edge endpoint outside node table")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Raw connection strengths must be positive and finite")
    # The same explicit no-autapse convention is applied to every dataset.
    keep = src != dst
    self_edges = int((~keep).sum())
    src, dst, weight = src[keep], dst[keep], weight[keep]
    if not len(src):
        raise ValueError("No non-autapse connections remain")
    order = np.lexsort((src, dst))
    src, dst, weight = src[order], dst[order], weight[order]
    starts = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
    indices = np.flatnonzero(starts)
    duplicate_edges = len(src) - len(indices)
    weight = np.add.reduceat(weight, indices)
    src, dst = src[indices].astype(np.uint32), dst[indices].astype(np.uint32)
    if not len(src) or not np.isfinite(weight).all():
        raise ValueError("No finite non-autapse connections remain")
    signs = np.zeros(n, dtype=np.int8) if signs is None else np.asarray(signs)
    if signs.shape != (n,) or not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError("Signs must be -1, 0 (unknown), or +1 per neuron")
    signs = signs.astype(np.int8)
    arrays = dict(node_ids=node_ids, src=src, dst=dst, weight=weight, signs=signs)
    path.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(path / f"{name}.npy", array, allow_pickle=False)
    summary = graph_statistics(n, src, dst, weight)
    summary.update(
        removed_self_edges=self_edges,
        merged_duplicate_edges=duplicate_edges,
        annotated_sign_fraction=float(np.mean(signs != 0)),
    )
    manifest = {
        "schema": "connectome-body-graph-v1",
        "provenance": provenance,
        "summary": summary,
        "files": {f"{name}.npy": digest_file(path / f"{name}.npy") for name in arrays},
    }
    manifest["fingerprint"] = digest_json(manifest)
    atomic_json(path / "manifest.json", manifest)
    return Graph.load(path)


def graph_statistics(n, src, dst, weight):
    indegree = np.bincount(dst, minlength=n)
    outdegree = np.bincount(src, minlength=n)
    matrix = sparse.csr_matrix((np.ones(len(src), dtype=np.int8), (dst, src)), shape=(n, n))
    weak, _ = connected_components(matrix, directed=True, connection="weak")
    strong, _ = connected_components(matrix, directed=True, connection="strong")
    return {
        "neurons": int(n),
        "edges": int(len(src)),
        "raw_weight_sum": float(weight.sum(dtype=np.float64)),
        "isolated_neurons": int(((indegree + outdegree) == 0).sum()),
        "weak_components": int(weak),
        "strong_components": int(strong),
        "in_degree_quantiles": np.quantile(indegree, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "out_degree_quantiles": np.quantile(outdegree, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def neuron_signs(graph: Graph, mode: str, fraction: float, seed: int):
    rng = np.random.default_rng(seed_for(seed, "signs"))
    # Exact population fraction; use the SAME assignments in paired nulls.
    signs = np.ones(graph.n, dtype=np.int8)
    signs[rng.permutation(graph.n)[: round(fraction * graph.n)]] = -1
    if mode == "annotated":
        if not np.any(graph.signs):
            raise ValueError("Annotated-sign condition requires actual sign annotations")
        signs = np.where(graph.signs != 0, graph.signs, signs).astype(np.int8)
    return signs


@njit(cache=True, inline="always")
def _edge_slot(table, key):
    # Mix all 64 bits: neuron-pair integers have structured low bits. A fixed
    # array avoids the allocation/reference-count overhead of a typed set.
    value = np.uint64(key)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    mask = len(table) - 1
    slot = np.int64(value & np.uint64(mask))
    while table[slot] != -1 and table[slot] != key:
        slot = (slot + 1) & mask
    return slot


@njit(cache=True, inline="always")
def _edge_remove(table, key):
    slot = _edge_slot(table, key)
    if table[slot] != key:
        raise RuntimeError("Topology null lost an existing edge")
    table[slot] = -1
    # Reinsert the following probe cluster. Avoid accumulating tombstones over
    # hundreds of millions of swaps; occupancy stays below 25 percent.
    slot = (slot + 1) & (len(table) - 1)
    while table[slot] != -1:
        displaced = table[slot]
        table[slot] = -1
        table[_edge_slot(table, displaced)] = displaced
        slot = (slot + 1) & (len(table) - 1)


@njit(cache=True)
def _edge_table(m):
    size = 8
    while size < 4 * m:
        size *= 2
    return np.full(size, -1, dtype=np.int64)


@njit(cache=True)
def _swap_edges(src, dst, signs, n, attempts, seed):
    """Directed simple-graph double swaps, keeping weights with their source.

    Same-sign sources preserve positive/negative incoming degree separately.
    Invalid moves (autapses, parallel edges, identical endpoints) are rejected.
    The result is a finite swap-chain sample, NOT a claim of uniform mixing.
    """
    np.random.seed(seed)
    occupied = _edge_table(len(src))
    for i in range(len(src)):
        key = np.int64(src[i]) * n + np.int64(dst[i])
        occupied[_edge_slot(occupied, key)] = key
    accepted = 0
    for _ in range(attempts):
        i = np.random.randint(len(src))
        j = np.random.randint(len(src))
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if a == c or b == d or a == d or c == b or signs[a] != signs[c]:
            continue
        new1, new2 = a * n + d, c * n + b
        if occupied[_edge_slot(occupied, new1)] != -1 or occupied[_edge_slot(occupied, new2)] != -1:
            continue
        _edge_remove(occupied, a * n + b)
        _edge_remove(occupied, c * n + d)
        occupied[_edge_slot(occupied, new1)] = new1
        occupied[_edge_slot(occupied, new2)] = new2
        dst[i], dst[j] = d, b
        accepted += 1
    return dst, accepted


@njit(cache=True)
def _random_edges(n, m, seed):
    np.random.seed(seed)
    if m > n * (n - 1):
        raise ValueError("Too many edges for a directed simple graph")
    occupied = _edge_table(m)
    src, dst = np.empty(m, np.uint32), np.empty(m, np.uint32)
    i = 0
    while i < m:
        a, b = np.random.randint(n), np.random.randint(n)
        key = a * n + b
        slot = _edge_slot(occupied, key)
        if a != b and occupied[slot] == -1:
            occupied[slot] = key
            src[i], dst[i] = a, b
            i += 1
    return src, dst


def transform_graph(
    graph: Graph, variant: str, signs: np.ndarray, seed: int, swaps_per_edge: int = 10
):
    """Return edges and metadata. Nulls never change node ordering or port seeds."""
    src, dst, weight = graph.src, graph.dst, graph.weight
    report = {"variant": variant, "seed": seed, "parent": graph.fingerprint}
    if variant == "degree_shuffled":
        dst, accepted = _swap_edges(
            np.asarray(src),
            np.array(dst),
            signs,
            graph.n,
            graph.m * swaps_per_edge,
            seed_for(seed, "swaps"),
        )
        report.update(
            attempts=graph.m * swaps_per_edge,
            accepted_swaps=int(accepted),
            changed_target_fraction=float(np.mean(dst != graph.dst)),
            preserves=[
                "N",
                "M",
                "in_degree",
                "out_degree",
                "signed_in_degree",
                "per_source_weight_multiset",
                "global_weight_multiset",
            ],
            does_not_preserve=["per_target_weighted_strength", "motifs", "communities"],
        )
        if accepted == 0:
            raise ValueError("No valid topology swaps; this graph cannot support this null")
        if not np.array_equal(
            np.bincount(dst, minlength=graph.n), np.bincount(graph.dst, minlength=graph.n)
        ):
            raise RuntimeError("Null violated in-degree invariance")
    elif variant == "matched_random":
        active = np.union1d(src, dst)
        src, dst = _random_edges(len(active), graph.m, seed_for(seed, "random_graph"))
        src, dst = active[src], active[dst]
        weight = np.asarray(weight)[
            np.random.default_rng(seed_for(seed, "weights")).permutation(graph.m)
        ]
        report.update(
            preserves=["N", "M", "global_weight_multiset", "node_signs", "original_isolates"],
            does_not_preserve=["degree_sequence", "weighted_strengths", "motifs"],
        )
    elif variant == "no_edges":
        src, dst, weight = src[:0], dst[:0], weight[:0]
        report["interpretation"] = "Disconnected input/output ports; only decoder bias can act"
    elif variant != "real":
        raise ValueError(f"Unknown graph transform {variant}")
    return src, dst, weight, report


def normalized_matrix(n, src, dst, raw_weight, signs, weight_transform="log1p"):
    weight = np.asarray(raw_weight, dtype=np.float64)
    if weight_transform == "log1p":
        weight = np.log1p(weight)
    incoming = np.bincount(dst, weights=weight, minlength=n)
    outgoing = np.bincount(src, weights=weight, minlength=n)
    denom = np.sqrt(incoming[dst] * outgoing[src])
    values = (weight * signs[src] / np.maximum(denom, np.finfo(float).tiny)).astype(np.float32)
    # ||D_in^-1/2 A^T D_out^-1/2 S||_2 <= 1. S has entries +/-1.
    # Consequently gain < 1 gives a topology-independent contraction bound.
    return sparse.csr_matrix((values, (dst, src)), shape=(n, n), dtype=np.float32)


def matched_subgraph(graph: Graph, path: Path, neurons: int, edges: int, seed: int):
    """Uniform node/edge subsampling for an explicit size/density sensitivity.

    Only existing edges are retained; this is not a reconstructed complete
    circuit. Reject an infeasible draw instead of selecting a luckier seed.
    """
    if not 4 <= neurons <= graph.n or not 1 <= edges <= graph.m:
        raise ValueError("Requested size must fit inside the source graph")
    rng = np.random.default_rng(seed_for(seed, "size-matched-subgraph"))
    nodes = np.sort(rng.choice(graph.n, neurons, replace=False))
    lookup = np.full(graph.n, -1, dtype=np.int64)
    lookup[nodes] = np.arange(neurons)
    eligible = np.flatnonzero((lookup[graph.src] >= 0) & (lookup[graph.dst] >= 0))
    if len(eligible) < edges:
        raise ValueError(
            f"This node draw retains {len(eligible)} edges, fewer than {edges}; "
            "lower the common edge target for every dataset, not just this graph"
        )
    chosen = rng.choice(eligible, edges, replace=False)
    provenance = {
        **graph.manifest["provenance"],
        "coverage": f"Uniform size-control subset of {graph.manifest['provenance'].get('coverage', 'source')}",
        "size_control": {
            "parent": graph.fingerprint,
            "seed": seed,
            "neurons": neurons,
            "edges": edges,
            "induced_edges_before_thinning": len(eligible),
            "method": "Uniform nodes, then uniform existing induced edges; no fabricated edges",
        },
    }
    return save_graph(
        path,
        graph.node_ids[nodes],
        lookup[graph.src[chosen]],
        lookup[graph.dst[chosen]],
        graph.weight[chosen],
        signs=graph.signs[nodes],
        provenance=provenance,
    )


def make_fixture(path: str | Path, n: int = 64, seed: int = 0) -> Graph:
    if n < 8:
        raise ValueError("Fixture requires at least 8 neurons")
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n, dtype=np.uint32), 4)
    dst = ((src.reshape(n, 4) + np.array([1, 2, 5, 7])) % n).ravel().astype(np.uint32)
    return save_graph(
        path,
        np.arange(n, dtype=np.uint64) + 2**63,
        src,
        dst,
        rng.integers(1, 8, len(src)),
        provenance={
            "dataset": "synthetic_fixture",
            "species": "none",
            "coverage": "synthetic",
            "release": "v1",
            "seed": seed,
            "evidence": "software validation only",
            "is_synthetic": True,
        },
    )
