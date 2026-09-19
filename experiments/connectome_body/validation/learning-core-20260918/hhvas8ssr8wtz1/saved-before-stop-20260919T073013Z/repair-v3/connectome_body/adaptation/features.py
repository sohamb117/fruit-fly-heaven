"""Deterministic O(N+M) structural descriptors; no anatomy or neuron identities."""

from __future__ import annotations

import numpy as np
from scipy import sparse

FEATURE_NAMES = (
    "log_in_degree",
    "log_out_degree",
    "log_in_strength",
    "log_out_strength",
    "reciprocal_out_fraction",
    "log_pagerank",
    "in_neighbor_log_in_degree",
    "in_neighbor_log_out_degree",
    "out_neighbor_log_in_degree",
    "out_neighbor_log_out_degree",
    "two_step_return_probability",
    "isolated",
)


def structural_features(n, src, dst, weight, *, pagerank_steps=100):
    adjacency = sparse.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    indegree = np.asarray(adjacency.sum(axis=0)).ravel()
    outdegree = np.asarray(adjacency.sum(axis=1)).ravel()
    in_strength = np.bincount(dst, weights=weight, minlength=n)
    out_strength = np.bincount(src, weights=weight, minlength=n)
    inv_in, inv_out = 1 / np.maximum(indegree, 1), 1 / np.maximum(outdegree, 1)
    transition = sparse.diags(inv_out) @ adjacency
    rank = np.full(n, 1 / n)
    residual = 0.0
    for _ in range(pagerank_steps):
        updated = 0.15 / n + 0.85 * (transition.T @ rank + rank[outdegree == 0].sum() / n)
        residual = float(np.abs(updated - rank).sum())
        rank = updated
    reciprocal = np.asarray(adjacency.multiply(adjacency.T).sum(axis=1)).ravel() * inv_out
    return_probability = np.asarray(transition.multiply(transition.T).sum(axis=1)).ravel()
    log_in, log_out = np.log1p(indegree), np.log1p(outdegree)
    values = np.column_stack(
        (
            log_in,
            log_out,
            np.log1p(in_strength),
            np.log1p(out_strength),
            reciprocal,
            np.log(np.maximum(rank, np.finfo(float).tiny)),
            adjacency.T @ log_in * inv_in,
            adjacency.T @ log_out * inv_in,
            transition @ log_in,
            transition @ log_out,
            return_probability,
            ((indegree + outdegree) == 0).astype(float),
        )
    )
    mean, scale = values.mean(axis=0), values.std(axis=0)
    normalized = np.clip((values - mean) / np.maximum(scale, 1e-8), -5, 5).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Nonfinite structural features")
    return normalized, {
        "names": FEATURE_NAMES,
        "normalization": "within-graph z-score, clipped [-5,5]",
        "mean": mean.tolist(),
        "std": scale.tolist(),
        "pagerank_steps": pagerank_steps,
        "pagerank_l1_residual": residual,
        "topology_source": "actual condition graph, after rewiring/randomization",
    }
