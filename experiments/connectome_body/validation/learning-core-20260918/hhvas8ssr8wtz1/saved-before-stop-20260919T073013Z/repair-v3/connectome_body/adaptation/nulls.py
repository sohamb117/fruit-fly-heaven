"""Weight- and sign-restricted swaps preserve both incoming and outgoing strengths."""

import numpy as np
from numba import njit

from ..graphs import _edge_remove, _edge_slot, _edge_table
from ..util import seed_for


@njit(cache=True)
def _weighted_swaps(src, dst, order, starts, ends, n, attempts, seed):
    np.random.seed(seed)
    table = _edge_table(len(src))
    for i in range(len(src)):
        key = np.int64(src[i]) * n + np.int64(dst[i])
        table[_edge_slot(table, key)] = key
    accepted = 0
    for _ in range(attempts):
        position = np.random.randint(len(order))
        i = order[position]
        j = order[np.random.randint(starts[position], ends[position])]
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if a == c or b == d or a == d or c == b:
            continue
        new1, new2 = a * n + d, c * n + b
        if table[_edge_slot(table, new1)] != -1 or table[_edge_slot(table, new2)] != -1:
            continue
        _edge_remove(table, a * n + b)
        _edge_remove(table, c * n + d)
        table[_edge_slot(table, new1)] = new1
        table[_edge_slot(table, new2)] = new2
        dst[i], dst[j] = d, b
        accepted += 1
    return dst, accepted


def weighted_degree_null(graph, signs, seed, attempts_per_edge=10):
    # Exact float32 weight equality, not rounded bins. Pairing equal source signs
    # and weights preserves signed incoming degree and strength as well.
    order = np.lexsort((graph.weight, signs[graph.src]))
    sign, weight = signs[graph.src[order]], graph.weight[order]
    first = np.flatnonzero(np.r_[True, (sign[1:] != sign[:-1]) | (weight[1:] != weight[:-1])])
    last = np.r_[first[1:], graph.m]
    starts, ends = np.repeat(first, last - first), np.repeat(last, last - first)
    dst, accepted = _weighted_swaps(
        np.asarray(graph.src),
        np.array(graph.dst),
        order,
        starts,
        ends,
        graph.n,
        attempts_per_edge * graph.m,
        seed_for(seed, "weighted-degree-null"),
    )
    if accepted == 0:
        raise ValueError(
            "No equal-weight/sign swaps accepted; this graph cannot support the primary weighted null"
        )
    for weights in (None, graph.weight, signs[graph.src], graph.weight * signs[graph.src]):
        before = np.bincount(graph.dst, weights=weights, minlength=graph.n)
        after = np.bincount(dst, weights=weights, minlength=graph.n)
        if not np.allclose(before, after, rtol=1e-12, atol=1e-9):
            raise AssertionError("Weighted null violated target degree/strength")
    return (
        graph.src,
        dst,
        graph.weight,
        {
            "variant": "degree_shuffled",
            "algorithm": "equal-weight/equal-source-sign directed swaps",
            "parent": graph.fingerprint,
            "seed": seed,
            "attempts": attempts_per_edge * graph.m,
            "accepted_swaps": int(accepted),
            "changed_target_fraction": float(np.mean(dst != graph.dst)),
            "preserves": [
                "N",
                "M",
                "in_degree",
                "out_degree",
                "signed_in_degree",
                "in_strength",
                "out_strength",
                "signed_in_strength",
                "per_source_weight_multiset",
                "global_weights",
            ],
            "limitation": "Finite restricted swap chain; no claim of uniform sampling or full mixing",
        },
    )
