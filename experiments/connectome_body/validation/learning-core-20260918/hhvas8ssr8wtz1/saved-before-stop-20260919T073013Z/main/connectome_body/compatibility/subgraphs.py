"""Auditable anatomical selections and equal-size comparison graphs for experiment 7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..graphs import save_graph
from ..util import atomic_json, digest_file, digest_json, seed_for


def read_selection(graph, path):
    value = json.loads(Path(path).read_text())
    if value.get("graph_fingerprint") != graph.fingerprint or not value.get("provenance"):
        raise ValueError(
            "Neuronal selection requires parent graph identity and annotation provenance"
        )
    raw = value.get("node_ids", [])
    if any(isinstance(node, (float, bool)) for node in raw):
        raise ValueError("Neuron IDs must not pass through floats or booleans")
    ids = list(map(str, raw))
    lookup = {str(node): index for index, node in enumerate(graph.node_ids)}
    if not ids or len(set(ids)) != len(ids) or any(node not in lookup for node in ids):
        raise ValueError("Selection contains duplicate, missing or unknown neurons")
    return value, np.array(sorted(lookup[node] for node in ids), dtype=np.int64)


def write_selection(graph, node_ids, output, *, role, task, provenance):
    if role not in ("relevant", "irrelevant", "random_selection") or not task or not provenance:
        raise ValueError("Declare anatomical role, task and evidence before selection")
    path = Path(output)
    if path.exists():
        raise FileExistsError("Anatomical selections are immutable")
    value = {
        "schema": "connectome-neuron-selection-v1",
        "graph_fingerprint": graph.fingerprint,
        "node_ids": list(node_ids),
        "role": role,
        "task": task,
        "provenance": provenance,
    }
    lookup = set(map(str, graph.node_ids))
    if any(isinstance(x, (float, bool)) for x in value["node_ids"]):
        raise ValueError("Neuron identities must remain lossless")
    ids = list(map(str, value["node_ids"]))
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= lookup:
        raise ValueError("Unknown or duplicated anatomical identities")
    value["node_ids"] = ids
    value["fingerprint"] = digest_json(value)
    atomic_json(path, value)
    return value


def induced_subgraph(graph, selection_file, output):
    selection, indices = read_selection(graph, selection_file)
    if len(indices) < 4:
        raise ValueError("A recurrent subgraph requires at least four selected neurons")
    remap = np.full(graph.n, -1, dtype=np.int64)
    remap[indices] = np.arange(len(indices))
    src, dst = remap[graph.src], remap[graph.dst]
    keep = (src >= 0) & (dst >= 0)
    if not keep.any():
        raise ValueError("No induced edges; do not substitute a denser artificial graph")
    return save_graph(
        output,
        graph.node_ids[indices],
        src[keep],
        dst[keep],
        graph.weight[keep],
        signs=graph.signs[indices],
        provenance={
            "dataset": graph.manifest["provenance"].get("dataset"),
            "parent_graph": graph.fingerprint,
            "selection_sha256": digest_file(selection_file),
            "selection": selection,
            "selection_role": selection.get("role"),
            "task": selection.get("task"),
            "operation": "induced directed graph; all selected isolates retained; no added synapses",
            "parent_neurons": graph.n,
            "parent_edges": graph.m,
            "boundary_edges_removed": int(np.count_nonzero((src >= 0) ^ (dst >= 0))),
            "weight_semantics": graph.manifest["provenance"].get("weight_semantics"),
            "is_synthetic": graph.manifest["provenance"].get("is_synthetic", False),
        },
    )


def prepare_subgraph_controls(graph, relevant_file, output, *, irrelevant_file=None, seed=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Subgraph controls are an immutable preregistered selection set")
    relevant, indices = read_selection(graph, relevant_file)
    if relevant.get("role") != "relevant" or not relevant.get("task"):
        raise ValueError("The reference must explicitly declare task relevance")
    complement = np.setdiff1d(np.arange(graph.n), indices)
    if len(complement) < len(indices):
        raise ValueError("Not enough unselected neurons for a disjoint equal-size random control")
    irrelevant = None
    if irrelevant_file:
        irrelevant, other = read_selection(graph, irrelevant_file)
        if irrelevant.get("role") != "irrelevant" or irrelevant.get("task") != relevant["task"]:
            raise ValueError("Irrelevance must have explicit evidence for the same task")
        if len(other) != len(indices) or np.intersect1d(other, indices).size:
            raise ValueError(
                "Irrelevant control must be equal-size and disjoint from the relevant selection"
            )
    rng = np.random.default_rng(seed_for(seed, "equal-size-subgraph-control"))
    random_indices = np.sort(rng.choice(complement, len(indices), replace=False))
    output.mkdir(parents=True)
    atomic_json(output / "relevant-selection.json", relevant)
    write_selection(
        graph,
        graph.node_ids[random_indices].tolist(),
        output / "random_selection-selection.json",
        role="random_selection",
        task=relevant["task"],
        provenance={
            "rule": "uniform subset of the complement, without replacement",
            "seed": seed,
            "reference_selection_sha256": digest_file(relevant_file),
            "not_an_anatomical_irrelevance_claim": True,
        },
    )
    if irrelevant is not None:
        atomic_json(output / "irrelevant-selection.json", irrelevant)
    graphs, failures = {}, {}
    for role in ("relevant", "random_selection", "irrelevant"):
        path = output / f"{role}-selection.json"
        if not path.exists():
            failures[role] = "No explicit anatomical irrelevance annotation supplied"
            continue
        try:
            subgraph = induced_subgraph(graph, path, output / role)
            graphs[role] = {
                "fingerprint": subgraph.fingerprint,
                "neurons": subgraph.n,
                "edges": subgraph.m,
            }
        except ValueError as exc:
            failures[role] = str(exc)
    result = {
        "schema": "task-subgraph-controls-v1",
        "parent_graph": graph.fingerprint,
        "task": relevant["task"],
        "seed": seed,
        "graphs": graphs,
        "missing_or_infeasible": failures,
        "matching": "neuron count; induced edge counts can differ and are reported",
        "randomized_relevant": "Use topology=random on the relevant graph to match both neuron and edge counts",
    }
    atomic_json(output / "controls.json", result)
    return result
