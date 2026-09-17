"""Audit raw Fish1 endpoint coverage independently of sparse edge aggregation."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from connectome_body.compatibility.fish1 import _read_page
from connectome_body.graphs import Graph
from connectome_body.util import atomic_json, digest_file, digest_json


def contains(sorted_ids, values):
    index = np.searchsorted(sorted_ids, values)
    return sorted_ids[np.minimum(index, len(sorted_ids) - 1)] == values


def main():
    directory = Path("data/raw/fish1-versioned-export")
    provenance = json.loads((directory / "provenance.json").read_text())
    graph = Graph.load("data/graphs/fish1")
    selected = np.asarray(graph.node_ids, dtype=np.uint64)
    assert np.all(selected[1:] > selected[:-1])
    somas = []
    for name, item in provenance["files"].items():
        if item["table"] == "somas":
            assert digest_file(directory / name) == item["sha256"]
            somas.extend(_read_page(directory / name))
    all_soma_roots = np.unique(np.array([row["pt_root_id"] for row in somas], dtype=np.uint64))
    all_soma_roots = all_soma_roots[all_soma_roots != 0]
    degree = np.bincount(graph.src, minlength=graph.n) + np.bincount(graph.dst, minlength=graph.n)
    samples = [int(selected[i]) for i in np.argsort(degree)[-3:]]
    samples += [int(row["pt_root_id"]) for row in somas if row["id"] == "173502"]
    samples = sorted(set(samples) - {0})
    root_counts = {root: {"raw_incoming": 0, "raw_outgoing": 0} for root in samples}
    counts = Counter()
    for name, item in provenance["files"].items():
        if item["table"] != "synapses_axde":
            continue
        path = directory / name
        assert digest_file(path) == item["sha256"]
        table = pq.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id"])
        pre = table["pre_pt_root_id"].to_numpy()
        post = table["post_pt_root_id"].to_numpy()
        assert pre.dtype == np.uint64 and post.dtype == np.uint64
        pre_selected, post_selected = contains(selected, pre), contains(selected, post)
        pre_soma, post_soma = contains(all_soma_roots, pre), contains(all_soma_roots, post)
        counts["raw_synapses"] += len(pre)
        masks = {
            "zero_pre_root": pre == 0,
            "zero_post_root": post == 0,
            "either_zero_root": (pre == 0) | (post == 0),
            "pre_has_any_soma": pre_soma,
            "post_has_any_soma": post_soma,
            "both_have_any_soma": pre_soma & post_soma,
            "neither_has_soma": ~pre_soma & ~post_soma,
            "pre_in_selected_population": pre_selected,
            "post_in_selected_population": post_selected,
            "both_selected_including_autapse": pre_selected & post_selected,
            "retained_nonself_synapses": pre_selected & post_selected & (pre != post),
        }
        for key, mask in masks.items():
            counts[key] += int(mask.sum())
        for root in samples:
            root_counts[root]["raw_incoming"] += int(np.count_nonzero(post == root))
            root_counts[root]["raw_outgoing"] += int(np.count_nonzero(pre == root))
    assert counts["retained_nonself_synapses"] == graph.manifest["summary"]["raw_weight_sum"]
    result = {
        "schema": "fish1-endpoint-membership-audit-v1",
        "materialization_version": provenance["materialization_version"],
        "graph_fingerprint": graph.fingerprint,
        "export_provenance_sha256": digest_file(directory / "provenance.json"),
        "counts": dict(counts),
        "retained_synapse_fraction": counts["retained_nonself_synapses"] / counts["raw_synapses"],
        "isolated_neuron_fraction": graph.manifest["summary"]["isolated_neurons"] / graph.n,
        "source_query_samples": [{"root_id": str(root), **root_counts[root]} for root in samples],
    }
    result["fingerprint"] = digest_json(result)
    atomic_json("data/raw/fish1-endpoint-audit-v700.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
