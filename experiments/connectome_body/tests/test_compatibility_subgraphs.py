import json

import numpy as np
import pytest

from connectome_body.compatibility.subgraphs import (
    induced_subgraph,
    prepare_subgraph_controls,
    write_selection,
)
from connectome_body.graphs import make_fixture


def test_subgraphs_only_retain_original_edges_and_random_controls_do_not_claim_irrelevance(
    tmp_path,
):
    graph = make_fixture(tmp_path / "parent", 64, 3)
    selection = tmp_path / "selection.json"
    write_selection(
        graph,
        graph.node_ids[:24].tolist(),
        selection,
        role="relevant",
        task="fly:hover",
        provenance="software test annotation",
    )
    subgraph = induced_subgraph(graph, selection, tmp_path / "induced")
    keep = (graph.src < 24) & (graph.dst < 24)
    assert subgraph.n == 24
    np.testing.assert_array_equal(subgraph.src, graph.src[keep])
    np.testing.assert_array_equal(subgraph.dst, graph.dst[keep])
    np.testing.assert_array_equal(subgraph.weight, graph.weight[keep])
    result = prepare_subgraph_controls(graph, selection, tmp_path / "controls", seed=5)
    random = json.loads((tmp_path / "controls/random_selection-selection.json").read_text())
    assert set(random["node_ids"]).isdisjoint(graph.node_ids[:24])
    assert len(random["node_ids"]) == 24
    assert random["provenance"]["not_an_anatomical_irrelevance_claim"]
    assert "irrelevant" in result["missing_or_infeasible"]
    bad = tmp_path / "irrelevant.json"
    write_selection(
        graph,
        graph.node_ids[24:48].tolist(),
        bad,
        role="irrelevant",
        task="fly:walking",
        provenance="different task",
    )
    with pytest.raises(ValueError, match="same task"):
        prepare_subgraph_controls(graph, selection, tmp_path / "invalid", irrelevant_file=bad)
