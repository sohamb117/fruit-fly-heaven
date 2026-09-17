import json

import numpy as np
import pytest

from connectome_body.compatibility.adapters import AnatomicalPorts
from connectome_body.compatibility.anatomy import prepare_anatomy, prepare_task_subgraphs
from connectome_body.graphs import Graph, save_graph
from connectome_body.util import atomic_json


@pytest.fixture
def annotated_graph(tmp_path):
    names = [f"SENSE{i}{side}" for i in range(6) for side in ("L", "R")]
    names += [f"DA{i:02d}" for i in range(1, 7)] + [
        f"SMD{i}{side}" for i in range(2) for side in ("L", "R")
    ]
    names += [f"INTER{i}" for i in range(44)]
    rng = np.random.default_rng(4)
    a, b = np.where((rng.random((len(names), len(names))) < 0.3) & ~np.eye(len(names), dtype=bool))
    graph = save_graph(
        tmp_path / "graph",
        np.array(names),
        a,
        b,
        np.ones(len(a)),
        provenance={"is_synthetic": True},
    )
    path = tmp_path / "cell-types.json"
    atomic_json(
        path,
        {
            "graph_fingerprint": graph.fingerprint,
            "source_sha256": "fixture",
            "cell_types": {
                name: "sensory"
                if name.startswith("SENSE")
                else "motorneuron"
                if name.startswith(("DA", "SMD"))
                else "interneuron"
                for name in names
            },
        },
    )
    return graph, path


def test_oracle_uses_only_pinned_sensory_motor_annotations(annotated_graph, tmp_path):
    graph, path = annotated_graph
    result = prepare_anatomy(graph, "celegans", path, tmp_path / "annotations", channels=(4,))
    ports = AnatomicalPorts(graph, result["ports"][0], 4)
    sensory = set(json.loads(path.read_text())["cell_types"])
    anatomy = json.loads((tmp_path / "annotations/anatomy.json").read_text())
    assert len(anatomy["sensory_ids"]) == 12 and len(anatomy["motor_ids"]) == 10
    assert set(anatomy["sensory_ids"]) <= sensory
    assert len(anatomy["bilateral_pairs"]) == 8
    assert not set(ports.input_nodes[ports.input_weights > 0].tolist()) & set(
        ports.output_nodes[ports.output_weights > 0].tolist()
    )
    assert sum(p.numel() for p in ports.parameters()) == 0
    assert (
        prepare_anatomy(graph, "celegans", path, tmp_path / "annotations", channels=(4,)) == result
    )
    value = json.loads(path.read_text())
    value["graph_fingerprint"] = "different"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="another graph"):
        prepare_anatomy(graph, "celegans", path, tmp_path / "other", channels=(4,))


def test_functional_hypotheses_keep_exact_equal_size_disjoint_controls(annotated_graph, tmp_path):
    graph, path = annotated_graph
    output = tmp_path / "modules"
    records = prepare_task_subgraphs(graph, "celegans", path, output, max_neurons=18, seed=7)
    assert len(records) == 10 and all(row["status"] == "prepared" for row in records.values())
    for task, row in records.items():
        root = output / task.replace(":", "-")
        real = Graph.load(root / "relevant")
        random = Graph.load(root / "random_selection")
        assert real.n == random.n == 18
        assert not set(real.node_ids) & set(random.node_ids)
        p = real.manifest["provenance"]["selection"]["provenance"]
        assert p["cross_species_homology_claim"] is False
        assert "hypothesis" in p["relevance_status"]
    assert (
        prepare_task_subgraphs(graph, "celegans", path, output, max_neurons=18, seed=7) == records
    )
    with pytest.raises(ValueError, match="null seed"):
        prepare_task_subgraphs(graph, "celegans", path, output, max_neurons=18, seed=8)
