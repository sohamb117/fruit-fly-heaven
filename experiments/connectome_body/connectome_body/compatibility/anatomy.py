"""Pinned anatomical oracles and explicit functional hypotheses for subgraphs.

These annotations are privileged information. Generic controller ports never
load them. Cross-species task selections are functional hypotheses, not claims
of neuron homology or established task relevance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..util import atomic_json, digest_file, digest_json
from .bodies import BODY_TASKS
from .subgraphs import prepare_subgraph_controls, write_selection


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(
                f"Keep the pinned anatomical artifact and choose a new path: {path}"
            )
    else:
        atomic_json(path, value)


def _read_annotations(graph, kind, source):
    source = Path(source)
    checksum = digest_file(source)
    if kind == "celegans":
        value = json.loads(source.read_text())
        if value.get("graph_fingerprint") != graph.fingerprint:
            raise ValueError("Cook cell annotations belong to another graph")
        raw = [
            {
                "id": node,
                "class": cell_type,
                "type": node[:-1] if node.endswith(("L", "R")) else node,
                "side": node[-1] if node.endswith(("L", "R")) else "",
                "region": "",
                "group": cell_type,
                "function": "",
            }
            for node, cell_type in value["cell_types"].items()
        ]
        sensory_class, motor_class = {"sensory"}, {"motorneuron"}
        evidence = {
            "kind": "Cook corrected cell-list classifications",
            "workbook_sha256": value["source_sha256"],
        }
    elif kind in ("banc", "malecns"):
        import pyarrow.feather as feather

        filename = "meta.feather" if kind == "banc" else "annotations.feather"
        expected = graph.manifest["provenance"].get("files", {}).get(filename, {}).get("sha256")
        if not expected or expected != checksum:
            raise ValueError("Anatomy source differs from the graph's pinned release metadata")
        if kind == "banc":
            columns = [
                "banc_888_id",
                "super_class",
                "cell_type",
                "side",
                "neuromere",
                "body_part_sensory",
                "body_part_effector",
                "cell_function",
            ]
            raw = []
            for row in feather.read_table(source, columns=columns).to_pylist():
                raw.append(
                    {
                        "id": str(row["banc_888_id"]),
                        "class": row["super_class"] or "",
                        "type": row["cell_type"] or "",
                        "side": {"left": "L", "right": "R"}.get(row["side"], ""),
                        "region": row["neuromere"] or "",
                        "group": row["body_part_effector"] or row["body_part_sensory"] or "",
                        "function": row["cell_function"] or "",
                    }
                )
            sensory_class = {"sensory", "sensory_ascending", "sensory_descending"}
            motor_class = {"motor"}
        else:
            columns = [
                "bodyId",
                "superclass",
                "type",
                "somaSide",
                "somaNeuromere",
                "subclass",
                "class",
            ]
            raw = [
                {
                    "id": str(row["bodyId"]),
                    "class": row["superclass"] or "",
                    "type": row["type"] or "",
                    "side": row["somaSide"] or "",
                    "region": row["somaNeuromere"] or "",
                    "group": row["subclass"] or "",
                    "function": row["class"] or "",
                }
                for row in feather.read_table(source, columns=columns).to_pylist()
            ]
            sensory_class = {"ol_sensory", "cb_sensory", "vnc_sensory"}
            motor_class = {"vnc_motor", "cb_motor"}
        evidence = {
            "kind": "published cell class, type, side and peripheral annotations",
            "source_file": filename,
            "source_url": graph.manifest["provenance"]["files"][filename]["url"],
        }
    else:
        raise ValueError(
            "Automatic anatomy import is defined only for BANC, MaleCNS and Cook C. elegans"
        )
    lookup = set(map(str, graph.node_ids))
    rows = [row for row in raw if row["id"] in lookup]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate canonical neurons in release annotations")
    for row in rows:
        row["role"] = (
            "sensory"
            if row["class"] in sensory_class
            else "motor"
            if row["class"] in motor_class
            else "other"
        )
    return rows, {
        **evidence,
        "source_sha256": checksum,
        "graph_citation": graph.manifest["provenance"].get("citation"),
        "unknown_annotation_neurons": graph.n - len(rows),
    }


def prepare_anatomy(graph, kind, source, output, channels=(16,)):
    rows, provenance = _read_annotations(graph, kind, source)
    pairs = defaultdict(lambda: {"L": [], "R": []})
    for row in rows:
        if row["type"] and row["side"] in ("L", "R"):
            pairs[(row["type"], row["class"], row["region"])][row["side"]].append(row["id"])
    bilateral = [
        value["L"] + value["R"]
        for _, value in sorted(pairs.items())
        if len(value["L"]) == len(value["R"]) == 1
    ]
    provenance["bilateral_pair_evidence"] = (
        "Conservative unique left/right type and region pairing; inferred from published labels, not an individually verified homology map. Multi-neuron types are left unpaired."
    )
    provenance["generic_port_access"] = False
    anatomy = {
        "schema": "connectome-anatomical-annotations-v1",
        "graph_fingerprint": graph.fingerprint,
        "sensory_ids": sorted(row["id"] for row in rows if row["role"] == "sensory"),
        "motor_ids": sorted(row["id"] for row in rows if row["role"] == "motor"),
        "bilateral_pairs": bilateral,
        "provenance": provenance,
    }
    anatomy["fingerprint"] = digest_json(anatomy)
    output = Path(output)
    _immutable_json(output / "anatomy.json", anatomy)
    ports = []
    for k in channels:
        if type(k) is not int or k < 1:
            raise ValueError("Anatomical channel counts must be positive integers")
        groups = {}
        for side, role in (("input", "sensory"), ("output", "motor")):
            ordered = sorted(
                (row for row in rows if row["role"] == role),
                key=lambda row: (
                    row["group"],
                    row["function"],
                    row["side"],
                    row["type"],
                    row["id"],
                ),
            )
            if len(ordered) < k:
                raise ValueError(
                    f"Only {len(ordered)} annotated {role} neurons for {k} anatomical ports"
                )
            groups[f"{side}_groups"] = [
                list(group) for group in np.array_split(np.array([row["id"] for row in ordered]), k)
            ]
        record = {
            "schema": "anatomical-ports-v1",
            "graph_fingerprint": graph.fingerprint,
            **groups,
            "provenance": {
                **provenance,
                "anatomy_fingerprint": anatomy["fingerprint"],
                "rule": "Separate annotated sensory and motor populations; sort by peripheral group, function, side and type, then divide into K contiguous groups. Fixed privileged routing, no cross-species neuron mapping.",
            },
        }
        record["fingerprint"] = digest_json(record)
        path = output / f"ports-k{k}.json"
        _immutable_json(path, record)
        ports.append(str(path.resolve()))
    return {
        "anatomy": str((output / "anatomy.json").resolve()),
        "ports": ports,
        "sensory_neurons": len(anatomy["sensory_ids"]),
        "motor_neurons": len(anatomy["motor_ids"]),
        "inferred_bilateral_pairs": len(bilateral),
        "fingerprint": anatomy["fingerprint"],
    }


def _task_seeds(rows, kind, task):
    motors = [row for row in rows if row["role"] == "motor"]
    mode = "locomotor"
    if task in ("hover", "controlled_flight"):
        mode = "flight" if kind != "celegans" else "postural_proxy"
    elif task in ("steering", "heading"):
        mode = "steering"
    elif task == "limb_control":
        mode = "foreleg" if kind != "celegans" else "postural_proxy"
    elif task in ("posture", "depth"):
        mode = "postural_proxy"
    if kind == "banc":
        groups = {
            "flight": {"wing", "haltere", "neck"},
            "steering": {"wing", "neck"},
            "foreleg": {"front_leg"},
            "locomotor": {"front_leg", "middle_leg", "hind_leg"},
            "postural_proxy": {"front_leg", "middle_leg", "hind_leg", "neck"},
        }[mode]
        selected = [row for row in motors if row["group"] in groups]
        rule = {"field": "body_part_effector", "values": sorted(groups)}
    elif kind == "malecns":
        groups = {
            "flight": {"wm", "hm", "nm"},
            "steering": {"wm", "nm"},
            "foreleg": {"fl"},
            "locomotor": {"fl", "ml", "hl"},
            "postural_proxy": {"fl", "ml", "hl", "nm"},
        }[mode]
        selected = [row for row in motors if row["group"] in groups]
        rule = {"field": "motor subclass", "values": sorted(groups)}
    else:
        prefixes = (
            ("SMD", "RMD", "SMB", "RME")
            if mode == "steering"
            else ("DA", "DB", "VA", "VB", "DD", "VD", "AS")
        )
        selected = [row for row in motors if row["id"].startswith(prefixes)]
        rule = {
            "field": "Cook motorneuron class and named motor-neuron classes",
            "prefixes": list(prefixes),
            "interpretation": "head/neck motor versus ventral-cord body-wall motor functional selection",
        }
    return sorted(row["id"] for row in selected), {"functional_module": mode, **rule}


def prepare_task_subgraphs(graph, kind, source, output, *, max_neurons=2048, seed=0):
    if type(max_neurons) is not int or max_neurons < 4:
        raise ValueError("At least four neurons per selected module are required")
    rows, source_provenance = _read_annotations(graph, kind, source)
    lookup = {str(node): index for index, node in enumerate(graph.node_ids)}
    limit = min(max_neurons, graph.n // 3)
    output = Path(output)
    records = {}
    for body, tasks in BODY_TASKS.items():
        for task in tasks:
            key = f"{body}:{task}"
            selected, rule = _task_seeds(rows, kind, task)
            if not selected or len(selected) > limit:
                records[key] = {
                    "status": "infeasible",
                    "seed_neurons": len(selected),
                    "limit": limit,
                }
                continue
            seeds = np.array([lookup[node] for node in selected], dtype=np.int64)
            mask = np.zeros(graph.n, bool)
            mask[seeds] = True
            score = np.bincount(
                graph.src, weights=graph.weight * mask[graph.dst], minlength=graph.n
            ) + np.bincount(graph.dst, weights=graph.weight * mask[graph.src], minlength=graph.n)
            candidates = np.flatnonzero((score > 0) & ~mask)
            order = np.lexsort((candidates, -score[candidates]))
            chosen = np.sort(np.r_[seeds, candidates[order[: limit - len(seeds)]]])
            provenance = {
                **source_provenance,
                "seed_rule": rule,
                "seed_neurons": selected,
                "selection_rule": "All declared motor seeds plus strongest directly connected neurons by total bidirectional synaptic magnitude, with canonical-index tie break",
                "maximum_neurons": limit,
                "relevance_status": "preregistered functional hypothesis, not established task specificity",
                "cross_species_homology_claim": False,
                "non_native_task_note": "The source animal's motor module is a functional proxy when the target body or task has no anatomical counterpart.",
            }
            directory = output / key.replace(":", "-")
            record = {
                "schema": "connectome-neuron-selection-v1",
                "graph_fingerprint": graph.fingerprint,
                "node_ids": list(map(str, graph.node_ids[chosen])),
                "role": "relevant",
                "task": key,
                "provenance": provenance,
            }
            record["fingerprint"] = digest_json(record)
            selection = output / "selections" / f"{key.replace(':', '-')}.json"
            if selection.exists():
                _immutable_json(selection, record)
            else:
                write_selection(
                    graph,
                    record["node_ids"],
                    selection,
                    role="relevant",
                    task=key,
                    provenance=provenance,
                )
            if directory.exists():
                # Validate source selection identity before accepting an earlier
                # preparation. Induced graph integrity is verified by the planner.
                existing = json.loads((directory / "relevant-selection.json").read_text())
                if existing != record:
                    raise ValueError("Previously prepared task subgraph uses another selection")
                controls = json.loads((directory / "controls.json").read_text())
                if (
                    controls.get("seed") != seed
                    or controls.get("parent_graph") != graph.fingerprint
                ):
                    raise ValueError(
                        "Previously prepared subgraph controls use another graph or null seed"
                    )
            else:
                controls = prepare_subgraph_controls(graph, selection, directory, seed=seed)
            records[key] = {"status": "prepared", "selected_neurons": len(chosen), **controls}
    _immutable_json(
        output / "preparation.json", {"graph_fingerprint": graph.fingerprint, "tasks": records}
    )
    return records
