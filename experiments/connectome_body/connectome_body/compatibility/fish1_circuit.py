"""Published Fish1 HMI circuit, explicitly separate from the whole-volume export.

The source lists contacts from both endpoint perspectives without shared synapse
IDs. This importer therefore uses binary directed adjacency, never a sum that
could double-count the same physical synapse. The circuit is a secondary input,
not a replacement for a reconstructed whole zebrafish nervous system.
"""

from __future__ import annotations

import ast
import io
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from ..data import download_verified
from ..graphs import save_graph
from ..util import atomic_json, digest_file
from .datasets import workbook_sheets

HMI_URL = "https://storage.googleapis.com/fish1-release/paper_data/HMI_analysis.zip"
HMI_SHA256 = "5507d2d32e0a111faf8a07ab69edb33f8325f4a1bbc3c74ddd5fb33f6383feca"
HMI_MEMBER = "HMI_analysis/data/em_zfish1_dataframe.xlsx"
RECONSTRUCTED_LABELS = (
    "soma, dendrite(c), axon(c)",
    "soma, dendrite(reconstructed), axon(reconstructed)",
)


def _lore_id(value):
    # Excel stores these small stable soma IDs numerically. Do not accept a
    # floating-point segmentation root or round a nonintegral value.
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or abs(value) >= 2**53:
            raise ValueError("HMI soma IDs must be exactly represented integers")
        value = int(value)
    if isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0:
        raise ValueError("Invalid HMI soma ID")
    return str(int(value))


def hmi_adjacency(rows):
    catalog = {}
    for row in rows:
        identity = _lore_id(row["Cell ID"])
        if identity in catalog:
            raise ValueError("Duplicate HMI soma ID")
        catalog[identity] = row
    selected = {
        key: row
        for key, row in catalog.items()
        if row.get("reconstruction_status") in RECONSTRUCTED_LABELS
    }
    nodes = sorted(selected, key=int)
    if len(nodes) < 4:
        raise ValueError("HMI reconstruction membership has fewer than four cells")
    lookup = {node: index for index, node in enumerate(nodes)}
    edges, counts = set(), Counter()
    for cell, row in catalog.items():
        for direction in ("inputs", "outputs"):
            raw = row.get(direction)
            if raw is None or str(raw).strip().lower() in ("", "n/a", "na", "[]"):
                continue
            try:
                contacts = ast.literal_eval(raw) if isinstance(raw, str) else raw
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Malformed HMI {direction} for soma {cell}") from exc
            if not isinstance(contacts, list):
                raise ValueError("HMI contact annotations must be lists")
            for contact in contacts:
                if not isinstance(contact, (list, tuple)) or len(contact) < 4:
                    raise ValueError("Malformed HMI contact tuple")
                counts["contact_annotations"] += 1
                if contact[0] is None or (
                    isinstance(contact[0], str)
                    and contact[0].strip().lower() in ("", "-", "no id???", "n/a")
                ):
                    counts["unidentified_partner_annotations"] += 1
                    continue
                partner = _lore_id(contact[0])
                src, dst = (partner, cell) if direction == "inputs" else (cell, partner)
                if src not in selected or dst not in selected:
                    counts["outside_membership_annotations"] += 1
                elif src == dst:
                    counts["autapse_annotations"] += 1
                else:
                    counts["retained_contact_annotations"] += 1
                    edges.add((lookup[src], lookup[dst]))
    if not edges:
        raise ValueError("No HMI connections remain under the declared membership")
    pairs = sorted(edges, key=lambda pair: (pair[1], pair[0]))
    signs = np.array(
        [
            {"Gad1B": -1, "VGluT2": 1}.get(selected[node].get("final_neurotransmitter_ID"), 0)
            for node in nodes
        ],
        dtype=np.int8,
    )
    audit = {
        "catalog_cells": len(catalog),
        "selected_cells": len(nodes),
        "reconstruction_labels": list(RECONSTRUCTED_LABELS),
        "selected_status_counts": dict(
            Counter(row["reconstruction_status"] for row in selected.values())
        ),
        "unique_directed_pairs": len(pairs),
        **counts,
    }
    return nodes, pairs, signs, selected, audit


def prepare_fish1_hmi(raw_directory, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Prepared Fish1 circuit graphs are immutable")
    archive_path = Path(raw_directory) / "HMI_analysis.zip"
    download_verified(HMI_URL, archive_path, HMI_SHA256)
    with zipfile.ZipFile(archive_path) as archive:
        workbook = archive.read(HMI_MEMBER)
    sheets = workbook_sheets(io.BytesIO(workbook))
    if list(sheets) != ["Sheet1"]:
        raise ValueError("Pinned HMI workbook layout changed")
    cells = sheets["Sheet1"]
    headings = {column: value for (row, column), value in cells.items() if row == 1}
    if headings.get(1) != "Cell ID":
        raise ValueError("Pinned HMI workbook has no leading soma ID column")
    rows = [
        {name: cells.get((row, column)) for column, name in headings.items()}
        for row in range(2, max(row for row, _ in cells) + 1)
        if cells.get((row, 1)) is not None
    ]
    nodes, pairs, signs, selected, audit = hmi_adjacency(rows)
    graph = save_graph(
        output,
        nodes,
        np.array([src for src, _ in pairs], dtype=np.uint32),
        np.array([dst for _, dst in pairs], dtype=np.uint32),
        np.ones(len(pairs), dtype=np.float32),
        signs=signs,
        provenance={
            "dataset": "Fish1-HMI-curated-binary",
            "species": "Danio rerio",
            "developmental_stage": "7 dpf",
            "resolution": "stable soma lore ID in the published HMI circuit catalog",
            "coverage": "secondary curated circuit; not a whole-brain reconstruction",
            "membership": "source axon/dendrite reconstruction labels, selected before observing control performance",
            "membership_audit": audit,
            "weight_semantics": "binary presence of an annotated directed cell pair; not synapse multiplicity",
            "sign_evidence": "source final_neurotransmitter_ID: Gad1B inhibitory, VGluT2 excitatory; other values unknown",
            "limitations": "Source reconstruction labels are retained, not independently revalidated. Contacts to unselected or unidentified partners are excluded. Binary union avoids double-counting input/output records without assuming coordinate equivalence.",
            "source_url": HMI_URL,
            "source_archive_sha256": digest_file(archive_path),
            "source_member": HMI_MEMBER,
            "citation": "https://doi.org/10.1101/2025.06.10.658982",
            "is_synthetic": False,
        },
    )
    atomic_json(
        output / "cell-annotations.json",
        {"graph_fingerprint": graph.fingerprint, "cells": selected, "generic_port_access": False},
    )
    return graph
