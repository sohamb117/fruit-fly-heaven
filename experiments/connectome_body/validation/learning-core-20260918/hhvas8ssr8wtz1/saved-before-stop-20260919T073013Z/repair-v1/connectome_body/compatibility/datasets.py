"""Pinned C. elegans ingestion and explicit versioned Fish1 export contracts."""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..data import download_verified
from ..graphs import save_graph
from ..util import atomic_json, digest_file

COOK_SOURCES = {
    "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx": {
        "url": "https://wormwiring.org/si/SI%205%20Connectome%20adjacency%20matrices%2C%20corrected%20July%202020.xlsx",
        "sha256": "1f4fdbf84746b69b49a8da0816f52787860ce349b638dce37924ba80f90c70c9",
    },
    "SI 4 Cell lists.xlsx": {
        "url": "https://wormwiring.org/si/SI%204%20Cell%20lists.xlsx",
        "sha256": "f5c524407e196ba8045cfb86fb996d41730e4b899a7c24b0c525a4d2ad0a7376",
    },
}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def workbook_sheets(path):
    """Read cell values without Excel execution, formula evaluation, or optional packages."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for value in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS):
                strings.append("".join(t.text or "" for t in value.iterfind(".//m:t", NS)))
        relations = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall("m:sheets/m:sheet", NS):
            target = relations[sheet.attrib[RID]]
            member = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join("xl", target))
            )
            if not member.startswith("xl/"):
                raise ValueError("Workbook relationship points outside xl/")
            cells = {}
            for cell in ET.fromstring(archive.read(member)).findall("m:sheetData/m:row/m:c", NS):
                if cell.find("m:f", NS) is not None:
                    raise ValueError("Formula cells are not accepted as source connectivity data")
                value = cell.find("m:v", NS)
                kind = cell.attrib.get("t", "n")
                if kind == "inlineStr":
                    parsed = "".join(t.text or "" for t in cell.iterfind(".//m:t", NS))
                elif value is None or value.text is None:
                    continue
                elif kind == "s":
                    parsed = strings[int(value.text)]
                elif kind == "n":
                    parsed = float(value.text)
                else:
                    raise ValueError(f"Unsupported source cell type: {kind}")
                address = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell.attrib["r"])
                if address is None:
                    raise ValueError("Malformed workbook cell address")
                column = 0
                for character in address[1]:
                    column = column * 26 + ord(character) - ord("A") + 1
                cells[(int(address[2]), column)] = parsed
            result[sheet.attrib["name"]] = cells
    return result


def prepare_celegans(raw_directory, output, *, connections="chemical"):
    if connections not in ("chemical", "electrical", "combined"):
        raise ValueError("Specify chemical, electrical, or explicitly combined connections")
    raw_directory, output = Path(raw_directory), Path(output)
    if output.exists():
        raise FileExistsError("Prepared biological graphs are immutable")
    for name, spec in COOK_SOURCES.items():
        download_verified(spec["url"], raw_directory / name, spec["sha256"])
    cell_tables = workbook_sheets(raw_directory / "SI 4 Cell lists.xlsx")
    neuron_types = {"sensory", "interneuron", "motorneuron", "neuron"}
    types = {}
    for name in ("pharynx", "sex-shared", "hermaphrodite specific"):
        cells = cell_tables[name]
        for (row, column), value in cells.items():
            if column == 1 and cells.get((row, 2)) in neuron_types:
                if not isinstance(value, str) or value in types:
                    raise ValueError("Cell list has duplicate or malformed neuronal identities")
                types[value] = cells[(row, 2)]
    if len(types) != 302:
        raise ValueError("Pinned adult-hermaphrodite cell list no longer yields 302 neurons")
    nodes = sorted(types)
    lookup = {node: index for index, node in enumerate(nodes)}
    workbook = workbook_sheets(
        raw_directory / "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
    )
    names = []
    if connections in ("chemical", "combined"):
        names.append("hermaphrodite chemical")
    if connections in ("electrical", "combined"):
        names.append("hermaphrodite gap jn symmetric")
    edges, counts = {}, {}
    for name in names:
        cells = workbook[name]
        # The source legend declares presynaptic rows and postsynaptic columns.
        rows = {row: value for (row, col), value in cells.items() if col == 3 and row >= 4}
        columns = {col: value for (row, col), value in cells.items() if row == 3 and col >= 4}
        retained, excluded = 0, 0
        for (row, col), value in cells.items():
            if row < 4 or col < 4 or not isinstance(value, float) or value == 0:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError("Connectivity entries must be finite, nonnegative extents")
            source, target = rows.get(row), columns.get(col)
            if source not in lookup or target not in lookup:
                excluded += 1
                continue
            key = (lookup[source], lookup[target])
            edges[key] = edges.get(key, 0.0) + value
            retained += 1
        counts[name] = {"retained_entries": retained, "excluded_non_neuronal_entries": excluded}
    pairs = sorted(edges)
    graph = save_graph(
        output,
        nodes,
        np.array([s for s, _ in pairs]),
        np.array([d for _, d in pairs]),
        np.array([edges[pair] for pair in pairs]),
        provenance={
            "dataset": "Cook2019Herm",
            "species": "Caenorhabditis elegans",
            "sex": "hermaphrodite",
            "developmental_stage": "adult",
            "release": "Cook2019-corrected-July2020",
            "resolution": "neuron",
            "coverage": "302-neuron whole-adult hermaphrodite; selected connection modalities",
            "connections": connections,
            "source_tables": counts,
            "weight_semantics": "total EM serial sections of connectivity, not raw synapse count",
            "assembly": "Multiple animals with source-documented extrapolation across reconstruction gaps",
            "sign_evidence": "No neurotransmitter signs supplied by these workbooks; all unknown",
            "citation": "https://doi.org/10.1038/s41586-019-1352-7",
            "documentation": "https://wormwiring.org/pages/adjacency.html",
            "files": COOK_SOURCES,
            "is_synthetic": False,
            "electrical_model_caveat": "Electrical edges, when requested, are represented as paired directed rate edges, not conductance gap-junction physiology",
        },
    )
    atomic_json(
        output / "cell-types.json",
        {
            "graph_fingerprint": graph.fingerprint,
            "cell_types": types,
            "purpose": "optional oracle/subgraph annotations; never generic port inputs",
            "source_sha256": digest_file(raw_directory / "SI 4 Cell lists.xlsx"),
        },
    )
    return graph


def validate_fish1_provenance(provenance):
    """Reject a changing/current-root dump masquerading as a versioned connectome."""
    required = (
        "datastack",
        "materialization_version",
        "segmentation_table",
        "membership_rule",
        "proofreading_coverage",
        "synapse_table",
        "exported_at_utc",
        "files",
    )
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        raise ValueError(f"Fish1 export is missing: {', '.join(missing)}")
    version = provenance["materialization_version"]
    if type(version) is not int or version < 1:
        raise ValueError("Fish1 requires a fixed materialization integer, never 'latest'")
    for spec in provenance["files"].values():
        if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", ""))):
            raise ValueError("Every exported Fish1 table needs a SHA-256 digest")
    return provenance
