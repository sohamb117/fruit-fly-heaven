"""Pinned source downloads and dataset-independent graph interchange."""

from __future__ import annotations

import json
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.feather as feather
import pyarrow.parquet as parquet

from .body import FLYBODY_REVISION, PROJECT, verify_flybody
from .graphs import save_graph
from .util import digest_file


def source_registry():
    return json.loads((PROJECT / "configs/sources.json").read_text())


def download_verified(url: str, path: Path, expected: str):
    if path.is_file():
        if digest_file(path) != expected:
            raise ValueError(f"Existing source checksum mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(
                url, timeout=120, context=ssl.create_default_context(cafile=certifi.where())
            ) as response,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if digest_file(tmp) != expected:
            raise ValueError(f"Upstream bytes changed: {url}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def bootstrap_flybody(destination: Path | None = None):
    destination = destination or PROJECT / "references/flybody"
    if destination.exists():
        return verify_flybody(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        archive = Path(tmp) / "source.tar.gz"
        with (
            urllib.request.urlopen(
                f"https://codeload.github.com/TuragaLab/flybody/tar.gz/{FLYBODY_REVISION}",
                timeout=120,
                context=ssl.create_default_context(cafile=certifi.where()),
            ) as response,
            archive.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        extracted = Path(tmp) / f"flybody-{FLYBODY_REVISION}"
        checksum = verify_flybody(str(extracted))
        extracted.rename(destination)
    return checksum


def _verify_sources(name, raw, download):
    source = source_registry()["datasets"][name]
    if source["status"] != "public_graph":
        raise ValueError(f"{name}: {source['reason']}")
    raw = Path(raw)
    for filename, spec in source["files"].items():
        path = raw / filename
        if download:
            download_verified(spec["url"], path, spec["sha256"])
        elif not path.is_file():
            raise FileNotFoundError(f"{path}; supply --download or a local --raw directory")
        elif digest_file(path) != spec["sha256"]:
            raise ValueError(f"Pinned source changed: {path}")
    return source


def prepare_banc(raw: Path, output: Path, download: bool = False):
    source = _verify_sources("banc", raw, download)
    meta = feather.read_table(raw / "meta.feather")
    edges = feather.read_table(raw / "edges-v3.feather", columns=["pre", "post", "count"])
    excluded = pc.is_in(
        meta["super_class"], value_set=pa.array(["glia", "trachea", "not_a_neuron"])
    )
    excluded_ids = meta["banc_888_id"].filter(excluded)
    keep = pc.and_(
        pc.invert(pc.is_in(edges["pre"], value_set=excluded_ids)),
        pc.invert(pc.is_in(edges["post"], value_set=excluded_ids)),
    )
    excluded_edges = len(edges) - pc.sum(keep).as_py()
    edges = edges.filter(keep)
    nodes = np.unique(
        np.concatenate(
            [
                meta["banc_888_id"].filter(pc.invert(excluded)).to_numpy(),
                pc.unique(edges["pre"]).to_numpy(zero_copy_only=False),
                pc.unique(edges["post"]).to_numpy(zero_copy_only=False),
            ]
        )
    )
    ids = pa.array(nodes)
    src = pc.index_in(edges["pre"], value_set=ids).to_numpy().astype(np.uint32)
    dst = pc.index_in(edges["post"], value_set=ids).to_numpy().astype(np.uint32)
    # This optional sensitivity condition is a transmitter-to-sign assumption,
    # not measured electrophysiology. Primary analyses use shared random Dale signs.
    lookup = {
        int(row["banc_888_id"]): row.get("neurotransmitter_verified")
        or row.get("neurotransmitter_predicted")
        for row in meta.to_pylist()
    }
    sign_map = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
    signs = np.array([sign_map.get(lookup.get(int(node)), 0) for node in nodes], dtype=np.int8)
    provenance = {
        **source,
        "filter": "Exclude explicit glia/trachea/not_a_neuron; retain all other metadata and edge endpoints; no count cutoff",
        "excluded_nodes": len(excluded_ids),
        "excluded_edges": int(excluded_edges),
        "sign_evidence": "Assumed insect transmitter effect; unknown/neuromodulatory labels remain 0",
        "unknown_annotated_nodes": pc.index_in(ids, value_set=meta["banc_888_id"]).null_count,
    }
    return save_graph(
        output, nodes, src, dst, edges["count"].to_numpy(), signs=signs, provenance=provenance
    )


def prepare_malecns(raw: Path, output: Path, download: bool = False):
    source = _verify_sources("malecns", raw, download)
    meta = feather.read_table(raw / "annotations.feather")
    # Full flat file includes millions of segmentation fragments. Membership is
    # fixed by the published curated 'Traced' annotations, never task performance.
    traced = pc.fill_null(pc.equal(meta["status"], "Traced"), False)
    nodes = np.sort(meta["bodyId"].filter(traced).to_numpy())
    ids = pa.array(nodes)
    pieces, raw_edges, dropped = [], 0, 0
    with pa.memory_map(str(raw / "edges.feather")) as mapped:
        reader = pa.ipc.open_file(mapped)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            src = pc.index_in(batch["body_pre"], value_set=ids)
            dst = pc.index_in(batch["body_post"], value_set=ids)
            keep = pc.and_(pc.is_valid(src), pc.is_valid(dst))
            raw_edges += len(batch)
            dropped += len(batch) - pc.sum(keep).as_py()
            pieces.append(
                (
                    src.filter(keep).to_numpy().astype(np.uint32),
                    dst.filter(keep).to_numpy().astype(np.uint32),
                    batch["weight"].filter(keep).to_numpy().astype(np.float32),
                )
            )
    src, dst, weight = (np.concatenate([p[j] for p in pieces]) for j in range(3))
    provenance = {
        **source,
        "filter": "Induced graph on curated status == Traced; no count cutoff",
        "annotation_rows": len(meta),
        "raw_edges": raw_edges,
        "excluded_edges": int(dropped),
        "sign_evidence": "No per-node transmitter signs imported; use random_dale primary condition",
    }
    return save_graph(output, nodes, src, dst, weight, provenance=provenance)


def _read_interchange(path: Path, string_columns):
    if path.suffix in (".feather", ".arrow"):
        table = feather.read_table(path)
    elif path.suffix == ".parquet":
        table = parquet.read_table(path)
    elif path.suffix == ".csv":
        table = csv.read_csv(
            path,
            convert_options=csv.ConvertOptions(
                column_types={name: pa.string() for name in string_columns}
            ),
        )
    else:
        raise ValueError("Use CSV, Feather, Arrow IPC, or Parquet")
    for name in string_columns:
        if name not in table.column_names or pa.types.is_floating(table[name].type):
            raise ValueError(f"Missing or floating-point ID column {name}")
        if table[name].null_count:
            raise ValueError(f"Null IDs in {name}")
        table = table.set_column(
            table.schema.get_field_index(name), name, pc.cast(table[name], pa.string())
        )
    return table


def import_edges(
    edge_path: Path,
    node_path: Path,
    provenance_path: Path,
    output: Path,
    pre="pre",
    post="post",
    weight="weight",
    node_id="id",
):
    provenance = json.loads(provenance_path.read_text())
    required = ("dataset", "species", "release", "coverage", "resolution", "citation", "license")
    if any(not provenance.get(key) for key in required):
        raise ValueError(f"Provenance must specify {required}")
    if provenance["resolution"] != "neuron_synapse":
        raise ValueError(
            "Region-level or modeled connectivity cannot be labeled a biological neuron graph"
        )
    edges = _read_interchange(edge_path, [pre, post])
    nodes = _read_interchange(node_path, [node_id])
    ids = nodes[node_id]
    if len(pc.unique(ids)) != len(ids):
        raise ValueError("Duplicate node IDs")
    src, dst = pc.index_in(edges[pre], value_set=ids), pc.index_in(edges[post], value_set=ids)
    if src.null_count or dst.null_count:
        raise ValueError("Edge endpoints absent from node table; membership must be explicit")
    if weight not in edges.column_names:
        raise ValueError("Weight column required; aggregate synapses to directed pair counts first")
    signs = nodes["sign"].to_numpy() if "sign" in nodes.column_names else None
    provenance["source_files"] = {
        "edges_sha256": digest_file(edge_path),
        "nodes_sha256": digest_file(node_path),
        "provenance_sha256": digest_file(provenance_path),
    }
    return save_graph(
        output,
        np.asarray(ids.to_pylist()),
        src.to_numpy(),
        dst.to_numpy(),
        edges[weight].to_numpy(),
        signs=signs,
        provenance=provenance,
    )
