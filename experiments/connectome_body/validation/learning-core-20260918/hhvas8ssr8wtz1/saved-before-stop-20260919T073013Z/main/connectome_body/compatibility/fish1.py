"""Versioned, resumable Fish1 table export and lossless neuron-root graph ingestion.

The client is injected. Authentication remains in CAVE's local credential store;
neither tokens nor client/session objects are serialized by this module.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..graphs import save_graph
from ..util import atomic_json, digest_file, digest_json
from .datasets import validate_fish1_provenance


def integer_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("Neuron IDs must never pass through floating point")
    if isinstance(value, str) and (not value.isdecimal() or not value.isascii()):
        raise ValueError("IDs must be unsigned decimal integers")
    result = int(value)
    if not 0 <= result < 2**64:
        raise ValueError("ID outside uint64 range")
    return str(result)


def annotation_id(value):
    """CAVE annotation IDs are signed int64; segmentation roots are uint64."""
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("Annotation IDs must never pass through floating point")
    if isinstance(value, str):
        digits = value[1:] if value.startswith("-") else value
        if not digits or not digits.isascii() or not digits.isdecimal():
            raise ValueError("Annotation IDs must be signed decimal integers")
    result = int(value)
    if not -(2**63) <= result < 2**63:
        raise ValueError("Annotation ID outside int64 range")
    return str(result)


@dataclass(frozen=True)
class Fish1ExportSpec:
    materialization_version: int
    proofreading_coverage: str
    membership_mode: str = "single_soma_roots"
    datastack: str = "fish1_full"
    segmentation_table: str = "fish1_v250915"
    soma_table: str = "somas"
    synapse_table: str = "synapses_axde"
    label_table: str = "synapses_axde_label"
    page_size: int = 50_000
    server_url: str = "https://global.brain-wire-test.org/"

    def validate(self):
        if type(self.materialization_version) is not int or self.materialization_version < 1:
            raise ValueError("Specify a fixed materialization integer, never latest")
        if not self.proofreading_coverage.strip():
            raise ValueError(
                "Describe reconstruction coverage; a soma-root graph is not a fully proofread connectome"
            )
        if self.membership_mode not in (
            "single_soma_roots",
            "molecularly_annotated_single_soma_roots",
        ):
            raise ValueError("Declare a supported membership rule before downloading")
        if type(self.page_size) is not int or not 1 <= self.page_size <= 500_000:
            raise ValueError("Page size must be between 1 and the Fish1 server cap of 500000")
        if (
            self.datastack != "fish1_full"
            or self.server_url != "https://global.brain-wire-test.org/"
        ):
            raise ValueError("This exporter is restricted to the documented Fish1 service")
        if any(
            not value
            for value in (
                self.segmentation_table,
                self.soma_table,
                self.synapse_table,
                self.label_table,
            )
        ):
            raise ValueError("Declare segmentation and annotation tables")
        if len({self.soma_table, self.synapse_table, self.label_table}) != 3:
            raise ValueError("Export somas, physical synapses and reference labels separately")


def _count(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict("records")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict) and set(value) == {"count"}:
        value = value["count"]
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError("CAVE count response must contain one nonnegative integer")
    return int(value)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_page(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".page-", dir=path.parent)
    try:
        arrays = {}
        for column in columns:
            if column.endswith("root_id"):
                arrays[column] = pa.array([int(row[column]) for row in rows], pa.uint64())
            elif column in ("id", "target_id"):
                arrays[column] = pa.array([int(row[column]) for row in rows], pa.int64())
            else:
                arrays[column] = pa.array([row[column] for row in rows], pa.string())
        with os.fdopen(fd, "wb") as stream:
            pq.write_table(
                pa.table(arrays),
                stream,
                compression="zstd",
                use_dictionary=[column for column in columns if column in ("tag", "cell_type")],
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_page(path):
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".csv":  # Immutable v1 exports remain readable.
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    raise ValueError("Unsupported Fish1 page format")


def _canonical_rows(rows, fields):
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("CAVE pagination ended before its declared count")
    canonical = []
    for row in rows:
        if not set(fields) <= row.keys():
            raise ValueError("Unexpected Fish1 table columns")
        values = {}
        for key in fields:
            if key.endswith("root_id"):
                values[key] = integer_id(row[key])
            elif key in ("id", "target_id"):
                values[key] = annotation_id(row[key])
            else:
                value = row[key]
                values[key] = str(value) if value is not None else "na"
        canonical.append(values)
    return canonical


def _bounded_id_pages(materialize, table, query, count_query, lower, count, page_size):
    """Fetch complete counted ID intervals; never depend on SQL result ordering."""
    pending = [(lower, 2**63 - 1, count)]
    capacity = page_size
    while pending:
        lo, hi, expected = pending.pop()
        if not expected:
            continue
        bounds = {"filter_greater_equal_dict": {"id": lo}, "filter_less_equal_dict": {"id": hi}}
        if expected > capacity:
            if lo == hi:
                raise ValueError("Duplicate Fish1 annotation IDs in a counted interval")
            mid = (lo + hi) // 2
            left = _count(
                materialize.query_table(
                    table,
                    **count_query,
                    filter_greater_equal_dict={"id": lo},
                    filter_less_equal_dict={"id": mid},
                )
            )
            if not 0 <= left <= expected:
                raise ValueError("Fish1 interval count contradicts its parent interval")
            pending.extend([(mid + 1, hi, expected - left), (lo, mid, left)])
            continue
        rows = _canonical_rows(
            materialize.query_table(table, **query, **bounds, limit=expected),
            query["select_columns"],
        )
        if len(rows) > expected:
            raise ValueError("Fish1 interval query exceeded its declared count")
        if len(rows) < expected:
            # A server may cap responses below our requested limit. Partition
            # more finely; an incomplete interval is never saved as a full page.
            capacity = min(capacity, len(rows))
            pending.append((lo, hi, expected))
            continue
        if any(not lo <= int(row["id"]) <= hi for row in rows):
            raise ValueError("Fish1 returned an annotation outside its requested ID interval")
        yield rows, (lo, hi)


def export_fish1(
    client, spec: Fish1ExportSpec, output, *, on_progress=None, request_page_size=None
):
    spec.validate()
    page_size = spec.page_size if request_page_size is None else request_page_size
    if type(page_size) is not int or not 1 <= page_size <= 500_000:
        raise ValueError("Request page size must be between 1 and 500000")
    output = Path(output).resolve()
    request = {"schema": "fish1-export-request-v2", "spec": asdict(spec)}
    request_id = digest_json(request)
    if (output / "provenance.json").exists():
        saved = json.loads((output / "provenance.json").read_text())
        if saved.get("request_identity") != request_id:
            raise FileExistsError("Completed export uses another fixed version or membership rule")
        for name, item in saved["files"].items():
            page = (output / name).resolve()
            if not page.is_relative_to(output) or digest_file(page) != item["sha256"]:
                raise ValueError("A completed Fish1 export page changed")
        return saved
    checkpoint = output / "export-progress.json"
    progress = {"request_identity": request_id, "files": {}, "tables": {}}
    if checkpoint.exists():
        progress = json.loads(checkpoint.read_text())
        if progress.get("request_identity") != request_id:
            raise ValueError("Resume requires exactly the original Fish1 export specification")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "request.json", request)
    source = client.info.segmentation_source(datastack_name=spec.datastack, use_stored=False)
    parsed = urlsplit(source.removeprefix("graphene://"))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or unquote(parsed.path.rstrip("/").split("/")[-1]) != spec.segmentation_table
    ):
        raise ValueError("Live CAVE segmentation source differs from the declared Fish1 table")
    # Store only the public endpoint. Credentials, query parameters and URL
    # fragments can never enter data provenance or an export checkpoint.
    public_source = urlunsplit((parsed.scheme, parsed.hostname, parsed.path, "", ""))
    if progress.get("segmentation_source", public_source) != public_source:
        raise ValueError("Fish1 segmentation endpoint changed since the export began")
    progress["segmentation_source"] = public_source
    materialize = client.materialize
    metadata = _jsonable(
        materialize.get_version_metadata(
            version=spec.materialization_version, datastack_name=spec.datastack
        )
    )
    if int(metadata.get("version", spec.materialization_version)) != spec.materialization_version:
        raise ValueError("CAVE returned metadata for a different materialization")
    tables = materialize.get_tables(
        version=spec.materialization_version, datastack_name=spec.datastack
    )
    if not {spec.soma_table, spec.synapse_table, spec.label_table} <= set(tables):
        raise ValueError("Required soma/synapse tables are absent from the pinned materialization")
    columns = {
        spec.soma_table: ["id", "pt_root_id", "cell_type"],
        spec.synapse_table: ["id", "pre_pt_root_id", "post_pt_root_id"],
        spec.label_table: ["id", "target_id", "tag"],
    }
    table_metadata = {}
    for table in columns:
        source_metadata = materialize.get_table_metadata(
            table, version=spec.materialization_version, datastack_name=spec.datastack
        )
        table_metadata[table] = _jsonable(
            {
                key: source_metadata[key]
                for key in (
                    "schema",
                    "reference_table",
                    "voxel_resolution",
                    "pcg_table_name",
                    "last_updated",
                    "flat_segmentation_source",
                )
                if key in source_metadata
            }
        )
    if table_metadata[spec.synapse_table].get("reference_table"):
        raise ValueError("Connectivity must come from physical synapses, not a reference join")
    if table_metadata[spec.label_table].get("reference_table") != spec.synapse_table:
        raise ValueError("Fish1 label table does not reference the declared physical synapses")
    # A disk-backed unique index detects overlapping or reordered pages without
    # retaining tens of millions of synapse IDs in a Python set.
    with tempfile.TemporaryDirectory(prefix="fish1-id-audit-", dir=output) as audit_dir:
        connection = sqlite3.connect(str(Path(audit_dir) / "ids.sqlite"))
        try:
            for table, fields in columns.items():
                connection.execute("DROP TABLE IF EXISTS seen")
                connection.execute("CREATE TABLE seen (row_id INTEGER PRIMARY KEY)")
                query = {
                    "materialization_version": spec.materialization_version,
                    "datastack_name": spec.datastack,
                    "return_df": True,
                    "select_columns": fields,
                    "merge_reference": False,
                }
                # The Fish1 server ignores get_counts on joined reference tables.
                # Query the physical table directly and bound a malformed reply.
                count_query = {**query, "return_df": False, "get_counts": True, "limit": 1}
                expected = _count(materialize.query_table(table, **count_query))
                if (
                    table in progress["tables"]
                    and progress["tables"][table]["expected_rows"] != expected
                ):
                    raise ValueError("Row count changed at a supposedly fixed materialization")
                offset, maximum_id = 0, -(2**63) - 1

                def audit(rows):
                    try:
                        connection.executemany(
                            "INSERT INTO seen VALUES (?)",
                            [(int(annotation_id(row["id"])),) for row in rows],
                        )
                        connection.commit()
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            "Duplicate Fish1 row ID: unstable pagination or overlapping table rows"
                        ) from exc

                cached = sorted(
                    (
                        (name, item)
                        for name, item in progress["files"].items()
                        if item["table"] == table
                    ),
                    key=lambda pair: pair[1]["offset"],
                )
                for relative, item in cached:
                    path = (output / relative).resolve()
                    if not path.is_relative_to(output) or digest_file(path) != item["sha256"]:
                        raise ValueError("Cached Fish1 page checksum mismatch")
                    if item["offset"] != offset:
                        raise ValueError("Missing or overlapping cached Fish1 page interval")
                    rows = _read_page(path)
                    if not rows or len(rows) != item["rows"]:
                        raise ValueError("Cached Fish1 page row count mismatch")
                    audit(rows)
                    maximum_id = max(maximum_id, max(int(annotation_id(row["id"])) for row in rows))
                    offset += len(rows)
                if offset > expected:
                    raise ValueError("Cached Fish1 export exceeds the declared table count")
                progress["tables"][table] = {"expected_rows": expected, "exported_rows": offset}
                if offset and offset < expected:
                    prefix = _count(
                        materialize.query_table(
                            table, **count_query, filter_less_equal_dict={"id": maximum_id}
                        )
                    )
                    if prefix != offset:
                        raise ValueError(
                            "Cached Fish1 pages are not a complete counted ID prefix; preserve them "
                            "and start a bounded-ID export in a new directory"
                        )
                    progress["tables"][table]["verified_cached_prefix_rows"] = prefix
                    progress["tables"][table]["verified_cached_prefix_max_id"] = str(maximum_id)
                atomic_json(checkpoint, progress)
                for rows, interval in _bounded_id_pages(
                    materialize,
                    table,
                    query,
                    count_query,
                    maximum_id + 1,
                    expected - offset,
                    page_size,
                ):
                    audit(rows)
                    relative = f"{table}/page-{offset:012d}.parquet"
                    path = output / relative
                    _write_page(path, rows, fields)
                    progress["files"][relative] = {
                        "sha256": digest_file(path),
                        "rows": len(rows),
                        "table": table,
                        "offset": offset,
                        "requested_limit": len(rows),
                        "id_range_inclusive": [str(value) for value in interval],
                        "pagination": "complete_counted_id_range",
                    }
                    offset += len(rows)
                    progress["tables"][table]["exported_rows"] = offset
                    atomic_json(checkpoint, progress)
                    if on_progress:
                        on_progress(
                            {"table": table, "exported_rows": offset, "expected_rows": expected}
                        )
                if (
                    offset != expected
                    or _count(materialize.query_table(table, **count_query)) != expected
                ):
                    raise ValueError("Fish1 count changed during a fixed-version export")
        finally:
            connection.close()
    membership = "one nonzero segmentation root with exactly one soma annotation; preserve disconnected selected roots"
    if spec.membership_mode.startswith("molecularly"):
        membership += "; restrict to soma cell_type exc or inh"
    provenance = {
        "schema": "fish1-versioned-export-v2",
        "request_identity": request_id,
        "datastack": spec.datastack,
        "materialization_version": spec.materialization_version,
        "segmentation_table": spec.segmentation_table,
        "segmentation_source": public_source,
        "soma_table": spec.soma_table,
        "synapse_table": spec.synapse_table,
        "label_table": spec.label_table,
        "page_format": "parquet-zstd",
        "pagination_audit": "complete counted nonoverlapping signed-ID intervals; cached legacy prefixes count-verified; global unique-ID count equals unjoined table count",
        "membership_rule": membership,
        "membership_mode": spec.membership_mode,
        "proofreading_coverage": spec.proofreading_coverage,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": progress["files"],
        "tables": progress["tables"],
        "materialization_metadata": metadata,
        "table_metadata": table_metadata,
        "documentation": "https://fish1-release.storage.googleapis.com/programmatic.html",
        "is_synthetic": False,
        "current_root_remapping": False,
    }
    validate_fish1_provenance(provenance)
    atomic_json(output / "provenance.json", provenance)
    return provenance


def import_fish1(export_directory, output, *, on_progress=None):
    directory, output = Path(export_directory).resolve(), Path(output)
    provenance = validate_fish1_provenance(json.loads((directory / "provenance.json").read_text()))
    if (
        provenance.get("schema") not in ("fish1-versioned-export-v1", "fish1-versioned-export-v2")
        or provenance.get("current_root_remapping") is not False
    ):
        raise ValueError("Only the fixed-version Fish1 export schema is accepted")
    if output.exists():
        raise FileExistsError("Prepared Fish1 graphs are immutable")
    files = provenance["files"]
    for name, item in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest_file(path) != item["sha256"]:
            raise ValueError("Fish1 page is outside the export or has changed")

    def table_rows(table):
        items = sorted(
            ((name, item) for name, item in files.items() if item["table"] == table),
            key=lambda pair: pair[1]["offset"],
        )
        offset = 0
        for name, item in items:
            if item["offset"] != offset:
                raise ValueError("Missing or overlapping Fish1 page interval")
            rows = _read_page(directory / name)
            if len(rows) != item["rows"]:
                raise ValueError("Fish1 page row count mismatch")
            yield from rows
            offset += len(rows)
            if on_progress:
                on_progress(
                    {
                        "stage": "read_table",
                        "table": table,
                        "processed_rows": offset,
                        "expected_rows": provenance["tables"][table]["expected_rows"],
                    }
                )
        if offset != provenance["tables"][table]["expected_rows"]:
            raise ValueError("Fish1 export is incomplete")

    roots, soma_ids = defaultdict(list), set()
    for row in table_rows(provenance["soma_table"]):
        soma, root = annotation_id(row["id"]), integer_id(row["pt_root_id"])
        if soma in soma_ids:
            raise ValueError("Duplicate soma annotation in Fish1 export")
        soma_ids.add(soma)
        roots[root].append((soma, row["cell_type"]))
    mode = provenance["membership_mode"]
    if mode not in ("single_soma_roots", "molecularly_annotated_single_soma_roots"):
        raise ValueError("Unknown Fish1 membership mode")
    nodes = sorted(
        (
            root
            for root, rows in roots.items()
            if root != "0"
            and len(rows) == 1
            and (mode == "single_soma_roots" or rows[0][1] in ("exc", "inh"))
        ),
        key=int,
    )
    if len(nodes) < 4:
        raise ValueError("The declared Fish1 membership yields fewer than four neurons")
    lookup = {node: i for i, node in enumerate(nodes)}
    signs = np.array(
        [{"exc": 1, "inh": -1}.get(roots[node][0][1], 0) for node in nodes], dtype=np.int8
    )
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="fish1-aggregate-", dir=directory) as aggregate_dir:
        connection = sqlite3.connect(str(Path(aggregate_dir) / "edges.sqlite"))
        try:
            # This database is disposable scratch space; source pages and the final
            # graph are separately checksummed and written atomically.
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE seen (synapse INTEGER PRIMARY KEY, source_sign INTEGER)"
            )
            connection.execute(
                "CREATE TABLE labels (id INTEGER PRIMARY KEY, target INTEGER, sign INTEGER)"
            )
            connection.execute(
                "CREATE TABLE edges (src INTEGER, dst INTEGER, n INTEGER, PRIMARY KEY(src, dst)) WITHOUT ROWID"
            )
            batch, ids, legacy_labels = [], [], []

            def flush():
                try:
                    connection.executemany("INSERT INTO seen VALUES (?, ?)", ids)
                    connection.executemany("INSERT INTO labels VALUES (?, ?, ?)", legacy_labels)
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate synapse annotation in Fish1 export") from exc
                connection.executemany(
                    "INSERT INTO edges VALUES (?, ?, ?) ON CONFLICT(src, dst) DO UPDATE SET n=n+1",
                    batch,
                )
                connection.commit()
                batch.clear()
                ids.clear()
                legacy_labels.clear()

            for row in table_rows(provenance["synapse_table"]):
                synapse = int(annotation_id(row["id"]))
                src, dst = integer_id(row["pre_pt_root_id"]), integer_id(row["post_pt_root_id"])
                counts["raw_synapses"] += 1
                source_sign = None
                if src not in lookup or dst not in lookup:
                    counts["outside_membership"] += 1
                elif src == dst:
                    counts["autapses_excluded"] += 1
                else:
                    source, target = lookup[src], lookup[dst]
                    source_sign = int(signs[source])
                    batch.append((source, target, 1))
                    counts["retained_synapses"] += 1
                ids.append((synapse, source_sign))
                if provenance["schema"] == "fish1-versioned-export-v1":
                    legacy_labels.append((synapse, synapse, {"1": -1, "2": 1}.get(row["tag"], 0)))
                if len(ids) >= 100_000:
                    flush()
            flush()
            if provenance["schema"] == "fish1-versioned-export-v2":
                labels = []
                for row in table_rows(provenance["label_table"]):
                    labels.append(
                        (
                            int(annotation_id(row["id"])),
                            int(annotation_id(row["target_id"])),
                            {"1": -1, "2": 1}.get(row["tag"], 0),
                        )
                    )
                    if len(labels) >= 100_000:
                        try:
                            connection.executemany("INSERT INTO labels VALUES (?, ?, ?)", labels)
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("Duplicate label annotation in Fish1 export") from exc
                        connection.commit()
                        labels.clear()
                try:
                    connection.executemany("INSERT INTO labels VALUES (?, ?, ?)", labels)
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate label annotation in Fish1 export") from exc
            connection.commit()
            if on_progress:
                on_progress({"stage": "audit_labels"})
            connection.execute("CREATE INDEX label_target ON labels(target)")
            counts["label_annotations"] = connection.execute(
                "SELECT COUNT(*) FROM labels"
            ).fetchone()[0]
            counts["labels_without_physical_synapse"] = connection.execute(
                "SELECT COUNT(*) FROM labels l LEFT JOIN seen s ON s.synapse=l.target WHERE s.synapse IS NULL"
            ).fetchone()[0]
            counts["edge_tag_disagrees_with_soma_label"] = connection.execute(
                "SELECT COUNT(*) FROM labels l JOIN seen s ON s.synapse=l.target "
                "WHERE l.sign != 0 AND s.source_sign != 0 AND l.sign != s.source_sign"
            ).fetchone()[0]
            counts["retained_synapses_without_label"] = connection.execute(
                "SELECT COUNT(*) FROM seen s WHERE s.source_sign IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM labels l WHERE l.target=s.synapse)"
            ).fetchone()[0]
            counts["physical_synapses_with_multiple_labels"] = connection.execute(
                "SELECT COUNT(*) FROM (SELECT l.target FROM labels l JOIN seen s ON s.synapse=l.target "
                "GROUP BY l.target HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            size = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            if on_progress:
                on_progress({"stage": "write_graph", "neurons": len(nodes), "edges": size})
            src, dst, weight = (
                np.empty(size, np.uint32),
                np.empty(size, np.uint32),
                np.empty(size, np.float32),
            )
            cursor = connection.execute("SELECT src, dst, n FROM edges ORDER BY dst, src")
            offset = 0
            while chunk := cursor.fetchmany(100_000):
                values = np.asarray(chunk, dtype=np.int64)
                if values[:, 2].max() > 2**24:
                    raise ValueError("Synapse count exceeds exact float32 integer representation")
                end = offset + len(values)
                src[offset:end], dst[offset:end], weight[offset:end] = values.T
                offset = end
        finally:
            connection.close()
    return save_graph(
        output,
        nodes,
        src,
        dst,
        weight,
        signs=signs,
        provenance={
            "dataset": "Fish1",
            "species": "Danio rerio",
            "developmental_stage": "7 dpf",
            "resolution": "versioned single-soma segmentation root, not guaranteed fully proofread neuron",
            "weight_semantics": "count of retained unique axon-to-dendrite synapse annotation IDs",
            "sign_evidence": "soma molecular cell_type exc/inh; unknown remains zero; edge tags do not override soma identity",
            "export": provenance,
            "export_provenance_sha256": digest_file(directory / "provenance.json"),
            "membership_audit": {
                "selected_roots": len(nodes),
                "multiple_soma_roots_excluded": sum(len(rows) > 1 for rows in roots.values()),
                "zero_root_somas_excluded": len(roots.get("0", [])),
                **counts,
            },
            "citation": "https://fish1-release.storage.googleapis.com/paper.html",
            "is_synthetic": provenance.get("is_synthetic", False),
        },
    )
