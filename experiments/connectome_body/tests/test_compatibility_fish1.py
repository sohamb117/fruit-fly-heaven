import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from connectome_body.compatibility.fish1 import (
    Fish1ExportSpec,
    annotation_id,
    export_fish1,
    import_fish1,
    integer_id,
)
from connectome_body.util import digest_file


class Tables:
    def __init__(self):
        self.roots = [2**63 + value for value in (5, 7, 11, 13, 17, 19)]
        labels = ("exc", "inh", "na", "exc", "na", "na", "na", "na")
        roots = [*self.roots[:5], 0, self.roots[5], self.roots[5]]
        self.data = {
            "somas": [
                {"id": i + 1, "pt_root_id": root, "cell_type": label}
                for i, (root, label) in enumerate(zip(roots, labels, strict=True))
            ]
        }
        synapses = [
            {
                "id": -(2**62) + i,
                "pre_pt_root_id": self.roots[s],
                "post_pt_root_id": self.roots[d],
                "tag": tag,
            }
            for i, (s, d, tag) in enumerate(
                [
                    (0, 1, "2"),
                    (0, 1, "1"),
                    (1, 2, "1"),
                    (2, 0, "na"),
                    (0, 0, "2"),
                    (3, 5, "2"),
                    (5, 0, "na"),
                    (3, 0, "2"),
                ]
            )
        ]
        self.data["synapses_axde"] = [
            {key: row[key] for key in ("id", "pre_pt_root_id", "post_pt_root_id")}
            for row in synapses
        ]
        self.data["synapses_axde_label"] = [
            {"id": row["id"] + 100, "target_id": row["id"], "tag": row["tag"]} for row in synapses
        ]
        self.calls, self.fail, self.label_reads = [], False, 0

    def get_version_metadata(self, **kwargs):
        assert kwargs == {"version": 574, "datastack_name": "fish1_full"}
        return {"version": 574, "time_stamp": "2025-09-15T00:00:00Z"}

    def get_tables(self, **kwargs):
        assert kwargs == {"version": 574, "datastack_name": "fish1_full"}
        return list(self.data)

    def get_table_metadata(self, table, **kwargs):
        assert kwargs == {"version": 574, "datastack_name": "fish1_full"}
        return {"reference_table": "synapses_axde" if table.endswith("_label") else None}

    def query_table(self, table, **kwargs):
        assert kwargs["materialization_version"] == 574
        assert kwargs["merge_reference"] is False
        assert kwargs["return_df"] is not bool(kwargs.get("get_counts"))
        self.calls.append((table, copy.deepcopy(kwargs)))
        assert "offset" not in kwargs  # CAVE does not guarantee stable row order.
        selected = self.data[table]
        if "filter_greater_equal_dict" in kwargs:
            selected = [
                row for row in selected if row["id"] >= kwargs["filter_greater_equal_dict"]["id"]
            ]
        if "filter_less_equal_dict" in kwargs:
            selected = [
                row for row in selected if row["id"] <= kwargs["filter_less_equal_dict"]["id"]
            ]
        if kwargs.get("get_counts"):
            assert kwargs["limit"] == 1
            return [{"count": len(selected)}]
        if table == "synapses_axde_label":
            if self.fail and self.label_reads == 1:
                raise ConnectionError("simulated interruption")
            self.label_reads += 1
        # Deliberately unordered, with a server cap smaller than the request.
        return list(reversed(selected))[: min(2, kwargs["limit"])]


def client_for(tables, source="graphene://https://example.test/segmentation/table/fish1_v250915"):
    def segmentation_source(**kwargs):
        assert kwargs == {"datastack_name": "fish1_full", "use_stored": False}
        return source

    return SimpleNamespace(
        materialize=tables, info=SimpleNamespace(segmentation_source=segmentation_source)
    )


def test_export_resume_and_import_preserve_large_ids_counts_signs_and_isolates(tmp_path):
    tables = Tables()
    client = client_for(tables)
    spec = Fish1ExportSpec(574, "Synthetic source for software testing", page_size=2)
    tables.fail = True
    with pytest.raises(ConnectionError):
        export_fish1(client, spec, tmp_path / "export")
    progress = json.loads((tmp_path / "export/export-progress.json").read_text())
    assert progress["tables"]["synapses_axde_label"]["exported_rows"] == 2
    tables.fail = False
    tables.calls.clear()
    provenance = export_fish1(client, spec, tmp_path / "export", request_page_size=5)
    assert not any(
        table == "somas" and not options.get("get_counts") for table, options in tables.calls
    )
    assert provenance["current_root_remapping"] is False
    assert all(
        item["pagination"] == "complete_counted_id_range" for item in provenance["files"].values()
    )
    graph = import_fish1(tmp_path / "export", tmp_path / "graph")
    assert graph.n == 5 and graph.m == 4
    assert set(graph.node_ids) == set(map(str, tables.roots[:5]))
    np.testing.assert_array_equal(graph.signs, [1, -1, 0, 1, 0])
    assert graph.weight.sum() == 5
    assert graph.manifest["summary"]["isolated_neurons"] == 1
    audit = graph.manifest["provenance"]["membership_audit"]
    assert audit["multiple_soma_roots_excluded"] == 1 and audit["autapses_excluded"] == 1
    assert audit["edge_tag_disagrees_with_soma_label"] == 1
    assert audit["retained_synapses_without_label"] == 0
    page = next((tmp_path / "export/somas").glob("*.parquet"))
    page.write_bytes(page.read_bytes() + b"bad\n")
    with pytest.raises(ValueError, match="changed"):
        import_fish1(tmp_path / "export", tmp_path / "bad")


def test_duplicate_synapses_and_float_ids_cannot_become_connectivity(tmp_path):
    tables = Tables()
    tables.data["synapses_axde"][2]["id"] = tables.data["synapses_axde"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate Fish1"):
        export_fish1(
            client_for(tables),
            Fish1ExportSpec(574, "fixture"),
            tmp_path / "duplicate",
        )
    for value in (float(2**63 + 5), True, "1.0", "-1", str(2**64)):
        with pytest.raises(ValueError):
            integer_id(value)
    with pytest.raises(ValueError, match="fixed materialization"):
        Fish1ExportSpec("latest", "unknown").validate()
    for value in (float(-(2**62)), True, "1.0", "+1", "-", str(2**63), str(-(2**63) - 1)):
        with pytest.raises(ValueError):
            annotation_id(value)
    assert annotation_id(-(2**63)) == str(-(2**63))
    assert annotation_id(str(2**63 - 1)) == str(2**63 - 1)


def test_reference_labels_never_multiply_or_remove_physical_connectivity(tmp_path):
    tables = Tables()
    labels = tables.data["synapses_axde_label"]
    labels.pop(0)
    labels.extend(
        [
            {"id": -10, "target_id": labels[0]["target_id"], "tag": "2"},
            {"id": -11, "target_id": -12, "tag": "1"},
        ]
    )
    export_fish1(client_for(tables), Fish1ExportSpec(574, "fixture"), tmp_path / "export")
    graph = import_fish1(tmp_path / "export", tmp_path / "graph")
    assert graph.m == 4 and graph.weight.sum() == 5
    audit = graph.manifest["provenance"]["membership_audit"]
    assert audit["retained_synapses_without_label"] == 1
    assert audit["physical_synapses_with_multiple_labels"] == 1
    assert audit["labels_without_physical_synapse"] == 1


def test_count_reply_cannot_be_mistaken_for_an_annotation(tmp_path):
    tables = Tables()
    query = tables.query_table

    def malformed_count(table, **kwargs):
        result = query(table, **kwargs)
        return [{"id": 1}] if kwargs.get("get_counts") else result

    tables.query_table = malformed_count
    with pytest.raises(ValueError, match="count response"):
        export_fish1(client_for(tables), Fish1ExportSpec(574, "fixture"), tmp_path / "export")
    assert not list((tmp_path / "export").rglob("*.parquet"))


def test_legacy_csv_export_remains_importable(tmp_path):
    tables = Tables()
    provenance = export_fish1(client_for(tables), Fish1ExportSpec(574, "fixture"), tmp_path / "v2")
    legacy = tmp_path / "v1"
    legacy.mkdir()
    tags = {row["target_id"]: row["tag"] for row in tables.data["synapses_axde_label"]}
    joined = [{**row, "tag": tags[row["id"]]} for row in tables.data["synapses_axde"]]
    provenance.update(
        schema="fish1-versioned-export-v1", synapse_table="synapses_axde_label", files={}, tables={}
    )
    for name, rows in (("somas", tables.data["somas"]), ("synapses_axde_label", joined)):
        path = legacy / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        provenance["files"][path.name] = {
            "sha256": digest_file(path),
            "rows": len(rows),
            "offset": 0,
            "table": name,
        }
        provenance["tables"][name] = {"expected_rows": len(rows), "exported_rows": len(rows)}
    (legacy / "provenance.json").write_text(json.dumps(provenance))
    graph = import_fish1(legacy, tmp_path / "graph")
    assert graph.n == 5 and graph.m == 4 and graph.weight.sum() == 5
    assert graph.manifest["provenance"]["membership_audit"]["retained_synapses_without_label"] == 0


def test_export_checks_the_actual_segmentation_endpoint_and_does_not_store_url_credentials(
    tmp_path,
):
    tables = Tables()
    with pytest.raises(ValueError, match="segmentation source"):
        export_fish1(
            client_for(tables, "graphene://https://example.test/table/other_release"),
            Fish1ExportSpec(574, "fixture"),
            tmp_path / "wrong",
        )
    assert not tables.calls
    result = export_fish1(
        client_for(
            tables,
            "graphene://https://user:private@example.test/table/fish1_v250915?token=private#private",
        ),
        Fish1ExportSpec(574, "fixture"),
        tmp_path / "correct",
    )
    assert result["segmentation_source"] == "https://example.test/table/fish1_v250915"
    assert "private" not in json.dumps(result)
