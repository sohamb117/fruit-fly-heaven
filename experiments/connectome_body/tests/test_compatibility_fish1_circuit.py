import pytest

from connectome_body.compatibility.fish1_circuit import RECONSTRUCTED_LABELS, hmi_adjacency


def rows():
    result = [
        {
            "Cell ID": float(i),
            "reconstruction_status": RECONSTRUCTED_LABELS[0],
            "inputs": "n/a",
            "outputs": "n/a",
            "final_neurotransmitter_ID": "VGluT2",
        }
        for i in range(1, 5)
    ]
    result[0]["outputs"] = "[(2, 1, 2, 3), (2, 4, 5, 6)]"
    result[1]["inputs"] = "[(1, 1, 2, 3)]"
    result[1]["outputs"] = (
        "[(3, 5, 6, 7), ('-', 1, 1, 1), (' ', 1, 1, 1), ('NO ID???', 1, 1, 1), (99, 1, 1, 1)]"
    )
    result[2]["final_neurotransmitter_ID"] = "Gad1B"
    result[3]["final_neurotransmitter_ID"] = "unlabeled"
    return result


def test_binary_union_preserves_direction_without_double_counting_contacts():
    nodes, pairs, signs, selected, audit = hmi_adjacency(rows())
    assert nodes == ["1", "2", "3", "4"]  # keep the selected isolate
    assert pairs == [(0, 1), (1, 2)]
    assert signs.tolist() == [1, 1, -1, 0]
    assert set(selected) == set(nodes)
    assert audit["retained_contact_annotations"] == 4
    assert audit["unique_directed_pairs"] == 2
    assert audit["unidentified_partner_annotations"] == 3
    assert audit["outside_membership_annotations"] == 1


@pytest.mark.parametrize("invalid_id", [1.5, float(2**60), True])
def test_unsafe_numeric_soma_ids_are_rejected(invalid_id):
    data = rows()
    data[0]["Cell ID"] = invalid_id
    with pytest.raises(ValueError, match="ID"):
        hmi_adjacency(data)


def test_only_declared_reconstruction_membership_is_selected():
    data = rows()
    data.append(
        {
            "Cell ID": 99,
            "reconstruction_status": "no status assigned",
            "inputs": "n/a",
            "outputs": "n/a",
        }
    )
    nodes, pairs, _, _, audit = hmi_adjacency(data)
    assert "99" not in nodes and len(pairs) == 2
    assert audit["catalog_cells"] == 5 and audit["selected_cells"] == 4


def test_malformed_contacts_and_duplicate_soma_ids_are_not_silently_dropped():
    data = rows()
    data[0]["outputs"] = "[(2,)]"
    with pytest.raises(ValueError, match="contact tuple"):
        hmi_adjacency(data)
    data = rows()
    data.append(dict(data[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        hmi_adjacency(data)
