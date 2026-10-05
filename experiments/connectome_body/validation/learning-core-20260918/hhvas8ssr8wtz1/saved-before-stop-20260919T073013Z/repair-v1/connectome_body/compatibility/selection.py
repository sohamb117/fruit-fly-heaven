"""Reviewable condition selection from a larger, immutable paper catalog."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from ..util import atomic_json, digest_json


def condition_fields(row):
    config = row["config"]
    adapter, substrate = config["controller"]["adapter"], config["controller"]["substrate"]
    return {
        "connectome": config["metadata"]["connectome"],
        "body": config["body"]["name"],
        "task": config["body"]["task"],
        "family": adapter["family"],
        "channels": adapter["channels"],
        "budget": adapter["budget"],
        "width": adapter.get("width"),
        "depth": adapter.get("depth"),
        "rank": adapter.get("rank"),
        "kind": substrate["kind"],
        "topology": substrate["topology"],
        "initialization": substrate["initialization"],
        "plasticity": substrate["plasticity"],
        "sign_mode": substrate["dynamics"]["sign_mode"],
        "train_seed": config["train_seed"],
        "subgraph_role": config["metadata"].get("subgraph_role"),
        "training_interactions": config["training"]["interactions"],
    }


def select_conditions(plan_path, filters_path, output):
    from .runner import read_plan

    plan, _ = read_plan(plan_path)
    specification = json.loads(Path(filters_path).read_text())
    if specification.get("schema") != "compatibility-selection-filter-v1":
        raise ValueError("Selection filter schema mismatch")
    include_controls = specification.get("include_matched_controls", False)
    if type(include_controls) is not bool:
        raise ValueError("include_matched_controls must be a boolean")
    filters = specification.get("filters", {})
    allowed = condition_fields(plan["conditions"][0])
    if (
        not filters
        or set(filters) - set(allowed)
        or any(not isinstance(values, list) or not values for values in filters.values())
    ):
        raise ValueError("Filters must be nonempty lists over declared condition fields")

    def matches_filters(row):
        fields = condition_fields(row)
        for name, values in filters.items():
            if name == "budget":
                # A capacity-invariant linear map is compiled once and shared
                # by multiple requested ceilings. Keep it in each such study.
                requested = row.get("requested_adapter_budgets", [fields["budget"]])
                if not set(requested).intersection(values):
                    return False
            elif fields[name] not in values:
                return False
        return True

    rows = [row for row in plan["conditions"] if matches_filters(row)]
    if not rows:
        raise ValueError("No planned conditions match this selection")
    direct_ids = {row["id"] for row in rows}
    if include_controls:
        # A trainable RNN divides its TOTAL budget between maps and recurrence.
        # Filtering its map budget against a biological adapter budget silently
        # drops the control. Follow the planner's explicit matching references.
        rows = [
            row
            for row in plan["conditions"]
            if row["id"] in direct_ids
            or any(match["reference"] in direct_ids for match in row.get("matches", []))
        ]
    ready = [row for row in rows if row["status"] == "ready"]
    evaluation_bound = 0
    for row in ready:
        learning = row["config"]["training"]
        episodes = (1 + math.ceil(learning["interactions"] / learning["eval_every"])) * learning[
            "eval_episodes"
        ] + 3 * learning["test_episodes"]
        evaluation_bound += episodes * row["config"]["body"]["horizon"]
    value = {
        "schema": "compatibility-condition-selection-v1",
        "plan_fingerprint": plan["fingerprint"],
        "filter_specification": specification,
        "condition_ids": [row["id"] for row in rows],
        "direct_condition_ids": [row["id"] for row in rows if row["id"] in direct_ids],
        "included_control_ids": [row["id"] for row in rows if row["id"] not in direct_ids],
        "statuses": dict(Counter(row["status"] for row in rows)),
        "ready_training_interactions": sum(
            row["config"]["training"]["interactions"] for row in ready
        ),
        "ready_evaluation_interactions_upper_bound": evaluation_bound,
        "rows": [
            {
                "id": row["id"],
                "status": row["status"],
                **condition_fields(row),
                "selection_reason": "filter" if row["id"] in direct_ids else "matched_control",
                "requested_adapter_budgets": row.get("requested_adapter_budgets", []),
                "matched_reference_ids": sorted(
                    {
                        match["reference"]
                        for match in row.get("matches", [])
                        if match["reference"] in direct_ids
                    }
                ),
            }
            for row in rows
        ],
        "note": "Selection does not launch work. Worker bounds remain mandatory; evaluation can exceed training experience.",
    }
    value["fingerprint"] = digest_json(value)
    target = Path(output)
    if target.exists() and json.loads(target.read_text()) != value:
        raise FileExistsError("Keep the existing selection and choose a new output")
    atomic_json(target, value)
    return value


def selection_ids(plan, path):
    value = json.loads(Path(path).read_text())
    if (
        value.get("schema") != "compatibility-condition-selection-v1"
        or value.get("plan_fingerprint") != plan["fingerprint"]
        or value.get("fingerprint")
        != digest_json({k: v for k, v in value.items() if k != "fingerprint"})
    ):
        raise ValueError("Condition selection belongs to another plan or has changed")
    return value["condition_ids"]
