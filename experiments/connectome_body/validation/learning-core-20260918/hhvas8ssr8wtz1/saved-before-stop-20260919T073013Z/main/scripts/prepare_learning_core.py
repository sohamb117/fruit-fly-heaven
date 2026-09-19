"""Prepare deterministic graph inputs on CPU before a paid qualification."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from connectome_body.compatibility.config import ControllerConfig
    from connectome_body.compatibility.learning_study import read_plan
    from connectome_body.compatibility.preparation import preparation_key, prepare_substrate
    from connectome_body.graphs import Graph
    from connectome_body.util import atomic_json, digest_json
    from scripts.qualify_learning_core import cases

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--all-seeds", action="store_true")
    a = p.parse_args()
    plan = read_plan(a.plan)
    rows = (
        [r for r in plan["conditions"] if r["regime"] == "bc_ppo"] if a.all_seeds else cases(plan)
    )
    graphs, seen, records = {}, set(), []
    started = time.monotonic()
    for row in rows:
        config = ControllerConfig.from_dict(row["config"]["controller"]).substrate
        if config.kind != "connectome":
            continue
        if config.graph not in graphs:
            graphs[config.graph] = Graph.load(config.graph)
        graph = graphs[config.graph]
        key = digest_json(preparation_key(graph, config))
        if key in seen:
            continue
        seen.add(key)
        result = prepare_substrate(graph, config)
        records.append({"graph": graph.fingerprint, **result.cache_report})
        atomic_json(
            a.output,
            {
                "plan": plan["fingerprint"],
                "entries": records,
                "complete": False,
                "wall_seconds": time.monotonic() - started,
            },
        )
        print(json.dumps(records[-1]), flush=True)
    atomic_json(
        a.output,
        {
            "plan": plan["fingerprint"],
            "entries": records,
            "complete": True,
            "wall_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
