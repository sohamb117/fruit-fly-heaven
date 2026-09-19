"""Generate a recorded device override without editing a study or an existing plan."""

import argparse
import json
from pathlib import Path

from connectome_body.body import PROJECT
from connectome_body.suite import make_plan

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study", type=Path, default=PROJECT / "configs/pilot.json")
parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
parser.add_argument("--graph-root", type=Path, default=PROJECT / "data/graphs")
parser.add_argument("--datasets", nargs="+")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
spec = json.loads(args.study.read_text())
spec.setdefault("base", {})["device"] = args.device
plan = make_plan(spec, args.graph_root, args.output, args.datasets)
print(
    json.dumps(
        {
            key: plan[key]
            for key in (
                "status",
                "fingerprint",
                "planned_runs",
                "ready_runs",
                "training_interactions",
                "missing_datasets",
            )
        },
        indent=2,
    )
)
