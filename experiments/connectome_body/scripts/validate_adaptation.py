"""Exercise native hover, imitation, DAgger, lesions, and analysis on a tiny graph.

Every output is explicitly engineering evidence. Optionally exercise full BANC
with the same native interface, using a short horizon and one offline stage.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from connectome_body.adaptation.analysis import analyze
from connectome_body.adaptation.collection import collect_resumable
from connectome_body.adaptation.hover import HoverConfig
from connectome_body.adaptation.pipeline import run_experiment
from connectome_body.body import PROJECT
from connectome_body.graphs import make_fixture
from connectome_body.util import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-banc", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError("Use a fresh validation directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    graph = root / "fixture"
    if not (graph / "manifest.json").exists():
        make_fixture(graph, 128, 13)
    body = HoverConfig(horizon=32)
    data = {}
    for split, episodes in (("train", 8), ("validation", 4), ("test", 4)):
        path = root / "data" / split
        cache = collect_resumable(
            body,
            path,
            split=split,
            episodes=episodes,
            smoke=True,
            resume=(path / "collection.json").exists(),
        )
        data[split] = str(cache.path)
    opt = dict(
        updates=4,
        batch_size=2,
        sequence_length=8,
        burn_in=8,
        eval_every=2,
        checkpoint_every=1,
        learning_rate=0.001,
        max_grad_norm=1.0,
    )
    records = []
    for variant in ("real", "degree_shuffled", "matched_random", "adapter_only", "trainable_gru"):
        output = root / "runs" / variant
        config = {
            "smoke": True,
            "dataset": "validation_fixture"
            if variant not in ("adapter_only", "trainable_gru")
            else "baseline",
            "body": dataclasses.asdict(body),
            "data": data,
            "adapter": {
                "graph": str(graph) if variant not in ("adapter_only", "trainable_gru") else None,
                "variant": variant,
                "budget": 5000,
                "channels": 16,
                "support": 8,
                "seed": 0,
                "port_seed": 0,
                "device": args.device,
                "threads": 1,
                "rate": {"control_dt": body.control_dt, "tau_seconds": 0.002, "substeps": 2},
            },
            "offline": opt,
            "dagger_optimization": dict(opt, updates=2),
            "online_budgets": [0, 16, 32],
            "dagger_beta": 0.0,
            "validation_episodes": 2,
            "test_episodes": 2,
            "success_threshold": 0.8,
            "interventions": True,
        }
        atomic_json(root / "configs" / f"{variant}.json", config)
        result = run_experiment(config, output, resume=(output / "manifest.json").exists())
        records.append(
            {
                "variant": variant,
                "status": result["status"],
                "evidence": result["evidence"],
                "actual_parameters": result["parameters"]["actual"],
                "online_interactions": result["online_interactions"],
                "result": str(output / "result.json"),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    if args.full_banc:
        config = json.loads((root / "configs/real.json").read_text())
        config["dataset"] = "banc"
        config["adapter"]["graph"] = str(PROJECT / "data/graphs/banc")
        config["adapter"]["support"] = 256
        config["offline"] = dict(opt, updates=2, batch_size=2, sequence_length=4, burn_in=4)
        config["online_budgets"] = [0]
        config["interventions"] = False
        output = root / "full-banc"
        atomic_json(root / "configs/full-banc.json", config)
        result = run_experiment(config, output, resume=(output / "manifest.json").exists())
        records.append(
            {
                "variant": "full-banc",
                "status": result["status"],
                "evidence": result["evidence"],
                "actual_parameters": result["parameters"]["actual"],
                "graph": result["graph"],
                "result": str(output / "result.json"),
            }
        )
        print(json.dumps({k: v for k, v in records[-1].items() if k != "graph"}), flush=True)
    summary = analyze(root / "runs", root / "analysis", include_validation=True)
    assert summary["completed_runs"] == 5
    assert not json.loads((root / "analysis/decision.json").read_text())["go"]
    atomic_json(
        root / "validation-summary.json",
        {"evidence": "native_hover_engineering_only", "runs": records},
    )


if __name__ == "__main__":
    main()
