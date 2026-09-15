"""Full-graph construction and gradient checks for the actual learned-port family."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from connectome_body.adaptation.imitation import source_identity
from connectome_body.adaptation.models import GenericAdapter
from connectome_body.adaptation.substrates import RateConfig
from connectome_body.graphs import Graph
from connectome_body.util import atomic_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--variants", nargs="+", default=["real", "degree_shuffled", "matched_random"]
    )
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Interface-check reports are immutable")
    if args.steps < 2 or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Need >=2 steps and an available explicit device")
    graph = Graph.load(args.graph)
    report = {
        "schema": "connectome-structural-interface-check-v1",
        "evidence": "engineering_only",
        "graph_fingerprint": graph.fingerprint,
        "source_identity": source_identity(),
        "device": args.device,
        "neurons": graph.n,
        "edges": graph.m,
        "rows": [],
    }
    for variant in args.variants:
        seed_everything(0, 1)
        start = time.perf_counter()
        actor = GenericAdapter(
            104, 12, 5000, 16, 256, graph, variant, RateConfig(control_dt=0.0002, tau_seconds=0.002)
        ).to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        setup = time.perf_counter() - start
        start = time.perf_counter()
        context, state = actor.context(), actor.reset(1)
        for _ in range(args.steps):
            action, state = actor(torch.randn(1, 104, device=args.device), state, context)
        action.square().mean().backward()
        if args.device == "cuda":
            torch.cuda.synchronize()
        gradients = {name: float(value.grad.norm()) for name, value in actor.named_parameters()}
        if not all(torch.isfinite(value.grad).all() for value in actor.parameters()):
            raise FloatingPointError("Nonfinite learned-interface gradient")
        for name in ("ports.input_queries", "ports.output_queries", "encoder.0.weight"):
            if name in gradients and gradients[name] <= 0:
                raise AssertionError(f"No gradient reached {name}")
        row = {
            "variant": variant,
            "status": "passed",
            "parameters": actor.parameter_report,
            "construction_seconds": setup,
            "forward_backward_seconds": time.perf_counter() - start,
            "bptt_length": args.steps,
            "batch": 1,
            "gradients": gradients,
            "substrate": actor.substrate.report if actor.substrate is not None else None,
        }
        report["rows"].append(row)
        atomic_json(args.output, report)
        print(
            {"variant": variant, "status": row["status"], "construction_seconds": setup}, flush=True
        )
        del actor, context, state, action
        gc.collect()
    report["status"] = "complete"
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
