"""Isolated sparse recurrent throughput; no simulator or learned controller."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .config import BodyConfig, DynamicsConfig, RunConfig
from .graphs import Graph
from .policy import FrozenConnectome
from .util import atomic_json, digest_file, seed_everything


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark(
    graph_path: Path,
    output: Path,
    *,
    device="cuda",
    batches=(1, 4, 16),
    lengths=(8, 32, 64),
    repeats=3,
    warmup=1,
    threads=1,
    seed=0,
    channels=16,
):
    if output.exists():
        raise FileExistsError(f"Use a fresh benchmark destination: {output}")
    if device not in ("cpu", "cuda"):
        raise ValueError("The sparse benchmark supports explicit cpu or cuda execution")
    if min(*batches, *lengths, repeats, threads, channels) < 1 or warmup < 0:
        raise ValueError("Positive counts and nonnegative warmup are required")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing a CPU substitution")
    seed_everything(seed, threads)
    graph = Graph.load(graph_path)
    config = RunConfig(
        graph=str(graph_path.resolve()),
        device=device,
        substrate_seed=seed,
        body=BodyConfig(),
        dynamics=DynamicsConfig(channels=channels),
    )
    started = time.perf_counter()
    brain = FrozenConnectome(graph, config).to(device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    source = Path(__file__).parent
    report = {
        "schema": "connectome-sparse-benchmark-v1",
        "scope": "Sparse frozen recurrence and fixed ports; no MuJoCo or adapter optimization",
        "graph_fingerprint": graph.fingerprint,
        "neurons": graph.n,
        "edges": graph.m,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "threads": threads,
        "dtype": "float32",
        "seed": seed,
        "repeats": repeats,
        "warmup": warmup,
        "substeps_per_decision": brain.substeps,
        "config": config.to_dict(),
        "setup_seconds": setup_seconds,
        "source_sha256": {
            name: digest_file(source / name)
            for name in ("sparse_benchmark.py", "policy.py", "graphs.py", "config.py", "util.py")
        },
        "transition_definition": "one recurrent substep for one batch member",
        "memory_definition": "CUDA process peak allocated/reserved bytes including fixed matrices; null on CPU",
        "rows": [],
    }
    for batch in batches:
        for length in lengths:
            row = {"batch": batch, "bptt_length": length}
            try:
                measurements = []
                for trial in range(warmup + repeats):
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    inputs = torch.randn(length, batch, channels, device=device) * 0.1
                    state = torch.zeros(batch, graph.n, device=device)
                    synchronize(device)
                    start = time.perf_counter()
                    with torch.no_grad():
                        for z in inputs:
                            _, state = brain(z, state)
                    synchronize(device)
                    inference_seconds = time.perf_counter() - start
                    inputs = inputs.detach().requires_grad_()
                    state = torch.zeros(batch, graph.n, device=device, requires_grad=True)
                    synchronize(device)
                    start = time.perf_counter()
                    loss = inputs.new_zeros(())
                    for z in inputs:
                        readout, state = brain(z, state)
                        loss = loss + readout.square().mean() / length
                    synchronize(device)
                    forward_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    loss.backward()
                    synchronize(device)
                    backward_seconds = time.perf_counter() - start
                    finite = bool(torch.isfinite(state).all() and torch.isfinite(inputs.grad).all())
                    gradient = float(inputs.grad.norm())
                    if not finite or gradient == 0:
                        raise FloatingPointError("Nonfinite dynamics or absent input gradient")
                    measurement = {
                        "inference_seconds": inference_seconds,
                        "forward_seconds": forward_seconds,
                        "backward_seconds": backward_seconds,
                        "input_gradient_norm": gradient,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                        if device == "cuda"
                        else None,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
                        if device == "cuda"
                        else None,
                    }
                    if trial >= warmup:
                        measurements.append(measurement)
                    del inputs, state, loss, readout
                row.update(status="passed", samples=measurements)
                transitions = batch * length * brain.substeps
                for name in ("inference", "forward", "backward"):
                    seconds = statistics.median(m[f"{name}_seconds"] for m in measurements)
                    row[f"{name}_seconds"] = seconds
                    row[f"{name}_transitions_per_second"] = transitions / seconds
                row["train_transitions_per_second"] = transitions / (
                    row["forward_seconds"] + row["backward_seconds"]
                )
                for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
                    row[name] = max(m[name] for m in measurements) if device == "cuda" else None
            except torch.cuda.OutOfMemoryError as exc:
                row.update(status="out_of_memory", error=str(exc))
                gc.collect()
                torch.cuda.empty_cache()
            report["rows"].append(row)
            atomic_json(output, report)
            print(json.dumps(row), flush=True)
    report["status"] = (
        "complete" if all(r["status"] == "passed" for r in report["rows"]) else "partial"
    )
    atomic_json(output, report)
    table = [
        "# Sparse recurrent benchmark",
        "",
        f"Device: **{device}**. Graph: {graph.n:,} neurons / {graph.m:,} edges. Float32.",
        "No CUDA result is implied by a CPU run. Rates count individual recurrent substeps across the batch.",
        "",
        "| Batch | BPTT | Inference transitions/s | Training forward/s | Backward/s | Combined training/s | Peak VRAM GiB | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        cells = [str(row["batch"]), str(row["bptt_length"])]
        for name in ("inference", "forward", "backward", "train"):
            value = row.get(f"{name}_transitions_per_second")
            cells.append(f"{value:.2f}" if value is not None else "—")
        memory = row.get("peak_allocated_bytes")
        cells += [f"{memory / 2**30:.3f}" if memory is not None else "n/a", row["status"]]
        table.append("| " + " | ".join(cells) + " |")
    output.with_suffix(".md").write_text("\n".join(table) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--lengths", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    args = vars(parser.parse_args(argv))
    args["graph_path"] = args.pop("graph")
    benchmark(**args)


if __name__ == "__main__":
    main()
