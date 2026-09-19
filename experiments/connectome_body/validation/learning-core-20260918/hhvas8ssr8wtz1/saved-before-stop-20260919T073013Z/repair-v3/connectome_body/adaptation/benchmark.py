"""Full learned-port adapter throughput on CPU, CUDA, or a local Apple GPU."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import platform
import resource
import statistics
import time
from pathlib import Path

import torch

from ..util import atomic_json, digest_file
from .imitation import AdapterSpec, build_adapter, source_identity
from .substrates import RateConfig, RateSubstrate


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def allocated(device):
    if device == "cuda":
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    if device == "mps":
        return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    return None, None


def empty_cache(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def verify_rate_parity(actor, graph, device):
    if device == "cpu":
        return {"status": "reference_cpu"}
    config = actor.substrate.config
    cpu = RateSubstrate(graph, "real", config, 0)
    torch.manual_seed(810)
    a = (torch.randn(2, graph.n) * 0.1).requires_grad_()
    b = a.detach().to(device).requires_grad_()
    initial = torch.randn(2, graph.n) * 0.1
    x, y = initial, initial.to(device)
    for _ in range(4):
        x, y = cpu.step(x, a), actor.substrate.step(y, b)
    x.square().mean().backward()
    y.square().mean().backward()
    torch.testing.assert_close(y.cpu(), x, rtol=2e-5, atol=3e-6)
    error = float((b.grad.cpu() - a.grad).norm() / a.grad.norm())
    if error > 1e-4 or not torch.isfinite(b.grad).all() or b.grad.norm() == 0:
        raise AssertionError("Accelerated recurrent gradient differs from CPU")
    return {
        "status": "passed",
        "max_state_error": float((y.cpu() - x).abs().max().detach()),
        "relative_input_gradient_error": error,
        "batch": 2,
        "decisions": 4,
    }


def benchmark(
    graph_path,
    output,
    *,
    device="mps",
    batches=(1, 4, 16),
    lengths=(8, 32, 64),
    repeats=3,
    warmup=1,
    threads=1,
    budget=5000,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Benchmark reports are immutable")
    if min(*batches, *lengths, repeats, threads) < 1 or warmup < 0:
        raise ValueError("Positive benchmark counts required")
    spec = AdapterSpec(
        graph=str(Path(graph_path).resolve()),
        budget=budget,
        device=device,
        threads=threads,
        rate=RateConfig(control_dt=0.0002, tau_seconds=0.002),
    )
    started = time.perf_counter()
    actor, graph = build_adapter(spec, 104, 12)
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    parity = verify_rate_parity(actor, graph, device)
    empty_cache(device)
    report = {
        "schema": "connectome-structural-benchmark-v1",
        "scope": "Full stateless E/D, learned structural ports, frozen recurrence; no MuJoCo or optimizer step",
        "evidence": "engineering_benchmark",
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "graph_fingerprint": graph.fingerprint,
        "neurons": graph.n,
        "edges": graph.m,
        "adapter": dataclasses.asdict(spec),
        "parameters": actor.parameter_report,
        "source_identity": source_identity(),
        "benchmark_source_sha256": digest_file(__file__),
        "setup_seconds": setup_seconds,
        "cpu_rate_parity": parity,
        "repeats": repeats,
        "warmup": warmup,
        "threads": threads,
        "transition_definition": "One whole-graph recurrent substep for one batch member",
        "memory_definition": "CUDA instrumented peak; MPS allocated/driver memory sampled after each decision and backward, not an instrumented peak; process RSS high-water is cumulative",
        "rows": [],
    }
    for batch in batches:
        for length in lengths:
            row = {"batch": batch, "bptt_length": length, "status": "passed"}
            samples = []
            for trial in range(warmup + repeats):
                empty_cache(device)
                actor.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                inputs = (torch.randn(length, batch, 104) * 0.1).to(device)
                actor.eval()
                with torch.no_grad():
                    context, state = actor.context(), actor.reset(batch)
                    synchronize(device)
                    start = time.perf_counter()
                    for observation in inputs:
                        _, state = actor(observation, state, context)
                    synchronize(device)
                    inference = time.perf_counter() - start
                del context, state
                actor.train()
                inputs = inputs.detach().requires_grad_()
                synchronize(device)
                start = time.perf_counter()
                context, state = actor.context(), actor.reset(batch)
                loss = inputs.new_zeros(())
                peak_allocated, peak_driver = allocated(device)
                for observation in inputs:
                    action, state = actor(observation, state, context)
                    loss = loss + action.square().mean() / length
                    now, driver = allocated(device)
                    if now is not None:
                        peak_allocated, peak_driver = (
                            max(peak_allocated, now),
                            max(peak_driver, driver),
                        )
                synchronize(device)
                forward = time.perf_counter() - start
                start = time.perf_counter()
                loss.backward()
                synchronize(device)
                backward = time.perf_counter() - start
                now, driver = allocated(device)
                if now is not None:
                    peak_allocated, peak_driver = max(peak_allocated, now), max(peak_driver, driver)
                if device == "cuda":
                    peak_allocated, peak_driver = (
                        torch.cuda.max_memory_allocated(),
                        torch.cuda.max_memory_reserved(),
                    )
                gradient = float(inputs.grad.norm())
                if not torch.isfinite(inputs.grad).all() or gradient <= 0:
                    raise FloatingPointError("No finite gradient through the complete adapter")
                for parameter in actor.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Missing or nonfinite adapter gradient")
                sample = {
                    "inference_seconds": inference,
                    "forward_seconds": forward,
                    "backward_seconds": backward,
                    "input_gradient_norm": gradient,
                    "allocated_bytes": peak_allocated,
                    "driver_or_reserved_bytes": peak_driver,
                    "process_highwater_rss_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * (1 if platform.system() == "Darwin" else 1024),
                }
                if trial >= warmup:
                    samples.append(sample)
                del inputs, context, state, action, loss
            row["samples"] = samples
            transitions = batch * length * spec.rate.substeps
            for phase in ("inference", "forward", "backward"):
                row[f"{phase}_seconds"] = statistics.median(s[f"{phase}_seconds"] for s in samples)
                row[f"{phase}_transitions_per_second"] = transitions / row[f"{phase}_seconds"]
            row["train_transitions_per_second"] = transitions / (
                row["forward_seconds"] + row["backward_seconds"]
            )
            row["allocated_bytes"] = (
                max(s["allocated_bytes"] for s in samples) if device != "cpu" else None
            )
            row["driver_or_reserved_bytes"] = (
                max(s["driver_or_reserved_bytes"] for s in samples) if device != "cpu" else None
            )
            row["process_highwater_rss_bytes"] = max(
                s["process_highwater_rss_bytes"] for s in samples
            )
            report["rows"].append(row)
            atomic_json(output, report)
            print(
                {
                    key: row[key]
                    for key in (
                        "batch",
                        "bptt_length",
                        "status",
                        "train_transitions_per_second",
                        "allocated_bytes",
                    )
                },
                flush=True,
            )
    report["status"] = "complete"
    atomic_json(output, report)
    table = [
        f"# Full BANC structural-adapter benchmark ({device})",
        "",
        report["scope"],
        "",
        report["memory_definition"],
        "",
        "| Batch | BPTT | Inference transitions/s | Forward+backward transitions/s | Allocated GiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        memory = (
            f"{row['allocated_bytes'] / 2**30:.3f}" if row["allocated_bytes"] is not None else "n/a"
        )
        table.append(
            f"| {row['batch']} | {row['bptt_length']} | {row['inference_transitions_per_second']:.2f} | {row['train_transitions_per_second']:.2f} | {memory} |"
        )
    output.with_suffix(".md").write_text("\n".join(table) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = vars(parser.parse_args())
    args["graph_path"] = args.pop("graph")
    benchmark(**args)


if __name__ == "__main__":
    main()
