"""Import immutable actor weights across runtimes, without claiming exact resume.

Checks the original source-file identity, parent checkpoint, graph, architecture,
and current CPU/CUDA numerical parity. The original optimizer and RNG are not
transferred. The resulting inference artifact explicitly records its provenance.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def import_weights(checkpoint, runtime_path, output, expected_sha256):
    import torch

    from connectome_body.adaptation.imitation import AdapterSpec, build_adapter, source_identity
    from connectome_body.util import atomic_json, atomic_torch_save, digest_file, digest_json

    if output.exists():
        raise FileExistsError("Runtime imports are immutable")
    if digest_file(checkpoint) != expected_sha256:
        raise ValueError("Original checkpoint bytes changed")
    parent = json.loads((checkpoint.parent / "manifest.json").read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        saved["identity"] != parent["identity"]
        or digest_json({k: v for k, v in parent.items() if k != "identity"}) != parent["identity"]
    ):
        raise ValueError("Original checkpoint identity is invalid")
    old_runtime = json.loads(runtime_path.read_text())
    if digest_json(old_runtime) != parent["source_identity"]:
        raise ValueError("Original source/runtime record does not match its manifest")
    base = ROOT / "connectome_body/adaptation"
    current_files = {name: digest_file(base / name) for name in old_runtime["files"]}
    if current_files != old_runtime["files"]:
        raise ValueError("Controller source changed; this only permits a runtime change")
    spec = AdapterSpec.from_dict(parent["adapter"])
    actor, graph = build_adapter(
        dataclasses.replace(spec, device="cpu"),
        saved["actor"]["encoder.0.weight"].shape[1],
        saved["actor"]["decoder.2.bias"].shape[0],
    )
    if actor.parameter_report != saved["parameter_report"]:
        raise ValueError("Imported actor architecture differs")
    if graph.fingerprint != parent["graph_fingerprint"]:
        raise ValueError("Imported graph differs")
    actor.load_state_dict(saved["actor"], strict=True)
    for name, value in actor.state_dict().items():
        if not torch.isfinite(value).all() or not torch.equal(value, saved["actor"][name]):
            raise ValueError("Imported actor weights differ or are nonfinite")
    observations = torch.randn(4, 2, actor.obs_dim, generator=torch.Generator().manual_seed(7321))

    def probe(device, fixed_nodes=None):
        from connectome_body.adaptation.ports import PortMap

        actor.to(device)
        actor.zero_grad(set_to_none=True)
        context = actor.context()
        if fixed_nodes is not None:
            # Top-k can choose different members of tied-score populations on
            # CPU and CUDA. Isolate recurrence parity using one shared support;
            # production training still uses the unchanged native CUDA allocator.
            selected = []
            for queries, nodes in zip(
                (actor.ports.input_queries, actor.ports.output_queries), fixed_nodes, strict=True
            ):
                nodes = nodes.to(device)
                scores = (queries[:, None, :] * actor.ports.features[nodes]).sum(-1)
                scores = scores / math.sqrt(actor.ports.features.shape[1])
                weights = torch.softmax(scores / actor.ports.temperature, dim=-1)
                weights = weights / weights.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
                selected.extend((nodes, weights))
            context = PortMap(*selected)
        state, actions = actor.reset(2), []
        for observation in observations:
            action, state = actor(observation.to(device), state, context)
            actions.append(action)
        prediction = torch.stack(actions)
        prediction.square().mean().backward()
        gradient = torch.cat([p.grad.detach().flatten().cpu() for p in actor.parameters()])
        return (
            prediction.detach().cpu(),
            state.detach().cpu(),
            gradient,
            context.input_nodes.detach().cpu(),
            context.output_nodes.detach().cpu(),
        )

    native_cpu, native_cuda = probe("cpu"), probe("cuda")
    support = native_cuda[3:]
    cpu, cuda = probe("cpu", support), probe("cuda", support)
    for expected, actual in zip(cpu[:3], cuda[:3], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    if not torch.isfinite(cuda[2]).all() or cuda[2].norm().item() <= 0:
        raise ValueError("Imported actor has invalid gradients")
    parity = {
        "status": "passed",
        "scope": "CPU/CUDA recurrence with identical port supports; not cross-version equivalence",
        "native_port_supports_differ": any(
            not torch.equal(a.sort(-1).values, b.sort(-1).values)
            for a, b in zip(native_cpu[3:], native_cuda[3:], strict=True)
        ),
        "native_allocator_action_max_abs_error": float(
            (native_cpu[0] - native_cuda[0]).abs().max()
        ),
        "native_allocator_state_max_abs_error": float((native_cpu[1] - native_cuda[1]).abs().max()),
        "training_allocator": "unchanged_native_cuda_topk",
        "action_max_abs_error": float((cpu[0] - cuda[0]).abs().max()),
        "state_max_abs_error": float((cpu[1] - cuda[1]).abs().max()),
        "gradient_max_abs_error": float((cpu[2] - cuda[2]).abs().max()),
        "gradient_relative_l2_error": float((cpu[2] - cuda[2]).norm() / cpu[2].norm()),
        "gradient_norm": float(cuda[2].norm()),
    }
    manifest = {
        "artifact_kind": "weights_only_runtime_import_not_training_resume",
        "source_identity": source_identity(),
        "adapter": parent["adapter"],
        "graph_fingerprint": graph.fingerprint,
        "parameter_report": actor.parameter_report,
        "data_metadata": parent["data_metadata"],
        "origin": {
            "checkpoint_sha256": expected_sha256,
            "identity": parent["identity"],
            "source_identity": parent["source_identity"],
            "updates": saved["updates"],
            "runtime": old_runtime,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "numba")},
        },
        "validation": parity,
        "weights_preserved_bitwise": True,
        "optimizer_and_rng_transferred": False,
    }
    manifest["identity"] = digest_json(manifest)
    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    atomic_torch_save(
        output / "weights.pt",
        {
            "identity": manifest["identity"],
            "actor": saved["actor"],
            "updates": 0,
            "parameter_report": actor.parameter_report,
        },
    )
    atomic_json(output / "validation.json", parity)
    print(
        json.dumps(
            {
                "status": "imported",
                "original_updates": saved["updates"],
                "identity": manifest["identity"],
                "parity": parity,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    import_weights(args.checkpoint, args.runtime, args.output, args.expected_sha256)


if __name__ == "__main__":
    main()
