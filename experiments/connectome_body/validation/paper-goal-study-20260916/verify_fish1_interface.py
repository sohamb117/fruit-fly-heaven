"""Validate the newly imported Fish1 graph on three native body interfaces.

Run from experiments/connectome_body with uv run --no-sync. This performs two
physical transitions and one backward pass per case, without optimizer steps.
It checks input integration, not learned control or a biological advantage.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from connectome_body.compatibility.bodies import BodySpec, make_body
from connectome_body.compatibility.config import (
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.controller import Controller
from connectome_body.compatibility.training import source_identity
from connectome_body.graphs import Graph
from connectome_body.util import atomic_json, digest_file, digest_json


def main():
    output = Path(__file__).with_name("fish1-interface-check.json")
    if output.exists():
        raise FileExistsError("Preserve the previous integration record")
    torch.set_num_threads(1)
    study = json.loads(Path("configs/paper/study.json").read_text())
    graph_path = study["graphs"]["fish1"]["path"]
    graph = Graph.load(graph_path)
    record = {
        "schema": "fish1-native-interface-check-v1",
        "scope": "finite native transitions and sparse gradients; no learning result",
        "goal_sha256": digest_file(Path("../GOAL.md")),
        "code_fingerprint": source_identity(),
        "graph_fingerprint": graph.fingerprint,
        "neurons": graph.n,
        "edges": graph.m,
        "device": "cpu",
        "optimizer_updates": 0,
        "cases": [],
    }
    for key in ("fly:hover", "worm:locomotion", "fish:swimming"):
        declaration = study["bodies"][key]
        body = make_body(BodySpec(**declaration["spec"]))
        try:
            for plasticity in ("adapters", "joint"):
                started = time.monotonic()
                config = ControllerConfig(
                    adapter=AdapterConfig(**study["adapter"]),
                    substrate=SubstrateConfig(
                        graph=graph_path,
                        plasticity=plasticity,
                        dynamics=DynamicsConfig(
                            **study["dynamics"], control_dt=declaration["control_dt"]
                        ),
                    ),
                )
                controller = Controller(body.obs_dim, body.action_dim, config, graph)
                observation = body.reset(0)
                state, context = controller.reset(1), controller.context()
                loss = torch.zeros(())
                for _ in range(2):
                    action, state = controller(
                        torch.as_tensor(observation).unsqueeze(0), state, context
                    )
                    assert torch.isfinite(action).all() and torch.isfinite(state).all()
                    observation, reward, terminated, truncated, _ = body.step(
                        action.detach().numpy()[0]
                    )
                    assert np.isfinite(observation).all() and np.isfinite(reward)
                    assert not terminated and not truncated
                    loss = loss + (action - 0.2).square().mean()
                loss.backward()
                gradients = {}
                for name, module in (
                    ("encoder", controller.encoder),
                    ("decoder", controller.decoder),
                    ("ports", controller.ports),
                    ("substrate", controller.core),
                ):
                    active = [p for p in module.parameters() if p.requires_grad]
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                    norm = sum(float(p.grad.detach().abs().sum()) for p in active)
                    if active:
                        assert norm > 0
                    gradients[name] = {"trainable_tensors": len(active), "gradient_l1": norm}
                case = {
                    "body_task": key,
                    "body_fingerprint": body.fingerprint,
                    "evidence": body.evidence,
                    "plasticity": plasticity,
                    "observation_dimension": body.obs_dim,
                    "action_dimension": body.action_dim,
                    "physical_transitions": 2,
                    "parameters": controller.parameter_report,
                    "gradients": gradients,
                    "elapsed_seconds": time.monotonic() - started,
                    "passed": True,
                }
                record["cases"].append(case)
                print(
                    json.dumps(
                        {
                            k: case[k]
                            for k in ("body_task", "plasticity", "passed", "elapsed_seconds")
                        }
                    ),
                    flush=True,
                )
                del controller, state, context, action, loss, module, active
                gc.collect()
        finally:
            body.close()
    record["passed"] = len(record["cases"]) == 6
    record["fingerprint"] = digest_json(record)
    atomic_json(output, record)


if __name__ == "__main__":
    main()
