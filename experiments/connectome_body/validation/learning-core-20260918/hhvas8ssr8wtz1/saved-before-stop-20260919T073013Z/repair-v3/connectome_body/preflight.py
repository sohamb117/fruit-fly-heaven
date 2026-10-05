"""Real graph/body forward-backward checks before committing a training budget."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .body import make_env
from .config import RunConfig
from .graphs import Graph
from .policy import BASELINES, Actor
from .util import atomic_json, seed_everything


def preflight(config: RunConfig, output: Path, steps=8):
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    seed_everything(config.train_seed, config.threads)
    graph = None if config.substrate in BASELINES else Graph.load(config.graph)
    env = make_env(config.body)
    try:
        start = time.monotonic()
        actor = Actor(env.obs_dim, env.action_dim, config, graph).to(config.device)
        setup_seconds = time.monotonic() - start
        obs = env.reset(config.train_seed)
        state = actor.initial_state(1)
        means, resets, body_seconds = [], 0, 0.0
        start = time.monotonic()
        for i in range(steps):
            mean, _, state = actor(torch.as_tensor(obs, device=config.device).unsqueeze(0), state)
            if not torch.isfinite(state).all() or not torch.isfinite(mean).all():
                raise FloatingPointError("Nonfinite preflight dynamics")
            means.append(mean)
            body_start = time.monotonic()
            obs, _, terminated, truncated, _ = env.step(torch.tanh(mean)[0].detach().cpu().numpy())
            body_seconds += time.monotonic() - body_start
            if terminated or truncated:
                resets += 1
                obs = env.reset(config.train_seed + i + 1)
                state = actor.initial_state(1)
        forward_seconds = time.monotonic() - start
        loss = torch.stack(means).square().mean()
        start = time.monotonic()
        loss.backward()
        if config.device == "cuda":
            torch.cuda.synchronize()
        backward_seconds = time.monotonic() - start
        gradients = {
            name: float(p.grad.norm()) for name, p in actor.named_parameters() if p.grad is not None
        }
        encoder_gradient = (
            sum(v * v for k, v in gradients.items() if k.startswith("encoder")) ** 0.5
        )
        if (
            config.substrate in ("real", "degree_shuffled", "matched_random")
            and encoder_gradient == 0
        ):
            raise RuntimeError("No action gradient reaches the encoder through this graph")
        report = {
            "schema": "connectome-body-preflight-v1",
            "status": "passed",
            "scope": "finite forward/backward and native body integration; not learned task performance",
            "graph": graph.manifest if graph else None,
            "body_backend": config.body.backend,
            "body_fingerprint": env.fingerprint,
            "observation_dim": env.obs_dim,
            "action_dim": env.action_dim,
            "actor_parameters": actor.num_parameters,
            "parameter_budget": config.adapter_budget,
            "learned_connectome_parameters": 0,
            "batch_size": 1,
            "steps": steps,
            "body_resets": resets,
            "setup_seconds": setup_seconds,
            "forward_seconds": forward_seconds,
            "body_seconds": body_seconds,
            "backward_seconds": backward_seconds,
            "encoder_gradient_norm": encoder_gradient,
            "state_rms": float(state.detach().square().mean().sqrt()) if state.numel() else 0.0,
            "state_saturation_fraction": float((state.detach().abs() > 0.95).float().mean())
            if state.numel()
            else 0.0,
            "action_rms": float(torch.stack(means).detach().square().mean().sqrt()),
            "config": config.to_dict(),
        }
        if actor.brain:
            report["ports_fingerprint"] = actor.brain.ports.fingerprint
            report["null"] = actor.brain.null_report
        atomic_json(output, report)
        return report
    finally:
        env.close()
