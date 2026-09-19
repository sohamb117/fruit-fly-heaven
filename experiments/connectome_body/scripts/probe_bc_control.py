"""Private development probes: readout conditioning and student-state action error.

Never changes a live policy or shared demonstrations. Teacher-blended trajectories
are explicitly diagnostic, not autonomous success or scientific evaluation.
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectome_body.compatibility.bodies import make_body
from connectome_body.compatibility.controller import Controller
from connectome_body.compatibility.imitation import (
    ObservationExpert,
    action_error,
    load_demonstrations,
)
from connectome_body.compatibility.run_config import RunSpec
from connectome_body.compatibility.training import source_identity
from connectome_body.util import atomic_json, atomic_torch_save, digest_file, seed_for


@torch.no_grad()
def fit_readout(actor, arrays):
    """Fit only a private linear decoder, with train-only feature statistics."""
    device = next(actor.parameters()).device
    layer = list(actor.decoder.modules())[-1]
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("Readout probe requires a linear decoder")
    captured = []
    hook = layer.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
    features, targets = [], []
    try:
        obs = torch.as_tensor(arrays["train_obs"], device=device)
        state, context = actor.reset(len(obs)), actor.context()
        for t in range(obs.shape[1]):
            _, state = actor(obs[:, t], state, context)
            z = captured.pop().cpu().numpy()
            valid = arrays["train_mask"][:, t]
            features.append(z[valid])
            targets.append(arrays["train_actions"][:, t][valid])
    finally:
        hook.remove()
    z = np.concatenate(features).astype(np.float64)
    y = np.arctanh(np.concatenate(targets).astype(np.float64).clip(-0.999, 0.999))
    if actor.config.normalize_actions:
        y = (y - actor.action_center.cpu().numpy()) / actor.action_scale.cpu().numpy()
    mean, scale = z.mean(0), np.maximum(z.std(0), 1e-8)
    zs = (z - mean) / scale
    covariance = zs.T @ zs / len(z)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    weights = np.linalg.solve(
        covariance + 1e-4 * np.eye(z.shape[1]), zs.T @ (y - y.mean(0)) / len(z)
    )
    weight = (weights / scale[:, None]).T
    bias = y.mean(0) - mean @ weight.T
    layer.weight.copy_(torch.as_tensor(weight, device=device, dtype=layer.weight.dtype))
    layer.bias.copy_(torch.as_tensor(bias, device=device, dtype=layer.bias.dtype))
    return dict(
        training_samples=len(z),
        ridge=1e-4,
        feature_std=scale.tolist(),
        correlation_eigenvalues=eigenvalues.tolist(),
        leading_component_fraction=float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-20)),
        participation_rank=float(eigenvalues.sum() ** 2 / max(eigenvalues @ eigenvalues, 1e-20)),
    )


@torch.no_grad()
def rollout(actor, body, expert, seed, teacher_fraction):
    device = next(actor.parameters()).device
    obs = body.reset(seed, "validation")
    expert.reset(seed)
    state, context = actor.reset(1), actor.context()
    errors, clips, trace = [], [], []
    reward_sum = 0.0
    body_spec = body.study_spec if hasattr(body, "study_spec") else body.spec
    for step in range(body_spec.horizon):
        x = torch.as_tensor(obs, device=device)[None]
        student, state = actor(x, state, context)
        student = student[0].cpu().numpy()
        teacher = expert.action(obs)
        error = (student - teacher) ** 2
        errors.append(error)
        clip_fraction = 0.0
        if actor.config.normalize_observations:
            standardized = (x - actor.observation_mean) / actor.observation_scale
            clip_fraction = float((standardized.abs() > 10).float().mean())
        clips.append(clip_fraction)
        action = teacher_fraction * teacher + (1 - teacher_fraction) * student
        obs, reward, terminated, truncated, info = body.step(action)
        reward_sum += reward
        if step < 20 or (step + 1) % 25 == 0:
            trace.append(
                dict(
                    step=step + 1,
                    action_mse=float(error.mean()),
                    clipped_observation_fraction=clip_fraction,
                    orientation_error=info.get("orientation_error"),
                    position_error=info.get("position_error"),
                )
            )
        if terminated or truncated:
            break
    return dict(
        teacher_fraction=teacher_fraction,
        seed=seed,
        **info,
        mean_return=reward_sum,
        action_mse=float(np.mean(errors)),
        action_mse_per_channel=np.mean(errors, axis=0).tolist(),
        mean_clipped_observation_fraction=float(np.mean(clips)),
        trace=trace,
        autonomous=teacher_fraction == 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing", choices=["positive", "balanced_disjoint"])
    parser.add_argument("--fit-readout", action="store_true")
    parser.add_argument("--teacher-fractions", nargs="+", type=float, default=[0, 0.9, 1])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.episodes < 1 or any(not 0 <= f <= 1 for f in args.teacher_fractions):
        raise ValueError("Positive episode count and teacher fractions in [0,1] required")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.bc_run / "best.pt", args.output / "parent.pt")
    manifest = json.loads((args.bc_run / "manifest.json").read_text())
    spec = RunSpec.from_dict(manifest["config"])
    if args.routing:
        if spec.controller.substrate.kind != "connectome":
            raise ValueError("Routing intervention requires a connectome")
        spec = replace(
            spec,
            controller=replace(
                spec.controller, adapter=replace(spec.controller.adapter, port_routing=args.routing)
            ),
        )
    body = make_body(spec.body)
    try:
        actor = Controller(body.obs_dim, body.action_dim, spec.controller).to(args.device)
        saved = torch.load(args.output / "parent.pt", map_location=args.device, weights_only=False)
        actor.load_state_dict(saved["actor"])
        actor.eval()
        data, arrays = load_demonstrations(args.dataset)
        if data["body_fingerprint"] != body.fingerprint:
            raise ValueError("Expert and student body identities differ")
        report = dict(
            source=source_identity(),
            parent_identity=manifest["identity"],
            parent_sha256=digest_file(args.output / "parent.pt"),
            config=spec.to_dict(),
            dataset=data["fingerprint"],
            fit_readout=args.fit_readout,
            teacher_fractions=args.teacher_fractions,
            rows=[],
            evidence="development-only private policy probe; assisted rollouts are not successes",
        )
        atomic_json(args.output / "request.json", report)
        started = time.monotonic()
        if args.fit_readout:
            report["readout"] = fit_readout(actor, arrays)
            atomic_torch_save(args.output / "private_readout.pt", {"actor": actor.state_dict()})
        report["validation_mse"] = action_error(actor, arrays, "validation", 4)
        print(
            json.dumps({k: v for k, v in report.items() if k in ("readout", "validation_mse")}),
            flush=True,
        )
        expert = ObservationExpert(body)
        for fraction in args.teacher_fractions:
            for episode in range(args.episodes):
                seed = seed_for(spec.train_seed, f"paper-evaluation-validation-{episode}")
                row = rollout(actor, body, expert, seed, fraction)
                report["rows"].append(row)
                report["wall_seconds"] = time.monotonic() - started
                atomic_json(args.output / "result.json", report)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ("trace", "action_mse_per_channel")
                        }
                    ),
                    flush=True,
                )
    finally:
        body.close()


if __name__ == "__main__":
    main()
