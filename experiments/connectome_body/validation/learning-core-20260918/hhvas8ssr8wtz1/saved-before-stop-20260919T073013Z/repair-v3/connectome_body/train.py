"""Recurrent PPO, held-out evaluation, and strict resumable runs.

Frozen graph weights still transmit gradients to E. Sequence minibatches retain
temporal order; stored rollout states seed a no-gradient burn-in followed by
truncated BPTT. Time limits bootstrap values; falls do not. Neither crosses an
episode boundary when computing GAE.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .body import PROJECT, make_env
from .config import RunConfig
from .graphs import Graph
from .policy import BASELINES, Actor, Critic, parameter_count
from .util import (
    atomic_json,
    atomic_torch_save,
    digest_json,
    restore_rng,
    rng_state,
    seed_everything,
    seed_for,
)


def code_fingerprint():
    digest = hashlib.sha256()
    paths = sorted((PROJECT / "connectome_body").glob("*.py"))
    paths += [PROJECT / "pyproject.toml", PROJECT / "uv.lock", PROJECT / "configs/sources.json"]
    for path in paths:
        digest.update(path.relative_to(PROJECT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def advantages(rewards, values, next_values, terminated, done, gamma, lam):
    """Time-limit bootstrap, but no GAE leakage into the next episode."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]).float() * next_values[t] - values[t]
        carry = delta + gamma * lam * (~done[t]).float() * carry
        result[t] = carry
    return result, result + values


@torch.no_grad()
def evaluate(
    actor: Actor,
    env,
    seed: int,
    episodes: int,
    split="validation",
    intervention="none",
    episode_seeds=None,
):
    records = []
    started = time.monotonic()
    device = actor.log_std.device
    for i in range(episodes):
        episode_seed = (
            episode_seeds[i]
            if episode_seeds is not None
            else seed_for(seed, f"evaluation-{split}-{i}")
        )
        obs = env.reset(episode_seed, split)
        state = actor.initial_state(1)
        for _ in range(env.config.horizon):
            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
            mean, _, state = actor(obs_tensor, state, intervention=intervention)
            if not torch.isfinite(mean).all() or not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite actor during evaluation")
            obs, _, terminated, truncated, info = env.step(torch.tanh(mean)[0].cpu().numpy())
            if terminated or truncated:
                break
        records.append({"seed": episode_seed, **info})
    return {
        "split": split,
        "intervention": intervention,
        "episodes": records,
        "success_rate": float(np.mean([r["success"] for r in records])),
        "mean_score": float(np.mean([r["score"] for r in records])),
        "mean_return": float(np.mean([r["episode_return"] for r in records])),
        "fall_rate": float(np.mean([r["fell"] for r in records])),
        "numerical_failure_rate": float(np.mean([r["numerical_failure"] for r in records])),
        "interactions": sum(r["steps"] for r in records),
        "wall_seconds": time.monotonic() - started,
    }


def threshold_experience(curve, threshold, confirmations, budget):
    streak = 0
    first = None
    for item in curve:
        if item["success_rate"] >= threshold:
            if not streak:
                first = item["training_interactions"]
            streak += 1
            if streak >= confirmations:
                return {
                    "event": True,
                    "first_crossing_step": first,
                    "certification_step": item["training_interactions"],
                    "censor_step": budget,
                }
        else:
            streak, first = 0, None
    return {
        "event": False,
        "first_crossing_step": None,
        "certification_step": None,
        "censor_step": budget,
    }


class Trainer:
    def __init__(self, config: RunConfig, output: Path, resume=False):
        config.validate()
        if config.ppo.eval_every % config.ppo.num_envs:
            raise ValueError("eval_every must be divisible by num_envs")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no device fallback")
        self.config, self.output = config, Path(output)
        self.device = torch.device(config.device)
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise RuntimeError(f"Another worker owns {self.output}") from exc
        if (self.output / "manifest.json").exists() and not resume:
            self.lock.close()
            raise FileExistsError("Existing run; use --resume, never silently overwrite")
        if resume and not (self.output / "latest.pt").is_file():
            self.lock.close()
            raise FileNotFoundError("--resume requires a complete checkpoint")
        self.envs = []
        self.eval_env = None
        try:
            seed_everything(seed_for(config.train_seed, "actor"), config.threads)
            self.graph = None if config.substrate in BASELINES else Graph.load(config.graph)
            for _ in range(config.ppo.num_envs):
                self.envs.append(make_env(config.body))
            env = self.envs[0]
            if any(e.fingerprint != env.fingerprint for e in self.envs):
                raise RuntimeError("Nonidentical body instances")
            self.eval_env = make_env(config.body)
            self.actor = Actor(env.obs_dim, env.action_dim, config, self.graph).to(self.device)
            # Critic initialization is independent of actor architecture/size.
            seed_everything(seed_for(config.train_seed, "critic"), config.threads)
            self.critic = Critic(env.obs_dim, config.ppo.critic_width).to(self.device)
            self.optimizer = torch.optim.Adam(
                [
                    {"params": list(self.actor.parameters())},
                    {"params": list(self.critic.parameters())},
                ],
                lr=config.ppo.learning_rate,
                eps=1e-5,
            )
            seed_everything(seed_for(config.train_seed, "learning"), config.threads)
            portable = config.to_dict()
            portable["graph"] = None if self.graph is None else self.graph.fingerprint
            portable["body"]["source"] = None
            self.manifest = {
                "schema": "connectome-body-run-v1",
                "config": config.to_dict(),
                "graph": self.graph.manifest if self.graph else None,
                "body_fingerprint": env.fingerprint,
                "body_backend": config.body.backend,
                "observation_schema": env.obs_schema,
                "observation_dim": env.obs_dim,
                "action_names": env.action_names,
                "action_dim": env.action_dim,
                "actor_parameters": self.actor.num_parameters,
                "critic_parameters": parameter_count(self.critic),
                "total_trainable_parameters": self.actor.num_parameters
                + parameter_count(self.critic),
                "actor_hidden_width": self.actor.hidden_width,
                "actor_state_size": self.actor.state_dim,
                "code_fingerprint": code_fingerprint(),
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "numpy", "scipy", "mujoco", "dm-control", "pyarrow")
                },
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cuda": torch.version.cuda,
                "evidence": (
                    "software_fixture_only"
                    if config.body.backend == "fixture"
                    or (
                        self.graph
                        and (
                            self.graph.manifest["provenance"].get("coverage") == "synthetic"
                            or self.graph.manifest["provenance"].get("is_synthetic", False)
                        )
                    )
                    else "native_flybody_smoke"
                    if config.purpose == "smoke"
                    else "native_flybody_experiment"
                ),
            }
            if self.actor.brain is not None:
                self.manifest["null"] = self.actor.brain.null_report
                self.manifest["ports_fingerprint"] = self.actor.brain.ports.fingerprint
            self.identity = digest_json(
                {
                    "config": portable,
                    "body": env.fingerprint,
                    "code": self.manifest["code_fingerprint"],
                    "versions": self.manifest["versions"],
                    "python": platform.python_version(),
                }
            )
            self.manifest["identity"] = self.identity
            self.interactions, self.updates, self.eval_interactions = 0, 0, 0
            self.next_eval = config.ppo.eval_every
            self.episode_counts = [0] * len(self.envs)
            self.seen_seeds = []
            self.obs = np.stack([self._reset_env(i) for i in range(len(self.envs))])
            self.state = self.actor.initial_state(len(self.envs))
            self.reset_mask = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
            self.curve, self.update_log, self.episode_log = [], [], []
            self.best = (-1.0, -1.0)
            self.best_step = 0
            self.elapsed_before = 0.0
            self.started = time.monotonic()
            if resume:
                self._restore()
            else:
                atomic_json(self.output / "manifest.json", self.manifest)
                atomic_json(self.output / "config.json", config.to_dict())
                self._validate()
                self._save()
        except BaseException:
            self.close()
            raise

    def _reset_env(self, i):
        seed = seed_for(self.config.train_seed, f"train-{i}-{self.episode_counts[i]}")
        self.episode_counts[i] += 1
        if len(self.seen_seeds) < self.config.ppo.test_episodes:
            self.seen_seeds.append(seed)
        return self.envs[i].reset(seed, "train")

    @torch.no_grad()
    def collect(self):
        p = self.config.ppo
        steps = min(
            p.rollout_steps,
            (p.interactions - self.interactions) // p.num_envs,
            (self.next_eval - self.interactions) // p.num_envs,
        )
        if steps <= 0:
            raise RuntimeError("Invalid rollout scheduling")
        snapshot_steps = {max(0, start - p.burn_in) for start in range(0, steps, p.sequence_length)}
        storage, snapshots = [], {}
        for t in range(steps):
            if t in snapshot_steps:
                snapshots[t] = self.state.detach().cpu().clone()
            obs = torch.as_tensor(self.obs, device=self.device)
            actions, raw, logp, new_state = self.actor.sample(obs, self.state, self.reset_mask)
            if not torch.isfinite(actions).all() or not torch.isfinite(new_state).all():
                raise FloatingPointError("Nonfinite actor; no optimizer or graph fallback")
            value = self.critic(obs)
            next_obs, rewards, terminated, done = [], [], [], []
            reset_obs = []
            for i, action in enumerate(actions.cpu().numpy()):
                following, reward, term, trunc, info = self.envs[i].step(action)
                next_obs.append(following)
                rewards.append(reward)
                terminated.append(term)
                done.append(term or trunc)
                if term or trunc:
                    self.episode_log.append(
                        {"training_interactions": self.interactions + p.num_envs, "env": i, **info}
                    )
                    reset_obs.append(self._reset_env(i))
                else:
                    reset_obs.append(following)
            bootstrap = self.critic(torch.as_tensor(np.stack(next_obs), device=self.device))
            storage.append(
                {
                    "obs": obs.cpu(),
                    "raw": raw.cpu(),
                    "logp": logp.cpu(),
                    "value": value.cpu(),
                    "next_value": bootstrap.cpu(),
                    "reward": torch.tensor(rewards, dtype=torch.float32),
                    "terminated": torch.tensor(terminated),
                    "done": torch.tensor(done),
                    "reset": self.reset_mask.cpu(),
                }
            )
            self.obs = np.stack(reset_obs)
            self.state = new_state.detach()
            self.reset_mask = torch.tensor(done, dtype=torch.bool, device=self.device)
            self.interactions += p.num_envs
        batch = {key: torch.stack([item[key] for item in storage]) for key in storage[0]}
        batch["advantage"], batch["return"] = advantages(
            batch["reward"],
            batch["value"],
            batch["next_value"],
            batch["terminated"],
            batch["done"],
            p.gamma,
            p.gae_lambda,
        )
        adv = batch["advantage"]
        batch["advantage"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return batch, snapshots

    def optimize(self, batch, snapshots):
        p = self.config.ppo
        steps = len(batch["obs"])
        starts = np.arange(0, steps, p.sequence_length)
        statistics = []
        for _ in range(p.epochs):
            for start in np.random.permutation(starts):
                start = int(start)
                end, burn = min(start + p.sequence_length, steps), max(0, start - p.burn_in)
                state = snapshots[burn].to(self.device)
                with torch.no_grad():
                    for t in range(burn, start):
                        _, _, state = self.actor(
                            batch["obs"][t].to(self.device),
                            state,
                            batch["reset"][t].to(self.device),
                        )
                logps, entropies = [], []
                for t in range(start, end):
                    mean, log_std, state = self.actor(
                        batch["obs"][t].to(self.device), state, batch["reset"][t].to(self.device)
                    )
                    logps.append(
                        self.actor.log_probability(mean, log_std, batch["raw"][t].to(self.device))
                    )
                    # Entropy of the pre-tanh Gaussian is the shared exploration
                    # regularizer; do not label it transformed-action entropy.
                    entropies.append((log_std + 0.5 * (1 + np.log(2 * np.pi))).sum(-1))
                logp = torch.stack(logps)
                old_logp = batch["logp"][start:end].to(self.device)
                adv = batch["advantage"][start:end].to(self.device)
                log_ratio = logp - old_logp
                ratio = torch.exp(log_ratio)
                actor_loss = -torch.minimum(
                    ratio * adv, ratio.clamp(1 - p.clip_ratio, 1 + p.clip_ratio) * adv
                ).mean()
                value = self.critic(batch["obs"][start:end].to(self.device))
                value_loss = (
                    0.5 * (value - batch["return"][start:end].to(self.device)).square().mean()
                )
                entropy = torch.stack(entropies).mean()
                loss = (
                    actor_loss + p.value_coefficient * value_loss - p.entropy_coefficient * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), p.max_grad_norm, error_if_nonfinite=True
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), p.max_grad_norm, error_if_nonfinite=True
                )
                encoder_norm = (
                    sum(
                        float(param.grad.square().sum())
                        for param in self.actor.encoder.parameters()
                        if param.grad is not None
                    )
                    ** 0.5
                )
                self.optimizer.step()
                statistics.append(
                    [
                        float(actor_loss.detach()),
                        float(value_loss.detach()),
                        float(((ratio - 1) - log_ratio).mean().detach()),
                        float(actor_norm),
                        encoder_norm,
                    ]
                )
        names = (
            "actor_loss",
            "value_loss",
            "approximate_kl",
            "actor_gradient_norm",
            "encoder_gradient_norm",
        )
        return dict(zip(names, np.mean(statistics, axis=0).tolist()))

    def _validate(self):
        saved_rng = rng_state()
        result = evaluate(
            self.actor, self.eval_env, self.config.train_seed, self.config.ppo.eval_episodes
        )
        restore_rng(saved_rng)
        self.eval_interactions += result["interactions"]
        result["training_interactions"] = self.interactions
        self.curve.append(result)
        metric = (result["success_rate"], result["mean_score"])
        if metric > self.best:
            self.best, self.best_step = metric, self.interactions
            self.best_actor = {
                key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()
            }
            atomic_torch_save(
                self.output / "best.pt",
                {
                    "identity": self.identity,
                    "actor": self.best_actor,
                    "training_interactions": self.interactions,
                },
            )

    def _save(self):
        state = {
            "schema": "connectome-body-checkpoint-v1",
            "identity": self.identity,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(),
            "environments": [env.state_dict() for env in self.envs],
            "obs": self.obs,
            "state": self.state.detach().cpu(),
            "reset_mask": self.reset_mask.cpu(),
            "interactions": self.interactions,
            "updates": self.updates,
            "eval_interactions": self.eval_interactions,
            "next_eval": self.next_eval,
            "episode_counts": self.episode_counts,
            "seen_seeds": self.seen_seeds,
            "curve": self.curve,
            "update_log": self.update_log,
            "episode_log": self.episode_log,
            "best": self.best,
            "best_step": self.best_step,
            "best_actor": self.best_actor,
            "elapsed": self.elapsed_before + time.monotonic() - self.started,
        }
        atomic_torch_save(self.output / "latest.pt", state)
        atomic_json(self.output / "learning_curve.json", self.curve)
        atomic_json(self.output / "updates.json", self.update_log)

    def _restore(self):
        # Checkpoints contain optimizer, NumPy, and MuJoCo states; load only
        # locally created/trusted run artifacts, never arbitrary downloaded .pt.
        # Keep RNG byte tensors on CPU. load_state_dict moves optimizer tensors
        # to their parameters' device; recurrent state is moved explicitly below.
        saved = torch.load(self.output / "latest.pt", map_location="cpu", weights_only=False)
        old = json.loads((self.output / "manifest.json").read_text())
        if saved["identity"] != self.identity or old["identity"] != self.identity:
            raise ValueError(
                "Resume identity mismatch: graph, body, code, versions, or configuration changed"
            )
        self.actor.load_state_dict(saved["actor"])
        self.critic.load_state_dict(saved["critic"])
        self.optimizer.load_state_dict(saved["optimizer"])
        for env, state in zip(self.envs, saved["environments"], strict=True):
            env.load_state_dict(state)
        for name in (
            "obs",
            "interactions",
            "updates",
            "eval_interactions",
            "next_eval",
            "episode_counts",
            "seen_seeds",
            "curve",
            "update_log",
            "episode_log",
            "best",
            "best_step",
        ):
            setattr(self, name, saved[name])
        self.state = saved["state"].to(self.device)
        self.reset_mask = saved["reset_mask"].to(self.device)
        self.elapsed_before = saved["elapsed"]
        self.best_actor = saved["best_actor"]
        # latest.pt is authoritative if a crash happened between writing an
        # improved best.pt and committing the corresponding training state.
        atomic_torch_save(
            self.output / "best.pt",
            {
                "identity": self.identity,
                "actor": self.best_actor,
                "training_interactions": self.best_step,
            },
        )
        # A crash can commit latest.pt before its derived JSON logs, including
        # at the final budget when no further optimization/save will occur.
        atomic_json(self.output / "learning_curve.json", self.curve)
        atomic_json(self.output / "updates.json", self.update_log)
        restore_rng(saved["rng"])

    def run(self, stop_after_updates: int | None = None):
        if (self.output / "result.json").exists():
            return json.loads((self.output / "result.json").read_text())
        p = self.config.ppo
        while self.interactions < p.interactions:
            started = time.monotonic()
            previous = self.interactions
            batch, snapshots = self.collect()
            stats = self.optimize(batch, snapshots)
            self.updates += 1
            seconds = time.monotonic() - started
            self.update_log.append(
                {
                    "training_interactions": self.interactions,
                    "update": self.updates,
                    "seconds": seconds,
                    "interactions_per_second": (self.interactions - previous) / seconds,
                    "rollout_end_state_rms": float(self.state.square().mean().sqrt())
                    if self.state.numel()
                    else 0.0,
                    "rollout_end_state_saturation_fraction": float(
                        (self.state.abs() > 0.95).float().mean()
                    )
                    if self.state.numel()
                    else 0.0,
                    **stats,
                }
            )
            if self.interactions >= self.next_eval:
                self._validate()
                self.next_eval += p.eval_every
            elif self.interactions == p.interactions:
                self._validate()
            self._save()
            print(
                json.dumps(
                    {
                        "step": self.interactions,
                        "budget": p.interactions,
                        "update": self.updates,
                        **stats,
                    }
                ),
                flush=True,
            )
            if (
                stop_after_updates is not None
                and self.updates >= stop_after_updates
                and self.interactions < p.interactions
            ):
                return {
                    "status": "paused_at_checkpoint",
                    "training_interactions": self.interactions,
                }
        return self._finish()

    def _finish(self):
        p = self.config.ppo
        best = torch.load(self.output / "best.pt", map_location=self.device, weights_only=False)
        if best["identity"] != self.identity:
            raise ValueError("Best checkpoint identity mismatch")
        self.actor.load_state_dict(best["actor"])
        scores = {}
        for split in ("test", "ood"):
            scores[split] = evaluate(
                self.actor, self.eval_env, self.config.train_seed, p.test_episodes, split
            )
        replay = self.seen_seeds[: p.test_episodes]
        scores["training_replay"] = evaluate(
            self.actor,
            self.eval_env,
            self.config.train_seed,
            len(replay),
            "train",
            episode_seeds=replay,
        )
        interventions = ["silence"]
        if self.actor.state_dim:
            interventions.append("reset")
        if self.actor.brain is not None:
            interventions.extend(["no_recurrence", "permute_readout"])
        causal = {
            kind: evaluate(
                self.actor, self.eval_env, self.config.train_seed, p.test_episodes, "test", kind
            )
            for kind in interventions
        }
        if self.config.substrate == "real":
            alternative = dataclasses.replace(self.config, substrate="degree_shuffled")
            rewired = Actor(self.actor.obs_dim, self.actor.action_dim, alternative, self.graph).to(
                self.device
            )
            rewired.load_state_dict(self.actor.state_dict())
            causal["acute_rewire"] = evaluate(
                rewired, self.eval_env, self.config.train_seed, p.test_episodes, "test"
            )
            causal["acute_rewire"]["intervention"] = "acute_rewire"
            causal["acute_rewire"]["null"] = rewired.brain.null_report
        self.eval_interactions += sum(
            item["interactions"] for item in [*scores.values(), *causal.values()]
        )
        result = {
            "schema": "connectome-body-result-v1",
            "status": "complete",
            "identity": self.identity,
            "evidence": self.manifest["evidence"],
            "training_interactions": self.interactions,
            "evaluation_interactions": self.eval_interactions,
            "total_environment_interactions": self.interactions + self.eval_interactions,
            "selected_checkpoint_step": best["training_interactions"],
            "adapter_budget": self.config.adapter_budget,
            "actor_parameters": self.actor.num_parameters,
            "experience_to_threshold": threshold_experience(
                self.curve, p.success_threshold, p.threshold_confirmations, p.interactions
            ),
            "evaluation": scores,
            "causal_interventions": causal,
            "generalization_gap": scores["training_replay"]["mean_score"]
            - scores["test"]["mean_score"],
            "training_wall_seconds": sum(item["seconds"] for item in self.update_log),
            "total_wall_seconds": self.elapsed_before + time.monotonic() - self.started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if self.config.device == "cuda"
            else None,
        }
        atomic_json(self.output / "episodes.json", self.episode_log)
        atomic_json(self.output / "result.json", result)
        return result

    def close(self):
        for env in [*self.envs, self.eval_env]:
            if env is not None:
                env.close()
        self.envs = []
        self.eval_env = None
        self.lock.close()


def train(config: RunConfig, output: Path, resume=False, stop_after_updates=None):
    trainer = Trainer(config, output, resume)
    try:
        result = trainer.run(stop_after_updates)
        (trainer.output / "failure.json").unlink(missing_ok=True)
        return result
    except BaseException as exc:
        atomic_json(
            trainer.output / "failure.json",
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "training_interactions": trainer.interactions,
                "exception": type(exc).__name__,
                "message": str(exc),
                "resume": "Resume from latest.pt with the identical configuration and code",
            },
        )
        raise
    finally:
        trainer.close()
