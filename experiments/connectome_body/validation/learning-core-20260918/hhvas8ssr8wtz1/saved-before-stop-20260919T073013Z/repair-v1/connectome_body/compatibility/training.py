"""One recurrent PPO procedure for every body, adapter, topology, and plasticity regime.

Only the neural substrate carries policy state. The critic is separate and its
capacity is reported separately. Exploration variance is fixed, so an
encoder-only or substrate-only condition cannot learn an extra policy parameter.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..body import PROJECT
from ..policy import Actor, Critic
from ..train import advantages, threshold_experience
from ..util import (
    atomic_json,
    atomic_torch_save,
    digest_file,
    digest_json,
    restore_rng,
    rng_state,
    seed_everything,
    seed_for,
)
from .adapters import count_parameters
from .bodies import make_body
from .controller import Controller
from .evaluation import evaluate
from .robustness import make_perturbed_body
from .run_config import RunSpec
from .runtime import hardware_profile


def source_identity():
    """Hash this implementation and its imported legacy helpers without editing them."""
    package = Path(__file__).parent.parent
    paths = sorted(package.glob("*.py"))
    paths += sorted((package / "adaptation").glob("*.py"))
    paths += sorted((package / "compatibility").glob("*.py"))
    paths += sorted((package / "compatibility" / "native").glob("*.cpp"))
    paths += [PROJECT / "pyproject.toml", PROJECT / "uv.lock"]
    return digest_json({str(path.relative_to(PROJECT)): digest_file(path) for path in paths})


def controller_identity(controller):
    portable = controller.config.to_dict()
    portable["substrate"]["graph"] = controller.graph_fingerprint
    for group, key in (("adapter", "anatomical_mapping"), ("substrate", "community_partition")):
        if portable[group][key]:
            portable[group][key] = digest_file(portable[group][key])
    return digest_json(
        {
            "config": portable,
            "observation_dim": controller.obs_dim,
            "action_dim": controller.action_dim,
            "parameters": controller.parameter_report,
        }
    )


def policy_digest(controller):
    digest = hashlib.sha256(controller_identity(controller).encode())
    for name, value in sorted(controller.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0" + array.dtype.str.encode())
        digest.update(str(array.shape).encode() + b"\0" + array.tobytes())
    return digest.hexdigest()


class Trainer:
    def __init__(self, config: RunSpec, output, resume=False):
        config.validate()
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no automatic device fallback")
        self.config, self.output, self.device = config, Path(output), torch.device(config.device)
        self.hardware = hardware_profile(config.device, config.threads)
        for matching in config.metadata.get("matches", []):
            calibration = matching.get("calibration")
            if calibration and calibration.get("hardware") != self.hardware:
                raise ValueError(
                    "Compute-matched condition requires its calibrated hardware and precision"
                )
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "run.lock").open("a")
        self.envs, self.eval_env = [], None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise RuntimeError("Another worker owns this run") from exc
        try:
            if (self.output / "manifest.json").exists() and not resume:
                raise FileExistsError("Run already exists; use --resume")
            if resume and not (self.output / "latest.pt").is_file():
                raise FileNotFoundError("Resume needs an atomically committed latest.pt")
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for _ in range(config.training.num_envs):
                self.envs.append(
                    make_body(config.body)
                    if config.perturbation is None
                    else make_perturbed_body(config.body, config.perturbation)
                )
            self.eval_env = (
                make_body(config.body)
                if config.perturbation is None
                else make_perturbed_body(config.body, config.perturbation)
            )
            env = self.envs[0]
            expected = config.metadata.get("expected_body_fingerprint")
            if expected and env.fingerprint != expected:
                raise ValueError("Body changed since the study was planned")
            for name, path in (
                ("anatomy", config.controller.adapter.anatomical_mapping),
                ("community", config.controller.substrate.community_partition),
            ):
                expected = config.metadata.get(f"expected_{name}_sha256")
                if expected and (not path or digest_file(path) != expected):
                    raise ValueError(f"The {name} input changed since planning")
            if any(item.fingerprint != env.fingerprint for item in [*self.envs, self.eval_env]):
                raise ValueError("Training/evaluation body definitions differ")
            if not math.isclose(config.controller.substrate.dynamics.control_dt, env.control_dt):
                raise ValueError("Neural control_dt must equal the body's decision interval")
            seed_everything(seed_for(config.train_seed, "paper-actor"), config.threads)
            self.actor = Controller(env.obs_dim, env.action_dim, config.controller).to(self.device)
            expected = config.metadata.get("expected_graph_fingerprint")
            if expected and self.actor.graph_fingerprint != expected:
                raise ValueError("Connectome changed since the study was planned")
            self.controller_identity = controller_identity(self.actor)
            seed_everything(seed_for(config.train_seed, "paper-critic"), config.threads)
            self.critic = Critic(env.obs_dim, config.training.critic_width).to(self.device)
            self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(
                [{"params": self.actor_parameters}, {"params": list(self.critic.parameters())}],
                lr=config.training.learning_rate,
                eps=1e-5,
            )
            self.log_std = torch.full(
                (env.action_dim,), config.training.exploration_log_std, device=self.device
            )
            self.transfer = self._initialize_transfer() if config.initial_checkpoint else None
            evidence = env.evidence
            if self.actor.topology is not None and self.actor.topology.report.get("is_synthetic"):
                evidence = "software_fixture_only"
            if evidence == "software_fixture_only" and config.purpose != "smoke":
                raise ValueError("Synthetic bodies/graphs require an explicitly labeled smoke run")
            versions = {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "mujoco", "dm-control", "numba")
            }
            portable = config.to_dict()
            portable.pop("metadata")
            portable["controller"] = self.controller_identity
            portable["body"] = env.fingerprint
            portable["initial_checkpoint"] = self.transfer
            self.manifest = {
                "schema": "compatibility-run-v1",
                "config": config.to_dict(),
                "code_fingerprint": source_identity(),
                "versions": versions,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cuda": torch.version.cuda,
                "device": str(self.device),
                "hardware": self.hardware,
                "body_fingerprint": env.fingerprint,
                "evidence": evidence,
                "observation_dim": env.obs_dim,
                "observation_schema": env.obs_schema,
                "action_dim": env.action_dim,
                "action_names": env.action_names,
                "controller_identity": self.controller_identity,
                "parameters": self.actor.parameter_report,
                "critic_parameters": count_parameters(self.critic, True),
                "total_optimization_parameters": count_parameters(self.actor, True)
                + count_parameters(self.critic, True),
                "topology": self.actor.topology.report if self.actor.topology is not None else None,
                "initialization": self.actor.initialization_report,
                "structural_features": self.actor.feature_report,
                "transfer": self.transfer,
                "exploration": {
                    "kind": "fixed_pre_tanh_gaussian",
                    "log_std": config.training.exploration_log_std,
                },
            }
            self.identity = digest_json(
                {
                    "config": portable,
                    "code": self.manifest["code_fingerprint"],
                    "versions": versions,
                    "python": platform.python_version(),
                    "hardware": self.hardware,
                }
            )
            self.manifest["identity"] = self.identity
            seed_everything(seed_for(config.train_seed, "paper-learning"), config.threads)
            self.interactions = self.updates = self.optimizer_steps = self.eval_interactions = 0
            self.optimized_decisions = self.burnin_decisions = 0
            self.training_seconds = self.elapsed_before = 0.0
            self.next_eval = config.training.eval_every
            self.episode_counts = [0] * len(self.envs)
            self.seen_seeds, self.curve, self.update_log, self.episode_log = [], [], [], []
            self.obs = np.stack([self._reset(i) for i in range(len(self.envs))])
            self.state = self.actor.reset(len(self.envs))
            self.reset_mask = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
            self.best, self.best_step, self.best_actor = (-1.0, -1.0), 0, None
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

    def _initialize_transfer(self):
        path = Path(self.config.initial_checkpoint)
        parent = json.loads((path.parent / "manifest.json").read_text())
        if parent.get("schema") == "compatibility-imitation-run-v1":
            return self._initialize_bc(path, parent)
        # This runner accepts its own trusted checkpoints, never arbitrary model
        # pickles. Transfer starts a new optimizer and counts new experience.
        if parent.get("schema") != "compatibility-run-v1":
            raise ValueError("Transfer requires a compatibility study checkpoint")
        checksum = digest_file(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if digest_file(path) != checksum:
            raise ValueError("Transfer checkpoint changed while it was being loaded")
        if (
            saved.get("identity") != parent["identity"]
            or parent["controller_identity"] != self.controller_identity
        ):
            raise ValueError("Transfer controller/graph/port identity mismatch")
        if (
            parent["config"]["body"]["name"] != self.config.body.name
            or parent["action_names"] != self.envs[0].action_names
        ):
            raise ValueError("Transfer requires the same body and actuator semantics")
        self.actor.load_state_dict(saved["actor"])
        selected_experience = saved.get("interactions", saved.get("training_interactions"))
        source_experience = selected_experience
        completed_path = path.parent / "result.json"
        if completed_path.exists():
            completed = json.loads(completed_path.read_text())
            if completed.get("identity") != parent["identity"] or completed.get(
                "manifest_sha256"
            ) != digest_json(parent):
                raise ValueError("Transfer source result or manifest seal mismatch")
            source_experience = completed["training_interactions"]
        return {
            "checkpoint_sha256": checksum,
            "source_run_identity": parent["identity"],
            "source_training_interactions": source_experience,
            "source_checkpoint_interactions": selected_experience,
            "source_task": parent["config"]["body"]["task"],
            "optimizer_reinitialized": True,
        }

    def _initialize_bc(self, path, parent):
        """BC-only and BC->PPO share a sealed, validation-selected policy."""
        result = json.loads((path.parent / "result.json").read_text())
        checksum = digest_file(path)
        if (
            result.get("status") != "complete"
            or result.get("identity") != parent.get("identity")
            or result.get("manifest_sha256") != digest_json(parent)
            or result.get("selected_checkpoint_sha256") != checksum
            or parent.get("controller_identity") != self.controller_identity
            or parent.get("body_fingerprint") != self.envs[0].fingerprint
            or parent.get("code_fingerprint") != source_identity()
            or parent.get("hardware") != self.hardware
            or parent.get("action_names") != self.envs[0].action_names
        ):
            raise ValueError("BC initialization identity, completion or checkpoint seal mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("identity") != parent["identity"] or digest_file(path) != checksum:
            raise ValueError("BC checkpoint changed or belongs to another run")
        self.actor.load_state_dict(saved["actor"])
        return {
            "kind": "behavioral_cloning",
            "checkpoint_sha256": checksum,
            "source_run_identity": parent["identity"],
            "dataset_fingerprint": parent["dataset_fingerprint"],
            "source_training_interactions": 0,
            "expert_samples": result["unique_expert_samples"],
            "expert_exposures": result["expert_exposures"],
            "teacher_environment_interactions": result["teacher_environment_interactions"],
            "bc_optimizer_steps": result["optimizer_steps"],
            "bc_wall_seconds": result["wall_seconds"],
            "bc_optimization_seconds": result["optimization_seconds"],
            "bc_closed_loop": result["selected_checkpoint_evaluation"],
            "expert_action_validation_mse": result["expert_action_validation_mse"],
            "optimizer_reinitialized": True,
            "recurrent_state_reset": True,
        }

    def _reset(self, index):
        seed = seed_for(self.config.train_seed, f"paper-train-{index}-{self.episode_counts[index]}")
        self.episode_counts[index] += 1
        if len(self.seen_seeds) < self.config.training.test_episodes:
            self.seen_seeds.append(seed)
        return self.envs[index].reset(seed, "train")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def collect(self):
        p = self.config.training
        steps = min(
            p.rollout_steps,
            (p.interactions - self.interactions) // p.num_envs,
            (self.next_eval - self.interactions) // p.num_envs,
        )
        if steps < 1:
            raise RuntimeError("Invalid exact-budget rollout schedule")
        snapshot_steps = {max(0, start - p.burn_in) for start in range(0, steps, p.sequence_length)}
        storage, snapshots = [], {}
        context = self.actor.context()
        for t in range(steps):
            if t in snapshot_steps:
                snapshots[t] = self.state.detach().cpu().clone()
            obs = torch.as_tensor(self.obs, device=self.device)
            mean, state = self.actor(obs, self.state, context, reset=self.reset_mask, squash=False)
            raw = mean + self.log_std.exp() * torch.randn_like(mean)
            action, logp = torch.tanh(raw), Actor.log_probability(mean, self.log_std, raw)
            if not torch.isfinite(action).all() or not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite policy during rollout")
            value = self.critic(self.actor.normalize_observation(obs))
            next_obs, reset_obs, rewards, terminated, done = [], [], [], [], []
            for i, command in enumerate(action.cpu().numpy()):
                following, reward, term, trunc, info = self.envs[i].step(command)
                next_obs.append(following)
                rewards.append(reward)
                terminated.append(term)
                done.append(term or trunc)
                if term or trunc:
                    self.episode_log.append(
                        {
                            "training_interactions": self.interactions + p.num_envs,
                            "env": i,
                            "seed": self.envs[i].case["seed"],
                            **info,
                        }
                    )
                    reset_obs.append(self._reset(i))
                else:
                    reset_obs.append(following)
            bootstrap = self.critic(
                self.actor.normalize_observation(
                    torch.as_tensor(np.stack(next_obs), device=self.device)
                )
            )
            storage.append(
                {
                    "obs": obs.cpu(),
                    "raw": raw.cpu(),
                    "mean": mean.cpu(),
                    "logp": logp.cpu(),
                    "value": value.cpu(),
                    "next_value": bootstrap.cpu(),
                    "reward": torch.tensor(rewards, dtype=torch.float32),
                    "terminated": torch.tensor(terminated),
                    "done": torch.tensor(done),
                    "reset": self.reset_mask.cpu(),
                }
            )
            self.obs, self.state = np.stack(reset_obs), state.detach()
            self.reset_mask = torch.tensor(done, dtype=torch.bool, device=self.device)
            self.interactions += p.num_envs
        batch = {name: torch.stack([row[name] for row in storage]) for name in storage[0]}
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
        p, statistics = self.config.training, []
        max_kl, stopped = 0.0, False
        steps = len(batch["obs"])
        for _ in range(p.epochs):
            for start in np.random.permutation(np.arange(0, steps, p.sequence_length)):
                start = int(start)
                end, burn = min(start + p.sequence_length, steps), max(0, start - p.burn_in)
                state = snapshots[burn].to(self.device)
                # Reconstruct burn-in with current weights; discard its graph.
                with torch.no_grad():
                    context = self.actor.context()
                    for t in range(burn, start):
                        _, state = self.actor(
                            batch["obs"][t].to(self.device),
                            state,
                            context,
                            reset=batch["reset"][t].to(self.device),
                        )
                context = self.actor.context()
                logps, means = [], []
                for t in range(start, end):
                    mean, state = self.actor(
                        batch["obs"][t].to(self.device),
                        state,
                        context,
                        reset=batch["reset"][t].to(self.device),
                        squash=False,
                    )
                    logps.append(
                        Actor.log_probability(mean, self.log_std, batch["raw"][t].to(self.device))
                    )
                    means.append(mean)
                log_ratio = torch.stack(logps) - batch["logp"][start:end].to(self.device)
                ratio = log_ratio.exp()
                # Equal fixed Gaussian variances give an exact conditional KL.
                # Tanh is bijective, so the action-space KL is the same. Unlike
                # a sampled log-ratio estimate, this cannot hide drift by noise.
                if p.target_kl is not None:
                    old_means = batch["mean"][start:end].to(self.device)
                    kl = float(
                        (
                            0.5
                            * ((torch.stack(means).detach() - old_means) / self.log_std.exp())
                            .square()
                            .sum(-1)
                        ).mean()
                    )
                    if not math.isfinite(kl):
                        raise FloatingPointError("Nonfinite PPO KL")
                    max_kl = max(max_kl, kl)
                    if kl > 1.5 * p.target_kl:
                        stopped = True
                        break  # No additional optimizer step on a stale rollout.
                adv = batch["advantage"][start:end].to(self.device)
                actor_loss = -torch.minimum(
                    ratio * adv, ratio.clamp(1 - p.clip_ratio, 1 + p.clip_ratio) * adv
                ).mean()
                value = self.critic(
                    self.actor.normalize_observation(batch["obs"][start:end].to(self.device))
                )
                value_loss = (
                    0.5 * (value - batch["return"][start:end].to(self.device)).square().mean()
                )
                loss = actor_loss + p.value_coefficient * value_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO objective")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(
                    self.actor_parameters, p.max_grad_norm, error_if_nonfinite=False
                )
                critic_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), p.max_grad_norm, error_if_nonfinite=False
                )
                if not torch.isfinite(norm) or not torch.isfinite(critic_norm):
                    raise FloatingPointError("Nonfinite actor or critic gradients")
                self.optimizer.step()
                self.optimizer_steps += 1
                self.optimized_decisions += (end - start) * p.num_envs
                self.burnin_decisions += (start - burn) * p.num_envs
                statistics.append(
                    [
                        float(actor_loss.detach()),
                        float(value_loss.detach()),
                        float(((ratio - 1) - log_ratio).mean().detach()),
                        float(norm),
                    ]
                )
            if stopped:
                break
        result = dict(
            zip(
                ("actor_loss", "value_loss", "approximate_kl", "actor_gradient_norm"),
                np.mean(statistics, axis=0).tolist() if statistics else [0.0] * 4,
                strict=True,
            )
        )
        result.update(
            kl_early_stopped=stopped,
            max_conditional_kl=max_kl if p.target_kl is not None else None,
            minibatch_optimizer_steps=len(statistics),
        )
        return result

    def _validate(self):
        saved_rng = rng_state()
        try:
            result = evaluate(
                self.actor,
                self.eval_env,
                self.config.train_seed,
                self.config.training.eval_episodes,
            )
        finally:
            restore_rng(saved_rng)
        self.eval_interactions += result["interactions"]
        result.update(
            training_interactions=self.interactions,
            optimizer_steps=self.optimizer_steps,
            training_wall_seconds=self.training_seconds,
        )
        self.curve.append(result)
        metric = (result["success_rate"], result["mean_score"])
        if metric > self.best:
            self.best, self.best_step = metric, self.interactions
            self.best_actor = {
                name: value.detach().cpu().clone()
                for name, value in self.actor.state_dict().items()
            }

    def _save_best(self):
        atomic_torch_save(
            self.output / "best.pt",
            {
                "schema": "compatibility-policy-v1",
                "identity": self.identity,
                "actor": self.best_actor,
                "training_interactions": self.best_step,
                "controller_identity": self.controller_identity,
            },
        )

    def _save(self):
        state = {
            name: getattr(self, name)
            for name in (
                "interactions",
                "updates",
                "optimizer_steps",
                "eval_interactions",
                "optimized_decisions",
                "burnin_decisions",
                "next_eval",
                "episode_counts",
                "seen_seeds",
                "curve",
                "update_log",
                "episode_log",
                "best",
                "best_step",
                "best_actor",
                "training_seconds",
                "obs",
            )
        }
        state.update(
            schema="compatibility-checkpoint-v1",
            identity=self.identity,
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            optimizer=self.optimizer.state_dict(),
            rng=rng_state(),
            environments=[env.state_dict() for env in self.envs],
            state=self.state.detach().cpu(),
            reset_mask=self.reset_mask.cpu(),
            elapsed=self.elapsed_before + time.monotonic() - self.started,
        )
        atomic_torch_save(self.output / "latest.pt", state)
        self._save_best()
        self._write_logs()

    def _write_logs(self):
        for name, value in (
            ("learning_curve", self.curve),
            ("updates", self.update_log),
            ("episodes", self.episode_log),
        ):
            atomic_json(self.output / f"{name}.json", value)

    def _restore(self):
        old = json.loads((self.output / "manifest.json").read_text())
        saved = torch.load(self.output / "latest.pt", map_location="cpu", weights_only=False)
        if (
            saved.get("schema") != "compatibility-checkpoint-v1"
            or saved.get("identity") != self.identity
            or old["identity"] != self.identity
        ):
            raise ValueError(
                "Resume identity mismatch: code, body, graph, runtime, or configuration changed"
            )
        self.actor.load_state_dict(saved["actor"])
        self.critic.load_state_dict(saved["critic"])
        self.optimizer.load_state_dict(saved["optimizer"])
        for env, snapshot in zip(self.envs, saved["environments"], strict=True):
            env.load_state_dict(snapshot)
        for name in (
            "interactions",
            "updates",
            "optimizer_steps",
            "eval_interactions",
            "optimized_decisions",
            "burnin_decisions",
            "next_eval",
            "episode_counts",
            "seen_seeds",
            "curve",
            "update_log",
            "episode_log",
            "best",
            "best_step",
            "best_actor",
            "training_seconds",
            "obs",
        ):
            setattr(self, name, saved[name])
        self.state, self.reset_mask = (
            saved["state"].to(self.device),
            saved["reset_mask"].to(self.device),
        )
        self.elapsed_before = saved["elapsed"]
        self._save_best()
        self._write_logs()
        restore_rng(saved["rng"])

    def run(self, stop_after_updates=None, max_seconds=None):
        if stop_after_updates is not None and (
            type(stop_after_updates) is not int or stop_after_updates < 1
        ):
            raise ValueError("stop_after_updates must be a positive integer")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")
        completed = self.output / "result.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if result["identity"] != self.identity:
                raise ValueError("Result identity mismatch")
            return result
        p = self.config.training
        while self.interactions < p.interactions:
            self._sync()
            started, previous = time.monotonic(), self.interactions
            batch, snapshots = self.collect()
            self._sync()
            collected = time.monotonic()
            stats = self.optimize(batch, snapshots)
            self._sync()
            optimized = time.monotonic()
            self.updates += 1
            self.training_seconds += optimized - started
            self.update_log.append(
                {
                    "training_interactions": self.interactions,
                    "update": self.updates,
                    "optimizer_steps": self.optimizer_steps,
                    "collection_seconds": collected - started,
                    "optimization_seconds": optimized - collected,
                    "training_seconds": self.training_seconds,
                    "interactions_per_second": (self.interactions - previous)
                    / (optimized - started),
                    **stats,
                }
            )
            if self.interactions >= self.next_eval or self.interactions == p.interactions:
                self._validate()
                while self.next_eval <= self.interactions:
                    self.next_eval += p.eval_every
            self._save()
            print(
                json.dumps(
                    {
                        "training_interactions": self.interactions,
                        "budget": p.interactions,
                        "rollouts": self.updates,
                        "optimizer_steps": self.optimizer_steps,
                        **stats,
                    }
                ),
                flush=True,
            )
            paused = (stop_after_updates is not None and self.updates >= stop_after_updates) or (
                max_seconds is not None and time.monotonic() - self.started >= max_seconds
            )
            if paused and self.interactions < p.interactions:
                return {
                    "status": "paused_at_checkpoint",
                    "identity": self.identity,
                    "training_interactions": self.interactions,
                    "optimizer_steps": self.optimizer_steps,
                }
        return self._finish()

    def _finish(self):
        p = self.config.training
        final_weights = {
            name: tensor.detach().clone() for name, tensor in self.actor.state_dict().items()
        }
        latest_scores = evaluate(
            self.actor, self.eval_env, self.config.train_seed, p.test_episodes, "test"
        )
        self.actor.load_state_dict(self.best_actor)
        selected = {
            split: evaluate(
                self.actor, self.eval_env, self.config.train_seed, p.test_episodes, split
            )
            for split in ("test", "ood")
        }
        policy = policy_digest(self.actor)
        self.actor.load_state_dict(final_weights)
        self.eval_interactions += latest_scores["interactions"] + sum(
            value["interactions"] for value in selected.values()
        )
        result = {
            "schema": "compatibility-result-v1",
            "status": "complete",
            "identity": self.identity,
            "manifest_sha256": digest_json(self.manifest),
            "evidence": self.manifest["evidence"],
            "training_interactions": self.interactions,
            "evaluation_interactions": self.eval_interactions,
            "total_environment_interactions": self.interactions + self.eval_interactions,
            "optimizer_steps": self.optimizer_steps,
            "rollout_updates": self.updates,
            "optimized_decisions": self.optimized_decisions,
            "burnin_decisions": self.burnin_decisions,
            "training_wall_seconds": self.training_seconds,
            "total_wall_seconds": self.elapsed_before + time.monotonic() - self.started,
            "parameters": self.actor.parameter_report,
            "pretraining": self.transfer,
            "selected_checkpoint_step": self.best_step,
            "selected_policy_digest": policy,
            "experience_to_threshold": threshold_experience(
                self.curve, p.success_threshold, p.threshold_confirmations, p.interactions
            ),
            "final_checkpoint_test": latest_scores,
            "selected_checkpoint_evaluation": selected,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if self.device.type == "cuda"
            else None,
        }
        atomic_json(self.output / "result.json", result)
        return result

    def close(self):
        for env in [*self.envs, self.eval_env]:
            if env is not None:
                env.close()
        self.envs, self.eval_env = [], None
        if not self.lock.closed:
            self.lock.close()


def train(config, output, resume=False, stop_after_updates=None, max_seconds=None):
    trainer = Trainer(config, output, resume)
    try:
        result = trainer.run(stop_after_updates, max_seconds)
        (trainer.output / "failure.json").unlink(missing_ok=True)
        return result
    except BaseException as exc:
        record = {
            "status": "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else ("failed" if isinstance(exc, FloatingPointError) else "execution_error"),
            "identity": trainer.identity,
            "manifest_sha256": digest_json(trainer.manifest),
            "training_interactions": trainer.interactions,
            "optimizer_steps": trainer.optimizer_steps,
            "exception": type(exc).__name__,
            "message": str(exc),
            "resume": "Restart from latest.pt with the identical run specification and code",
        }
        record["fingerprint"] = digest_json(record)
        atomic_json(trainer.output / "failure.json", record)
        raise
    finally:
        trainer.close()
