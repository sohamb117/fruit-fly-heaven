"""Strict run specification for embodied learning, including exact experience budgets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bodies import BodySpec
from .config import ControllerConfig, _positive_integer
from .robustness import PerturbationSpec


@dataclass(frozen=True)
class LearningConfig:
    interactions: int = 1_000_000
    num_envs: int = 4
    rollout_steps: int = 128
    sequence_length: int = 16
    burn_in: int = 16
    epochs: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    critic_width: int = 64
    exploration_log_std: float = -1.0
    eval_every: int = 20_000
    eval_episodes: int = 20
    test_episodes: int = 100
    success_threshold: float = 0.8
    threshold_confirmations: int = 3

    def validate(self):
        for name in (
            "interactions",
            "num_envs",
            "rollout_steps",
            "sequence_length",
            "epochs",
            "critic_width",
            "eval_every",
            "eval_episodes",
            "test_episodes",
            "threshold_confirmations",
        ):
            _positive_integer(getattr(self, name), name)
        _positive_integer(self.burn_in, "burn_in", minimum=0)
        if self.interactions % self.num_envs or self.eval_every % self.num_envs:
            raise ValueError("Training and evaluation intervals must divide by num_envs exactly")
        for name in ("learning_rate", "value_coefficient", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0 < self.clip_ratio < 1 or not 0 < self.success_threshold <= 1:
            raise ValueError("Invalid PPO clipping or success threshold")
        if not -5 <= self.exploration_log_std <= 1:
            raise ValueError("Fixed exploration_log_std must lie in [-5, 1]")


@dataclass(frozen=True)
class RunSpec:
    body: BodySpec = field(default_factory=BodySpec)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    training: LearningConfig = field(default_factory=LearningConfig)
    train_seed: int = 0
    device: str = "cpu"
    threads: int = 1
    purpose: str = "experiment"
    metadata: dict = field(default_factory=dict)
    initial_checkpoint: str | None = None
    perturbation: PerturbationSpec | None = None

    def validate(self):
        self.body.validate()
        self.controller.validate()
        self.training.validate()
        if self.perturbation is not None:
            self.perturbation.validate()
        _positive_integer(self.train_seed, "train_seed", minimum=0)
        _positive_integer(self.threads, "threads")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("The paper's sparse plasticity runner supports CPU or CUDA")
        if self.purpose not in ("experiment", "smoke") or not isinstance(self.metadata, dict):
            raise ValueError("Invalid run purpose or metadata")
        if self.initial_checkpoint is not None and not isinstance(self.initial_checkpoint, str):
            raise ValueError("Transfer initialization requires a checkpoint filename")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["body"] = BodySpec(**value.get("body", {}))
        value["controller"] = ControllerConfig.from_dict(value.get("controller", {}))
        value["training"] = LearningConfig(**value.get("training", {}))
        if value.get("perturbation") is not None:
            value["perturbation"] = PerturbationSpec(**value["perturbation"])
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        # Paths in a materialized run are resolved by the planner, never by an
        # implicit cwd change in the training process.
        return cls.from_dict(json.loads(Path(path).read_text()))
