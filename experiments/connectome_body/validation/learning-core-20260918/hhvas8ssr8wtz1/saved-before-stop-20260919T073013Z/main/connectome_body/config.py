"""All choices that can affect an experiment are serializable and fingerprinted."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DynamicsConfig:
    channels: int = 16
    gain: float = 0.9
    tau_seconds: float = 0.02
    substeps: int = 2
    input_gain: float = 1.0
    inhibitory_fraction: float = 0.2
    sign_mode: str = "random_dale"
    weight_transform: str = "log1p"

    def validate(self):
        if self.channels < 2 or self.substeps < 1:
            raise ValueError("channels >= 2 and substeps >= 1 are required")
        if self.tau_seconds <= 0 or self.gain < 0 or self.input_gain <= 0:
            raise ValueError("Invalid dynamical scale")
        if not 0 <= self.inhibitory_fraction <= 1:
            raise ValueError("Invalid inhibitory fraction")
        if self.sign_mode not in ("random_dale", "annotated"):
            raise ValueError("Unknown sign_mode")
        if self.weight_transform not in ("log1p", "count"):
            raise ValueError("Unknown weight_transform")


@dataclass(frozen=True)
class BodyConfig:
    backend: str = "flybody"
    task: str = "balance"
    source: str | None = None
    horizon: int = 200
    action_repeat: int = 5
    perturbations: bool = True

    @property
    def control_dt(self) -> float:
        return 0.002 * self.action_repeat

    def validate(self):
        if self.backend not in ("flybody", "fixture"):
            raise ValueError("Body must be flybody or the explicitly synthetic fixture")
        if self.task not in ("balance", "walk", "reach"):
            raise ValueError("Unknown task")
        if self.horizon < 2 or self.action_repeat < 1:
            raise ValueError("Invalid body time budget")


@dataclass(frozen=True)
class PPOConfig:
    interactions: int = 1_000_000
    num_envs: int = 4
    rollout_steps: int = 128
    sequence_length: int = 16
    burn_in: int = 16
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    critic_width: int = 64
    eval_every: int = 20_000
    eval_episodes: int = 20
    test_episodes: int = 100
    success_threshold: float = 0.8
    threshold_confirmations: int = 3

    def validate(self):
        positive = (
            self.interactions,
            self.num_envs,
            self.rollout_steps,
            self.sequence_length,
            self.epochs,
            self.eval_every,
            self.eval_episodes,
            self.test_episodes,
            self.threshold_confirmations,
            self.critic_width,
        )
        if min(positive) < 1 or self.burn_in < 0:
            raise ValueError("Invalid PPO counts")
        if self.interactions % self.num_envs:
            raise ValueError("interactions must be divisible by num_envs; no hidden overshoot")
        if not 0 < self.success_threshold <= 1:
            raise ValueError("success_threshold must be in (0, 1]")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid discount")


@dataclass(frozen=True)
class RunConfig:
    graph: str | None = None
    substrate: str = "real"
    adapter_budget: int = 5_000
    train_seed: int = 0
    substrate_seed: int = 0
    device: str = "cpu"
    threads: int = 1
    purpose: str = "experiment"
    body: BodyConfig = field(default_factory=BodyConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self):
        self.body.validate()
        self.dynamics.validate()
        self.ppo.validate()
        if self.substrate not in (
            "real",
            "degree_shuffled",
            "matched_random",
            "no_edges",
            "adapter_only",
            "adapter_gru",
            "trainable_gru",
        ):
            raise ValueError("Unknown substrate")
        if (
            self.substrate not in ("adapter_only", "adapter_gru", "trainable_gru")
            and self.graph is None
        ):
            raise ValueError("A graph is required; synthetic fallbacks are forbidden")
        if self.adapter_budget < 1 or self.threads < 1:
            raise ValueError("Invalid resource budget")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Use cpu or cuda; sparse MPS is not supported")
        if self.purpose not in ("experiment", "smoke"):
            raise ValueError("purpose must be experiment or smoke")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RunConfig":
        value = dict(value)
        for name, kind in (("body", BodyConfig), ("dynamics", DynamicsConfig), ("ppo", PPOConfig)):
            value[name] = kind(**value.get(name, {}))
        result = cls(**value)
        result.validate()
        return result
