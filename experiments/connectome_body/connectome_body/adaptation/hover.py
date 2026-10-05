"""Native FlyBody hovering with its published wing-pattern generator."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..body import FLYBODY_REVISION, PROJECT, quaternion, verify_flybody
from ..util import digest_file, digest_json, seed_for


@dataclass(frozen=True)
class HoverConfig:
    horizon: int = 5000
    action_repeat: int = 1
    initial_angle: float = 0.04
    initial_velocity: float = 0.5
    linear_kick: float = 2.0
    angular_kick: float = 3.0
    perturbations: bool = True
    position_tolerance: float = 0.25
    angle_tolerance: float = math.pi / 9
    velocity_tolerance: float = 5.0
    final_dwell_seconds: float = 0.1

    @property
    def control_dt(self):
        return 0.0002 * self.action_repeat

    def validate(self):
        if self.horizon < 8 or self.action_repeat < 1:
            raise ValueError("Invalid hover time grid")
        if min(self.initial_angle, self.initial_velocity, self.linear_kick, self.angular_kick) < 0:
            raise ValueError("Negative disturbance scale")
        if (
            min(
                self.position_tolerance,
                self.angle_tolerance,
                self.velocity_tolerance,
                self.final_dwell_seconds,
            )
            <= 0
        ):
            raise ValueError("Invalid success criterion")


def hover_scenario(seed, split, config):
    if split not in ("train", "validation", "test", "ood"):
        raise ValueError("Unknown hover scenario split")
    rng = np.random.default_rng(seed_for(seed, f"hover-{split}"))
    scale = 2.0 if split == "ood" else 1.0
    disturbances = []
    if config.perturbations:
        for fraction in (0.25, 0.60):
            disturbances.append(
                {
                    "step": int(config.horizon * fraction),
                    "linear_velocity": (rng.normal(size=3) * config.linear_kick * scale).tolist(),
                    "angular_velocity": (rng.normal(size=3) * config.angular_kick * scale).tolist(),
                }
            )
    return {
        "seed": int(seed),
        "split": split,
        "initial_angles": (rng.uniform(-1, 1, 3) * config.initial_angle * scale).tolist(),
        "initial_velocity": (rng.normal(size=3) * config.initial_velocity * scale).tolist(),
        "disturbances": disturbances,
    }


class FlyBodyInterface:
    """No feedback controller is hidden in this body: only the fixed upstream WPG."""

    def __init__(self, config=HoverConfig(), asset_root=None):
        config.validate()
        self.config = config
        source = str(PROJECT / "references/flybody")
        source_digest = verify_flybody(source)
        os.environ.setdefault("MUJOCO_GL", "disable")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
        import mujoco
        from dm_control import mjcf
        from dm_control.locomotion.arenas import floors
        from dm_control.rl.control import PhysicsError
        from flybody.fruitfly.fruitfly import FruitFly
        from flybody.tasks.flight_imitation import FlightImitationWBPG
        from flybody.tasks.pattern_generators import WingBeatPatternGenerator
        from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader

        from .assets import prepare_assets

        asset_root = Path(asset_root or PROJECT / "data/teacher-assets")
        prepare_assets(asset_root)
        pattern = asset_root / "flight-data/wing_pattern_fmech.npy"
        self.mujoco, self.PhysicsError = mujoco, PhysicsError
        self.wpg = WingBeatPatternGenerator(base_pattern_path=str(pattern))
        self.loader = InferenceFlightTrajectoryLoader()
        self._set_reference()
        self.task = FlightImitationWBPG(
            wbpg=self.wpg,
            traj_generator=self.loader,
            walker=FruitFly,
            arena=floors.Floor(size=(10, 10)),
            time_limit=config.horizon * config.control_dt + 0.02,
            force_actuators=False,
            disable_legs=True,
            joint_filter=0.0,
            future_steps=5,
            initialize_qvel=True,
            terminal_com_dist=5.0,
            trajectory_sites=False,
        )
        self.walker = self.task.walker
        self.rng = np.random.RandomState(0)
        self.task.initialize_episode_mjcf(self.rng)
        self.physics = mjcf.Physics.from_mjcf_model(self.task.root_entity.mjcf_model)
        self.observables = {k: v for k, v in sorted(self.task.observables.items()) if v.enabled}
        spec = self.walker.get_action_spec(self.physics)
        self.low, self.high = spec.minimum.copy(), spec.maximum.copy()
        self.action_names = spec.name.split("\t")
        self.action_dim = len(self.low)
        self.integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.native_substeps = round(0.0002 / self.physics.timestep())
        self.reset(0)
        self.obs_schema = [
            {"name": key, "shape": list(value.shape), "scale": self._scale(key)}
            for key, value in self.raw_observation().items()
        ]
        self.obs_dim = len(self.observation())
        self.fingerprint = digest_json(
            {
                "task": "hover",
                "config": dataclasses.asdict(config),
                "source": source_digest,
                "revision": FLYBODY_REVISION,
                "wrapper_sha256": digest_file(__file__),
                "mujoco": mujoco.__version__,
                "wpg_sha256": digest_file(pattern),
                "observation_schema": self.obs_schema,
                "action_names": self.action_names,
                "low": self.low.tolist(),
                "high": self.high.tolist(),
                "xml_sha256": hashlib.sha256(
                    self.task.root_entity.mjcf_model.to_xml_string().encode()
                ).hexdigest(),
                "action_semantics": "native residual joint commands plus WPG frequency; affine normalized [-1,1]",
            }
        )

    def _set_reference(self):
        length = self.config.horizon * self.config.action_repeat + 16
        self.reference_quaternion = quaternion(0, -math.radians(47.5), 0)
        self.target_com = np.array([0.0, 0.0, 1.0])
        qpos = np.tile(np.r_[self.target_com, self.reference_quaternion], (length, 1))
        self.loader.set_next_trajectory(qpos, np.zeros((length, 6)))

    @staticmethod
    def _scale(key):
        if "joints_vel" in key:
            return 2000.0
        if "joints_pos" in key:
            return math.pi
        if "accelerometer" in key:
            return 2000.0
        if "gyro" in key:
            return 200.0
        if "velocimeter" in key:
            return 50.0
        return 1.0

    def raw_observation(self):
        return {
            key: np.asarray(value(self.physics)).astype(np.float32)
            for key, value in self.observables.items()
        }

    def observation(self):
        return np.clip(
            np.concatenate(
                [
                    value.reshape(-1) / self._scale(key)
                    for key, value in self.raw_observation().items()
                ]
            ),
            -10,
            10,
        ).astype(np.float32)

    def normalize_teacher_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Invalid teacher action")
        return np.clip(2 * (action - self.low) / (self.high - self.low) - 1, -1, 1).astype(
            np.float32
        )

    def reset(self, seed, split="train"):
        from flybody.quaternions import mult_quat

        self.case = hover_scenario(seed, split, self.config)
        self.rng = np.random.RandomState(seed)
        self.steps, self.episode_return, self.held_steps = 0, 0.0, 0
        self.failed = False
        self.metric_history, self.recovery_steps = [], {}
        self._set_reference()
        self.task.initialize_episode_mjcf(self.rng)
        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.physics.model.qpos0
            self.physics.data.qvel[:] = 0
            self.physics.data.act[:] = 0
            self.physics.data.ctrl[:] = 0
            self.physics.data.time = 0
            self.walker.initialize_episode(self.physics, self.rng)
            self.task.initialize_episode(self.physics, self.rng)
            position, rotation = self.walker.get_pose(self.physics)
            local = quaternion(*self.case["initial_angles"])
            self.walker.set_pose(self.physics, position, mult_quat(rotation, local))
            self.physics.bind(self.task._root_joint).qvel[:3] = self.case["initial_velocity"]
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        return self.observation()

    def metrics(self):
        position = np.asarray(self.physics.named.data.subtree_com["walker/"])
        _, rotation = self.walker.get_pose(self.physics)
        dot = np.clip(abs(np.dot(rotation, self.reference_quaternion)), 0, 1)
        qvel = np.asarray(self.physics.bind(self.task._root_joint).qvel)
        return {
            "position_error": float(np.linalg.norm(position - self.target_com)),
            "orientation_error": float(2 * np.arccos(dot)),
            "height": float(position[2]),
            "speed": float(np.linalg.norm(qvel[:3])),
            "control_magnitude": float(np.mean(self.last_action**2)),
        }

    def step(self, action):
        if self.failed or self.steps >= self.config.horizon:
            raise RuntimeError("Reset the completed hover episode before stepping")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Invalid normalized action")
        action = np.clip(action, -1, 1)
        self.last_action = action.copy()
        for disturbance in self.case["disturbances"]:
            if disturbance["step"] == self.steps:
                velocity = self.physics.bind(self.task._root_joint).qvel.copy()
                velocity += np.r_[disturbance["linear_velocity"], disturbance["angular_velocity"]]
                self.physics.bind(self.task._root_joint).qvel = velocity
                self.recovery_steps[self.steps] = None
        numerical_failure = False
        try:
            for _ in range(self.config.action_repeat):
                native_action = self.low + (action + 1) * 0.5 * (self.high - self.low)
                self.task.before_step(self.physics, native_action.copy(), self.rng)
                self.physics.step(nstep=self.native_substeps)
            numerical_failure = (
                not np.isfinite(self.physics.data.qpos).all()
                or not np.isfinite(self.physics.data.qvel).all()
            )
        except self.PhysicsError:
            numerical_failure = True
        self.steps += 1
        metrics = (
            self.metrics()
            if not numerical_failure
            else {
                "position_error": 10.0,
                "orientation_error": math.pi,
                "height": 0.0,
                "speed": 100.0,
                "control_magnitude": float(np.mean(action**2)),
            }
        )
        c = self.config
        self.failed = (
            numerical_failure
            or metrics["height"] < 0.2
            or metrics["position_error"] > 3.0
            or metrics["orientation_error"] > math.pi / 2
        )
        recovered = (
            metrics["position_error"] < c.position_tolerance
            and metrics["orientation_error"] < c.angle_tolerance
            and metrics["speed"] < c.velocity_tolerance
        )
        self.held_steps = self.held_steps + 1 if recovered else 0
        for kick, recovered_at in self.recovery_steps.items():
            if recovered_at is None and recovered:
                self.recovery_steps[kick] = self.steps
        reward = math.exp(
            -((metrics["position_error"] / 0.5) ** 2) - (metrics["orientation_error"] / 0.5) ** 2
        ) * (1 - 0.05 * metrics["control_magnitude"])
        if numerical_failure:
            reward = 0.0
        self.episode_return += reward
        self.metric_history.append(metrics)
        truncated = self.steps == c.horizon and not self.failed
        dwell = min(c.horizon, math.ceil(c.final_dwell_seconds / c.control_dt))
        info = {
            **metrics,
            "success": bool(truncated and self.held_steps >= dwell),
            "score": self.episode_return / c.horizon,
            "survival": self.steps / c.horizon,
            "numerical_failure": numerical_failure,
            "recovery": [
                {
                    "kick_step": k,
                    "seconds": ((v if v is not None else self.steps) - k) * c.control_dt,
                    "censored": v is None,
                }
                for k, v in self.recovery_steps.items()
            ],
        }
        observation = (
            self.observation() if not numerical_failure else np.zeros(self.obs_dim, np.float32)
        )
        return observation, float(reward), bool(self.failed), truncated, info

    def state_dict(self):
        integration = np.empty(
            self.mujoco.mj_stateSize(self.physics.model.ptr, self.integration_spec)
        )
        self.mujoco.mj_getState(
            self.physics.model.ptr, self.physics.data.ptr, integration, self.integration_spec
        )
        return {
            "integration": integration,
            "warmstart": self.physics.data.qacc_warmstart.copy(),
            "case": copy.deepcopy(self.case),
            "rng": self.rng.get_state(),
            "steps": self.steps,
            "held_steps": self.held_steps,
            "return": self.episode_return,
            "failed": self.failed,
            "last_action": self.last_action.copy(),
            "task_steps": self.task._step_counter,
            "prev_action": self.walker.prev_action.copy(),
            "metrics": copy.deepcopy(self.metric_history),
            "recovery": copy.deepcopy(self.recovery_steps),
            "wpg": {
                name: copy.deepcopy(getattr(self.wpg, name))
                for name in ("_step", "_freq_idx", "_ctrl_freq")
            },
        }

    def load_state_dict(self, state):
        self.reset(state["case"]["seed"], state["case"]["split"])
        self.mujoco.mj_setState(
            self.physics.model.ptr,
            self.physics.data.ptr,
            state["integration"],
            self.integration_spec,
        )
        self.physics.forward()
        self.physics.data.qacc_warmstart[:] = state["warmstart"]
        self.rng.set_state(state["rng"])
        for name, key in (
            ("steps", "steps"),
            ("held_steps", "held_steps"),
            ("episode_return", "return"),
            ("failed", "failed"),
            ("last_action", "last_action"),
            ("metric_history", "metrics"),
            ("recovery_steps", "recovery"),
        ):
            setattr(self, name, copy.deepcopy(state[key]))
        self.task._step_counter = state["task_steps"]
        self.walker.prev_action[:] = state["prev_action"]
        for name, value in state["wpg"].items():
            setattr(self.wpg, name, value)
        self.wpg._traj = self.wpg.traj_ctrl[self.wpg._freq_idx]["traj"]
        self.wpg._cycle_len = len(self.wpg._traj)

    def close(self):
        self.physics.free()
