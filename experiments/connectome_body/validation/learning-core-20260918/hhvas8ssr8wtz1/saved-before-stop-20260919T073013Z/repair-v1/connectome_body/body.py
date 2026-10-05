"""Three tasks on one pinned, stock FlyBody walking configuration.

FlyBody uses CGS units. Policy actions are normalized to [-1,1]; their affine
mapping into the stock actuator ranges is fixed. No pretrained teacher, gait,
reference trajectory, anatomical decoder, or brain-dependent observation exists.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import BodyConfig
from .util import digest_json

PROJECT = Path(__file__).resolve().parents[1]
FLYBODY_REVISION = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"


@lru_cache(maxsize=4)
def verify_flybody(source: str):
    source = Path(source).resolve()
    package = source / "flybody"
    if not (package / "fruitfly/assets/fruitfly.xml").is_file():
        raise FileNotFoundError(
            f"Pinned FlyBody source not found at {source}. Run cbbench bootstrap, "
            "or set FLYBODY_SOURCE to an existing pinned checkout."
        )
    digest = hashlib.sha256()
    for path in sorted(p for p in package.rglob("*") if p.suffix in (".py", ".xml", ".obj")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    result = digest.hexdigest()
    lock_path = PROJECT / "configs/sources.json"
    if lock_path.exists():
        expected = json.loads(lock_path.read_text())["flybody"]["content_sha256"]
        if result != expected:
            raise ValueError("FlyBody source differs from the pinned model/code/assets")
    sys.path.insert(0, str(source))
    return result


def scenario(seed: int, split: str, task: str, horizon: int, dt: float, perturbations: bool):
    if split not in ("train", "validation", "test", "ood"):
        raise ValueError("Unknown scenario split")
    rng = np.random.default_rng(seed)
    ood = split == "ood"
    yaw = rng.uniform(-math.pi, math.pi) if ood else rng.uniform(-0.35, 0.35)
    bearing = yaw + rng.uniform(-math.pi if ood else -0.8, math.pi if ood else 0.8)
    distance = rng.uniform(1.2, 2.0) if ood else rng.uniform(0.6, 1.2)
    speed = rng.uniform(3.0, 5.0) if ood else rng.uniform(1.5, 3.0)
    command = speed * np.array([math.cos(bearing), math.sin(bearing)])
    if task != "walk":
        command[:] = 0
    target = distance * np.array([math.cos(bearing), math.sin(bearing)])
    if task != "reach":
        target[:] = 0
    kicks = {}
    if perturbations:
        for fraction in (0.25, 0.625):
            direction = rng.uniform(-math.pi, math.pi)
            amplitude = rng.uniform(1.0, 2.0) * (2 if ood else 1)
            kicks[int(horizon * fraction)] = [
                amplitude * math.cos(direction),
                amplitude * math.sin(direction),
            ]
    return {
        "yaw": float(yaw),
        "target": target,
        "command": command,
        "roll": float(rng.uniform(-0.10, 0.10) * (2 if ood else 1)),
        "pitch": float(rng.uniform(-0.10, 0.10) * (2 if ood else 1)),
        "kicks": kicks,
        "split": split,
        "seed": int(seed),
    }


def quaternion(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def task_reward(task, metrics):
    upright = np.clip((metrics["upright"] - 0.2) / 0.8, 0, 1)
    height = math.exp(-(((metrics["height"] - 0.1278) / 0.07) ** 2))
    if task == "balance":
        progress = math.exp(-((metrics["speed"] / 1.0) ** 2) - (metrics["distance"] / 0.25) ** 2)
        accomplished = (
            metrics["upright"] > 0.85
            and metrics["speed"] < 0.5
            and metrics["distance"] < 0.15
            and metrics["height"] > 0.07
        )
    elif task == "walk":
        progress = math.exp(-((metrics["velocity_error"] / 1.5) ** 2))
        accomplished = (
            metrics["upright"] > 0.8
            and metrics["velocity_error"] < 0.75
            and metrics["height"] > 0.07
        )
    else:
        progress = math.exp(-((metrics["distance"] / 0.5) ** 2))
        accomplished = (
            metrics["upright"] > 0.8
            and metrics["distance"] < 0.15
            and metrics["speed"] < 0.5
            and metrics["height"] > 0.07
        )
    effort = max(0.0, 1 - 0.05 * metrics["action_squared"])
    return float(upright * height * progress * effort), bool(accomplished)


class FlyBodyEnv:
    def __init__(self, config: BodyConfig):
        self.config = config
        config.validate()
        source = config.source or os.environ.get(
            "FLYBODY_SOURCE", str(PROJECT / "references/flybody")
        )
        source_digest = verify_flybody(source)
        # These tasks use proprioception and geometry, so no window, renderer,
        # OpenGL context, or display server is needed on training workers.
        os.environ.setdefault("MUJOCO_GL", "disable")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
        import mujoco
        from dm_control import mjcf
        from dm_control.locomotion.arenas import floors
        from dm_control.rl.control import PhysicsError
        from flybody.fruitfly.fruitfly import FruitFly
        from flybody.tasks.base import Walking

        self.mujoco = mujoco
        self.PhysicsError = PhysicsError

        class BenchmarkWalking(Walking):
            def get_reward_factors(self, physics):
                # The outer environment computes the documented task reward.
                return (1.0,)

        self.task = BenchmarkWalking(
            walker=FruitFly,
            arena=floors.Floor(size=(20.0, 20.0)),
            time_limit=config.horizon * config.control_dt,
            disable_wings=True,
            joint_filter=0.01,
            adhesion_filter=0.007,
            force_actuators=False,
            add_ghost=False,
        )
        self.walker = self.task.walker
        self.rng = np.random.RandomState(0)
        self.task.initialize_episode_mjcf(self.rng)
        self.physics = mjcf.Physics.from_mjcf_model(self.task.root_entity.mjcf_model)
        spec = self.walker.get_action_spec(self.physics)
        self.low, self.high = spec.minimum.copy(), spec.maximum.copy()
        self.action_dim = len(self.low)
        self.action_names = spec.name.split("\t")
        self.thorax = self.physics.model.name2id("walker/thorax", "body")
        self.joints = self.walker.mjcf_model.find_all("joint")
        self.touch_sensors = list(self.walker.mjcf_model.sensor.touch)
        self.native_substeps = round(0.002 / self.physics.timestep())
        self.integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.obs_schema = []
        self.reset(0)
        self.obs_dim = len(self.observation())
        self.fingerprint = digest_json(
            {
                "backend": "flybody",
                "source": source_digest,
                "revision": FLYBODY_REVISION,
                "mujoco": mujoco.__version__,
                "dm_control": importlib.metadata.version("dm-control"),
                "xml": hashlib.sha256(self.physics.model.to_xml_string().encode()).hexdigest()
                if hasattr(self.physics.model, "to_xml_string")
                else hashlib.sha256(
                    self.task.root_entity.mjcf_model.to_xml_string().encode()
                ).hexdigest(),
                "obs_schema": self.obs_schema,
                "action_names": self.action_names,
                "low": self.low.tolist(),
                "high": self.high.tolist(),
                "horizon": config.horizon,
                "action_repeat": config.action_repeat,
                "perturbations": config.perturbations,
                "task": config.task,
            }
        )

    def reset(self, seed: int, split: str = "train"):
        self.rng = np.random.RandomState(seed)
        self.case = scenario(
            seed,
            split,
            self.config.task,
            self.config.horizon,
            self.config.control_dt,
            self.config.perturbations,
        )
        self.steps, self.held_steps, self.episode_return = 0, 0, 0.0
        self.failed = False
        with self.physics.reset_context():
            self.physics.data.qpos[:] = self.physics.model.qpos0
            self.physics.data.qvel[:] = 0
            self.physics.data.act[:] = 0
            self.physics.data.ctrl[:] = 0
            self.physics.data.time = 0
            self.walker.initialize_episode(self.physics, self.rng)
            self.task.initialize_episode(self.physics, self.rng)
            pose = self.walker.upright_pose
            self.walker.set_pose(
                self.physics,
                position=pose.xpos,
                quaternion=quaternion(self.case["roll"], self.case["pitch"], self.case["yaw"]),
            )
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        return self.observation()

    def observation(self):
        bind = self.physics.bind(self.walker.root_body)
        rotation = np.asarray(bind.xmat).reshape(3, 3)
        pos = np.asarray(bind.xpos)
        twist = np.empty(6)
        self.mujoco.mj_objectVelocity(
            self.physics.model.ptr,
            self.physics.data.ptr,
            self.mujoco.mjtObj.mjOBJ_BODY,
            self.thorax,
            twist,
            1,
        )
        local_angular, local_velocity = twist[:3], twist[3:]
        target = rotation.T @ np.r_[self.case["target"] - pos[:2], 0.0]
        command = rotation.T @ np.r_[self.case["command"], 0.0]
        joint = self.physics.bind(self.joints)
        parts = [
            ("joint_position_rad", np.asarray(joint.qpos) / math.pi),
            ("joint_velocity_50_rad_s", np.asarray(joint.qvel) / 50.0),
            ("actuator_activation", np.asarray(self.physics.data.act)),
            ("world_up_in_body", rotation[2]),
            ("body_linear_velocity_5_cm_s", local_velocity / 5.0),
            ("body_angular_velocity_50_rad_s", local_angular / 50.0),
            ("thorax_height_0.1278_cm", np.array([pos[2] / 0.1278])),
            ("target_displacement_cm", target[:2]),
            ("velocity_command_5_cm_s", command[:2] / 5.0),
            ("task_one_hot", np.eye(3)[("balance", "walk", "reach").index(self.config.task)]),
        ]
        # Raw touch forces vary by orders of magnitude; this fixed transform
        # needs no learned normalizer or privileged information.
        if self.touch_sensors:
            touch = np.asarray(self.physics.bind(self.touch_sensors).sensordata)
            parts.append(("touch_log1p_100_dyne", np.log1p(np.maximum(touch, 0) * 100.0)))
        self.obs_schema = [(name, int(np.size(value))) for name, value in parts]
        obs = np.concatenate([np.asarray(value).ravel() for _, value in parts])
        if not np.isfinite(obs).all():
            raise FloatingPointError("Nonfinite FlyBody observation")
        return np.clip(obs, -10, 10).astype(np.float32)

    def metrics(self, action):
        pos = np.asarray(self.physics.bind(self.walker.root_body).xpos)
        rotation = np.asarray(self.physics.bind(self.walker.root_body).xmat).reshape(3, 3)
        velocity = self.physics.data.qvel[:2]
        return {
            "upright": float(rotation[2, 2]),
            "height": float(pos[2]),
            "speed": float(np.linalg.norm(velocity)),
            "distance": float(np.linalg.norm(self.case["target"] - pos[:2])),
            "velocity_error": float(np.linalg.norm(velocity - self.case["command"])),
            "action_squared": float(np.mean(np.square(action))),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if (
            action.shape != (self.action_dim,)
            or not np.isfinite(action).all()
            or np.max(np.abs(action)) > 1.000001
        ):
            raise ValueError("Invalid canonical action")
        if self.steps >= self.config.horizon or self.failed:
            raise RuntimeError("Reset a completed environment before stepping")
        if self.steps in self.case["kicks"]:
            self.physics.data.qvel[:2] += self.case["kicks"][self.steps]
        real_action = self.low + (action + 1) * 0.5 * (self.high - self.low)
        self.walker.apply_action(self.physics, real_action, self.rng)
        numerical_failure = False
        try:
            self.physics.step(self.native_substeps * self.config.action_repeat)
            if (
                not np.isfinite(self.physics.data.qacc).all()
                or np.linalg.norm(self.physics.data.qacc) > 1e14
            ):
                raise FloatingPointError("Invalid generalized acceleration")
            obs = self.observation()
            metrics = self.metrics(action)
        except (FloatingPointError, self.PhysicsError):
            # Controller-induced physics failures count as failed episodes;
            # they are never dropped from performance/stability denominators.
            numerical_failure = True
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            metrics = {
                "upright": -1.0,
                "height": 0.0,
                "speed": 0.0,
                "distance": 0.0,
                "velocity_error": 0.0,
                "action_squared": float(np.mean(action**2)),
            }
        self.steps += 1
        self.last_action = action.copy()
        reward, accomplished = task_reward(self.config.task, metrics)
        self.failed = metrics["upright"] < 0.2 or metrics["height"] < 0.035
        if self.failed:
            reward, accomplished = 0.0, False
        self.held_steps = self.held_steps + 1 if accomplished else 0
        self.episode_return += reward
        truncated = self.steps >= self.config.horizon and not self.failed
        # Success requires performance throughout the LAST 0.25 seconds.
        hold = min(self.config.horizon, math.ceil(0.25 / self.config.control_dt))
        info = {
            **metrics,
            "success": bool(truncated and self.held_steps >= hold),
            "episode_return": self.episode_return,
            "score": self.episode_return / self.config.horizon,
            "steps": self.steps,
            "numerical_failure": numerical_failure,
            "fell": bool(self.failed),
        }
        return obs, reward, self.failed, truncated, info

    def state_dict(self):
        state = np.empty(self.mujoco.mj_stateSize(self.physics.model.ptr, self.integration_spec))
        self.mujoco.mj_getState(
            self.physics.model.ptr, self.physics.data.ptr, state, self.integration_spec
        )
        return {
            "integration": state,
            "rng": self.rng.get_state(),
            "case": copy.deepcopy(self.case),
            "steps": self.steps,
            "held_steps": self.held_steps,
            "episode_return": self.episode_return,
            "failed": self.failed,
            "last_action": self.last_action.copy(),
            "prev_action": self.walker.prev_action.copy(),
        }

    def load_state_dict(self, state):
        self.rng.set_state(state["rng"])
        for key in ("case", "steps", "held_steps", "episode_return", "failed", "last_action"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.mujoco.mj_setState(
            self.physics.model.ptr,
            self.physics.data.ptr,
            state["integration"],
            self.integration_spec,
        )
        warmstart = self.physics.data.qacc_warmstart.copy()
        self.physics.forward()
        self.physics.data.qacc_warmstart[:] = warmstart
        self.walker._prev_action[:] = state["prev_action"]

    def close(self):
        self.physics.free()


class FixtureEnv:
    """Small analytic controller fixture. Never classified as a FlyBody result."""

    obs_dim, action_dim = 12, 3

    def __init__(self, config: BodyConfig):
        self.config = config
        self.fingerprint = digest_json(
            {
                "backend": "synthetic_fixture_v1",
                "task": config.task,
                "horizon": config.horizon,
                "dt": config.control_dt,
            }
        )
        self.obs_schema = [("synthetic_fixture", 12)]
        self.action_names = ["force_x", "force_y", "balance_torque"]
        self.reset(0)

    def reset(self, seed, split="train"):
        self.rng = np.random.default_rng(seed)
        self.case = scenario(
            seed,
            split,
            self.config.task,
            self.config.horizon,
            self.config.control_dt,
            self.config.perturbations,
        )
        self.position, self.velocity = np.zeros(2), np.zeros(2)
        self.angle, self.omega = self.rng.uniform(-0.1, 0.1), 0.0
        self.steps, self.held_steps, self.episode_return = 0, 0, 0.0
        self.failed = False
        return self.observation()

    def observation(self):
        return np.r_[
            self.position,
            self.velocity / 5,
            self.angle,
            self.omega,
            self.case["target"] - self.position,
            self.case["command"] / 5,
            [self.config.task == "walk", self.config.task == "reach"],
        ].astype(np.float32)

    def step(self, action):
        dt = self.config.control_dt
        if self.steps in self.case["kicks"]:
            self.velocity += self.case["kicks"][self.steps]
        self.velocity += dt * (20 * np.asarray(action[:2]) - self.velocity)
        self.position += dt * self.velocity
        self.omega += dt * (3 * self.angle + 8 * action[2] - self.omega)
        self.angle += dt * self.omega
        self.steps += 1
        self.failed = bool(abs(self.angle) > 1.3)
        metrics = {
            "upright": math.cos(self.angle),
            "height": 0.1278,
            "speed": float(np.linalg.norm(self.velocity)),
            "distance": float(np.linalg.norm(self.case["target"] - self.position)),
            "velocity_error": float(np.linalg.norm(self.velocity - self.case["command"])),
            "action_squared": float(np.mean(np.square(action))),
        }
        reward, accomplished = task_reward(self.config.task, metrics)
        self.held_steps = self.held_steps + 1 if accomplished and not self.failed else 0
        self.episode_return += reward
        truncated = self.steps >= self.config.horizon and not self.failed
        hold = min(self.config.horizon, math.ceil(0.25 / dt))
        return (
            self.observation(),
            reward,
            self.failed,
            truncated,
            {
                **metrics,
                "success": bool(truncated and self.held_steps >= hold),
                "episode_return": self.episode_return,
                "score": self.episode_return / self.config.horizon,
                "steps": self.steps,
                "numerical_failure": False,
                "fell": self.failed,
            },
        )

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state))

    def close(self):
        pass


def make_env(config: BodyConfig):
    return FlyBodyEnv(config) if config.backend == "flybody" else FixtureEnv(config)
