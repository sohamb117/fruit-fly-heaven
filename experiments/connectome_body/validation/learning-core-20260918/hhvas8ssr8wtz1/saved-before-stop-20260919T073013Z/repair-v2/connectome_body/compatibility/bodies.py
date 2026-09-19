"""Standardized body contract and tasks for the crossed compatibility experiments.

Fly tasks use the existing pinned native FlyBody implementation. Worm/fish
models must come with a hashed model manifest: a placeholder mesh or toy
swimmer is never silently substituted for a declared biological body.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..adaptation.hover import FlyBodyInterface, HoverConfig
from ..body import FlyBodyEnv, quaternion
from ..config import BodyConfig as GroundConfig
from ..util import digest_file, digest_json, seed_for

BODY_TASKS = {
    "fly": ("hover", "controlled_flight", "walking", "limb_control"),
    "worm": ("locomotion", "steering", "posture"),
    "fish": ("swimming", "heading", "depth"),
}


@dataclass(frozen=True)
class BodySpec:
    name: str = "fly"
    task: str = "hover"
    horizon: int = 5000
    action_repeat: int = 1
    model_manifest: str | None = None
    parameters: dict = field(default_factory=dict)

    def validate(self):
        from .drone import DRONE_TASKS, DroneParameters

        tasks = {**BODY_TASKS, "drone": DRONE_TASKS}
        if self.name not in tasks or self.task not in tasks[self.name]:
            raise ValueError("Task does not belong to the declared body")
        if type(self.horizon) is not int or self.horizon < 8:
            raise ValueError("At least eight body decisions are required")
        if type(self.action_repeat) is not int or self.action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        if self.name in ("worm", "fish") and not self.model_manifest:
            raise ValueError("Worm/fish experiments require a pinned biomechanical model manifest")
        if self.name == "fly" and self.model_manifest:
            raise ValueError("Fly tasks use the pinned native FlyBody assets")
        if not isinstance(self.parameters, dict):
            raise ValueError("Body parameters must be an object")
        if self.name == "drone":
            if self.model_manifest:
                raise ValueError("The engineered drone model is defined by its pinned parameters")
            DroneParameters(**self.parameters).validate()


class FlightBody(FlyBodyInterface):
    def __init__(self, spec: BodySpec):
        spec.validate()
        self.study_spec = spec
        hover_names = set(HoverConfig.__dataclass_fields__) - {"horizon", "action_repeat"}
        unknown = set(spec.parameters) - hover_names - {"target_speed", "trajectory_frequency"}
        if unknown:
            raise ValueError(f"Unknown flight body parameters: {sorted(unknown)}")
        values = {key: value for key, value in spec.parameters.items() if key in hover_names}
        super().__init__(
            HoverConfig(horizon=spec.horizon, action_repeat=spec.action_repeat, **values)
        )
        self.fingerprint = digest_json(
            {"native_body": self.fingerprint, "study_task": asdict(spec)}
        )
        self.evidence = "native_flybody"
        self.control_dt = self.config.control_dt

    def _set_reference(self):
        if self.study_spec.task == "hover":
            return super()._set_reference()
        length = self.config.horizon * self.config.action_repeat + 16
        case = getattr(self, "case", {"seed": 0, "split": "train"})
        rng = np.random.default_rng(seed_for(case["seed"], "controlled-flight-target"))
        angle = rng.uniform(-math.pi, math.pi)
        speed = float(self.study_spec.parameters.get("target_speed", 5.0))
        frequency = float(self.study_spec.parameters.get("trajectory_frequency", 0.5))
        if min(speed, frequency) <= 0 or not np.isfinite([speed, frequency]).all():
            raise ValueError("Positive finite flight target speed/frequency required")
        if case["split"] == "ood":
            speed, frequency = speed * 1.5, frequency * 1.5
        time = np.arange(length) * 0.0002
        heading = np.array([math.cos(angle), math.sin(angle), 0.0])
        lateral = np.array([-heading[1], heading[0], 0.0])
        phase = 2 * math.pi * frequency * time
        self.target_positions = (
            np.array([0.0, 0.0, 1.0])
            + speed * time[:, None] * heading
            + (speed / (8 * math.pi * frequency)) * (1 - np.cos(phase[:, None])) * lateral
        )
        self.target_velocities = speed * heading + speed / 4 * np.sin(phase[:, None]) * lateral
        self.reference_quaternion = quaternion(0, -math.radians(47.5), angle)
        self.target_com = self.target_positions[0].copy()
        qpos = np.column_stack(
            (self.target_positions, np.tile(self.reference_quaternion, (length, 1)))
        )
        qvel = np.column_stack((self.target_velocities, np.zeros((length, 3))))
        self.loader.set_next_trajectory(qpos, qvel)

    def metrics(self):
        if self.study_spec.task == "controlled_flight":
            index = min(self.steps * self.config.action_repeat, len(self.target_positions) - 1)
            self.target_com = self.target_positions[index]
        result = super().metrics()
        if self.study_spec.task == "controlled_flight":
            velocity = np.asarray(self.physics.bind(self.task._root_joint).qvel)[:3]
            result["absolute_speed"] = result["speed"]
            result["speed"] = float(np.linalg.norm(velocity - self.target_velocities[index]))
            result["target_velocity_error"] = result["speed"]
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(episode_return=self.episode_return, steps=self.steps, fell=self.failed)
        return observation, reward, terminated, truncated, info


class GroundBody(FlyBodyEnv):
    def __init__(self, spec: BodySpec):
        spec.validate()
        if spec.parameters:
            raise ValueError("Ground task parameters must be declared in the task implementation")
        self.study_spec = spec
        self.limb_held_steps = 0
        super().__init__(
            GroundConfig(
                task="walk" if spec.task == "walking" else "balance",
                horizon=spec.horizon,
                action_repeat=spec.action_repeat,
            )
        )
        self.control_dt = self.config.control_dt
        self.evidence = "native_flybody"
        if spec.task == "limb_control":
            # Hinge joints belonging to legs, selected solely from the BODY's
            # declared joint names. This supplies task goals, never neural labels.
            names = [
                self.physics.model.id2name(i, "joint") or "" for i in range(self.physics.model.njnt)
            ]
            selected = [
                i
                for i, name in enumerate(names)
                if any(part in name.lower() for part in ("coxa", "femur", "tibia", "tarsus"))
                and int(self.physics.model.jnt_type[i]) == 3
            ]
            if not selected:
                raise ValueError("Pinned FlyBody has no declared leg hinge joints")
            self.limb_qpos = self.physics.model.jnt_qposadr[selected].copy()
            self.limb_center = self.physics.model.qpos0[self.limb_qpos].copy()
            self.limb_phase = np.arange(len(selected)) * 2 * math.pi / len(selected)
            self.obs_dim = len(self.observation())
        self.fingerprint = digest_json(
            {"native_body": self.fingerprint, "study_task": asdict(spec)}
        )

    def _limb_target(self):
        phase = 2 * math.pi * self.steps * self.control_dt + self.limb_phase
        return self.limb_center + 0.25 * np.sin(phase), 0.25 * 2 * math.pi * np.cos(phase)

    def observation(self):
        base = super().observation()
        if not hasattr(self, "limb_qpos"):
            return base
        self.obs_schema += [
            ("limb_target_error", len(self.limb_qpos)),
            ("limb_target_velocity", len(self.limb_qpos)),
        ]
        target, velocity = self._limb_target()
        return np.r_[base, target - self.physics.data.qpos[self.limb_qpos], velocity].astype(
            np.float32
        )

    def step(self, action):
        previous_return = self.episode_return
        obs, reward, terminated, truncated, info = super().step(action)
        if self.study_spec.task == "limb_control":
            target, _ = self._limb_target()
            error = float(np.sqrt(np.mean((target - self.physics.data.qpos[self.limb_qpos]) ** 2)))
            reward = 0.0 if terminated else float(math.exp(-((error / 0.15) ** 2)))
            self.episode_return = previous_return + reward
            self.limb_held_steps = self.limb_held_steps + 1 if error < 0.1 and not terminated else 0
            info.update(
                limb_angle_rmse=error,
                episode_return=self.episode_return,
                score=self.episode_return / self.config.horizon,
                success=bool(truncated and self.limb_held_steps * self.control_dt >= 0.25),
            )
        return obs, reward, terminated, truncated, info

    def reset(self, seed, split="train"):
        self.limb_held_steps = 0
        return super().reset(seed, split)

    def state_dict(self):
        return {**super().state_dict(), "limb_held_steps": self.limb_held_steps}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.limb_held_steps = state["limb_held_steps"]


class ManifestBody:
    """Native MuJoCo worm/fish model with common task, snapshot, and action contracts."""

    def __init__(self, spec: BodySpec):
        import mujoco

        spec.validate()
        path = Path(spec.model_manifest).resolve()
        manifest = json.loads(path.read_text())
        required = (
            "schema",
            "body",
            "species",
            "source",
            "files",
            "model",
            "root_body",
            "body_length",
            "forward_axis",
            "control_dt",
            "evidence",
        )
        if (
            any(name not in manifest for name in required)
            or manifest["schema"] != "mujoco-embodiment-v1"
        ):
            raise ValueError("Incomplete biomechanical model manifest")
        if manifest["body"] != spec.name:
            raise ValueError("Model manifest belongs to a different body")
        if manifest["model"] not in manifest["files"]:
            raise ValueError("Root model XML is not checksummed")
        for name, expected in manifest["files"].items():
            candidate = (path.parent / name).resolve()
            if not candidate.is_relative_to(path.parent) or digest_file(candidate) != expected:
                raise ValueError(
                    "Body asset is missing, changed, or outside its manifest directory"
                )
        self.spec = self.config = spec
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path.parent / manifest["model"]))
        # The compiled digest includes all resolved meshes, textures, includes,
        # and plugin settings, even if a source manifest omitted a dependency.
        compiled = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=compiled)
        compiled_digest = hashlib.sha256(compiled.tobytes()).hexdigest()
        self.data = mujoco.MjData(self.model)
        self.root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, manifest["root_body"]
        )
        if self.root_id < 1 or self.model.nu < 1:
            raise ValueError("Body needs an articulated root and native actuators")
        self.length = float(manifest["body_length"])
        self.control_dt = float(manifest["control_dt"]) * spec.action_repeat
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.forward_axis = np.asarray(manifest["forward_axis"], dtype=float)
        if (
            self.length <= 0
            or self.substeps < 1
            or not np.isfinite(self.length)
            or not math.isclose(self.control_dt, self.substeps * self.model.opt.timestep)
            or self.forward_axis.shape != (3,)
            or not np.isclose(np.linalg.norm(self.forward_axis), 1)
        ):
            raise ValueError("Invalid body scale, time grid, or unit forward axis")
        unknown = set(spec.parameters) - {
            "target_speed",
            "target_depth",
            "initial_noise",
            "dwell_seconds",
        }
        if unknown:
            raise ValueError(f"Unknown body task parameters: {sorted(unknown)}")
        for name, value in spec.parameters.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Body task parameter {name} must be a finite number")
            if name in ("target_speed", "initial_noise") and value < 0:
                raise ValueError(f"Body task parameter {name} must be nonnegative")
            if name == "dwell_seconds" and value <= 0:
                raise ValueError("dwell_seconds must be positive")
        self.low, self.high = self.model.actuator_ctrlrange.T.copy()
        if (
            not np.all(self.model.actuator_ctrllimited)
            or np.any(self.high <= self.low)
            or not np.isfinite([self.low, self.high]).all()
        ):
            raise ValueError("Every native actuator needs a finite ordered control range")
        self.action_dim = self.model.nu
        self.action_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.action_dim)
        ]
        self.hinges = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        if not len(self.hinges):
            raise ValueError("Body needs native articulated hinge joints")
        self.qpos_indices = self.model.jnt_qposadr[self.hinges]
        self.qvel_indices = self.model.jnt_dofadr[self.hinges]
        self.reference_angles = self.model.qpos0[self.qpos_indices].copy()
        self.integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.evidence = manifest["evidence"]
        self.model_manifest = manifest
        portable_task = asdict(spec)
        portable_task["model_manifest"] = None
        self.fingerprint = digest_json(
            {
                "manifest": manifest,
                "task": portable_task,
                "compiled_model_sha256": compiled_digest,
                "mujoco_version": mujoco.__version__,
            }
        )
        self.reset(0)
        self.obs_dim = len(self.observation())
        self.obs_schema = [
            {"name": "hinge_position", "size": len(self.hinges)},
            {"name": "hinge_velocity", "size": len(self.hinges)},
            {"name": "actuator_activation", "size": self.model.na},
            {"name": "body_frame_velocity_angular_up_target", "size": 12},
            {"name": "target_speed_and_depth_error", "size": 2},
        ]

    def reset(self, seed, split="train"):
        if split not in ("train", "validation", "test", "ood"):
            raise ValueError("Unknown scenario split")
        self.case = {"seed": int(seed), "split": split}
        self.rng = np.random.default_rng(seed_for(seed, f"{self.spec.name}-{split}"))
        self.mujoco.mj_resetData(self.model, self.data)
        noise = float(self.spec.parameters.get("initial_noise", 0.05))
        self.data.qpos[self.qpos_indices] += self.rng.normal(0, noise, len(self.hinges))
        self.steps = self.held_steps = 0
        self.episode_return = 0.0
        self.failed = False
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.mujoco.mj_forward(self.model, self.data)
        self.origin = self.position().copy()
        angle = self.rng.uniform(-math.pi, math.pi)
        self.target_direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        self.target_speed = float(self.spec.parameters.get("target_speed", 1.0))
        self.target_depth = float(
            self.spec.parameters.get("target_depth", self.origin[2] / self.length)
        )
        if split == "ood":
            self.target_speed *= 1.5
            self.target_depth += self.rng.uniform(-1, 1)
        if self.spec.task in ("posture", "depth"):
            self.target_speed = 0.0
        self.warning_counts = np.array([w.number for w in self.data.warning])
        return self.observation()

    def _kinematics(self):
        rotation = self.data.xmat[self.root_id].reshape(3, 3)
        velocity = np.empty(6)
        self.mujoco.mj_objectVelocity(
            self.model,
            self.data,
            self.mujoco.mjtObj.mjOBJ_XBODY,
            self.root_id,
            velocity,
            1,
        )
        return rotation, velocity

    def position(self):
        return self.data.xpos[self.root_id]

    def observation(self):
        rotation, velocity = self._kinematics()
        depth_error = self.target_depth - self.position()[2] / self.length
        observation = np.r_[
            self.data.qpos[self.qpos_indices] / math.pi,
            self.data.qvel[self.qvel_indices] / 20,
            self.data.act,
            velocity[3:] / self.length,
            velocity[:3] / 20,
            rotation.T @ np.array([0, 0, 1]),
            rotation.T @ self.target_direction,
            self.target_speed,
            depth_error,
        ]
        return np.clip(observation, -20, 20).astype(np.float32)

    def step(self, action):
        if self.failed or self.steps >= self.spec.horizon:
            raise RuntimeError("Reset a completed body episode before stepping")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Malformed actuator command")
        self.last_action = np.clip(action, -1, 1)
        self.data.ctrl[:] = self.low + (self.last_action + 1) / 2 * (self.high - self.low)
        self._advance_physics()
        self.steps += 1
        warnings = np.array([w.number for w in self.data.warning])
        bad_warning_ids = [
            int(self.mujoco.mjtWarning.mjWARN_BADQPOS),
            int(self.mujoco.mjtWarning.mjWARN_BADQVEL),
            int(self.mujoco.mjtWarning.mjWARN_BADQACC),
        ]
        numerical_failure = bool(
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or np.any(warnings[bad_warning_ids] > self.warning_counts[bad_warning_ids])
        )
        self.warning_counts = warnings
        if numerical_failure:
            heading_error, speed_error, depth_error, posture_error, upright_error = (
                math.pi,
                100.0,
                100.0,
                math.pi,
                2.0,
            )
        else:
            rotation, velocity = self._kinematics()
            forward = rotation @ self.forward_axis
            heading_error = math.acos(float(np.clip(forward @ self.target_direction, -1, 1)))
            world_velocity = rotation @ velocity[3:] / self.length
            speed_error = float(
                np.linalg.norm(world_velocity - self.target_speed * self.target_direction)
            )
            depth_error = float(abs(self.position()[2] / self.length - self.target_depth))
            posture_error = float(
                np.sqrt(np.mean((self.data.qpos[self.qpos_indices] - self.reference_angles) ** 2))
            )
            upright_error = float(1 - rotation[2, 2])
        if self.spec.task == "posture":
            errors, success = (
                (posture_error / 0.2) ** 2 + speed_error**2,
                posture_error < 0.15 and speed_error < 0.2,
            )
        elif self.spec.task == "depth":
            errors, success = (
                (depth_error / 0.5) ** 2 + upright_error**2,
                depth_error < 0.2 and upright_error < 0.2,
            )
        else:
            errors = (speed_error / 0.5) ** 2 + (heading_error / 0.5) ** 2
            success = speed_error < 0.25 and heading_error < 0.3
            if self.spec.task in ("heading", "steering"):
                errors = (heading_error / 0.4) ** 2 + 0.25 * (speed_error / 0.5) ** 2
        self.failed = (
            numerical_failure
            or float(np.linalg.norm(self.position() - self.origin)) > 100 * self.length
        )
        reward = (
            0.0
            if self.failed
            else math.exp(-errors) * (1 - 0.01 * float(np.mean(self.last_action**2)))
        )
        self.episode_return += reward
        self.held_steps = self.held_steps + 1 if success and not self.failed else 0
        truncated = self.steps == self.spec.horizon and not self.failed
        dwell = min(
            self.spec.horizon,
            math.ceil(float(self.spec.parameters.get("dwell_seconds", 0.25)) / self.control_dt),
        )
        info = {
            "steps": self.steps,
            "success": bool(truncated and self.held_steps >= dwell),
            "score": self.episode_return / self.spec.horizon,
            "episode_return": self.episode_return,
            "survival": self.steps / self.spec.horizon,
            "numerical_failure": numerical_failure,
            "fell": self.failed,
            "heading_error": heading_error,
            "speed_error": speed_error,
            "depth_error": depth_error,
            "posture_error": posture_error,
        }
        observation = (
            np.zeros(self.obs_dim, np.float32) if numerical_failure else self.observation()
        )
        return observation, reward, self.failed, truncated, info

    def _advance_physics(self):
        self.mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def state_dict(self):
        integration = np.empty(self.mujoco.mj_stateSize(self.model, self.integration_spec))
        self.mujoco.mj_getState(self.model, self.data, integration, self.integration_spec)
        values = {
            name: copy.deepcopy(getattr(self, name))
            for name in (
                "case",
                "steps",
                "held_steps",
                "episode_return",
                "failed",
                "last_action",
                "origin",
                "target_direction",
                "target_speed",
                "target_depth",
                "warning_counts",
            )
        }
        return {
            "body_fingerprint": self.fingerprint,
            "integration": integration,
            "warmstart": self.data.qacc_warmstart.copy(),
            "values": values,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if state["body_fingerprint"] != self.fingerprint:
            raise ValueError("Cannot restore a different body/task")
        self.reset(state["values"]["case"]["seed"], state["values"]["case"]["split"])
        self.mujoco.mj_setState(self.model, self.data, state["integration"], self.integration_spec)
        self.mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = state["warmstart"]
        for name, value in state["values"].items():
            setattr(self, name, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        for warning, count in zip(self.data.warning, self.warning_counts, strict=True):
            warning.number = count

    def close(self):
        self.data = self.model = None


def make_body(spec: BodySpec):
    spec.validate()
    if spec.name == "drone":
        from .drone import DroneBody

        return DroneBody(spec)
    if spec.name == "fly":
        if spec.task in ("hover", "controlled_flight"):
            return FlightBody(spec)
        return GroundBody(spec)
    manifest = json.loads(Path(spec.model_manifest).read_text())
    if manifest.get("schema") == "published-worm-embodiment-v1":
        from .worm_body import PublishedWormBody

        return PublishedWormBody(spec)
    if manifest.get("backend") == "simzfish-3d-extension-v1":
        from .fish_body import FishBody

        return FishBody(spec)
    return ManifestBody(spec)
