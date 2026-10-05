"""Three control tasks using the published 50-segment C. elegans mechanics.

The body retains viscoelastic and muscle state. It contains no neural circuit,
oscillator, time/phase observation, or automatic steering/posture controller.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict

import numpy as np

from ..util import digest_json, seed_for
from .worm_native import WormEngine


class PublishedWormBody:
    def __init__(self, spec):
        spec.validate()
        if spec.name != "worm":
            raise ValueError("Published worm mechanics require a worm task")
        self.spec = self.config = spec
        allowed = {
            "target_speed",
            "initial_noise",
            "dwell_seconds",
            "posture_amplitude",
            "heading_span",
            "disturbance_amplitude",
        }
        if set(spec.parameters) - allowed:
            raise ValueError(
                f"Unknown worm task parameters: {sorted(set(spec.parameters) - allowed)}"
            )
        for name, value in spec.parameters.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Worm task parameter {name} must be a finite number")
            if value < 0 or (
                name in ("target_speed", "dwell_seconds", "posture_amplitude") and value == 0
            ):
                raise ValueError(f"Worm task parameter {name} is outside its positive range")
            if name in ("initial_noise", "disturbance_amplitude") and value > 1:
                raise ValueError("Muscle perturbation amplitude must be in [0, 1]")
            if name == "heading_span" and value > math.pi:
                raise ValueError("Initial heading span must not exceed pi")
        self.engine = WormEngine(spec.model_manifest)
        try:
            self.model_manifest = self.engine.manifest
            expected = {
                "body_length": 0.001,
                "control_dt": 0.01,
                "physics_dt": 0.001,
                "muscles": 48,
                "segments": 50,
                "rods": 51,
                "muscle_time_constant": 0.1,
            }
            if any(self.model_manifest.get(k) != v for k, v in expected.items()):
                raise ValueError("Manifest does not describe the compiled published worm constants")
            self.length = 0.001
            self.control_dt = 0.01 * spec.action_repeat
            self.substeps = 10 * spec.action_repeat
            self.action_dim = 48
            self.action_names = [
                f"{side}_muscle_{i:02d}" for side in ("dorsal", "ventral") for i in range(1, 25)
            ]
            self.indices = np.round(np.linspace(1, 48, 24)).astype(int)
            self.obs_schema = [
                {"name": "segment_curvature", "size": 24},
                {"name": "segment_curvature_velocity", "size": 24},
                {"name": "dorsal_ventral_strain", "size": 48},
                {"name": "filtered_muscle_activation", "size": 48},
                {"name": "body_frame_velocity", "size": 2},
                {"name": "egocentric_target_direction", "size": 2},
                {"name": "target_speed_body_lengths_per_second", "size": 1},
                {"name": "target_curvature", "size": 24},
            ]
            self.obs_dim = sum(field["size"] for field in self.obs_schema)
            self.actuator_capacity = np.ones(48)
            self.evidence = "published_worm_mechanics"
            self.task_parameters = {
                **self.model_manifest["task_defaults"],
                "initial_noise": 0.15,
                "heading_span": math.pi / 3,
                "disturbance_amplitude": 0.15,
                **spec.parameters,
            }
            portable = asdict(spec)
            portable["model_manifest"] = None
            self.fingerprint = digest_json(
                {
                    "manifest": self.model_manifest,
                    "task": portable,
                    "task_parameters": self.task_parameters,
                    "observation_schema": self.obs_schema,
                    "task_interface_version": 1,
                }
            )
            self.reset(0)
        except BaseException:
            self.engine.close()
            raise

    def _shape(self):
        angles = np.diff(self.engine.rods[:, 2])
        return np.arctan2(np.sin(angles), np.cos(angles))[self.indices]

    def _frame(self):
        # Rod zero is the head; initial forward direction is negative world x.
        forward = self.engine.rods[0, :2] - self.engine.rods[-1, :2]
        forward = forward / max(np.linalg.norm(forward), 1e-12)
        return np.column_stack((forward, [-forward[1], forward[0]]))

    def reset(self, seed, split="train"):
        if split not in ("train", "validation", "test", "ood"):
            raise ValueError("Unknown scenario split")
        self.case = {"seed": int(seed), "split": split}
        self.rng = np.random.default_rng(seed_for(seed, f"worm-{split}"))
        self.engine.reset()
        # Reach a consistent DAE initial condition through the actual mechanics.
        # Arbitrary edits of the rod coordinates would invalidate the solver.
        phase = self.rng.uniform(-math.pi, math.pi)
        profile = np.sin(np.linspace(0, 2 * math.pi, 24) + phase)
        amplitude = self.task_parameters["initial_noise"]
        if split == "ood":
            amplitude = min(1.0, 1.5 * amplitude)
        drive = amplitude * np.r_[np.maximum(profile, 0), np.maximum(-profile, 0)]
        if not self.engine.step(drive, self.actuator_capacity, 0.001, 200):
            raise RuntimeError("Published worm initialization became nonfinite")
        self.origin = self.engine.rods[:, :2].mean(0).copy()
        self.previous_center = self.origin.copy()
        self.previous_shape = self._shape()
        self.velocity = np.zeros(2)
        self.curvature_velocity = np.zeros(24)
        forward = self._frame()[:, 0]
        angle = math.atan2(forward[1], forward[0])
        if self.spec.task == "steering":
            # A nonzero turn is required; straight passive relaxation cannot win.
            span = self.task_parameters["heading_span"]
            angle += self.rng.choice([-1, 1]) * self.rng.uniform(span / 2, span)
        self.target_direction = np.array([math.cos(angle), math.sin(angle)])
        self.target_speed = self.task_parameters["target_speed"] * (1.5 if split == "ood" else 1)
        self.target_curvature = np.zeros(24)
        if self.spec.task == "posture":
            self.target_speed = 0.0
            # Hold a prescribed bend, not merely the passive straight equilibrium.
            self.target_curvature = self.task_parameters["posture_amplitude"] * np.sin(
                np.linspace(0, 2 * math.pi, 24) + self.rng.uniform(-math.pi, math.pi)
            )
        self.disturbance_profile = self.rng.uniform(-1, 1, 24)
        self.steps = self.held_steps = 0
        self.episode_return = 0.0
        self.failed = False
        self.last_action = np.full(48, -1.0, np.float32)
        return self.observation()

    def observation(self):
        frame = self._frame()
        values = np.r_[
            self._shape() / 0.3,
            self.curvature_velocity / 10,
            self.engine.strains[:, self.indices].ravel() / 0.2,
            self.engine.muscles,
            frame.T @ self.velocity,
            frame.T @ self.target_direction,
            self.target_speed,
            self.target_curvature / 0.3,
        ]
        return np.clip(values, -20, 20).astype(np.float32)

    def step(self, action):
        if self.failed or self.steps >= self.spec.horizon:
            raise RuntimeError("Reset a completed worm episode before stepping")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (48,) or not np.isfinite(action).all():
            raise ValueError("Malformed worm muscle command")
        self.last_action = np.clip(action, -1, 1)
        activation = (self.last_action.astype(float) + 1) / 2
        # Exogenous muscle pulses, identically seeded across policies. These are
        # disturbances, not a pattern generator or a feedback controller.
        pulse = max(1, round(0.05 / self.control_dt))
        if any(
            start <= self.steps < start + pulse
            for start in (round(0.25 * self.spec.horizon), round(0.6 * self.spec.horizon))
        ):
            shift = self.task_parameters["disturbance_amplitude"] * self.disturbance_profile
            activation = np.clip(activation + np.r_[shift, -shift], 0, 1)
        valid = self.engine.step(activation, self.actuator_capacity, 0.001, self.substeps)
        self.steps += 1
        if valid:
            center, shape = self.engine.rods[:, :2].mean(0), self._shape()
            self.velocity = (center - self.previous_center) / (self.length * self.control_dt)
            self.curvature_velocity = (shape - self.previous_shape) / self.control_dt
            self.previous_center, self.previous_shape = center.copy(), shape.copy()
            heading = math.acos(float(np.clip(self._frame()[:, 0] @ self.target_direction, -1, 1)))
            speed = float(np.linalg.norm(self.velocity - self.target_speed * self.target_direction))
            posture = float(np.sqrt(np.mean((shape - self.target_curvature) ** 2)))
            self.failed = bool(np.linalg.norm(center - self.origin) > 100 * self.length)
        else:
            heading, speed, posture = math.pi, 100.0, math.pi
            self.failed = True
        if self.spec.task == "posture":
            errors = (posture / 0.025) ** 2 + (speed / 0.05) ** 2
            success = posture < 0.01 and speed < 0.03
        else:
            errors = (speed / 0.15) ** 2 + (heading / 0.5) ** 2
            success = speed < 0.1 and heading < 0.3
            if self.spec.task == "steering":
                errors = (heading / 0.4) ** 2 + 0.25 * (speed / 0.15) ** 2
        reward = (
            0.0 if self.failed else math.exp(-errors) * (1 - 0.01 * float(np.mean(activation**2)))
        )
        self.episode_return += reward
        self.held_steps = self.held_steps + 1 if success and not self.failed else 0
        truncated = self.steps == self.spec.horizon and not self.failed
        dwell = min(
            self.spec.horizon, math.ceil(self.task_parameters["dwell_seconds"] / self.control_dt)
        )
        info = {
            "steps": self.steps,
            "success": bool(truncated and self.held_steps >= dwell),
            "score": self.episode_return / self.spec.horizon,
            "episode_return": self.episode_return,
            "survival": self.steps / self.spec.horizon,
            "fell": self.failed,
            "numerical_failure": not valid,
            "heading_error": heading,
            "speed_error": speed,
            "posture_error": posture,
        }
        observation = self.observation() if valid else np.zeros(self.obs_dim, np.float32)
        return observation, reward, self.failed, truncated, info

    def state_dict(self):
        fields = (
            "case",
            "origin",
            "previous_center",
            "previous_shape",
            "velocity",
            "curvature_velocity",
            "target_direction",
            "target_speed",
            "target_curvature",
            "disturbance_profile",
            "steps",
            "held_steps",
            "episode_return",
            "failed",
            "last_action",
        )
        return {
            "body_fingerprint": self.fingerprint,
            "engine": self.engine.state_dict(),
            "actuator_capacity": self.actuator_capacity.copy(),
            "values": {name: copy.deepcopy(getattr(self, name)) for name in fields},
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if state["body_fingerprint"] != self.fingerprint or not np.array_equal(
            state["actuator_capacity"], self.actuator_capacity
        ):
            raise ValueError("Cannot restore a different worm body, task, or actuation condition")
        self.engine.load_state_dict(state["engine"])
        for name, value in state["values"].items():
            setattr(self, name, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def configure_perturbation(self, perturbation):
        if (
            perturbation.mass_scale != 1
            or perturbation.inertia_scale != 1
            or perturbation.fluid_density_scale != 1
            or perturbation.viscosity_scale != 1
            or any(perturbation.wind)
            or perturbation.terrain_slope
        ):
            raise ValueError(
                "Published overdamped agar mechanics have no inertia, fluid, or terrain intervention"
            )
        if any(i >= self.action_dim for i in perturbation.failed_actuators):
            raise ValueError("Failed actuator index outside this body")
        if perturbation.friction_scale != 1:
            replacement = WormEngine(
                self.spec.model_manifest, drag_scale=perturbation.friction_scale
            )
            self.engine.close()
            self.engine = replacement
        self.actuator_capacity[:] = perturbation.actuator_scale
        self.actuator_capacity[list(perturbation.failed_actuators)] = 0

    def structural_descriptors(self):
        return {
            "observation_dimension": self.obs_dim,
            "action_dimension": self.action_dim,
            "control_dt_seconds": self.control_dt,
            "articulated_hinges": 0,
            "moving_bodies": 51,
            "generalized_velocities": 153,
            "actuator_state_dimension": 48,
            "median_passive_damping_time_seconds": 0.0175,
            "fluid_density_native_units": None,
            "fluid_viscosity_native_units": None,
            "viscoelastic_segments": 50,
            "muscle_time_constant_seconds": 0.1,
            "medium_drag_anisotropy": 40.0,
            "task_target_speed_native_units": self.task_parameters["target_speed"]
            if self.spec.task != "posture"
            else 0.0,
            "task_trajectory_frequency_hz": None,
        }

    def close(self):
        self.engine.close()
