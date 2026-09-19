"""Deterministic SI-unit SE(3) hexrotor and twenty sensorimotor tasks.

An engineered rigid body, not a validated digital twin. Individual rotor thrusts
are the policy actions; no controller or estimator is hidden in the environment.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from ..util import digest_json, seed_for

DRONE_TASKS = (
    "attitude",
    "hover",
    "setpoint",
    "waypoints",
    "trajectory",
    "pursuit",
    "gust",
    "turbulence",
    "sensor_delay",
    "sensor_dropout",
    "sensor_noise",
    "actuator_delay",
    "motor_degradation",
    "motor_failure",
    "payload",
    "energy",
    "color_navigation",
    "obstacles",
    "landing",
    "mission",
)
CORE_DRONE = (
    "attitude",
    "hover",
    "trajectory",
    "gust",
    "sensor_delay",
    "sensor_dropout",
    "motor_degradation",
)


@dataclass(frozen=True)
class DroneParameters:
    dt: float = 0.02
    substeps: int = 4
    mass: float = 1.0
    arm: float = 0.22
    max_thrust: float = 5.0
    motor_tau: float = 0.04
    drag: float = 0.15
    angular_drag: float = 0.015
    yaw_coefficient: float = 0.02
    delay: int = 5
    dropout_steps: int = 15
    noise: float = 0.04
    gust_impulse: float = 0.7
    wind_sigma: float = 0.7
    motor_fraction: float = 0.5
    trajectory: str = "mixed"
    radius: float = 0.8
    frequency: float = 0.12
    position_tolerance: float = 0.25
    angle_tolerance: float = 0.2
    recovery_hold: float = 0.3

    def validate(self):
        for name in (
            "dt",
            "mass",
            "arm",
            "max_thrust",
            "motor_tau",
            "radius",
            "frequency",
            "position_tolerance",
            "angle_tolerance",
            "recovery_hold",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Positive finite drone {name} required")
        if type(self.substeps) is not int or self.substeps < 1 or self.dt / self.substeps > 0.01:
            raise ValueError("Drone integration requires positive substeps at most 10 ms")
        for name in ("delay", "dropout_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"Nonnegative integer {name} required")
        for name in (
            "drag",
            "angular_drag",
            "yaw_coefficient",
            "noise",
            "gust_impulse",
            "wind_sigma",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Nonnegative finite {name} required")
        if not 0 <= self.motor_fraction <= 1:
            raise ValueError("Motor fraction must be in [0,1]")
        if self.trajectory not in ("mixed", "circle", "figure_eight", "spiral", "spline"):
            raise ValueError("Unknown trajectory")


class DroneBody:
    evidence = "engineered_se3_hexrotor_v1"
    action_dim = 6
    action_names = tuple(f"rotor_{i}_thrust" for i in range(6))
    obs_schema = [
        ("measured_position", 3),
        ("measured_velocity", 3),
        ("measured_rotation_matrix", 9),
        ("measured_body_rates", 3),
        ("command_position", 3),
        ("command_velocity", 3),
        ("command_heading_cos_sin", 2),
        ("range_rays", 8),
        ("colored_target_positions", 6),
        ("color_cue", 2),
        ("mission_phase", 3),
        ("sensor_valid", 1),
    ]
    obs_dim = sum(n for _, n in obs_schema)

    def __init__(self, spec):
        spec.validate()
        self.config = self.study_spec = spec
        self.p = DroneParameters(**spec.parameters)
        self.p.validate()
        self.control_dt = self.p.dt * spec.action_repeat
        self.fingerprint = digest_json(
            {
                "backend": self.evidence,
                "spec": asdict(spec),
                "parameters": asdict(self.p),
                "obs": self.obs_schema,
            }
        )
        theta = np.arange(6) * np.pi / 3
        self.rotor_positions = self.p.arm * np.c_[np.cos(theta), np.sin(theta), np.zeros(6)]
        self.spin = np.array([1, -1, 1, -1, 1, -1])
        self.nominal_inertia = np.array([0.018, 0.018, 0.032])
        self.reset(0)

    def allocation(self):
        r = self.rotor_positions - self.com
        return np.vstack((np.ones(6), r[:, 1], -r[:, 0], self.spin * self.p.yaw_coefficient))

    def reset(self, seed, split="train"):
        if split not in ("train", "validation", "test", "ood"):
            raise ValueError("Unknown drone split")
        self.case = {"seed": int(seed), "split": split}
        self.rng = np.random.default_rng(seed_for(seed, f"drone-{split}"))
        self.steps = 0
        self.episode_return = 0.0
        self.failed = False
        self.position = np.array([0.0, 0.0, 1.5]) + self.rng.normal(0, 0.08, 3)
        self.velocity = self.rng.normal(0, 0.05, 3)
        angles = self.rng.uniform(
            -0.45 if self.config.task == "attitude" else -0.08,
            0.45 if self.config.task == "attitude" else 0.08,
            3,
        )
        self.rotation = Rotation.from_euler("xyz", angles).as_matrix()
        self.omega = self.rng.normal(0, 0.05, 3)
        self.mass, self.inertia, self.com = self.p.mass, self.nominal_inertia.copy(), np.zeros(3)
        if self.config.task == "payload":
            scale = self.rng.uniform(1.3, 1.6) if split == "ood" else self.rng.uniform(0.8, 1.2)
            self.mass *= scale
            self.inertia *= scale * self.rng.uniform(0.9, 1.1, 3)
            self.com = self.rng.uniform(-0.025, 0.025, 3) * (2 if split == "ood" else 1)
        self.motor_efficiency = np.ones(6)
        self.thrust = np.full(6, self.mass * 9.81 / 6)
        self.last_action = self.thrust / self.p.max_thrust * 2 - 1
        self.wind = np.zeros(3)
        self.event_step = int(
            self.rng.integers(max(1, self.config.horizon // 4), max(2, self.config.horizon // 2))
        )
        self.fault_motor = int(self.rng.integers(6))
        self.phase = float(self.rng.uniform(0, 2 * np.pi))
        self.path_kind = self.p.trajectory
        if self.path_kind == "mixed":
            self.path_kind = str(self.rng.choice(["circle", "figure_eight", "spiral", "spline"]))
        self.frequency = self.p.frequency * (1.4 if split == "ood" else 1)
        self.duration = self.config.horizon * self.control_dt
        knots = np.linspace(0, self.duration + self.control_dt, 7)
        points = self.rng.uniform(-self.p.radius, self.p.radius, (7, 3))
        points[:, 2] = 1.5 + points[:, 2] * 0.3
        self.spline = CubicSpline(knots, points, bc_type="natural")
        self.waypoints = np.array([[0.8, 0, 1.5], [0.8, 0.8, 1.8], [0, 0.8, 1.5], [0, 0, 1.5]])
        self.waypoints[:, :2] = (
            self.waypoints[:, :2] @ Rotation.from_euler("z", self.phase).as_matrix()[:2, :2]
        )
        self.waypoint_index = 0
        self.colors = np.array([[0.8, 0.6, 1.5], [0.8, -0.6, 1.5]])
        self.rng.shuffle(self.colors)
        self.color_cue = int(self.rng.integers(2))
        self.obstacles = np.array([[0.65, 0.18, 1.5, 0.22], [1.1, -0.25, 1.5, 0.22]])
        self.pursuit_position = np.array([0.0, 0.0, 1.5])
        self.pursuit_velocity = np.zeros(3)
        self.platform = np.array([0.8, 0.0, 0.12])
        self.mission_phase = 0
        if self.config.task == "mission":
            self.position = np.array([0.0, 0.0, 0.12])
            self.velocity[:] = 0
            self.rotation = np.eye(3)
            self.omega[:] = 0
            self.thrust[:] = 0
        self.landed = self.collision = self.wrong_target = self.correct_target = False
        self.touchdown_velocity = None
        self.path_length = self.energy = 0.0
        self.history = []
        self.recovered_at = None
        self.recovery_run = 0
        self.event_position = None
        self.action_queue = [self.last_action.copy() for _ in range(self.p.delay + 1)]
        self.sensor_queue = [self._clean_sensor() for _ in range(self.p.delay + 1)]
        self._sensor = self.sensor_queue[-1].copy()
        self.sensor_valid = True
        self._update_goal()
        self._capture_sensor()
        return self.observation()

    def _update_goal(self):
        task, t = self.config.task, self.steps * self.control_dt
        self.goal, self.goal_velocity, self.goal_yaw = np.array([0.0, 0.0, 1.5]), np.zeros(3), 0.0
        if task in ("trajectory", "energy"):
            a, w = self.p.radius, 2 * np.pi * self.frequency
            phase = w * t + self.phase
            if self.path_kind == "spline":
                self.goal, self.goal_velocity = self.spline(t), self.spline(t, 1)
            elif self.path_kind == "figure_eight":
                self.goal += [a * np.sin(phase), a / 2 * np.sin(2 * phase), 0]
                self.goal_velocity = a * w * np.array([np.cos(phase), np.cos(2 * phase), 0])
            else:
                self.goal += [
                    a * np.cos(phase),
                    a * np.sin(phase),
                    0.25 * np.sin(phase / 2) if self.path_kind == "spiral" else 0,
                ]
                self.goal_velocity = a * w * np.array([-np.sin(phase), np.cos(phase), 0])
                if self.path_kind == "spiral":
                    self.goal_velocity[2] = 0.125 * w * np.cos(phase / 2)
        elif task == "setpoint":
            self.goal = self.waypoints[min(3, int(4 * t / max(self.duration, 0.01)))].copy()
            self.goal_yaw = (int(4 * t / max(self.duration, 0.01)) % 4) * np.pi / 2
        elif task == "waypoints":
            self.goal = self.waypoints[min(3, self.waypoint_index)].copy()
        elif task == "pursuit":
            self.goal, self.goal_velocity = (
                self.pursuit_position.copy(),
                self.pursuit_velocity.copy(),
            )
        elif task == "obstacles":
            self.goal = np.array([1.6, 0.0, 1.5])
        elif task == "color_navigation":
            self.goal = self.colors[self.color_cue].copy()
        elif task in ("landing", "motor_failure"):
            if task == "landing" or self.steps >= self.event_step:
                start = 0 if task == "landing" else self.event_step * self.control_dt
                f = np.clip((t - start) / max(0.7 * (self.duration - start), 0.01), 0, 1)
                self.goal = np.r_[self.platform[:2] * min(1, f * 3), max(0.12, 1.5 * (1 - f))]
                self.goal_velocity[2] = (
                    -1.5 / max(0.7 * (self.duration - start), 0.01) if f < 1 else 0
                )
        elif task == "mission":
            if self.mission_phase == 0:
                self.goal = np.array([0.0, 0.0, 1.5])
            elif self.mission_phase == 1:
                self.goal = np.r_[self.platform[:2], 1.5]
            else:
                self.goal = self.platform.copy()

    def _clean_sensor(self):
        return np.r_[self.position, self.velocity, self.rotation.ravel(), self.omega]

    def _capture_sensor(self):
        sensor = self._clean_sensor()
        if self.config.task == "sensor_noise":
            scale = self.p.noise * (2 if self.case["split"] == "ood" else 1)
            sensor[:6] += self.rng.normal(0, scale, 6)
            sensor[6:15] = (
                self.rotation @ Rotation.from_rotvec(self.rng.normal(0, scale, 3)).as_matrix()
            ).ravel()
            sensor[15:] += self.rng.normal(0, scale, 3)
        self.sensor_queue.append(sensor)
        self.sensor_queue = self.sensor_queue[-(self.p.delay + 1) :]
        self._sensor = self.sensor_queue[0 if self.config.task == "sensor_delay" else -1].copy()
        self.sensor_valid = not (
            self.config.task == "sensor_dropout"
            and self.event_step <= self.steps < self.event_step + self.p.dropout_steps
        )
        if not self.sensor_valid:
            self._sensor[:] = 0

    def observation(self):
        # Goals are exogenous commands. Never derive a clean-state tracking error
        # that bypasses the sensor-delay/dropout path.
        ranges = np.ones(8)
        if self.config.task == "obstacles" and self.sensor_valid:
            for i, angle in enumerate(np.arange(8) * np.pi / 4):
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                for obstacle in self.obstacles:
                    offset = obstacle[:3] - self._sensor[:3]
                    along = offset @ direction
                    perpendicular = offset @ offset - along * along
                    if along > 0 and perpendicular < obstacle[3] ** 2:
                        ranges[i] = min(
                            ranges[i], max(0, along - np.sqrt(obstacle[3] ** 2 - perpendicular)) / 3
                        )
        if not self.sensor_valid:
            ranges[:] = 0
        color = self.config.task == "color_navigation"
        phase = np.eye(3)[self.mission_phase] if self.config.task == "mission" else np.zeros(3)
        return np.r_[
            self._sensor,
            np.zeros(3) if color else self.goal,
            self.goal_velocity,
            np.cos(self.goal_yaw),
            np.sin(self.goal_yaw),
            ranges,
            self.colors.ravel() if color else np.zeros(6),
            np.eye(2)[self.color_cue] if color else np.zeros(2),
            phase,
            self.sensor_valid,
        ].astype(np.float32)

    def _physics(self, command):
        dt = self.p.dt / self.p.substeps
        allocation = self.allocation()
        for _ in range(self.p.substeps * self.config.action_repeat):
            self.thrust += (command - self.thrust) * (-np.expm1(-dt / self.p.motor_tau))
            force_torque = allocation @ (self.thrust * self.motor_efficiency)
            acceleration = (
                self.rotation[:, 2] * force_torque[0] + self.wind - self.p.drag * self.velocity
            ) / self.mass
            acceleration[2] -= 9.81
            angular = (
                force_torque[1:]
                - np.cross(self.omega, self.inertia * self.omega)
                - self.p.angular_drag * self.omega
            ) / self.inertia
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
            self.omega += angular * dt
            self.rotation = self.rotation @ Rotation.from_rotvec(self.omega * dt).as_matrix()
            self.energy += (
                float(np.sum(np.maximum(self.thrust * self.motor_efficiency, 0) ** 1.5)) * dt
            )

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Six finite drone actuator commands required")
        action = np.clip(action, -1, 1)
        old_position, old_action = self.position.copy(), self.last_action.copy()
        if self.steps == self.event_step:
            self.event_position = self.position.copy()
            self.recovery_run = 0
            if self.config.task == "gust":
                self.velocity += self.rng.normal(0, self.p.gust_impulse, 3) / self.mass
            if self.config.task in ("motor_degradation", "motor_failure"):
                self.motor_efficiency[self.fault_motor] = (
                    self.p.motor_fraction if self.config.task == "motor_degradation" else 0
                )
        if self.config.task == "turbulence":
            rho = np.exp(-self.control_dt / 0.5)
            self.wind = rho * self.wind + np.sqrt(
                1 - rho * rho
            ) * self.p.wind_sigma * self.rng.normal(size=3)
        if self.config.task == "pursuit":
            self.pursuit_velocity = 0.98 * self.pursuit_velocity + self.rng.normal(0, 0.04, 3)
            self.pursuit_velocity -= 0.04 * (self.pursuit_position - [0, 0, 1.5])
            self.pursuit_position += self.pursuit_velocity * self.control_dt
        self.action_queue.append(action.copy())
        self.action_queue = self.action_queue[-(self.p.delay + 1) :]
        effective = action
        if self.config.task == "actuator_delay":
            effective = self.action_queue[-1 - int(self.rng.integers(self.p.delay + 1))]
        self._physics((effective + 1) * self.p.max_thrust / 2)
        self.steps += 1
        self.last_action = action.copy()
        self.path_length += float(np.linalg.norm(self.position - old_position))
        numerical = not np.isfinite(self._clean_sensor()).all()
        self.collision = bool(
            self.config.task == "obstacles"
            and any(np.linalg.norm(self.position - o[:3]) < o[3] + 0.1 for o in self.obstacles)
        )
        ground = self.position[2] <= 0.12
        allowed_landing = self.config.task in ("landing", "mission", "motor_failure")
        tilt = float(np.arccos(np.clip(self.rotation[2, 2], -1, 1)))
        if (
            ground
            and allowed_landing
            and not (self.config.task == "mission" and self.mission_phase == 0)
        ):
            self.touchdown_velocity = float(np.linalg.norm(self.velocity))
            self.landed = bool(
                np.linalg.norm(self.position[:2] - self.platform[:2]) < 0.25
                and self.touchdown_velocity < 0.6
                and tilt < 0.3
            )
        if (
            ground
            and self.config.task == "mission"
            and self.mission_phase == 0
            and self.steps < self.config.horizon // 3
        ):
            self.position[2], self.velocity[2] = 0.12, max(0, self.velocity[2])
            ground = False
        self.failed = bool(
            numerical
            or self.collision
            or (ground and not self.landed)
            or np.linalg.norm(self.position) > 12
            or tilt > 1.5
        )
        error = float(np.linalg.norm(self.position - self.goal))
        angle = float(
            Rotation.from_matrix(
                Rotation.from_euler("z", self.goal_yaw).as_matrix().T @ self.rotation
            ).magnitude()
        )
        if self.config.task == "waypoints" and error < 0.25:
            self.waypoint_index = min(4, self.waypoint_index + 1)
        if self.config.task == "mission" and error < 0.2 and self.mission_phase < 2:
            self.mission_phase += 1
        if self.config.task == "color_navigation":
            self.correct_target |= error < 0.25
            self.wrong_target |= (
                np.linalg.norm(self.position - self.colors[1 - self.color_cue]) < 0.25
            )
        stable = (
            angle < self.p.angle_tolerance
            if self.config.task == "attitude"
            else error < self.p.position_tolerance and tilt < 0.3
        )
        self.recovery_run = self.recovery_run + 1 if stable else 0
        start = (
            self.event_step
            if self.config.task in ("gust", "motor_degradation", "motor_failure")
            else 0
        )
        if (
            self.steps >= start
            and self.recovery_run * self.control_dt >= self.p.recovery_hold
            and self.recovered_at is None
        ):
            self.recovered_at = max(
                0.0, (self.steps - start) * self.control_dt - self.p.recovery_hold
            )
        reward = float(
            np.exp(-((angle / 0.3) ** 2))
            if self.config.task == "attitude"
            else np.exp(-((error / 0.6) ** 2) - (tilt / 0.5) ** 2)
        )
        if self.config.task == "energy":
            reward *= np.exp(-0.03 * np.mean((self.thrust / self.p.max_thrust) ** 1.5))
        if self.failed or self.wrong_target:
            reward = 0.0
        self.episode_return += reward
        self.history.append(
            {
                "position": self.position.copy(),
                "goal": self.goal.copy(),
                "error": error,
                "angle": angle,
                "action": action.copy(),
                "action_delta": float(np.mean((action - old_action) ** 2)),
                "stable": stable,
                "dropout": self.config.task == "sensor_dropout"
                and self.event_step <= self.steps < self.event_step + self.p.dropout_steps,
            }
        )
        terminated = self.failed or self.landed
        truncated = self.steps >= self.config.horizon
        self._update_goal()
        self._capture_sensor()
        info = (
            self.metrics()
            if terminated or truncated
            else {
                "success": False,
                "score": self.episode_return / self.config.horizon,
                "episode_return": self.episode_return,
                "steps": self.steps,
                "numerical_failure": numerical,
                "fell": self.failed,
            }
        )
        return self.observation(), reward, bool(terminated), bool(truncated), info

    def metrics(self):
        h = self.history
        errors = np.array([r["error"] for r in h])
        angles = np.array([r["angle"] for r in h])
        stable = np.mean([r["stable"] for r in h]) if h else 0.0
        success = not self.failed and stable >= 0.8 and self.steps >= self.config.horizon
        if self.config.task in ("landing", "mission", "motor_failure"):
            success = self.landed
        if self.config.task == "waypoints":
            success = self.waypoint_index == 4 and not self.failed
        if self.config.task == "color_navigation":
            success = self.correct_target and not self.wrong_target and not self.failed
        if self.config.task == "obstacles":
            success = bool(errors.size and errors[-1] < 0.25 and not self.failed)
        phase_lag = None
        if len(h) > 10 and self.config.task in ("trajectory", "energy"):
            actual, target = np.array([r["position"] for r in h]), np.array([r["goal"] for r in h])
            lags = range(min(len(h) // 4, 50) + 1)
            phase_lag = (
                min(lags, key=lambda k: np.mean((actual[k:] - target[: len(h) - k]) ** 2))
                * self.control_dt
            )
        switches = [0] + [
            i for i in range(1, len(h)) if np.linalg.norm(h[i]["goal"] - h[i - 1]["goal"]) > 0.25
        ]
        settling, overshoot = [], []
        if self.config.task == "setpoint":
            for a, b in zip(switches, switches[1:] + [len(h)]):
                good = errors[a:b] < self.p.position_tolerance
                indices = [j for j in range(len(good)) if good[j:].all()]
                settling.append(indices[0] * self.control_dt if indices else None)
                direction = h[a]["goal"] - (h[a - 1]["goal"] if a else np.array([0, 0, 1.5]))
                unit = direction / max(np.linalg.norm(direction), 1e-12)
                overshoot.append(max(0.0, max((r["position"] - r["goal"]) @ unit for r in h[a:b])))
        drops = [r["error"] for r in h if r["dropout"]]
        return {
            "success": bool(success),
            "score": self.episode_return / self.config.horizon,
            "episode_return": self.episode_return,
            "steps": self.steps,
            "numerical_failure": not np.isfinite(self._clean_sensor()).all(),
            "fell": self.failed,
            "position_rmse": float(np.sqrt(np.mean(errors**2))) if h else None,
            "angular_rmse": float(np.sqrt(np.mean(angles**2))) if h else None,
            "survival_seconds": self.steps * self.control_dt,
            "crash": self.failed,
            "recovery_seconds": self.recovered_at,
            "recovery_censored": self.recovered_at is None,
            "path_length": self.path_length,
            "power_proxy_integral": self.energy,
            "power_proxy_per_meter": self.energy / self.path_length
            if self.path_length > 0.01
            else None,
            "control_smoothness": float(np.mean([r["action_delta"] for r in h])) if h else None,
            "action_variance": float(np.var([r["action"] for r in h], axis=0).mean())
            if h
            else None,
            "phase_lag_seconds": phase_lag,
            "settling_seconds": settling,
            "overshoot_meters": overshoot,
            "steady_state_error": float(np.mean(errors[-max(1, len(h) // 5) :])) if h else None,
            "dropout_drift": float(np.mean(drops)) if drops else None,
            "waypoint_completion": self.waypoint_index / 4,
            "collision": self.collision,
            "interception_fraction": float(np.mean(errors < 0.25)) if h else 0.0,
            "correct_target": bool(self.correct_target),
            "wrong_target": bool(self.wrong_target),
            "landing_success": self.landed,
            "touchdown_velocity": self.touchdown_velocity,
            "mission_phase": self.mission_phase,
            "scenario": copy.deepcopy(self.case),
            "peak_position_error": float(errors.max()) if h else None,
            "peak_post_event_displacement": max(
                (
                    float(np.linalg.norm(r["position"] - self.event_position))
                    for r in h[self.event_step :]
                ),
                default=None,
            )
            if self.event_position is not None
            else None,
        }

    def state_dict(self):
        return {
            "fingerprint": self.fingerprint,
            "state": copy.deepcopy(
                {
                    k: v
                    for k, v in self.__dict__.items()
                    if k not in ("config", "study_spec", "p", "fingerprint", "rng")
                }
            ),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, value):
        if value["fingerprint"] != self.fingerprint:
            raise ValueError("Drone checkpoint belongs to another task/model")
        self.__dict__.update(copy.deepcopy(value["state"]))
        self.rng.bit_generator.state = copy.deepcopy(value["rng"])

    def close(self):
        pass


def reference_action(body):
    """Privileged analytic SE(3) PD reference for mechanical qualification only."""
    acceleration = 3.0 * (body.goal - body.position) + 2.5 * (body.goal_velocity - body.velocity)
    force = body.mass * (acceleration + [0, 0, 9.81]) + body.p.drag * body.velocity
    z = force / max(np.linalg.norm(force), 1e-12)
    heading = np.array([np.cos(body.goal_yaw), np.sin(body.goal_yaw), 0.0])
    y = np.cross(z, heading)
    y /= max(np.linalg.norm(y), 1e-12)
    desired = np.column_stack((np.cross(y, z), y, z))
    skew = desired.T @ body.rotation - body.rotation.T @ desired
    error = 0.5 * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    torque = -0.4 * error - 0.12 * body.omega + np.cross(body.omega, body.inertia * body.omega)
    wrench = np.r_[max(0, force @ body.rotation[:, 2]), torque]
    thrust = np.linalg.lstsq(body.allocation() * body.motor_efficiency, wrench, rcond=None)[0]
    return np.clip(thrust / body.p.max_thrust * 2 - 1, -1, 1)


def qualify_drone():
    """Static allocation, SI-force and integration checks; no learned policy."""
    from .bodies import BodySpec

    body = DroneBody(BodySpec(name="drone", task="hover", horizon=100))
    body.position[:] = [0, 0, 1.5]
    body.velocity[:] = body.omega[:] = 0
    body.rotation = np.eye(3)
    trim = np.full(6, body.mass * 9.81 / 6)
    body.thrust[:] = trim
    for _ in range(100):
        body._physics(trim)
    drift = float(np.linalg.norm(body.position - [0, 0, 1.5]))
    failures = []
    for motor in range(6):
        active = np.arange(6) != motor
        result = lsq_linear(
            body.allocation()[:, active], [body.mass * 9.81, 0, 0, 0], bounds=(0, body.p.max_thrust)
        )
        failures.append(
            {
                "motor": motor,
                "residual": float(np.linalg.norm(result.fun)),
                "thrust": result.x.tolist(),
                "rank": int(np.linalg.matrix_rank(body.allocation()[:, active])),
            }
        )
    closed_loop = []
    for task in ("hover", "attitude", "gust"):
        probe = DroneBody(BodySpec(name="drone", task=task, horizon=250))
        probe.reset(7, "validation")
        for _ in range(250):
            _, _, terminated, truncated, info = probe.step(reference_action(probe))
            if terminated or truncated:
                break
        closed_loop.append(
            {
                "task": task,
                "crash": bool(info["crash"]),
                "final_error": float(np.linalg.norm(probe.position - probe.goal)),
                "angular_rmse": info["angular_rmse"],
            }
        )
    return {
        "schema": "drone-qualification-v1",
        "body_fingerprint": body.fingerprint,
        "hover_drift_meters": drift,
        "single_failure_allocation": failures,
        "privileged_reference": closed_loop,
        "passed": drift < 1e-8
        and all(x["residual"] < 1e-4 for x in failures)
        and all(not x["crash"] and x["final_error"] < 0.25 for x in closed_loop),
        "scope": "ideal hover trim, static fault allocation and privileged analytic recovery; no hardware or learned-task validation",
    }
