"""Observable sensor, actuator, target, and native-physics distribution shifts."""

from __future__ import annotations

import copy
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..util import digest_json, seed_for
from .bodies import BodySpec, make_body


@dataclass(frozen=True)
class PerturbationSpec:
    name: str = "nominal"
    seed: int = 0
    sensor_noise: float = 0.0
    sensor_delay: int = 0
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    friction_scale: float = 1.0
    fluid_density_scale: float = 1.0
    viscosity_scale: float = 1.0
    actuator_scale: float = 1.0
    failed_actuators: tuple[int, ...] = ()
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    terrain_slope: float = 0.0
    target_speed_scale: float = 1.0
    trajectory_frequency_scale: float = 1.0

    def validate(self):
        if not self.name or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Perturbation needs a name and nonnegative integer seed")
        if type(self.sensor_delay) is not int or self.sensor_delay < 0:
            raise ValueError("Sensor delay is a nonnegative number of decisions")
        for name in (
            "mass_scale",
            "inertia_scale",
            "friction_scale",
            "fluid_density_scale",
            "viscosity_scale",
            "target_speed_scale",
            "trajectory_frequency_scale",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.sensor_noise)
            or self.sensor_noise < 0
            or not 0 <= self.actuator_scale <= 1
        ):
            raise ValueError("Invalid sensor noise or actuator strength")
        if len(self.wind) != 3 or not np.isfinite(self.wind).all():
            raise ValueError("Wind/current must be a finite 3D vector in native body units/second")
        if not math.isfinite(self.terrain_slope) or abs(self.terrain_slope) > math.pi / 4:
            raise ValueError("Terrain slope must be finite and within +/-45 degrees")
        if len(set(self.failed_actuators)) != len(self.failed_actuators) or any(
            type(i) is not int or i < 0 for i in self.failed_actuators
        ):
            raise ValueError("Failed actuator indices must be unique nonnegative integers")


def _scale_actuator(model, mujoco, index, scale):
    gain, bias = model.actuator_gaintype[index], model.actuator_biastype[index]
    if gain == mujoco.mjtGain.mjGAIN_FIXED:
        model.actuator_gainprm[index, 0] *= scale
    elif gain == mujoco.mjtGain.mjGAIN_AFFINE:
        model.actuator_gainprm[index, :3] *= scale
    elif gain == mujoco.mjtGain.mjGAIN_MUSCLE:
        # MuJoCo muscle force is parameter 2; negative force selects automatic
        # scaling via parameter 3. Preserve length/velocity response parameters.
        field = 3 if model.actuator_gainprm[index, 2] < 0 else 2
        model.actuator_gainprm[index, field] *= scale
    else:
        raise ValueError("Actuator weakness is undefined for this custom gain model")
    if bias == mujoco.mjtBias.mjBIAS_AFFINE:
        model.actuator_biasprm[index, :3] *= scale
    elif bias == mujoco.mjtBias.mjBIAS_MUSCLE:
        field = 3 if model.actuator_biasprm[index, 2] < 0 else 2
        model.actuator_biasprm[index, field] *= scale
    elif bias != mujoco.mjtBias.mjBIAS_NONE:
        raise ValueError("Actuator weakness is undefined for this custom bias model")


class PerturbedBody:
    def __init__(self, body, spec: PerturbationSpec, target_adjusted=False):
        spec.validate()
        self.body, self.perturbation = body, spec
        self.target_adjusted = target_adjusted
        if hasattr(body, "configure_perturbation"):
            body.configure_perturbation(spec)
        else:
            self._configure_mujoco(body, spec)
        self.fingerprint = digest_json({"body": body.fingerprint, "perturbation": asdict(spec)})
        self.observation_buffer = deque(maxlen=spec.sensor_delay + 1)
        self.rng = np.random.default_rng(spec.seed)

    @staticmethod
    def _configure_mujoco(body, spec):
        model = body.physics.model.ptr if hasattr(body, "physics") else body.model
        data = body.physics.data.ptr if hasattr(body, "physics") else body.data
        mujoco = body.mujoco
        if any(i >= model.nu for i in spec.failed_actuators):
            raise ValueError("Failed actuator index outside this body")
        if spec.mass_scale != 1 or spec.inertia_scale != 1:
            # Automatic muscle force scaling uses the model's reference
            # acceleration. Changing mass must not silently strengthen muscles.
            for index in range(model.nu):
                for kinds, parameters, muscle_kind in (
                    (model.actuator_gaintype, model.actuator_gainprm, mujoco.mjtGain.mjGAIN_MUSCLE),
                    (model.actuator_biastype, model.actuator_biasprm, mujoco.mjtBias.mjBIAS_MUSCLE),
                ):
                    if kinds[index] == muscle_kind and parameters[index, 2] < 0:
                        if model.actuator_acc0[index] <= 0:
                            raise ValueError(
                                "Cannot preserve muscle strength with zero reference acceleration"
                            )
                        parameters[index, 2] = parameters[index, 3] / model.actuator_acc0[index]
        model.body_mass[1:] *= spec.mass_scale
        model.body_inertia[1:] *= spec.inertia_scale
        if spec.friction_scale != 1:
            if not np.any(model.geom_contype | model.geom_conaffinity):
                raise ValueError("This body has no contact surfaces for a friction intervention")
            model.geom_friction[:] *= spec.friction_scale
        if spec.fluid_density_scale != 1 and model.opt.density <= 0:
            raise ValueError("This body has no native fluid density to perturb")
        if spec.viscosity_scale != 1 and model.opt.viscosity <= 0:
            raise ValueError("This body has no native fluid viscosity to perturb")
        model.opt.density *= spec.fluid_density_scale
        model.opt.viscosity *= spec.viscosity_scale
        if any(spec.wind):
            if model.opt.density <= 0 and model.opt.viscosity <= 0:
                raise ValueError("Wind/current requires native fluid forces")
            model.opt.wind[:] = spec.wind
        if spec.terrain_slope:
            planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            if not len(planes):
                raise ValueError("Unseen sloped terrain requires a native ground plane")
            rotation = np.array(
                [math.cos(spec.terrain_slope / 2), 0, math.sin(spec.terrain_slope / 2), 0]
            )
            for index in planes:
                composed = np.empty(4)
                mujoco.mju_mulQuat(composed, rotation, model.geom_quat[index])
                model.geom_quat[index] = composed
        for index in range(model.nu):
            scale = 0.0 if index in spec.failed_actuators else spec.actuator_scale
            if scale != 1:
                _scale_actuator(model, mujoco, index, scale)
        mujoco.mj_setConst(model, data)

    def __getattr__(self, name):
        return getattr(self.body, name)

    def _sense(self, observation):
        observed = (
            observation + self.rng.normal(0, self.perturbation.sensor_noise, observation.shape)
        ).astype(np.float32)
        if not self.observation_buffer:
            for _ in range(self.perturbation.sensor_delay):
                self.observation_buffer.append(observed.copy())
        self.observation_buffer.append(observed)
        return self.observation_buffer[0].copy()

    def reset(self, seed, split="train"):
        observation = self.body.reset(seed, split)
        if self.perturbation.target_speed_scale != 1 and not self.target_adjusted:
            if "command" not in self.body.case:
                raise ValueError("Target speed variation is not defined for this task")
            self.body.case["command"] = (
                np.asarray(self.body.case["command"]) * self.perturbation.target_speed_scale
            ).tolist()
            observation = self.body.observation()
        self.rng = np.random.default_rng(
            seed_for(seed, f"body-perturbation-{self.perturbation.seed}-{split}")
        )
        self.observation_buffer.clear()
        return self._sense(observation)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.body.step(action)
        return self._sense(observation), reward, terminated, truncated, info

    def state_dict(self):
        return {
            "fingerprint": self.fingerprint,
            "body": self.body.state_dict(),
            "sensor_history": list(self.observation_buffer),
            "sensor_rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("Robustness condition differs from checkpoint")
        self.body.load_state_dict(state["body"])
        self.observation_buffer.clear()
        self.observation_buffer.extend(copy.deepcopy(state["sensor_history"]))
        self.rng.bit_generator.state = copy.deepcopy(state["sensor_rng"])

    def close(self):
        self.body.close()


def make_perturbed_body(body: BodySpec, spec: PerturbationSpec):
    spec.validate()
    parameters = dict(body.parameters)
    target_adjusted = False
    if spec.target_speed_scale != 1:
        if body.task == "controlled_flight" or (
            body.name != "fly" and body.task not in ("posture", "depth")
        ):
            parameters["target_speed"] = (
                parameters.get("target_speed", _default_target_speed(body))
                * spec.target_speed_scale
            )
            target_adjusted = True
        elif body.task != "walking":
            raise ValueError("This stabilization task has no locomotion speed command")
    if spec.trajectory_frequency_scale != 1:
        if body.task != "controlled_flight":
            raise ValueError("Trajectory frequency shift requires controlled flight")
        parameters["trajectory_frequency"] = (
            parameters.get("trajectory_frequency", 0.5) * spec.trajectory_frequency_scale
        )
    instance = make_body(replace(body, parameters=parameters))
    try:
        return PerturbedBody(instance, spec, target_adjusted)
    except BaseException:
        instance.close()
        raise


def _default_target_speed(body):
    if body.name == "fly":
        return 5.0
    if body.model_manifest:
        manifest = json.loads(Path(body.model_manifest).read_text())
        return manifest.get("task_defaults", {}).get("target_speed", 1.0)
    return 1.0
