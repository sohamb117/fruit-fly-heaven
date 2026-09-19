"""Actuator-only tasks on the explicitly declared simZFish-derived 3D body.

Hydrostatics supply displacement-based buoyancy, not a stabilizing controller.
All translation, turning and changes in depth must arise from joint commands
and the body's hydrodynamics. No root forces depend on task targets.
"""

from __future__ import annotations

import math

import numpy as np

from .bodies import ManifestBody
from .fish_model import REVISION, SOURCE_FILES


class FishBody(ManifestBody):
    def __init__(self, spec):
        super().__init__(spec)
        manifest = self.model_manifest
        if (
            spec.name != "fish"
            or manifest.get("backend") != "simzfish-3d-extension-v1"
            or manifest["source"].get("revision") != REVISION
            or any(
                manifest["files"].get(f"source/{name}") != checksum
                for name, checksum in SOURCE_FILES.items()
            )
            or manifest.get("native_body_invariants", {}).get("automatic_stabilizer") is not False
        ):
            raise ValueError("Fish body must use the pinned, declared simZFish extension")
        if self.model.opt.integrator == self.mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("Split hydrostatic integration does not support RK4")
        self.displaced_volume = np.zeros(self.model.nbody)
        self.center_of_volume = np.zeros((self.model.nbody, 3))
        for index, kind in enumerate(self.model.geom_type):
            size = self.model.geom_size[index]
            if kind == self.mujoco.mjtGeom.mjGEOM_CYLINDER:
                volume = math.pi * size[0] ** 2 * 2 * size[1]
            elif kind == self.mujoco.mjtGeom.mjGEOM_BOX:
                volume = 8 * np.prod(size)
            elif kind == self.mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                volume = 4 / 3 * math.pi * np.prod(size)
            else:
                raise ValueError("Unregistered displaced-volume geometry in the fish body")
            body = self.model.geom_bodyid[index]
            self.displaced_volume[body] += volume
            self.center_of_volume[body] += volume * self.model.geom_pos[index]
        active = self.displaced_volume > 0
        self.center_of_volume[active] /= self.displaced_volume[active, None]

    def reset(self, seed, split="train"):
        self.last_world_velocity = np.zeros(3)
        super().reset(seed, split)
        defaults = self.model_manifest["task_defaults"]
        self.target_speed = float(
            self.spec.parameters.get("target_speed", defaults["target_speed"])
        )
        rotation, _ = self._kinematics()
        forward = rotation @ self.forward_axis
        heading = math.atan2(forward[1], forward[0])
        if self.spec.task == "heading":
            span = defaults["initial_heading_span"]
            heading += self.rng.choice([-1, 1]) * self.rng.uniform(span / 2, span)
        self.target_direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        self.target_depth = float(
            self.spec.parameters.get("target_depth", self.origin[2] / self.length)
        )
        if self.spec.task == "depth":
            self.target_speed = 0.0
            if "target_depth" not in self.spec.parameters:
                low, high = defaults["depth_offset_body_lengths"]
                self.target_depth += self.rng.choice([-1, 1]) * self.rng.uniform(low, high)
        if split == "ood":
            self.target_speed *= 1.5
            if self.spec.task == "depth":
                self.target_depth += self.rng.choice([-1, 1]) * 0.5
        self.case["target_direction"] = self.target_direction.tolist()
        self.case["target_speed"] = self.target_speed
        self.case["target_depth"] = self.target_depth
        return self.observation()

    def _apply_buoyancy(self):
        # Each source collision primitive denotes displaced volume. The native
        # source cylinder/box volumes are additive (also used for source mass).
        # xfrc_applied expects a world-frame force and a torque about the COM.
        density = self.model.opt.density
        force = -density * self.displaced_volume[:, None] * self.model.opt.gravity
        rotation = self.data.xmat.reshape(-1, 3, 3)
        centers = self.data.xpos + np.einsum("bij,bj->bi", rotation, self.center_of_volume)
        self.data.xfrc_applied[:, :3] = force
        self.data.xfrc_applied[:, 3:] = np.cross(centers - self.data.xipos, force)

    def position(self):
        return self.data.subtree_com[self.root_id]

    def _kinematics(self):
        rotation, velocity = super()._kinematics()
        # The mean over the last physical control interval avoids aliasing
        # impulsive joint-servomotor velocities at the action update boundary.
        # This is a physical velocity measurement, never a task controller.
        velocity[3:] = rotation.T @ self.last_world_velocity
        return rotation, velocity

    def _advance_physics(self):
        previous = self.position().copy()
        for _ in range(self.substeps):
            self.mujoco.mj_step1(self.model, self.data)
            self._apply_buoyancy()
            self.mujoco.mj_step2(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self.last_world_velocity = (self.position() - previous) / self.control_dt

    def state_dict(self):
        return {**super().state_dict(), "last_world_velocity": self.last_world_velocity.copy()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.last_world_velocity = np.asarray(state["last_world_velocity"]).copy()
