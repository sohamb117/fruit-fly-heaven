#!/usr/bin/env python3
"""Isolated published-controller/contact compatibility probes, never BANC.

Uses the released SavedModels and upstream native task factories. Adapted cases
project observations/actions by name; their added actuator behavior is declared
in the report. No production body, controller, XML, or browser state is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import mujoco
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import type_spec_registry
from flybody.fly_envs import flight_imitation, walk_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real
from flybody.tasks.constants import _WING_PARAMS

ROOT = Path(__file__).resolve().parents[1]
PATTERN = ROOT / "data/raw/flybody-baseline/wing_pattern_fmech.npy"


def setup_policy(name):
    normal = tfp.distributions.Normal(tf.zeros([1, 12]), 1.)
    independent = tfp.distributions.Independent(normal, reinterpreted_batch_ndims=1)
    type_spec_registry._NAME_TO_TYPE_SPEC["tensorflow_probability.python.distributions.normal.Normal_ACTTypeSpec"] = type(tf.type_spec_from_value(normal))
    type_spec_registry._NAME_TO_TYPE_SPEC["tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec"] = type(tf.type_spec_from_value(independent))
    return tf.saved_model.load(str(ROOT / f"data/raw/flybody-policy/{name}"))


def make_environment(walking, articulated=False, wings=False):
    if walking:
        return walk_imitation(disable_wings=not wings, random_state=np.random.RandomState(42))
    return flight_imitation(wpg_pattern_path=str(PATTERN), disable_legs=not articulated, random_state=np.random.RandomState(42))


def configure_common(env, walking):
    # One physical configuration for walking and flight: all leg and wing
    # joints present, walking's filtered nonwing actuators/contact settings,
    # flight's wing actuator/fluid parameters and 50us physics. Policies and
    # task-specific sensor/action projections remain separate.
    env.task.set_timesteps(physics_timestep=.00005, control_timestep=.002 if walking else .0002)
    walker = env.task.walker
    for actuator in walker.mjcf_model.find_all("actuator"):
        if actuator.tag == "adhesion":
            continue
        is_wing = actuator.name.startswith("wing_")
        actuator.dyntype = "none" if is_wing else "filterexact"
        actuator.dynprm = (1.,) if is_wing else (.01,)
    for i, axis in enumerate(["yaw", "roll", "pitch"]):
        walker.mjcf_model.find("default", axis).general.gainprm = (_WING_PARAMS["gainprm"][i],)
    wing = walker.mjcf_model.find("default", "wing")
    wing.joint.damping = _WING_PARAMS["damping"]
    wing.joint.stiffness = _WING_PARAMS["stiffness"]
    for geom in walker.mjcf_model.find_all("geom"):
        if "fluid" in geom.name:
            geom.fluidshape = "ellipsoid"
            geom.fluidcoef = _WING_PARAMS["fluidcoef"]
    walker.mjcf_model.find("default", "adhesion-collision").geom.friction = (1.,)
    for geom in env.task._arena.ground_geoms:
        geom.contype = 1
        geom.conaffinity = 1
        geom.friction = (.5,)
        geom.solref = (.001, 1.)
        geom.solimp = (.95, .99, .01)
    for name in ["gyro", "accelerometer", "velocimeter", "force", "touch"]:
        getattr(walker.observables, name).buffer_size = round(env.control_timestep()/.00005)


def run_case(case, policy, template):
    walking = case.startswith("walking")
    common = "common" in case
    articulated = "articulated" in case or (common and not walking)
    wings = case == "walking_wings_enabled" or (walking and common)
    env = make_environment(walking, articulated, wings)
    if common:
        configure_common(env, walking)
    duration = 2. if walking else 1.2
    env._time_limit = duration
    env.task._time_limit = duration
    if walking:
        env.task._max_episode_steps = round(duration/env.control_timestep()) + 1
    if not walking:
        for geom in env.task._arena.ground_geoms:
            geom.contype = 1
            geom.conaffinity = 1
    future = env.task._future_steps
    nsteps = round(duration/env.control_timestep()) + future + 1
    ref, vel = constant_speed_trajectory(n_steps=nsteps, speed=2 if walking else 20,
        init_pos=(0, 0, .1278 if walking else 1), body_rot_angle_y=0 if walking else -47.5,
        control_timestep=env.control_timestep())
    if case == "flight_floor_descent":
        ref[:, 2] = np.linspace(1, .08, nsteps)
        vel[:, 2] = (ref[-1, 2]-ref[0, 2])/((nsteps-1)*env.control_timestep())
    env.task._traj_generator.set_next_trajectory(ref, vel)
    ts = env.reset()
    spec = env.action_spec()
    names = spec.name.split("\t")
    baseline_names = template.action_spec().name.split("\t")
    action_indices = np.array([names.index(name) for name in baseline_names])
    actual_joint_names = [joint.name for joint in env.task.walker.observable_joints]
    original_joint_names = [joint.name for joint in template.task.walker.observable_joints]
    joint_indices = np.array([actual_joint_names.index(name) for name in original_joint_names])
    template_observation = template.reset().observation
    missing_actions = [name for name in names if name not in baseline_names]
    added_actuators = [i for i in range(env.physics.model.nu) if env.physics.model.id2name(i, "actuator").split("/")[-1] not in baseline_names]
    added_mode = "none"
    default_action = np.zeros(spec.shape, dtype=np.float32)
    if articulated and "passive" in case:
        # Explicit actuator ablation: passive joints retain their native
        # stiffness/damping, but added leg and adhesion actuators exert no force.
        for i in added_actuators:
            env.physics.model.actuator_gainprm[i] = 0
            env.physics.model.actuator_biasprm[i] = 0
        added_mode = "Added leg/adhesion actuator gain and bias are zero; native passive joint springs and damping retained."
    elif articulated:
        for name in missing_actions:
            aid = env.physics.model.name2id("walker/"+name, "actuator")
            if aid >= 0 and "adhere" not in name:
                default_action[names.index(name)] = env.physics.data.actuator_length[aid]
        added_mode = "Added leg position/tendon actuators target their initial retracted length; adhesion target zero. Fixed posture actuator targets, not pose constraints."
    elif wings:
        added_mode = "Added wing force actuators receive zero; native retraction springs remain."
    fields = list(template_observation)
    ground_ids = {env.physics.model.name2id(g.full_identifier, "geom") for g in env.task._arena.ground_geoms}
    initial_position = np.asarray(env.task.walker.get_pose(env.physics)[0]).copy()
    rows = []
    contacts = 0
    first_contact = None
    ground_contact_legs = set()
    max_reference = 0.
    max_tilt = 0.
    minheight = float("inf")
    max_external_force = 0.
    max_actuator_root_force = 0.
    steps = 0
    finite = True
    start = time.perf_counter()
    while not ts.last():
        observation = {key: np.asarray(ts.observation[key], np.float32) for key in fields}
        observation["walker/joints_pos"] = observation["walker/joints_pos"][joint_indices]
        observation["walker/joints_vel"] = observation["walker/joints_vel"][joint_indices]
        # Flight policy has no actuator-activation input. For walking with
        # added wings, only original actuator states enter the policy.
        if template_observation["walker/actuator_activation"].size == 0:
            observation["walker/actuator_activation"] = np.empty(0, np.float32)
        elif observation["walker/actuator_activation"].shape != template_observation["walker/actuator_activation"].shape:
            activation_indices = []
            # Actuation-state observation follows XML actuator order, which
            # differs from the action-spec order (adhesion actions are first).
            for original_aid in range(template.physics.model.nu):
                if template.physics.model.actuator_actadr[original_aid] < 0:
                    continue
                name = template.physics.model.id2name(original_aid, "actuator")
                aid = env.physics.model.name2id(name, "actuator")
                if aid >= 0 and env.physics.model.actuator_actadr[aid] >= 0:
                    activation_indices.append(env.physics.model.actuator_actadr[aid])
            observation["walker/actuator_activation"] = np.asarray(env.physics.data.act[activation_indices], np.float32)
        canonical = policy({k: tf.convert_to_tensor(v[None]) for k, v in observation.items()}).mean()[0].numpy()
        action = default_action.copy()
        action[action_indices] = canonical2real(canonical.copy(), template.action_spec())
        ts = env.step(action)
        steps += 1
        data = env.physics.data.ptr
        position, quat = env.task.walker.get_pose(env.physics)
        reference = np.asarray(ts.observation["walker/ref_displacement"])[0]
        error = float(np.linalg.norm(reference))
        tilt = float(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1))))
        minheight = min(minheight, float(position[2]))
        max_tilt = max(max_tilt, tilt)
        max_reference = max(max_reference, error)
        max_external_force = max(max_external_force, float(np.abs(data.xfrc_applied).max()), float(np.abs(data.qfrc_applied).max()))
        max_actuator_root_force = max(max_actuator_root_force, float(np.abs(data.qfrc_actuator[:6]).max()))
        finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(canonical).all())
        active_ground_contacts = 0
        force = np.zeros(6)
        for cid, contact in enumerate(data.contact):
            if contact.geom1 not in ground_ids and contact.geom2 not in ground_ids:
                continue
            mujoco.mj_contactForce(env.physics.model.ptr, data, cid, force)
            if force[0] <= 1e-10:
                continue
            active_ground_contacts += 1
            name = env.physics.model.id2name(contact.geom2 if contact.geom1 in ground_ids else contact.geom1, "geom")
            match = re.search(r"T([123])_(left|right)", name)
            if match:
                ground_contact_legs.add("_".join(match.groups()))
        if active_ground_contacts:
            contacts += 1
            if first_contact is None:
                first_contact = float(data.time)
        if steps % (10 if walking else 100) == 0 or ts.last():
            rows.append({"time": float(data.time), "position_cm": position.tolist(), "quaternion": quat.tolist(), "reference_error_cm": error, "active_ground_contacts": active_ground_contacts})
        if not finite or steps > nsteps:
            break
    last = rows[-1] if rows else {}
    last_position = np.asarray(env.task.walker.get_pose(env.physics)[0])
    result = {"case": case, "seconds": float(env.physics.data.time), "wall_seconds": time.perf_counter()-start,
        "steps": steps, "discount": float(ts.discount), "reached_trajectory_end": bool(env.task._reached_traj_end),
        "finite": finite, "min_height_cm": minheight, "max_body_tilt_degrees": max_tilt, "max_root_reference_error_cm": max_reference,
        "displacement_cm": (last_position-initial_position).tolist(), "max_external_applied_force": max_external_force,
        "max_native_actuator_root_generalized_force": max_actuator_root_force,
        "root_actuator_note": "Native claw-adhesion body transmissions contribute to root generalized forces during foot contact; this is not a free-root actuator or an externally applied stabilizer.",
        "ground_contact_control_samples": contacts, "first_ground_contact_s": first_contact, "ground_contact_legs": sorted(ground_contact_legs),
        "actual_action_size": int(spec.shape[0]), "policy_action_size": int(template.action_spec().shape[0]),
        "actual_observation_shapes": {k:list(v.shape) for k,v in ts.observation.items()}, "policy_observation_shapes": {k:list(v.shape) for k,v in template_observation.items()},
        "missing_actions": missing_actions, "added_actuator_mode": added_mode,
        "common_physical_configuration": common,
        "physics_timestep_s": float(env.physics.timestep()), "control_timestep_s": float(env.control_timestep()),
        "physical_dimensions": {key:int(getattr(env.physics.model,key)) for key in ["nq","nv","nu","na","njnt","nbody"]},
        "original_task_termination": "walking: root reference distance>.3cm, linvel>50cm/s, angvel>200rad/s, qacc>1e14; flight: height<.2cm, root reference distance>2cm, qacc>1e14",
        "reward_note": "Walking inference reward is identically1 and is not evidence of task performance." if walking else "Original flight imitation reward.",
        "trace": rows, "last": last}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["walking_native", "walking_wings_enabled", "flight_floor", "flight_articulated_passive", "flight_articulated_retracted", "flight_floor_descent"])
    p.add_argument("--output", type=Path, default=ROOT / "reports/flybody-contact-reuse.json")
    args = p.parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    policies = {kind: setup_policy(kind) for kind in {"walking" if case.startswith("walking") else "flight" for case in args.cases}}
    templates = {kind: make_environment(kind == "walking") for kind in policies}
    report = {"scope": "Isolated native published-controller compatibility probes; no BANC or production changes.",
        "source_revision": "d015e9bfe441bd90ae431bac24c55cb74bdbce26", "mujoco": mujoco.__version__,
        "policy_source": "https://ndownloader.figshare.com/files/44815195", "cases": []}
    for case in args.cases:
        kind = "walking" if case.startswith("walking") else "flight"
        try:
            result = run_case(case, policies[kind], templates[kind])
        except Exception as error:
            result = {"case": case, "exception": repr(error)}
        report["cases"].append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ["trace", "actual_observation_shapes", "policy_observation_shapes", "missing_actions", "last"]}), flush=True)
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
