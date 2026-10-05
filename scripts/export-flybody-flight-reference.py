#!/usr/bin/env python3
"""Export the upstream flight task for independent MuJoCo WASM reproduction.

Removes visual mesh geometry and mesh assets only, replacing mesh-derived body
inertias with the full compiled model's values. Keeps all joints, actuators,
sensors, collision/fluid geometry, body frames, and original task dynamics.
Requires the isolated FlyBody baseline environment. This never edits the BANC
production body or controller.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import mujoco
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import type_spec_registry

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real
from flybody.tasks.constants import _TERMINAL_HEIGHT, _TERMINAL_QACC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"
REVISION = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def array(value):
    if isinstance(value, dict):
        return {k: array(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def snapshot(data):
    return {"time": float(data.time), **{key: getattr(data, key).tolist() for key in ["qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "sensordata"]}}


def main():
    assert subprocess.check_output(["git", "-C", str(ROOT / "references/flybody"), "rev-parse", "HEAD"], text=True).strip() == REVISION
    pattern = ROOT / "data/raw/flybody-baseline/wing_pattern_fmech.npy"
    env = flight_imitation(wpg_pattern_path=str(pattern), random_state=np.random.RandomState(42))
    # Match the validated continuous native baseline's horizon. Reference
    # preview padding is not permission to execute beyond the upstream task's
    # own trajectory termination step.
    env._time_limit = 1.2
    env.task._time_limit = 1.2
    qpos_ref, qvel_ref = constant_speed_trajectory(n_steps=6020, speed=20, init_pos=(0, 0, 1), body_rot_angle_y=-47.5, control_timestep=env.control_timestep())
    env.task._traj_generator.set_next_trajectory(qpos_ref, qvel_ref)
    timestep = env.reset()
    full = env.physics.model.ptr
    xml_full = env.task.root_entity.mjcf_model.to_xml_string()
    root = ET.fromstring(xml_full)
    removed = []
    for body in root.findall(".//worldbody//body"):
        bid = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_BODY, body.get("name"))
        assert bid >= 0
        for inertial in list(body.findall("inertial")):
            body.remove(inertial)
        if full.body_mass[bid] > 0:
            ET.SubElement(body, "inertial", pos=numbers(full.body_ipos[bid]), quat=numbers(full.body_iquat[bid]), mass=format(float(full.body_mass[bid]), ".17g"), diaginertia=numbers(full.body_inertia[bid]))
        for geom in list(body.findall("geom")):
            gid = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_GEOM, geom.get("name"))
            assert gid >= 0
            if full.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                assert full.geom_contype[gid] == 0 and full.geom_conaffinity[gid] == 0
                removed.append(geom.get("name"))
                body.remove(geom)
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    root.find("compiler").set("inertiafromgeom", "false")
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode") + "\n"
    xml_path = OUT / "flybody-flight-reference.xml"
    xml_path.write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.nu, model.nbody, model.nsensor) == (full.nq, full.nv, full.nu, full.nbody, full.nsensor)
    comparisons = {}
    for field in ["body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "jnt_pos", "jnt_axis", "jnt_range", "dof_damping", "dof_armature", "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange", "actuator_trntype", "actuator_trnid", "sensor_type", "sensor_adr", "sensor_dim"]:
        delta = np.abs(getattr(model, field) - getattr(full, field))
        comparisons[field] = float(delta.max(initial=0))
        assert comparisons[field] < 1e-12, (field, comparisons[field])
    walker = env.task.walker
    observable_joint_ids = [model.joint(j.full_identifier).id for j in walker.observable_joints]
    ghost_joint = model.joint(env.task._ghost_joint.full_identifier)
    wbpg = env.task._wbpg
    spec = env.action_spec()
    sensors = [{"name": model.sensor(i).name, "id": i, "type": int(model.sensor_type[i]), "address": int(model.sensor_adr[i]), "dimension": int(model.sensor_dim[i]), "object_id": int(model.sensor_objid[i]), "object_type": int(model.sensor_objtype[i])} for i in range(model.nsensor)]
    actuators = []
    for i in range(model.nu):
        transmission = int(model.actuator_trntype[i])
        actuator = {"name": model.actuator(i).name, "id": i, "transmission_type": transmission,
                    "transmission_name": mujoco.mjtTrn(transmission).name,
                    "ctrlrange": model.actuator_ctrlrange[i].tolist()}
        if transmission == mujoco.mjtTrn.mjTRN_JOINT:
            actuator["joint"] = int(model.actuator_trnid[i, 0])
        elif transmission == mujoco.mjtTrn.mjTRN_TENDON:
            actuator["tendon"] = int(model.actuator_trnid[i, 0])
        actuators.append(actuator)
    metadata = {
        "schema": 1,
        "kind": "upstream_trained_policy_flight_reference_not_BANC",
        "source": {"repository": "https://github.com/TuragaLab/flybody", "revision": REVISION, "full_task_xml_sha256": hashlib.sha256(xml_full.encode()).hexdigest(), "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "mujoco": mujoco.__version__, "license": "Apache-2.0 for model source; flight waveform and policy are separately published data"},
        "changes": ["Removed non-colliding visual mesh geoms and mesh assets", "Inserted exact compiled body inertias and disabled inertia inference"],
        "unchanged_native_parameters_max_error": comparisons,
        "removed_visual_mesh_geoms": removed,
        "dimensions": {key: int(getattr(model, key)) for key in ["nq", "nv", "nu", "na", "nbody", "ngeom", "njnt", "nsensor", "nsensordata"]},
        "physics_timestep": float(model.opt.timestep),
        "control_timestep": env.control_timestep(),
        "episode": {"time_limit_s": float(env._time_limit), "upstream_default_time_limit_s": 0.6,
                    "trajectory_timesteps": int(env.task._traj_timesteps),
                    "maxControlSteps": min(int(env.task._traj_timesteps), round(env._time_limit / env.control_timestep()))},
        "termination": {"minimum_height_cm": float(_TERMINAL_HEIGHT), "maximum_qacc_norm": float(_TERMINAL_QACC),
                        "maximum_reference_displacement_cm": float(env.task._terminal_com_dist),
                        "height_body_id": model.body(walker.thorax.full_identifier).id,
                        "reference_displacement": "Norm of the current root-frame walker/ref_displacement[0], as in the upstream check_termination method",
                        "comparisons": "Fail on height < minimum, qacc norm > maximum, or reference displacement > maximum; good termination at trajectory_timesteps."},
        "physics_substeps_per_control": 4,
        "dm_control_legacy_step": bool(env.physics.legacy_step),
        "stepping": "For each of four physics substeps: mj_step2 then mj_step1, as dm_control legacy Euler. After first substep only, mj_forward reproduces the dirty-state sensor-binding refresh caused by writing the ghost pose. Then sample sensors after each substep; average all four samples for gyro/accelerometer/velocimeter. Joint/world-z observations are from final substep.",
        "initial_state": snapshot(env.physics.data.ptr),
        "initial_observation": array(timestep.observation),
        "joints": [{"name": model.joint(i).name, "id": i, "type": int(model.jnt_type[i]), "qpos": int(model.jnt_qposadr[i]), "dof": int(model.jnt_dofadr[i]), "body": int(model.jnt_bodyid[i])} for i in range(model.njnt)],
        "actuators": actuators,
        "action_mapping": {"names": spec.name.split("\t"), "minimum": spec.minimum.tolist(), "maximum": spec.maximum.tolist(), "action_indices": dict(walker._action_indices), "ctrl_indices": dict(walker._ctrl_indices), "wing_indices": env.task._wing_inds_action, "frequency_user_index": env.task._user_idx_action, "transform": "canonical2real clipped [-1,1], then add WBPG target minus current wing qpos to the six wing actions; first 11 actions become ctrl"},
        "observations": {"observable_joint_ids": observable_joint_ids, "qpos_indices": model.jnt_qposadr[observable_joint_ids].tolist(), "qvel_indices": model.jnt_dofadr[observable_joint_ids].tolist(), "world_zaxis_body_id": model.body(walker.root_body.full_identifier).id, "sensors": sensors, "sensor_mean_buffer_size": 4, "actuator_activation_size": int(model.na), "future_steps": 5, "reference_formula": "At control step counter, displacement of next6 reference root positions relative to model root, rotated by the root_body xmat transpose. Relative quaternions are reciprocal(model_root_quat)*reference_root_quat; preserve the upstream sign."},
        "body_to_leg": [(int(m[1])-1+(3 if m[2] == "right" else 0)) if (m := re.search(r"_T([123])_(left|right)$", model.body(i).name)) else -1 for i in range(model.nbody)],
        "bodies": [{"id": i, "name": model.body(i).name, "parent": int(model.body_parentid[i]), "pos": model.body_pos[i].tolist(), "quat": model.body_quat[i].tolist()} for i in range(model.nbody)],
        "ghost": {"qpos": int(ghost_joint.qposadr[0]), "dof": int(ghost_joint.dofadr[0]), "offset": array(env.task._ghost_offset), "note": "Ghost is an unactuated non-colliding reference marker; reset its root qpos/qvel to current reference before every control step, exactly as upstream."},
        "reference": {"root_qpos": env.task._ref_qpos.tolist(), "qvel": env.task._ref_qvel.tolist(), "center_of_mass_qpos": qpos_ref.tolist()},
        "wingbeat_generator": {"base_beat_freq": wbpg.base_beat_freq, "rel_freq_range": wbpg.rel_freq_range, "ctrl_filter": wbpg.ctrl_filter, "rate": wbpg._rate, "beat_freqs": wbpg.beat_freqs.tolist(), "initial_state": {"frequency": float(wbpg._ctrl_freq), "frequency_index": int(wbpg._freq_idx), "step": int(wbpg._step)}, "trajectories": array({str(i): value for i, value in enumerate(wbpg.traj_ctrl)})},
        "limitations": ["Starts airborne; legs frozen and floor contacts disabled by original task", "Published trained policy, not a BANC controller", "The ghost is not an additional controlled fly and cannot transfer forces to the physical fly"],
    }
    normal = tfp.distributions.Normal(tf.zeros([1, 12]), 1.)
    independent = tfp.distributions.Independent(normal, reinterpreted_batch_ndims=1)
    type_spec_registry._NAME_TO_TYPE_SPEC["tensorflow_probability.python.distributions.normal.Normal_ACTTypeSpec"] = type(tf.type_spec_from_value(normal))
    type_spec_registry._NAME_TO_TYPE_SPEC["tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec"] = type(tf.type_spec_from_value(independent))
    policy = tf.saved_model.load(str(ROOT / "data/raw/flybody-policy/flight"))
    reduced = mujoco.MjData(model)
    reduced.qpos[:] = env.physics.data.qpos
    reduced.qvel[:] = env.physics.data.qvel
    reduced.qacc_warmstart[:] = env.physics.data.qacc_warmstart
    mujoco.mj_forward(model, reduced)
    frames = []
    max_state_error = 0.
    max_sensor_mean_error = 0.
    for step in range(10):
        observation = {k: np.asarray(v, np.float32) for k, v in timestep.observation.items()}
        canonical = policy({k: tf.convert_to_tensor(v[None]) for k, v in observation.items()}).mean()[0].numpy()
        real_action = canonical2real(canonical.copy(), spec)
        frame = {"step": step, "observation_before": array(observation), "canonical_action": canonical.tolist(), "real_action_before_wbpg": real_action.tolist()}
        timestep = env.step(real_action)
        reduced.qpos[int(ghost_joint.qposadr[0]):int(ghost_joint.qposadr[0])+7] = env.task._ref_qpos[step]
        reduced.qpos[int(ghost_joint.qposadr[0]):int(ghost_joint.qposadr[0])+3] += env.task._ghost_offset
        reduced.qvel[int(ghost_joint.dofadr[0]):int(ghost_joint.dofadr[0])+6] = env.task._ref_qvel[step]
        reduced.ctrl[:] = env.physics.data.ctrl
        subsensors = []
        for substep in range(4):
            mujoco.mj_step2(model, reduced)
            mujoco.mj_step1(model, reduced)
            if substep == 0:
                # Ghost pose writes set dm_control's dirty flag; reading the
                # first sensor binding refreshes acceleration-dependent fields.
                mujoco.mj_forward(model, reduced)
            subsensors.append(reduced.sensordata.tolist())
        error = max(float(np.abs(reduced.qpos - env.physics.data.qpos).max()), float(np.abs(reduced.qvel - env.physics.data.qvel).max()))
        max_state_error = max(max_state_error, error)
        mean_sensors = np.mean(subsensors, axis=0)
        sensor_error = max(float(np.abs(mean_sensors[offset:offset+3] - timestep.observation[f"walker/{name}"]).max()) for name, offset in [("accelerometer", 0), ("gyro", 3), ("velocimeter", 6)])
        max_sensor_mean_error = max(max_sensor_mean_error, sensor_error)
        frame.update({"ctrl": env.physics.data.ctrl.tolist(), "state_after": snapshot(env.physics.data.ptr), "observation_after": array(timestep.observation), "wingbeat_state_after": {"frequency": float(wbpg._ctrl_freq), "frequency_index": int(wbpg._freq_idx), "step": int(wbpg._step)}, "mesh_free_replay_error": error, "mesh_free_sensor_mean_error": sensor_error, "mesh_free_substep_sensors": subsensors})
        frames.append(frame)
    metadata["native_mesh_free_replay_max_state_error"] = max_state_error
    metadata["native_mesh_free_sensor_mean_max_error"] = max_sensor_mean_error
    assert max_state_error < 1e-8, max_state_error
    assert max_sensor_mean_error < 1e-8, max_sensor_mean_error
    (OUT / "flybody-flight-reference.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    fixture = {"xml_sha256": metadata["source"]["xml_sha256"], "mujoco": mujoco.__version__, "initial_state": metadata["initial_state"], "native_mesh_free_replay_max_state_error": max_state_error, "frames": frames}
    (OUT / "flybody-flight-reference-fixtures.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(json.dumps({"dimensions": metadata["dimensions"], "xml_sha256": metadata["source"]["xml_sha256"], "mesh_free_state_error": max_state_error, "mesh_free_sensor_mean_error": max_sensor_mean_error, "policy_fixtures": len(frames)}))


if __name__ == "__main__":
    main()
