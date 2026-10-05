#!/usr/bin/env python3
"""Reproduce the released FlyBody flight policy on the unmodified upstream task.

This is a trained controller baseline, not a BANC controller. The native upstream
flight task disables legs/floor contacts and starts in flight. No root force,
pose correction, hand-written stabilizer, or local wing adapter is used here.

Create an isolated environment with scripts/flybody-baseline-requirements.txt,
then run from the repository root. The two complete default episodes total about
1.2 seconds of simulated flight, with an independent random wing phase at reset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # Registers SavedModel distribution types.

from flybody.fly_envs import flight_imitation
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "data/raw/flybody-policy/flight")
    parser.add_argument("--wing-pattern", type=Path, default=ROOT / "data/raw/flybody-baseline/wing_pattern_fmech.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/flybody-upstream-baseline.json")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "reports/flybody-upstream-policy-fixtures.npz")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--duration", type=float, default=0.6, help="Episode horizon; 0.6 is the upstream default. Only the time limit changes for longer tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.duration <= .002:
        parser.error("--duration must exceed .002 seconds")

    # Same input conversion and deterministic distribution mean as the upstream
    # SinglePrecisionWrapper and TestPolicyWrapper, without importing the Acme
    # training stack (Reverb has no macOS wheel). canonical2real is upstream code
    # and performs the CanonicalSpecWrapper action rescaling and clipping.
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    # TFP renamed its serialized type registry namespace after the published
    # checkpoint. Supply aliases for the same distribution TypeSpecs; model
    # operations, tensors and weights are not edited.
    from tensorflow.python.framework import type_spec_registry
    normal = tfp.distributions.Normal(tf.zeros([1, 12]), 1.)
    independent = tfp.distributions.Independent(normal, reinterpreted_batch_ndims=1)
    type_spec_registry._NAME_TO_TYPE_SPEC["tensorflow_probability.python.distributions.normal.Normal_ACTTypeSpec"] = type(tf.type_spec_from_value(normal))
    type_spec_registry._NAME_TO_TYPE_SPEC["tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec"] = type(tf.type_spec_from_value(independent))
    policy = tf.saved_model.load(str(args.policy))
    env = flight_imitation(
        wpg_pattern_path=str(args.wing_pattern),
        random_state=np.random.RandomState(args.seed),
    )
    # Only extend the reference/time horizon when requested; equations, initial
    # state, control rate, model, actuators and trained policy remain upstream.
    env._time_limit = args.duration
    env.task._time_limit = args.duration
    control_dt = env.control_timestep()
    qpos, qvel = constant_speed_trajectory(
        n_steps=round(args.duration / control_dt) + 20,
        speed=20,
        init_pos=(0, 0, 1),
        body_rot_angle_y=-47.5,
        control_timestep=control_dt,
    )
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    spec = env.action_spec()
    report = {
        "kind": "released_trained_policy_baseline_not_BANC",
        "upstream": {
            "repository": "https://github.com/TuragaLab/flybody",
            "commit": subprocess.check_output(["git", "-C", str(ROOT / "references/flybody"), "rev-parse", "HEAD"], text=True).strip(),
            "task": "flybody.fly_envs.flight_imitation",
            "policy_url": "https://ndownloader.figshare.com/files/44815195",
            "policy_sha256": {str(p.relative_to(args.policy)): sha256(p) for p in sorted(args.policy.rglob("*")) if p.is_file()},
            "wing_pattern_sha256": sha256(args.wing_pattern),
            "wrapper_equivalence": "float32 observations + leading batch dimension + distribution.mean()[0] + upstream canonical2real(..., clip=True)",
            "tfp_compatibility": "Aliases legacy Normal/Independent serialized TypeSpec names to the same classes in TFP0.23; no graph or weight modifications.",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "tensorflow", "tensorflow-probability", "dm-control", "mujoco", "flybody"]},
            "physics_timestep_s": float(env.physics.timestep()),
            "control_timestep_s": control_dt,
            "episode_time_limit_s": args.duration,
            "upstream_default_time_limit_s": 0.6,
            "only_time_horizon_extended": args.duration != 0.6,
            "seed": args.seed,
            "population": 1,
            "legs_disabled": True,
            "floor_contacts_disabled": True,
            "joint_filter_s": 0,
            "root_dofs": 6,
            "action_shape": list(spec.shape),
            "action_names": spec.name.split("\t"),
        },
        "reference": {
            "generator": "upstream constant_speed_trajectory",
            "speed_cm_s": 20,
            "height_cm": 1,
            "body_pitch_degrees": -47.5,
            "future_steps": 5,
            "inputs_are_privileged_reference_and_proprioception": True,
        },
        "limitations": [
            "Starts already airborne with prescribed reference velocity; not a takeoff test.",
            "Legs frozen and floor contact disabled by upstream task; not a landing or combined walking-flight test.",
            "Published learned controller and wingbeat generator, not BANC neurons, muscle-specific physiology, or autonomous food seeking.",
            "TensorFlow 2.15.1 is used for Apple Silicon inference of a model originally published with TensorFlow 2.8 dependencies.",
        ],
        "episodes": [],
    }
    fixtures = {}
    fixture_actions = []
    video_frames = []
    policy_total = 0.0
    env_total = 0.0

    for episode in range(args.episodes):
        timestep = env.reset()
        if episode == 0:
            report["environment"]["observation_shapes"] = {k: list(v.shape) for k, v in timestep.observation.items()}
            report["environment"]["nq"] = int(env.physics.model.nq)
            report["environment"]["nv"] = int(env.physics.model.nv)
            report["environment"]["nu"] = int(env.physics.model.nu)
        steps = 0
        trace = []
        rewards = []
        heights = []
        distances = []
        attitude_errors = []
        upright = []
        root_wrench_max = 0.0
        root_actuator_force_max = 0.0
        all_finite = True
        wall_start = time.perf_counter()
        while not timestep.last():
            observation = {k: np.asarray(v, dtype=np.float32) for k, v in timestep.observation.items()}
            policy_start = time.perf_counter()
            distribution = policy({k: tf.convert_to_tensor(v[None]) for k, v in observation.items()})
            canonical = distribution.mean()[0].numpy()
            policy_total += time.perf_counter() - policy_start
            if steps % 100 == 0:
                for key, value in observation.items():
                    fixtures.setdefault(key, []).append(value)
                fixture_actions.append(canonical.copy())
            action = canonical2real(canonical.copy(), spec)
            env_start = time.perf_counter()
            timestep = env.step(action)
            env_total += time.perf_counter() - env_start
            steps += 1
            physics = env.physics
            position, quat = env.task.walker.get_pose(physics)
            velocity, angular = env.task.walker.get_velocity(physics)
            reference = env.task._ref_qpos[min(steps, len(env.task._ref_qpos) - 1)]
            distance = float(np.linalg.norm(physics.named.data.subtree_com["walker/"] - qpos[min(steps, len(qpos) - 1), :3]))
            attitude_error = float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(quat, reference[3:7])), 0, 1))))
            body_up = float(1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
            height = float(position[2])
            root_wrench_max = max(root_wrench_max, float(np.abs(physics.data.xfrc_applied).max()), float(np.abs(physics.data.qfrc_applied).max()))
            root_actuator_force_max = max(root_actuator_force_max, float(np.abs(physics.data.qfrc_actuator[:6]).max()))
            all_finite &= bool(np.isfinite(physics.data.qpos).all() and np.isfinite(physics.data.qvel).all() and np.isfinite(canonical).all())
            rewards.append(float(timestep.reward))
            heights.append(height)
            distances.append(distance)
            attitude_errors.append(attitude_error)
            upright.append(body_up)
            if steps % 10 == 0 or timestep.last():
                trace.append({"time_s": float(physics.data.time), "position_cm": position.tolist(), "quaternion_wxyz": quat.tolist(), "velocity_cm_s": velocity.tolist(), "angular_velocity_rad_s": angular.tolist(), "reference_error_cm": distance, "attitude_error_degrees": attitude_error, "reward": float(timestep.reward)})
            if args.video and episode == 0 and steps % 20 == 0:
                video_frames.append(physics.render(height=480, width=640, camera_id=1))
            if not all_finite or steps > round(args.duration / control_dt) + 100:
                break
        good_termination = bool(env.task._reached_traj_end and timestep.last() and timestep.discount == 1)
        entry = {
            "episode": episode,
            "steps": steps,
            "simulated_seconds": float(env.physics.data.time),
            "wall_seconds": time.perf_counter() - wall_start,
            "completed_upstream_episode": good_termination,
            "discount": float(timestep.discount),
            "all_finite": all_finite,
            "minimum_height_cm": min(heights),
            "maximum_height_cm": max(heights),
            "minimum_body_up_z": min(upright),
            "maximum_reference_error_cm": max(distances),
            "mean_reference_error_cm": float(np.mean(distances)),
            "maximum_attitude_error_degrees": max(attitude_errors),
            "mean_reward": float(np.mean(rewards)),
            "minimum_reward": min(rewards),
            "max_external_applied_force_or_torque": root_wrench_max,
            "max_direct_root_actuator_force": root_actuator_force_max,
            "trace": trace,
        }
        report["episodes"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "trace"}), flush=True)

    report["total_simulated_seconds"] = sum(e["simulated_seconds"] for e in report["episodes"])
    report["timing"] = {"policy_seconds": policy_total, "environment_step_seconds": env_total, "includes_rendering": bool(args.video)}
    report["passed"] = all(e["completed_upstream_episode"] and e["all_finite"] and e["minimum_height_cm"] > .2 and e["minimum_body_up_z"] > 0 and e["mean_reward"] > .8 and e["max_external_applied_force_or_torque"] == 0 and e["max_direct_root_actuator_force"] == 0 for e in report["episodes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.fixtures, **{k: np.asarray(v) for k, v in fixtures.items()}, policy_action=np.asarray(fixture_actions))
    if args.video:
        import mediapy
        args.video.parent.mkdir(parents=True, exist_ok=True)
        # 250 frames per simulation second, played at 50 fps for wing visibility.
        mediapy.write_video(str(args.video), video_frames, fps=50)
    print(f"Report: {args.output}; passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
