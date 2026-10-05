"""Recorded actuator/mechanics checks, deliberately separate from paper learning.

The probe waveforms and feedback law live here, outside the body interface and
outside every experimental controller. Passing does not establish realistic
zebrafish hydrodynamics, policy trainability, or a biological topology advantage.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..util import atomic_json, digest_file, digest_json
from .bodies import BodySpec, make_body
from .robustness import PerturbationSpec, make_perturbed_body
from .runtime import hardware_profile


def body_interface_identity(name):
    root = Path(__file__).parent
    files = ["bodies.py", "robustness.py", "qualification.py"]
    files += (
        ["worm_body.py", "worm_native.py", "native/worm_bridge.cpp"]
        if name == "worm"
        else ["fish_body.py", "fish_model.py"]
    )
    return digest_json({name: digest_file(root / name) for name in files})


def _pose(body, name):
    if name == "worm":
        return body.engine.rods[:, :2].mean(0).copy(), body._frame()[:, 0].copy()
    return body.position().copy(), body._kinematics()[0][:, 0].copy()


def _probe_action(body, name, probe, step):
    t = step * body.control_dt
    if name == "worm":
        if probe == "passive":
            return np.full(48, -1.0)
        if probe == "posture_feedback":
            drive = np.clip(20 * (body.target_curvature - body._shape()), -1, 1)
        else:
            direction = -1 if probe == "backward" else 1
            drive = np.sin(2 * np.pi * (direction * 0.5 * t - np.linspace(0, 1, 24)))
            if probe.startswith("turn_"):
                drive[:8] += 0.5 if probe == "turn_positive" else -0.5
        activation = np.clip(np.r_[np.maximum(drive, 0), np.maximum(-drive, 0)], 0, 1)
        return 2 * activation - 1
    action = np.zeros(10)
    if probe == "passive":
        return action
    direction = -1 if probe == "backward" else 1
    amplitude = 0.4 if probe == "backward" else 0.6
    action[:6] = amplitude / 2 * np.sin(2 * np.pi * 30 * t - direction * np.arange(6) * 0.6)
    if probe.startswith("turn_"):
        action[:6] += 0.025 if probe == "turn_positive" else -0.025
    if probe in ("ascend", "descend"):
        pitch = -0.8 if probe == "ascend" else 0.8
        action[7], action[9] = pitch / 1.2, -pitch / 1.2
    return action


def qualify_body(name, manifest, output, *, seed=0):
    if name not in ("worm", "fish") or type(seed) is not int or seed < 0:
        raise ValueError("Qualify worm or fish using a nonnegative seed")
    manifest, output = Path(manifest).resolve(), Path(output)
    request = {
        "schema": "body-mechanics-qualification-v1",
        "body": name,
        "model_manifest_sha256": digest_file(manifest),
        "body_interface_identity": body_interface_identity(name),
        "seed": seed,
    }
    if output.exists():
        record = json.loads(output.read_text())
        if any(record.get(key) != value for key, value in request.items()) or record.get(
            "fingerprint"
        ) != digest_json({k: v for k, v in record.items() if k != "fingerprint"}):
            raise FileExistsError("Keep the existing qualification and choose a new output")
        return record
    probes = ["passive", "forward", "backward", "turn_positive", "turn_negative", "failed"]
    probes += ["posture_feedback"] if name == "worm" else ["ascend", "descend"]
    records = {}
    for probe in probes:
        task = "locomotion" if name == "worm" else "swimming"
        if probe == "posture_feedback":
            task = "posture"
        parameters = {"initial_noise": 0}
        if name == "worm":
            parameters["disturbance_amplitude"] = 0
        spec = BodySpec(
            name,
            task,
            horizon=1000 if name == "worm" else 400,
            model_manifest=str(manifest),
            parameters=parameters,
        )
        body = (
            make_perturbed_body(spec, PerturbationSpec(name="failed", actuator_scale=0))
            if probe == "failed"
            else make_body(spec)
        )
        try:
            body.reset(seed, "validation")
            start, forward = _pose(body, name)
            started = time.perf_counter()
            for step in range(spec.horizon):
                action = _probe_action(body, name, probe, step)
                obs, _, terminated, truncated, info = body.step(action)
                if not np.isfinite(obs).all():
                    raise FloatingPointError("Body qualification produced a nonfinite observation")
                if terminated or truncated:
                    break
            seconds = time.perf_counter() - started
            end, last_forward = _pose(body, name)
            displacement = (end - start) / body.length
            turn = math.atan2(
                forward[0] * last_forward[1] - forward[1] * last_forward[0],
                forward[:2] @ last_forward[:2],
            )
            records[probe] = {
                "displacement_body_lengths": displacement.tolist(),
                "forward_displacement_body_lengths": float(displacement @ forward),
                "heading_change_radians": turn,
                "physical_seconds": (step + 1) * body.control_dt,
                "wall_seconds": seconds,
                "control_steps_per_second": (step + 1) / seconds,
                "completed_horizon": step + 1 == spec.horizon,
                "final_metrics": info,
                "body_fingerprint": body.fingerprint,
            }
        finally:
            body.close()
    forward = records["forward"]["forward_displacement_body_lengths"]
    backward = records["backward"]["forward_displacement_body_lengths"]
    failed = records["failed"]["forward_displacement_body_lengths"]
    left, right = [
        records[key]["heading_change_radians"] for key in ("turn_positive", "turn_negative")
    ]
    criteria = {
        "all_probes_finite_and_complete": all(
            row["completed_horizon"] and not row["final_metrics"]["numerical_failure"]
            for row in records.values()
        ),
        "forward_actuation": forward > 0.5,
        "reverse_actuation": backward < -0.5,
        "bidirectional_steering": left * right < 0 and min(abs(left), abs(right)) > 0.05,
        "failed_actuation_removes_propulsion": abs(failed) < 0.1 * abs(forward),
    }
    if name == "worm":
        criteria["nonpassive_posture_tracking"] = bool(
            records["posture_feedback"]["final_metrics"]["success"]
        )
    else:
        criteria["vertical_actuation_both_directions"] = (
            records["ascend"]["displacement_body_lengths"][2] > 0.5
            and records["descend"]["displacement_body_lengths"][2] < -0.5
        )
    record = {
        **request,
        "hardware": hardware_profile("cpu", 1),
        "probe_definitions": "Fixed functions in qualification.py; no learned controller and no built-in neural circuit",
        "criteria": criteria,
        "passed": all(criteria.values()),
        "probes": records,
        "scope": "Actuation, finite mechanics and measurable task variables only; not biological fidelity or evidence of learned control",
    }
    record["fingerprint"] = digest_json(record)
    atomic_json(output, record)
    return record
