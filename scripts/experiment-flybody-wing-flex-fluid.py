#!/usr/bin/env python3
"""Matched-state audit of distal inertia-fluid fallback; no production changes."""
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/flybody-wing-flex-prototype"
BEFORE = OUT / "before-distal-fluid-correction"
CAPTURE = ROOT / "reports/flybody-solid-wing-repair/after-com/onset/capture.json"
PASSIVE_SOURCE = "https://raw.githubusercontent.com/google-deepmind/mujoco/3.13.0/src/engine/engine_passive.c"
DERIVATIVE_SOURCE = "https://raw.githubusercontent.com/google-deepmind/mujoco/3.13.0/src/engine/engine_derivative.c"
INTERACTION = 1e-300


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_sentinel(xml):
    root = ET.fromstring(xml)
    for side in ("left", "right"):
        geom = root.find(f".//geom[@name='wing_{side}_distal_inertial']")
        assert geom is not None
        geom.set("fluidshape", "ellipsoid")
        geom.set("fluidcoef", "0 0 0 0 0")
    return ET.tostring(root, encoding="unicode")


def model(xml, suppress):
    result = mujoco.MjModel.from_xml_string(configure_sentinel(xml) if suppress else xml)
    assert int(result.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
    if suppress:
        for side in ("left", "right"):
            geom = result.geom(f"wing_{side}_distal_inertial").id
            assert result.geom_fluid[geom, 0] == 1
            result.geom_fluid[geom, 0] = INTERACTION
    return result


def mapping(original, other):
    original_q, other_q, original_v, other_v, labels = [], [], [], [], []
    for j in range(original.njnt):
        target = other.joint(original.joint(j).name).id
        kind = int(original.jnt_type[j])
        assert kind == int(other.jnt_type[target])
        nq, nv = (7, 6) if kind == 0 else (4, 3) if kind == 1 else (1, 1)
        original_q.extend(range(original.jnt_qposadr[j], original.jnt_qposadr[j] + nq))
        other_q.extend(range(other.jnt_qposadr[target], other.jnt_qposadr[target] + nq))
        original_v.extend(range(original.jnt_dofadr[j], original.jnt_dofadr[j] + nv))
        other_v.extend(range(other.jnt_dofadr[target], other.jnt_dofadr[target] + nv))
        labels.extend(f"{original.joint(j).name}:{axis}" for axis in range(nv))
    return tuple(np.asarray(x) for x in (original_q, other_q, original_v, other_v)), labels


def forward(original, target, state, indices):
    original_q, target_q, original_v, target_v = indices
    data = mujoco.MjData(target)
    data.qpos[target_q] = np.asarray(state["qpos"])[original_q]
    data.qvel[target_v] = np.asarray(state["qvel"])[original_v]
    # The new flex coordinate and its velocity remain exactly zero in this
    # matched-state *force* assay; this is not a clamped flight rollout.
    for actuator in range(original.nu):
        j = target.actuator(original.actuator(actuator).name).id
        data.ctrl[j] = state["ctrl"][actuator]
        na = original.actuator_actnum[actuator]
        if na:
            data.act[target.actuator_actadr[j]:target.actuator_actadr[j]+na] = state["act"][original.actuator_actadr[actuator]:original.actuator_actadr[actuator]+na]
    mujoco.mj_forward(target, data)
    return data


def main():
    started = time.monotonic()
    assert (BEFORE / "manifest.json").exists(), "Preserve original prototype and replay first"
    capture = json.loads(CAPTURE.read_text())
    initial = [capture["initial"]["native"]]
    states = initial + [row["after"]["native"] for row in capture["frames"] if row["after"]["native"]["time"] < .13805]
    source_path = ROOT / "models/flybody-mujoco.xml"
    original = mujoco.MjModel.from_xml_path(str(source_path))
    models = {"baseline": original}
    for label, file in (("rigid", "rigid-split-control.xml"), ("flex_neutral", "flex.xml")):
        xml = (BEFORE / file).read_text()
        models[label + "_before"] = model(xml, False)
        # Audit the actual regenerated artifact, not only an in-memory sketch.
        corrected_xml = (OUT / file).read_text()
        for side in ("left", "right"):
            sentinel = ET.fromstring(corrected_xml).find(f".//geom[@name='wing_{side}_distal_inertial']")
            assert sentinel.get("fluidshape") == "ellipsoid"
            assert sentinel.get("fluidcoef") == "0 0 0 0 0"
        models[label + "_suppressed"] = model(corrected_xml, True)
    maps = {name: mapping(original, value) for name, value in models.items()}
    channels = ("qfrc_fluid", "qfrc_passive", "qfrc_spring", "qfrc_damper", "qfrc_gravcomp", "qfrc_bias")
    summaries = {name: {channel: {"maximumAbsoluteDifference": 0.} for channel in channels} for name in models if name != "baseline"}
    samples = []
    for state in states:
        data = {name: forward(original, value, state, maps[name][0]) for name, value in models.items()}
        row = {"time": state["time"], "conditions": {}}
        for name in summaries:
            ids, labels = maps[name]
            condition = {}
            for channel in channels:
                before = getattr(data["baseline"], channel)[ids[2]]
                after = getattr(data[name], channel)[ids[3]]
                error = after - before
                at = int(np.argmax(abs(error)))
                maximum = float(abs(error[at]))
                condition[channel] = maximum
                if maximum > summaries[name][channel]["maximumAbsoluteDifference"]:
                    summaries[name][channel] = {"maximumAbsoluteDifference": maximum, "signedDifference": float(error[at]),
                                                 "time": state["time"], "dof": labels[at],
                                                 "baseline": float(before[at]), "candidate": float(after[at])}
            row["conditions"][name] = condition
        samples.append(row)
    # Dense inertia matrices on shared coordinates must still agree for neutral
    # flexion. This catches frame and parallel-axis errors independently of drag.
    matrices = {}
    for name, value in models.items():
        matrix = np.zeros((value.nv, value.nv))
        mujoco.mj_fullM(value, data[name], matrix)
        ids = maps[name][0][3]
        matrices[name] = matrix[np.ix_(ids, ids)]
    mass_errors = {name: float(np.max(abs(value - matrices["baseline"]))) for name, value in matrices.items() if name != "baseline"}
    for name in ("rigid_suppressed", "flex_neutral_suppressed"):
        for channel in channels:
            assert summaries[name][channel]["maximumAbsoluteDifference"] < 1e-9, (name, channel, summaries[name][channel])
        assert mass_errors[name] < 1e-15
    assert summaries["rigid_before"]["qfrc_fluid"]["maximumAbsoluteDifference"] > 1e-5
    assert summaries["rigid_before"]["qfrc_spring"]["maximumAbsoluteDifference"] == 0
    assert summaries["rigid_before"]["qfrc_damper"]["maximumAbsoluteDifference"] == 0
    sentinel = []
    for side in ("left", "right"):
        name = "wing_" + side
        old_geom = original.geom(name + "_fluid").id
        rigid = models["rigid_suppressed"]
        new_geom = rigid.geom(name + "_fluid").id
        for field in ("geom_size", "geom_pos", "geom_quat", "geom_fluid"):
            np.testing.assert_array_equal(getattr(original, field)[old_geom], getattr(rigid, field)[new_geom])
        distal = rigid.body(name + "_distal").id
        sentinel_geom = rigid.geom(name + "_distal_inertial").id
        assert rigid.geom_bodyid[sentinel_geom] == distal and rigid.geom_fluid[sentinel_geom, 0] > 0
        sentinel.append({"body": name + "_distal", "geom": name + "_distal_inertial",
                         "interaction": float(rigid.geom_fluid[sentinel_geom, 0]), "originalProximalFluidUnchanged": True})
    report = {
        "passed": True, "nativeVersion": mujoco.__version__, "wallSeconds": time.monotonic() - started,
        "scope": "Matched q/qvel/ctrl/act at every captured 2 ms state before first baseline wing impact; no integration, tuning or neural rerun.",
        "statesCompared": len(states), "intervalSeconds": [states[0]["time"], states[-1]["time"]],
        "sourceHashes": {str(p.relative_to(ROOT)): hash_file(p) for p in (Path(__file__), source_path, CAPTURE, BEFORE / "flex.xml", BEFORE / "rigid-split-control.xml", OUT / "flex.xml", OUT / "rigid-split-control.xml", OUT / "result.json")},
        "engineSources": {"passive": PASSIVE_SOURCE, "derivative": DERIVATIVE_SOURCE},
        "finding": "New massive distal bodies acquired inertia-box fluid forces, including welded rigid children. All matched passive-force differences are fluid differences; original spring/damper channels are unchanged.",
        "correction": "Enable an ellipsoid sentinel on existing distal invisible inertial geom, then set only its compiled interaction to 1e-300. The positive coefficient suppresses inertia fallback; all explicit fluid forces are scaled to negligible magnitude. This is numerical suppression, not mathematical zero.",
        "limitations": ["MJCF parses fluidshape as a boolean; compiled-model override is mandatory.",
                        "Setting interaction to zero restores the unwanted inertia model.",
                        "fluidcoef=0 alone retains Stokes viscosity and added-mass terms.",
                        "Only current Euler integrator is supported: MuJoCo 3.13 implicit fluid derivatives are not interaction-scaled.",
                        "This validates instantaneous force and neutral-inertia equivalence, not trajectory/contact equivalence or flexible flight."],
        "sentinels": sentinel, "forceDifferencesByNamedDof": summaries,
        "denseSharedMassMatrixMaximumErrors": mass_errors, "samples": samples,
    }
    (OUT / "fluid-fallback-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "wallSeconds", "statesCompared", "forceDifferencesByNamedDof", "denseSharedMassMatrixMaximumErrors")}, indent=2))


if __name__ == "__main__":
    main()
