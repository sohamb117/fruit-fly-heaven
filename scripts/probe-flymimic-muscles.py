#!/usr/bin/env python3
"""Test the published FlyMimic Hill-muscle boundary in isolation.

Downloads the public meshes used by FlyGym, loads its unchanged musculoskeletal
XML, and compares 15 independent activation pulses with a zero-input reference.
This is the original tethered left-front-leg model, not a whole-fly simulation.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references/flygym"
XML = SOURCE / "src/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml"
CACHE = ROOT / "data/raw/flymimic-meshes"
REMOTE = "https://datasets.epfl.ch/nely-public-share/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a"
JOINTS = ["joint_LFCoxa_yaw", "joint_LFCoxa_pitch", "joint_LFCoxa_roll", "joint_LFTrochanter_yaw", "joint_LFTrochanter_pitch", "joint_LFTrochanter_roll", "joint_LFTibia_pitch"]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(XML.read_text())
    meshes = root.findall("./asset/mesh")
    def download(mesh):
        name = Path(mesh.get("file")).name
        path = CACHE / name
        if not path.exists():
            urllib.request.urlretrieve(f"{REMOTE}/{name}", path)
        mesh.set("file", str(path))
        return name, hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=6) as pool:
        mesh_sha = dict(pool.map(download, meshes))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    qids = [int(model.joint(name).qposadr[0]) for name in JOINTS]
    vids = [int(model.joint(name).dofadr[0]) for name in JOINTS]
    mocap = SOURCE / "src/flygym_demo/muscle_imitation/assets/mocap"
    qinit, vinit = np.load(mocap / "qpos/0002.npy")[0], np.load(mocap / "qvel/0002.npy")[0]
    def run(muscle=None):
        data = mujoco.MjData(model)
        data.qpos[qids] = qinit
        data.qvel[vids] = vinit
        mujoco.mj_forward(model, data)
        peak_force = 0.
        peak_activation = 0.
        if muscle is not None:
            data.ctrl[muscle] = .25
        for _ in range(200):
            mujoco.mj_step(model, data)
            if muscle is not None:
                peak_force = max(peak_force, abs(float(data.actuator_force[muscle])))
                peak_activation = max(peak_activation, float(data.act[model.actuator_actadr[muscle]]))
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
        return {"qpos": data.qpos[qids].tolist(), "activation": peak_activation, "force_native_units": peak_force,
                "thorax_position": data.xpos[model.body("Thorax").id].tolist()}
    baseline = run()
    cases = []
    for i in range(model.nu):
        measured = run(i)
        delta = np.asarray(measured["qpos"])-baseline["qpos"]
        cases.append({"name": model.actuator(i).name, "id": i, "tendon": model.tendon(int(model.actuator_trnid[i, 0])).name,
            "activation_time_constants_s": model.actuator_dynprm[i, :2].tolist(), "control": .25, "seconds": .02,
            "max_joint_difference_from_zero_rad": float(np.abs(delta).max()), "joint_delta_rad": delta.tolist(), **measured})
    report = {"scope": "Published FlyMimic left-front-leg Hill-muscle pulse test; original thorax tether retained. No BANC, whole-body locomotion, takeoff or landing claim.",
        "sources": {"flygym_revision": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
                    "xml_sha256": hashlib.sha256(XML.read_bytes()).hexdigest(), "mesh_sha256": mesh_sha,
                    "model_repository": "https://github.com/gizemozd/FlyMimic", "documentation": "https://neuromechfly.org/tutorials/6_muscle_imitation/"},
        "mujoco": mujoco.__version__, "dimensions": {k:int(getattr(model,k)) for k in ["nq","nv","nu","na","nbody","ntendon"]},
        "has_free_root_joint": bool(np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)),
        "all_actuators_are_muscle": bool(np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE) and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)),
        "tracked_joints": JOINTS, "baseline": baseline, "cases": cases,
        "passed": all(case["activation"] > .2 and case["max_joint_difference_from_zero_rad"] > 1e-6 for case in cases),
        "limitations": ["Only left-front leg is muscle-driven; right-front joints locked and other legs passive.", "Thorax is anchored in the published XML; this does not demonstrate support of a free body.", "Muscle parameter values are the published model fits, not measured BANC motor-neuron physiology.", "Native muscle/tendon naming and body topology differ from FlyBody and need explicit anatomical mapping and unit conversion."]}
    (ROOT / "reports/flymimic-muscle-probe.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["sources","cases","baseline"]},indent=2))


if __name__ == "__main__":
    main()
