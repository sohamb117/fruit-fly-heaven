#!/usr/bin/env python3
"""Native wing registration fixtures, including original-Euler comparison."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    xml=ROOT/"models/flybody-mujoco.xml"
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    metadata=json.loads((ROOT/"models/flybody-mujoco.json").read_text())
    report={"scope":"Read-only native wing geometry and original procedural renderer registration; no control or dynamics changes.",
        "xml_sha256":hashlib.sha256(xml.read_bytes()).hexdigest(),"mujoco":mujoco.__version__,"native_units":"cm",
        "joints":{},"cases":[],"interpretation":[]}
    cases=[("resting",[1.5,.7,-1.],[1.5,.7,-1.]),("zero_joints",[0,0,0],[0,0,0]),
        ("asymmetric",[-.8,.5,.7],[.2,-.7,-.4]),("opposite_strokes",[-1.2,.8,2.3],[1.2,-.7,-.9])]
    rng=np.random.default_rng(888)
    for i in range(12):
        cases.append((f"random_root_{i}",[float(rng.uniform(-1.3,1.3)),float(rng.uniform(-.8,1.4)),float(rng.uniform(-1.1,2.7))],
                     [float(rng.uniform(-1.3,1.3)),float(rng.uniform(-.8,1.4)),float(rng.uniform(-1.1,2.7))]))
    for index,(name,left,right) in enumerate(cases):
        mujoco.mj_resetData(model,data)
        for joint in metadata["joints"]:data.qpos[joint["qpos"]]=joint["neutral"]
        data.qpos[:7]=[0,0,1,1,0,0,0]
        if index>=4:
            data.qpos[:3]=rng.normal(size=3)
            quat=rng.normal(size=4);data.qpos[3:7]=quat/np.linalg.norm(quat)
        for side,angles in [("left",left),("right",right)]:
            for axis,angle in zip(["yaw","roll","pitch"],angles):
                joint=model.joint(f"wing_{axis}_{side}");data.qpos[joint.qposadr[0]]=angle
                report["joints"][joint.name]={"id":joint.id,"axis":joint.axis.tolist(),"qpos":int(joint.qposadr[0])}
        mujoco.mj_forward(model,data)
        case={"name":name,"qpos":data.qpos.tolist(),"root_rotation":data.xmat[1].tolist(),"wings":[]}
        for k,(side,angles) in enumerate([("left",left),("right",right)]):
            geom=model.geom(f"wing_{side}_fluid");body=model.body(f"wing_{side}")
            rotation=data.geom_xmat[geom.id].reshape(3,3);center=data.geom_xpos[geom.id];anchor=data.xpos[body.id]
            sign=1 if np.dot(rotation[:,2],center-anchor)>=0 else -1
            span=sign*rotation[:,2];tip=center+geom.size[2]*span
            relative=data.xmat[1].reshape(3,3).T@span
            # THREE Euler XYZ is Rx*Ry*Rz for column vectors. The original
            # renderer supplies [roll,yaw,pitch] with a whole-angle side sign.
            roll,yaw,pitch=np.asarray([angles[1],angles[0],angles[2]])*(1 if k else -1)
            vertical=np.cos(roll)*np.sin(pitch)+np.sin(roll)*np.sin(yaw)*np.cos(pitch)
            case["wings"].append({"side":side,"body":body.id,"geom":geom.id,"angles_yaw_roll_pitch":angles,
                "world_anchor_cm":anchor.tolist(),"world_center_cm":center.tolist(),"world_distal_span_axis":span.tolist(),
                "world_distal_tip_cm":tip.tolist(),"geom_rotation":rotation.reshape(-1).tolist(),"geom_size_cm":geom.size.tolist(),
                "native_span_elevation_degrees":float(np.degrees(np.arcsin(abs(relative[2])))),
                "original_euler_span_elevation_degrees":float(np.degrees(np.arcsin(abs(vertical))))})
        report["cases"].append(case)
    report["interpretation"]=[
        "At resting joint values [1.5,.7,-1], both native wing spans are near horizontal; the old Euler renderer makes one wing almost vertical. This is a rendering mismatch.",
        "Native joint order is yaw about Z, roll about X, pitch about Y, within nonidentity and mirrored native wing body frames. Directly assigning reordered angles to a THREE Euler XYZ group does not preserve that kinematic chain.",
        "Use the native fluid-geom center and frame to orient the original ellipsoid. Its native axes are thickness/chord/span; the existing UI ellipsoid axes are span/thickness/chord.",
        "The helper preserves the original wing mesh center offset and shape; it changes only its parent group pose. Its group origin is decorative and need not equal the native anatomical wing hinge; the returned anchor is the true hinge.",
        "Original span radius is 1.18 scene units, native span radius 1.14; preserving the old shape leaves a 0.04 scene-unit distal outline difference. Optional nativeHalfExtents can align the full ellipsoid envelope.",
        "Zero wing-power is not proof that a moving physical wing is at rest: inertia and springs still evolve its joints. Historical observer frames lack wing qpos, so individual past wing attitudes cannot be reconstructed uniquely from those logs."]
    output=ROOT/"reports/flybody-wing-landmarks.json";output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"resting":report["cases"][0]["wings"],"cases":len(cases)},indent=2))

if __name__=="__main__":main()
