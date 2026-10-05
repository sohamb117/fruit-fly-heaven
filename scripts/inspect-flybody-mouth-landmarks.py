#!/usr/bin/env python3
"""Export native mouth landmarks in the existing procedural fly mesh frame.

This is a kinematic inspection, not a controller or change to the body model.
Native lengths are centimetres. The existing scene uses ten units per cm.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/flybody-mujoco.xml"
SOURCE = ROOT / "references/flybody/flybody/fruitfly/assets/fruitfly.xml"
OFFSET = np.array([.13, .91, 0.])
SWAP_YZ = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def skin_point(data, world):
    rotation = data.xmat[1].reshape(3, 3)
    return OFFSET + 10 * SWAP_YZ @ rotation.T @ (world - data.qpos[:3])


def lowest_ellipsoid_point(model, data, gid, normal=np.array([0., 0., 1.])):
    rotation = data.geom_xmat[gid].reshape(3, 3)
    quadratic = rotation @ np.diag(model.geom_size[gid] ** 2) @ rotation.T
    return data.geom_xpos[gid] - quadratic @ normal / np.sqrt(normal @ quadratic @ normal)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    root = ET.parse(SOURCE).getroot()
    names = ["head", "rostrum", "haustellum", "labrum_left", "labrum_right"]
    bodies = {name: int(model.body(name).id) for name in names}
    geometries = {side: int(model.geom(f"labrum_{side}_lower_collision").id) for side in ["left", "right"]}
    report = {
        "scope": "Native mouth kinematics and original procedural UI mesh registration; no dynamics changed.",
        "production_xml_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "source_xml_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "native_units": "centimetres",
        "render_units_per_cm": 10,
        "mesh_root_anchor": OFFSET.tolist(),
        "world_to_existing_mesh": "[.13,.91,0] + 10 * swapYZ * transpose(rootRotation) * (worldPoint-rootPosition)",
        "existing_mesh_world_position": "10*swapYZ*rootPosition - (swapYZ*rootRotation*swapYZ)*[.13,.91,0]",
        "note": "The root registration must rotate the mesh offset with the body. Do not apply an additional world-vertical mouth offset.",
        "body_ids": bodies,
        "joint_ids": {name: int(model.joint(name).id) for name in ["rostrum", "haustellum"]},
        "mouth_geometries": [],
        "source_visual_meshes": [],
        "poses": [],
    }
    for side, gid in geometries.items():
        report["mouth_geometries"].append({"side": side, "id": gid,
            "body_id": int(model.geom_bodyid[gid]), "type": "ellipsoid",
            "semiaxes_cm": model.geom_size[gid].tolist(),
            "position_in_body_cm": model.geom_pos[gid].tolist(),
            "quaternion_in_body_wxyz": model.geom_quat[gid].tolist()})
    mesh_nodes = {mesh.attrib["name"]: mesh for mesh in root.findall("./asset/mesh")}
    for name in ["rostrum", "rostrum_bristle-brown", "haustellum", "haustellum_black", "labrum_left_lower", "labrum_right_lower"]:
        geom = root.find(f".//geom[@name='{name}']")
        mesh = mesh_nodes[geom.attrib["mesh"]]
        report["source_visual_meshes"].append({"name": name, "file": str(SOURCE.parent.relative_to(ROOT) / mesh.attrib["file"]),
            "obj_scale": [.1, .1, .1], "geom_position_in_body_cm": [float(x) for x in geom.attrib["pos"].split()],
            "geom_quaternion_in_body_wxyz": [float(x) for x in geom.attrib["quat"].split()],
            "transform_order": "Scale OBJ vertices by .1, apply XML geom quaternion/position, then current native body pose; do not treat OBJ origin as its joint anchor."})
    cases = [("neutral", 0., 0.), ("rostrum_extended", -1.24, 0.),
             ("haustellum_extended", 0., -1.59), ("both_extended", -1.24, -1.59),
             ("both_retracted", .183, .7)]
    maximum_surface_equation_error = 0.
    maximum_roundtrip_error = 0.
    for name, rostrum, haustellum in cases:
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
        for joint, angle in [("rostrum", rostrum), ("haustellum", haustellum)]:
            data.qpos[model.joint(joint).qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        points = {name: data.xpos[bid].copy() for name, bid in bodies.items()}
        points["labrum_hinge_midpoint"] = (points["labrum_left"] + points["labrum_right"]) / 2
        for side, gid in geometries.items():
            points[f"labrum_{side}_center"] = data.geom_xpos[gid].copy()
            points[f"labrum_{side}_floor_support"] = lowest_ellipsoid_point(model, data, gid)
            local = data.geom_xmat[gid].reshape(3, 3).T @ (points[f"labrum_{side}_floor_support"] - data.geom_xpos[gid])
            maximum_surface_equation_error = max(maximum_surface_equation_error, abs(np.sum((local / model.geom_size[gid]) ** 2) - 1))
        points["labrum_center_midpoint"] = (points["labrum_left_center"] + points["labrum_right_center"]) / 2
        points["labrum_floor_support_midpoint"] = (points["labrum_left_floor_support"] + points["labrum_right_floor_support"]) / 2
        landmarks = {key: {"world_cm": point.tolist(), "mesh_local": skin_point(data, point).tolist()} for key, point in points.items()}
        report["poses"].append({"name": name, "angles_rad": [rostrum, haustellum], "landmarks": landmarks,
            "polyline_landmarks": ["rostrum", "haustellum", "labrum_hinge_midpoint", "labrum_floor_support_midpoint"]})
    # Verify the renderer registration for nonidentity body attitudes too.
    rng = np.random.default_rng(888)
    for _ in range(20):
        data.qpos[:3] = rng.normal(size=3)
        quaternion = rng.normal(size=4); quaternion /= np.linalg.norm(quaternion)
        data.qpos[3:7] = quaternion
        mujoco.mj_forward(model, data)
        rotation = data.xmat[1].reshape(3, 3)
        mesh_rotation = SWAP_YZ @ rotation @ SWAP_YZ
        mesh_position = 10 * SWAP_YZ @ data.qpos[:3] - mesh_rotation @ OFFSET
        for bid in bodies.values():
            point = data.xpos[bid]
            reconstructed = mesh_position + mesh_rotation @ skin_point(data, point)
            maximum_roundtrip_error = max(maximum_roundtrip_error, float(np.max(np.abs(reconstructed - 10 * SWAP_YZ @ point))))
    report["validation"] = {"max_ellipsoid_surface_equation_error": float(maximum_surface_equation_error),
        "max_rotated_mesh_registration_error_scene_units": maximum_roundtrip_error,
        "random_root_poses_checked": 20}
    report["implementation_guidance"] = [
        "Reuse the existing material and segment primitives, but place each mouth segment from the exported native landmarks every frame, as the legs already do.",
        "Rostrum body origin is its hinge in the head. Haustellum body origin is the distal rostrum hinge. Labrum body origins are the labellar hinge endpoints, not the labellar tips.",
        "Render lower labella at native collision-ellipsoid centers with their actual orientations and radii; then the visible volume and contact volume coincide.",
        "A line/polyline ending at the labrum body origin is too short. A line ending at the ellipsoid center misses its surface by approximately 0.01 cm depending on pose.",
        "For a sloped surface with world outward normal n, C=R*diag(semiaxes^2)*transpose(R), support=center-C*n/sqrt(dot(n,C*n)). Prefer actual MuJoCo contact.pos for an active contact marker.",
        "Original FlyBody OBJ assets contain offset geometry: preserve their XML geom transforms if adopting them. The current original UI uses procedural ellipsoids, so these OBJ transforms are reference data rather than a request to replace its appearance.",
    ]
    assert maximum_surface_equation_error < 1e-12
    assert maximum_roundtrip_error < 1e-12
    output = ROOT / "reports/flybody-mouth-landmarks.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "validation": report["validation"],
                      "both_extended_landmarks": report["poses"][3]["landmarks"]}, indent=2))


if __name__ == "__main__":
    main()
