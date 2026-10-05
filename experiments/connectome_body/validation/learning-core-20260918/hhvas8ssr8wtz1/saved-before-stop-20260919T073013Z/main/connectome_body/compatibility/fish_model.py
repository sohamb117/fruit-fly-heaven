"""A declared 3D MuJoCo extension of the pinned simZFish collision morphology.

The original simulator is Webots. This is NOT a numerically equivalent port:
native MuJoCo fluid forces replace Webots hydrodynamics, and two articulated
pectoral fins provide additional actuation. Every deviation is in the manifest.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ..data import download_verified
from ..util import atomic_json, digest_file, digest_json

REVISION = "eb5cf9cff445821e4e3f1636361aa7232e2d9988"
SOURCE_FILES = {
    "protos/Zebrafish_4mm_demo.proto": "046a8b5cb6fd8a78ac6aae6564b3478a7605e8dd21ab9f497385be6e46dd97a3",
    "worlds/40 OMR_Free_Zebrafish_L-L demo.wbt": "e05b7613cb92c2202ade091e83d2e3416279995c0ca93e201ec7b39d3b71ba85",
    "LICENSE": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
    "NOTICE": "3fe52344a49f0fd8847348922ee6b004ab509dec7115a087648596f6a92cb860",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# Original coordinates: x=lateral, y=up, z=tailward. New frame: x=forward,z=up.
ROTATION = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


@dataclass(frozen=True)
class FishModelConfig:
    physics_dt: float = 0.00005
    control_dt: float = 0.005
    tail_position_gain: float = 0.0001
    tail_velocity_gain: float = 0.0000001
    fin_position_gain: float = 0.000002
    fin_velocity_gain: float = 0.00000002
    fin_half_chord: float = 0.010
    fin_half_span: float = 0.020
    fin_half_thickness: float = 0.001
    fluid_blunt_drag: float = 0.5
    fluid_slender_drag: float = 0.1
    fluid_angular_drag: float = 1.0
    fluid_kutta_lift: float = 1.0
    fluid_magnus_lift: float = 1.0

    def validate(self):
        if any(not math.isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("Fish extension parameters must be positive finite numbers")
        if not math.isclose(
            self.control_dt / self.physics_dt, round(self.control_dt / self.physics_dt)
        ):
            raise ValueError(
                "Fish control interval must contain an integer number of physical steps"
            )


def _block(text, prefix):
    matches = list(re.finditer(prefix + r"\s*\{", text))
    if len(matches) != 1:
        raise ValueError(f"Expected one pinned VRML block: {prefix}")
    start, depth = matches[0].end(), 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if not depth:
            return text[start:end]
    raise ValueError("Unbalanced pinned VRML body definition")


def _numbers(text, name, count, bracket=False):
    prefix = rf"\b{re.escape(name)}\s+" + (r"\[\s*" if bracket else "")
    match = re.search(prefix + rf"((?:{NUMBER}\s*){{{count}}})", text)
    if match is None:
        raise ValueError(f"Missing {name} in pinned body source")
    return np.array(list(map(float, re.findall(NUMBER, match[1]))))


def source_morphology(path):
    text = re.sub(r"#[^\n]*", "", Path(path).read_text())
    records = []
    for index in range(8):
        if index == 0:
            bounds = _block(text, r"DEF HEAD_SEGMENT_BOUNDING Group")
            physics = _block(text, r"physics DEF DEVICE_PHYSICS Physics")
            translation, anchor = np.zeros(3), np.zeros(3)
        else:
            bounds = (
                _block(text, rf"DEF SMALL_SEGMENT_{index} Group")
                if index < 7
                else _block(text, r"DEF TAIL_SEGMENT_BOUNDING Group")
            )
            physics = (
                _block(text, rf"physics DEF SEGMENT_PHYSICS_SEG_{index} Physics")
                if index < 7
                else _block(text, r"physics DEF SEGMENT_PHYSICS_TAIL Physics")
            )
            solid = _block(text, rf"endPoint DEF SOLID_{index} Solid")
            joint = _block(text, rf"DEF HINGE_JOINT_{index} HingeJoint")
            translation = _numbers(solid, "translation", 3)
            anchor = _numbers(joint, "anchor", 3)
        cylinder = _block(bounds, r"geometry Cylinder")
        box = _block(bounds, r"geometry Box")
        translations = list(re.finditer(rf"\btranslation\s+((?:{NUMBER}\s*){{3}})", bounds))
        # Root has an explicit zero cylinder translation; other cylinders use
        # their parent frame. The final Transform always positions the box.
        box_position = np.array(list(map(float, re.findall(NUMBER, translations[-1][1]))))
        cylinder_position = (
            np.array(list(map(float, re.findall(NUMBER, translations[0][1]))))
            if len(translations) > 1
            else np.zeros(3)
        )
        inertia = _numbers(physics, "inertiaMatrix", 6, bracket=True)
        matrix = np.array(
            [
                [inertia[0], inertia[3], inertia[4]],
                [inertia[3], inertia[1], inertia[5]],
                [inertia[4], inertia[5], inertia[2]],
            ]
        )
        matrix = ROTATION @ matrix @ ROTATION.T
        moments, axes = np.linalg.eigh(matrix)
        if moments.min() <= 0:
            raise ValueError("Published fish inertia is not positive definite")
        source_inertia = matrix.copy()
        gap = moments[2] - moments[0] - moments[1]
        repair = None
        if gap > 0:
            if gap / moments.sum() > 0.005:
                raise ValueError("Published inertia needs a material, unapproved correction")
            # Project principal moments onto A+B>=C with the smallest Euclidean
            # change, keeping eigenvectors. Stay just inside the constraint so
            # serialization/diagonalization cannot flip a boundary inequality.
            corrected = moments + (gap / 3 + 1e-12 * moments.sum()) * np.array([1, 1, -1])
            matrix = (axes * corrected) @ axes.T
            repair = {
                "method": "minimal_principal_moment_triangle_projection",
                "original_principal_moments": moments.tolist(),
                "corrected_principal_moments": corrected.tolist(),
                "constraint_gap_over_original_trace": float(gap / moments.sum()),
                "relative_frobenius_change": float(
                    np.linalg.norm(matrix - source_inertia) / np.linalg.norm(source_inertia)
                ),
            }
        records.append(
            {
                "index": index,
                "translation": (ROTATION @ translation).tolist(),
                "joint_position": (ROTATION @ (anchor - translation)).tolist(),
                "mass": float(_numbers(physics, "mass", 1)[0]),
                "center_of_mass": (
                    ROTATION @ _numbers(physics, "centerOfMass", 3, bracket=True)
                ).tolist(),
                "inertia": [
                    matrix[0, 0],
                    matrix[1, 1],
                    matrix[2, 2],
                    matrix[0, 1],
                    matrix[0, 2],
                    matrix[1, 2],
                ],
                "source_inertia": [
                    source_inertia[0, 0],
                    source_inertia[1, 1],
                    source_inertia[2, 2],
                    source_inertia[0, 1],
                    source_inertia[0, 2],
                    source_inertia[1, 2],
                ],
                "inertia_feasibility_repair": repair,
                "cylinder_radius": float(_numbers(cylinder, "radius", 1)[0]),
                "cylinder_half_height": float(_numbers(cylinder, "height", 1)[0] / 2),
                "cylinder_position": (ROTATION @ cylinder_position).tolist(),
                "box_half_sizes": (np.abs(ROTATION) @ _numbers(box, "size", 3) / 2).tolist(),
                "box_position": (ROTATION @ box_position).tolist(),
            }
        )
    return records


def _vector(values):
    return " ".join(f"{float(value):.15g}" for value in values)


def model_xml(records, config):
    config.validate()
    root = ET.Element("mujoco", model="simZFish-derived-3D-extension")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="auto")
    ET.SubElement(
        root,
        "option",
        timestep=str(config.physics_dt),
        gravity="0 0 -981",
        density="0.001",
        viscosity="0.00001",
        integrator="implicitfast",
        iterations="30",
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        contype="0",
        conaffinity="0",
        fluidshape="ellipsoid",
        fluidcoef=_vector(
            [
                config.fluid_blunt_drag,
                config.fluid_slender_drag,
                config.fluid_angular_drag,
                config.fluid_kutta_lift,
                config.fluid_magnus_lift,
            ]
        ),
        rgba=".25 .65 .75 1",
    )
    world = ET.SubElement(root, "worldbody")
    head = ET.SubElement(world, "body", name="head", pos="0 0 0")
    ET.SubElement(head, "freejoint", name="root")
    parent = head
    joints = []
    for index, record in enumerate(records):
        body = (
            head
            if index == 0
            else ET.SubElement(
                parent, "body", name=f"segment_{index}", pos=_vector(record["translation"])
            )
        )
        if 0 < index < 7:
            name = f"tail_yaw_{index}"
            ET.SubElement(
                body,
                "joint",
                name=name,
                type="hinge",
                pos=_vector(record["joint_position"]),
                axis="0 0 1",
                range="-2 2",
            )
            joints.append((name, "tail"))
        ET.SubElement(
            body,
            "inertial",
            mass=str(record["mass"]),
            pos=_vector(record["center_of_mass"]),
            fullinertia=_vector(record["inertia"]),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"cylinder_{index}",
            type="cylinder",
            pos=_vector(record["cylinder_position"]),
            size=_vector([record["cylinder_radius"], record["cylinder_half_height"]]),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"box_{index}",
            type="box",
            pos=_vector(record["box_position"]),
            size=_vector(record["box_half_sizes"]),
        )
        parent = body
    for side, direction in (("left", 1), ("right", -1)):
        fin = ET.SubElement(
            head, "body", name=f"{side}_pectoral_fin", pos=_vector([-0.025, direction * 0.014, 0])
        )
        for movement, axis in (("sweep", [0, 0, direction]), ("pitch", [0, direction, 0])):
            name = f"{side}_fin_{movement}"
            ET.SubElement(
                fin, "joint", name=name, type="hinge", axis=_vector(axis), range="-1.2 1.2"
            )
            joints.append((name, "fin"))
        ET.SubElement(
            fin,
            "geom",
            name=f"{side}_fin_surface",
            type="ellipsoid",
            pos=_vector([0, direction * config.fin_half_span, 0]),
            size=_vector([config.fin_half_chord, config.fin_half_span, config.fin_half_thickness]),
            density="0.000915",
            rgba=".5 .8 .85 1",
        )
    actuators = ET.SubElement(root, "actuator")
    for name, kind in joints:
        kp = config.tail_position_gain if kind == "tail" else config.fin_position_gain
        kv = config.tail_velocity_gain if kind == "tail" else config.fin_velocity_gain
        limit = 2 if kind == "tail" else 1.2
        ET.SubElement(
            actuators,
            "position",
            name=name + "_target",
            joint=name,
            kp=str(kp),
            kv=str(kv),
            ctrlrange=_vector([-limit, limit]),
            forcerange="-0.01 0.01" if kind == "tail" else "-0.0001 0.0001",
        )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def prepare_fish_body(output, config=None):
    config = config or FishModelConfig()
    config.validate()
    output = Path(output).resolve()
    for name, checksum in SOURCE_FILES.items():
        url = f"https://ponyo.epfl.ch/api/v4/projects/124/repository/files/{quote(name, safe='')}/raw?ref={REVISION}"
        download_verified(url, output / "source" / name, checksum)
    records = source_morphology(output / "source/protos/Zebrafish_4mm_demo.proto")
    xml = model_xml(records, config)
    xml_path = output / "fish.xml"
    target = output / "manifest.json"
    if target.exists() and xml_path.read_text() != xml:
        raise FileExistsError("Fish extension parameters changed; prepare a new directory")
    # Compile before publishing the manifest; malformed or inertially invalid
    # models never become a runnable study input.
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    xml_path.write_text(xml)
    morphology = {
        "source_sha256": SOURCE_FILES["protos/Zebrafish_4mm_demo.proto"],
        "records": records,
        "extension": asdict(config),
        "total_model_mass_kg": float(model.body_mass.sum()),
        "neuronal_controller_included": False,
    }
    path = output / "morphology.json"
    if path.exists() and json.loads(path.read_text()) != morphology:
        raise FileExistsError("Pinned fish morphology changed")
    atomic_json(path, morphology)
    manifest = {
        "schema": "mujoco-embodiment-v1",
        "backend": "simzfish-3d-extension-v1",
        "body": "fish",
        "species": "Danio rerio larva, model extension",
        "source": {
            "repository": "https://ponyo.epfl.ch/proj/zebrafish/simzfish",
            "revision": REVISION,
            "citation": "https://doi.org/10.1126/scirobotics.adv4408",
            "license": "Apache-2.0",
            "is_synthetic": False,
        },
        "files": {
            "fish.xml": digest_file(xml_path),
            "morphology.json": digest_file(path),
            **{f"source/{name}": checksum for name, checksum in SOURCE_FILES.items()},
        },
        "model": "fish.xml",
        "root_body": "head",
        "body_length": 0.4,
        "forward_axis": [1, 0, 0],
        "control_dt": config.control_dt,
        "evidence": "simzfish_derived_3d_extension",
        "metres_per_native_length_unit": 0.01,
        "task_defaults": {
            "target_speed": 2.5,
            "dwell_seconds": 0.25,
            "initial_heading_span": math.pi / 3,
            "depth_offset_body_lengths": [0.5, 1.0],
        },
        "native_body_invariants": {
            "source_collision_geometry_and_masses": True,
            "source_inertias_with_recorded_feasibility_repair": True,
            "source_neural_circuit_included": False,
            "automatic_stabilizer": False,
            "prescribed_root_thrust": False,
            "state_clamping_or_teleportation": False,
        },
        "deviations_from_published_simulator": [
            "MuJoCo ellipsoid hydrodynamics replace Webots projected-area hydrodynamics; numerical equivalence is not claimed.",
            "The six tail joints use bounded position actuators with explicitly declared gains, not the Webots velocity servo implementation.",
            "One source inertia tensor violates the principal-moment triangle constraint by 0.0884% of its trace. A minimal principal-moment projection repairs it; morphology.json retains the original tensor and exact correction.",
            "Two pectoral fins each have sweep and pitch actuation. Their dimensions, gains and fluid coefficients are modeling assumptions, not measured parameters from the original simZFish release.",
            "All bodies are fully submerged. Hydrostatic buoyancy acts at each geometric center of volume; there is no water-surface stabilization or teleportation.",
            "The seventh almost-fixed tail hinge is fused. Visual meshes, cameras, the original controller and the display arena are omitted from these proprioceptive tasks.",
        ],
        "validation_status": "requires recorded propulsion, steering and vertical-control qualification; not an empirically validated zebrafish digital twin",
    }
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError("Keep the existing fish manifest and prepare a new directory")
    atomic_json(target, manifest)
    return {
        "manifest": str(target),
        "actuators": model.nu,
        "mass_kg": float(model.body_mass.sum()),
        "manifest_fingerprint": digest_json(manifest),
    }
