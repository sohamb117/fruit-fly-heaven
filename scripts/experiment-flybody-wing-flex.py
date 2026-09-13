#!/usr/bin/env python3
"""Offline, one-hinge-per-wing mechanical prototype. Never edits production.

Run after other benchmarks finish:
  uv run --offline --with mujoco --with numpy python scripts/experiment-flybody-wing-flex.py

This is a local force/deflection calibration, not a flight or collision remedy.
The anatomical bending line, damping, and distributed aerodynamics are not known.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/flybody-wing-flex-prototype"
SOURCE = ROOT / "models/flybody-mujoco.xml"
CAUSAL = ROOT / "reports/flybody-solid-wing-repair/after-com/wing-contact-causal.json"
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6361194/"
DISTAL_FLUID_INTERACTION = 1e-300


def numbers(text):
    return np.fromstring(text, sep=" ")


def fmt(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).flat)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(quat):
    result = np.zeros(9)
    q = np.asarray(quat, dtype=float)
    mujoco.mju_quat2Mat(result, q / np.linalg.norm(q))
    return result.reshape(3, 3)


def configure_native_model(model):
    """Required prototype override: prevent extra distal inertia-box air drag.

    The original whole-wing fluid geom remains proximal. MJCF fluidshape is a
    boolean, so a positive-negligible compiled interaction cannot be set in XML.
    Exactly zero would restore fallback. 3.13 implicit fluid Jacobians are not
    interaction-scaled; only the existing Euler model is supported here.
    """
    assert int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
    for side in ("left", "right"):
        geom = model.geom(f"wing_{side}_distal_inertial").id
        assert model.geom_fluid[geom, 0] == 1
        model.geom_fluid[geom, 0] = DISTAL_FLUID_INTERACTION


def clipped_ellipsoid(geom, hinge, span):
    """Two convex point clouds clipped at one common physical hinge plane.

    Boundary-ring points are on the exact ellipsoid/plane intersection. Surface
    tessellation is an approximation: a rigid split control is emitted as well.
    """
    size = numbers(geom.get("size"))
    center = numbers(geom.get("pos"))
    rot = rotation(numbers(geom.get("quat")))
    longitude = np.arange(96) * (2 * np.pi / 96)
    latitude = np.linspace(0, np.pi, 49)
    unit = np.stack([
        (np.sin(latitude[:, None]) * np.cos(longitude)).ravel(),
        (np.sin(latitude[:, None]) * np.sin(longitude)).ravel(),
        np.broadcast_to(np.cos(latitude[:, None]), (49, 96)).ravel(),
    ], axis=1)
    points = (unit * size) @ rot.T + center
    plane = size * (rot.T @ span)
    offset = np.dot(hinge - center, span) / np.linalg.norm(plane)
    assert abs(offset) < 1, "Hinge must intersect each original collision ellipsoid"
    normal = plane / np.linalg.norm(plane)
    tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    ring = offset * normal + np.sqrt(1 - offset**2) * (
        np.cos(longitude[:, None]) * tangent + np.sin(longitude[:, None]) * bitangent)
    ring = (ring * size) @ rot.T + center
    distance = (points - hinge) @ span
    parts = [np.vstack([points[distance <= 0], ring]), np.vstack([points[distance >= 0], ring])]
    # A numerical support-function error over a fixed spherical sample; not a
    # claim of an exact Hausdorff bound or exact primitive collision equivalence.
    directions = unit[::13]
    exact = directions @ center + np.linalg.norm((directions @ rot) * size, axis=1)
    approximate = np.max(directions @ np.vstack(parts).T, axis=1)
    return parts, float(np.max(exact - approximate))


def generate(stiffness, damping_ratio):
    source = ET.parse(SOURCE)
    root = source.getroot()
    original = mujoco.MjModel.from_xml_path(str(SOURCE))
    asset = root.find("asset")
    contact = root.find("contact")
    original_exclusions = list(contact)
    wing_specs = []
    for side in ("left", "right"):
        name = "wing_" + side
        body = root.find(f".//body[@name='{name}']")
        inertial = body.find("inertial")
        center = numbers(inertial.get("pos"))
        quat = numbers(inertial.get("quat"))
        rot = rotation(quat)
        mass = float(inertial.get("mass"))
        old_inertia = numbers(inertial.get("diaginertia"))
        inertial_geom = body.find(f"geom[@name='{name}_inertial']")
        half_size = numbers(inertial_geom.get("size"))
        a, b, length = half_size
        expected = mass / 3 * np.array([b*b + length*length, a*a + length*length, a*a + b*b])
        np.testing.assert_allclose(old_inertia, expected, atol=1e-20, rtol=1e-12)
        outward_sign = 1 if np.dot(rot[:, 2], center) > 0 else -1
        span = outward_sign * rot[:, 2]
        # Chord-axis bend; +q moves the distal tip toward the original box normal.
        axis = outward_sign * rot[:, 1]
        normal = rot[:, 0]
        distal_name = name + "_distal"
        distal = ET.SubElement(body, "body", {"name": distal_name, "pos": fmt(center)})
        spring = stiffness * 1000 * length**2  # N/m -> (g cm/s²)/cm, then lever².
        half_mass = mass / 2
        half_inertia = half_mass / 3 * np.array([b*b + (length/2)**2, a*a + (length/2)**2, a*a + b*b])
        distal_hinge_inertia = half_inertia[1] + half_mass * (length/2)**2
        damping = 2 * damping_ratio * np.sqrt(spring * distal_hinge_inertia)
        ET.SubElement(distal, "joint", {
            "name": name + "_bend", "type": "hinge", "axis": fmt(axis),
            "limited": "false", "stiffness": fmt([spring]), "springref": "0",
            "damping": fmt([damping]), "armature": "0", "frictionloss": "0",
        })
        # Split the existing *box* inertia exactly; collision shape is independent
        # because the production compiler has inertiafromgeom=false.
        inertial.set("mass", fmt([half_mass]))
        inertial.set("pos", fmt(center - span * length/2))
        inertial.set("diaginertia", fmt(half_inertia))
        ET.SubElement(distal, "inertial", {
            "mass": fmt([half_mass]), "pos": fmt(span * length/2),
            "quat": fmt(quat), "diaginertia": fmt(half_inertia),
        })
        inertial_geom.set("size", fmt([a, b, length/2]))
        inertial_geom.set("pos", fmt(center - span * length/2))
        distal_inertial_geom = copy.deepcopy(inertial_geom)
        distal_inertial_geom.set("name", distal_name + "_inertial")
        distal_inertial_geom.set("pos", fmt(span * length/2))
        # Existing invisible, noncolliding geom acts as fluid-selection sentinel.
        # Its compiled interaction MUST be changed by configure_native_model.
        distal_inertial_geom.set("fluidshape", "ellipsoid")
        distal_inertial_geom.set("fluidcoef", "0 0 0 0 0")
        distal.append(distal_inertial_geom)
        errors = {}
        for geom in list(body.findall("geom")):
            if not geom.get("name", "").endswith("_collision"):
                continue
            parts, support_error = clipped_ellipsoid(geom, center, span)
            errors[geom.get("name")] = support_error
            body.remove(geom)
            for i, parent in enumerate((body, distal)):
                new_name = geom.get("name") + ("_proximal" if i == 0 else "_distal")
                ET.SubElement(asset, "mesh", {
                    "name": new_name + "_mesh", "scale": "1 1 1",
                    "vertex": fmt(parts[i] - (center if i else 0)),
                })
                attributes = {key: value for key, value in geom.attrib.items()
                              if key not in ("name", "type", "pos", "quat", "size")}
                attributes.update(name=new_name, type="mesh", mesh=new_name + "_mesh")
                ET.SubElement(parent, "geom", attributes)
        ET.SubElement(distal, "site", {
            "name": name + "_tip_probe", "pos": fmt(span * length),
            "size": ".001", "rgba": "1 0 0 1", "group": "3",
        })
        wing_specs.append({
            "side": side, "originalBody": name, "distalBody": distal_name,
            "massG": mass, "originalCenterCm": center.tolist(), "originalInertiaGcm2": old_inertia.tolist(),
            "hingeInOriginalWingFrameCm": center.tolist(), "bendAxis": axis.tolist(),
            "outwardSpan": span.tolist(), "surfaceNormal": normal.tolist(),
            "probeLeverCm": length, "tipStiffnessNPerM": stiffness,
            "rotationalSpringNative": spring, "dampingNative": damping,
            "dampingRatioPrior": damping_ratio, "distalInertiaAboutHingeGcm2": distal_hinge_inertia,
            "distalHeldProximalNaturalFrequencyHz": float(np.sqrt(spring / distal_hinge_inertia) / (2*np.pi)),
            "inputMeshSupportSampleErrorCm": errors,
            "compiledHullSupportErrorMeasured": False,
        })
    # Preserve every existing explicit exclusion for all corresponding segments.
    expansions = {s["originalBody"]: [s["originalBody"], s["distalBody"]] for s in wing_specs}
    exclusions = {tuple(sorted([x.get("body1"), x.get("body2")])) for x in contact}

    def exclude(a, b):
        pair = tuple(sorted([a, b]))
        if a != b and pair not in exclusions:
            ET.SubElement(contact, "exclude", {"name": "prototype_" + a + "__" + b, "body1": a, "body2": b})
            exclusions.add(pair)

    for item in original_exclusions:
        for a in expansions.get(item.get("body1"), [item.get("body1")]):
            for b in expansions.get(item.get("body2"), [item.get("body2")]):
                exclude(a, b)
    for spec in wing_specs:
        old_body = original.body(spec["originalBody"]).id
        parent_weld = original.body_weldid[original.body_parentid[old_body]]
        exclude(spec["originalBody"], spec["distalBody"])
        # Original wing/parent-weld collisions were filtered by MuJoCo. A new
        # grandchild must not accidentally create those self-contact candidates.
        for bid in range(1, original.nbody):
            if original.body_weldid[bid] == parent_weld:
                exclude(spec["distalBody"], original.body(bid).name)
    ET.indent(source, space="  ")
    source.write(OUT / "flex.xml", encoding="unicode")
    rigid = copy.deepcopy(source)
    for spec in wing_specs:
        body = rigid.find(f".//body[@name='{spec['distalBody']}']")
        body.remove(body.find("joint"))
    rigid.write(OUT / "rigid-split-control.xml", encoding="unicode")
    return original, wing_specs


def spatial_moments(model, data, ids):
    mass = float(sum(model.body_mass[i] for i in ids))
    center = sum(model.body_mass[i] * data.xipos[i] for i in ids) / mass
    inertia = np.zeros((3, 3))
    for i in ids:
        rot = data.ximat[i].reshape(3, 3)
        offset = data.xipos[i] - center
        inertia += rot @ np.diag(model.body_inertia[i]) @ rot.T
        inertia += model.body_mass[i] * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return mass, center, inertia


def verify_invariants(original, candidate, specs):
    a, b = mujoco.MjData(original), mujoco.MjData(candidate)
    # Map original coordinates by name; adding a child changes later addresses.
    for i in range(original.njnt):
        new = candidate.joint(original.joint(i).name).id
        assert original.jnt_type[i] == candidate.jnt_type[new]
        n = 7 if int(original.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE) else 1
        b.qpos[candidate.jnt_qposadr[new]:candidate.jnt_qposadr[new]+n] = a.qpos[original.jnt_qposadr[i]:original.jnt_qposadr[i]+n]
    mujoco.mj_forward(original, a)
    mujoco.mj_forward(candidate, b)
    rows = []
    for spec in specs:
        before = spatial_moments(original, a, [original.body(spec["originalBody"]).id])
        after = spatial_moments(candidate, b, [candidate.body(spec["originalBody"]).id, candidate.body(spec["distalBody"]).id])
        errors = [float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x, y in zip(before, after)]
        assert errors[0] < 1e-18 and errors[1] < 1e-13 and errors[2] < 1e-18
        rows.append({"side": spec["side"], "massErrorG": errors[0], "comErrorCm": errors[1], "inertiaTensorErrorGcm2": errors[2]})
    before = spatial_moments(original, a, range(1, original.nbody))
    after = spatial_moments(candidate, b, range(1, candidate.nbody))
    whole_errors = [float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x, y in zip(before, after)]
    assert whole_errors[0] < 1e-18 and whole_errors[1] < 1e-13 and whole_errors[2] < 1e-18
    original_dofs = {original.joint(i).name for i in range(original.njnt)}
    new_names = {candidate.joint(i).name for i in range(candidate.njnt)} - original_dofs
    assert new_names == {"wing_left_bend", "wing_right_bend"}
    assert candidate.nv == original.nv + 2 and candidate.nu == original.nu and candidate.neq == original.neq
    assert int(candidate.joint("free").type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    new_ids = {candidate.joint(name).id for name in new_names}
    assert not any(int(candidate.actuator_trnid[i, 0]) in new_ids
                   for i in range(candidate.nu)
                   if int(candidate.actuator_trntype[i]) in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)))
    for i in range(original.nu):
        assert original.actuator(i).name == candidate.actuator(i).name
        assert original.actuator_trntype[i] == candidate.actuator_trntype[i]
        assert original.actuator_dyntype[i] == candidate.actuator_dyntype[i]
        transmission = int(original.actuator_trntype[i])
        old_target, new_target = int(original.actuator_trnid[i, 0]), int(candidate.actuator_trnid[i, 0])
        if transmission in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            assert original.joint(old_target).name == candidate.joint(new_target).name
        elif transmission == int(mujoco.mjtTrn.mjTRN_BODY):
            assert original.body(old_target).name == candidate.body(new_target).name
        else:
            raise AssertionError("Unreviewed actuator transmission in source model")
        assert original.actuator_trnid[i, 1] == candidate.actuator_trnid[i, 1] == -1
        for field in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange", "actuator_gear"):
            np.testing.assert_array_equal(getattr(original, field)[i], getattr(candidate, field)[i])
    return {"perWing": rows, "wholeBodyErrorsMassComInertia": whole_errors,
            "additionalDofs": 2, "newActuators": 0, "newEqualityConstraints": 0,
            "rootFreeJointPreserved": True, "originalActuatorParametersPreserved": True,
            "originalActuatorTransmissionTypesAndTargetNamesPreserved": True}


def force_probe(xml, specs, duration):
    model = mujoco.MjModel.from_xml_path(str(xml))
    configure_native_model(model)
    assert int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
    # Isolate elastic response. No gravity, fluid, contact, or active muscles.
    # All original joints (including free root) and their passive mechanics stay.
    model.opt.gravity[:] = 0
    model.opt.density = 0
    model.opt.viscosity = 0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    rows = []
    for spec in specs:
        proximal = model.body(spec["originalBody"]).id
        distal = model.body(spec["distalBody"]).id
        joint = model.joint(spec["originalBody"] + "_bend")
        dof, coordinate = int(joint.dofadr[0]), int(joint.qposadr[0])
        site = model.site(spec["originalBody"] + "_tip_probe").id
        local_normal = np.asarray(spec["surfaceNormal"])
        for force_micronewton in (-2., -1., -.5, -.25, .25, .5, 1., 2.):
            data = mujoco.MjData(model)
            data.qpos[:] = model.qpos_spring
            mujoco.mj_forward(model, data)
            initial_com = data.subtree_com[1].copy()
            max_root_generalized_force = 0.
            samples = []
            for step in range(round(duration / model.opt.timestep)):
                # Refresh kinematics before computing body-fixed force vectors
                # and point Jacobians; post-step caches otherwise lag qpos.
                mujoco.mj_step1(model, data)
                # Corotating opposite forces at the *same world point* produce
                # zero external wrench. The proximal reaction is not a clamp.
                force = data.xmat[proximal].reshape(3, 3) @ local_normal * (force_micronewton * .1)
                point = data.site_xpos[site].copy()
                data.qfrc_applied[:] = 0
                mujoco.mj_applyFT(model, data, force, np.zeros(3), point, distal, data.qfrc_applied)
                mujoco.mj_applyFT(model, data, -force, np.zeros(3), point, proximal, data.qfrc_applied)
                max_root_generalized_force = max(max_root_generalized_force, float(np.max(abs(data.qfrc_applied[:6]))))
                mujoco.mj_step2(model, data)
                if step >= round(.8 * duration / model.opt.timestep):
                    samples.append([float(data.qpos[coordinate]), float(data.qvel[dof])])
            mujoco.mj_forward(model, data)
            values = np.asarray(samples)
            angle = float(values[:, 0].mean())
            deflection_m = spec["probeLeverCm"] * .01 * np.sin(angle)
            stiffness = force_micronewton * 1e-6 / deflection_m
            # Native spring, applied moment, and explicit damping should balance.
            residual = float(data.qfrc_applied[dof] + data.qfrc_passive[dof])
            row = {"side": spec["side"], "forceMicroNewton": force_micronewton,
                   "finalAngleRad": float(data.qpos[coordinate]), "meanAngleRad": angle,
                   "tailAnglePeakToPeakRad": float(np.ptp(values[:, 0])),
                   "deflectionMicrometers": deflection_m * 1e6, "effectiveStiffnessNPerM": float(stiffness),
                   "fractionalStiffnessError": float(abs(stiffness / spec["tipStiffnessNPerM"] - 1)),
                   "maxRootAppliedGeneralizedForce": max_root_generalized_force,
                   "centerOfMassDriftCm": float(np.linalg.norm(data.subtree_com[1] - initial_com)),
                   "finalElasticAppliedDampingResidualNative": residual,
                   "finalRootVelocity": data.qvel[:6].tolist(),
                   "nativeWarningCounts": data.warning.number.tolist()}
            assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
            assert row["fractionalStiffnessError"] < .015, row
            assert row["tailAnglePeakToPeakRad"] < 2e-5, row
            assert max_root_generalized_force < 1e-12, row
            assert not any(row["nativeWarningCounts"]), row
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiffness", type=float, default=.024)
    parser.add_argument("--damping-ratio", type=float, default=.2)
    parser.add_argument("--duration", type=float, default=.12)
    args = parser.parse_args()
    assert .020 <= args.stiffness <= .031
    assert args.damping_ratio > 0 and args.duration >= .05
    OUT.mkdir(parents=True, exist_ok=True)
    assert CAUSAL.exists(), "Require the source-matched wing/environment contact ablation report"
    causal = json.loads(CAUSAL.read_text())
    assert causal.get("passed"), "Contact-causality audit must have passed"
    causal_cases = {case["mode"]: case for case in causal["cases"]}
    assert causal_cases["baseline"]["metrics"]["firstOverturnedSeconds"] is not None
    assert causal_cases["no_wing_environment"]["metrics"]["firstOverturnedSeconds"] is None
    assert causal["preImpactParity"]["maximumRootQposError"] == 0
    assert causal["preImpactParity"]["maximumRootQvelError"] == 0
    started = time.monotonic()
    original, specs = generate(args.stiffness, args.damping_ratio)
    candidate = mujoco.MjModel.from_xml_path(str(OUT / "flex.xml"))
    configure_native_model(candidate)
    invariants = verify_invariants(original, candidate, specs)
    probes = force_probe(OUT / "flex.xml", specs, args.duration)
    report = {
        "status": "offline mechanical prototype only; not installed or validated in flight",
        "passed": True, "nativeVersion": mujoco.__version__, "wallSeconds": time.monotonic() - started,
        "sourceHashes": {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE, CAUSAL, Path(__file__), OUT / "flex.xml", OUT / "rigid-split-control.xml")},
        "contactCausalityGate": {"scope": "Recorded motor replay, .4 s; removal of wing/environment pairs prevents inversion while prior tilt remains.",
                                "baseline": causal_cases["baseline"]["metrics"],
                                "noWingEnvironment": causal_cases["no_wing_environment"]["metrics"],
                                "preImpactParity": causal["preImpactParity"]},
        "requiredNativeOverrides": {
            "reason": "Preserve original proximal whole-wing fluid model while suppressing unintended extra distal inertia-box fluid fallback.",
            "integrator": "Euler", "integratorId": 0,
            "source": "https://raw.githubusercontent.com/google-deepmind/mujoco/3.13.0/src/engine/engine_passive.c",
            "fluidGeometryOverrides": [{"geom": f"wing_{side}_distal_inertial", "array": "geom_fluid", "stride": 12,
                                         "component": 0, "expectedCompiledValue": 1., "value": DISTAL_FLUID_INTERACTION}
                                        for side in ("left", "right")],
            "warning": "XML alone is insufficient. Apply every listed override after compiling flex or rigid-split XML, before any forward/step. Positive-negligible coefficient is numerical suppression, not exact zero. Unsupported with implicit integration in MuJoCo3.13."
        },
        "paper": {"url": PAPER, "measurement": "Fig. 4C, load point 2 at end of third longitudinal vein in Drosophila",
                  "medianNPerM": .024, "rangeNPerM": [.020, .031], "linearForceRangeMicroNewton": [0, 3.3],
                  "wingsTested": 3, "spatialStiffnessVariationUpTo": 77,
                  "limitation": "Local static stiffness, not universal wing stiffness, a hinge location, or a damping measurement."},
        "assumptions": [
            "Single chord-axis hinge at existing inertial-box midspan, not a measured anatomical hinge.",
            "Box-inertial distal tip is a proxy for the paper's load point; exact vein-to-mesh registration is absent.",
            "Paper wings are about 2.15 mm and its example Drosophila mass about 1.6 mg; current FlyBody mass is about .985 mg. No unmeasured size scaling is applied.",
            "Damping ratio is an explicit numerical prior relative to distal inertia with proximal segment held; it is not measured Drosophila damping.",
            "A low-load static fit does not validate response at 235 Hz or extrapolation to the recorded millinewton-scale collision peaks.",
            "Inertial loading can flex this segment before contact. A later flexible-vs-rigid onset comparison must quantify pre-impact divergence and include a no-environment free-wing drive check.",
            "No active actuator, armature, friction loss, hard joint limit, or root constraint is added.",
            "Collision ellipsoids become clipped convex meshes; neutral collision response is not exactly identical. Compare against the emitted rigid split control before any impact claim.",
            "Original whole-wing fluid ellipsoid remains on proximal body unchanged. Aerodynamic deformation/distribution is NOT represented; this is not ready for flight integration.",
            "Required native overrides suppress additional distal inertia-box fluid fallback; loading XML without these overrides is an invalid aerodynamic control.",
            "Production renderer, metadata joint/body indices, wing force adapter, and rotation sensors are not modified; this XML is not a drop-in production replacement.",
        ],
        "probe": {"durationSecondsPerLoad": args.duration, "timestepSeconds": float(candidate.opt.timestep),
                  "scope": "Internal equal/opposite point-force pair; original free root unconstrained, no root pose writes after initialization. Measure distal displacement relative to proximal segment.",
                  "isolation": "Only force probe disables gravity, air, contact, actuation; original passive joints remain. Generated XML retains original environment options.",
                  "notDemonstrated": "Collision impact mitigation, pre-impact balance, closed-loop neural flight, feeding or task completion.", "rows": probes},
        "invariants": invariants, "wings": specs,
    }
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "wallSeconds": report["wallSeconds"], "invariants": invariants,
                      "stiffnessRangeNPerM": [min(p["effectiveStiffnessNPerM"] for p in probes), max(p["effectiveStiffnessNPerM"] for p in probes)]}, indent=2))


if __name__ == "__main__":
    main()
