#!/usr/bin/env python3
"""Compare the tested common controller body with the production BANC body."""
import hashlib
import importlib.util
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("reuse", ROOT/"scripts/experiment-flybody-contact-reuse.py")
reuse=importlib.util.module_from_spec(spec); spec.loader.exec_module(reuse)

def main():
    env=reuse.make_environment(False,articulated=True)
    reuse.configure_common(env,False)
    env.reset()
    common=env.physics.model.ptr
    production=mujoco.MjModel.from_xml_path(str(ROOT/"models/flybody-mujoco.xml"))
    report={"scope":"Read-only comparison against common physical body tested with both published controllers.",
        "production_xml_sha256":hashlib.sha256((ROOT/"models/flybody-mujoco.xml").read_bytes()).hexdigest(),
        "source_revision":"d015e9bfe441bd90ae431bac24c55cb74bdbce26",
        "production":{k:int(getattr(production,k)) for k in ["nq","nv","nu","na","njnt","nbody","ngeom","nsensor"]},
        "common":{k:int(getattr(common,k)) for k in ["nq","nv","nu","na","njnt","nbody","ngeom","nsensor"]},
        "joints":[],"bodies":[],"actuators":[],"additional_common_joints":[]}
    names={mujoco.mj_id2name(common,mujoco.mjtObj.mjOBJ_JOINT,i):i for i in range(common.njnt)}
    existing=set()
    for i in range(production.njnt):
        name=production.joint(i).name
        candidate="walker/"+name if name!="free" else "walker/"
        j=names.get(candidate)
        row={"name":name,"common_name":candidate,"found":j is not None}
        if j is not None:
            existing.add(j)
            row["kinematic_max_error"]=max(float(np.abs(getattr(production,f)[i]-getattr(common,f)[j]).max()) for f in ["jnt_type","jnt_axis","jnt_pos","jnt_range"])
            for key,field in [("stiffness","jnt_stiffness")]:
                a,b=float(getattr(production,field)[i]),float(getattr(common,field)[j])
                if a!=b:row[key]={"production":a,"common":b}
            pd,cd=production.jnt_dofadr[i],common.jnt_dofadr[j]
            for field in ["dof_damping","dof_armature"]:
                a,b=float(getattr(production,field)[pd]),float(getattr(common,field)[cd])
                if a!=b:row[field]={"production":a,"common":b}
        report["joints"].append(row)
    report["additional_common_joints"]=[common.joint(i).name for i in range(common.njnt) if i not in existing]
    for i in range(1,production.nbody):
        name=production.body(i).name
        j=mujoco.mj_name2id(common,mujoco.mjtObj.mjOBJ_BODY,"walker/"+name)
        row={"name":name,"found":j>=0}
        if j>=0:
            quaternion_error=min(float(np.abs(production.body_quat[i]-common.body_quat[j]).max()),float(np.abs(production.body_quat[i]+common.body_quat[j]).max()))
            row["local_frame_max_error"]=max(float(np.abs(production.body_pos[i]-common.body_pos[j]).max()),quaternion_error)
            if row["local_frame_max_error"]>1e-12:
                row["production_quaternion_wxyz"]=production.body_quat[i].tolist()
                row["common_quaternion_wxyz"]=common.body_quat[j].tolist()
                row["orientation_difference_degrees"]=float(2*np.degrees(np.arccos(np.clip(abs(production.body_quat[i]@common.body_quat[j]),0,1))))
            row["mass_error_g"]=float(production.body_mass[i]-common.body_mass[j])
            row["inertia_max_error"]=float(np.abs(production.body_inertia[i]-common.body_inertia[j]).max())
        report["bodies"].append(row)
    for i in range(production.nu):
        name=production.actuator(i).name
        j=mujoco.mj_name2id(common,mujoco.mjtObj.mjOBJ_ACTUATOR,"walker/"+name)
        row={"name":name,"found":j>=0}
        if j>=0:
            row["differences"]={}
            for f in ["actuator_trntype","actuator_dyntype","actuator_gaintype","actuator_biastype","actuator_gainprm","actuator_biasprm","actuator_dynprm","actuator_ctrlrange","actuator_forcerange","actuator_gear"]:
                a,b=getattr(production,f)[i],getattr(common,f)[j]
                if np.any(a!=b):row["differences"][f]={"production":np.asarray(a).tolist(),"common":np.asarray(b).tolist()}
        report["actuators"].append(row)
    report["summary"]={
        "matched_production_joints":sum(r["found"] for r in report["joints"]),
        "max_joint_kinematic_error":max(r.get("kinematic_max_error",0) for r in report["joints"]),
        "matched_production_bodies":sum(r["found"] for r in report["bodies"]),
        "max_body_local_frame_error":max(r.get("local_frame_max_error",0) for r in report["bodies"]),
        "sum_common_mass_g":float(common.body_mass.sum()),"sum_production_mass_g":float(production.body_mass.sum()),
        "common_real_body_mass_g":float(sum(common.body_mass[mujoco.mj_name2id(common,mujoco.mjtObj.mjOBJ_BODY,"walker/"+production.body(i).name)] for i in range(1,production.nbody))),
        "mass_note":"sum_common_mass_g includes the noncolliding visual reference ghost; common_real_body_mass_g excludes it.",
        "additional_common_joint_count":len(report["additional_common_joints"]),
        "matched_production_actuators":sum(r["found"] for r in report["actuators"]),
        "matched_actuators_with_different_parameters":sum(bool(r.get("differences")) for r in report["actuators"])}
    report["boundary"]=["Name and joint-frame compatibility do not imply policy compatibility: retain exact sensor definitions, ordering, action normalization, and control timestep.",
        "The common body uses upstream position servos and filtered claw adhesion. The current production actuator bank adds mouth actuation and uses unfiltered claw adhesion; the controlling inputs still differ substantially.",
        "The common body retains 52 additional biological joints plus one noncolliding ghost free joint. Freezing those biological joints was not included in the passing common-model policy tests.",
        "No published controller is inserted into the BANC path by this inspection."]
    (ROOT/"reports/flybody-common-model-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"summary":report["summary"],"dimensions":{"production":report["production"],"common":report["common"]},"missing_actuators":[r["name"] for r in report["actuators"] if not r["found"]],"missing_bodies":[r["name"] for r in report["bodies"] if not r["found"]]},indent=2))

if __name__=="__main__":main()
