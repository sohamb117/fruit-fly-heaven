#!/usr/bin/env python3
"""Record the tested common-body landing/walking teacher, without BANC.

Run using the pinned environment in scripts/flybody-baseline-lock.txt.
Produces isolated mesh-free models, complete native control/state sequences,
policy input/action projections, and small independent replay fixtures.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/flybody-landing-teacher"
REVISION="d015e9bfe441bd90ae431bac24c55cb74bdbce26"
spec=importlib.util.spec_from_file_location("landing",ROOT/"scripts/experiment-flybody-landing.py")
landing=importlib.util.module_from_spec(spec);spec.loader.exec_module(landing)

def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,(np.ndarray,np.generic)):return value.tolist()
    return value

def snapshot(data):
    return {"time":float(data.time),**{k:getattr(data,k).copy() for k in ["qpos","qvel","act","ctrl","qacc","qacc_warmstart","qfrc_applied","xfrc_applied"]}}

def write_json(path,value):
    path.write_text(json.dumps(plain(value),separators=(",",":"))+"\n")

def numbers(value):return " ".join(format(float(v),".17g") for v in value)

def mesh_free(env,path):
    full=env.physics.model.ptr
    source=env.task.root_entity.mjcf_model.to_xml_string()
    xml=ET.fromstring(source);removed=[]
    for body in xml.findall(".//worldbody//body"):
        bid=mujoco.mj_name2id(full,mujoco.mjtObj.mjOBJ_BODY,body.attrib["name"])
        for inertial in list(body.findall("inertial")):body.remove(inertial)
        if full.body_mass[bid]>0:
            ET.SubElement(body,"inertial",pos=numbers(full.body_ipos[bid]),quat=numbers(full.body_iquat[bid]),
                mass=format(float(full.body_mass[bid]),".17g"),diaginertia=numbers(full.body_inertia[bid]))
        for geom in list(body.findall("geom")):
            gid=mujoco.mj_name2id(full,mujoco.mjtObj.mjOBJ_GEOM,geom.attrib["name"])
            if full.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                assert full.geom_contype[gid]==0 and full.geom_conaffinity[gid]==0
                removed.append(geom.attrib["name"]);body.remove(geom)
    for asset in xml.findall("asset"):
        for mesh in list(asset.findall("mesh")):asset.remove(mesh)
    xml.find("compiler").set("inertiafromgeom","false")
    ET.indent(xml);text=ET.tostring(xml,encoding="unicode")+"\n";path.write_text(text)
    model=mujoco.MjModel.from_xml_string(text)
    fields=["body_mass","body_inertia","body_ipos","body_iquat","body_pos","body_quat","jnt_type","jnt_pos","jnt_axis","jnt_range","jnt_stiffness","dof_damping","dof_armature","actuator_dyntype","actuator_gainprm","actuator_biasprm","actuator_dynprm","actuator_ctrlrange","actuator_trntype","actuator_trnid","sensor_type","sensor_adr","sensor_dim"]
    errors={key:float(np.abs(getattr(model,key)-getattr(full,key)).max(initial=0)) for key in fields}
    assert max(errors.values())<1e-12
    return model,{"file":path.name,"sha256":hashlib.sha256(text.encode()).hexdigest(),
        "full_task_xml_sha256":hashlib.sha256(source.encode()).hexdigest(),"removed_visual_mesh_geoms":removed,
        "unchanged_parameter_max_errors":errors,"dimensions":{k:int(getattr(model,k)) for k in ["nq","nv","na","nu","nbody","ngeom","njnt","nsensor"]}}

class TeacherArchive:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.phases={};self.pending=None
        self.report={"schema":1,"kind":"published_policy_common_body_landing_walking_teacher_not_BANC",
            "command":"/tmp/flybody-baseline-venv/bin/python scripts/export-flybody-landing-teacher.py",
            "recreate_environment":"uv venv --python 3.11 /tmp/flybody-baseline-venv && uv pip sync --python /tmp/flybody-baseline-venv/bin/python scripts/flybody-baseline-lock.txt",
            "source":{"repository":"https://github.com/TuragaLab/flybody","revision":REVISION,"mujoco":mujoco.__version__,
                "policy_archive":"https://ndownloader.figshare.com/files/44815195","policy_archive_sha256":"2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582"},
            "source_files_sha256":{},"phases":{},
            "units":{"length":"cm","mass":"g","time":"s"},
            "limitations":["Explicit trajectory and contact-triggered controller switch; not autonomous food localization, not BANC, not a measured VNC.",
                "Native position servos and claw adhesion are retained; this teacher does not solve the motor-neuron-to-muscle adapter.",
                "Grounded walking-to-flight handoff did not stabilize in the separate takeoff probe.",
                "No production/default UI changes are made by this exporter."]}
        for file in ["scripts/experiment-flybody-contact-reuse.py","scripts/experiment-flybody-transition.py","scripts/experiment-flybody-landing.py","scripts/export-flybody-landing-teacher.py","data/raw/flybody-baseline/wing_pattern_fmech.npy"]:
            self.report["source_files_sha256"][file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        for name in ["flight","walking"]:
            directory=ROOT/f"data/raw/flybody-policy/{name}"
            self.report["source"][f"{name}_policy_files_sha256"]={str(f.relative_to(directory)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(directory.rglob("*")) if f.is_file()}

    def start_phase(self,name,env,adapter,targets):
        model,model_info=mesh_free(env,OUT/f"{name}.xml")
        original_step=env.physics.step
        phase={"frames":[],"model":model,"adapter":adapter}
        self.phases[name]=phase;self.phase_name=name
        def record_first_step(*args,**kwargs):
            if self.pending is not None and "before_first_substep" not in self.pending:
                self.pending["before_first_substep"]=snapshot(env.physics.data)
            return original_step(*args,**kwargs)
        env.physics.step=record_first_step
        keys=sorted(adapter.fields)
        initial_observation=adapter.project_observation(env._observation_updater.get_observation())
        phase["policy_keys"]=keys
        policy_shapes={key:list(initial_observation[key].shape) for key in keys}
        spec=env.action_spec();template_spec=adapter.template.action_spec()
        metadata={"model":model_info,"physics_timestep_s":float(model.opt.timestep),
            "control_timestep_s":env.control_timestep(),"substeps_per_control":env._n_sub_steps,
            "initial_state":snapshot(env.physics.data),"initial_policy_observation":initial_observation,
            "targets":targets,"actual_root_reference_qpos":env.task._ref_qpos.copy(),"actual_reference_qvel":env.task._ref_qvel.copy(),
            "policy_projection":{"keys_sorted":keys,"shapes":policy_shapes,"joint_observation_indices":adapter.joint_indices,
                "actuator_activation_indices_in_compiled_xml_order":adapter.activation_indices,
                "action_indices_by_exact_name":adapter.action_indices,"policy_action_names":template_spec.name.split("\t"),
                "full_action_names":spec.name.split("\t"),"policy_real_minimum":template_spec.minimum,"policy_real_maximum":template_spec.maximum,
                "mapping":"Policy observation fields projected by original joint and XML actuator names. Deterministic policy mean; canonical2real uses original float32 operation order. Full actuator controls below include native WBPG corrections."},
            "joints":[{"id":i,"name":model.joint(i).name,"qpos":int(model.jnt_qposadr[i]),"dof":int(model.jnt_dofadr[i])} for i in range(model.njnt)],
            "actuator_names":[model.actuator(i).name for i in range(model.nu)],
            "stepping":"Native legacy Euler: mj_step2, mj_step1 per 50 us substep; a first-substep mj_forward reproduces the dirty sensor binding refresh. Flight uses 4 substeps/control; walking 40. The noncolliding ghost reference pose is updated before each control. Fixtures contain the exact state AFTER before_step and BEFORE the first physics substep."}
        if name=="descent":
            w=env.task._wbpg
            metadata["wingbeat_initial_state"]={"frequency":float(w._ctrl_freq),"frequency_index":int(w._freq_idx),"step":int(w._step)}
        self.report["phases"][name]=metadata

    def before_control(self,env,adapter,action):
        phase=self.phases[self.phase_name]
        self.pending={"step":len(phase["frames"]),"policy_input":np.concatenate([adapter.last_observation[k].reshape(-1) for k in phase["policy_keys"]]),
            "canonical_action":adapter.last_canonical_action.copy(),"real_action_before_task":action.copy()}

    def after_control(self,env,timestep):
        assert "before_first_substep" in self.pending
        self.pending.update({"after_control":snapshot(env.physics.data),"reward":float(timestep.reward),"discount":float(timestep.discount)})
        self.phases[self.phase_name]["frames"].append(self.pending);self.pending=None

    def replay_frame(self,model,frame,substeps):
        data=mujoco.MjData(model);initial=frame["before_first_substep"]
        for key,value in initial.items():
            if key=="time":data.time=value
            else:getattr(data,key)[:]=value
        mujoco.mj_forward(model,data)
        for substep in range(substeps):
            mujoco.mj_step2(model,data);mujoco.mj_step1(model,data)
            if substep==0:mujoco.mj_forward(model,data)
        return {key:float(np.abs(getattr(data,key)-frame["after_control"][key]).max(initial=0)) for key in ["qpos","qvel","act"]}

    def finish(self,outcome):
        arrays={};fixtures={"kind":self.report["kind"],"phases":{}}
        for name,phase in self.phases.items():
            frames=phase["frames"];metadata=self.report["phases"][name]
            for field in ["policy_input","canonical_action","real_action_before_task"]:
                arrays[f"{name}/{field}"]=np.stack([frame[field] for frame in frames])
            for when in ["before_first_substep","after_control"]:
                for field in ["time","qpos","qvel","act","ctrl","qacc_warmstart"]:
                    arrays[f"{name}/{when}/{field}"]=np.asarray([frame[when][field] for frame in frames])
            indices=sorted(set(range(min(10,len(frames))))|set(range(max(0,len(frames)-10),len(frames))))
            selected=[];errors={key:0. for key in ["qpos","qvel","act"]}
            for i in indices:
                frame=frames[i];error=self.replay_frame(phase["model"],frame,metadata["substeps_per_control"])
                errors={key:max(errors[key],value) for key,value in error.items()}
                selected.append({**frame,"mesh_free_native_replay_error":error})
            fixtures["phases"][name]={"xml":metadata["model"]["file"],"xml_sha256":metadata["model"]["sha256"],"substeps_per_control":metadata["substeps_per_control"],"frames":selected}
            metadata["control_steps"]=len(frames)
            metadata["simulated_seconds"]=float(frames[-1]["after_control"]["time"])
            metadata["fixture_replay_max_errors"]=errors
            metadata["max_external_applied_force"]=max(float(np.abs(frame["after_control"][key]).max(initial=0)) for frame in frames for key in ["qfrc_applied","xfrc_applied"])
        np.savez_compressed(OUT/"native-teacher-sequences.npz",**arrays)
        self.report["complete_native_sequence_file"]="native-teacher-sequences.npz"
        self.report["complete_native_sequence_sha256"]=hashlib.sha256((OUT/"native-teacher-sequences.npz").read_bytes()).hexdigest()
        self.report["result"]=outcome["outcome"]
        self.report["transition_priors"]=outcome["priors"]
        write_json(OUT/"native-replay-fixtures.json",fixtures)
        write_json(OUT/"teacher.json",self.report)
        print(json.dumps({"teacher_directory":str(OUT.relative_to(ROOT)),"replay_errors":{name:phase["fixture_replay_max_errors"] for name,phase in self.report["phases"].items()}},indent=2))

def main():
    assert subprocess.check_output(["git","-C",str(ROOT/"references/flybody"),"rev-parse","HEAD"],text=True).strip()==REVISION
    landing.main(TeacherArchive())

if __name__=="__main__":main()
