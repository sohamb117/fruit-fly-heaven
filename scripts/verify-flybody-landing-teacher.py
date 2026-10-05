#!/usr/bin/env python3
"""Continuous native replay of recorded teacher controls on mesh-free models."""
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/"reports/flybody-landing-teacher"

def main():
    metadata=json.loads((DIRECTORY/"teacher.json").read_text())
    sequences=np.load(DIRECTORY/metadata["complete_native_sequence_file"])
    assert hashlib.sha256((DIRECTORY/metadata["complete_native_sequence_file"]).read_bytes()).hexdigest()==metadata["complete_native_sequence_sha256"]
    report={"scope":"Recorded controls, continuous native dynamics on mesh-free models; no policy inference or physical-state corrections within either phase.","mujoco":mujoco.__version__,"phases":{}}
    for name,phase in metadata["phases"].items():
        xml=DIRECTORY/phase["model"]["file"]
        assert hashlib.sha256(xml.read_bytes()).hexdigest()==phase["model"]["sha256"]
        model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
        ghost=model.joint("ghost/");gq=int(ghost.qposadr[0]);gv=int(ghost.dofadr[0])
        before={k:sequences[f"{name}/before_first_substep/{k}"] for k in ["time","qpos","qvel","act","ctrl","qacc_warmstart"]}
        after={k:sequences[f"{name}/after_control/{k}"] for k in ["qpos","qvel","act"]}
        for key,values in before.items():
            if key=="time":data.time=float(values[0])
            else:getattr(data,key)[:]=values[0]
        mujoco.mj_forward(model,data)
        errors={"qpos":0.,"qvel":0.,"act":0.,"real_body_qpos":0.,"real_body_qvel":0.}
        poses=[];start=time.perf_counter()
        for step in range(len(before["ctrl"])):
            # This is the task's noncolliding trajectory marker, never the fly.
            data.qpos[gq:gq+7]=before["qpos"][step,gq:gq+7]
            data.qvel[gv:gv+6]=before["qvel"][step,gv:gv+6]
            data.ctrl[:]=before["ctrl"][step]
            for substep in range(phase["substeps_per_control"]):
                mujoco.mj_step2(model,data);mujoco.mj_step1(model,data)
                if substep==0:mujoco.mj_forward(model,data)
            for key in ["qpos","qvel","act"]:
                difference=np.abs(getattr(data,key)-after[key][step])
                errors[key]=max(errors[key],float(difference.max(initial=0)))
                if key!="act":errors["real_body_"+key]=max(errors["real_body_"+key],float(difference[:gq if key=="qpos" else gv].max(initial=0)))
            if step%max(1,len(before["ctrl"])//20)==0 or step==len(before["ctrl"])-1:
                poses.append({"step":step,"time":float(data.time),"root_qpos":data.qpos[:7].tolist()})
        report["phases"][name]={"xml_sha256":phase["model"]["sha256"],"control_steps":len(before["ctrl"]),"simulated_seconds":float(data.time),"wall_seconds":time.perf_counter()-start,"maximum_absolute_errors":errors,"poses":poses}
    report["passed"]=all(p["maximum_absolute_errors"]["real_body_qpos"]<1e-6 and p["maximum_absolute_errors"]["real_body_qvel"]<1e-3 for p in report["phases"].values())
    (DIRECTORY/"continuous-native-replay.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"phases":{name:{k:v for k,v in p.items() if k!="poses"} for name,p in report["phases"].items()}},indent=2))

if __name__=="__main__":main()
