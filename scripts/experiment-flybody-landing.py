#!/usr/bin/env python3
"""Contact-triggered handoff between published controllers on one physical state.

Explicit experiment: a reference descent, leg deployment, then the walking
policy at first real ground contact. No BANC/autonomous-task claim.
"""
import importlib.util
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import tensorflow as tf
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("transition",ROOT/"scripts/experiment-flybody-transition.py")
transition=importlib.util.module_from_spec(spec);spec.loader.exec_module(transition)
reuse=transition.reuse

def ground_contacts(env):
    active=0;force=np.zeros(6)
    ground={env.physics.model.name2id(g.full_identifier,"geom") for g in env.task._arena.ground_geoms}
    for cid,c in enumerate(env.physics.data.contact):
        if c.geom1 in ground or c.geom2 in ground:
            mujoco.mj_contactForce(env.physics.model.ptr,env.physics.data.ptr,cid,force)
            active+=force[0]>1e-10
    return int(active)

def sample(env,phase,elapsed):
    q=env.physics.data.qpos[3:7]
    return {"phase":phase,"time":elapsed+float(env.physics.data.time),"position_cm":env.physics.data.qpos[:3].tolist(),
            "tilt_degrees":float(np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))),"ground_contacts":ground_contacts(env),
            "external_force":max(float(np.abs(env.physics.data.xfrc_applied).max()),float(np.abs(env.physics.data.qfrc_applied).max()))}

def main(teacher=None):
    tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(1)
    fly=reuse.make_environment(False,articulated=True);walk=reuse.make_environment(True,wings=True)
    reuse.configure_common(fly,False);reuse.configure_common(walk,True)
    walk.reset();standing_length={walk.physics.model.id2name(i,"actuator"):float(walk.physics.data.actuator_length[i]) for i in range(walk.physics.model.nu)}
    fly._time_limit=fly.task._time_limit=1.2
    n=6020;t=np.arange(n)*.0002;s=np.clip(t/1.1,0,1);smooth=s*s*(3-2*s)
    q,v=constant_speed_trajectory(n,speed=20,init_pos=(0,0,1),body_rot_angle_y=-47.5,control_timestep=.0002)
    speed=20-18*smooth;q[:,0]=np.cumsum(speed)*.0002;q[:,0]-=q[0,0];q[:,2]=1-.88*smooth
    v[:,:3]=np.gradient(q[:,:3],.0002,axis=0)
    fly.task._traj_generator.set_next_trajectory(q,v);ft=fly.reset()
    fp=transition.PolicyAdapter(fly,reuse.make_environment(False),reuse.setup_policy("flight"))
    wp=transition.PolicyAdapter(walk,reuse.make_environment(True),reuse.setup_policy("walking"))
    folded=fp.defaults.copy();deployed=folded.copy();names=fly.action_spec().name.split("\t")
    original_flight_names=fp.template.action_spec().name.split("\t")
    for i,name in enumerate(names):
        if name not in original_flight_names and "adhere" not in name:
            if "walker/"+name in standing_length:deployed[i]=standing_length["walker/"+name]
    if teacher:
        teacher.start_phase("descent",fly,fp,{"com_qpos":q,"qvel":v,"folded_leg_action":folded,"standing_leg_action":deployed})
    trace=[];first_contact=None;maxforce=0.;steps=0;start=time.perf_counter()
    while not ft.last():
        # Declared deployment prior: extend legs while approaching the floor.
        blend=np.clip((.5-fly.physics.data.qpos[2])/.15,0,1)
        fp.defaults[:]=(1-blend)*folded+blend*deployed
        action=fp.act(ft.observation)
        if teacher:teacher.before_control(fly,fp,action)
        ft=fly.step(action);steps+=1
        if teacher:teacher.after_control(fly,ft)
        if steps%50==0:trace.append(sample(fly,"descent",0))
        if ground_contacts(fly)>0:
            first_contact=float(fly.physics.data.time);trace.append(sample(fly,"first_contact",0));break
    outcome={"flight_seconds":float(fly.physics.data.time),"first_contact_s":first_contact,"flight_discount":float(ft.discount)}
    if first_contact is not None:
        state={key:getattr(fly.physics.data,key).copy() for key in ["qpos","qvel","act","ctrl","qacc_warmstart"]}
        walk._time_limit=walk.task._time_limit=.6;walk.task._max_episode_steps=301
        q,v=constant_speed_trajectory(365,speed=2,init_pos=(0,0,.1278),control_timestep=.002)
        walk.task._traj_generator.set_next_trajectory(q,v);walk.reset()
        walk.task._ref_qpos[:,:2]+=state["qpos"][:2]
        for key,value in state.items():getattr(walk.physics.data,key)[:]=value
        walk.physics.data.time=0.;walk.physics.forward()
        continuity=max(float(np.abs(getattr(walk.physics.data,key)-value).max(initial=0)) for key,value in state.items() if key in ["qpos","qvel","act"])
        assert continuity==0
        walk._observation_updater.reset(walk.physics,walk.random_state);obs=walk._observation_updater.get_observation()
        if teacher:
            teacher.start_phase("walking_handoff",walk,wp,{"com_qpos":q,"qvel":v,"incoming_physical_state":state,"continuity_error":continuity})
        count=0;contact_steps=0
        while True:
            action=wp.act(obs)
            if teacher:teacher.before_control(walk,wp,action)
            wt=walk.step(action);obs=wt.observation;count+=1;contact_steps+=ground_contacts(walk)>0
            if teacher:teacher.after_control(walk,wt)
            if count%5==0 or wt.last():trace.append(sample(walk,"walking_handoff",first_contact))
            if wt.last():break
        final=sample(walk,"final",first_contact)
        outcome.update({"continuity_error":continuity,"walking_seconds":float(walk.physics.data.time),"walking_discount":float(wt.discount),"reached_end":bool(walk.task._reached_traj_end),"contact_fraction":contact_steps/count,"final":final})
    report={"scope":"Explicit reference descent and contact-triggered controller handoff on common physical body; not BANC or autonomous food localization.","outcome":outcome,"wall_seconds":time.perf_counter()-start,"trace":trace,
            "priors":["Reference speed 20 to 2 cm/s, altitude 1 to .12 cm, pitch 47.5 degrees.","Leg targets blend from retracted to original walking initial lengths between heights .5 and .35 cm.","Switch to the trained walking controller on first nonzero native ground contact force; wing force commands are zero afterward.","All physical qpos, qvel, and activation states are copied exactly between task wrappers with identical dynamics; no root forces or corrections."]}
    (ROOT/"reports/flybody-landing-probe.json").write_text(json.dumps(report,indent=2)+"\n")
    if teacher:teacher.finish(report)
    print(json.dumps({k:v for k,v in report.items() if k!="trace"},indent=2))

if __name__=="__main__":main()
