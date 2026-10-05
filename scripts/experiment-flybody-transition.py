#!/usr/bin/env python3
"""A declared reference-conditioned walking-to-flight transition experiment.

The same physical parameters support both published policies. Task wrappers are
exchanged with continuous qpos/qvel/activation state. No root force, velocity
injection, or pose correction is used at the transition. This is not BANC.
"""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import tensorflow as tf
import flybody.tasks.flight_imitation as flight_task
from flybody.tasks.synthetic_trajectories import constant_speed_trajectory
from flybody.tasks.task_utils import canonical2real

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contact_reuse", ROOT / "scripts/experiment-flybody-contact-reuse.py")
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


class PolicyAdapter:
    def __init__(self, env, template, policy):
        self.env, self.template, self.policy = env, template, policy
        original = template.reset().observation
        self.fields = list(original)
        self.activation_shape = original["walker/actuator_activation"].shape
        actual_joints = [j.name for j in env.task.walker.observable_joints]
        self.joint_indices = [actual_joints.index(j.name) for j in template.task.walker.observable_joints]
        names = env.action_spec().name.split("\t")
        oldnames = template.action_spec().name.split("\t")
        self.action_indices = [names.index(name) for name in oldnames]
        self.activation_indices = []
        for aid in range(template.physics.model.nu):
            if template.physics.model.actuator_actadr[aid] >= 0:
                other = env.physics.model.name2id(template.physics.model.id2name(aid, "actuator"), "actuator")
                self.activation_indices.append(env.physics.model.actuator_actadr[other])
        self.defaults = np.zeros(env.action_spec().shape, np.float32)
        if len(oldnames) == 12:
            for name in names:
                if name not in oldnames and "adhere" not in name:
                    aid = env.physics.model.name2id("walker/"+name, "actuator")
                    if aid >= 0:
                        self.defaults[names.index(name)] = env.physics.data.actuator_length[aid]

    def project_observation(self, observation):
        obs = {key:np.asarray(observation[key], np.float32) for key in self.fields}
        for key in ["walker/joints_pos", "walker/joints_vel"]:
            obs[key] = obs[key][self.joint_indices]
        obs["walker/actuator_activation"] = np.asarray(self.env.physics.data.act[self.activation_indices],np.float32) if self.activation_indices else np.empty(0,np.float32)
        assert obs["walker/actuator_activation"].shape == self.activation_shape
        return obs

    def act(self, observation):
        obs = self.project_observation(observation)
        canonical = self.policy({key:tf.convert_to_tensor(value[None]) for key,value in obs.items()}).mean()[0].numpy()
        self.last_observation = obs
        self.last_canonical_action = canonical.copy()
        action = self.defaults.copy()
        action[self.action_indices] = canonical2real(canonical.copy(),self.template.action_spec())
        return action


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    walk = reuse.make_environment(True,wings=True)
    fly = reuse.make_environment(False,articulated=True)
    reuse.configure_common(walk,True)
    reuse.configure_common(fly,False)
    walk._time_limit = walk.task._time_limit = .4
    walk.task._max_episode_steps = 201
    q,v = constant_speed_trajectory(265,speed=2,init_pos=(0,0,.1278),control_timestep=.002)
    walk.task._traj_generator.set_next_trajectory(q,v)
    wt = walk.reset()
    fly._time_limit = fly.task._time_limit = 1.2
    q,v = constant_speed_trajectory(6020,speed=20,init_pos=(0,0,1),body_rot_angle_y=-47.5,control_timestep=.0002)
    fly.task._traj_generator.set_next_trajectory(q,v)
    fly.reset()
    # Compare every dynamic scalar array relevant to this shared body. Visual
    # trajectory sites/colors, observation buffers, and task timers may differ.
    fields = ["body_mass","body_inertia","body_ipos","body_iquat","body_pos","body_quat","jnt_type","jnt_pos","jnt_axis","jnt_stiffness","jnt_range","dof_damping","dof_armature","geom_type","geom_size","geom_pos","geom_quat","geom_contype","geom_conaffinity","geom_friction","geom_solref","geom_solimp","geom_fluid","actuator_gainprm","actuator_biasprm","actuator_dynprm","actuator_dyntype","actuator_ctrlrange","actuator_trntype","actuator_trnid"]
    differences = {key:float(np.abs(getattr(walk.physics.model,key)-getattr(fly.physics.model,key)).max(initial=0)) for key in fields}
    assert max(differences.values()) < 1e-12, differences
    prod = mujoco.MjModel.from_xml_path(str(ROOT/"models/flybody-mujoco.xml"))
    comparison = {"production":{k:int(getattr(prod,k)) for k in ["nq","nv","nu","na","njnt"]}, "common":{k:int(getattr(fly.physics.model,k)) for k in ["nq","nv","nu","na","njnt"]}, "joint_differences":[]}
    for jid in range(prod.njnt):
        name=prod.joint(jid).name
        other=mujoco.mj_name2id(fly.physics.model.ptr,mujoco.mjtObj.mjOBJ_JOINT,"walker/"+name)
        if other < 0:
            comparison["joint_differences"].append({"name":name,"in_common":False})
        else:
            comparison["joint_differences"].append({"name":name,"in_common":True,"axis_error":float(np.abs(prod.jnt_axis[jid]-fly.physics.model.jnt_axis[other]).max()),"position_error":float(np.abs(prod.jnt_pos[jid]-fly.physics.model.jnt_pos[other]).max()),"range_error":float(np.abs(prod.jnt_range[jid]-fly.physics.model.jnt_range[other]).max())})
    wp = PolicyAdapter(walk,reuse.make_environment(True),reuse.setup_policy("walking"))
    fp = PolicyAdapter(fly,reuse.make_environment(False),reuse.setup_policy("flight"))
    trace=[]
    while not wt.last():
        wt=walk.step(wp.act(wt.observation))
        if round(walk.physics.data.time/.002)%10==0:
            trace.append({"phase":"walking","time":float(walk.physics.data.time),"position_cm":walk.physics.data.qpos[:3].tolist()})
    before={key:getattr(walk.physics.data,key).copy() for key in ["qpos","qvel","act","ctrl","qacc_warmstart"]}
    position=before["qpos"][:3].copy()
    quaternion=before["qpos"][3:7].copy()
    com=walk.physics.named.data.subtree_com["walker/"].copy()
    n=6020; t=np.arange(n)*.0002
    smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
    speed=2+18*smooth(t/.3)
    ref=np.zeros((n,7));ref[:,0]=np.cumsum(speed)*.0002;ref[:,0]-=ref[0,0]
    ref[:,2]=com[2]+(1-com[2])*smooth(t/.4)
    target=np.array([np.cos(np.deg2rad(47.5)/2),0,-np.sin(np.deg2rad(47.5)/2),0])
    blend=smooth(t/.2)[:,None];ref[:,3:]=(1-blend)*quaternion+blend*target
    ref[:,3:]/=np.linalg.norm(ref[:,3:],axis=1)[:,None]
    vel=np.zeros((n,6));vel[:,:3]=np.gradient(ref[:,:3],.0002,axis=0)
    fly.task._traj_generator.set_next_trajectory(ref,vel)
    fly.reset()
    fly.task._ref_qpos[:,:2]+=com[:2]
    for key,value in before.items():
        getattr(fly.physics.data,key)[:]=value
    fly.physics.data.time=0.
    fly.physics.forward()
    continuity=max(float(np.abs(getattr(fly.physics.data,key)-value).max(initial=0)) for key,value in before.items() if key in ["qpos","qvel","act"])
    assert continuity == 0
    # Original flight task ends below0.2cm, which excludes a grounded start.
    # Lower only this experiment's failure cutoff; do not alter dynamics.
    flight_task._TERMINAL_HEIGHT=.02
    fly._observation_updater.reset(fly.physics,fly.random_state)
    obs=fly._observation_updater.get_observation()
    step=0;maxheight=float(position[2]);maxforce=0.;maxerror=0.;lift_samples=0;start=time.perf_counter()
    while True:
        ft=fly.step(fp.act(obs));obs=ft.observation;step+=1
        height=float(fly.physics.data.qpos[2]);maxheight=max(maxheight,height)
        error=float(np.linalg.norm(obs["walker/ref_displacement"][0]));maxerror=max(maxerror,error)
        maxforce=max(maxforce,float(np.abs(fly.physics.data.xfrc_applied).max()),float(np.abs(fly.physics.data.qfrc_applied).max()))
        if height>.3:lift_samples+=1
        if step%50==0 or ft.last():
            trace.append({"phase":"flight_attempt","time":.4+float(fly.physics.data.time),"position_cm":fly.physics.data.qpos[:3].tolist(),"quaternion":fly.physics.data.qpos[3:7].tolist(),"reference_error_cm":error})
        if ft.last() or step>=5994:
            break
    result={"scope":"Explicit reference-conditioned native walking-to-flight transition; original learned policies, not BANC.","shared_physical_parameter_max_errors":differences,"production_body_comparison":comparison,
        "transition_state_continuity_error":continuity,"walking_seconds":.4,"flight_attempt_seconds":float(fly.physics.data.time),"flight_wall_seconds":time.perf_counter()-start,"discount":float(ft.discount),"reached_trajectory_end":bool(fly.task._reached_traj_end),"maximum_height_cm":maxheight,"samples_above_3mm":lift_samples,"maximum_reference_error_cm":maxerror,"maximum_external_applied_force":maxforce,"trace":trace,
        "priors":["At0.4s switch from releasedwalkingpolicy to releasedflightpolicy; leg targets change to fixedretractedposition andadhesion tozero.","Futureflightreference explicitly ramps altitude to1cm,speed2to20cm/s,andpitchto47.5degrees.","Flightfailureheightcutoff loweredfrom0.2cmto0.02cm solely toallowgroundedstart.","Allqpos,qvel,activationstate arecontinuous; taskwrapper clock resets butphysicalrootisnotmoved."]}
    (ROOT/"reports/flybody-transition-probe.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key not in ["trace","production_body_comparison","shared_physical_parameter_max_errors"]},indent=2))


if __name__=="__main__":main()
