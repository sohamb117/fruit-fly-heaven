// The released FlyBody flight task, preserved as an independent reference.
// Its learned controller and prescribed trajectory do NOT implement BANC.
import {FlyBodyPolicy,flattenFlyBodyObservation} from './flybody-policy.js';

const f32=Math.fround;
const nearest=(values,target,phase=false)=>{
 let best=0,distance=Infinity;
 for(let i=0;i<values.length;i++){
  const d=Math.abs((phase?values[i]%1:values[i])-target);
  if(d<distance){best=i;distance=d;}
 }
 return best;
};

export class ReferenceWingbeat{
 constructor(config){this.config=config;Object.assign(this,config.initial_state);}
 advance(command){
  const c=this.config;
  this.step=(this.step+1)%c.trajectories[this.frequency_index].traj.length;
  const requested=c.base_beat_freq*(1+c.rel_freq_range*command);
  this.frequency=c.ctrl_filter===0?requested:this.frequency*c.rate+requested*(1-c.rate);
  const index=nearest(c.beat_freqs,this.frequency);
  if(index!==this.frequency_index){
   const phase=c.trajectories[this.frequency_index].phase[this.step]%1;
   this.step=nearest(c.trajectories[index].phase,phase,true);
   this.frequency_index=index;
  }
  return c.trajectories[this.frequency_index].traj[this.step];
 }
}

export class FlyBodyFlightReference{
 constructor(mj,xml,metadata,policyModel){
  if(metadata.kind!=='upstream_trained_policy_flight_reference_not_BANC')throw new Error('Wrong flight reference model');
  Object.assign(this,{mj,metadata,policyModel});
  this.model=mj.MjModel.from_xml_string(xml);this.data=new mj.MjData(this.model);
  this.policy=new FlyBodyPolicy(policyModel);
  this.input=new Float32Array(this.policy.inputSize);
  this.realAction=new Float32Array(policyModel.output.size);
  this.wingJoints=metadata.action_mapping.wing_indices.map(i=>metadata.joints.find(j=>j.name==='walker/'+metadata.action_mapping.names[i]));
  this.rootBody=metadata.observations.world_zaxis_body_id;
  this.sensorFields=metadata.observations.sensors.filter(s=>['walker/accelerometer','walker/gyro','walker/velocimeter'].includes(s.name));
  this.reset();
 }
 reset(){
  const {mj,model,data,metadata:m}=this;
  mj.mj_resetData(model,data);
  for(const field of ['qpos','qvel','ctrl','act','qacc_warmstart'])data[field].set(m.initial_state[field]);
  mj.mj_forward(model,data);
  this.controlStep=0;this.wingbeat=new ReferenceWingbeat(m.wingbeat_generator);
  this.terminationReason=null;
  // Composer starts its four-sample buffers with three zero observations.
  this.observation=Object.fromEntries(Object.entries(m.initial_observation).map(([k,v])=>[k,Float64Array.from(v.flat())]));
  this.sensorMeans=Object.fromEntries(this.sensorFields.map(s=>[s.name,new Float64Array(s.dimension)]));
  this.lastCanonicalAction=new Float32Array(this.policyModel.output.size);
  return this;
 }
 get done(){return this.terminationReason!==null;}
 step(canonical,{actuated=true}={}){
  if(this.done)return false;
  const {mj,model,data,metadata:m}=this,a=this.realAction,map=m.action_mapping;
  if(!canonical)canonical=this.policy.predict(flattenFlyBodyObservation(this.policyModel,this.observation,this.input));
  this.lastCanonicalAction.set(canonical);
  // Match NumPy's in-place float32 CanonicalSpecWrapper transformation.
  for(let i=0;i<a.length;i++){
   a[i]=f32(.5*f32(Math.max(-1,Math.min(1,canonical[i]))+1));
   a[i]=f32(a[i]*(map.maximum[i]-map.minimum[i]));
   a[i]=f32(a[i]+map.minimum[i]);
  }
  const targets=this.wingbeat.advance(a[map.frequency_user_index]);
  for(let k=0;k<map.wing_indices.length;k++)a[map.wing_indices[k]]=f32(a[map.wing_indices[k]]+targets[k]-data.qpos[this.wingJoints[k].qpos]);
  for(const [kind,indices] of Object.entries(map.ctrl_indices))if(indices)for(let k=0;k<indices.length;k++)data.ctrl[indices[k]]=a[map.action_indices[kind][k]];
  if(!actuated)data.ctrl.fill(0);
  const ref=m.reference.root_qpos[this.controlStep],vel=m.reference.qvel[this.controlStep],ghost=m.ghost;
  data.qpos.set(ref,ghost.qpos);for(let k=0;k<3;k++)data.qpos[ghost.qpos+k]+=ghost.offset[k];
  data.qvel.set(vel,ghost.dof);
  for(const field of this.sensorFields)this.sensorMeans[field.name].fill(0);
  for(let sub=0;sub<m.physics_substeps_per_control;sub++){
   // dm_control defaults to legacy Euler stepping, not mj_step's order.
   mj.mj_step2(model,data);mj.mj_step1(model,data);
   // The upstream ghost-pose binding dirties physics; its first subsequent
   // sensor read invokes forward, including acceleration at this first state.
   if(sub===0)mj.mj_forward(model,data);
   for(const field of this.sensorFields)for(let k=0;k<field.dimension;k++)this.sensorMeans[field.name][k]+=data.sensordata[field.address+k]/m.physics_substeps_per_control;
  }
  this.controlStep++;
  this.readObservation();
  this.checkTermination();
  return true;
 }
 checkTermination(){
  const {data:d,metadata:m}=this,t=m.termination;
  if(!d.qpos.every(Number.isFinite)||!d.qvel.every(Number.isFinite))this.terminationReason='nonfinite state';
  else if(t&&d.xpos[this.rootBody*3+2]<t.minimum_height_cm)this.terminationReason='below minimum height';
  else if(t&&Math.hypot(...this.observation['walker/ref_displacement'].subarray(0,3))>t.maximum_reference_displacement_cm)this.terminationReason='reference distance exceeded';
  else if(t&&Math.hypot(...d.qacc)>t.maximum_qacc_norm)this.terminationReason='acceleration limit exceeded';
  else if(this.controlStep>=(m.episode?.maxControlSteps??m.reference.root_qpos.length-m.observations.future_steps-1))this.terminationReason='trajectory complete';
 }
 readObservation(){
  const {data:d,metadata:m,observation:o}=this,s=m.observations;
  for(const field of this.sensorFields)o[field.name].set(this.sensorMeans[field.name]);
  for(let k=0;k<s.qpos_indices.length;k++){
   o['walker/joints_pos'][k]=d.qpos[s.qpos_indices[k]];
   o['walker/joints_vel'][k]=d.qvel[s.qvel_indices[k]];
  }
  const r=d.xmat.subarray(this.rootBody*9,this.rootBody*9+9),pos=d.xpos.subarray(this.rootBody*3,this.rootBody*3+3),q=d.xquat.subarray(this.rootBody*4,this.rootBody*4+4);
  o['walker/world_zaxis'].set(r.subarray(6));
  const displacement=o['walker/ref_displacement'],orientation=o['walker/ref_root_quat'];
  const norm=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3],w=q[0]/norm,x=-q[1]/norm,y=-q[2]/norm,z=-q[3]/norm;
  for(let k=0;k<=s.future_steps;k++){
   const target=m.reference.root_qpos[this.controlStep+k],dx=target[0]-pos[0],dy=target[1]-pos[1],dz=target[2]-pos[2];
   for(let axis=0;axis<3;axis++)displacement[k*3+axis]=r[axis]*dx+r[3+axis]*dy+r[6+axis]*dz;
   const [a,b,c,e]=target.slice(3);
   orientation.set([w*a-x*b-y*c-z*e,w*b+x*a+y*e-z*c,w*c-x*e+y*a+z*b,w*e+x*c-y*b+z*a],k*4);
  }
  return o;
 }
 sample(){
  const {data:d,metadata:m}=this,ref=m.reference.root_qpos[this.controlStep],com=m.reference.center_of_mass_qpos[this.controlStep];
  const q=Array.from(d.xquat.subarray(this.rootBody*4,this.rootBody*4+4));
  return {time:d.time,position:Array.from(d.xpos.subarray(this.rootBody*3,this.rootBody*3+3)),quaternion:q,
   referenceError:Math.hypot(...Array.from(d.subtree_com.subarray(3,6),(v,i)=>v-com[i])),
   attitudeError:2*Math.acos(Math.min(1,Math.abs(q.reduce((sum,v,i)=>sum+v*ref[i+3],0))))*180/Math.PI,
   frequency:this.wingbeat.frequency,
   maximumAppliedForce:Math.max(...Array.from(d.xfrc_applied,Math.abs),...Array.from(d.qfrc_applied,Math.abs)),
   maximumRootActuatorForce:Math.max(...Array.from(d.qfrc_actuator.subarray(0,6),Math.abs)),
   finite:d.qpos.every(Number.isFinite)&&d.qvel.every(Number.isFinite),terminationReason:this.terminationReason};
 }
 dispose(){this.data.delete();this.model.delete();}
}
