// Prescribed bounded steering commands for the published FlyBody controller.
// These change only its reference trajectory, never the fly state or forces.
const TAU=2*Math.PI;
const quatMultiply=(a,b)=>[
  a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
  a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
  a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
  a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
];
const rotate=(q,v)=>{
  const r=quatMultiply(quatMultiply(q,[0,...v]),[q[0],-q[1],-q[2],-q[3]]);
  return r.slice(1);
};
const arrayLike=(length,at)=>new Proxy({length},{get(target,key){
  if(typeof key==='string'&&/^\d+$/.test(key))return at(Number(key));
  return target[key];
}});

/** Returns new reference metadata sharing the immutable physical assets.
 * `brake-hover` smoothly decelerates the original flight speed to hover.
 * `orbit` follows a circle with tangent heading and an optional physical bank.
 * Both are explicit high-level tasks supplied to the learned joint policy.
 */
export function createBoundedFlightTrajectory(metadata,{
  kind='brake-hover',durationSeconds=10,brakingSeconds=.3,radiusCm=2.5,
  bankScale=1,
}={}){
  if(!['brake-hover','orbit'].includes(kind)||!(durationSeconds>0)||!(brakingSeconds>0)||!(radiusCm>0))throw new Error('Invalid bounded flight trajectory');
  const dt=metadata.control_timestep,start=metadata.reference.center_of_mass_qpos[0],root0=metadata.reference.root_qpos[0];
  const q0=start.slice(3),inverse=[q0[0],-q0[1],-q0[2],-q0[3]];
  const rootOffset=rotate(inverse,root0.slice(0,3).map((v,i)=>v-start[i]));
  const initialVelocity=metadata.initial_state.qvel.slice(0,3),speed=Math.hypot(initialVelocity[0],initialVelocity[1]);
  const heading=Math.atan2(initialVelocity[1],initialVelocity[0]);
  const maxControlSteps=Math.round(durationSeconds/dt),length=maxControlSteps+metadata.observations.future_steps+2;
  let cacheStart=-1;const cache=new Map();
  function row(index){
    if(!Number.isInteger(index)||index<0||index>=length)throw new RangeError('Flight reference index outside trajectory');
    if(cache.has(index))return cache.get(index);
    const t=index*dt;let position,velocity,quaternion,angular;
    if(kind==='brake-hover'){
      const u=Math.min(1,t/brakingSeconds),distance=speed*brakingSeconds*(u-u**3+.5*u**4);
      const v=speed*(1-3*u*u+2*u*u*u);
      position=[start[0]+Math.cos(heading)*distance,start[1]+Math.sin(heading)*distance,start[2]];
      velocity=[Math.cos(heading)*v,Math.sin(heading)*v,0];quaternion=q0;angular=[0,0,0];
    }else{
      const omega=speed/radiusCm,theta=omega*t;
      const a=radiusCm*Math.sin(theta),b=radiusCm*(1-Math.cos(theta));
      position=[start[0]+a*Math.cos(heading)-b*Math.sin(heading),start[1]+a*Math.sin(heading)+b*Math.cos(heading),start[2]];
      velocity=[speed*Math.cos(heading+theta),speed*Math.sin(heading+theta),0];
      const yaw=[Math.cos(theta/2),0,0,Math.sin(theta/2)];
      const bank=-bankScale*Math.atan(speed*omega/981)*(1-Math.exp(-t/.1));
      quaternion=quatMultiply(yaw,quatMultiply([Math.cos(bank/2),Math.sin(bank/2),0,0],q0));
      angular=[0,0,omega];
    }
    const shifted=rotate(quaternion,rootOffset);
    const value={com:[...position,...quaternion],root:[...position.map((v,i)=>v+shifted[i]),...quaternion],velocity:[...velocity,...angular]};
    if(cacheStart<0||index-cacheStart>64||index<cacheStart){cache.clear();cacheStart=index;}
    cache.set(index,value);return value;
  }
  const reference={
    center_of_mass_qpos:arrayLike(length,i=>row(i).com),
    root_qpos:arrayLike(length,i=>row(i).root),
    qvel:arrayLike(length,i=>row(i).velocity),
  };
  const initialObservation={...metadata.initial_observation},displacement=[],orientation=[];
  for(let i=0;i<=metadata.observations.future_steps;i++){
    const target=reference.root_qpos[i];
    displacement.push(rotate(inverse,target.slice(0,3).map((v,k)=>v-root0[k])));
    orientation.push(quatMultiply(inverse,target.slice(3)));
  }
  initialObservation['walker/ref_displacement']=displacement;
  initialObservation['walker/ref_root_quat']=orientation;
  return {...metadata,initial_observation:initialObservation,reference,
    episode:{...metadata.episode,time_limit_s:durationSeconds,maxControlSteps,trajectory_timesteps:maxControlSteps},
    prescribedTrajectory:{kind,durationSeconds,brakingSeconds,radiusCm,bankScale,initialSpeedCmPerSecond:speed,
      nominalDistanceCm:kind==='brake-hover'?speed*brakingSeconds/2:2*radiusCm,
      periodSeconds:kind==='orbit'?TAU*radiusCm/speed:null,
      note:'Explicit desired motion for the published learned policy; no body-position overwrite, root force, takeoff, landing, or BANC control.'},
  };
}
