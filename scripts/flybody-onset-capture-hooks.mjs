// Test-only observer hooks, shared by the browser capture and Node fixture.
// This module has no imports, DOM requirements or physical control writes.
const copy=x=>x===undefined?null:structuredClone(x);
const array=x=>x?Array.from(x):[];

export function onsetState(body){
 const d=body.data,muscles=body.muscles,w=body.wings;
 return {native:{time:d.time,...Object.fromEntries(['qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied'].map(k=>[k,array(d[k])]))},
  muscleState:array(muscles.core.HEAPF32.slice(muscles.state/4,muscles.state/4+muscles.count*3)),
  muscleInput:array(body.input),activation:array(body.activation),restPose:array(body.restPose),restHeight:body.restHeight,
  remainder:body.remainder,internal:copy(body.internal),food:copy(body.food),
  wing:{phase:w.phase,deployment:array(w.deployment),power:array(w.power),opening:array(w.opening),
   target:array(w.target),residuals:w.residuals.map(array)},
  diagnostics:{wingDriveLeft:body.wingDriveLeft??0,wingDriveRight:body.wingDriveRight??0,wingPower:body.wingPower,
   airborne:body.airborne,onFood:body.onFood,mouthContact:body.mouthContact,contacts:body.contactCount,
   environmentContacts:body.environmentContactCount,legLoads:array(body.legLoads),
   quaternion:array(body.quaternion),proboscis:body.proboscis,pump:body.pump,
   actuatorForce:array(d.actuator_force),actuatorJointTorque:array(d.qfrc_actuator),passiveJointTorque:array(d.qfrc_passive),
   wingDriveVsEffectivePower:'Drive is mean DLM/DVM muscle force; effective power is after opening/deployment.'}};
}

export function onsetScene(world,scene,metadata,io){
 const capture=globalThis.__flybodyOnset;
 if(!capture?.enabled)return;
 if(capture.scene)throw new Error('Onset recorder supports exactly one created physical world');
 capture.scene={xml:scene.xml,heights:array(scene.heights),fruitGeomNames:copy(scene.fruitGeomNames),metadata:copy(metadata),io:copy(io),fruit:copy(world.habitat.fruit),
  ceiling:world.habitat.ceiling,initialWorldOptions:{flightEnabled:world.flightEnabled,motorCoupling:world.motorCoupling,movementMode:world.movementMode}};
}

export function installOnsetCapture(Physics){
 if(Physics.prototype.__onsetInstalled)return;
 const step=Physics.prototype.step;
 Object.defineProperty(Physics.prototype,'__onsetInstalled',{value:true});
 Physics.prototype.step=function(rates,duration,options={}){
  const capture=globalThis.__flybodyOnset;
  if(!capture?.enabled||capture.done)return step.call(this,rates,duration,options);
  if(capture.body&&capture.body!==this)throw new Error('Onset capture expected one body');
  capture.body=this;
  if(Math.abs(duration-.002)>1e-10)throw new Error(`Expected actual 2 ms body step, received ${duration}`);
  const row={index:capture.frames.length,duration,options:copy(options),rates:Array.from(rates.entries()),before:onsetState(this),muscleSteps:[],wingSteps:[]};
  if(!capture.frames.length){
   if(this.data.time>this.metadata.timestep/2)throw new Error('Recorder missed the first physical step');
   capture.initial=row.before;
   capture.startedWallMs=globalThis.performance?.now?.()??0;
  }
  const muscleStep=this.muscles.step,wingStep=this.wings.step;
  this.muscles.step=(input,dt)=>{
   const state=muscleStep.call(this.muscles,input,dt);
   row.muscleSteps.push({time:this.data.time,dt,input:array(input),state:array(state)});return state;
  };
  this.wings.step=(q,ctrl,left,right,steering,dt)=>{
   const result=wingStep.call(this.wings,q,ctrl,left,right,steering,dt);
   row.wingSteps.push({time:this.data.time,dt,left,right,steering:copy(steering),ctrl:array(ctrl),target:array(this.wings.target),
    deployment:array(this.wings.deployment),opening:array(this.wings.opening),effectivePower:array(this.wings.power)});
   return result;
  };
  let result;
  try{result=step.call(this,rates,duration,options);}
  finally{this.muscles.step=muscleStep;this.wings.step=wingStep;}
  row.after=onsetState(this);capture.frames.push(row);
  if(this.data.time>=capture.seconds-1e-10){
   capture.done=true;capture.finishedWallMs=globalThis.performance?.now?.()??0;
   // A normal UI Pause click is delivered after this real step completes.
   // Never suppress a call or write a physical state to enforce the horizon.
   if(globalThis.document)queueMicrotask(()=>{if(globalThis.heaven?.state&&!globalThis.heaven.state.paused)document.getElementById('pause')?.click();});
  }
  return result;
 };
}

export function exportOnsetCapture(){
 const capture=globalThis.__flybodyOnset;
 return {schema:1,kind:capture.kind??'actual-original-ui-onset',seconds:capture.seconds,done:capture.done??false,
  scene:capture.scene,initial:capture.initial,frames:capture.frames,
  observationWallSeconds:(capture.finishedWallMs-capture.startedWallMs)/1000,
  physicalStateWritesByRecorder:0};
}

export function restoreOnsetState(body,state){
 const d=body.data;
 for(const [key,value] of Object.entries(state.native)){if(key==='time')d.time=value;else d[key].set(value);}
 body.mj.mj_forward(body.model,d);
 // mj_forward computes accelerations, but the next original mj_step started
 // from the recorded warm start. Restore it after the required cache rebuild.
 d.qacc_warmstart.set(state.native.qacc_warmstart);
 body.muscles.core.HEAPF32.set(state.muscleState,body.muscles.state/4);
 body.muscleState=Float32Array.from(state.muscleState);body.input.set(state.muscleInput);body.activation.set(state.activation);
 body.restPose=Float64Array.from(state.restPose);body.restHeight=state.restHeight;body.remainder=state.remainder;
 Object.assign(body.internal,state.internal);body.food=copy(state.food);
 body.wings.phase=state.wing.phase;
 for(const key of ['deployment','power','opening','target'])body.wings[key].set(state.wing[key]);
 state.wing.residuals.forEach((row,index)=>body.wings.residuals[index].set(row));
 body.wings.lastSteering=undefined;
 body.wingPhase=state.wing.phase;[body.wingPowerLeft,body.wingPowerRight]=state.wing.power;
 body.wingPower=(body.wingPowerLeft+body.wingPowerRight)/2;
 body.refresh();
}
