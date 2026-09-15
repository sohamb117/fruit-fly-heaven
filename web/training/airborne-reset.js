// Explicit initial-condition fixture. The only restraints operate during the
// declared unscored warm-up. Neural/event/muscle/native clocks never reset.
const check=(ok,message)=>{if(!ok)throw new Error(message);};
const finite=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
const copyRoot=data=>({qpos:Array.from(data.qpos.slice(0,7)),qvel:Array.from(data.qvel.slice(0,6))});
function sameRoot(data,expected){
 for(const [actual,wanted,name]of[[data.qpos,expected.qpos,'qpos'],[data.qvel,expected.qvel,'qvel']])
  for(let i=0;i<wanted.length;i++)check(Object.is(actual[i],wanted[i]),'Root '+name+' changed outside the fixture/native integrator at '+i);
}
function zeroForces(data){for(const a of[data.qfrc_applied,data.xfrc_applied])for(const v of a)check(v===0,'Unexpected applied external force');}
export function beginAirborneWarmup({body,world,fly,settings}){
 check(settings&&settings.profile==='airborne-live-warmup-v1'&&settings.schemaVersion===1,'Unknown airborne initialization contract');
 check(settings.bodyVariant==='full-native'&&/^[0-9a-f]{64}$/.test(settings.bodyVariantHash),'Only a pinned full native body is implemented by this fixture');
 check(Number.isFinite(settings.warmupSeconds)&&settings.warmupSeconds>=.1&&settings.warmupSeconds<=2&&Math.abs(settings.warmupSeconds/.002-Math.round(settings.warmupSeconds/.002))<1e-9,'Warm-up must be100–2000ms in exact2ms blocks');
 check(finite(settings.rootQpos,7)&&Math.abs(Math.hypot(...settings.rootQpos.slice(3))-1)<1e-12,'Explicit normalized airborne root pose required');
 check(settings.rootQpos[2]>0&&Math.hypot(...settings.rootQpos.slice(0,2))<6.5,'Airborne reset pose outside habitat');
 const {mj,model,data}=body;check(world.mj===mj&&world.model===model,'Body/world native identity mismatch');
 check(model.nq===57&&model.nv===56&&data.qpos.length===57&&data.qvel.length===56&&model.opt.timestep===.00005,'Expected50us native free-root model');
 check(body.time===0&&data.time===0&&body.remainder===0,'Airborne warm-up requires a fresh zero-time body');
 const event=body._wingMotorEvents;
 check(event&&event.initialized&&event.elapsedMs===0&&event.observedMs===0&&event.adapter.snapshot().pending===null,'Initialized wing event packet0 required before warm-up');
 zeroForces(data);
 const original=Object.getOwnPropertyDescriptor(mj,'mj_step');
 check(original&&typeof original.value==='function'&&(original.writable||original.configurable),'Cannot install explicit native warm-up fixture');
 const held=Object.freeze([...settings.rootQpos]),warmSeconds=settings.warmupSeconds;
 let mode='warmup',directRootWrites=0,warmNativeSteps=0,releasedNativeSteps=0,releaseNativeTime=null,releaseNeuralTimeMs=null,lastRoot,disposed=false,releaseVelocity=null;
 const resetRoot=()=>{data.qpos.set(held,0);data.qvel.fill(0,0,6);directRootWrites++;lastRoot=copyRoot(data);};
 const audit=()=>({schemaVersion:1,profile:settings.profile,mode,bodyVariant:settings.bodyVariant,rootQpos:[...held],warmupSeconds:warmSeconds,
  rootRestraintActive:mode==='warmup',rootWriteCount:directRootWrites,warmNativeSteps,releasedNativeSteps,releaseNativeTime,releaseNeuralTimeMs,
  warmupForces:'Explicit root kinematic restraint during setup only; not counted as aerodynamic support or task progress.',
  nonwingJoints:'All remain dynamic and receive their original native muscle controls; this fixture never writes nonwing positions/velocities.',
  release:'Retains native integration/warmstart, muscles, wing phase/deployment, event queues and BANC state. No clock reset or synthetic motor stream.',
  externalForceArraysWritten:false,...(releaseVelocity?{releaseVelocity:[...releaseVelocity],releaseVelocityStatus:'Explicit one-time recovery initial condition in native cm/s and rad/s; no subsequent root intervention.'}:{})});
 try{
  resetRoot();mj.mj_forward(model,data);
  if(body._wingLoadFeedback)body._wingLoadFeedback.capture(data,data.time);
  body.refresh();check(body.environmentContactCount===0,'Initial airborne pose touches the environment');world.copyPose(fly,body,0);
  body.monitor?.resetContinuity('explicit unscored airborne initialization');
  lastRoot=copyRoot(data);
  Object.defineProperty(mj,'mj_step',{...original,value:function(givenModel,givenData){
   check(!disposed&&givenModel===model&&givenData===data,'Unexpected native body during airborne fixture');
   sameRoot(data,lastRoot);zeroForces(data);
   if(mode==='warmup'){
    check(data.time+.00005<=warmSeconds+1e-9,'Warm-up reached release boundary; finish before another step');
    resetRoot();const result=original.value.apply(this,arguments);warmNativeSteps++;
    check(data.qpos.every(Number.isFinite)&&data.qvel.every(Number.isFinite),'Nonfinite native warm-up state');
    resetRoot();zeroForces(data);return result;
   }
   check(mode==='released','Native integration after fixture disposal');
   const result=original.value.apply(this,arguments);releasedNativeSteps++;lastRoot=copyRoot(data);zeroForces(data);return result;
  }});
 }catch(error){Object.defineProperty(mj,'mj_step',original);disposed=true;mode='aborted';throw error;}
 return {audit,
  finish({neuralTimeMs,releaseVelocity:requestedVelocity}={}){
   check(requestedVelocity===undefined||finite(requestedVelocity,6),'Recovery release velocity requires six finite native components');
   check(!disposed&&mode==='warmup','Airborne fixture is not warming');
   check(Math.abs(data.time-warmSeconds)<1e-8&&Math.abs(body.time-data.time)<1e-10,'Warm-up has not reached exact release boundary');
   check(Number.isFinite(neuralTimeMs)&&Math.abs(neuralTimeMs/1000-data.time)<1e-8,'Neural/native release clocks disagree');
   check(warmNativeSteps===Math.round(warmSeconds/.00005),'Unexpected native warm-up step count');
   check(body.remainder===0&&event.elapsedMs===Math.round(warmSeconds*1000)&&event.observedMs===event.elapsedMs&&event.adapter.snapshot().pending===null,'Wing event history not fully consumed at release');
   sameRoot(data,lastRoot);zeroForces(data);
   if(requestedVelocity!==undefined){
    releaseVelocity=[...requestedVelocity];data.qvel.set(releaseVelocity,0);directRootWrites++;
    mj.mj_forward(model,data);
    // Velocity-only release retains the last completed-step wing-load sample.
    // A current-time mj_forward force is not a completed native-step sample.
    lastRoot=copyRoot(data);
   }
   releaseNativeTime=data.time;releaseNeuralTimeMs=neuralTimeMs;mode='released';
   body.monitor?.resetContinuity('airborne warm-up released; scoring begins');body.refresh();world.copyPose(fly,body,0);
   sameRoot(data,lastRoot);
   return audit();
  },
  abort(){
   if(!disposed){Object.defineProperty(mj,'mj_step',original);disposed=true;mode=mode==='released'?'finished':'aborted';}
   return audit();
  },
 };
}
