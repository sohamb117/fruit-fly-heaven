import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_RECOVERY_DISTURBANCE,RECOVERY_CRITERIA,createRecoveryDisturbance,validateRecoveryDisturbance,createRecoveryScore} from '../training/recovery-objective.js';
import {beginAirborneWarmup} from '../training/airborne-reset.js';
import {createMaintainedFlightClock} from '../training/airborne-reset-contract.js';
import {createTegulaStrainPrior,createTegulaInputMapper} from '../banc-tegula.js';

const settings={schemaVersion:1,profile:'airborne-live-warmup-v1',warmupSeconds:.5,bodyVariant:'full-native',bodyVariantHash:'a'.repeat(64),rootQpos:[0,0,3,1,0,0,0]};
const obs=(patch={})=>({finite:true,externalForce:false,up:1,angularSpeed:0,speedCmPerSecond:0,verticalSpeed:0,height:3,radius:0,ceiling:50,wingPower:.9,environmentContacts:0,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0,...patch});
function run(make){const score=createRecoveryScore(5,make(0));for(let k=1;k<=2500&&!score.state.reason;k++)score.step(make(k*.002),.002);return score.state;}

test('release disturbances repeat by seed, vary held-out directions and remain inside declared native-unit bounds',()=>{
 const a=createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,888),b=createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,888),c=createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,889);
 assert.deepEqual(a,b);assert.notDeepEqual(a.releaseVelocity,c.releaseVelocity);
 for(let seed=0;seed<64;seed++){
  const d=createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,seed),angular=Math.hypot(...d.angularVelocityRootRadS),linear=Math.hypot(...d.linearVelocityWorldCmS);
  assert(angular>=24&&angular<=36);assert(linear>=2&&linear<=6);
  assert.deepEqual(d.releaseVelocity,[...d.linearVelocityWorldCmS,...d.angularVelocityRootRadS]);assert(Object.isFrozen(d.releaseVelocity));
 }
 for(const change of [x=>x.angularSpeedRangeRadS=[0,36],x=>x.linearSpeedRangeCmS=[2,Infinity],x=>x.angularSpeedRangeRadS=[36,24],x=>x.extra=1]){
  const x=structuredClone(DEFAULT_RECOVERY_DISTURBANCE);change(x);assert.throws(()=>validateRecoveryDisturbance(x),/Flight recovery/);
 }
 assert.throws(()=>createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,-1),/seed/);
});

test('recovery requires timely settled powered flight and the full five-second horizon',()=>{
 const score=createRecoveryScore(5,obs({angularSpeed:30}));
 for(let k=1;k<2500;k++){
  score.step(obs({angularSpeed:k<200?24:0}),.002);assert.equal(score.state.success,false);
 }
 const result=score.step(obs(),.002);assert.equal(result.reason,'recovery_success');assert.equal(result.success,true);
 assert.equal(score.state.stage,'recovery');assert(score.state.recoveredAt<1);assert(score.state.recoveredSeconds>4.5);
 assert.equal(score.state.hasTakenOff,false);assert.equal(score.state.takeoffTime,null);assert.equal(score.state.elapsed,5);
 assert.equal(RECOVERY_CRITERIA.minimumFinalRecoveredSeconds,1);assert(score.state.return>9);
});

test('late recovery, contacts, late instability, passive fall and broad upright posture cannot win',()=>{
 for(const make of [
  t=>obs({angularSpeed:t<2?24:0}),
  t=>obs(t===1?{environmentContacts:1,nonFootEnvironmentContacts:1}:{}),
  t=>obs({angularSpeed:t>4.5?24:0}),
  ()=>obs({up:.75}),
  t=>obs({height:3-.5*981*t*t,verticalSpeed:-981*t,speedCmPerSecond:981*t})
 ])assert.equal(run(make).success,false);
 const forced=run(t=>obs({externalForce:t>.1}));assert.equal(forced.reason,'unexpected_external_force');assert.equal(forced.return,-10);
});

function fixture(){
 const data={time:0,qpos:new Float64Array(57),qvel:new Float64Array(56),qfrc_applied:new Float64Array(56),xfrc_applied:new Float64Array(6)},
  model={nq:57,nv:56,opt:{timestep:.00005}},mj={mj_step(_m,d){d.time+=.00005;},mj_forward(){}},
  body={data,model,mj,get time(){return data.time;},remainder:0,environmentContactCount:0,refresh(){},monitor:{resetContinuity(){}},
   _wingMotorEvents:{initialized:true,elapsedMs:0,observedMs:0,adapter:{snapshot:()=>({pending:null})}}},
  world={mj,model,copyPose(){}},fly={},captures=[];
 body._wingLoadFeedback={capture(_data,time){captures.push(time);}};
 data.qpos.set(settings.rootQpos);data.qpos[10]=.4;data.qvel[12]=.7;
 const original=mj.mj_step,warmup=beginAirborneWarmup({body,world,fly,settings});
 for(let k=0;k<10000;k++)mj.mj_step(model,data);
 body._wingMotorEvents.elapsedMs=500;body._wingMotorEvents.observedMs=500;
 const clock=createMaintainedFlightClock(settings,{bodyMetadataSha256:settings.bodyVariantHash});
 const clockSample=()=>({nativeTimeSeconds:data.time,neuralTimeMs:500,bodyEventElapsedMs:500,bodyEventObservedMs:500,lastPacketTimeMs:500,remainderSeconds:0,pendingEvents:false,rootRestraintActive:warmup.audit().rootRestraintActive,externalForceApplied:false,rootWriteCount:warmup.audit().rootWriteCount});
 return {data,model,mj,body,warmup,original,clock,clockSample,captures};
}

test('audited release writes only the six initial velocities and preserves pose, clocks and nonroot state',()=>{
 const f=fixture();try{
  const time=f.data.time,qpos=Array.from(f.data.qpos),nonroot=Array.from(f.data.qvel.slice(6)),before=f.warmup.audit(),v=createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,888).releaseVelocity;
  const audit=f.warmup.finish({neuralTimeMs:500,releaseVelocity:v});
  assert.deepEqual(Array.from(f.data.qvel.slice(0,6)),v);assert.deepEqual(Array.from(f.data.qvel.slice(6)),nonroot);assert.deepEqual(Array.from(f.data.qpos),qpos);assert.equal(f.data.time,time);
  assert.equal(audit.rootWriteCount,before.rootWriteCount+1);assert.equal(audit.rootRestraintActive,false);assert.equal(audit.externalForceArraysWritten,false);
  f.clock.release(f.clockSample());
  for(let k=0;k<40;k++)f.mj.mj_step(f.model,f.data);
  assert.equal(f.warmup.audit().rootWriteCount,audit.rootWriteCount);
  assert.equal(f.clock.advance({...f.clockSample(),nativeTimeSeconds:f.data.time,neuralTimeMs:502,bodyEventElapsedMs:502,bodyEventObservedMs:502,lastPacketTimeMs:502}).scoredSteps,1);
  f.data.qvel[3]++;assert.throws(()=>f.mj.mj_step(f.model,f.data),/changed outside/);
 }finally{f.warmup.abort();assert.equal(f.mj.mj_step,f.original);}
});

test('invalid release vectors do not change state; absent vector preserves the legacy zero-velocity release',()=>{
 const f=fixture();try{
  const before=f.warmup.audit(),vel=Array.from(f.data.qvel),time=f.data.time;
  for(const v of [[0,0,0],[0,0,0,0,NaN,0],null]){
   assert.throws(()=>f.warmup.finish({neuralTimeMs:500,releaseVelocity:v}),/six finite/);
   assert.deepEqual(f.warmup.audit(),before);assert.deepEqual(Array.from(f.data.qvel),vel);assert.equal(f.data.time,time);
  }
  const audit=f.warmup.finish({neuralTimeMs:500});assert.equal(audit.rootWriteCount,before.rootWriteCount);assert(f.data.qvel.slice(0,6).every(x=>x===0));
  f.mj.mj_step(f.model,f.data);
 }finally{f.warmup.abort();}
});

test('velocity-only release retains the last Euler force sample so the first scored tegula timestamp remains valid',()=>{
 const f=fixture(),mapper=createTegulaInputMapper(createTegulaStrainPrior({halfLoadNative:.02}));
 const feedback=(time,force=time===0?0:time-.00005)=>({wingLoad:{kind:'native-wing-aerodynamic-moment-v1',units:'g cm^2/s^2',localFrame:'native-thorax',
  momentThorax:{left:[.02,0,0],right:[-.02,0,0]},bodyTimeSeconds:time,forceTimeSeconds:force,localFrameTimeSeconds:force}});
 try{
  for(let k=0;k<250;k++)mapper.rates(feedback(k*.002),true,k*.002);
  const count=f.captures.length,releaseVelocity=createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,888).releaseVelocity;
  f.warmup.finish({neuralTimeMs:500,releaseVelocity});assert.equal(f.captures.length,count,'release must not manufacture a same-time forward-force receptor sample');
  const completed=feedback(f.data.time);assert.doesNotThrow(()=>mapper.rates(completed,true,f.data.time));
  assert.throws(()=>mapper.rates(feedback(f.data.time,f.data.time),true,f.data.time),/clock mismatch/);
 }finally{f.warmup.abort();}
});
