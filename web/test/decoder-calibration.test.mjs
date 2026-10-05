import test from 'node:test';
import assert from 'node:assert/strict';
import {buildMotorDecoderContract,createMotorDecoder} from '../motor-decoder.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';
import {fitBoundedRidge} from '../../scripts/calibrate-motor-decoder.mjs';
import {fitBoundedRidgeStatistics,createDecoderStatistics,fitDecoderStatistics,createDemonstrationCollector,
  createDecoderCalibrationRunner} from '../training/decoder-calibration.js';
import {controlNames,createFlightTeacher,validateFlightTeacherCalibration} from '../training/flight-teacher.js';

function fixture(){let n=1000;const muscles=[];
  for(const side of ['left','right'])for(const [target,kind,count]of [
    ['dorsal_longitudinal_muscle','asynchronous_wing',5],['dorsoventral_muscle','asynchronous_wing',7],
    ...STEERING_MUSCLE_TYPES.map(type=>[type,'wing_steering_assumption',1])]){
    const indices=Array.from({length:count},()=>n++);muscles.push({kind,target,joint:(kind==='asynchronous_wing'?'wing_power_':'wing_steer_')+side,
      sign:1,indices,root_ids:indices.map(i=>(720575940000000000n+BigInt(i)).toString())});}
  return {muscles};
}
const io=fixture(),contract=buildMotorDecoderContract(io),prior=contract.parameters.map(p=>p.initial);
const required=['/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json','/body-engine/mujoco.wasm','/flybody-wings.js','/flybody-physics.js'];
function calibration(){const J=Array.from({length:4},()=>Array(20).fill(0));J[0][0]=J[0][1]=1;J[1][2]=1;J[2][5]=1;J[3][8]=1;
  return {schemaVersion:1,kind:'flight-state-teacher-calibration-v1',source:{kind:'native-control-calibration',sha256:'a'.repeat(64),
    description:'Synthetic unit-test calibration only; no claim of native evidence'},mechanics:Object.fromEntries(required.map(k=>[k,'b'.repeat(64)])),
    controlNames:[...controlNames],trim:Array.from({length:20},(_,i)=>i<2?.5:0),J,response:[1,0,0,0],
    inertia:[[.000001,0,0],[0,.000002,0],[0,0,.000003]],massG:.001,gains:{naturalFrequency:12,dampingRatio:.9}};
}
const close=(a,b,e=1e-10)=>assert(Math.abs(a-b)<=e,`${a} vs ${b}`);

test('browser normal equations match existing row-based bounded ridge, including signed bounds and unused inputs',()=>{
  const rows=[[1,0,0],[.5,1,0],[-.5,.2,0],[.25,-1,0]],y=[2,-1,.3,.7],weights=[.1,.2,.3,.4],initial=[.1,-.1,.4],bounds=[[-.25,.25],[-.25,.25],[0,1]],
    G=new Float64Array(9),b=new Float64Array(3);let y2=0;
  rows.forEach((row,i)=>{y2+=weights[i]*y[i]**2;row.forEach((x,j)=>{b[j]+=weights[i]*x*y[i];row.forEach((z,k)=>G[j*3+k]+=weights[i]*x*z);});});
  const old=fitBoundedRidge(rows,y,weights,initial,bounds),now=fitBoundedRidgeStatistics(G,b,y2,initial,bounds);
  now.values.forEach((v,i)=>close(v,old.values[i]));assert.equal(now.values[2],.4);assert.deepEqual(now.unexcitedColumns,[2]);
  assert(now.objectiveAfter<=now.objectiveBefore);assert.throws(()=>fitBoundedRidgeStatistics([NaN],[0],0,[0],[[-1,1]]));
});
function syntheticTrial(seed,split,{zeroFeatures=false,zeroTargets=false,validationShift=0}={}){
  const d=createMotorDecoder(io,prior),stats=createDecoderStatistics(contract);let state=seed;
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/2**32;};
  for(let k=0;k<32;k++){
    d.advance(Array.from({length:48},()=>zeroFeatures?0:random()));
    for(let sub=0;sub<5;sub++){
      const phase=(k+sub*.2)*1.4,features=d.features(phase),power=[0,0],steering=[[0,0,0],[0,0,0]];
      // Explicit algorithm fixture: targets are a declared mathematical
      // instrument, not biological/native training evidence.
      if(!zeroTargets)for(let side=0;side<2;side++){
        power[side]=.8*contract.parameters.reduce((s,p,j)=>s+(p.kind==='power'&&p.sideIndex===side?features[j]:0),0)+validationShift;
        for(let axis=0;axis<3;axis++)steering[side][axis]=.01*features[contract.parameters.findIndex(p=>p.kind==='steering'&&p.sideIndex===side&&p.axisIndex===axis)];
      }
      stats.add(features,{power,steering});
    }
  }
  return {seed,split,statistics:stats.finish()};
}
test('actual production features fit bounded672 vector; validation labels never influence updates',async()=>{
  const train=syntheticTrial(1,'train'),validation=syntheticTrial(2,'validation'),
    result=await fitDecoderStatistics(contract,[train,validation],prior,{maxSweeps:500});
  assert.equal(result.parameters.length,672);assert(result.changedParameterCount>0);assert(result.after.validation.mixedUnitRawRmse<result.before.validation.mixedUnitRawRmse);
  const shifted=await fitDecoderStatistics(contract,[train,syntheticTrial(2,'validation',{validationShift:.2})],prior,{maxSweeps:500});
  assert.deepEqual(result.parameters,shifted.parameters);assert.notEqual(result.after.validation.mixedUnitRawRmse,shifted.after.validation.mixedUnitRawRmse);
  result.parameters.forEach((v,i)=>assert(v>=contract.parameters[i].min&&v<=contract.parameters[i].max));
  await assert.rejects(()=>fitDecoderStatistics(contract,[train,{...validation,seed:1}],prior),/independent/);
});
test('no motor information and zero teacher targets fail explicitly',async()=>{
  const validation=syntheticTrial(2,'validation');
  await assert.rejects(()=>fitDecoderStatistics(contract,[syntheticTrial(1,'train',{zeroFeatures:true}),validation],prior),/no motor-feature/);
  await assert.rejects(()=>fitDecoderStatistics(contract,[syntheticTrial(1,'train',{zeroTargets:true}),validation],prior),/no nonzero teacher/);
  const s=createDecoderStatistics(contract);assert.throws(()=>s.finish(),/no applied/);
});
test('privileged teacher is bounded and deterministic; raw targets remain distinct from clipping',()=>{
  const c=calibration(),a=createFlightTeacher({calibration:c,targetHeight:3.5}),b=createFlightTeacher({calibration:c,targetHeight:3.5});
  const observation={quaternion:[Math.cos(.1),Math.sin(.1),0,0],omegaRoot:[1,0,0],height:3,verticalSpeed:-1,dt:.002};
  assert.deepEqual(a.update(observation),b.update(observation));assert(a.sample(0).controls.steering[0][0]<0);
  assert(a.sample(0).controls.power.every(x=>x>=.5&&x<=1));
  const high=calibration();high.trim[2]=high.trim[3]=high.trim[4]=.25;
  const paired=createFlightTeacher({calibration:high,targetHeight:3.5}).sample(Math.PI/4);
  assert(paired.targets.steering[0][0]>.25);assert.equal(paired.controls.steering[0][0],.25);
  const bad=calibration();bad.source.kind='synthetic-teacher';assert.throws(()=>validateFlightTeacherCalibration(bad),/native calibration/);
});
function bodyFixture(){const decoder=createMotorDecoder(io,prior),data={time:0,qfrc_applied:new Float64Array(6),xfrc_applied:new Float64Array(6),
  qpos:Float64Array.from([0,0,3.5,1,0,0,0]),qvel:new Float64Array(6),subtree_com:Float64Array.from([0,0,0,0,0,3.5]),subtree_linvel:new Float64Array(6)};
  return {motorDecoder:decoder,time:0,data,metadata:{mass_g:.001},model:{jnt_bodyid:[1],jnt_type:[0],body_subtreemass:[0,.001]},
    mj:{mj_kinematics(){},mj_comPos(){},mj_comVel(){},mj_subtreeVel(){}}};
}
function block(body,collector,phase){collector.beforeStep({body,phase,durationSeconds:.002,neuralMs:(body.time+.002)*1000});
  for(let ms=0;ms<2;ms++){for(let sub=0;sub<5;sub++){
    // Match FlyBodyPhysics: data.time advances natively; body.time is stale
    // until refresh after the entire1ms interval and decoder history update.
    body.motorDecoder.sample(body.data.time*2*Math.PI*236);body.data.time+=.0002;
  }body.motorDecoder.advance(Array(48).fill(.4));body.time=body.data.time;}}
test('collector reads the native substep clock with stale body.time, preserves history and restores exact descriptor',()=>{
  const body=bodyFixture(),descriptor=Object.getOwnPropertyDescriptor(body,'motorDecoder'),original=body.motorDecoder,
    collector=createDemonstrationCollector({body,contract,initialParameters:prior,calibration:calibration(),targetHeight:3.5});
  block(body,collector,'warmup');collector.release(3.5);block(body,collector,'scored');
  const captured=collector.finish();assert.equal(captured.summary.samples,20);assert.equal(captured.summary.warmupSamples,10);assert.equal(captured.summary.scoredSamples,10);
  assert.equal(original.timeMs,4);assert.deepEqual(original.snapshot().history[3],Array(48).fill(.4));
  assert(captured.summary.auditSamples.every(s=>s.nativeTimeSeconds*1000>=s.decoderTimeMs-1e-8));
  assert.deepEqual(captured.summary.auditSamples.slice(0,5).map(s=>s.decoderTimeMs),[0,0,0,0,0]);
  captured.summary.auditSamples.slice(0,5).forEach((s,i)=>assert(Math.abs(s.nativeTimeSeconds-i*.0002)<1e-12));
  collector.restore();assert.deepEqual(Object.getOwnPropertyDescriptor(body,'motorDecoder'),descriptor);collector.restore();
  assert(body.data.qfrc_applied.every(x=>x===0));assert.deepEqual(Array.from(body.data.qpos),[0,0,3.5,1,0,0,0]);
});
test('failed teacher returns no candidate and restores decoder; bad source fails before evaluation',async()=>{
  const c=calibration(),text=JSON.stringify(c),hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),x=>x.toString(16).padStart(2,'0')).join(''),
    config={motorDecoderContract:contract,parameters:contract.parameters,assets:{...c.mechanics,'/teacher.json':hash},modelFingerprint:'test',
      initialCondition:{rootQpos:[0,0,3.5,1,0,0,0]},stages:[{id:'maintained_flight',durationSeconds:5}]};
  let calls=0,lastBody,original;
  const environment={configHash:'config',modelFingerprint:'test',async evaluate(job,callbacks){
    calls++;lastBody=bodyFixture();original=lastBody.motorDecoder;
    callbacks.onBeforePhysicsStep({body:lastBody,phase:'warmup',durationSeconds:.002,neuralMs:2});
    lastBody.motorDecoder.sample(0);
    return {success:false,simSeconds:0,reason:'physical_failure',cancelled:false,metrics:{error:null}};
  }};
  const runner=createDecoderCalibrationRunner({config,environment,initialParameters:prior,teacherCalibrationText:text,teacherCalibrationSha256:hash});
  const trials=[{seed:1,split:'train',stage:'maintained_flight',durationSeconds:5},{seed:2,split:'validation',stage:'maintained_flight',durationSeconds:5}];
  const result=await runner.run({trials});assert.equal(result.status,'teacher_failed');assert.equal(result.candidate,null);assert.equal(lastBody.motorDecoder,original);assert.equal(calls,1);
  const bad=createDecoderCalibrationRunner({config,environment,initialParameters:prior,teacherCalibrationText:text+' ',teacherCalibrationSha256:hash});
  await assert.rejects(()=>bad.run({trials}),/hash mismatch/);assert.equal(calls,1);
});
