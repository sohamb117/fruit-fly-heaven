import {test} from 'node:test';
import assert from 'node:assert/strict';
import {median,sampleValidity,compareSamples} from './benchmark-transplant-metrics.mjs';

const config={movementMode:'direct',movementControl:'direct',bodyClock:'neural',fast:false,referenceController:false,flightEnabled:true,motorCoupling:true,
  sensorySwitches:{odor:true,taste:true,vision:true,'body-sense':true},bodyBackend:'mujoco-wasm'};
function validSample(){
  const point=(time,frame)=>({wall:time*100000,neural:time*1000,paused:false,population:1,configuration:structuredClone(config),backend:'webgpu',eyesReady:1,gradedReady:1,colorReady:1,
    bodies:[{id:1,time,neuralMs:time*1000,position:[1,2,3],eyeSequence:frame}],anatomy:{frames:frame,scanPixels:100},rendererFrames:frame,errors:[]});
  return {dataset:'banc',mode:'reference',population:1,trial:1,wallMs:20000,neuralMs:200,cohortSpeed:.01,before:point(.02,1),after:point(.22,5),errors:[]};
}
test('two-trial median averages both trials rather than selecting the faster one',()=>{
  assert(Math.abs(median([.2,.1])-.15)<1e-15);assert.equal(median([4,1,3]),3);assert.equal(median([]),null);
});
test('valid native neural-clock sample remains valid',()=>assert.deepEqual(sampleValidity(validSample(),{seconds:20}),[]));
test('frozen body, pause, wrong clocks/configuration and stale retinal/anatomy work fail validity',()=>{
  for(const change of [s=>s.after.bodies[0].time=s.before.bodies[0].time,s=>s.after.paused=true,s=>s.after.configuration.bodyClock='live',
    s=>s.after.configuration.fast=true,s=>s.after.configuration.movementMode='behavior',s=>s.after.configuration.sensorySwitches.taste=false,
    s=>s.after.bodies[0].time+=.1,s=>s.after.bodies[0].position[0]=NaN,s=>s.after.bodies[0].eyeSequence=1,
    s=>s.after.anatomy.frames=1,s=>s.wallMs=19000]){
    const s=validSample();change(s);assert(sampleValidity(s,{seconds:20}).length>0);
  }
});
test('FlyWire fixed-step residual is allowed but a stalled body is rejected',()=>{
  const s=validSample();s.dataset='flywire';for(const point of [s.before,s.after]){point.backend='wasm';point.configuration.bodyBackend='kinematic';point.bodies[0].time-=.015;}
  assert.deepEqual(sampleValidity(s,{seconds:20}),[]);s.after.bodies[0].time=s.before.bodies[0].time;assert(sampleValidity(s,{seconds:20}).length>0);
});
function pairSamples(){return [['flywire',1,.1],['banc',1,.02],['flywire',2,.3],['banc',2,.02]].map(([dataset,trial,cohortSpeed])=>({dataset,trial,cohortSpeed,valid:true,population:1,mode:'reference',after:{backend:dataset==='banc'?'webgpu':'wasm',configuration:{bodyBackend:dataset==='banc'?'mujoco-wasm':'kinematic'}}}));}
test('paired comparisons report a true median and expose a failing individual trial',()=>{
  const c=compareSamples(pairSamples(),{population:1,mode:'reference',trials:2});
  assert.equal(c.slowdown,10);assert.equal(c.withinOneOrderOfMagnitude,true);assert.equal(c.everyPairedTrialWithinOneOrderOfMagnitude,false);assert.deepEqual(c.pairedSlowdownRange,[5,15]);
});
test('failed/missing/duplicate trials and backend drift cannot yield a same-order claim',()=>{
  for(const alter of [rows=>rows[0].valid=false,rows=>rows.pop(),rows=>rows[2].trial=1,rows=>rows[3].after.backend='wasm']){
    const rows=pairSamples();alter(rows);const c=compareSamples(rows,{population:1,mode:'reference',trials:2});assert.equal(c.eligible,false);assert.equal(c.withinOneOrderOfMagnitude,null);assert.equal(c.slowdown,null);
  }
});
