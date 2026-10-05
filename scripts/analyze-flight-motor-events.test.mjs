import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {analyzeMotorEventPair} from './analyze-flight-motor-events.mjs';

const ioBytes=fs.readFileSync(new URL('../data/prepared/banc888/io.json',import.meta.url));
const io=JSON.parse(ioBytes),ioSha=createHash('sha256').update(ioBytes).digest('hex');
const mappings=io.muscles.filter(row=>['asynchronous_wing','wing_steering_assumption'].includes(row.kind));
const indices=[...new Set(mappings.flatMap(row=>row.indices))].sort((a,b)=>a-b);
const b1=mappings.find(row=>row.target==='b1_muscle'&&row.joint.endsWith('_left')).indices[0];
const dlm=mappings.find(row=>row.target==='dorsal_longitudinal_muscle'&&row.joint.endsWith('_left')).indices;
const TAU=2*Math.PI,wrap=value=>((value%TAU)+TAU)%TAU;
function fixture(){
 const counts=indices.map(()=>0),plan=[{index:b1,timeMs:100},{index:b1,timeMs:104},{index:b1,timeMs:280},
  {index:dlm[0],timeMs:105},{index:dlm[1],timeMs:105}].sort((a,b)=>a.timeMs-b.timeMs||a.index-b.index);
 const packets=Array.from({length:141},(_,p)=>{
  const events=plan.filter(event=>event.timeMs>(p-1)*2&&event.timeMs<=p*2);
  for(const event of events)counts[indices.indexOf(event.index)]++;
  return {initialized:p===0,fromTimeMs:p===0?null:(p-1)*2,timeMs:p*2,indices:[...indices],ratesHz:indices.map((_,k)=>k+.25),counts:[...counts],events,
   bodyTimeSeconds:p===0?0:(p-1)*.002,wingPhaseRadians:wrap(TAU-.1+TAU*250*(p===0?0:(p-1)*.002)),wingFrequencyHz:250};
 });
 const assignment={name:'capture',seed:888,stage:'flight',durationSeconds:.28,parameters:Array(27).fill(0),capturePhysicsDigest:true,captureMotorEvents:true};
 const evaluation={parameters:assignment.parameters,seed:888,stage:'flight',environmentVersion:5,backend:'wasm',bodyBackend:'mujoco-wasm',dtMs:.5,bodyBlockMs:2,
  configHash:'1'.repeat(64),modelFingerprint:'2'.repeat(64),return:0,success:false,terminated:false,truncated:true,reason:'time_limit',cancelled:false,simSeconds:.28,steps:140,provenance:{seed:888},metrics:{error:null}};
 const captured={schemaVersion:1,kind:'native-held-out-evaluation',assignment,evaluation,complete:true,error:null,configHash:evaluation.configHash,modelFingerprint:evaluation.modelFingerprint,
  backendLabel:'wasm',nativeWebGPU:null,sourceHashes:{'/banc-data/io.json':ioSha},motorEvents:packets,
  physicsDigest:{sha256:'3'.repeat(64),rows:141,valuesPerRow:630,encoding:'Native-endian float64',fields:['body.time','qpos','qvel','act','ctrl','muscleState','wingPhase','wingDeployment','wingPower','wingTarget']}};
 const control=structuredClone(captured);control.assignment.name='control';control.assignment.captureMotorEvents=false;delete control.motorEvents;
 return {captured,control};
}
const analyze=({captured,control})=>analyzeMotorEventPair(captured,control,io,ioSha);

test('counts exact open-left window separately from EMA and extrapolates pre-body table phase',()=>{
 const pair=fixture(),before=JSON.stringify(pair),r=analyze(pair),row=r.perNeuron.find(row=>row.index===b1);
 assert.equal(r.perNeuron.length,48);assert.equal(r.perMuscle.length,28);
 assert.equal(row.count,2);assert.equal(row.rawRateHz,2/.18);assert.notEqual(row.rawRateHz,row.emaRateHz.mean);
 assert.deepEqual(row.events.map(event=>event.timeMs),[104,280]);
 for(const event of row.events)assert.ok(Math.abs(event.tablePhaseRadians-(TAU-.1))<1e-10);
 assert.ok(Math.abs(row.phase.resultantLength-1)<1e-12);
 const silent=r.perNeuron.find(row=>row.count===0);assert.equal(silent.phase.resultantLength,null);assert.equal(silent.phase.meanRadians,null);
 assert.equal(JSON.stringify(pair),before,'analysis must not mutate records');
});
test('per-unit DLM counts and same-tick coincidence retain identities',()=>{
 const r=analyze(fixture()),left=r.dlmCoincidences.find(row=>row.side==='left');
 assert.equal(left.motorIndices.length,5);assert.equal(left.totalSpikes,2);assert.equal(left.simultaneousTicks,1);
 assert.equal(left.ticksByNumberOfSpikingNeurons[2],1);assert.deepEqual(left.simultaneous[0].motorIndices,[dlm[0],dlm[1]].sort((a,b)=>a-b));
});
test('capture ordering is immaterial but both explicit flags are required',()=>{
 const p=fixture();assert.deepEqual(analyzeMotorEventPair(p.control,p.captured,io,ioSha),analyze(p));
 delete p.control.assignment.captureMotorEvents;assert.throws(()=>analyze(p),/explicit captureMotorEvents/);
});
test('rejects unpaired parameters, changed digest, or unpinned metadata',()=>{
 for(const mutate of [p=>{p.control.assignment.parameters[0]=.1;},p=>{p.control.physicsDigest.sha256='4'.repeat(64);},p=>{p.control.sourceHashes['/banc-data/io.json']='5'.repeat(64);}]){
  const p=fixture();mutate(p);assert.throws(()=>analyze(p),/mismatch/);
 }
 const p=fixture();assert.throws(()=>analyzeMotorEventPair(p.captured,p.control,io,'6'.repeat(64)),/metadata is not pinned/);
});
test('rejects cancelled/incomplete records and insufficient window coverage',()=>{
 for(const mutate of [p=>{p.captured.complete=false;},p=>{p.captured.evaluation.cancelled=true;},p=>{p.captured.error='failure';}]){
  const p=fixture();mutate(p);assert.throws(()=>analyze(p));
 }
 const p=fixture();for(const r of [p.captured,p.control]){r.evaluation.steps=130;r.evaluation.simSeconds=.26;r.evaluation.reason='overturned';r.evaluation.terminated=true;r.physicsDigest.rows=131;}
 p.captured.motorEvents.length=131;assert.throws(()=>analyze(p),/complete .*analysis window/);
});
test('rejects missing/reordered packets and shifted body context',()=>{
 for(const mutate of [p=>{p.captured.motorEvents.splice(20,1);},p=>{p.captured.motorEvents[10].fromTimeMs=16;},p=>{p.captured.motorEvents[10].bodyTimeSeconds=.02;}]){
  const p=fixture();mutate(p);assert.throws(()=>analyze(p));
 }
});
test('rejects lost events, duplicate events, unknown neurons, and timestamps off neural ticks',()=>{
 for(const mutate of [p=>{p.captured.motorEvents[52].events=[];},p=>{p.captured.motorEvents[52].events.push({...p.captured.motorEvents[52].events[0]});},p=>{p.captured.motorEvents[52].events[0].index=999999;},p=>{p.captured.motorEvents[52].events[0].timeMs=103.7;}]){
  const p=fixture();mutate(p);assert.throws(()=>analyze(p));
 }
});
test('rejects counter drift, reordered IDs, invalid rate, and inconsistent phase',()=>{
 for(const mutate of [p=>{p.captured.motorEvents[50].counts[0]++;},p=>{p.captured.motorEvents[50].indices.reverse();},p=>{p.captured.motorEvents[50].ratesHz[0]=NaN;},p=>{p.captured.motorEvents[50].wingPhaseRadians=.75;}]){
  const p=fixture();mutate(p);assert.throws(()=>analyze(p));
 }
});
