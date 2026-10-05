import test from 'node:test';
import assert from 'node:assert/strict';
import {createBrainObserver,MAX_BRAIN_SAMPLE} from '../training/observer-runtime.js';
import {createObservedTrainingWorker} from '../training/observer-worker.js';
import {createWasmCore,WasmBrain} from '../../packages/banc-runtime/src/wasm.js';
import {fixture as modelFixture} from '../../packages/banc-runtime/test/fixture.mjs';

function fakeBrain(){
 class Brain{
  constructor(){
   this.n=5;this.tick=0;this.states=[0,this.n*8*4];this.kinetics=this.n*16*4;
   this.core={HEAPF32:new Float32Array(this.n*35)};this.reads=0;this.result={owned:'original result'};
   this.core.HEAPF32.fill(-1e30,this.kinetics/4+this.n*18);this.update();
  }
  get timeMs(){return this.tick*.5;}
  update(){for(let i=0;i<this.n;i++){const offset=this.states[this.tick%2]/4+i*8;this.core.HEAPF32[offset]=-60+i+this.tick;this.core.HEAPF32[offset+4]=i*2+this.tick;}}
  step(){this.tick++;this.update();}
  readState(...args){this.reads++;this.args=args;if(this.error)throw this.error;return this.result;}
 }
 return Brain;
}
function observerFixture(){
 const Brain=fakeBrain(),messages=[];let wall=0;
 const observer=createBrainObserver({Brain,postMessage:(message,transfer)=>messages.push(structuredClone(message,{transfer})),now:()=>wall});
 return {Brain,observer,messages,advance(ms){wall+=ms;},brains:()=>messages.filter(message=>message.type==='brain')};
}

test('observer copies real selected values after the original read and owns every transferred buffer',()=>{
 const f=observerFixture(),brain=new f.Brain(),originalBytes=brain.core.HEAPF32.slice();
 try{
  const indices=[3,0];f.observer.configure({enabled:true,indices});indices[0]=4;f.observer.beginJob('trial-a');
  const arg={marker:true};assert.equal(brain.readState([1],arg),brain.result);assert.equal(brain.args[1],arg);assert.equal(brain.reads,1);
  const snapshot=f.brains()[0].snapshot;
  assert.deepEqual(snapshot.indices,Uint32Array.from([3,0]));assert.deepEqual(snapshot.voltage,Float32Array.from([-57,-60]));
  assert.deepEqual(snapshot.rates,Float32Array.from([6,0]));assert(snapshot.lastSpikeMs.every(value=>value<0));
  assert.equal(snapshot.neuralTimeMs,0);assert.equal(snapshot.jobId,'trial-a');assert.equal(snapshot.sampleSequence,1);
  for(const field of ['indices','voltage','rates','lastSpikeMs'])assert.notEqual(snapshot[field].buffer,brain.core.HEAPF32.buffer);
  assert.deepEqual(brain.core.HEAPF32,originalBytes);snapshot.voltage.fill(999);snapshot.indices.fill(4);
  f.advance(500);brain.step();brain.readState();assert.deepEqual(f.brains()[1].snapshot.indices,Uint32Array.from([3,0]));
  assert.deepEqual(f.brains()[1].snapshot.voltage,Float32Array.from([-56,-59]));assert.equal(brain.reads,2);
 }finally{f.observer.dispose();}
});

test('off, calibration, pause-like inactivity and job boundaries never publish stale brain state',()=>{
 const f=observerFixture(),a=new f.Brain(),b=new f.Brain();
 try{
  a.readState();assert.equal(f.brains().length,0);
  f.observer.configure({enabled:true,indices:[0]});a.readState();assert.equal(f.brains().length,0,'ready/calibration has no job');
  const first=f.observer.beginJob('a');a.readState();assert.equal(f.brains().length,1);
  f.advance(1000);assert.equal(f.brains().length,1,'no timer invents new samples without a read');
  f.observer.configure({enabled:false});a.readState();assert.equal(f.brains().length,1);
  f.observer.configure({enabled:true,indices:[0]});f.observer.endJob(first);a.readState();assert.equal(f.brains().length,1);
  const second=f.observer.beginJob('b');f.observer.endJob(first);b.readState();assert.equal(f.brains().at(-1).snapshot.jobId,'b');
  f.observer.endJob(second);f.advance(500);b.readState();assert.equal(f.brains().length,2);
  f.observer.beginJob(undefined);a.readState();assert.equal(f.brains().length,2);
 }finally{f.observer.dispose();}
});

test('sampling is at most twice per wall second even across toggles, selections and trials',()=>{
 const f=observerFixture(),a=new f.Brain(),b=new f.Brain();
 try{
  f.observer.configure({enabled:true,indices:[0]});f.observer.beginJob('a');a.readState();
  for(let k=0;k<5;k++){f.advance(99);a.step();a.readState();}
  assert.equal(f.brains().length,1);f.observer.configure({enabled:false});f.observer.configure({enabled:true,indices:[1]});
  f.observer.beginJob('b');b.readState();assert.equal(f.brains().length,1);
  f.advance(5);b.readState();assert.equal(f.brains().length,2);assert.equal(f.brains()[1].snapshot.sampleSequence,2);
 }finally{f.observer.dispose();}
});

test('invalid sample requests and observer failures cannot change a read result or its original error',()=>{
 const f=observerFixture(),brain=new f.Brain();
 try{
  for(const indices of [[],[0,0],[-1],[.5],[2**32],new Float32Array([1]),Array.from({length:MAX_BRAIN_SAMPLE+1},(_,i)=>i)]){
   assert.throws(()=>f.observer.configure({enabled:true,indices}),/indices/);assert.equal(f.observer.state.enabled,false);
  }
  assert.throws(()=>f.observer.configure({enabled:1,indices:[0]}),/boolean/);
  f.observer.configure({enabled:true,indices:[5]});f.observer.beginJob('a');
  assert.equal(brain.readState(),brain.result);assert.equal(f.messages.at(-1).type,'brain-error');assert.equal(f.observer.state.enabled,false);
  f.observer.configure({enabled:true,indices:[0]});const error=new Error('original read failed');brain.error=error;
  assert.throws(()=>brain.readState(),candidate=>candidate===error);assert.equal(f.brains().length,0);
  brain.error=null;brain.core.HEAPF32[0]=NaN;assert.equal(brain.readState(),brain.result);
  assert.equal(f.messages.at(-1).type,'brain-error');assert.equal(f.observer.state.enabled,false);
 }finally{f.observer.dispose();}
});

test('delivery failures are isolated and disposing restores the original prototype exactly',()=>{
 const Brain=fakeBrain(),descriptor=Object.getOwnPropertyDescriptor(Brain.prototype,'readState'),brain=new Brain();
 const observer=createBrainObserver({Brain,postMessage(){throw new Error('sink failed');},now:()=>0});
 observer.configure({enabled:true,indices:[0]});observer.beginJob('a');
 assert.equal(brain.readState(),brain.result);assert.equal(observer.state.enabled,false);assert.equal(brain.reads,1);
 observer.dispose();observer.dispose();assert.deepEqual(Object.getOwnPropertyDescriptor(Brain.prototype,'readState'),descriptor);
 assert.equal(brain.readState(),brain.result);assert.equal(brain.reads,2);
});

test('tiny actual WASM run produces identical states, events and read counts with observation enabled',async()=>{
 const core=await createWasmCore(),model=modelFixture({n:8}),input=new Float32Array(8).fill(48),selected=Uint32Array.from([5,1,0]);
 const original=WasmBrain.prototype.readState;let reads=0,wall=0;
 WasmBrain.prototype.readState=function(){reads++;return original.apply(this,arguments);};
 const counted=WasmBrain.prototype.readState;
 async function run(observe){
  const brain=new WasmBrain(core,model),snapshots=[],messages=[];let observer;
  try{
   if(observe){observer=createBrainObserver({Brain:WasmBrain,postMessage:(message,transfer)=>messages.push(structuredClone(message,{transfer})),now:()=>wall});observer.configure({enabled:true,indices:selected});observer.beginJob('native-fixture');}
   for(let block=0;block<12;block++){brain.step(4,input);wall+=500;snapshots.push(brain.readState(selected,{includeSpikeTime:true}));}
   return {snapshots,messages,tick:brain.tick,states:brain.states.map(offset=>core.HEAPF32.slice(offset/4,offset/4+brain.n*8)),
    kinetics:core.HEAPF32.slice(brain.kinetics/4,brain.kinetics/4+brain.n*19)};
  }finally{observer?.dispose();brain.dispose();}
 }
 try{
  const baseline=await run(false),baselineReads=reads,observed=await run(true);
  assert.equal(reads-baselineReads,baselineReads);assert.equal(baselineReads,12);assert.equal(WasmBrain.prototype.readState,counted);
  assert.deepEqual(observed.snapshots,baseline.snapshots);assert.deepEqual(observed.states,baseline.states);assert.deepEqual(observed.kinetics,baseline.kinetics);assert.equal(observed.tick,baseline.tick);
  assert.equal(observed.messages.length,12);
  observed.messages.forEach(({snapshot},block)=>selected.forEach((_,k)=>{
   assert.equal(snapshot.voltage[k],observed.snapshots[block][k*9]);assert.equal(snapshot.rates[k],observed.snapshots[block][k*9+4]);assert.equal(snapshot.lastSpikeMs[k],observed.snapshots[block][k*9+8]);
  }));
 }finally{WasmBrain.prototype.readState=original;}
});

test('worker wrapper attributes only accepted evaluations and keeps results, frames and lifecycle unchanged',async()=>{
 const Brain=fakeBrain(),original=Brain.prototype.readState,messages=[];let wall=0,closed=0,disposed=0;
 const worker=createObservedTrainingWorker({Brain,postMessage:(message,transfer)=>messages.push(structuredClone(message,{transfer})),now:()=>wall,close:()=>closed++,
  createEnvironment:async()=>({ready(){new Brain().readState();return {backend:'fixture'};},dispose(){disposed++;},async evaluate(job,{onFrame,checkpoint}){
   const brain=new Brain();for(let block=0;block<2;block++){await checkpoint();wall+=500;brain.step();brain.readState();onFrame({time:brain.timeMs});}
   return {return:1,steps:2,parameters:job.parameters};
  }}),
 });
 try{
  await worker.handle({type:'observe-brain',enabled:true,indices:[0,2]});await worker.handle({type:'initialize',id:1});
  assert.equal(messages.filter(m=>m.type==='brain').length,0);
  await worker.handle({type:'evaluate',id:2,job:{jobId:'first',parameters:[.25]}});
  await worker.handle({type:'evaluate',id:3,job:{jobId:'second',parameters:[.5]}});
  assert.deepEqual(messages.filter(m=>m.type==='brain').map(m=>m.snapshot.jobId),['first','first','second','second']);
  assert.deepEqual(messages.filter(m=>m.type==='evaluation').map(m=>m.result),[{return:1,steps:2,parameters:[.25]},{return:1,steps:2,parameters:[.5]}]);
  assert.equal(messages.filter(m=>m.type==='frame').length,4);assert.equal(worker.state.observation.jobId,null);
  await worker.handle({type:'observe-brain',enabled:true,indices:[-1]});assert.equal(messages.at(-1).type,'brain-error');
  await worker.handle({type:'stop'});assert.equal(closed,1);assert.equal(disposed,1);assert.equal(Brain.prototype.readState,original);
 }finally{worker.dispose();}
});

test('rejected concurrent commands cannot steal trial identity, while cancellation clears it immediately',async()=>{
 const Brain=fakeBrain(),messages=[];let release,wall=0;
 const worker=createObservedTrainingWorker({Brain,now:()=>wall,postMessage:(message,transfer)=>messages.push(structuredClone(message,{transfer})),
  createEnvironment:async()=>({ready:()=>({}),dispose(){},async evaluate(){
   const brain=new Brain();brain.readState();await new Promise(resolve=>{release=resolve;});wall=500;brain.readState();return {return:0};
  }}),
 });
 try{
  await worker.handle({type:'observe-brain',enabled:true,indices:[0]});await worker.handle({type:'initialize'});
  const evaluation=worker.handle({type:'evaluate',id:1,job:{jobId:'accepted'}});
  await worker.handle({type:'evaluate',id:2,job:{jobId:'rejected'}});await worker.handle({type:'initialize',id:3});
  assert.equal(worker.state.observation.jobId,'accepted');assert.deepEqual(messages.filter(m=>m.type==='brain').map(m=>m.snapshot.jobId),['accepted']);
  await worker.handle({type:'cancel'});assert.equal(worker.state.observation.jobId,null);release();await evaluation;
  assert.equal(messages.filter(m=>m.type==='brain').length,1);
 }finally{worker.dispose();}
});
