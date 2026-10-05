import test from 'node:test';
import assert from 'node:assert/strict';
import {createWingEventExcitation} from '../flybody-wing-event-excitation.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';
import {createWingMotorEventReader} from '../../packages/banc-runtime/src/motor-events.js';
const muscles=[];let id=0;
for(const side of ['left','right']){
 for(const target of STEERING_MUSCLE_TYPES)muscles.push({joint:'wing_steer_'+side,kind:'wing_steering_assumption',target,sign:1,indices:[id++]});
 for(const [target,count]of [['dorsal_longitudinal_muscle',5],['dorsoventral_muscle',7]])muscles.push({joint:'wing_power_'+side,kind:'asynchronous_wing',target,sign:1,indices:Array.from({length:count},()=>id++)});
}
const io={muscles},indices=Uint32Array.from({length:48},(_,i)=>i),dlm=muscles.filter(m=>m.target==='dorsal_longitudinal_muscle').flatMap(m=>m.indices),unit=dlm[0];
const eventContract={schema:1,profile:'dlm-snl-2023-v1',indices:dlm};
function setup(){
 const params=new Float32Array(48*16);for(let i=0;i<48;i++)params[i*16+5]=2;
 const reader=createWingMotorEventReader({indices,params,dtMs:.5,bodyBlockMs:2,eventContract});
 const adapter=createWingEventExcitation({io,eventContract});const state=new Float32Array(48*9);
 for(let i=0;i<48;i++)state[i*9+8]=-1e30;
 adapter.accept(reader.read(state,0));return{reader,adapter,state};
}
test('0.1 ms DLM events pass from float32 reader into precise effector intervals and snapshots',()=>{
 const {reader,adapter,state}=setup();const clone=createWingEventExcitation({io,eventContract});
 for(let t=2;t<=12;t+=2){
  if(t===2||t===12){state[unit*9+3]++;state[unit*9+8]=Math.fround(t===2?.1:10.1);}
  const p=reader.read(state,t);adapter.accept(p);clone.restore(adapter.snapshot());
  assert.deepEqual(adapter.finishInterval(t-1),clone.finishInterval(t-1));
  clone.restore(adapter.snapshot());assert.deepEqual(adapter.finishInterval(t),clone.finishInterval(t));
 }
 assert.equal(adapter.snapshot().schemaVersion,2);assert.equal(adapter.readState().lastEventTimesMs[unit],10.1);
 assert(adapter.readState().decayState[unit]>0);assert.equal(adapter.readState().decayState[0],0);
});
test('DLM substep phase affects excitation within the same 1 ms interval',()=>{
 const run=time=>{
  const {reader,adapter,state}=setup();state[unit*9+3]=1;state[unit*9+8]=Math.fround(time);adapter.accept(reader.read(state,2));return adapter.finishInterval(1).unitExcitation[unit];
 };
 assert(run(.1)>run(.9));assert.equal(run(1),0);
});
test('mixed timing contracts reject incorrect guard, wrong family, aliases, and snapshot histories transactionally',()=>{
 const {reader,adapter,state}=setup();state[unit*9+3]=1;state[unit*9+8]=Math.fround(.1);adapter.accept(reader.read(state,2));adapter.finishInterval(1);adapter.finishInterval(2);
 const before=adapter.snapshot();
 const badContract={...eventContract,indices:[0]};assert.throws(()=>createWingEventExcitation({io,eventContract:badContract}),/DLM/);
 const p={initialized:false,fromTimeMs:2,timeMs:4,indices,counts:new Uint32Array(48),ratesHz:new Float32Array(48),events:[{index:unit,timeMs:3.1}]};p.counts[unit]=2;
 assert.throws(()=>adapter.accept(p),/interval/);assert.deepEqual(adapter.snapshot(),before);
 for(const mutate of [s=>s.schemaVersion=1,s=>s.contract.eventContract.indices[0]=0,s=>s.lastEventTimesMs[unit]=Math.fround(.1),s=>s.counts[unit]=2]){
  const s=structuredClone(before);mutate(s);assert.throws(()=>adapter.restore(s));assert.deepEqual(adapter.snapshot(),before);
 }
 assert.throws(()=>createWingEventExcitation({io}).restore(before),/contract/);
});
test('no opt-in keeps the existing exact-half-millisecond snapshot schema and rejects finer events',()=>{
 const a=createWingEventExcitation({io});assert.equal(a.snapshot().schemaVersion,1);assert(!Object.hasOwn(a.snapshot().contract,'eventContract'));
 const {adapter}=setup();assert.throws(()=>a.restore(adapter.snapshot()),/contract/);
});
