import fs from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './fixture.mjs';
import {validateModel,initialBuffers} from '../src/model.js';
import {intrinsicLayout,decodeDlmEventTime,DLM_CELLS} from '../src/cell-models.js';
import {createWasmCore,WasmBrain} from '../src/wasm.js';
import {createWingMotorEventReader} from '../src/motor-events.js';
const core=await createWasmCore();
export function dlmFixture(options={}){
 const m=fixture(options);m.manifest.intrinsic_models={schema:1,profile:'dlm-snl-2023-v1',cells:[{index:1,root_id:'fixture1'}],ionic_step_ms:.1,initial_gates:{h:.146,b:.146},event_policy:'threshold-10ms-guard'};return m;
}
test('DLM selection validates explicit profile, frozen priors, IDs, spiking and no invented gaps',()=>{
 for(const change of [m=>m.manifest.intrinsic_models.extra=1,m=>m.manifest.intrinsic_models.initial_gates.h=.2,
  m=>m.manifest.intrinsic_models.profile='other',m=>m.manifest.intrinsic_models.cells.push({index:1,root_id:'x'}),
  m=>m.manifest.fixture=false,m=>m.manifest.dt_ms=.1,m=>m.params[24]=1]){
  const m=dlmFixture();change(m);assert.throws(()=>validateModel(m));
 }
 assert.throws(()=>validateModel(dlmFixture({gaps:[{a:0,b:1,weight:.04}]})),/gap/);
 assert.equal(validateModel(dlmFixture()).manifest.intrinsic_models.profile,'dlm-snl-2023-v1');
});
test('legacy buffer layout stays exact; optional slots are immutable and ionic state sparse',()=>{
 const m=fixture(),legacy=initialBuffers(m);assert.equal(legacy.packed.length,4*17+27);assert.equal(legacy.kinetics.length,4*19);
 const selected=dlmFixture(),layout=intrinsicLayout(selected),a=initialBuffers(selected),b=initialBuffers(selected);
 assert.equal(layout.count,1);assert.equal(a.kinetics.length,4*19+4);assert.deepEqual(Array.from(a.packed.slice(4*17+27)),[0,1,0,0]);
 assert.deepEqual(a.packed.slice(0,4*17+27),legacy.packed);assert.deepEqual(a.state,legacy.state);
 assert.deepEqual(Array.from(a.kinetics.slice(4*19)),[Math.fround(.146),Math.fround(.146),0,0]);
 a.kinetics[4*19]=0;assert.equal(b.kinetics[4*19],Math.fround(.146));assert(Object.isFrozen(layout.eventContract.indices));
});
test('separate native entry preserves every unselected state value bitexact',()=>{
 const regular=new WasmBrain(core,fixture()),selected=new WasmBrain(core,dlmFixture());
 try{for(let i=0;i<20;i++){
  const input=Float32Array.from([40,108.75,25,0]);regular.step(5,input);selected.step(5,input);
  assert.deepEqual(selected.readState([0,2,3]).slice(),regular.readState([0,2,3]).slice());
 }}finally{regular.dispose();selected.dispose();}
});
test('ionic state is per-brain, shared graph lifecycle and chunking remain independent',()=>{
 const m=dlmFixture(),a=new WasmBrain(core,m),b=new WasmBrain(core,m,{shared:a}),c=new WasmBrain(core,m,{shared:a});
 try{
  const input=Float32Array.from([0,108.75,0,0]);a.step(7,input);for(let i=0;i<7;i++)b.step(1,input);
  assert.deepEqual(a.readState([1],{includeSpikeTime:true}),b.readState([1],{includeSpikeTime:true}));assert.deepEqual(a.readIntrinsicState(),b.readIntrinsicState());
  assert.equal(c.readState([1])[0],-60);assert.equal(c.readIntrinsicState()[0],Math.fround(.146));
  a.dispose();b.step(1,input);assert.equal(b.timeMs,4);
 }finally{a.dispose();b.dispose();c.dispose();}
});
test('the opt-in membrane is continuous, with no LIF reset, adaptation, or physical refractory clamp',()=>{
 const a=new WasmBrain(core,dlmFixture());let previous=0,events=0;
 try{
  for(let i=0;i<1000;i++){
   a.step(1,Float32Array.from([0,108.75,0,0]));const s=a.readState([1],{includeSpikeTime:true});
   assert.equal(s[1],0);assert.equal(s[2],0);assert(s.every(Number.isFinite));
   if(s[3]>previous){events++;assert.notEqual(s[0],-60);assert.equal(s[5],2);decodeDlmEventTime(s[8]);}
   previous=s[3];
  }
  assert(events>0);assert.equal(a.readIntrinsicState()[3],0);
 }finally{a.dispose();}
});
test('intrinsic numerical failure poisons only its brain and cannot return an accepted state',()=>{
 const m=dlmFixture(),a=new WasmBrain(core,m),b=new WasmBrain(core,m,{shared:a});
 try{
  core.HEAPF32[a.kinetics/4+a.n*19]=NaN;
  assert.throws(()=>a.step(1,new Float32Array(4)),/ionic integration failed/);assert.throws(()=>a.readState([1]),/ionic integration failed/);
  b.step(1,new Float32Array(4));assert(b.readState([1]).every(Number.isFinite));
 }finally{a.dispose();b.dispose();}
});
test('decimal event storage decodes only uniquely represented 0.1 ms grid values',()=>{
 for(const t of [.1,.3,10.1,33.4,1234.7])assert.equal(decodeDlmEventTime(Math.fround(t)),t);
 for(const t of [NaN,Infinity,-1,.100001,2**21])assert.throws(()=>decodeDlmEventTime(t));
});
test('mixed reader retains substep time and rejects an unflagged fractional event',()=>{
 const m=dlmFixture(),contract=intrinsicLayout(m).eventContract,reader=createWingMotorEventReader({indices:[0,1],params:m.params,dtMs:.5,bodyBlockMs:2,eventContract:contract});
 const s=new Float32Array(18);s[8]=s[17]=-1e30;reader.read(s,0);s[12]=1;s[17]=Math.fround(.1);
 assert.deepEqual(reader.read(s,2).events,[{index:1,timeMs:.1}]);
 s[3]=1;s[8]=Math.fround(2.1);assert.throws(()=>reader.read(s,4),/0.5 ms/);
});

test('timestamp rounding bound accepts 2/3 ULP, rejects 4 ULP and overlapping large-clock neighbourhoods',()=>{
 const base=Math.fround(.9),ulp=2**-24;
 assert.equal(decodeDlmEventTime(Math.fround(base+2*ulp)),.9);
 assert.equal(decodeDlmEventTime(Math.fround(base+3*ulp)),.9);
 assert.throws(()=>decodeDlmEventTime(Math.fround(base+4*ulp)));
 assert.throws(()=>decodeDlmEventTime(Math.fround(262144.1)));
 assert.throws(()=>decodeDlmEventTime(Math.fround(.95)));
});

test('opted-in source releases through existing chemical history and receptor delay',()=>{
 const m=dlmFixture({edges:[{source:1,post:2,weight:2,receptor:0,delay:3}]});
 const a=new WasmBrain(core,m);let launchTick=null,arrivalTick=null;
 try{
  for(let tick=0;tick<1000;tick++){
   a.step(1,Float32Array.from([0,108.75,0,0]));const state=a.readState([1,2]);
   if(launchTick===null&&state[3]>0)launchTick=tick;
   if(arrivalTick===null&&state[14]>1)arrivalTick=tick;
   if(arrivalTick!==null)break;
  }
  assert.notEqual(launchTick,null);assert.equal(arrivalTick-launchTick,3);
 }finally{a.dispose();}
});
test('incoming chemical drive changes the ionic membrane through actual receptor kinetics',()=>{
 const wired=dlmFixture({edges:[{source:0,post:1,weight:2,receptor:0,delay:3}]}),empty=dlmFixture();
 const a=new WasmBrain(core,wired),b=new WasmBrain(core,empty);const input=Float32Array.from([100,0,0,0]);
 try{
  a.step(80,input);b.step(80,input);assert.notEqual(a.readState([1])[0],b.readState([1])[0]);assert.notEqual(a.readIntrinsicState()[0],b.readIntrinsicState()[0]);
 }finally{a.dispose();b.dispose();}
});
test('an edited opt-in declaration cannot silently reuse a cached layout or shared graph',()=>{
 const m=dlmFixture(),a=new WasmBrain(core,m);try{
  m.manifest.intrinsic_models.cells[0].index=2;
  assert.throws(()=>a.step(1,new Float32Array(4)),/declaration changed/);assert.throws(()=>new WasmBrain(core,m,{shared:a}),/declaration changed/);
  assert.equal(intrinsicLayout(m).cells[0].index,2);
 }finally{a.dispose();}
});

test('actual DLM IO/root-ID joins are mandatory outside bounded synthetic fixtures',()=>{
 const m=fixture({n:175401});m.manifest.fixture=false;
 m.io=JSON.parse(fs.readFileSync(new URL('../../../data/prepared/banc888/io.json',import.meta.url)));
 m.manifest.intrinsic_models={schema:1,profile:'dlm-snl-2023-v1',cells:DLM_CELLS.map(x=>({...x})),ionic_step_ms:.1,initial_gates:{h:.146,b:.146},event_policy:'threshold-10ms-guard'};
 assert.equal(intrinsicLayout(m).count,10);
 const row=m.io.motor_neurons.find(x=>x.index===DLM_CELLS[0].index),saved=row.root_id;row.root_id='incorrect';assert.throws(()=>intrinsicLayout(m),/identity/);row.root_id=saved;
 const muscle=m.io.muscles.find(x=>x.target==='dorsal_longitudinal_muscle');muscle.root_ids[0]='incorrect';assert.throws(()=>intrinsicLayout(m),/join/);
 m.manifest.fixture=true;assert.throws(()=>intrinsicLayout(m),/64/);
});
