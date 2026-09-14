import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createWingMotorEventReader} from '../src/motor-events.js';
import {createWasmCore,WasmBrain} from '../src/wasm.js';
import {fixture} from './fixture.mjs';

const model=()=>fixture({n:4});
const reader=(overrides={})=>createWingMotorEventReader({indices:[2,0],params:model().params,dtMs:.5,bodyBlockMs:2,...overrides});
function state(rows){
 const output=new Float32Array(rows.length*9);
 rows.forEach(({count=0,time=-1e30,rate=0},k)=>{output[k*9+3]=count;output[k*9+4]=rate;output[k*9+8]=time;});
 return output;
}

test('first read establishes a baseline, preserves selected rates, and emits no historical spikes',()=>{
 const r=reader(),initial=r.read(state([{count:3,time:8,rate:37.5},{count:2,time:7.5,rate:12.25}]),10);
 assert.equal(initial.initialized,true);assert.equal(initial.fromTimeMs,null);assert.deepEqual(initial.events,[]);
 assert.deepEqual(Array.from(initial.indices),[2,0]);assert.deepEqual(Array.from(initial.ratesHz),[37.5,12.25]);
 const next=r.read(state([{count:4,time:12,rate:41},{count:2,time:7.5,rate:11}]),12);
 assert.equal(next.initialized,false);assert.equal(next.fromTimeMs,10);
 assert.deepEqual(next.events,[{index:2,timeMs:12}]);
 r.reset();assert.throws(()=>r.read(state([{count:1,time:.5}]),2),/nine/);
 const reset=r.read(state([{count:1,time:.5},{}]),2);assert(reset.initialized);assert.deepEqual(reset.events,[]);
});

test('constructor owns indices; returned arrays cannot change the reader baseline',()=>{
 const indices=[2,0],r=reader({indices});indices[0]=1;
 const baseline=r.read(state([{},{}]),0);baseline.indices[0]=3;baseline.counts[0]=99;baseline.ratesHz[0]=999;
 const next=r.read(state([{count:1,time:2,rate:1},{count:1,time:.5,rate:2}]),2);
 assert.deepEqual(Array.from(next.indices),[2,0]);
 assert.deepEqual(next.events,[{index:0,timeMs:.5},{index:2,timeMs:2}]);
 next.counts.fill(99);next.indices.fill(3);
 const quiet=r.read(state([{count:1,time:2,rate:.9},{count:1,time:.5,rate:1.9}]),4);
 assert.deepEqual(quiet.events,[]);assert.deepEqual(Array.from(quiet.counts),[1,1]);
});

test('same-time events sort by neuron index, independent of selection order or global ring',()=>{
 const r=reader();r.read(state([{},{}]),0);
 const sample=state([{count:1,time:2},{count:1,time:2}]);
 Object.defineProperty(sample,'spikes',{get(){throw new Error('Global ring must not be read');}});
 assert.deepEqual(r.read(sample,2).events,[{index:0,timeMs:2},{index:2,timeMs:2}]);
});

test('current-time boundary is included once; preceding boundary is never newly emitted',()=>{
 const r=reader();r.read(state([{},{}]),0);
 assert.deepEqual(r.read(state([{count:1,time:2},{}]),2).events,[{index:2,timeMs:2}]);
 assert.deepEqual(r.read(state([{count:1,time:2},{}]),4).events,[]);
 assert.throws(()=>r.read(state([{count:1,time:2},{count:1,time:4}]),6),/outside/);
 assert.deepEqual(r.read(state([{count:1,time:2},{count:1,time:6}]),6).events,[{index:0,timeMs:6}]);
});

test('rejects unsafe timing, graded neurons, weak refractory guarantees and invalid indices',()=>{
 for(const config of [{dtMs:1},{bodyBlockMs:2.5},{bodyBlockMs:1},{dtMs:.25}])assert.throws(()=>reader(config),/0.5 ms/);
 for(const indices of [[],[0,0],[-1],[4],[.5]])assert.throws(()=>reader({indices}),/indices/);
 assert.throws(()=>reader({indices:new Float32Array([0])}),/indices/);
 assert.throws(()=>reader({params:new Float32Array(17)}),/stride 16/);
 for(const value of [0,1.5,1.9999,NaN,Infinity]){
  const m=model();m.params[2*16+5]=value;assert.throws(()=>reader({params:m.params}),/refractory/);
 }
 const m=model();m.params[2*16+8]=1;assert.throws(()=>reader({params:m.params}),/spiking/);
});

test('rejects skipped, repeated, regressed and non-tick read times without advancing state',()=>{
 const r=reader();r.read(state([{},{}]),2);
 for(const time of [6,2,0,4.25,NaN,Infinity,2**23])assert.throws(()=>r.read(state([{},{}]),time),/interval|tick/);
 assert.deepEqual(r.read(state([{},{}]),4).events,[]);
});

test('rejects count regression, overrun, precision exhaustion and inconsistent timestamps transactionally',()=>{
 const r=reader();r.read(state([{count:1,time:2},{}]),2);
 const invalid=[
  [state([{},{}]),/regression/],
  [state([{count:3,time:4},{}]),/overrun/],
  [state([{count:1,time:2},{count:2**24,time:4}]),/2\^24/],
  [state([{count:1.5,time:2},{}]),/integer/],
  [state([{count:1,time:4},{}]),/without/],
  [state([{count:2,time:2},{}]),/outside/],
  [state([{count:2,time:4},{}]),/refractory interval/],
  [state([{count:1,time:2},{count:1,time:4.25}]),/tick/],
  [state([{count:1,time:2},{count:1,time:4.5}]),/history/],
  [state([{count:1,time:2},{count:0,time:0}]),/sentinel/],
  [state([{count:1,time:2},{count:1,time:0}]),/history/],
  [state([{count:1,time:2},{rate:-1}]),/Negative/],
  [state([{count:1,time:2},{rate:NaN}]),/Nonfinite/],
 ];
 for(const [sample,pattern]of invalid)assert.throws(()=>r.read(sample,4),pattern);
 assert.deepEqual(r.read(state([{count:1,time:2},{count:1,time:4}]),4).events,[{index:0,timeMs:4}]);
});

test('requires exact Float32 stride nine; explicit reset safely establishes a new baseline',()=>{
 const r=reader();
 assert.throws(()=>r.read(new Float32Array(16),0),/nine/);
 assert.throws(()=>r.read(new Float64Array(18),0),/nine/);
 r.read(state([{count:1,time:2},{}]),2);r.reset();
 const baseline=r.read(state([{},{}]),0);assert(baseline.initialized);assert.deepEqual(baseline.events,[]);
 assert.deepEqual(r.read(state([{},{}]),2).events,[]);
});

test('changed selected model declarations fail closed',()=>{
 const m=model(),r=reader({params:m.params});r.read(state([{},{}]),0);
 m.params[2*16+5]=1;assert.throws(()=>r.read(state([{},{}]),2),/params changed/);
 m.params[2*16+5]=2;assert.deepEqual(r.read(state([{},{}]),2).events,[]);
});

test('native four-neuron tick trace and block readout recover the same exact spike times',async()=>{
 const core=await createWasmCore(),m=model(),brain=new WasmBrain(core,m),indices=Uint32Array.from([2,0,3]);
 const r=createWingMotorEventReader({indices,params:m.params,dtMs:.5,bodyBlockMs:2});
 const input=new Float32Array([1000,0,1000,0]),internal={hunger:0,insulin:0,akh:0};
 let counts=new Float32Array(4),expected=[],allEvents=[];
 try{
  assert.deepEqual(r.read(brain.readState(indices,{includeSpikeTime:true}),brain.timeMs).events,[]);
  for(let tick=1;tick<=40;tick++){
   brain.step(1,input,internal);const eachTick=brain.readState();
   for(const index of indices){
    const count=eachTick[index*8+3];if(count>counts[index])expected.push({index,timeMs:brain.timeMs});counts[index]=count;
   }
   if(tick%4===0){
    expected.sort((a,b)=>a.timeMs-b.timeMs||a.index-b.index);
    const selected=brain.readState(indices,{includeSpikeTime:true}),decoded=r.read(selected,brain.timeMs);
    assert.deepEqual(decoded.events,expected);assert.deepEqual(Array.from(decoded.ratesHz),Array.from(indices,index=>eachTick[index*8+4]));
    allEvents.push(...decoded.events);expected=[];
   }
  }
  assert(allEvents.some(event=>event.timeMs===8),'Exercise an event exactly at a block boundary');
  assert(allEvents.every(event=>event.index!==3),'Quiet selected neurons must emit no event');
  for(const index of [0,2]){
   const times=allEvents.filter(event=>event.index===index).map(event=>event.timeMs);
   assert(times.length>3);assert.equal(times[0],.5);
   for(let k=1;k<times.length;k++)assert(times[k]-times[k-1]>=2.5);
  }
  // Creating a reader after accumulated real spikes must not replay them.
  const late=createWingMotorEventReader({indices,params:m.params,dtMs:.5,bodyBlockMs:2});
  assert.deepEqual(late.read(brain.readState(indices,{includeSpikeTime:true}),brain.timeMs).events,[]);
 }finally{brain.dispose();}
});
