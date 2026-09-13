import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createWasmCore,WasmBrain,WasmMuscles,WasmJoints} from '../src/wasm.js';
import {fixture} from './fixture.mjs';
import {validateModel} from '../src/model.js';
const core=await createWasmCore();
const internal={hunger:0,insulin:0,akh:0};
test('batched trace retains every timestep across odd and even state-buffer swaps',()=>{
  const model=fixture({n:3,edges:[{source:0,post:1,weight:2,receptor:0}]}),batched=new WasmBrain(core,model),single=new WasmBrain(core,model);
  const input=new Float32Array([35,20,0]);
  for(const steps of [3,4,1,17]){
    const trace=batched.step(steps,input,internal,true,{traceIndex:1});
    for(let k=0;k<steps;k++){
      single.step(1,input,internal);const state=single.readState(new Uint32Array([1]));
      assert.deepEqual(Array.from(trace.subarray(k*8,k*8+8)),Array.from(state));
    }
    assert.deepEqual(batched.readState(),single.readState());
  }
  assert.throws(()=>batched.step(1,input,internal,true,{traceIndex:3}),/trace/);
  batched.dispose();single.dispose();
});
test('shared immutable WASM wiring keeps inputs/state independent and outlives the first brain',()=>{
  const model=fixture({n:3,edges:[{source:0,post:1,weight:2,receptor:0}]}),first=new WasmBrain(core,model);
  const second=new WasmBrain(core,model,{shared:first}),reference=new WasmBrain(core,model);
  const quiet=new Float32Array(3),drive=new Float32Array([35,0,0]);
  first.step(64,drive,internal);second.step(64,quiet,internal);reference.step(64,quiet,internal);
  assert(first.readState().totalSpikes>0);assert.deepEqual(second.readState(),reference.readState());
  first.dispose();second.step(64,drive,internal);reference.step(64,drive,internal);
  assert.deepEqual(second.readState(),reference.readState());
  const observed=second.readState(new Uint32Array([0]),{includeSpikeTime:true});assert(observed[8]>0);assert(observed[8]<=second.timeMs);
  assert.equal(observed.spikes.filter(([,i])=>i===0).length,observed[3]);assert(observed.spikes.filter(([,i])=>i===0).length>1);
  second.dispose();reference.dispose();
});
function run(brain,input,steps){while(steps>0){const k=Math.min(128,steps);brain.step(k,input,internal);steps-=k;}return brain.readState();}

test('passive membrane matches the independently derived discrete RC solution',()=>{
  const b=new WasmBrain(core,fixture({n:1})),input=new Float32Array([5]);b.step(100,input,internal);
  const expected=-60+5*(1-Math.pow(1+.5/20,-100));
  assert(Math.abs(b.readState()[0]-expected)<1e-4);assert.equal(b.readState()[3],0);b.dispose();
});

test('spiking and graded cells have distinct release and spike-count semantics',()=>{
  const b=new WasmBrain(core,fixture({n:2,graded:[1]}));
  const s=run(b,new Float32Array([40,40]),400);
  assert(s[3]>0);assert.equal(s[11],0);assert(s[13]>0&&s[13]<.1);assert(s.every(Number.isFinite));b.dispose();
});
test('synaptic delay and receptor rise/decay persist beyond presynaptic activity',()=>{
  const b=new WasmBrain(core,fixture({n:2,edges:[{source:0,post:1,weight:8,delay:4}]}));
  b.step(1,new Float32Array([1000,0]),internal);assert.equal(b.readState()[3],1);
  b.step(3,new Float32Array(2),internal);assert.equal(b.readState()[14],1);
  b.step(1,new Float32Array(2),internal);assert(b.readState()[14]>1);
  b.step(15,new Float32Array(2),internal);assert(b.readState()[14]>1);
  b.step(100,new Float32Array(2),internal);assert(Math.abs(b.readState()[14]-1)<1e-5);b.dispose();
});
test('postsynaptic receptor reversal changes excitation to inhibition',()=>{
  const values=[0,1,2,3,4].map(receptor=>{
    const b=new WasmBrain(core,fixture({n:2,graded:[0,1],edges:[{source:0,post:1,weight:20,receptor}]}));
    const s=run(b,new Float32Array([30,0]),500);b.dispose();return s[8];
  });
  assert(values[0]>-50&&values[4]>-50);assert(values[1]<-65&&values[2]<-65&&values[3]<-65);
});
test('electrical coupling is symmetric, independently disconnectable and subthreshold',()=>{
  const m=fixture({n:2,gaps:[{a:0,b:1,weight:2}]});
  const a=new WasmBrain(core,m),b=new WasmBrain(core,m),off=new WasmBrain(core,m);
  run(a,new Float32Array([10,0]),300);run(b,new Float32Array([0,10]),300);
  off.step(100,new Float32Array([10,0]),internal,false);
  assert(Math.abs(a.readState()[0]-b.readState()[8])<1e-5);assert(a.readState()[8]>-60);assert.equal(off.readState()[8],-60);
  for(const brain of [a,b,off])brain.dispose();
});
test('monoamines require configured receptor sensitivity; hormones affect configured cells',()=>{
  const testRun=gain=>{const b=new WasmBrain(core,fixture({n:2,graded:[0,1],edges:[{source:0,post:1,weight:20,receptor:6}],overrides:{1:{13:gain}}}));const s=run(b,new Float32Array([30,10]),3000);b.dispose();return s[8];};
  assert(testRun(2)>testRun(0)+3);
  const m=fixture({n:2,graded:[0,1],overrides:{1:{15:10}}}),b=new WasmBrain(core,m);
  b.step(100,new Float32Array(2),{hunger:1,akh:1,insulin:0});const s=b.readState();assert(s[8]>s[0]+5);b.dispose();
});
test('independent state, chunk invariance and invalid inputs',()=>{
  const m=fixture({edges:[{source:0,post:1,weight:4}]}),a=new WasmBrain(core,m),b=new WasmBrain(core,m),input=new Float32Array([40,0,0,0]);
  a.step(40,input,internal);for(let i=0;i<40;i++)b.step(1,input,internal);assert.deepEqual(a.readState(),b.readState());
  assert.throws(()=>a.step(1,new Float32Array([NaN,0,0,0]),internal),/input/);
  const broken=fixture();broken.offsets[1]=3;assert.throws(()=>validateModel(broken),/CSR/);
  a.dispose();assert.throws(()=>a.readState(),/disposed/);b.dispose();
});
test('muscle activation, energy dependence, fatigue and passive decay',()=>{
  const m=new WasmMuscles(core,2),input=new Float32Array([1,1,0,1,1,1,1,0,1,.5]);
  let s;for(let i=0;i<100;i++)s=m.step(input,.01);assert(s[0]>.99);assert(Math.abs(s[2]/s[5]-2)<1e-5);assert(s[1]>0);
  input[0]=input[5]=0;for(let i=0;i<100;i++)s=m.step(input,.01);assert(s[2]<1e-7);m.dispose();assert.throws(()=>m.step(input,.01),/disposed/);
});
test('distilled joints remain passive at rest and bounded under sustained torque',()=>{
  const joints=[{inertia:.001,damping:.1,stiffness:.2,neutral:.1,range:[-1,1]}],j=new WasmJoints(core,joints);
  assert(Math.abs(j.step(new Float32Array([0]),.001)[0]-.1)<1e-6);
  let s;for(let i=0;i<1000;i++)s=j.step(new Float32Array([10]),.001);assert(s[0]<=.450001&&s[1]===0);j.dispose();
});
