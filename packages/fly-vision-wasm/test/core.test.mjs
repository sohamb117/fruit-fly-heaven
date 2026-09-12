import test from 'node:test';
import assert from 'node:assert/strict';
import {createVisionModule} from '../dist/index.js';

const graph=()=>({rowOffsets:new Uint32Array([0,0,1,3]),sources:new Uint32Array([0,0,1]),
  weights:new Float32Array([.3,-.2,.7]),bias:new Float32Array([.1,.2,.3]),
  tauSeconds:new Float32Array([.05,.1,.02]),inputIndices:new Uint32Array([0]),dtSeconds:.01});

test('graded WASM matches an independent dense Euler reference including inhibition',async()=>{
  const r=await createVisionModule(),g=graph(),m=r.createModel(g),n=m.createNetwork();
  let state=Array.from(g.bias);
  for(let k=0;k<200;k++){
    const input=k<70?.04:k<130?-.2:.1,rect=state.map(x=>Math.max(0,x));
    state=state.map((x,i)=>x+.01/Math.max(g.tauSeconds[i],.01)*(-x+g.bias[i]+(i===0?input:i===1?.3*rect[0]:-.2*rect[0]+.7*rect[1])));
    n.step(new Float32Array([input]));
    n.readActivations().forEach((x,i)=>assert.ok(Math.abs(x-state[i])<1e-6));
  }
  n.dispose();m.dispose();
});

test('subthreshold signals transmit, 100 networks stay independent, and reads cannot mutate state',async()=>{
  const r=await createVisionModule(),m=r.createModel(graph()),ns=Array.from({length:100},()=>m.createNetwork());
  ns.forEach((n,i)=>n.step(new Float32Array([i===0?.03:0]),20));
  assert.ok(ns[0].readActivations()[1]>ns[1].readActivations()[1]);
  const before=ns[0].readActivations();ns[0].readActivations().fill(999);assert.deepEqual(ns[0].readActivations(),before);
  m.dispose();assert.throws(()=>m.createNetwork(),/disposed/);ns[0].step(new Float32Array([0]));
  ns[0].reset();assert.deepEqual(ns[0].readActivations(),graph().bias);assert.equal(ns[0].timeSeconds,0);
  ns.forEach(n=>n.dispose());assert.throws(()=>ns[0].step(new Float32Array([0])),/disposed/);
});

test('malformed graphs and inputs are rejected before calling native code',async()=>{
  const r=await createVisionModule();
  for(const change of [{sources:new Uint32Array([99,0,1])},{rowOffsets:new Uint32Array([0,2,1,3])},{tauSeconds:new Float32Array([.1,0,.1])},{weights:new Float32Array([NaN,0,1])},{inputIndices:new Uint32Array([3])}])assert.throws(()=>r.createModel({...graph(),...change}),TypeError);
  const m=r.createModel(graph()),n=m.createNetwork();
  for(const input of [new Float32Array(),new Float32Array([Infinity]),[1]])assert.throws(()=>n.step(input),TypeError);
  assert.throws(()=>n.step(new Float32Array([0]),.5),TypeError);n.dispose();m.dispose();
});
