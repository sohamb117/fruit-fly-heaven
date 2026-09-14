import test from 'node:test';
import assert from 'node:assert/strict';
import {createTrainingBrainResources} from '../training/brain-resources.js';
import {createWasmCore,WasmBrain} from '../../packages/banc-runtime/src/wasm.js';
import {fixture} from '../../packages/banc-runtime/test/fixture.mjs';

function fakeBrain(shared=null){
  return {shared,state:[0],disposed:false,disposeCount:0,dispose(){if(!this.disposed){this.disposed=true;this.disposeCount++;}}};
}

test('explicit WASM never attempts GPU or fallback and keeps fresh per-episode state',async()=>{
  const created=[];let gpuCalls=0,fallbackCalls=0;
  const resources=createTrainingBrainResources({model:{},backend:'wasm',
    createGPU:async()=>{gpuCalls++;throw new Error('GPU must not be touched');},
    createWasm:(_model,options)=>{const brain=fakeBrain(options?.shared);created.push(brain);return brain;},
    onFallback:()=>fallbackCalls++});
  const first=await resources.create();first.state[0]=99;first.dispose();
  const second=await resources.create();assert.deepEqual(second.state,[0]);
  assert.equal(resources.backend,'wasm');assert.equal(first.shared,created[0]);assert.equal(second.shared,created[0]);
  assert.equal(gpuCalls,0);assert.equal(fallbackCalls,0);assert.equal(created.length,3);
  second.dispose();resources.dispose();assert(created.every(brain=>brain.disposed));
});

test('explicit backend failures never silently select another engine',async()=>{
  for(const backend of ['wasm','webgpu']){
    let gpuCalls=0,wasmCalls=0,fallbacks=0;
    const resources=createTrainingBrainResources({model:{},backend,
      createGPU:async()=>{gpuCalls++;throw new Error('GPU failed');},
      createWasm:()=>{wasmCalls++;throw new Error('WASM failed');},onFallback:()=>fallbacks++});
    await assert.rejects(resources.create(),backend==='wasm'?/WASM failed/:/GPU failed/);
    assert.equal(gpuCalls,backend==='webgpu'?1:0);assert.equal(wasmCalls,backend==='wasm'?1:0);assert.equal(fallbacks,0);resources.dispose();
  }
  assert.throws(()=>createTrainingBrainResources({backend:'unknown'}),/Unknown training neural backend/);
});

test('episodes get fresh states while one GPU anchor survives until final disposal',async()=>{
  const model={},created=[];
  const resources=createTrainingBrainResources({model,createGPU:async(value,options)=>{
    assert.equal(value,model);assert.equal(options?.shared?.disposed??false,false);
    const brain=fakeBrain(options?.shared);created.push(brain);return brain;
  },createWasm:()=>{throw new Error('Unexpected fallback');}});
  const first=await resources.create();first.state[0]=73;first.dispose();
  const second=await resources.create();assert.equal(resources.backend,'webgpu');
  assert.notEqual(first,second);assert.deepEqual(second.state,[0]);
  assert.equal(first.shared,created[0]);assert.equal(second.shared,created[0]);
  assert.equal(created[0].disposed,false);assert.equal(created.length,3);
  second.dispose();resources.dispose();resources.dispose();
  assert.equal(created[0].disposeCount,1);await assert.rejects(resources.create(),/disposed/);
});

test('an established GPU backend never silently falls back after a later failure',async()=>{
  const anchor=fakeBrain();let gpuCalls=0,fallbacks=0;
  const resources=createTrainingBrainResources({model:{},createGPU:async()=>{
    if(++gpuCalls===1)return anchor;
    throw new Error('device lost during child creation');
  },createWasm:()=>{fallbacks++;return fakeBrain();}});
  await assert.rejects(resources.create(),/device lost/);assert.equal(resources.backend,'webgpu');
  assert.equal(fallbacks,0);resources.dispose();assert.equal(anchor.disposeCount,1);
});

test('disposal during asynchronous GPU anchor creation cleans up the late resource',async()=>{
  let resolve;const anchor=fakeBrain(),pending=new Promise(r=>{resolve=r;});
  const resources=createTrainingBrainResources({model:{},createGPU:()=>pending,createWasm:()=>{throw new Error('Unexpected fallback');}});
  const creation=resources.create();resources.dispose();resolve(anchor);
  await assert.rejects(creation,/disposed/);assert.equal(anchor.disposeCount,1);
});

test('disposal during asynchronous episode creation cleans up anchor and late child',async()=>{
  let resolve,started;const ready=new Promise(r=>{started=r;}),pending=new Promise(r=>{resolve=r;}),anchor=fakeBrain(),child=fakeBrain(anchor);
  const resources=createTrainingBrainResources({model:{},createGPU:async(_model,options)=>{
    if(!options?.shared)return anchor;started();return pending;
  },createWasm:()=>{throw new Error('Unexpected fallback');}});
  const creation=resources.create();await ready;resources.dispose();resolve(child);
  await assert.rejects(creation,/disposed/);assert.equal(anchor.disposeCount,1);assert.equal(child.disposeCount,1);
});

test('actual WASM fallback shares only graph wiring and starts each episode bit-identically',async()=>{
  const core=await createWasmCore(),model=fixture({n:2,edges:[{source:0,post:1,weight:20,receptor:0}]}),created=[];
  let attempts=0;
  const resources=createTrainingBrainResources({model,createGPU:async()=>{attempts++;throw new Error('Fixture has no GPU');},
    createWasm:(value,options)=>{const brain=new WasmBrain(core,value,options);created.push(brain);return brain;}});
  const reference=new WasmBrain(core,model),input=new Float32Array([30,0]),internal={hunger:.65,insulin:0,akh:.65};
  let first,second;
  try{
    first=await resources.create();assert.equal(resources.backend,'wasm');
    assert.deepEqual(first.readState(),reference.readState());first.step(64,input,internal);first.dispose();
    second=await resources.create();assert.equal(attempts,1);
    assert.equal(second.tick,0);assert.equal(second.edges,created[0].edges);assert.notEqual(second.states[0],created[0].states[0]);
    assert.deepEqual(second.readState(),reference.readState());
    second.step(64,input,internal);reference.step(64,input,internal);
    assert.deepEqual(second.readState(),reference.readState());
    assert.equal(created[0].tick,0);assert.equal(created[0].disposed,false);
  }finally{first?.dispose();second?.dispose();reference.dispose();resources.dispose();}
  assert(created.every(brain=>brain.disposed));
});
