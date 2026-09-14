import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {fixture} from './fixture.mjs';
import {createWasmCore,WasmBrain} from '../src/wasm.js';
import {WebGPUBrain} from '../src/webgpu.js';
import {DLM_MAX_TIME_MS} from '../src/cell-models.js';

const internal={hunger:.7,insulin:.2,akh:.4};
function modelFor(dlm){
  const model=fixture({n:6,graded:[2],edges:[
    {source:0,post:1,weight:.03,receptor:0,delay:3},
    {source:1,post:3,weight:.7,receptor:0,delay:7},
    {source:2,post:4,weight:.8,receptor:1,delay:2},
    {source:3,post:5,weight:.5,receptor:2,delay:31},
    {source:0,post:5,weight:.4,receptor:5,delay:5},
  ],gaps:[{a:0,b:4,weight:.08}],overrides:{0:{11:2,12:-1,13:1},4:{11:-1,12:2,13:.5}}});
  if(dlm)model.manifest.intrinsic_models={schema:1,profile:'dlm-snl-2023-v1',
    cells:[{index:1,root_id:'fixture1'}],ionic_step_ms:.1,
    initial_gates:{h:.146,b:.146},event_policy:'threshold-10ms-guard'};
  return model;
}
function inputAt(tick){return Float32Array.from([
  tick%11<5?65:0,108.75+7*Math.sin(tick*.037),
  18+12*Math.sin(tick*.13),tick%23<11?31:-9,10+tick%9,12,
]);}
async function snapshot(brain){return {tick:brain.tick,timeMs:brain.timeMs,
  state:await brain.readState(undefined,{includeSpikeTime:true}),ionic:brain.readIntrinsicState()};}

async function nonconstantParity(create,dlm,gaps){
  const model=modelFor(dlm),sequence=await create(model),single=await create(model,sequence);
  try{
    // Existing events and delay history must survive each sequence upload.
    const warmup=Float32Array.from([65,108.75,25,30,10,15]);
    await sequence.step(64,warmup,internal,gaps);await single.step(64,warmup,internal,gaps);
    const initial=await snapshot(single);assert(initial.state.totalSpikes>0);
    for(const count of [1,4,7,128,31,128,65,128]){
      const inputs=Array.from({length:count},(_,k)=>inputAt(sequence.tick+k));
      assert.equal(await sequence.stepSequence(inputs,internal,gaps),undefined);
      for(const input of inputs)await single.step(1,input,internal,gaps);
      assert.deepEqual(await snapshot(sequence),await snapshot(single));
    }
    const result=await snapshot(sequence);
    assert(result.state.totalSpikes>initial.state.totalSpikes);
    assert(result.state.spikes.length>initial.state.spikes.length);
    assert(result.state.spikes.some(([time])=>time<=initial.timeMs),'older events retained');
    assert(result.state[2*9+7]>0,'graded neuron has a nonzero output');
    assert(result.state[3*9+6]>0,'delayed excitatory receptor receives spikes');
    assert(result.state[4*9+6]>0,'graded source drives receptor kinetics');
    if(dlm){assert(result.state[1*9+3]>0,'ionic cell emits events');assert.notEqual(result.ionic[0],Math.fround(.146));}
  }finally{sequence.dispose();single.dispose();}
}

async function constantParity(create,dlm,gaps){
  const model=modelFor(dlm),sequence=await create(model),block=await create(model,sequence);
  try{
    for(const count of [1,4,7,128,3,128,128]){
      const input=Float32Array.from([48,108.75,27,35,15,18]);
      await sequence.stepSequence(Array.from({length:count},()=>input),internal,gaps);
      await block.step(count,input,internal,gaps);
      assert.deepEqual(await snapshot(sequence),await snapshot(block));
    }
  }finally{sequence.dispose();block.dispose();}
}

async function invalidSequences(create){
  const brain=await create(modelFor(true));
  try{
    await brain.step(31,inputAt(0),internal);const before=await snapshot(brain),valid=inputAt(1);
    const nan=valid.slice();nan[5]=NaN;const infinite=valid.slice();infinite[0]=Infinity;
    for(const input of [null,valid,[],Array(129).fill(valid),[Array.from(valid)],
      [new Float32Array(5)],[valid,nan],[valid,infinite],[valid,new Float64Array(6)]]){
      await assert.rejects(async()=>brain.stepSequence(input,internal));
      assert.deepEqual(await snapshot(brain),before);
    }
    for(const state of [{...internal,hunger:NaN},{...internal,insulin:2},{...internal,akh:-1}]){
      await assert.rejects(async()=>brain.stepSequence([valid,valid],state));
      assert.deepEqual(await snapshot(brain),before);
    }
    brain.tick=DLM_MAX_TIME_MS*2-1;
    await assert.rejects(async()=>brain.stepSequence([valid]),/timestamp precision/);
    assert.equal(brain.tick,DLM_MAX_TIME_MS*2-1);brain.tick=before.tick;
    brain.model.manifest.intrinsic_models.cells[0].index=3;
    await assert.rejects(async()=>brain.stepSequence([valid]),/declaration changed/);
    brain.model.manifest.intrinsic_models.cells[0].index=1;
    assert.deepEqual(await snapshot(brain),before);
    brain.dispose();await assert.rejects(async()=>brain.stepSequence([valid]),/disposed/);
  }finally{brain.dispose();}
}

test('WASM: nonconstant schedules preserve all readout/events and ionic state exactly',async t=>{
  const core=await createWasmCore(),create=(model,shared)=>new WasmBrain(core,model,{shared});
  for(const dlm of [false,true])for(const gaps of [false,true])await t.test(`ionic=${dlm}, gaps=${gaps}`,()=>nonconstantParity(create,dlm,gaps));
});
test('WASM: constant schedules equal the unchanged block step exactly',async t=>{
  const core=await createWasmCore(),create=(model,shared)=>new WasmBrain(core,model,{shared});
  for(const dlm of [false,true])for(const gaps of [false,true])await t.test(`ionic=${dlm}, gaps=${gaps}`,()=>constantParity(create,dlm,gaps));
});
test('WASM: validation is atomic and ionic failure poisons only its brain',async()=>{
  const core=await createWasmCore(),create=(model,shared)=>new WasmBrain(core,model,{shared});
  await invalidSequences(create);
  const model=modelFor(true),brain=create(model),sibling=create(model,brain);
  try{
    core.HEAPF32[brain.kinetics/4+brain.n*19]=NaN;
    assert.throws(()=>brain.stepSequence([inputAt(0),inputAt(1)]),/ionic integration failed/);
    assert.throws(()=>brain.readState(),/ionic integration failed/);assert.equal(brain.tick,0);
    sibling.stepSequence([inputAt(0)]);assert.equal(sibling.tick,1);
  }finally{brain.dispose();sibling.dispose();}
});

test('WASM: HEAP-backed future currents use values from sequence entry',async()=>{
  const core=await createWasmCore();
  for(const dlm of [false,true])for(const source of ['state','currents']){
    const model=modelFor(dlm),brain=new WasmBrain(core,model),single=new WasmBrain(core,model,{shared:brain});
    try{
      const offset=source==='state'?brain.states[1]/4:brain.params/4+brain.n*16;
      const aliased=core.HEAPF32.subarray(offset,offset+brain.n);
      const inputs=[inputAt(0),aliased,inputAt(2),aliased],entry=inputs.map(input=>input.slice());
      brain.stepSequence(inputs,internal);
      for(const input of entry)single.step(1,input,internal);
      assert.deepEqual(await snapshot(brain),await snapshot(single),`${source}, ionic=${dlm}`);
      assert.deepEqual(inputs[0],entry[0]);assert.deepEqual(inputs[2],entry[2]);
      if(source==='state')assert.notDeepEqual(aliased,entry[1],'earlier dispatch overwrites the future input view');
    }finally{brain.dispose();single.dispose();}
  }
});

// Optional native backend is supplied explicitly; the regular runtime gains no
// dependency on Dawn. The staging runner supplies the pinned Metal provider.
test('Dawn: exact sequence parity, upload budget, and lifecycle',{
  skip:!process.env.BANC_NATIVE_WEBGPU_BACKEND,
},async t=>{
  const {installNativeWebGPU}=await import(pathToFileURL(process.env.BANC_NATIVE_WEBGPU_BACKEND));
  const installed=await installNativeWebGPU(),previousFetch=globalThis.fetch;
  globalThis.fetch=async(url,...args)=>url instanceof URL&&url.protocol==='file:'?
    new Response(await fs.readFile(url)):previousFetch(url,...args);
  const create=(model,shared)=>WebGPUBrain.create(model,{gpu:installed.gpu,shared});
  try{
    t.diagnostic(JSON.stringify(installed.provenance));
    for(const dlm of [false,true])for(const gaps of [false,true]){
      await t.test(`nonconstant ionic=${dlm}, gaps=${gaps}`,()=>nonconstantParity(create,dlm,gaps));
      await t.test(`constant ionic=${dlm}, gaps=${gaps}`,()=>constantParity(create,dlm,gaps));
    }
    await t.test('invalid vectors are atomic; timestamps and declarations remain guarded',()=>invalidSequences(create));
    await t.test('old-compatible adapter limit allows block steps and short sequences; oversize rejects atomically',async()=>{
      // This fixture's graph fits 192 KiB, while its 128-tick schedule needs
      // 288 KiB. Expose a synthetic limit on an actual adapter/device: Dawn
      // may retain its larger default even when a smaller limit is requested.
      const adapter=await installed.gpu.requestAdapter({powerPreference:'high-performance'}),limit=192*1024;
      let requested;
      const gpu={requestAdapter:async()=>({
        info:adapter.info,
        limits:{maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:limit},
        async requestDevice(options){
          requested=options;
          const device=await adapter.requestDevice(options);
          assert(device.limits.maxBufferSize>=limit,'physical device fits the synthetic budget');
          const limits=new Proxy(device.limits,{get(target,key){
            return key==='maxBufferSize'?limit:Reflect.get(target,key,target);
          }});
          return new Proxy(device,{get(target,key){
            if(key==='limits')return limits;
            const value=Reflect.get(target,key,target);
            return typeof value==='function'?value.bind(target):value;
          }});
        },
      })};
      const model=fixture({n:512}),brain=await WebGPUBrain.create(model,{gpu});
      try{
        assert.equal(requested.requiredLimits.maxBufferSize,limit);
        assert.equal(brain.device.limits.maxBufferSize,limit);
        const input=new Float32Array(512).fill(48);
        await brain.step(4,input);await brain.stepSequence(Array(64).fill(input));
        const before=await snapshot(brain),buffer=brain.sequenceBuffer,bytes=brain.allocatedBytes;
        await assert.rejects(brain.stepSequence(Array(128).fill(input)),/GPU buffer limit too small.*128-tick/);
        assert.deepEqual(await snapshot(brain),before);
        assert.equal(brain.sequenceBuffer,buffer);assert.equal(brain.sequenceCapacity,64);
        assert.equal(brain.allocatedBytes,bytes);assert.equal(brain.busy,false);
        await brain.step(1,input);assert.equal(brain.tick,69);
      }finally{brain.dispose();}
    });
    await t.test('one upload, one submission, one completion; schedule allocation grows then reuses',async()=>{
      for(const dlm of [false,true]){
      const brain=await create(modelFor(dlm)),original=brain.device,counts={writes:0,submits:0,waits:0,maps:0,copies:[]};
      if(dlm){const map=brain.intrinsicStaging.mapAsync;Object.defineProperty(brain.intrinsicStaging,'mapAsync',{value(...args){counts.maps++;return map.apply(this,args);}});}
      const queue=new Proxy(original.queue,{get(target,key){const value=Reflect.get(target,key,target);return typeof value!=='function'?value:(...args)=>{
        if(key==='writeBuffer')counts.writes++;if(key==='submit')counts.submits++;if(key==='onSubmittedWorkDone')counts.waits++;
        return value.apply(target,args);
      };}});
      brain.device=new Proxy(original,{get(target,key){if(key==='queue')return queue;
        const value=Reflect.get(target,key,target);if(key==='createCommandEncoder')return (...args)=>{
          const encoder=value.apply(target,args);return new Proxy(encoder,{get(e,k){const f=Reflect.get(e,k,e);return typeof f!=='function'?f:(...a)=>{
            if(k==='copyBufferToBuffer')counts.copies.push(a);return f.apply(e,a);
          };}});
        };return typeof value==='function'?value.bind(target):value;
      }});
      try{
        const initialBytes=brain.allocatedBytes;assert.equal(brain.sequenceBuffer,undefined);
        let previous=null;
        for(const count of [4,3,7,5,128,1]){
          counts.writes=counts.submits=counts.waits=counts.maps=0;counts.copies=[];
          const previousCapacity=brain.sequenceCapacity??0;
          await brain.stepSequence(Array.from({length:count},(_,k)=>inputAt(k)));
          assert.equal(counts.writes,1);assert.equal(counts.submits,1);assert.equal(counts.waits,dlm?0:1);assert.equal(counts.maps,dlm?1:0);
          assert.equal(counts.copies.length,count+1+Number(dlm));
          assert.equal(counts.copies[0][2],brain.config);assert.equal(counts.copies[0][4],count*256);
          for(const copy of counts.copies.slice(1,count+1)){assert.equal(copy[2],brain.inputs);assert.equal(copy[3],0);assert.equal(copy[4],brain.n*4);}
          if(dlm){assert.equal(counts.copies.at(-1)[2],brain.intrinsicStaging);assert.equal(counts.copies.at(-1)[4],brain.intrinsic.count*16);}
          if(count<=previousCapacity)assert.equal(brain.sequenceBuffer,previous);
          assert(brain.sequenceCapacity<=128);assert(brain.sequenceCapacity>=count);
          assert.equal(brain.allocatedBytes,initialBytes+brain.sequenceBuffer.size);
          assert.equal(brain.buffers.filter(b=>b===brain.sequenceBuffer).length,1);previous=brain.sequenceBuffer;
        }
      }finally{brain.dispose();assert.equal(brain.sequenceValues,null);}
      }
    });
    await t.test('busy and device loss reject sequence operations; ionic failure is isolated',async()=>{
      const model=modelFor(true),brain=await create(model),sibling=await create(model,brain);
      try{
        const pending=brain.stepSequence([inputAt(0),inputAt(1)]);
        await assert.rejects(brain.stepSequence([inputAt(0)]),/Concurrent GPU operation/);
        await assert.rejects(brain.step(1,inputAt(0)),/Concurrent GPU operation/);
        await assert.rejects(brain.readState(),/Concurrent GPU operation/);
        await pending;assert.equal(brain.busy,false);assert.equal(brain.intrinsicStaging.mapState,'unmapped');
        brain.device.queue.writeBuffer(brain.kinetics,brain.n*19*4,Float32Array.from([NaN,.146,0,0]));
        await assert.rejects(brain.stepSequence([inputAt(0)]),/ionic/);
        assert.equal(brain.busy,false);assert.equal(brain.intrinsicStaging.mapState,'unmapped');
        await assert.rejects(brain.readState(),/ionic/);assert.equal(brain.tick,2);
        await sibling.stepSequence([inputAt(0)]);assert.equal(sibling.tick,1);
        const lost=new Error('fixture device loss');sibling.health.lost=lost;
        await assert.rejects(sibling.stepSequence([inputAt(0)]),lost);
      }finally{brain.dispose();sibling.dispose();}
    });
  }finally{globalThis.fetch=previousFetch;installed.uninstall();}
});
