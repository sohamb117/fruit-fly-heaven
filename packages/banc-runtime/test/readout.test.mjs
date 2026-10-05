import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {fixture} from './fixture.mjs';
import {createWasmCore,WasmBrain} from '../src/wasm.js';
import {WebGPUBrain} from '../src/webgpu.js';
import {createWingMotorEventReader} from '../src/motor-events.js';
import {SPIKE_CAPACITY} from '../src/model.js';

const selected=Uint32Array.from([2,0,3]);
const selectedOnly={includeSpikeTime:true,includeStatistics:false,includeSpikeHistory:false};
const bytes=values=>new Uint8Array(values.buffer,values.byteOffset,values.byteLength);
function modelFor(dlm){
  // Two reduction groups exercise the GPU's optional global statistics.
  const model=fixture({n:dlm?8:260});
  if(dlm)model.manifest.intrinsic_models={schema:1,profile:'dlm-snl-2023-v1',
    cells:[{index:2,root_id:'fixture2'}],ionic_step_ms:.1,
    initial_gates:{h:.146,b:.146},event_policy:'threshold-10ms-guard'};
  return model;
}

async function selectedParity(create,dlm){
  const model=modelFor(dlm),brain=await create(model),input=new Float32Array(model.manifest.neuron_count);
  input[0]=1000;input[1]=1000;input[2]=dlm?108.75:1000;
  const reader=()=>createWingMotorEventReader({indices:selected,params:model.params,dtMs:.5,bodyBlockMs:2,eventContract:brain.eventContract});
  const legacyReader=reader(),leanReader=reader();let eventCount=0;
  try{
    for(let block=0;block<=20;block++){
      if(block)await brain.step(4,input);
      const legacy=await brain.readState(selected,{includeSpikeTime:true}),lean=await brain.readState(selected,selectedOnly);
      assert.equal(lean.length,selected.length*9);assert.deepEqual(bytes(lean),bytes(legacy));
      for(const key of ['totalSpikes','activeEver','spikes'])assert.equal(Object.hasOwn(lean,key),false,key);
      const expected=legacyReader.read(legacy,brain.timeMs),actual=leanReader.read(lean,brain.timeMs);
      assert.deepEqual(actual,expected);eventCount+=actual.events.length;
      const full=await brain.readState();
      assert.equal(legacy.totalSpikes,full.totalSpikes);assert.equal(legacy.activeEver,full.activeEver);
      assert.equal(Object.hasOwn(full,'spikes'),false,'default stride eight keeps its legacy shape');
      if(block)assert(legacy.spikes.some(([,index])=>index===1),'default history includes an unselected neuron');
    }
    assert(eventCount>0,'the parity comparison includes actual selected events');
    const legacy=await brain.readState(selected,{includeSpikeTime:true});
    assert.deepEqual(await brain.readState(selected,{includeSpikeTime:true,includeStatistics:true,includeSpikeHistory:true}),legacy);
    for(const includeStatistics of [false,true])for(const includeSpikeHistory of [false,true]){
      const result=await brain.readState(selected,{includeSpikeTime:true,includeStatistics,includeSpikeHistory});
      assert.deepEqual(bytes(result),bytes(legacy));
      assert.equal(Object.hasOwn(result,'totalSpikes'),includeStatistics);
      assert.equal(Object.hasOwn(result,'activeEver'),includeStatistics);
      assert.equal(Object.hasOwn(result,'spikes'),includeSpikeHistory);
      if(includeStatistics){assert.equal(result.totalSpikes,legacy.totalSpikes);assert.equal(result.activeEver,legacy.activeEver);}
      if(includeSpikeHistory)assert.deepEqual(result.spikes,legacy.spikes);
    }
    const lean8=await brain.readState(selected,{includeStatistics:false,includeSpikeHistory:true});
    assert.equal(lean8.length,selected.length*8);assert.equal(Object.hasOwn(lean8,'spikes'),false);
    assert.equal(Object.hasOwn(lean8,'totalSpikes'),false);
    const snapshot=await brain.readState(selected,{includeSpikeTime:true});
    for(const flag of ['includeStatistics','includeSpikeHistory'])for(const value of [0,1,null,'false',{},new Boolean(false)]){
      await assert.rejects(async()=>brain.readState(selected,{[flag]:value}),/flags must be boolean/);
      assert.deepEqual(await brain.readState(selected,{includeSpikeTime:true}),snapshot);
    }
    return brain;
  }catch(error){brain.dispose();throw error;}
}

test('WASM selected event readout preserves values and packets while diagnostics remain opt-out',async t=>{
  const core=await createWasmCore();
  for(const dlm of [false,true])await t.test(`ionic=${dlm}`,async()=>{
    const brain=await selectedParity(model=>new WasmBrain(core,model),dlm);
    try{
      // A disabled history must not even attempt to create its native view.
      const events=brain.events;brain.events=core.HEAPU8.byteLength;
      try{assert.equal(brain.readState(selected,selectedOnly).length,27);assert.throws(()=>brain.readState(selected,{includeSpikeTime:true}),RangeError);}
      finally{brain.events=events;}
    }finally{brain.dispose();}
  });
});

// Optional hardware fixture, never a full model, body, coordinator, or trainer.
test('Dawn selected readout compiles, preserves events, and skips global output/history transfer',{
  skip:!process.env.BANC_NATIVE_WEBGPU_BACKEND,
},async t=>{
  const {installNativeWebGPU}=await import(pathToFileURL(process.env.BANC_NATIVE_WEBGPU_BACKEND));
  const installed=await installNativeWebGPU(),previousFetch=globalThis.fetch;
  globalThis.fetch=async(url,...args)=>url instanceof URL&&url.protocol==='file:'?
    new Response(await fs.readFile(url)):previousFetch(url,...args);
  try{
    for(const dlm of [false,true])await t.test(`ionic=${dlm}`,async()=>{
      const brain=await selectedParity(model=>WebGPUBrain.create(model,{gpu:installed.gpu}),dlm);
      try{
        await brain.readState(selected,selectedOnly);
        const lean=brain.readCaches.find(cache=>cache.stride===9&&!cache.includeStatistics&&!cache.eventBytes);
        assert.equal(lean.output.size,selected.length*9*4);assert.equal(lean.staging.size,lean.output.size);
        await brain.readState(selected,{includeSpikeTime:true});
        const legacy=brain.readCaches.find(cache=>cache.stride===9&&cache.includeStatistics&&cache.eventBytes);
        assert.equal(legacy.output.size,(selected.length*9+Math.ceil(brain.n/256)*2)*4);
        assert.equal(legacy.eventBytes,(2+SPIKE_CAPACITY*2)*4);
        assert.equal(legacy.staging.size,legacy.output.size+legacy.eventBytes);
        const original=brain.device,copies=[];
        brain.device=new Proxy(original,{get(target,key){
          const value=Reflect.get(target,key,target);
          if(key==='createCommandEncoder')return (...args)=>{
            const encoder=value.apply(target,args);
            return new Proxy(encoder,{get(object,method){
              const fn=Reflect.get(object,method,object);
              if(method==='copyBufferToBuffer')return (...values)=>{copies.push(values);return fn.apply(object,values);};
              return typeof fn==='function'?fn.bind(object):fn;
            }});
          };
          return typeof value==='function'?value.bind(target):value;
        }});
        await brain.readState(selected,selectedOnly);
        assert.equal(copies.length,1);assert.equal(copies[0][4],selected.length*9*4);
        assert(copies.every(copy=>copy[0]!==brain.inputs));
        await brain.readState(selected,{includeSpikeTime:true});
        assert.equal(copies.length,3);assert.equal(copies.at(-1)[0],brain.inputs);
      }finally{brain.dispose();}
    });
  }finally{globalThis.fetch=previousFetch;installed.uninstall();}
});
