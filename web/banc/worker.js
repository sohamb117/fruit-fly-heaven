import {loadBancModel,createRuntime} from '/banc-engine/src/index.js';
import {ReducedBody,sensoryCurrents} from './embodiment.js';
let runtime,model,body,indices,paused=false,alive=true,singleSteps=0;
const control={odor:true,vision:true,taste:true,proprioception:true,gaps:true,coupling:true,flight:true};
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function init(args){
  paused=args.paused??false;Object.assign(control,args.control||{});
  self.postMessage({type:'progress',message:'Verifying BANC v888 graph and annotations…'});
  model=await loadBancModel();
  const response=await fetch('/body-model/flybody-reduced.json');if(!response.ok)throw new Error('Run scripts/distill-flybody.py to prepare the body');
  const mechanics=await response.json();
  self.postMessage({type:'progress',message:'Allocating neural state and compiling compute kernels…'});
  runtime=await createRuntime(model,{backend:args.backend,onFallback:message=>self.postMessage({type:'fallback',message})});
  body=new ReducedBody(mechanics,model.io,runtime,{energy:args.energy,initialCondition:args.initialCondition});
  indices=Uint32Array.from(model.io.motor_neurons.map(m=>m.index));
  self.postMessage({type:'ready',backend:runtime.brain.backend,adapter:runtime.brain.adapterInfo,manifest:model.manifest,muscles:model.io.muscles,motors:model.io.motor_neurons,body:mechanics});
  await run();
}
async function run(){
  while(alive){
    if(paused&&singleSteps===0){await sleep(20);continue;}
    if(singleSteps>0)singleSteps--;
    const started=performance.now(),input=sensoryCurrents(model,body,control);
    await runtime.brain.step(10,input,body.internal,control.gaps);
    const state=await runtime.brain.readState(indices),rates=new Map(Array.from(indices,(id,k)=>[id,state[k*8+4]]));
    body.step(rates,.005,control);
    self.postMessage({type:'update',timeMs:runtime.brain.timeMs,wallMs:performance.now()-started,body:body.snapshot(),state},[state.buffer]);
    await sleep(0);
  }
}
self.onmessage=({data})=>{
  if(data.type==='init')init(data).catch(fail);
  if(data.type==='control'){Object.assign(control,data.control||{});if('paused' in data)paused=data.paused;}
  if(data.type==='step'){paused=true;singleSteps++;}
};
function fail(error){alive=false;body?.dispose();runtime?.dispose();self.postMessage({type:'error',message:error.stack||error.message||String(error)});}
self.addEventListener('unhandledrejection',event=>fail(event.reason));
