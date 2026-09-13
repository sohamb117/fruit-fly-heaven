import {createWasmCore,WasmBrain,WasmMuscles,WasmJoints} from './wasm.js';
import {WebGPUBrain} from './webgpu.js';
export {loadBancModel,validateModel} from './model.js';
export {WasmBrain,WasmMuscles,WasmJoints,WebGPUBrain,createWasmCore};

export async function createRuntime(model,{backend='auto',onFallback=()=>{},wasmOptions}={}){
  if(!['auto','webgpu','wasm'].includes(backend))throw new Error('Unknown backend');
  const core=await createWasmCore(wasmOptions);let brain;
  if(backend!=='wasm'){
    try{brain=await WebGPUBrain.create(model);}catch(error){if(backend==='webgpu')throw error;onFallback(error.message);}
  }
  brain??=new WasmBrain(core,model);
  return {brain,createMuscles:count=>new WasmMuscles(core,count),createJoints:joints=>new WasmJoints(core,joints),dispose:()=>brain.dispose()};
}
