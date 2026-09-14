// Keep immutable graph/device resources alive between evaluations. Each call
// still returns a fresh neural state; callers dispose that episode's brain.
export function createTrainingBrainResources({model,createGPU,createWasm,onFallback=()=>{},backend:requestedBackend='auto'}){
  if(!['auto','webgpu','wasm'].includes(requestedBackend))throw new Error('Unknown training neural backend');
  let anchor=null,backend=null,disposed=false,busy=false;
  const check=()=>{if(disposed)throw new Error('Training brain resources disposed');};
  return {
    get backend(){return backend;},
    async create(){
      check();if(busy)throw new Error('Concurrent training brain creation');busy=true;
      try{
        if(!anchor){
          if(requestedBackend==='wasm'){anchor=createWasm(model);backend='wasm';}
          else{
            try{anchor=await createGPU(model);backend='webgpu';}
            catch(error){check();if(requestedBackend==='webgpu')throw error;onFallback(error);anchor=createWasm(model);backend='wasm';}
          }
          if(disposed){anchor.dispose();anchor=null;check();}
        }
        // In particular, a later GPU/device failure must not silently change
        // backend or reuse a previous episode's state.
        const brain=backend==='webgpu'?await createGPU(model,{shared:anchor}):createWasm(model,{shared:anchor});
        if(disposed){brain.dispose();check();}
        return brain;
      }finally{busy=false;}
    },
    dispose(){if(!disposed){disposed=true;anchor?.dispose();anchor=null;}}
  };
}
