import createCore from './core.js';
const assert=(ok,message)=>{if(!ok)throw new TypeError(message);};
const floats=(v,n,name)=>{assert(v instanceof Float32Array&&v.length===n,`${name} must have ${n} Float32 values`);assert(v.every(Number.isFinite),`${name} must be finite`);};

/** Generic graded recurrent networks. Models and camera mappings are host data. */
export async function createVisionModule(options={}){
  const core=await createCore({...options.wasmBinary?{wasmBinary:options.wasmBinary}:{},locateFile:p=>options.wasmUrl?String(options.wasmUrl):new URL(p,import.meta.url).href});
  const alloc=a=>{const p=core._malloc(Math.max(8,a.byteLength));if(!p)throw new Error('WASM allocation failed');core.HEAPU8.set(new Uint8Array(a.buffer,a.byteOffset,a.byteLength),p);return p;};
  class Model{
    #n;#k;#dt;#ps=[];#refs=1;#disposed=false;
    constructor({rowOffsets,sources,weights,bias,tauSeconds,inputIndices,dtSeconds=.02}){
      assert(bias instanceof Float32Array&&bias.length>0,'bias must be a nonempty Float32Array');
      this.#n=bias.length;this.#k=inputIndices?.length;this.#dt=dtSeconds;
      floats(bias,this.#n,'bias');floats(tauSeconds,this.#n,'tauSeconds');
      assert(tauSeconds.every(v=>v>0),'time constants must be positive');
      assert(Number.isFinite(dtSeconds)&&dtSeconds>0,'dtSeconds must be positive');
      assert(sources instanceof Uint32Array&&sources.every(i=>i<this.#n),'Invalid source indices');floats(weights,sources.length,'weights');
      assert(rowOffsets instanceof Uint32Array&&rowOffsets.length===this.#n+1&&rowOffsets[0]===0&&rowOffsets[this.#n]===sources.length&&rowOffsets.every((v,i)=>!i||v>=rowOffsets[i-1]),'Invalid CSR row offsets');
      assert(inputIndices instanceof Uint32Array&&inputIndices.every(i=>i<this.#n)&&new Set(inputIndices).size===this.#k,'Invalid input indices');
      try{for(const a of [rowOffsets,sources,weights,bias,tauSeconds,inputIndices])this.#ps.push(alloc(a));}catch(e){this.#release();throw e;}
    }
    #release(){if(--this.#refs===0){for(const p of this.#ps)core._free(p);this.#ps=[];}}
    get neuronCount(){return this.#n;}get inputCount(){return this.#k;}get dtSeconds(){return this.#dt;}
    createNetwork(){
      assert(!this.#disposed,'Model has been disposed');
      const n=this.#n,k=this.#k,dt=this.#dt,ps=this.#ps,state=[];
      try{for(const a of [core.HEAPF32.slice(ps[3]/4,ps[3]/4+n),new Float32Array(n),new Float32Array(n),new Float32Array(k)])state.push(alloc(a));}catch(e){for(const p of state)core._free(p);throw e;}
      ++this.#refs;let disposed=false,time=0;
      const live=()=>assert(!disposed,'Network has been disposed');
      return Object.freeze({
        get timeSeconds(){live();return time;},
        step(input,steps=1){
          live();floats(input,k,'input');assert(Number.isInteger(steps)&&steps>=0&&steps<=100000,'Invalid step count');
          core.HEAPF32.set(input,state[3]/4);
          if(core._gv_step(n,...ps.slice(0,5),k,ps[5],state[3],state[0],state[1],state[2],dt,steps)<0)throw new Error('Graded network became nonfinite');
          time+=dt*steps;return time;
        },
        readActivations(){live();return core.HEAPF32.slice(state[0]/4,state[0]/4+n);},
        reset(){live();core.HEAPF32.copyWithin(state[0]/4,ps[3]/4,ps[3]/4+n);time=0;},
        dispose:()=>{if(!disposed){state.forEach(p=>core._free(p));disposed=true;this.#release();}},
      });
    }
    dispose(){if(!this.#disposed){this.#disposed=true;this.#release();}}
  }
  return Object.freeze({createModel:data=>new Model(data),get allocatedHeapBytes(){return core.HEAPU8.buffer.byteLength;}});
}
