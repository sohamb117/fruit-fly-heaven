import {validateModel,initialBuffers,validateStep,historySlots,SPIKE_CAPACITY,decodeSpikeHistory} from './model.js';

export async function createWasmCore(options={}) {
  const {default:create}=await import('../dist/core.js');
  return create(options);
}
export class WasmBrain {
  constructor(core,model,{shared=null}={}) {
    if(shared){shared.live();if(shared.core!==core||shared.model!==model)throw new Error('Shared WASM graph must use the same core and model');}
    else validateModel(model);
    this.core=core;this.model=model;this.n=model.manifest.neuron_count;
    this.tick=0;this.backend='wasm';this.allocations=[];this.disposed=false;
    const {packed,state,history,kinetics}=initialBuffers(model);
    const alloc=a=>{const p=core._malloc(Math.max(8,a.byteLength));if(!p)throw new Error('WASM allocation failed');this.allocations.push(p);core.HEAPU8.set(new Uint8Array(a.buffer,a.byteOffset,a.byteLength),p);return p;};
    try{
      if(shared){this.shared=shared.shared;this.shared.references++;this.offsets=shared.offsets;this.edges=shared.edges;}
      else{this.offsets=alloc(model.offsets);this.edges=alloc(model.edges);this.shared={references:1,pointers:this.allocations.splice(0)};}
      this.params=alloc(packed);
      this.states=[alloc(state),alloc(state)];this.history=alloc(history);this.kinetics=alloc(kinetics);
      this.events=alloc(new Uint32Array(2+SPIKE_CAPACITY*2));
    }catch(e){this.dispose();throw e;}
  }
  get timeMs(){return this.tick*this.model.manifest.dt_ms;}
  live(){if(this.disposed)throw new Error('Brain disposed');}
  step(steps,input,internal={hunger:0,insulin:0,akh:0},gaps=true,{traceIndex=null}={}) {
    this.live();validateStep(steps,input,this.n,internal);
    if(traceIndex!==null&&(!Number.isInteger(traceIndex)||traceIndex<0||traceIndex>=this.n))throw new Error('Invalid trace index');
    const trace=traceIndex===null?undefined:new Float32Array(steps*8);
    const c=this.core,m=this.model.manifest;
    c.HEAPF32.set(input,this.params/4+this.n*16);
    for(let i=0;i<steps;i++){
      c._br_step(this.n,this.tick,historySlots(this.model),+gaps,m.dt_ms,internal.hunger,internal.insulin,internal.akh,
        this.offsets,this.edges,this.params,this.states[this.tick%2],this.states[1-this.tick%2],this.history,this.kinetics,this.events);
      this.tick++;
      if(trace){const offset=this.states[this.tick%2]/4+traceIndex*8;trace.set(c.HEAPF32.subarray(offset,offset+8),i*8);}
    }
    return trace;
  }
  readState(indices,{includeSpikeTime=false}={}) {
    this.live();const state=this.core.HEAPF32.subarray(this.states[this.tick%2]/4,this.states[this.tick%2]/4+this.n*8);
    indices??=Uint32Array.from({length:this.n},(_,i)=>i);
    const stride=includeSpikeTime?9:8,out=new Float32Array(indices.length*stride);
    indices.forEach((i,k)=>{if(!Number.isInteger(i)||i<0||i>=this.n)throw new Error('Invalid readout index');out.set(state.subarray(i*8,i*8+8),k*stride);if(includeSpikeTime)out[k*stride+8]=this.core.HEAPF32[this.kinetics/4+this.n*18+i];});
    out.totalSpikes=0;out.activeEver=0;
    for(let i=0;i<this.n;i++){out.totalSpikes+=state[i*8+3];out.activeEver+=state[i*8+3]>0;}
    if(includeSpikeTime)out.spikes=decodeSpikeHistory(new Uint32Array(this.core.HEAPU8.buffer,this.events,2+SPIKE_CAPACITY*2));
    return out;
  }
  dispose(){if(!this.disposed){this.allocations.forEach(p=>this.core._free(p));if(this.shared&&--this.shared.references===0)this.shared.pointers.forEach(p=>this.core._free(p));this.disposed=true;}}
}

export class WasmMuscles {
  constructor(core,count){
    if(!Number.isInteger(count)||count<1)throw new Error('Invalid muscle count');
    this.core=core;this.count=count;this.disposed=false;this.input=core._malloc(count*20);this.state=core._malloc(count*12);
    if(!this.input||!this.state){this.dispose();throw new Error('Muscle allocation failed');}
    core.HEAPF32.fill(0,this.state/4,this.state/4+count*3);
  }
  step(input,dt){
    if(this.disposed)throw new Error('Muscles disposed');
    if(!(input instanceof Float32Array)||input.length!==this.count*5||!input.every(Number.isFinite)||!Number.isFinite(dt)||dt<=0||dt>.05)throw new Error('Invalid muscle step');
    this.core.HEAPF32.set(input,this.input/4);this.core._muscle_step(this.count,dt,this.input,this.state);
    return this.core.HEAPF32.slice(this.state/4,this.state/4+this.count*3);
  }
  dispose(){if(!this.disposed){if(this.input)this.core._free(this.input);if(this.state)this.core._free(this.state);this.disposed=true;}}
}

export class WasmJoints {
  constructor(core,joints){
    this.core=core;this.count=joints.length;this.allocations=[];this.disposed=false;
    const alloc=a=>{const p=core._malloc(a.byteLength);if(!p)throw new Error('Joint allocation failed');this.allocations.push(p);core.HEAPF32.set(a,p/4);return p;};
    const params=new Float32Array(joints.length*6),state=new Float32Array(joints.length*2);
    joints.forEach((j,i)=>{if(!(j.inertia>0&&j.damping>=0&&j.stiffness>=0))throw new Error('Invalid joint mechanics');params.set([j.inertia,j.damping,j.stiffness,j.neutral,Math.max(j.range[0],j.neutral-.35),Math.min(j.range[1],j.neutral+.35)],i*6);state[i*2]=j.neutral;});
    try{this.params=alloc(params);this.state=alloc(state);this.torque=alloc(new Float32Array(joints.length));}catch(e){this.dispose();throw e;}
  }
  step(torque,dt){
    if(this.disposed||!(torque instanceof Float32Array)||torque.length!==this.count||!torque.every(Number.isFinite)||!(dt>0&&dt<=.01))throw new Error('Invalid joint step');
    this.core.HEAPF32.set(torque,this.torque/4);this.core._joint_step(this.count,dt,this.params,this.torque,this.state);
    return this.core.HEAPF32.slice(this.state/4,this.state/4+this.count*2);
  }
  dispose(){if(!this.disposed){this.allocations.forEach(p=>this.core._free(p));this.disposed=true;}}
}
