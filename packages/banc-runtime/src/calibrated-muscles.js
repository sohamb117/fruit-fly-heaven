// Optional v2 muscle ABI. The ordinary WasmMuscles/brain paths are unchanged.
// Per-muscle profiles: activation rise/fall seconds, fatigue/recovery per second.
export class CalibratedWasmMuscles {
  constructor(core,profiles){
    if(typeof core._muscle_step_configured!=='function')throw new Error('Configured muscle ABI is unavailable; rebuild the BANC runtime');
    if(!(profiles instanceof Float32Array)||!profiles.length||profiles.length%4||!profiles.every(Number.isFinite))throw new Error('Invalid muscle kinetics profile');
    for(let i=0;i<profiles.length;i+=4)if(!(profiles[i]>=.001&&profiles[i]<=5&&profiles[i+1]>=.001&&profiles[i+1]<=5&&profiles[i+2]>=0&&profiles[i+2]<=1&&profiles[i+3]>=0&&profiles[i+3]<=1))throw new Error('Muscle kinetics outside supported limits');
    this.core=core;this.count=profiles.length/4;this.disposed=false;this.allocations=[];
    const alloc=bytes=>{const p=core._malloc(bytes);if(!p)throw new Error('Muscle allocation failed');this.allocations.push(p);return p;};
    try{
      this.input=alloc(this.count*20);this.state=alloc(this.count*12);this.profiles=alloc(profiles.byteLength);
      core.HEAPF32.fill(0,this.state/4,this.state/4+this.count*3);core.HEAPF32.set(profiles,this.profiles/4);
    }catch(error){this.dispose();throw error;}
  }
  step(input,dt){
    if(this.disposed)throw new Error('Muscles disposed');
    if(!(input instanceof Float32Array)||input.length!==this.count*5||!input.every(Number.isFinite)||!Number.isFinite(dt)||dt<=0||dt>.05)throw new Error('Invalid muscle step');
    const c=this.core;c.HEAPF32.set(input,this.input/4);c._muscle_step_configured(this.count,dt,this.input,this.state,this.profiles);
    return c.HEAPF32.slice(this.state/4,this.state/4+this.count*3);
  }
  dispose(){if(!this.disposed){for(const pointer of this.allocations)this.core._free(pointer);this.disposed=true;}}
}
