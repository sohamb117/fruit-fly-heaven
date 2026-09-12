import createCore64 from './core.js';

export const DEFAULT_PARAMETERS=Object.freeze({
  dtMs:.1,tauMembraneMs:20,tauSynapseMs:5,restMv:-52,thresholdMv:-45,resetMv:-52,
  refractoryMs:2.2,delayMs:1.8,weightScaleMv:.275,spikeHistoryCapacity:8192,
});
const FIELDS={voltage:0,synapticDrive:1,spikeCount:2};
const UNITS={voltage:'mV',synapticDrive:'mV',spikeCount:'spikes'};
const PARAMS=['dtMs','tauMembraneMs','tauSynapseMs','restMv','thresholdMv','resetMv','refractoryMs','delayMs','weightScaleMv'];
const ensure=(condition,message)=>{if(!condition)throw new TypeError(message);};
const integer=(v,name,min=0,max=0xffffffff)=>ensure(Number.isInteger(v)&&v>=min&&v<=max,`${name} must be an integer in ${min}…${max}`);

/** A module owns a WASM heap; connectomes in that heap can be shared by many brains. */
export async function createBrainModule(options={}){
  const precision=options.precision??'float64';
  ensure(precision==='float64'||precision==='float32','precision must be float64 or float32');
  const createCore=precision==='float32'?(await import('./core-f32.js')).default:createCore64;
  const core=await createCore({
    ...(options.wasmBinary?{wasmBinary:options.wasmBinary}:{}),
    locateFile:path=>options.wasmUrl?String(options.wasmUrl):new URL(path,import.meta.url).href,
  });
  ensure(core._fb_precision_bits?.()===(precision==='float32'?32:64),'WASM binary does not match requested precision; use the matching release artifact');
  function check(value){if(value===-1)throw new Error(core.UTF8ToString(core._fb_error()));return value;}
  function pointer(value){if(!value)throw new Error(core.UTF8ToString(core._fb_error())||'WASM allocation failed');return value;}
  function alloc(typed){
    const p=pointer(core._malloc(Math.max(8,typed.byteLength)));
    core.HEAPU8.set(new Uint8Array(typed.buffer,typed.byteOffset,typed.byteLength),p);return p;
  }
  function withBuffers(arrays,fn){const ps=[];try{for(const a of arrays)ps.push(alloc(a));return fn(...ps);}finally{for(const p of ps)core._free(p);}}
  function indices(values,n){ensure(values instanceof Uint32Array,'indices must be Uint32Array');for(const i of values)ensure(i<n,'Neuron index outside graph');return values;}
  function floats(values,length,name){ensure(values instanceof Float32Array,`${name} must be Float32Array`);ensure(values.length===length,`${name} length mismatch`);for(const v of values)ensure(Number.isFinite(v),`${name} must be finite`);return values;}
  function seedParts(seed){
    if(typeof seed==='number'){ensure(Number.isSafeInteger(seed)&&seed>=0,'seed must be a nonnegative safe integer or bigint');seed=BigInt(seed);}
    ensure(typeof seed==='bigint'&&seed>=0n&&seed<=0xffffffffffffffffn,'seed must fit uint64');
    return [Number(seed&0xffffffffn),Number(seed>>32n)];
  }
  class Brain {
    #ptr;#graph;#parameters;
    constructor(graph,options={}){
      this.#graph=graph;const {seed=1,...overrides}=options;
      for(const key of Object.keys(overrides))ensure(key in DEFAULT_PARAMETERS,`Unknown neuron parameter: ${key}`);
      this.#parameters=Object.freeze({...DEFAULT_PARAMETERS,...overrides});
      integer(this.#parameters.spikeHistoryCapacity,'spikeHistoryCapacity',1,1048576);
      const [lo,hi]=seedParts(seed);
      this.#ptr=withBuffers([new Float64Array(PARAMS.map(k=>this.#parameters[k]))],p=>pointer(core._fb_brain_create(graph._handle(),lo,hi,p,this.#parameters.spikeHistoryCapacity)));
    }
    #live(){ensure(this.#ptr,'Brain has been disposed');return this.#ptr;}
    get neuronCount(){return this.#graph.neuronCount;}
    get precision(){return precision;}
    get parameters(){return this.#parameters;}
    get timeMs(){return core._fb_time(this.#live());}
    get totalSpikes(){return core._fb_spike_total(this.#live());}
    get connectome(){return this.#graph;}
    /** Replace all Poisson voltage-input channels. Rates are Hz; amplitudes are mV. */
    setPoissonInputs({indices:ids,ratesHz,amplitudesMv}){
      indices(ids,this.neuronCount);floats(ratesHz,ids.length,'ratesHz');
      amplitudesMv=amplitudesMv??new Float32Array(ids.length).fill(68.75);
      floats(amplitudesMv,ids.length,'amplitudesMv');
      withBuffers([ids,ratesHz,amplitudesMv],(i,r,a)=>check(core._fb_inputs(this.#live(),ids.length,i,r,a)));return this;
    }
    clearPoissonInputs(){return this.setPoissonInputs({indices:new Uint32Array(),ratesHz:new Float32Array()});}
    /** Set a constant depolarizing drive in the dv/dt equation, measured in mV. */
    setCurrentInputs(ids,driveMv){indices(ids,this.neuronCount);floats(driveMv,ids.length,'driveMv');withBuffers([ids,driveMv],(i,v)=>check(core._fb_currents(this.#live(),ids.length,i,v)));return this;}
    /** Deliver an instantaneous voltage pulse at the current neural time. */
    injectVoltage(ids,deltaMv){indices(ids,this.neuronCount);floats(deltaMv,ids.length,'deltaMv');withBuffers([ids,deltaMv],(i,v)=>check(core._fb_inject(this.#live(),ids.length,i,v)));return this;}
    setRefractoryPeriod(ids,ms){indices(ids,this.neuronCount);withBuffers([ids],i=>check(core._fb_refractory(this.#live(),ids.length,i,ms)));return this;}
    /** Advance exactly durationMs, an integer multiple of dtMs; returns neural time. */
    step(durationMs){
      ensure(Number.isFinite(durationMs)&&durationMs>=0,'durationMs must be finite and nonnegative');
      const ticks=durationMs/this.#parameters.dtMs;
      ensure(Math.abs(ticks-Math.round(ticks))<1e-7,'durationMs must be an integer multiple of dtMs');
      integer(Math.round(ticks),'ticks',0,1000000);check(core._fb_step(this.#live(),Math.round(ticks)));return this.timeMs;
    }
    /** A copy: inspecting state never mutates neuron dynamics or exposes a stale heap view. */
    readActivations({field='voltage',indices:ids}={}){
      ensure(field in FIELDS,'Unknown activation field');if(ids)indices(ids,this.neuronCount);
      const n=ids?.length??this.neuronCount,out=pointer(core._malloc(Math.max(8,n*8)));
      try{
        if(ids)withBuffers([ids],i=>check(core._fb_read(this.#live(),FIELDS[field],n,i,out)));
        else check(core._fb_read(this.#live(),FIELDS[field],n,0,out));
        return core.HEAPF64.slice(out/8,out/8+n);
      }finally{core._free(out);}
    }
    /** Bounded history, with explicit cumulative dropped-event count. */
    readSpikes(){
      const cap=this.#parameters.spikeHistoryCapacity;
      const times=pointer(core._malloc(cap*8)),ids=pointer(core._malloc(cap*4));
      try{
        const n=core._fb_spikes(this.#live(),times,ids),total=this.totalSpikes;
        return {timesMs:core.HEAPF64.slice(times/8,times/8+n),neuronIndices:core.HEAPU32.slice(ids/4,ids/4+n),total,dropped:Math.max(0,total-n)};
      }finally{core._free(times);core._free(ids);}
    }
    dispose(){if(this.#ptr){core._fb_brain_destroy(this.#ptr);this.#ptr=0;}}
  }
  class Connectome {
    #ptr;#n;#edges;
    constructor({neuronCount,rowOffsets,targets,weights}){
      integer(neuronCount,'neuronCount',1);
      ensure(rowOffsets instanceof Uint32Array && rowOffsets.length===neuronCount+1,'rowOffsets must have neuronCount + 1 entries');
      ensure(targets instanceof Uint32Array,'targets must be Uint32Array');floats(weights,targets.length,'weights');
      this.#n=neuronCount;this.#edges=targets.length;
      this.#ptr=withBuffers([rowOffsets,targets,weights],(r,c,w)=>pointer(core._fb_graph_create(neuronCount,targets.length,r,c,w)));
    }
    _handle(){ensure(this.#ptr,'Connectome has been disposed');return this.#ptr;}
    get neuronCount(){return this.#n;}get edgeCount(){return this.#edges;}
    createBrain(options={}){return new Brain(this,options);}
    createPopulation(count,options={}){
      integer(count,'count',1,10000);const first=options.seed??1;seedParts(first);
      const brains=[];try{for(let i=0;i<count;i++)brains.push(this.createBrain({...options,seed:BigInt(first)+BigInt(i)}));return brains;}catch(error){for(const b of brains)b.dispose();throw error;}
    }
    // Existing brains retain shared ownership of native graph memory.
    dispose(){if(this.#ptr){core._fb_graph_destroy(this.#ptr);this.#ptr=0;}}
  }
  return Object.freeze({
    precision,
    createConnectome:csr=>new Connectome(csr),
    readActivationMatrix(brains,options={}){
      ensure(Array.isArray(brains)&&brains.length>0,'Provide at least one brain');
      const graph=brains[0].connectome;
      ensure(brains.every(b=>b instanceof Brain&&b.connectome===graph),'Matrix rows must use the same connectome and module');
      const field=options.field??'voltage',cols=options.indices?.length??graph.neuronCount;
      ensure(field in FIELDS,'Unknown activation field');
      const values=new Float64Array(brains.length*cols),timesMs=new Float64Array(brains.length);
      brains.forEach((b,row)=>{values.set(b.readActivations(options),row*cols);timesMs[row]=b.timeMs;});
      return {values,shape:[brains.length,cols],order:'row-major',field,unit:UNITS[field],timesMs,precision};
    },
    get allocatedHeapBytes(){return core.HEAPU8.buffer.byteLength;},
  });
}

/** Load raw little-endian CSR arrays. Dataset licensing stays with the caller. */
export async function loadConnectome(url,{fetch:fetcher=globalThis.fetch}={}){
  ensure(typeof fetcher==='function','A fetch implementation is required');
  const base=new URL(url,globalThis.location?.href??import.meta.url);
  async function get(name){const response=await fetcher(new URL(name,base));if(!response.ok)throw new Error(`Connectome ${name}: HTTP ${response.status}`);return response;}
  const meta=await(await get('metadata.json')).json();
  const [r,c,w]=await Promise.all(['indptr.bin','targets.bin','weights.bin'].map(async name=>(await get(name)).arrayBuffer()));
  return {metadata:meta,neuronCount:meta.neurons_per_brain??meta.neuronCount,rowOffsets:new Uint32Array(r),targets:new Uint32Array(c),weights:new Float32Array(w)};
}
