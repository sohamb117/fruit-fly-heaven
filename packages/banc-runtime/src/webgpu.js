import {validateModel,initialBuffers,validateStep,historySlots,SPIKE_CAPACITY,decodeSpikeHistory} from './model.js';

const readoutShader=`
@group(0) @binding(0) var<storage,read> state:array<f32>;
struct ReadConfig {count:u32,stride:u32,ids:array<u32>}
@group(0) @binding(1) var<storage,read> config:ReadConfig;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read> kinetics:array<f32>;
@compute @workgroup_size(64) fn readout(@builtin(global_invocation_id) id:vec3<u32>){
 let n=arrayLength(&state)/8u;
 if(id.x<config.count){
   for(var k=0u;k<8u;k++){output[id.x*config.stride+k]=state[config.ids[id.x]*8u+k];}
   if(config.stride==9u){output[id.x*9u+8u]=kinetics[n*18u+config.ids[id.x]];}
 }
 if(id.x<(n+255u)/256u){
   var spikes=0.0;var activeCount=0.0;
   for(var i=id.x*256u;i<min(n,(id.x+1u)*256u);i++){let count=state[i*8u+3u];spikes+=count;if(count>0.0){activeCount+=1.0;}}
   output[config.count*config.stride+id.x*2u]=spikes;output[config.count*config.stride+id.x*2u+1u]=activeCount;
 }
}`;

export class WebGPUBrain {
  static async create(model,{gpu=globalThis.navigator?.gpu,shared=null}={}) {
    if(shared){
      shared.live();
      if(shared.model!==model)throw new Error('Shared GPU graph must use the same model');
      const brain=new WebGPUBrain(shared.device,model);brain.shared=shared.shared;brain.adapterInfo=shared.adapterInfo;
      brain.shared.references++;
      try{await brain.init();return brain;}catch(e){brain.dispose();throw e;}
    }
    validateModel(model);
    if(!gpu)throw new Error('WebGPU unavailable');
    const adapter=await gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw new Error('No WebGPU adapter');
    const n=model.manifest.neuron_count;
    const largest=Math.max(model.edges.byteLength,n*historySlots(model)*4,n*19*4,(n*17+27)*4,(n+2+SPIKE_CAPACITY*2)*4,8);
    if(largest>adapter.limits.maxStorageBufferBindingSize||largest>adapter.limits.maxBufferSize)throw new Error(`GPU storage limit too small for BANC (${Math.ceil(largest/1048576)} MiB required)`);
    const device=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:largest,maxBufferSize:Math.max(largest,32768),maxStorageBuffersPerShaderStage:8}});
    const brain=new WebGPUBrain(device,model);brain.adapterInfo={vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,device:adapter.info?.device,description:adapter.info?.description};
    try{await brain.init();return brain;}catch(e){brain.dispose();throw e;}
  }
  constructor(device,model){this.device=device;this.model=model;this.n=model.manifest.neuron_count;this.tick=0;this.backend='webgpu';this.buffers=[];this.disposed=false;this.lost=null;this.busy=false;
    device.lost.then(info=>{if(!this.disposed)this.lost=new Error(`WebGPU device lost: ${info.message}. Restart to reset neural state.`);});
    device.addEventListener('uncapturederror',event=>{this.lost=event.error;});
  }
  buffer(array,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC){
    const b=this.device.createBuffer({size:Math.max(4,array.byteLength),usage,mappedAtCreation:true});
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(array.buffer,array.byteOffset,array.byteLength));b.unmap();this.buffers.push(b);return b;
  }
  async init(){
    const d=this.device,{packed,state,history,kinetics}=initialBuffers(this.model);
    if(!this.shared){
    const response=await fetch(new URL('./neural.wgsl',import.meta.url));if(!response.ok)throw new Error('Missing neural WGSL');
    const module=d.createShaderModule({code:await response.text()});
    const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==='error');
    if(errors.length)throw new Error(errors.map(e=>e.message).join('\n'));
    this.pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'step'}});
    const readModule=d.createShaderModule({code:readoutShader}),readInfo=await readModule.getCompilationInfo();
    const readErrors=readInfo.messages.filter(m=>m.type==='error');if(readErrors.length)throw new Error(readErrors.map(e=>e.message).join('\n'));
    this.readPipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module:readModule,entryPoint:'readout'}});
    // Even an empty CSR needs one struct-sized storage binding for WGSL.
    this.offsets=this.buffer(this.model.offsets);this.edges=this.buffer(this.model.edges.length?this.model.edges:new Uint32Array(4));this.params=this.buffer(packed);
    this.shared={references:1,buffers:this.buffers.splice(0),offsets:this.offsets,edges:this.edges,params:this.params,pipeline:this.pipeline,readPipeline:this.readPipeline};
    }else for(const key of ['offsets','edges','params','pipeline','readPipeline'])this[key]=this.shared[key];
    this.states=[this.buffer(state),this.buffer(state)];this.history=this.buffer(history);this.kinetics=this.buffer(kinetics);
    this.inputs=this.buffer(new Uint32Array(this.n+2+SPIKE_CAPACITY*2));
    this.traceStaging=this.buffer(new Float32Array(128*8),GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    this.config=this.buffer(new Uint32Array(128*64),GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    // Each dispatch has a distinct aligned uniform range. Queue writes cannot
    // provide per-dispatch values when they overwrite one shared uniform slot.
    this.bindings=Array.from({length:128},(_,k)=>[0,1].map(parity=>d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:
      [this.offsets,this.edges,this.params,this.states[parity],this.states[1-parity],this.history,this.kinetics].map((buffer,binding)=>({binding,resource:{buffer}}))
        .concat({binding:7,resource:{buffer:this.config,offset:k*256,size:32}},{binding:8,resource:{buffer:this.inputs}})})));
    this.allocatedBytes=this.buffers.reduce((sum,b)=>sum+b.size,0);
  }
  live(){if(this.disposed)throw new Error('Brain disposed');if(this.lost)throw this.lost;}
  get timeMs(){return this.tick*this.model.manifest.dt_ms;}
  async step(steps,input,internal={hunger:0,insulin:0,akh:0},gaps=true,{traceIndex=null}={}){
    this.live();if(this.busy)throw new Error('Concurrent GPU operation');validateStep(steps,input,this.n,internal);
    if(traceIndex!==null&&(!Number.isInteger(traceIndex)||traceIndex<0||traceIndex>=this.n))throw new Error('Invalid trace index');
    this.busy=true;
    try{
      const d=this.device,uniform=new ArrayBuffer(steps*256),u=new Uint32Array(uniform),f=new Float32Array(uniform);
      const slots=historySlots(this.model);
      for(let k=0;k<steps;k++){u.set([this.n,this.tick+k,slots,+gaps],k*64);f.set([this.model.manifest.dt_ms,internal.hunger,internal.insulin,internal.akh],k*64+4);}
      d.queue.writeBuffer(this.inputs,0,input);d.queue.writeBuffer(this.config,0,uniform);
      const encoder=d.createCommandEncoder();let pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);
      for(let k=0;k<steps;k++){
        pass.setBindGroup(0,this.bindings[k][(this.tick+k)%2]);pass.dispatchWorkgroups(Math.ceil(this.n/128));
        if(traceIndex!==null){
          // Preserve every timestep on the GPU; one readback serves the block.
          pass.end();encoder.copyBufferToBuffer(this.states[1-(this.tick+k)%2],traceIndex*32,this.traceStaging,k*32,32);
          if(k+1<steps){pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);}
        }
      }
      if(traceIndex===null)pass.end();d.queue.submit([encoder.finish()]);
      let trace;
      if(traceIndex!==null){await this.traceStaging.mapAsync(GPUMapMode.READ,0,steps*32);trace=new Float32Array(this.traceStaging.getMappedRange(0,steps*32)).slice();}
      else await d.queue.onSubmittedWorkDone();
      this.live();this.tick+=steps;return trace;
    }finally{if(this.traceStaging.mapState==='mapped')this.traceStaging.unmap();this.busy=false;}
  }
  async readState(indices=Uint32Array.from({length:this.n},(_,i)=>i),{includeSpikeTime=false}={}){
    this.live();if(this.busy)throw new Error('Concurrent GPU operation');
    if(!(indices instanceof Uint32Array)||!indices.length||indices.some(i=>i>=this.n))throw new Error('Invalid readout indices');
    this.busy=true;let staging;
    try{
      const d=this.device;
      const stride=includeSpikeTime?9:8,partials=Math.ceil(this.n/256),bytes=(indices.length*stride+partials*2)*4,eventBytes=includeSpikeTime?(2+SPIKE_CAPACITY*2)*4:0;
      // The live console alternates small motor readouts and full inspection.
      // Keep both buffers instead of destroying/reallocating them each switch.
      this.readCaches??=[];
      let cache=this.readCaches.find(c=>c.stride===stride&&c.ids.length===indices.length&&indices.every((v,i)=>v===c.ids[i]));
      if(!cache){
        if(this.readCaches.length>=2){const old=this.readCaches.shift();for(const key of ['index','output','staging'])old[key].destroy();}
        const config=new Uint32Array(indices.length+2);config.set([indices.length,stride]);config.set(indices,2);
        const index=d.createBuffer({size:config.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});d.queue.writeBuffer(index,0,config);
        const output=d.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
        const buffer=d.createBuffer({size:bytes+eventBytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
        const bindings=this.states.map(state=>d.createBindGroup({layout:this.readPipeline.getBindGroupLayout(0),entries:[state,index,output,this.kinetics].map((buffer,binding)=>({binding,resource:{buffer}}))}));
        cache={ids:indices.slice(),stride,index,output,staging:buffer,bindings};this.readCaches.push(cache);
      }
      staging=cache.staging;const encoder=d.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(this.readPipeline);pass.setBindGroup(0,cache.bindings[this.tick%2]);pass.dispatchWorkgroups(Math.ceil(Math.max(indices.length,partials)/64));pass.end();
      encoder.copyBufferToBuffer(cache.output,0,staging,0,bytes);
      if(includeSpikeTime)encoder.copyBufferToBuffer(this.inputs,this.n*4,staging,bytes,eventBytes);
      d.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);this.live();
      const mapped=new Float32Array(staging.getMappedRange()),result=mapped.slice(0,indices.length*stride);
      result.totalSpikes=0;result.activeEver=0;
      for(let i=indices.length*stride;i<bytes/4;i+=2){result.totalSpikes+=mapped[i];result.activeEver+=mapped[i+1];}
      if(includeSpikeTime)result.spikes=decodeSpikeHistory(new Uint32Array(mapped.buffer,bytes,2+SPIKE_CAPACITY*2));
      return result;
    }finally{if(staging?.mapState==='mapped')staging.unmap();this.busy=false;}
  }
  dispose(){if(!this.disposed){this.disposed=true;this.buffers.forEach(b=>b.destroy());for(const cache of this.readCaches||[])for(const key of ['index','output','staging'])cache[key].destroy();if(!this.shared||--this.shared.references===0){this.shared?.buffers.forEach(b=>b.destroy());this.device.destroy();}}}
}
