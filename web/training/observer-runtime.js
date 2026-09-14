// Presentation-only reads of the running WASM brain. Never advances a clock,
// calls readState again, or retains/transfers a view of native memory.
const installed=new WeakMap();
export const MAX_BRAIN_SAMPLE=4096;
export const BRAIN_SAMPLE_INTERVAL_MS=500;

export function createBrainObserver({Brain,postMessage,now=()=>performance.now()}){
 if(!Brain?.prototype||typeof Brain.prototype.readState!=='function'||typeof postMessage!=='function')throw new TypeError('Brain observer requires a WASM brain and message sink');
 const prototype=Brain.prototype,descriptor=Object.getOwnPropertyDescriptor(prototype,'readState');
 if(installed.has(prototype))throw new Error('Brain observer already installed');
 const original=prototype.readState;
 let enabled=false,disposed=false,indices=null,job=null,brain=null,validatedRevision=-1,revision=0,sequence=0,lastWall=-Infinity;
 const report=error=>{try{postMessage({type:'brain-error',message:error?.message||'Neural preview unavailable'});}catch{}};
 function afterRead(value){
  if(!job||!indices)return;
  try{
   const wall=now();if(!Number.isFinite(wall))throw new Error('Invalid neural preview clock');
   if(wall-lastWall<BRAIN_SAMPLE_INTERVAL_MS)return;
   if(brain&&brain!==value)throw new Error('Neural preview brain changed during its trial');
   brain=value;
   const n=brain.n,heap=brain.core?.HEAPF32;
   if(!Number.isInteger(n)||n<1||!(heap instanceof Float32Array)||brain.disposed)throw new Error('Neural preview state unavailable');
   if(validatedRevision!==revision){
    if(indices.some(index=>index>=n))throw new RangeError('Neural preview index is outside the running brain');
    validatedRevision=revision;
   }
   const offset=brain.states?.[brain.tick%2]/4,lastOffset=brain.kinetics/4+n*18,neuralTimeMs=brain.timeMs;
   if(!Number.isSafeInteger(offset)||offset<0||offset+n*8>heap.length||!Number.isSafeInteger(lastOffset)||lastOffset<0||lastOffset+n>heap.length||!Number.isFinite(neuralTimeMs))throw new Error('Neural preview memory layout unavailable');
   const count=indices.length,voltage=new Float32Array(count),rates=new Float32Array(count),lastSpikeMs=new Float32Array(count);
   for(let k=0;k<count;k++){
    const index=indices[k];voltage[k]=heap[offset+index*8];rates[k]=heap[offset+index*8+4];lastSpikeMs[k]=heap[lastOffset+index];
    if(!Number.isFinite(voltage[k])||!Number.isFinite(rates[k])||!Number.isFinite(lastSpikeMs[k]))throw new Error('Nonfinite neural preview value');
   }
   const selected=indices.slice(),snapshot={jobId:job.id,neuralTimeMs,sampleSequence:++sequence,indices:selected,voltage,rates,lastSpikeMs};
   // Account before delivery: a failing or reentrant observer cannot flood.
   lastWall=wall;
   postMessage({type:'brain',snapshot},[selected.buffer,voltage.buffer,rates.buffer,lastSpikeMs.buffer]);
  }catch(error){enabled=false;brain=null;validatedRevision=-1;report(error);}
 }
 function wrappedReadState(){
  const result=original.apply(this,arguments);
  if(enabled)afterRead(this);
  return result;
 }
 Object.defineProperty(prototype,'readState',{configurable:true,writable:true,...descriptor,value:wrappedReadState});installed.set(prototype,wrappedReadState);
 function endJob(token){if(token!==undefined&&token!==job)return;job=null;brain=null;validatedRevision=-1;}
 return {
  configure(message){
   if(disposed)throw new Error('Brain observer disposed');
   enabled=false;indices=null;brain=null;validatedRevision=-1;
   if(typeof message?.enabled!=='boolean')throw new TypeError('Neural preview enabled must be boolean');
   if(!message.enabled)return;
   const requested=message.indices;
   if((!Array.isArray(requested)&&!(requested instanceof Uint32Array))||!requested.length||requested.length>MAX_BRAIN_SAMPLE||
    requested.some(index=>!Number.isInteger(index)||index<0||index>0xffffffff)||new Set(requested).size!==requested.length)
    throw new RangeError('Neural preview requires 1–4096 unique neuron indices');
   indices=Uint32Array.from(requested);revision++;enabled=true;
  },
  beginJob(id){
   endJob();
   // A missing assignment is not a reason to stop computation; simply provide
   // no attributed preview for calibration or an unknown local evaluation.
   if(typeof id!=='string'||!id.length||id.length>256)return null;
   job={id};return job;
  },
  endJob,
  report,
  dispose(){
   if(disposed)return;disposed=true;enabled=false;indices=null;endJob();
   if(prototype.readState===wrappedReadState){if(descriptor)Object.defineProperty(prototype,'readState',descriptor);else delete prototype.readState;}
   if(installed.get(prototype)===wrappedReadState)installed.delete(prototype);
  },
  get state(){return {enabled,disposed,jobId:job?.id??null,sampleCount:indices?.length??0,sampleSequence:sequence};},
 };
}
