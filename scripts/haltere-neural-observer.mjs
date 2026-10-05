// Diagnostic-only selected readout expansion. Never imported by live runtime.
const installations=new WeakMap();
const WING_COUNT=48,HALTERE_COUNT=328,STRIDE=9;
const HALTERE_MOTOR_IDS=Object.freeze([97021,118683]);

function selected(values,count,n,label,{sorted=false}={}){
 if(!Array.isArray(values)&&!(values instanceof Uint32Array))throw new TypeError(label+' must be an array or Uint32Array');
 if(values.length!==count)throw new RangeError(label+' has the wrong count');
 const copy=Array.from(values);
 if(copy.some(value=>!Number.isInteger(value)||value<0||value>=n)||new Set(copy).size!==copy.length)
  throw new RangeError(label+' must be unique integer indices within the declared model');
 if(sorted&&copy.some((value,k)=>k>0&&value<=copy[k-1]))throw new RangeError(label+' must be strictly sorted');
 return Uint32Array.from(copy);
}

/**
 * Install around one explicit class's own prototype readState method.
 * onSample is synchronous and receives owned {state,indices,haltereIndices,
 * haltereMotorIndices,timeMs,stride}; state has 328 sensor rows then two hDVM
 * rows. Other reads return the original result/promise untouched.
 * Restore in a finally block. Restore also suppresses callbacks for an already
 * pending read, while that caller still receives its correctly sliced result.
 */
export function installHaltereNeuralObserver({BrainClass,neuronCount,wingIndices,haltereIndices,haltereMotorIndices,onSample}={}){
 if(typeof BrainClass!=='function'||!BrainClass.prototype)throw new TypeError('BrainClass must be an explicit constructor');
 if(!Number.isInteger(neuronCount)||neuronCount<1||neuronCount>2**32)throw new RangeError('Invalid model neuron count');
 if(typeof onSample!=='function')throw new TypeError('onSample must be a function');
 const wing=selected(wingIndices,WING_COUNT,neuronCount,'Wing indices',{sorted:true});
 const sensory=selected(haltereIndices,HALTERE_COUNT,neuronCount,'Haltere indices');
 const motor=selected(haltereMotorIndices,2,neuronCount,'Haltere motor indices');
 if(motor.some((value,k)=>value!==HALTERE_MOTOR_IDS[k]))throw new RangeError('Expected hDVM motor indices [97021,118683]');
 const extra=new Uint32Array(HALTERE_COUNT+2);extra.set(sensory);extra.set(motor,HALTERE_COUNT);
 const combined=new Uint32Array(WING_COUNT+extra.length);combined.set(wing);combined.set(extra,WING_COUNT);
 if(new Set(combined).size!==combined.length)throw new RangeError('Wing, haltere sensory and hDVM selections must be disjoint');
 const prototype=BrainClass.prototype,descriptor=Object.getOwnPropertyDescriptor(prototype,'readState');
 if(!descriptor||typeof descriptor.value!=='function'||(!descriptor.writable&&!descriptor.configurable))throw new TypeError('Expected a replaceable own prototype readState method');
 if(installations.has(prototype))throw new Error('Haltere neural observer is already installed');
 const original=descriptor.value;let active=true;
 function wrappedReadState(indices,options){
  const match=active&&indices instanceof Uint32Array&&indices.length===wing.length&&
   indices.every((value,k)=>value===wing[k])&&options?.includeSpikeTime===true;
  if(!match)return original.apply(this,arguments);
  if(this.n!==neuronCount)throw new Error('Observed brain neuron count differs from the declared model');
  // One original call performs one gather/map. Never issue a second GPU read.
  return Promise.resolve(original.call(this,combined,options)).then(result=>{
   if(!(result instanceof Float32Array)||result.length!==combined.length*STRIDE)throw new Error('Unexpected expanded neural readout shape');
   const prefix=result.slice(0,WING_COUNT*STRIDE);
   for(const name of ['totalSpikes','activeEver','spikes']){
    const property=Object.getOwnPropertyDescriptor(result,name);
    if(!property)throw new Error('Expanded neural readout is missing '+name);
    Object.defineProperty(prefix,name,property);
   }
   if(active){
    const timeMs=this.timeMs;
    if(!Number.isFinite(timeMs)||timeMs<0)throw new Error('Invalid observed neural time');
    const returned=onSample({timeMs,stride:STRIDE,state:result.slice(WING_COUNT*STRIDE),indices:extra.slice(),
     haltereIndices:sensory.slice(),haltereMotorIndices:motor.slice()});
    if(returned&&typeof returned.then==='function')throw new TypeError('onSample must be synchronous');
   }
   return prefix;
  });
 }
 Object.defineProperty(prototype,'readState',{...descriptor,value:wrappedReadState});
 installations.set(prototype,wrappedReadState);
 return Object.freeze({
  restore(){
   if(!active)return;
   if(prototype.readState!==wrappedReadState)throw new Error('Cannot restore: another owner replaced readState');
   Object.defineProperty(prototype,'readState',descriptor);active=false;installations.delete(prototype);
  },
  get active(){return active;},
 });
}
