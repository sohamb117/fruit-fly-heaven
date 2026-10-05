import {validateMotorEventContract,decodeDlmEventTime} from './cell-models.js';
// Selected-neuron event observation only. No recruitment, muscle excitation,
// calcium model, rate-to-impulse conversion, or neural-state writes.
const PARAM_STRIDE=16,READ_STRIDE=9,MAX_EXACT_FLOAT32_INTEGER=2**24;

/**
 * Recover lossless events for the current 0.5 ms / 2 ms motor-readout contract.
 *
 * Supply the full Float32 model params and the SAME selected-index order used
 * by brain.readState(indices,{includeSpikeTime:true}). Each read must follow
 * exactly one body block on the same continuously running brain. The first
 * read, and the first read after explicit reset(), establish a baseline and
 * emit no historical events. `initialized` is true on that baseline read.
 *
 * The native and WGSL kernels test OLD refractory state before permitting a
 * spike. With dt=0.5 and refractory>=2, four subsequent ticks are forbidden;
 * the next possible spike is at least 2.5 ms later. Thus every half-open
 * (previousTime,currentTime] 2 ms window contains at most one event per cell.
 * This conservative bound deliberately rejects other timing configurations.
 *
 * Current timestamps/counts must remain in their exact float32 tick/integer
 * ranges. Neither the global event ring nor readState's extra properties are
 * inspected. Returned arrays are owned copies; a failed read is transactional.
 */
export function createWingMotorEventReader({indices,params,dtMs,bodyBlockMs,eventContract}){
 if(dtMs!==.5||bodyBlockMs!==2)throw new RangeError('Motor events require 0.5 ms neural ticks and 2 ms body blocks');
 if(!(params instanceof Float32Array)||!params.length||params.length%PARAM_STRIDE)
  throw new TypeError('Motor events require full Float32 model params with stride 16');
 if(!Array.isArray(indices)&&!(indices instanceof Uint32Array))throw new TypeError('Selected indices must be an array or Uint32Array');
 const n=params.length/PARAM_STRIDE,ids=Array.from(indices);
 if(!ids.length||ids.some(index=>!Number.isInteger(index)||index<0||index>=n)||new Set(ids).size!==ids.length)
  throw new RangeError('Selected indices must be nonempty, unique and in model bounds');
 const dlm=new Set(validateMotorEventContract(eventContract));
 if([...dlm].some(i=>i>=n))throw new RangeError('DLM timing cell outside model');
 const selected=Uint32Array.from(ids),contract=ids.map(index=>{
  const refractory=params[index*PARAM_STRIDE+5],graded=params[index*PARAM_STRIDE+8];
  if(!Number.isFinite(graded)||graded>.5)throw new RangeError('Selected motor neurons must all be spiking');
  if(!Number.isFinite(refractory)||(!dlm.has(index)&&refractory<2))throw new RangeError('Selected motor refractory time must be at least 2 ms');
  return {refractory,graded};
 });
 let previous=null;
 const exactTime=(value,label)=>{
  const tick=value/dtMs;
  if(!Number.isFinite(value)||value<0||!Number.isInteger(tick)||tick>=MAX_EXACT_FLOAT32_INTEGER||Math.fround(value)!==value)
   throw new RangeError(label+' must lie on an exactly representable 0.5 ms tick');
 };
 function read(state,timeMs){
  if(!(state instanceof Float32Array)||state.length!==selected.length*READ_STRIDE)
   throw new TypeError('Expected exactly nine Float32 readState values per selected neuron');
  exactTime(timeMs,'Read time');
  if(previous&&timeMs-previous.timeMs!==bodyBlockMs)
   throw new RangeError('Unobserved or repeated interval: read exactly once per 2 ms body block');
  // A changed model declaration invalidates the timing guarantee. This reader
  // cannot certify an independently mutated packed/native parameter buffer;
  // callers must keep the declared model and running brain identical.
  for(let k=0;k<selected.length;k++){
   const offset=selected[k]*PARAM_STRIDE,c=contract[k];
   if(params[offset+5]!==c.refractory||params[offset+8]!==c.graded)
    throw new Error('Selected motor model params changed after reader creation');
  }
  const ratesHz=new Float32Array(selected.length),counts=new Uint32Array(selected.length),lastTimes=new Float64Array(selected.length),events=[];
  for(let k=0;k<selected.length;k++){
   const offset=k*READ_STRIDE;
   for(let j=0;j<READ_STRIDE;j++)if(!Number.isFinite(state[offset+j]))throw new TypeError('Nonfinite selected motor state');
   const count=state[offset+3],rate=state[offset+4],rawLast=state[offset+8];
   const last=count>0&&dlm.has(selected[k])?decodeDlmEventTime(rawLast):rawLast;
   if(!Number.isInteger(count)||count<0||count>=MAX_EXACT_FLOAT32_INTEGER)
    throw new RangeError('Spike count must remain a nonnegative exact float32 integer below 2^24');
   if(rate<0||state[offset+2]<0)throw new RangeError('Negative motor rate or refractory state');
   if(count===0){if(!(last<0))throw new Error('Unspiked neuron requires a negative last-spike sentinel');}
   else{
    if(!dlm.has(selected[k]))exactTime(last,'Last-spike time');
    if(last<=0||last>timeMs)throw new Error('Last-spike time lies outside the observed history');
   }
   if(previous){
    const delta=count-previous.counts[k];
    if(delta<0)throw new Error('Spike count regression; rebaseline explicitly after a brain reset');
    if(delta>1)throw new Error('Spike count overrun: last timestamp cannot recover multiple events');
    if(delta===0){
     if(last!==previous.lastTimes[k])throw new Error('Last-spike timestamp changed without a new spike count');
    }else{
     if(!(last>previous.timeMs&&last<=timeMs))throw new Error('New spike timestamp is outside (previousTime,currentTime]');
     if(previous.counts[k]>0&&(dlm.has(selected[k])?Math.round(last*10)-Math.round(previous.lastTimes[k]*10)<100:last-previous.lastTimes[k]<2.5))
      throw new Error('Spike timestamps violate the conservative refractory interval');
     events.push({index:selected[k],timeMs:last});
    }
   }
   counts[k]=count;ratesHz[k]=rate;lastTimes[k]=last;
  }
  events.sort((a,b)=>a.timeMs-b.timeMs||a.index-b.index);
  const result={initialized:previous===null,fromTimeMs:previous?.timeMs??null,timeMs,indices:selected.slice(),ratesHz,counts,events};
  previous={timeMs,counts:counts.slice(),lastTimes};
  return result;
 }
 return Object.freeze({read,reset(){previous=null;}});
}
