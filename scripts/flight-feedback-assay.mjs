// Diagnostic only: exact WASM branches from an actual completed flight block.
// No native body integration, parameter fitting, coordinator writes or CLI run.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {WasmBrain} from '../packages/banc-runtime/src/wasm.js';
import {historySlots,SPIKE_CAPACITY} from '../packages/banc-runtime/src/model.js';
import {decodeDlmEventTime} from '../packages/banc-runtime/src/cell-models.js';
import {createWingMotorEventReader} from '../packages/banc-runtime/src/motor-events.js';
import {createWingEventExcitation} from '../web/flybody-wing-event-excitation.js';
import {createMotorDecoder} from '../web/motor-decoder.js';

const records=new WeakMap(),TAU=2*Math.PI,EPS=1e-8;
assert.equal(new Uint8Array(new Uint32Array([1]).buffer)[0],1,'Diagnostic encoding requires a little-endian host');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const bytes=value=>new Uint8Array(value.buffer,value.byteOffset,value.byteLength);
const pack=value=>Buffer.from(bytes(value)).toString('base64');
const clone=value=>structuredClone(value);
const near=(a,b,label)=>assert(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<EPS,label);
const ownArray=(value,n,label)=>{assert((Array.isArray(value)||ArrayBuffer.isView(value))&&value.length===n&&Array.from(value).every(Number.isFinite),label);return Array.from(value);};
function indices(value,n,label){const result=Uint32Array.from(value??[]);assert(result.length>0&&new Set(result).size===result.length&&Array.from(value??[]).every(x=>Number.isInteger(x)&&x>=0&&x<n),label);return result;}
function regions(brain){
 const n=brain.n,layout=brain.intrinsic;
 return {params:[brain.params,layout.packedLength*4],state0:[brain.states[0],n*8*4],state1:[brain.states[1],n*8*4],
  history:[brain.history,n*historySlots(brain.model)*4],kinetics:[brain.kinetics,layout.kineticsLength*4],events:[brain.events,(2+SPIKE_CAPACITY*2)*4]};
}
function forwardHashes(brain){
 brain.live();const result={tick:brain.tick};
 for(const [name,[pointer,length]]of Object.entries(regions(brain)))result[name]=sha(brain.core.HEAPU8.subarray(pointer,pointer+length));
 return result;
}
function fork(source){
 source.live();const before=forwardHashes(source),child=new WasmBrain(source.core,source.model,{shared:source});
 try{
  const target=regions(child);
  // Allocation may grow the shared WASM heap. Acquire every heap view afresh.
  for(const [name,[pointer,length]]of Object.entries(regions(source))){assert.equal(target[name][1],length);child.core.HEAPU8.set(source.core.HEAPU8.subarray(pointer,pointer+length),target[name][0]);}
  child.tick=source.tick;assert.deepEqual(forwardHashes(child),before,'Cloned neural forward state');
  assert.deepEqual(forwardHashes(source),before,'Capture changed source neural state');return child;
 }catch(error){child.dispose();throw error;}
}
function get(snapshot){const record=records.get(snapshot);assert(record&&!record.disposed,'Flight snapshot disposed or unknown');return record;}

/** Synchronous observer capture. A retained, unadvanced child owns shared graph
 * references after the source episode is disposed. Its heap state stays private.
 * input/inputSequence are the JUST-CONSUMED production block, not invented future
 * sensory currents. Later branches explicitly hold this context fixed. */
export function captureWasmFlightState(context,{afferentIndices}={}){
 const {brain,body,world,input,inputSequence}=context;
 assert(brain?.backend==='wasm'&&brain.model.manifest.dt_ms===.5,'Actual 0.5 ms WASM brain required');brain.live();
 assert(context.phase!=='warmup','Capture only after native warmup release');
 const timeMs=brain.timeMs;assert(timeMs>=500&&Number.isInteger(timeMs/2),'Capture at a completed post-warmup 2 ms boundary');
 near(body.time,timeMs/1000,'Body/neural capture clocks');near(body.remainder??0,0,'Unconsumed body remainder');
 for(const name of ['qfrc_applied','xfrc_applied'])assert(body.data[name]?.every(x=>x===0),'Capture has an external force');
 assert.equal(body.environmentContactCount,0,'Capture must be contact-free');
 const event=body._wingMotorEvents,eventSnapshot=event?.adapter.snapshot();
 assert(event?.initialized&&event.elapsedMs===timeMs&&event.observedMs===timeMs&&eventSnapshot.pending===null,'Capture requires fully consumed motor events');
 assert.equal(eventSnapshot.integratedThroughMs,timeMs);assert.equal(eventSnapshot.observedThroughMs,timeMs);
 const decoderSnapshot=body.motorDecoder?.snapshot?.();assert(decoderSnapshot,'Motor decoder snapshot API required');
 assert.equal(decoderSnapshot.timeMs,timeMs,'Decoder/neural capture clocks');
 const io=world?.io??brain.model.io,decoder=createMotorDecoder(io,decoderSnapshot.weights);decoder.restore(decoderSnapshot);
 const wingIndices=indices(decoder.contract.indices,brain.n,'Invalid actual wing identities');assert.equal(wingIndices.length,48);
 assert.deepEqual(Array.from(wingIndices),eventSnapshot.contract.indices,'Decoder/event identities');
 assert(input instanceof Float32Array&&input.length===brain.n&&input.every(Number.isFinite),'Capture needs actual full current vector');
 const schedule=inputSequence===null||inputSequence===undefined?[input,input,input,input]:inputSequence;
 assert(Array.isArray(schedule)&&schedule.length===4&&schedule.every(row=>row instanceof Float32Array&&row.length===brain.n&&row.every(Number.isFinite)),'Invalid captured 0.5 ms current schedule');
 const internal=Object.fromEntries(['hunger','insulin','akh'].map(key=>[key,body.internal[key]]));
 assert(Object.values(internal).every(x=>Number.isFinite(x)&&x>=0&&x<=1),'Invalid captured internal state');
 const nativeSample={halterePower:ownArray(body.halterePower,2,'Native haltere power required'),
  omegaRootRadS:ownArray(body.data.qvel.slice(3,6),3,'Native angular velocity required'),
  wingPhaseRadians:body.wings.phase,wingFrequencyHz:body.wings.frequencyHz};
 assert(Number.isFinite(nativeSample.wingPhaseRadians)&&Number.isFinite(nativeSample.wingFrequencyHz)&&nativeSample.wingFrequencyHz>=0,'Invalid native wing oscillator');
 assert(nativeSample.halterePower.every(x=>x>=0&&x<=1),'Invalid native haltere power');
 const retained=fork(brain),baselineHashes=forwardHashes(retained);
 try{
  const encoder=context.encoder,encoderState=encoder?Object.fromEntries(['ratesHz','adapted','lightDrive','lastSequence','lastFrameTime','lastKey','contrast'].filter(key=>encoder[key]!==undefined).map(key=>[key,clone(encoder[key])])):null;
  const record={brain:retained,disposed:false,baselineHashes,io,wingIndices,afferentIndices:afferentIndices===undefined?null:indices(afferentIndices,brain.n,'Invalid captured afferent identities'),
   input:input.slice(),schedule:schedule.map(row=>row.slice()),internal,nativeSample,eventSnapshot:clone(eventSnapshot),decoderSnapshot:clone(decoderSnapshot)};
  const snapshot=Object.freeze({kind:'wasm-flight-forward-snapshot',timeMs,bodyTimeSeconds:body.time,baselineHashes:Object.freeze({...baselineHashes}),
   inputContext:'Repeated last-delivered 2 ms sensory current schedule; frozen body/internal context with advancing prescribed wing phase',
   source:clone(context.provenance??null),sensoryFeedback:clone(context.fly?.feedback??null),sensorySample:clone(encoder?.sample??null),encoderState,
   nativeSample:clone(nativeSample),decoderFeatureSha256:sha(bytes(decoder.features(nativeSample.wingPhaseRadians))),
   dispose(){if(!record.disposed){record.disposed=true;record.brain.dispose();}}});
  records.set(snapshot,record);return snapshot;
 }catch(error){retained.dispose();throw error;}
}

function read(brain,selected){return brain.readState(selected,{includeSpikeTime:true,includeStatistics:false,includeSpikeHistory:false});}
function tickEvents(previous,current,selected,from,to,model){
 const events=[],dlm=new Set(model.manifest.intrinsic_models?.cells.map(c=>c.index)??[]);
 for(let k=0;k<selected.length;k++){
  const id=selected[k],count=current[k*9+3],delta=count-previous[k*9+3];
  assert(Number.isInteger(count)&&count>=0&&count<2**24&&Number.isInteger(delta)&&delta>=0&&delta<=1,'Selected event count overrun');
  if(delta){assert(model.params[id*16+8]<=.5,'Graded afferent cannot emit invented spikes');
   const raw=current[k*9+8],timeMs=dlm.has(id)?decodeDlmEventTime(raw):raw;
   assert(timeMs>from&&timeMs<=to,'Selected event timestamp outside neural interval');
   events.push({index:id,timeMs,relativeMs:timeMs-from});
  }
 }
 return events.sort((a,b)=>a.timeMs-b.timeMs||a.index-b.index);
}
function summarize(arm,afferents,wingIndices,cap){
 const all=Array.from(afferents),wing=Array.from(wingIndices),index=new Map([...all,...wing].map((id,k)=>[id,k]));
 const counts=Array(index.size).fill(0);let firstAfferentEventMs=null,firstWingEventMs=null,maximumRateHz=0,currentSum=0,currentMax=0,saturated=0,currentCount=0;
 for(const tick of arm.trace){
  for(const event of tick.events){counts[index.get(event.index)]++;const elapsed=event.timeMs-arm.startTimeMs;if(index.get(event.index)<all.length)firstAfferentEventMs??=elapsed;else firstWingEventMs??=elapsed;}
  for(const value of tick.actualCurrentsPa){currentSum+=value;currentMax=Math.max(currentMax,value);saturated+=cap>0&&value>=cap*.95;currentCount++;}
  maximumRateHz=Math.max(maximumRateHz,tick.maximumSelectedRateHz);
 }
 return {afferentEventCounts:counts.slice(0,all.length),wingEventCounts:counts.slice(all.length),firstAfferentEventMs,firstWingEventMs,
  maximumSelectedRateHz:maximumRateHz,meanDrivenCurrentPa:currentCount?currentSum/currentCount:0,maximumDrivenCurrentPa:currentMax,
  fractionDrivenCurrentsAtLeast95PercentOfCap:currentCount?saturated/currentCount:0};
}

function firstEventContrast(positive,negative,selected,startTimeMs){
 const wanted=new Set(selected),key=e=>e.index+':'+e.timeMs;
 const left=positive.trace.flatMap(t=>t.events).filter(e=>wanted.has(e.index));
 const right=negative.trace.flatMap(t=>t.events).filter(e=>wanted.has(e.index));
 const leftKeys=new Set(left.map(key)),rightKeys=new Set(right.map(key));
 const changed=[...left.filter(e=>!rightKeys.has(key(e))),...right.filter(e=>!leftKeys.has(key(e)))];
 return changed.length?Math.min(...changed.map(e=>e.timeMs))-startTimeMs:null;
}

function firstDifferenceTime(left,right,field,startTimeMs){
 const row=left.find((value,k)=>JSON.stringify(value[field])!==JSON.stringify(right[k][field]));
 return row?row.timeMs-startTimeMs:null;
}

function unpack(value,Type){const buffer=Buffer.from(value,'base64');return new Type(buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength));}
function meanState(arm){
 const sum=new Float64Array(unpack(arm.trace[0].state9,Float32Array).length);
 for(const row of arm.trace){const state=unpack(row.state9,Float32Array);for(let k=0;k<sum.length;k++)sum[k]+=state[k]/arm.trace.length;}
 return sum;
}
const norm=values=>Math.hypot(...values);
function cosine(a,b){const denominator=norm(a)*norm(b);return denominator>0?a.reduce((sum,value,k)=>sum+value*b[k],0)/denominator:null;}
function layerContrasts(sham,positive,negative,selected){
 const states=[sham,positive,negative].map(meanState),fields={voltage:0,rateHz:4,release:5};
 const meanNeuralResponse=Array.from(selected,(index,k)=>({index,...Object.fromEntries(Object.entries(fields).map(([field,slot])=>{
  const base=states[0][k*9+slot],p=states[1][k*9+slot],n=states[2][k*9+slot];
  return [field,{positiveMinusSham:p-base,negativeMinusSham:n-base,odd:(p-n)/2,even:(p+n)/2-base}];
 }))}));
 const meanOddFeature=new Float64Array(672),meanEvenFeature=new Float64Array(672),featureResponse=[];
 for(let row=0;row<sham.decoder.length;row++){
  const [base,p,n]=[sham,positive,negative].map(arm=>unpack(arm.decoder[row].features,Float64Array));
  const odd=Float64Array.from(p,(value,k)=>(value-n[k])/2),even=Float64Array.from(p,(value,k)=>(value+n[k])/2-base[k]);
  for(let k=0;k<672;k++){meanOddFeature[k]+=odd[k]/sham.decoder.length;meanEvenFeature[k]+=even[k]/sham.decoder.length;}
  featureResponse.push({timeMs:sham.decoder[row].timeMs,relativeMs:sham.decoder[row].timeMs-sham.startTimeMs,oddNorm:norm(odd),evenNorm:norm(even),
   positiveOutput:clone(positive.decoder[row].output),negativeOutput:clone(negative.decoder[row].output),shamOutput:clone(sham.decoder[row].output)});
 }
 return {meanNeuralResponse,meanOddFeature:Array.from(meanOddFeature),meanEvenFeature:Array.from(meanEvenFeature),featureResponse,
  interpretation:'Odd/even means are descriptive for this finite pulse and frozen state. No axis tuning, flight stabilization or significance is inferred.'};
}

/** haltereMapper is either an existing mapper for one matching cap, or a
 * factory({maxCurrentPa}) returning the pinned mapper at each declared cap.
 * Signed controls replace native omega_y with +/-pitchRateRadS during a pulse;
 * omega_x/z are retained, so + and - have identical angular-speed magnitudes.
 * Folded controls feed abs(omega_y) to the SAME mapper: explicitly artificial
 * sign blindness, not a reproduction of the legacy sensory-rate encoder. */
export async function runFlightFeedbackAssay(snapshot,{haltereMapper,maxCurrentPa=[800],phaseOffsetsRadians=[0,Math.PI],
 afferentIndices,pitchRateRadS=5,pulseMs=10,recoveryMs=30,checkpoint=async()=>{},onArm}={}){
 const source=get(snapshot),caps=Array.isArray(maxCurrentPa)?maxCurrentPa:[maxCurrentPa];
 assert(caps.length>0&&caps.length<=8&&new Set(caps).size===caps.length&&caps.every(x=>Number.isFinite(x)&&x>0&&x<=10000),'Invalid current-cap grid');
 assert(Array.isArray(phaseOffsetsRadians)&&phaseOffsetsRadians.length>0&&phaseOffsetsRadians.length<=4&&phaseOffsetsRadians.every(Number.isFinite),'Invalid phase offsets');
 assert(Number.isFinite(pitchRateRadS)&&pitchRateRadS>0&&pitchRateRadS<=100,'Invalid pitch-rate probe');
 assert([pulseMs,recoveryMs].every(x=>Number.isInteger(x)&&x>=2&&x<=100&&x%2===0),'Probe windows must be positive even milliseconds');
 assert(typeof checkpoint==='function'&&(onArm===undefined||typeof onArm==='function'),'Invalid diagnostic callback');
 const report={schemaVersion:1,kind:'phase-resolved-flight-feedback-assay',completed:false,startTimeMs:snapshot.timeMs,
  source:snapshot.source,baselineHashes:source.baselineHashes,nativeSample:clone(source.nativeSample),
  scope:'Actual live-conditioned WASM neural state; subsequent sensory/body/internal context frozen. Neural branches only; no body integration or successful-flight claim.',
  protocol:{caps,phaseOffsetsRadians,pitchRateRadS,pulseMs,recoveryMs,neuralDtMs:.5,motorPacketMs:2,decoderIntervalMs:1,
   sham:'Recompute the mechanical haltere prior at captured native angular velocity; not an unmodified live continuation',
   unsignedControl:'Fold the prescribed pitch-rate sign before the same mechanical prior; not legacy-encoder parity',
   currentContext:snapshot.inputContext,phaseOffsets:'Explicit sensory-prior phase offsets; decoder samples an analytic continuation of the captured phase/frequency at 1ms boundaries. No future native phase is observed.'},
  encoding:{state9:'Float32 native little-endian base64, afferents then48 wing units, fields V,adaptation,refractory,count,rate,release,conductance,current,lastSpikeTime',
   features:'Float64 native little-endian base64 in the pinned672-parameter order'},groups:[]};
 assert.deepEqual(forwardHashes(source.brain),source.baselineHashes,'Retained snapshot mutated before assay');
 for(const cap of caps){
  const mapper=typeof haltereMapper==='function'?await haltereMapper({maxCurrentPa:cap}):haltereMapper;
  assert(mapper&&typeof mapper.writeInto==='function'&&mapper.maxCurrentPa===cap&&mapper.neuronCount===source.brain.n,'Mapper/cap/model mismatch');
  const driven=indices(mapper.indices,source.brain.n,'Invalid haltere current identities');
  const afferents=indices(afferentIndices??source.afferentIndices??driven,source.brain.n,'Invalid afferent readout identities');
  assert(Array.from(driven).every(id=>afferents.includes(id)),'Readout must include every driven afferent');
  assert(!Array.from(afferents).some(id=>source.wingIndices.includes(id)),'Afferent/motor identities overlap');
  const selected=Uint32Array.from([...afferents,...source.wingIndices]),drivenMask=new Uint8Array(source.brain.n);
  for(const id of driven)drivenMask[id]=1;
  report.afferentIndices??=Array.from(afferents);assert.deepEqual(Array.from(afferents),report.afferentIndices,'Mapper grid changes afferent identities');
  report.wingIndices??=Array.from(source.wingIndices);
  report.gradedAfferents??=Array.from(afferents).filter(id=>source.brain.model.params[id*16+8]>.5);
  report.initialState9??=pack(read(source.brain,selected));
  report.initialDecoderSnapshot??=clone(source.decoderSnapshot);
  report.initialEventAdapterSnapshot??=clone(source.eventSnapshot);
  report.inputScheduleSha256??=source.schedule.map(input=>sha(bytes(input)));
  report.frozenInternal??=clone(source.internal);
  report.gradedReadout='A graded afferent may have no spikes; its voltage and release are retained in state9.';
  for(const phaseOffset of phaseOffsetsRadians){
   const group={maxCurrentPa:cap,phaseOffsetRadians:phaseOffset,arms:[],duplicateShamExact:false,unsignedPairExact:false};report.groups.push(group);
   for(const name of ['sham-a','sham-b','pitch-positive','pitch-negative','unsigned-positive','unsigned-negative']){
    await checkpoint();const brain=fork(source.brain);
    const arm={name,startTimeMs:snapshot.timeMs,trace:[],motorPackets:[],decoder:[],finalHashes:null};
    try{
    const adapter=createWingEventExcitation({io:source.io,config:source.eventSnapshot.contract.config,eventContract:source.eventSnapshot.contract.eventContract});
    adapter.restore(source.eventSnapshot);const decoder=createMotorDecoder(source.io,source.decoderSnapshot.weights);decoder.restore(source.decoderSnapshot);
    assert.equal(sha(bytes(decoder.features(source.nativeSample.wingPhaseRadians))),snapshot.decoderFeatureSha256,'Exact decoder history restoration');
    const reader=createWingMotorEventReader({indices:source.wingIndices,params:brain.model.params,dtMs:.5,bodyBlockMs:2,eventContract:brain.eventContract});
    const baseline=read(brain,source.wingIndices),initialPacket=reader.read(baseline,brain.timeMs);
    assert(initialPacket.initialized&&initialPacket.timeMs===snapshot.timeMs,'Nonzero-time reader baseline');
    assert.deepEqual(Array.from(initialPacket.counts),source.eventSnapshot.counts,'Reader/adapter baseline counts');
    // Do not deliver this baseline to the restored adapter: it already owns T.
     let previous=read(brain,selected);
     for(let tick=0;tick<(pulseMs+recoveryMs)*2;tick++){
      await checkpoint();const elapsed=tick*.5,block=Math.floor(tick/4),offset=tick%4;
      const omega=source.nativeSample.omegaRootRadS.slice(),sign=name.endsWith('negative')?-1:1;
      if(elapsed<pulseMs&&!name.startsWith('sham'))omega[1]=sign*pitchRateRadS;
      if(name.startsWith('unsigned')&&elapsed<pulseMs)omega[1]=Math.abs(omega[1]);
      const input=source.schedule[offset].slice(),sample={...source.nativeSample,omegaRootRadS:omega,
       wingPhaseRadians:source.nativeSample.wingPhaseRadians+TAU*source.nativeSample.wingFrequencyHz*block*.002+phaseOffset,elapsedSeconds:offset*.0005};
      const diagnostics=mapper.writeInto(input,sample),requested=Array.from(driven,id=>input[id]);
      // Only the declared mapper indices may overwrite the frozen current vector.
      for(let i=0;i<input.length;i++)if(!drivenMask[i]&&input[i]!==source.schedule[offset][i])throw new Error('Unexpected non-haltere input change');
      const from=brain.timeMs;brain.step(1,input,source.internal,true);const current=read(brain,selected);
      assert(current.every(Number.isFinite),'Nonfinite selected neural readout');
      const actual=Array.from(driven,id=>brain.core.HEAPF32[brain.params/4+brain.n*16+id]);assert.deepEqual(actual,requested,'Current actually consumed by native WASM');
      const events=tickEvents(previous,current,selected,from,brain.timeMs,brain.model).map(event=>({...event,relativeMs:event.timeMs-snapshot.timeMs}));
      arm.trace.push({fromTimeMs:from,timeMs:brain.timeMs,relativeMs:brain.timeMs-snapshot.timeMs,omegaRootRadS:omega,
       sensoryPhaseRadians:sample.wingPhaseRadians+TAU*sample.wingFrequencyHz*sample.elapsedSeconds,
       requestedCurrentsPa:requested,actualCurrentsPa:actual,state9:pack(current),events,
       maximumSelectedRateHz:Math.max(...Array.from(selected,(_,k)=>current[k*9+4])),
       prior:diagnostics?{kind:diagnostics.kind,maxCurrentPa:diagnostics.maxCurrentPa}:null});previous=current;
      if(offset===3){
       const packet=reader.read(read(brain,source.wingIndices),brain.timeMs);adapter.accept(packet);
       arm.motorPackets.push({timeMs:packet.timeMs,counts:Array.from(packet.counts),ratesHz:Array.from(packet.ratesHz),events:clone(packet.events)});
       for(const endpoint of [brain.timeMs-1,brain.timeMs]){
        const interval=adapter.finishInterval(endpoint);decoder.advance(interval.unitExcitation);
        assert.equal(decoder.timeMs,endpoint,'Decoder/adapter absolute clock');
        const phase=source.nativeSample.wingPhaseRadians+TAU*source.nativeSample.wingFrequencyHz*(endpoint-snapshot.timeMs)/1000;
        arm.decoder.push({timeMs:endpoint,phaseRadians:phase,unitExcitation:pack(interval.unitExcitation),features:pack(decoder.features(phase)),output:decoder.sample(phase)});
       }
      }
     }
     arm.finalHashes=forwardHashes(brain);arm.summary=summarize(arm,afferents,source.wingIndices,cap);
    }finally{brain.dispose();}
    group.arms.push(arm);
    if(name==='sham-b'){assert.deepEqual(arm.trace,group.arms[0].trace,'Duplicate sham trace differs');assert.deepEqual(arm.motorPackets,group.arms[0].motorPackets);assert.deepEqual(arm.decoder,group.arms[0].decoder);assert.deepEqual(arm.finalHashes,group.arms[0].finalHashes);group.duplicateShamExact=true;}
    if(name==='unsigned-negative'){const positive=group.arms.find(a=>a.name==='unsigned-positive');assert.deepEqual(arm.trace,positive.trace,'Sign-folded control differs');assert.deepEqual(arm.finalHashes,positive.finalHashes);assert.deepEqual(arm.decoder,positive.decoder);group.unsignedPairExact=true;}
    await onArm?.(clone(arm),{maxCurrentPa:cap,phaseOffsetRadians:phaseOffset});
   }
   const sham=group.arms[0],positive=group.arms[2],negative=group.arms[3];
   const contrast=(field)=>positive.summary[field].map((v,i)=>({positiveMinusSham:v-sham.summary[field][i],negativeMinusSham:negative.summary[field][i]-sham.summary[field][i],odd:(v-negative.summary[field][i])/2,even:(v+negative.summary[field][i])/2-sham.summary[field][i]}));
   const currentDifference=positive.trace.find((row,i)=>JSON.stringify(row.actualCurrentsPa)!==JSON.stringify(negative.trace[i].actualCurrentsPa));
   group.response={afferentCounts:contrast('afferentEventCounts'),wingCounts:contrast('wingEventCounts'),
    firstDifferentCurrentMs:currentDifference?currentDifference.fromTimeMs-snapshot.timeMs:null,
    firstDifferentAfferentEventMs:firstEventContrast(positive,negative,afferents,snapshot.timeMs),
    firstDifferentWingEventMs:firstEventContrast(positive,negative,source.wingIndices,snapshot.timeMs),
    firstDifferentWingPacketMs:firstDifferenceTime(positive.motorPackets,negative.motorPackets,'events',snapshot.timeMs),
    firstDifferentDecoderFeatureMs:firstDifferenceTime(positive.decoder,negative.decoder,'features',snapshot.timeMs),
    ...layerContrasts(sham,positive,negative,selected)};
  }
 }
 report.phaseConsistency=caps.flatMap(cap=>{
  const groups=report.groups.filter(group=>group.maxCurrentPa===cap),base=groups[0];
  return groups.slice(1).map(group=>({maxCurrentPa:cap,referencePhaseOffsetRadians:base.phaseOffsetRadians,phaseOffsetRadians:group.phaseOffsetRadians,
   meanOddFeatureCosine:cosine(base.response.meanOddFeature,group.response.meanOddFeature),
   meanOddWingCountCosine:cosine(base.response.wingCounts.map(row=>row.odd),group.response.wingCounts.map(row=>row.odd)),
   meaning:'Descriptive alignment under the same native decoder phase and different sensory-prior phases. Null means no nonzero contrast; this is not a generalization or controllability proof.'}));
 });
 assert.deepEqual(forwardHashes(source.brain),source.baselineHashes,'Retained snapshot changed during branches');report.completed=true;
 return report;
}
