// Fixed diagnostic: timestamped MN events -> excitation -> native WASM muscle.
// No neural simulation, body, oscillator, optimizer or gain fitting.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {createWingEventExcitation,DEFAULT_WING_EVENT_PRIORS} from '../web/flybody-wing-event-excitation.js';
import {analyzeMotorEventPair} from './analyze-flight-motor-events.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [outputArg,flag]=process.argv.slice(2);
assert(outputArg&&(!flag||flag==='--prepare-only')&&process.argv.length<=4,
 'Usage: node scripts/characterize-wing-event-muscles.mjs reports/new-output [--prepare-only]');
const output=path.resolve(outputArg),prepareOnly=flag==='--prepare-only',reports=path.join(root,'reports');
assert(output.startsWith(reports+path.sep));await fs.mkdir(output,{recursive:true});
assert((await fs.realpath(output)).startsWith((await fs.realpath(reports))+path.sep));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const sourceFiles=['scripts/characterize-wing-event-muscles.mjs','scripts/analyze-flight-motor-events.mjs',
 'web/flybody-wing-event-excitation.js','web/training/flight-parameters.js','packages/banc-runtime/src/wasm.js','packages/banc-runtime/src/model.js',
 'packages/banc-runtime/dist/core.js','packages/banc-runtime/dist/core.wasm','packages/banc-runtime/native/core.cpp',
 'data/prepared/banc888/io.json'];
const sourceBytes=Object.fromEntries(await Promise.all(sourceFiles.map(async file=>[file,await fs.readFile(path.join(root,file))])));
const sourceHashes=Object.fromEntries(Object.entries(sourceBytes).map(([file,bytes])=>[file,sha(bytes)]));
const captureFile='reports/flight-motor-event-observer/evaluation/001-with-motor-events-709cbeba-c183-46ce-8939-55a12b7d7e35.json';
const controlFile='reports/flight-motor-event-observer/evaluation/000-without-motor-events-709cbeba-c183-46ce-8939-55a12b7d7e35.json';
const captureBytes=await fs.readFile(path.join(root,captureFile)),controlBytes=await fs.readFile(path.join(root,controlFile));
const captured=JSON.parse(captureBytes),control=JSON.parse(controlBytes),io=JSON.parse(sourceBytes['data/prepared/banc888/io.json']);
const pair=analyzeMotorEventPair(captured,control,io,sourceHashes['data/prepared/banc888/io.json']);
assert.equal(pair.coverage.totalEvents,1460);assert.equal(captured.motorEvents.at(-1).timeMs,388);
for(const [file,url]of [['packages/banc-runtime/src/wasm.js','/banc-engine/src/wasm.js'],['packages/banc-runtime/src/model.js','/banc-engine/src/model.js'],
 ['packages/banc-runtime/dist/core.js','/banc-engine/dist/core.js'],['packages/banc-runtime/dist/core.wasm','/banc-engine/dist/core.wasm']])assert.equal(sourceHashes[file],captured.sourceHashes[url],'Recorded native muscle dependency changed: '+file);
const mappings=io.muscles.map((row,mappingIndex)=>({...row,mappingIndex})).filter(row=>['asynchronous_wing','wing_steering_assumption'].includes(row.kind));
const familyOf=m=>m.kind==='wing_steering_assumption'?'steering':m.target==='dorsal_longitudinal_muscle'?'dlm':'dvm';
const indices=[...new Set(mappings.flatMap(row=>row.indices))].sort((a,b)=>a-b),slotById=new Map(indices.map((id,k)=>[id,k]));
assert.equal(mappings.length,28);assert.equal(indices.length,48);
const unitIdentity=indices.map(index=>{const mapping=mappings.find(row=>row.indices.includes(index));assert(mapping);return {index,
 rootId:io.motor_neurons.find(row=>row.index===index).root_id,family:familyOf(mapping),side:mapping.joint.endsWith('_left')?'left':'right',
 target:mapping.target,mappingIndex:mapping.mappingIndex,mappingPosition:mappings.indexOf(mapping)};});
const cases=[...['steering','dlm','dvm'].flatMap(family=>(family==='steering'?[50,100,200]:[3,8,12]).map(rateHz=>({
 name:family+'-'+rateHz+'hz',kind:'periodic-synthetic',family,rateHz,durationMs:6000,analysisFromMs:4000,analysisToMs:6000}))),
 {name:'recorded-388ms',kind:'recorded-events',family:'all',durationMs:388,analysisFromMs:100,analysisToMs:280}];
const config=structuredClone(DEFAULT_WING_EVENT_PRIORS);
function* syntheticPackets(condition){
 const active=unitIdentity.filter(row=>row.family===condition.family).map(row=>row.index),byTick=new Map();
 for(let ordinal=1;ordinal<=condition.rateHz*condition.durationMs/1000;ordinal++){
  const tick=Math.round(2000*ordinal/condition.rateHz);assert(tick>0&&tick<=condition.durationMs*2);assert(!byTick.has(tick));byTick.set(tick,active);
 }
 const counts=Array(indices.length).fill(0),ratesHz=Array(indices.length).fill(0);
 yield {initialized:true,fromTimeMs:null,timeMs:0,indices:[...indices],ratesHz:[...ratesHz],counts:[...counts],events:[]};
 for(let endMs=2;endMs<=condition.durationMs;endMs+=2){
  const events=[];for(let tick=(endMs-2)*2+1;tick<=endMs*2;tick++)for(const index of byTick.get(tick)||[]){
   events.push({index,timeMs:tick*.5});counts[slotById.get(index)]++;
  }
  yield {initialized:false,fromTimeMs:endMs-2,timeMs:endMs,indices:[...indices],ratesHz:[...ratesHz],counts:[...counts],events};
 }
}
const packetsFor=condition=>condition.kind==='recorded-events'?captured.motorEvents:syntheticPackets(condition);
function packetDigest(condition){
 const hash=createHash('sha256'),counts=Array(indices.length).fill(0);let packets=0,events=0;
 for(const packet of packetsFor(condition)){hash.update(JSON.stringify(packet)+'\n');packets++;events+=packet.events.length;
  for(const event of packet.events)counts[slotById.get(event.index)]++;
 }
 return {sha256:hash.digest('hex'),packets,events,eventsPerUnit:counts};
}
const caseDefinitions=cases.map(row=>({...row,input:packetDigest(row)}));
const plan={schemaVersion:1,kind:'wing-event-native-muscle-characterization-plan',sourceHashes,
 scope:'One28-group native muscle object at a time. Nine fixed synthetic event conditions plus one unchanged recorded48-MN event train. No body/brain/controller/optimizer or force-normalization fitting.',
 artifacts:{[captureFile]:sha(captureBytes),[controlFile]:sha(controlBytes)},recordedPair:{configHash:pair.pairing.configHash,
 modelFingerprint:pair.pairing.modelFingerprint,physicsDigest:pair.pairing.physicsDigest,events:pair.coverage.totalEvents,simSeconds:pair.pairing.result.simSeconds},
 config,indices,unitIdentity,mappings: mappings.map(row=>({mappingIndex:row.mappingIndex,target:row.target,joint:row.joint,kind:row.kind,indices:row.indices})),
 cases:caseDefinitions,
 syntheticTiming:{gridMs:.5,formula:'eventTimeMs = 0.5*Math.round(2000*ordinal/rateHz), ordinal=1,2,...',
  ties:'Nearest0.5ms; exact half-grid ties round toward the later time. No event occurs at0.',
  phase:'Every unit of the selected family receives the same fixed phase. Other families are silent. This is a transfer-function test, not biological synchrony.',
  metadataRates:'Synthetic packet ratesHz are zero placeholders and never drive excitation. Reported rates count the generated events.',
  packetConvention:'Fresh time0 baseline, then consecutive2ms packets with events in(fromTimeMs,timeMs].'},
 muscle:{implementation:'WasmMuscles / native _muscle_step',count:28,intervalMs:1,inputStride:5,stateStride:3,
  mechanicalInput:{normalizedLength:1,positiveShorteningVelocity:0,Fmax:1,energy:1},initialState:'Zero activation,fatigue,force; zero per-neuron event kernels.',
  updateTiming:'At each1ms right boundary, pass that completed interval mean excitation to the native kernel with dt=.001. Returned force is available after the boundary; no body is integrated.',
  forceUnits:'Normalized native model force with Fmax=1. Not measured force or a reproduction of the moving/fatigued recorded body.'},
 analysis:{syntheticWarmupMs:4000,syntheticWindowMs:2000,recordedWindow:'(100,280]ms; full0–388ms input preserved and not extended.',
  periodicSettleCheck:'Compare intervalmean kernel/excitation and native activation against the same phase1000ms earlier throughout(4000,6000]ms. All chosen integer rates repeat their quantized schedule every1000ms.',
  settleTolerance:1e-6,fatigue:'Fatigue may drift despite periodic excitation/activation. Report its start,end and slope; do not declare force stationary.',
  highExcitationThreshold:.95,waveforms:'At most a few hundred selected1ms waveform rows per condition; all48units/all28mappings retain aggregate statistics.'},
 assumptions:['Steering1/5ms kernels and DLM6.2/82ms calcium-like kernels are explicit effector priors, not validated force kinetics.',
  'DVM initially borrows DLM kernel constants as a separate configurable assumption. Native15/40ms activation is an additional stage.',
  'Per-neuron nonlinearity precedes equal-weight averaging into each original muscle mapping. No rate-to-force normalization or learned gains.',
  'The actual motor-event input is unchanged; normalized fixed muscle conditions deliberately differ from its original moving body.']};
async function writeSameOrNew(file,bytes){try{assert((await fs.readFile(file)).equals(bytes),'Immutable prepared artifact changed: '+file);}catch(error){if(error.code!=='ENOENT')throw error;await fs.writeFile(file,bytes,{flag:'wx'});}}
await writeSameOrNew(path.join(output,'plan.json'),Buffer.from(JSON.stringify(plan,null,2)+'\n'));
await writeSameOrNew(path.join(output,'source.used.mjs'),sourceBytes['scripts/characterize-wing-event-muscles.mjs']);
await writeSameOrNew(path.join(output,'excitation-source.used.js'),sourceBytes['web/flybody-wing-event-excitation.js']);
if(prepareOnly){console.log(JSON.stringify({prepared:true,nativeExecution:false,plan:path.join(output,'plan.json'),cases:cases.length,recordedEvents:1460}));process.exit(0);}
assert(!(await fs.readdir(output)).some(name=>name==='result.json'||cases.some(row=>name===row.name+'.json')),'Use a directory without prior characterization results');
const summary={schemaVersion:1,kind:'wing-event-native-muscle-characterization',startedAt:new Date().toISOString(),
 planSha256:sha(await fs.readFile(path.join(output,'plan.json'))),sourceHashes,nodeVersion:process.version,cases:[],completed:false,error:null};
let stop=false,core;process.on('SIGINT',()=>{stop=true;});process.on('SIGTERM',()=>{stop=true;});
async function checkpoint(){await new Promise(resolve=>setTimeout(resolve,0));let marker=false;try{await fs.access(path.join(output,'STOP'));marker=true;}catch(error){if(error.code!=='ENOENT')throw error;}if(stop||marker){const error=new Error('Operator stopped characterization');error.name='AbortError';throw error;}}
async function verifyPins(){for(const [file,digest]of Object.entries(sourceHashes))assert.equal(sha(await fs.readFile(path.join(root,file))),digest);assert.equal(sha(await fs.readFile(path.join(root,captureFile))),sha(captureBytes));assert.equal(sha(await fs.readFile(path.join(root,controlFile))),sha(controlBytes));}
function accumulator(){return {n:0,sum:0,sum2:0,min:Infinity,max:-Infinity,high:0};}
function add(a,value){assert(Number.isFinite(value));a.n++;a.sum+=value;a.sum2+=value*value;a.min=Math.min(a.min,value);a.max=Math.max(a.max,value);if(value>=.95)a.high++;}
function finish(a){return a.n?{samples:a.n,min:a.min,max:a.max,mean:a.sum/a.n,rms:Math.sqrt(a.sum2/a.n),fractionAtLeast095:a.high/a.n}:null;}
function makeStats(fields,count){return Array.from({length:count},()=>Object.fromEntries(fields.map(key=>[key,accumulator()])));}
function finishStats(rows){return rows.map(row=>Object.fromEntries(Object.entries(row).map(([key,value])=>[key,finish(value)])));}
function waveformSelection(condition){
 const wanted=condition.family==='all'?['steering','dlm','dvm']:[condition.family],selected=[];
 for(const family of wanted)for(const side of ['left','right']){
  const rows=unitIdentity.filter(row=>row.family===family&&row.side===side);
  const row=rows.find(row=>family==='steering'?row.target==='i1_muscle':family==='dvm'?io.motor_neurons.find(n=>n.index===row.index)?.cell_type==='DVM1a-c':true)||rows[0];selected.push(row.index);
 }
 return selected;
}
try{
 await verifyPins();await checkpoint();core=await createWasmCore({wasmBinary:sourceBytes['packages/banc-runtime/dist/core.wasm']});assert.equal(typeof core._muscle_step,'function');
 for(const condition of cases){
  await checkpoint();await verifyPins();const interfaceModel=createWingEventExcitation({io,config}),muscles=new WasmMuscles(core,28);
  const record={name:condition.name,condition,input:caseDefinitions.find(row=>row.name===condition.name).input,completed:false,error:null,
   units:unitIdentity,mappings:plan.mappings,sourcePinsVerified:false,waveforms:[],quadratureDiscrepancyMax:0,excitationFloat32RoundingMax:0,
   settle:{required:condition.kind==='periodic-synthetic',checkedIntervals:0,kernelMaxError:0,unitExcitationMaxError:0,mappedExcitationMaxError:0,activationMaxError:0,passed:null},
   waveformUnits:waveformSelection(condition),intervals:0};
  const unitStats=makeStats(['meanKernel','meanExcitation'],48),mappedStats=makeStats(['intervalMeanExcitation','nativeActivation','nativeFatigue','nativeForce'],28);
  const mechanical=new Float32Array(28*5);for(let m=0;m<28;m++)mechanical.set([0,1,0,1,1],m*5);
  const ring=new Array(1000),selectedPositions=record.waveformUnits.map(id=>slotById.get(id));
  const selectedMappings=[...new Set(selectedPositions.map(k=>unitIdentity[k].mappingPosition))];let state=new Float32Array(84),startFatigue,endFatigue,analysisSamples=0;
  const hash=createHash('sha256');let eventTotal=0;
  try{
   const initialContract=interfaceModel.readState();assert.deepEqual(Array.from(initialContract.indices),indices);
   assert.deepEqual(Array.from(initialContract.mappingIndices),mappings.map(row=>row.mappingIndex));
   for(const packet of packetsFor(condition)){
    hash.update(JSON.stringify(packet)+'\n');eventTotal+=packet.events.length;interfaceModel.accept(packet);
    if(packet.initialized)continue;
    for(const endMs of [packet.timeMs-1,packet.timeMs]){
     const interval=interfaceModel.finishInterval(endMs);assert.equal(interval.timeMs,endMs);assert.equal(interval.fromTimeMs,endMs-1);
     assert.equal(interval.excitation.length,28);assert.equal(interval.unitExcitation.length,48);assert.equal(interval.unitMeanKernel.length,48);
     record.quadratureDiscrepancyMax=Math.max(record.quadratureDiscrepancyMax,interval.quadratureDiscrepancy);
     for(let m=0;m<28;m++){mechanical[m*5]=interval.excitation[m];record.excitationFloat32RoundingMax=Math.max(record.excitationFloat32RoundingMax,Math.abs(mechanical[m*5]-interval.excitation[m]));}
     state=muscles.step(mechanical,.001);record.intervals++;
     if(condition.kind==='periodic-synthetic'){
      unitIdentity.forEach((row,k)=>{if(row.family!==condition.family){assert.equal(interval.unitMeanKernel[k],0);assert.equal(interval.unitExcitation[k],0);}});
      mappings.forEach((row,m)=>{if(familyOf(row)!==condition.family){assert.equal(interval.excitation[m],0);assert(state.subarray(m*3,m*3+3).every(value=>value===0));}});
     }
     if(endMs===condition.analysisFromMs)startFatigue=Array.from({length:28},(_,m)=>state[m*3+1]);
     if(endMs>condition.analysisFromMs&&endMs<=condition.analysisToMs){
      analysisSamples++;for(let k=0;k<48;k++){add(unitStats[k].meanKernel,interval.unitMeanKernel[k]);add(unitStats[k].meanExcitation,interval.unitExcitation[k]);}
      for(let m=0;m<28;m++){add(mappedStats[m].intervalMeanExcitation,interval.excitation[m]);add(mappedStats[m].nativeActivation,state[m*3]);add(mappedStats[m].nativeFatigue,state[m*3+1]);add(mappedStats[m].nativeForce,state[m*3+2]);}
      if(condition.kind==='periodic-synthetic'){
       const previous=ring[(endMs-1)%1000];assert(previous&&previous.timeMs===endMs-1000);
       for(let k=0;k<48;k++){record.settle.kernelMaxError=Math.max(record.settle.kernelMaxError,Math.abs(previous.kernel[k]-interval.unitMeanKernel[k]));record.settle.unitExcitationMaxError=Math.max(record.settle.unitExcitationMaxError,Math.abs(previous.unit[k]-interval.unitExcitation[k]));}
       for(let m=0;m<28;m++){record.settle.mappedExcitationMaxError=Math.max(record.settle.mappedExcitationMaxError,Math.abs(previous.excitation[m]-interval.excitation[m]));record.settle.activationMaxError=Math.max(record.settle.activationMaxError,Math.abs(previous.activation[m]-state[m*3]));}
       record.settle.checkedIntervals++;
      }
     }
     if(endMs===condition.analysisToMs)endFatigue=Array.from({length:28},(_,m)=>state[m*3+1]);
     ring[(endMs-1)%1000]={timeMs:endMs,kernel:interval.unitMeanKernel.slice(),unit:interval.unitExcitation.slice(),excitation:interval.excitation.slice(),activation:Float32Array.from({length:28},(_,m)=>state[m*3])};
     const firstEvent=condition.kind==='periodic-synthetic'?.5*Math.round(2000/condition.rateHz):100;
     const sample=condition.kind==='periodic-synthetic'?(endMs<=20||(endMs>=firstEvent-3&&endMs<=firstEvent+40)||(endMs>4000&&endMs<=4040)):
      ((endMs>=95&&endMs<=140)||(endMs>=240&&endMs<=260));
     if(sample)record.waveforms.push({timeMs:endMs,unitMeanKernel:selectedPositions.map(k=>interval.unitMeanKernel[k]),unitMeanExcitation:selectedPositions.map(k=>interval.unitExcitation[k]),
      mappingPositions:selectedMappings,intervalMeanExcitation:selectedMappings.map(m=>interval.excitation[m]),nativeState:selectedMappings.map(m=>Array.from(state.subarray(m*3,m*3+3)))});
    }
    if(packet.timeMs%200===0)await checkpoint();
   }
   assert.equal(hash.digest('hex'),record.input.sha256,'Event packet sequence mutated');assert.equal(eventTotal,record.input.events);
   assert.equal(record.intervals,condition.durationMs);assert.equal(analysisSamples,condition.analysisToMs-condition.analysisFromMs);
   if(record.settle.required){record.settle.passed=['kernelMaxError','unitExcitationMaxError','mappedExcitationMaxError','activationMaxError'].every(key=>record.settle[key]<=plan.analysis.settleTolerance);
    assert(record.settle.passed,'Synthetic kernel/excitation/activation did not settle to the declared periodic tolerance');}
   record.analysisSamples=analysisSamples;record.unitStatistics=finishStats(unitStats);record.mappingStatistics=finishStats(mappedStats);
   record.fatigue={atAnalysisStart:startFatigue,atAnalysisEnd:endFatigue,meanSlopePerSecond:endFatigue.map((value,m)=>(value-startFatigue[m])/((condition.analysisToMs-condition.analysisFromMs)/1000)),
    interpretation:'Native fatigue is retained. A nonzero slope means force is not stationary even when kernel and activation are periodic.'};
   record.finalNativeState=Array.from(state);record.finalExcitationState=interfaceModel.snapshot();
   await verifyPins();record.sourcePinsVerified=true;record.completed=true;
  }catch(error){record.error=error.stack||error.message;throw error;}
  finally{muscles.dispose();await fs.writeFile(path.join(output,condition.name+'.json'),JSON.stringify(record)+'\n',{flag:'wx'});summary.cases.push({name:condition.name,file:condition.name+'.json',completed:record.completed,error:record.error});}
  console.log(JSON.stringify({kind:'case-complete',name:condition.name,intervals:record.intervals,analysisSamples,settled:record.settle.passed}));
 }
 summary.completed=true;
}catch(error){summary.error=error.stack||error.message;process.exitCode=1;}
finally{summary.finishedAt=new Date().toISOString();await fs.writeFile(path.join(output,'result.json'),JSON.stringify(summary,null,2)+'\n',{flag:'wx'});}
console.log(JSON.stringify({output,completed:summary.completed,cases:summary.cases.length,error:summary.error}));
