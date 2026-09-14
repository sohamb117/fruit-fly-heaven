// Artificial frozen-context neural identification. No body integration,
// receptor-axis assignments, optimizer, coordinator writes or browser activity.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {parseArgs} from 'node:util';
import {loadBancModel} from '../packages/banc-runtime/src/model.js';
import {WebGPUBrain} from '../packages/banc-runtime/src/webgpu.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {createBancTasteMapper} from '../web/banc-taste.js';
import {createBancSensoryCurrentMapper} from '../web/banc-sensory-current.js';
import {createHabitat} from '../web/body-world.js';
import {createMotorExcitation} from '../web/flybody-motor-excitation.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const cli=parseArgs({options:{'prefix-ms':{type:'string',default:'100'},'max-pulse-hz':{type:'string',default:'5'},
 groups:{type:'string',default:'all4'},'reference-result':{type:'string'},describe:{type:'boolean',default:false}},allowPositionals:true});
const [endpoint,outputArg]=cli.positionals;
if(cli.positionals.length!==2)throw new Error('Usage: node scripts/identify-flight-afferent-response.mjs http://127.0.0.1:PORT/ reports/output [--prefix-ms 100|300] [--max-pulse-hz 1.25|2.5|5] [--groups active3|all4] [--reference-result reports/historical/result.json] [--describe]');
const prefixMs=Number(cli.values['prefix-ms']),maxPulseHz=Number(cli.values['max-pulse-hz']),groupSelection=cli.values.groups;
assert([100,300].includes(prefixMs),'prefix-ms must be100 or300');
assert([1.25,2.5,5].includes(maxPulseHz),'max-pulse-hz must be1.25,2.5 or historical5');
assert(['active3','all4'].includes(groupSelection),'groups must be active3 or all4');
const selectedGroupKeys=groupSelection==='active3'?['haltere/right','wing_base/left','wing_base/right']:['haltere/left','haltere/right','wing_base/left','wing_base/right'];
const expectedArmCount=2+2*selectedGroupKeys.length;
const origin=new URL(endpoint),output=path.resolve(outputArg);
assert(origin.protocol==='http:'&&['127.0.0.1','localhost'].includes(origin.hostname)&&origin.pathname==='/'&&!origin.username&&!origin.password&&!origin.search&&!origin.hash,'Loopback origin required');
assert(output.startsWith(path.join(root,'reports')+path.sep));
if(cli.values.describe){console.log(JSON.stringify({endpoint:origin.href,output,prefixMs,maxPulseHz,groupSelection,selectedGroupKeys,expectedArmCount,referenceResult:cli.values['reference-result']??null,neuralExecution:false}));process.exit(0);}
await fs.mkdir(output,{recursive:true});
assert((await fs.realpath(output)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep));
assert(!(await fs.readdir(output)).some(name=>name.endsWith('.json')),'Use a new output directory');
const sha=value=>createHash('sha256').update(value).digest('hex');
const pack=values=>{const bytes=Buffer.alloc(values.length*4);for(let i=0;i<values.length;i++)bytes.writeFloatLE(values[i],i*4);return bytes.toString('base64');};
const unpack=text=>{const bytes=Buffer.from(text,'base64');return Float32Array.from({length:bytes.length/4},(_,i)=>bytes.readFloatLE(i*4));};
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
function localAsset(url){const entry=prefixes.find(([prefix])=>url.startsWith(prefix));const file=path.resolve(root,entry?entry[1]+url.slice(entry[0].length):'web/'+url.slice(1));assert(file.startsWith(root+path.sep));return file;}
const networkFetch=globalThis.fetch;globalThis.location={href:origin.href};
globalThis.fetch=(input,options={})=>{
 let url=new URL(typeof input==='string'?input:input.url||String(input),origin);
 if(url.href===pathToFileURL(path.join(root,'packages/banc-runtime/src/neural.wgsl')).href)url=new URL('/banc-engine/src/neural.wgsl',origin);
 const method=String(options.method||input?.method||'GET').toUpperCase();
 assert(url.origin===origin.origin&&['GET','HEAD'].includes(method),'Only same-origin reads are allowed');
 return networkFetch(url,{...options,method,redirect:'error',signal:options.signal||AbortSignal.timeout(30000)});
};
async function get(url){const response=await fetch(url);assert(response.ok,`Missing ${url}`);return Buffer.from(await response.arrayBuffer());}
// Pin only dependencies used by this neural-only assay. Body controller/model
// changes are deliberately outside its scope; no coordinator config is needed.
const actualAssets=['/banc-engine/src/model.js','/banc-engine/src/webgpu.js','/banc-engine/src/wasm.js',
 '/banc-engine/dist/core.js','/banc-engine/dist/core.wasm','/banc-engine/src/neural.wgsl',
 '/sensory-encoder.js','/body-world.js','/color-vision.js','/banc-ground-sense.js','/banc-taste.js',
 '/banc-sensory-current.js','/flybody-motor-excitation.js','/banc-data/manifest.json','/banc-data/io.json',
 '/banc-data/console/groups.json','/banc-data/console/sensory-inputs.json','/body-model/banc-taste-peg-annotations.json'];
const sourceHashes={},assetBytes={};
for(const url of actualAssets){
 const bytes=await get(url),digest=sha(bytes);
 assert.equal(sha(await fs.readFile(localAsset(url))),digest,'Local executable/data differs from served asset: '+url);
 sourceHashes[url]=digest;assetBytes[url]=bytes;
}
const sourceFiles={capture:'reports/flight-live-variants/no-adhesion-power15-steering005.json',
 context:'reports/flight-sensory-observability/result.json',muscles:'reports/steering-recruitment-calibration/legacy-replay/muscle-profiles.json',
 recruitment:'reports/steering-recruitment-calibration/hill80-n1/candidate.json',
 backend:'reports/native-webgpu-tooling/backend.mjs',backendLock:'reports/native-webgpu-tooling/package-lock.json'};
const sourceBytes=Object.fromEntries(await Promise.all(Object.entries(sourceFiles).map(async([key,file])=>[key,await fs.readFile(file)])));
const original=JSON.parse(sourceBytes.capture),capture=original.evaluation.motorReplay,context=JSON.parse(sourceBytes.context),muscleCapture=JSON.parse(sourceBytes.muscles);
assert.equal(muscleCapture.baselineGate.passed,true);assert.equal(context.probeTimeSeconds,.2);
assert.equal(context.sourceHashes['reports/flight-live-variants/no-adhesion-power15-steering005.json'],sha(sourceBytes.capture));
assert.equal(muscleCapture.captureSha256,sha(sourceBytes.capture));
assert.equal(context.sourceHashes[sourceFiles.muscles],sha(sourceBytes.muscles));
const [manifest,groups,taste]=['/banc-data/console/sensory-inputs.json','/banc-data/console/groups.json','/body-model/banc-taste-peg-annotations.json'].map(url=>JSON.parse(assetBytes[url]));
for(const url of ['/banc-data/console/sensory-inputs.json','/banc-data/console/groups.json','/body-model/banc-taste-peg-annotations.json'])
 assert.equal(sourceHashes[url],context.sourceHashes[path.relative(root,localAsset(url))],'Frozen context data changed: '+url);
const qbytes=Buffer.from(capture.steps[99].postQpos,'base64'),q=Array.from({length:qbytes.length/8},(_,i)=>qbytes.readDoubleLE(i*8));
const pose={bodyTime:.2,x:q[0]*10,y:(q[2]-capture.initial.restHeight)*10,z:q[1]*10,
 heading:Math.atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]**2+q[6]**2)),contact:false,feedback:structuredClone(context.baselineFeedback)};
const habitat=createHabitat(structuredClone(capture.scene.fruit));
const tasteMapper=createBancTasteMapper([...capture.scene.io.sensory,...taste.annotations],groups.sweet);
const encoder=new SensoryEncoder(manifest,groups,habitat,{tasteMapper});
const encoded=encoder.update(pose,null,{odor:true,taste:true,vision:false,bodySense:true});
const [model,core]=await Promise.all([loadBancModel(),createWasmCore()]);
assert.equal(model.manifest.dt_ms,.5);assert.deepEqual(model.io.muscles,capture.scene.io.muscles);
assert.deepEqual(model.manifest,JSON.parse(assetBytes['/banc-data/manifest.json']));
assert.equal(model.manifest.files['params.bin'].sha256,context.sourceHashes['data/prepared/banc888/params.bin']);
const neuralFingerprint=sha(JSON.stringify({assets:Object.entries(sourceHashes).sort(([a],[b])=>a.localeCompare(b)),graphFiles:model.manifest.files}));
console.log('Calibrating the existing sensory current mapper');
const mapper=await createBancSensoryCurrentMapper(core,model,{indices:encoded.indices});
const currents=new Float32Array(model.manifest.neuron_count),rateById=new Map();
encoded.indices.forEach((id,k)=>{currents[id]=mapper.current(id,encoded.ratesHz[k]);rateById.set(id,encoded.ratesHz[k]);});
const wingMappings=model.io.muscles.flatMap((mapping,index)=>['asynchronous_wing','wing_steering_assumption'].includes(mapping.kind)?[{...mapping,mappingIndex:index}]:[]);
assert.equal(wingMappings.length,28);
const rotation=manifest.body_transducers.filter(sensor=>sensor.kind==='rotation');assert.equal(rotation.length,449);
const wingIds=[...new Set(wingMappings.flatMap(mapping=>mapping.indices))].sort((a,b)=>a-b);assert.equal(wingIds.length,48);
const readIds=Uint32Array.from([...new Set([...wingIds,...rotation.map(sensor=>sensor.index)])].sort((a,b)=>a-b));
const readOffset=new Map(Array.from(readIds,(id,k)=>[id,k]));
const inputGroups=selectedGroupKeys.map(key=>{
 const cells=rotation.filter(sensor=>sensor.organ+'/'+sensor.side===key).map(sensor=>sensor.index);
 const deltaHz=Math.min(maxPulseHz,...cells.map(id=>.5*(rateById.get(id)-mapper.limits(id).minimumHz)),...cells.map(id=>.5*(Math.min(100,mapper.limits(id).maximumHz)-rateById.get(id))));
 assert(deltaHz>0&&Number.isFinite(deltaHz),'No symmetric pulse above the current-map floor: '+key);
 const pulses=[-1,1].map(sign=>({sign,cells:cells.map(id=>{const requestedHz=rateById.get(id)+sign*deltaHz;
  return {index:id,baselineHz:rateById.get(id),requestedHz,baselineCurrent:currents[id],current:Math.fround(mapper.current(id,requestedHz))};})}));
 return {key,indices:cells,count:cells.length,deltaHz,pulses};
});
const recruitmentConfig=JSON.parse(sourceBytes.recruitment).recruitment;
assert.deepEqual(recruitmentConfig,{steering:{kind:'hill',halfActivationHz:80,exponent:1}});
const recruitment=createMotorExcitation(recruitmentConfig);
const nativeState=unpack(muscleCapture.profiles[199].state),nativeInput=unpack(muscleCapture.profiles[199].input);
assert.deepEqual(muscleCapture.mappings,capture.scene.io.muscles);
assert.equal(muscleCapture.profiles[199].frameIndex,99);
assert(Math.abs(muscleCapture.profiles[199].timeBefore+.001-.2)<1e-12);
assert.equal(muscleCapture.profiles[199].dt,.001);
const frozenMuscleInput=new Float32Array(28*5),initialMuscleState=new Float32Array(28*3);
wingMappings.forEach((mapping,k)=>{frozenMuscleInput.set(nativeInput.subarray(mapping.mappingIndex*5,mapping.mappingIndex*5+5),k*5);initialMuscleState.set(nativeState.subarray(mapping.mappingIndex*3,mapping.mappingIndex*3+3),k*3);});
const internal=Object.fromEntries(['hunger','insulin','akh'].map(key=>[key,capture.initial.internal[key]]));
const report={schemaVersion:2,kind:'frozen-context-afferent-identification',startedAt:new Date().toISOString(),neuralFingerprint,
 scope:`Artificial neural identification on frozen sensory input reconstructed from a recorded200ms airborne state. Fresh neural initialization and a common${prefixMs}ms conditioning prefix; not a restored live-flight brain, flight evaluation or receptor-axis calibration.`,
 context:{pose,staticSnapshot:true,finiteDifferenceJointSpeeds:'dt=0 snapshot convention from the source observability report',
  internal,internalSource:'Captured initial internal values held fixed; not claimed to be the internal state at200ms.',indices:Array.from(encoded.indices),ratesHz:pack(encoded.ratesHz),currentsPa:pack(currents)},
 protocol:{prefixMs,maxPulseHz,groupSelection,selectedGroupKeys,expectedArmCount,branchMs:100,baselineMs:20,pulseMs:20,recoveryMs:60,dtMs:.5,readoutMs:2,gaps:true,
  pulse:`Symmetric requested-rate change per afferent; delta=min(${maxPulseHz}Hz, half distance to calibrated minimum/100Hz maximum). Realized float32 currents are recorded; current changes need not be symmetric.`,
  snapshotBuffers:['states[0]','states[1]','history','kinetics','inputs (including diagnostic event ring)'],
  baselineEquality:'Literal byte equality of every readout/muscle trace and final states[0],states[1],history,kinetics,current-input bytes. Atomic event-ring ordering is diagnostic only and is excluded from final-state equality; it is copied when branching.'},
 sourceHashes,artifactHashes:Object.fromEntries(Object.entries(sourceFiles).map(([key,file])=>[file,sha(sourceBytes[key])])),
 scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),cli:process.argv.slice(2),neuralManifest:model.manifest,
 sourceCaptureConfigHash:original.configHash,sourceCaptureModelFingerprint:original.modelFingerprint,
 inputGroups,readIds:Array.from(readIds),wingIds,wingMappings,recruitment:recruitmentConfig,
 muscleReadoutScope:`The 28 independent native wing muscles use frozen captured length, velocity, maximum force and energy, plus the explicitly selected Hill80/n1 steering recruitment. Initial captured legacy muscle states receive the common${prefixMs}ms prefix. No body integration, wing controller gains or force-reference trim is applied.`,
 sensoryCurrentProfiles:mapper.profiles.map(({current,...profile})=>profile),baselineGate:{passed:false},interpretationAllowed:false,arms:[],error:null};
let provider,master,stop=false;
process.on('SIGINT',()=>{stop=true;});process.on('SIGTERM',()=>{stop=true;});
async function checkpoint(){await new Promise(resolve=>setTimeout(resolve,0));let marker=false;try{await fs.access(path.join(output,'STOP'));marker=true;}catch(error){if(error.code!=='ENOENT')throw error;}if(stop||marker){const error=new Error('Operator stopped identification');error.name='AbortError';throw error;}}
async function verifyPins(){
 for(const [url,digest]of Object.entries(sourceHashes))assert.equal(sha(await fs.readFile(localAsset(url))),digest,'Dependency changed during identification: '+url);
 for(const key of ['backend','backendLock'])assert.equal(sha(await fs.readFile(sourceFiles[key])),sha(sourceBytes[key]));
}
function newMuscles(state){const muscles=new WasmMuscles(core,28);core.HEAPF32.set(state,muscles.state/4);return muscles;}
function muscleTick(muscles,state){
 const input=frozenMuscleInput.slice(),rates=new Float32Array(28);
 wingMappings.forEach((mapping,k)=>{const rate=mapping.indices.reduce((sum,id)=>sum+state[readOffset.get(id)*8+4],0)/mapping.indices.length;
  rates[k]=rate;input[k*5]=recruitment.fromRate(mapping.kind,rate);});
 const one=muscles.step(input,.001),two=muscles.step(input,.001);
 return {rates,input,one,two};
}
async function sample(brain,muscles,relativeMs,drive){
 await checkpoint();await brain.step(4,drive,internal,true);
 const state=await brain.readState(readIds);assert(state.every(Number.isFinite));
 const muscle=muscleTick(muscles,state);
 return {relativeMs,neuralMs:brain.timeMs,neuralState:pack(state),globalSpikes:state.totalSpikes,
  wingGroupRatesHz:pack(muscle.rates),wingMuscleInput:pack(muscle.input),wingMuscleStateAt1ms:pack(muscle.one),wingMuscleStateAt2ms:pack(muscle.two)};
}
async function forwardState(brain){
 const entries=[['state0',brain.states[0]],['state1',brain.states[1]],['history',brain.history],['kinetics',brain.kinetics],['inputCurrents',brain.inputs,brain.n*4]],output={};
 for(const [name,buffer,length=buffer.size]of entries){
  const staging=brain.device.createBuffer({size:length,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const encoder=brain.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,staging,0,length);brain.device.queue.submit([encoder.finish()]);
   await staging.mapAsync(GPUMapMode.READ);output[name]=Buffer.from(new Uint8Array(staging.getMappedRange()).slice());staging.unmap();
  }finally{staging.destroy();}
 }
 return output;
}
async function fork(){
 const child=await WebGPUBrain.create(model,{shared:master}),encoder=master.device.createCommandEncoder();
 for(const [from,to]of [[master.states[0],child.states[0]],[master.states[1],child.states[1]],[master.history,child.history],[master.kinetics,child.kinetics],[master.inputs,child.inputs]]){assert.equal(from.size,to.size);encoder.copyBufferToBuffer(from,0,to,0,from.size);}
 master.device.queue.submit([encoder.finish()]);await master.device.queue.onSubmittedWorkDone();child.tick=master.tick;return child;
}
async function runArm(name,group=null,sign=0){
 const brain=await fork(),muscles=newMuscles(report.prefixMuscleState),trace=[],drive=currents.slice();
 const pulse=group?.pulses.find(row=>row.sign===sign);
 if(pulse)for(const cell of pulse.cells)drive[cell.index]=cell.current;
 const arm={name,group:group?.key??null,sign,trace,error:null,completed:false};let finalState;
 try{
  await verifyPins();
  assert.equal(brain.tick,master.tick,'Forked neural clock differs');
  const initialHashes=Object.fromEntries(Object.entries(await forwardState(brain)).map(([key,value])=>[key,sha(value)]));
  assert.deepEqual(initialHashes,report.prefixForwardStateHashes,'Forked forward state differs from the conditioned master');
  arm.initialStateHashes=initialHashes;
  for(let block=0;block<50;block++){const time=block*2,active=time>=20&&time<40;
   trace.push(await sample(brain,muscles,time+2,active?drive:currents));}
  finalState=await forwardState(brain);arm.finalStateHashes=Object.fromEntries(Object.entries(finalState).map(([key,value])=>[key,sha(value)]));
  arm.completed=true;
 }catch(error){arm.error=error.stack||error.message;throw error;}
 finally{muscles.dispose();brain.dispose();report.arms.push({name,group:arm.group,sign,completed:arm.completed,file:name+'.json',error:arm.error});await fs.writeFile(path.join(output,name+'.json'),JSON.stringify(arm)+'\n',{flag:'wx'});}
 return {arm,finalState};
}
try{
 if(cli.values['reference-result']){
  const file=path.resolve(cli.values['reference-result']),bytes=await fs.readFile(file),reference=JSON.parse(bytes);
  assert(reference.baselineGate.passed&&reference.interpretationAllowed,'Reference experiment did not pass its gates');
  assert.equal(neuralFingerprint,reference.neuralFingerprint,'Neural dependencies differ from the declared reference');
  assert.deepEqual(report.artifactHashes,reference.artifactHashes,'Frozen source artifacts differ from the declared reference');
  assert.deepEqual(report.context,reference.context,'Frozen sensory/internal context differs from the declared reference');
  assert.deepEqual(report.recruitment,reference.recruitment,'Muscle recruitment differs from the declared reference');
  report.referenceGate={passed:true,file:path.relative(root,file),sha256:sha(bytes),referenceScriptSha256:reference.scriptSha256};
 }
 await verifyPins();
 const {installNativeWebGPU}=await import(pathToFileURL(path.join(root,sourceFiles.backend)));provider=await installNativeWebGPU();
 report.backend=provider.provenance;assert.equal(report.backend.moduleSha256,sha(sourceBytes.backend));assert.equal(report.backend.packageLockSha256,sha(sourceBytes.backendLock));
 master=await WebGPUBrain.create(model,{gpu:provider.gpu});
 const muscles=newMuscles(initialMuscleState);report.prefixTrace=[];
 try{for(let block=0;block<prefixMs/2;block++)report.prefixTrace.push(await sample(master,muscles,(block+1)*2,currents));
  report.prefixMuscleState=Float32Array.from(core.HEAPF32.subarray(muscles.state/4,muscles.state/4+28*3));
 }finally{muscles.dispose();}
 const frozenHashes=Object.fromEntries(Object.entries(await forwardState(master)).map(([key,value])=>[key,sha(value)]));report.prefixForwardStateHashes=frozenHashes;
 console.log(JSON.stringify({kind:'conditioned',neuralFingerprint,backend:'dawn-metal',prefixMs,maxPulseHz,expectedArmCount,groups:inputGroups.map(group=>({key:group.key,count:group.count,deltaHz:group.deltaHz}))}));
 const baselineA=await runArm('baseline-a'),baselineB=await runArm('baseline-b');
 assert.equal(baselineA.arm.trace.length,baselineB.arm.trace.length);
 for(let i=0;i<baselineA.arm.trace.length;i++)assert.deepEqual(baselineA.arm.trace[i],baselineB.arm.trace[i],'Duplicate baseline trace differs at block '+i);
 for(const [key,value]of Object.entries(baselineA.finalState))assert(value.equals(baselineB.finalState[key]),'Duplicate baseline full forward state differs: '+key);
 report.baselineGate={passed:true,checkedTraceBlocks:50,checkedForwardStateBytes:Object.values(baselineA.finalState).reduce((sum,value)=>sum+value.length,0),finalStateHashes:baselineA.arm.finalStateHashes};
 baselineA.finalState=null;baselineB.finalState=null;
 console.log(JSON.stringify({kind:'baseline-byte-gate',...report.baselineGate}));
 const responses=[];
 for(const group of inputGroups){
  const minus=await runArm(group.key.replace('/','-')+'-minus',group,-1),plus=await runArm(group.key.replace('/','-')+'-plus',group,1);
  const response={group:group.key,deltaHz:group.deltaHz,meanCurrentSpanPerCell:group.pulses[1].cells.reduce((sum,cell,k)=>sum+cell.current-group.pulses[0].cells[k].current,0)/group.count,
   trace:[],meanCentralWingRateGain:Array(28).fill(0),meanCentralWingForceGain:Array(28).fill(0)};
  for(let t=0;t<50;t++){
   const pr=unpack(plus.arm.trace[t].wingGroupRatesHz),mr=unpack(minus.arm.trace[t].wingGroupRatesHz),br=unpack(baselineA.arm.trace[t].wingGroupRatesHz);
   const pf=unpack(plus.arm.trace[t].wingMuscleStateAt2ms),mf=unpack(minus.arm.trace[t].wingMuscleStateAt2ms),bf=unpack(baselineA.arm.trace[t].wingMuscleStateAt2ms);
   const entry={relativeMs:(t+1)*2,plusMinusRateDifference:Array.from(pr,(v,k)=>v-mr[k]),plusMinusForceDifference:Array.from({length:28},(_,k)=>pf[k*3+2]-mf[k*3+2]),
    plusBaselineRateDifference:Array.from(pr,(v,k)=>v-br[k]),minusBaselineRateDifference:Array.from(mr,(v,k)=>v-br[k]),
    plusBaselineForceDifference:Array.from({length:28},(_,k)=>pf[k*3+2]-bf[k*3+2]),minusBaselineForceDifference:Array.from({length:28},(_,k)=>mf[k*3+2]-bf[k*3+2])};
   response.trace.push(entry);
   if(t>=10)for(let k=0;k<28;k++){response.meanCentralWingRateGain[k]+=entry.plusMinusRateDifference[k]/(2*group.deltaHz*40);response.meanCentralWingForceGain[k]+=entry.plusMinusForceDifference[k]/(2*group.deltaHz*40);}
  }
  responses.push(response);await fs.writeFile(path.join(output,'responses.json'),JSON.stringify({scope:'Signed response to anatomical population stimulation; no rotation-axis meaning is assigned.',baselineGate:report.baselineGate,responses})+'\n');
  console.log(JSON.stringify({kind:'group-complete',group:group.key,rateGainNorm:Math.hypot(...response.meanCentralWingRateGain),forceGainNorm:Math.hypot(...response.meanCentralWingForceGain)}));
 }
 const afterHashes=Object.fromEntries(Object.entries(await forwardState(master)).map(([key,value])=>[key,sha(value)]));assert.deepEqual(afterHashes,frozenHashes,'Conditioned master changed during branch experiments');
 await verifyPins();assert.equal(report.arms.length,expectedArmCount);report.interpretationAllowed=true;
}catch(error){report.error=error.stack||error.message;process.exitCode=1;}
finally{master?.dispose();provider?.uninstall();if(report.prefixMuscleState)report.prefixMuscleState=pack(report.prefixMuscleState);report.finishedAt=new Date().toISOString();await fs.writeFile(path.join(output,'result.json'),JSON.stringify(report)+'\n');}
console.log(JSON.stringify({kind:'identification-finished',output,arms:report.arms.length,baselineGate:report.baselineGate,interpretationAllowed:report.interpretationAllowed,error:report.error}));
