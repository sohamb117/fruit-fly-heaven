// Read-only paired checkpoint evaluation. One actual BANC/native-body fly at a time;
// no coordinator leases, result uploads, optimizer updates, or early time cap.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {registerHooks} from 'node:module';
import {createHash,randomUUID} from 'node:crypto';
import {createFlightHistoryCapture} from './flight-history-capture.mjs';
import {disableHabitatBoundaryContacts} from './flight-boundary-intervention.mjs';
import {installHaltereNeuralObserver} from './haltere-neural-observer.mjs';
import {prescribeLeftHaltereSensorMotion} from './prescribe-left-haltere-sensor-motion.mjs';
import {createMotorReplayCapture,packCaptureArray} from '../web/test/training-motor-capture.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [endpoint,planArg,outputArg]=process.argv.slice(2);
if(!endpoint||!planArg||!outputArg)throw new Error('Usage: node scripts/evaluate-training-native.mjs http://127.0.0.1:7849/ plan.json reports/output');
const url=new URL(endpoint),output=path.resolve(outputArg);
if(url.protocol!=='http:'||!['localhost','127.0.0.1'].includes(url.hostname)||url.username||url.password||url.search||url.hash||url.pathname!=='/')throw new Error('Use a loopback coordinator origin');
if(!output.startsWith(path.join(root,'reports')+path.sep))throw new Error('Evaluation output must be inside reports');
await fs.mkdir(output,{recursive:true});
if(!(await fs.realpath(output)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep))throw new Error('Evaluation output escaped reports');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex'),runId=randomUUID();
const planBytes=await fs.readFile(path.resolve(planArg)),plan=JSON.parse(planBytes);
if(plan.schemaVersion!==1||!Array.isArray(plan.jobs)||!plan.jobs.length||plan.jobs.length>1000)throw new Error('Invalid evaluation plan');
const backendLabel=plan.backend||'wasm';
if(!['wasm','dawn-metal'].includes(backendLabel))throw new Error('Unsupported diagnostic neural backend');
const expectedBackend=backendLabel==='dawn-metal'?'webgpu':'wasm';
let nativeWebGPU=null;
if(backendLabel==='dawn-metal'){
 const backendPath=path.join(root,'reports/native-webgpu-tooling/backend.mjs');
 if(!plan.nativeWebGPU||sha(await fs.readFile(backendPath))!==plan.nativeWebGPU.moduleSha256)throw new Error('Native WebGPU loader is not pinned by the plan');
 const {installNativeWebGPU}=await import(pathToFileURL(backendPath));
 nativeWebGPU=await installNativeWebGPU();
 if(nativeWebGPU.provenance.packageLockSha256!==plan.nativeWebGPU.packageLockSha256)throw new Error('Native WebGPU installation differs from the plan');
}
const networkFetch=globalThis.fetch;
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
function localAsset(asset){const entry=prefixes.find(([prefix])=>asset.startsWith(prefix));const file=path.resolve(root,entry?entry[1]+asset.slice(entry[0].length):'web/'+asset.slice(1));if(!file.startsWith(root+path.sep))throw new Error('Asset escaped checkout');return file;}
registerHooks({resolve(specifier,context,next){if(prefixes.some(([prefix])=>specifier.startsWith(prefix)))return {url:pathToFileURL(localAsset(specifier)).href,shortCircuit:true};return next(specifier,context);}});
globalThis.location={href:url.href};
globalThis.fetch=(input,options={})=>{
 let requested=new URL(typeof input==='string'?input:input.url||String(input),url);
 // Node resolves this existing module-relative shader URL as file://; fetch
 // the exact corresponding pinned HTTP asset, not arbitrary local files.
 for(const shader of ['neural.wgsl','neural-dlm.wgsl'])if(requested.href===pathToFileURL(path.join(root,'packages/banc-runtime/src',shader)).href)requested=new URL('/banc-engine/src/'+shader,url);
 const method=String(options.method||input?.method||'GET').toUpperCase();
 if(requested.origin!==url.origin||!['GET','HEAD'].includes(method))throw new Error('Evaluator permits only same-origin read-only requests');
 return networkFetch(requested,{...options,method,redirect:'error',signal:options.signal||AbortSignal.timeout(30000)});
};
const response=await fetch('/training/config.json');if(!response.ok)throw new Error('Cannot load coordinator configuration');
const configText=await response.text(),config=JSON.parse(configText),configHash=sha(configText);
if(plan.configHash!==configHash||plan.modelFingerprint!==config.modelFingerprint)throw new Error('Plan does not match coordinator configuration');
const names=new Set();
for(const job of plan.jobs){
 if(typeof job.name!=='string'||!/^[-a-zA-Z0-9._]+$/.test(job.name)||job.name.length>150||names.has(job.name))throw new Error('Invalid or duplicate evaluation job name');
 if(job.captureMotorEvents!==undefined&&typeof job.captureMotorEvents!=='boolean')throw new Error('Motor-event capture must be an explicit boolean');
 if(job.capturePhysicsDigest!==undefined&&typeof job.capturePhysicsDigest!=='boolean')throw new Error('Physics digest capture must be an explicit boolean');
 if(job.captureTegulaFeedback!==undefined&&typeof job.captureTegulaFeedback!=='boolean')throw new Error('Tegula feedback capture must be an explicit boolean');
 if(job.captureHaltereFeedback!==undefined&&typeof job.captureHaltereFeedback!=='boolean')throw new Error('Haltere feedback capture must be an explicit boolean');
 if(job.captureMotorReplay!==undefined&&(typeof job.captureMotorReplay!=='boolean'||(job.captureMotorReplay&&job.captureMotorEvents!==true)))throw new Error('Motor replay capture requires explicit motor-event capture');
 if(job.captureHaltereNeuralState!==undefined&&(typeof job.captureHaltereNeuralState!=='boolean'||(job.captureHaltereNeuralState&&(backendLabel!=='dawn-metal'||job.captureMotorEvents!==true))))throw new Error('Haltere neural observation requires explicit native Metal and motor-event capture');
 if(job.prescribedLeftHaltereMotion!==undefined){
  const control=job.prescribedLeftHaltereMotion;
  if(!control||typeof control!=='object'||Array.isArray(control)||Object.keys(control).length!==2||
   !Number.isFinite(control.onsetSeconds)||control.onsetSeconds<0||control.onsetSeconds>=job.durationSeconds||
   !Number.isFinite(control.leftAmplitudeRadians)||control.leftAmplitudeRadians<0||control.leftAmplitudeRadians>Math.PI/4||
   config.haltereFeedback===undefined||job.captureHaltereFeedback!==true||job.captureMotorEvents!==true)
   throw new Error('Prescribed haltere motion requires an explicit bounded diagnostic, mechanical mapper and input/event capture');
 }
 if(job.captureFlightHistory!==undefined&&typeof job.captureFlightHistory!=='boolean')throw new Error('Flight history capture must be an explicit boolean');
 if(job.disableBoundaryContacts!==undefined&&typeof job.disableBoundaryContacts!=='boolean')throw new Error('Boundary intervention must be an explicit boolean');
 if(job.physicsPrefixSeconds!==undefined&&(!job.capturePhysicsDigest||!Number.isFinite(job.physicsPrefixSeconds)||job.physicsPrefixSeconds<=0||job.physicsPrefixSeconds>job.durationSeconds||Math.abs(job.physicsPrefixSeconds*1000/config.bodyBlockMs-Math.round(job.physicsPrefixSeconds*1000/config.bodyBlockMs))>1e-9))throw new Error('Physics prefix must end at an observed body block');
 names.add(job.name);
 const stage=config.stages.find(value=>value.id===job.stage);
 if(!Number.isInteger(job.seed)||job.seed<0||job.seed>0xffffffff||!stage||job.durationSeconds!==stage.durationSeconds)throw new Error('Invalid evaluation seed, stage or full horizon');
 if(!Array.isArray(job.parameters)||job.parameters.length!==config.parameters.length||job.parameters.some((value,index)=>!Number.isFinite(value)||value<config.parameters[index].min||value>config.parameters[index].max))throw new Error('Evaluation parameters are out of bounds');
}
async function verifyCode(){
 const hashes={};
 for(const [relative,digest]of Object.entries(plan.diagnosticSourceHashes||{})){
  const file=path.resolve(root,relative);
  if(!file.startsWith(root+path.sep)||!/^[a-f0-9]{64}$/.test(digest)||sha(await fs.readFile(file))!==digest)throw new Error('Diagnostic source differs from plan: '+relative);
 }
 if(nativeWebGPU){
  if(sha(await fs.readFile(path.join(root,'reports/native-webgpu-tooling/backend.mjs')))!==plan.nativeWebGPU.moduleSha256||sha(await fs.readFile(path.join(root,'reports/native-webgpu-tooling/package-lock.json')))!==plan.nativeWebGPU.packageLockSha256)throw new Error('Native WebGPU tooling changed during evaluation');
 }
 for(const [asset,digest]of Object.entries(config.assets))if(!asset.startsWith('/body-model/')){
  hashes[asset]=sha(await fs.readFile(localAsset(asset)));
  if(hashes[asset]!==digest)throw new Error('Local/served asset mismatch: '+asset);
 }
 return hashes;
}
const sourceHashes=await verifyCode();
const replaySourceXml=plan.jobs.some(job=>job.captureMotorReplay)?await (await fetch('/body-model/flybody-mujoco.xml')).text():null;
if(replaySourceXml!==null&&sha(replaySourceXml)!==config.assets['/body-model/flybody-mujoco.xml'])throw new Error('Replay model XML differs from configuration');
let haltereNeuralSelection;
if(plan.jobs.some(job=>job.captureHaltereNeuralState)){
 const io=JSON.parse(await fs.readFile(localAsset('/banc-data/io.json'))),sensory=JSON.parse(await fs.readFile(localAsset('/banc-data/console/sensory-inputs.json')));
 const wingIndices=Uint32Array.from([...new Set(io.muscles.filter(m=>m.kind==='asynchronous_wing'||m.kind==='wing_steering_assumption').flatMap(m=>m.indices))].sort((a,b)=>a-b));
 const cells=sensory.body_transducers.filter(cell=>cell.organ==='haltere').sort((a,b)=>a.index-b.index);
 const motor=io.motor_neurons.filter(m=>[97021,118683].includes(m.index)).sort((a,b)=>a.index-b.index);
 if(motor.length!==2||motor[0].root_id!=='720575941546998972'||motor[0].side!=='left'||motor[1].root_id!=='720575941572052381'||motor[1].side!=='right'||motor.some(m=>m.peripheral_target_type!=='haltere_dorsoventral_muscle'))throw new Error('Haltere motor identities differ from diagnostic contract');
 const {WebGPUBrain}=await import('../packages/banc-runtime/src/webgpu.js');
 haltereNeuralSelection={BrainClass:WebGPUBrain,neuronCount:sensory.neuron_count,wingIndices,haltereIndices:Uint32Array.from(cells,c=>c.index),haltereMotorIndices:[97021,118683],cells,motor};
}
const run={schemaVersion:1,kind:'native-checkpoint-evaluation',runId,date:new Date().toISOString(),endpoint:url.href,
 scope:plan.scope||'Held-out native WASM checkpoint comparison. No optimizer or training upload.',
 planFile:path.resolve(planArg),planSha256:sha(planBytes),plan,configText,configHash,modelFingerprint:config.modelFingerprint,
 sourceHashes,scriptSha256:sha(await fs.readFile(fileURLToPath(import.meta.url))),nodeVersion:process.version,backendLabel,nativeWebGPU:nativeWebGPU?.provenance??null,jobs:[],stopped:false,error:null};
const runFile=path.join(output,`run-${runId}.json`);
await fs.writeFile(runFile,JSON.stringify(run)+'\n',{flag:'wx'});
let stop=false,environment;
process.on('SIGINT',()=>{stop=true;});process.on('SIGTERM',()=>{stop=true;});
async function stopping(){if(stop)return true;try{await fs.access(path.join(output,'STOP'));return true;}catch(error){if(error.code!=='ENOENT')throw error;return false;}}
try{
 const {createTrainingEnvironment}=await import('../web/training/environment.js');
 environment=await createTrainingEnvironment(config,{onProgress:value=>console.log(value.message)});
 run.ready=await environment.ready();
 if(run.ready.backend!==expectedBackend||run.ready.configHash!==configHash)throw new Error('Evaluator requires its explicitly planned native neural backend; fallback is not accepted');
 console.log(JSON.stringify({kind:'evaluation-ready',configHash,modelFingerprint:config.modelFingerprint,jobs:plan.jobs.length,output}));
 for(const [index,job]of plan.jobs.entries()){
  if(await stopping()){run.stopped=true;break;}
  const frames=[],record={schemaVersion:1,kind:'native-held-out-evaluation',runId,index,assignment:job,
   configHash,modelFingerprint:config.modelFingerprint,planSha256:run.planSha256,sourceHashes,backendLabel,nativeWebGPU:run.nativeWebGPU,
   scope:run.scope,evaluation:null,frames,error:null,complete:false};
  if(job.captureMotorEvents)record.motorEvents=[];
  if(job.captureTegulaFeedback)record.tegulaFeedback=[];
  if(job.captureHaltereFeedback)record.haltereFeedback=[];
  const motorReplay=job.captureMotorReplay?createMotorReplayCapture({sourceXml:replaySourceXml,seconds:.5,bodyBlockMs:config.bodyBlockMs,sourceAssets:plan.diagnosticSourceHashes}):null;
  if(motorReplay)record.motorForceSamples=[];
  if(job.prescribedLeftHaltereMotion)record.prescribedLeftHaltereMotion=[];
  if(job.captureHaltereNeuralState)record.haltereNeuralState={cells:haltereNeuralSelection.cells,motor:haltereNeuralSelection.motor,
   fields:['voltageMv','adaptationPa','refractoryMs','cumulativeSpikeCount','rateHz','releasePerMs','totalConductanceNs','externalAndIntrinsicCurrentPa','lastSpikeMs'],samples:[]};
  const flightHistory=job.captureFlightHistory?createFlightHistoryCapture():null;
  const physicsHash=job.capturePhysicsDigest?createHash('sha256'):null;
  let physicsRows=0,physicsValuesPerRow=null;
  const capturePhysics=physicsHash?({body})=>{
   const values=Float64Array.from([body.time,...body.data.qpos,...body.data.qvel,...body.data.act,...body.data.ctrl,
    ...body.muscleState,body.wings.phase,...body.wings.deployment,...body.wings.power,...body.wings.target]);
   if(physicsValuesPerRow!==null&&physicsValuesPerRow!==values.length)throw new Error('Physics digest layout changed');
   physicsValuesPerRow=values.length;physicsHash.update(new Uint8Array(values.buffer));physicsRows++;
   if(job.physicsPrefixSeconds!==undefined&&Math.abs(body.time-job.physicsPrefixSeconds)<1e-9)
    record.physicsPrefixDigest={throughSeconds:job.physicsPrefixSeconds,sha256:physicsHash.copy().digest('hex'),rows:physicsRows,valuesPerRow:physicsValuesPerRow};
  }:undefined;
  const captureBody=physicsHash||job.captureTegulaFeedback||job.captureHaltereFeedback||flightHistory||job.disableBoundaryContacts||job.prescribedLeftHaltereMotion||motorReplay?context=>{
   if(context.initialObservation&&job.disableBoundaryContacts)record.intervention=disableHabitatBoundaryContacts(context.body);
   capturePhysics?.(context);
   if(flightHistory){if(context.initialObservation)flightHistory.start(context);else flightHistory.sample(context);}
   if(motorReplay){
    if(context.initialObservation){
     motorReplay.onInitialState(context);
     const event=context.body._wingMotorEvents;
     if(!event||event.elapsedMs!==0||event.observedMs!==0||!event.initialized)throw new Error('Replay requires an initialized zero-time wing event path');
     record.initialWingMotorState={adapterSnapshot:event.adapter.snapshot(),
      nativeWasmState:{encoding:'float32-little-endian-base64',data:packCaptureArray(event.muscles.core.HEAPF32.subarray(event.muscles.state/4,event.muscles.state/4+84),32)},
      nativeInput:Array.from(event.input),nativeState:Array.from(event.nativeState),elapsedMs:event.elapsedMs,observedMs:event.observedMs,initialized:event.initialized,lastInterval:null};
    }else motorReplay.onPhysicsStep(context);
    if(context.body.time<=.5+1e-9)record.motorForceSamples.push({bodyTimeSeconds:context.body.time,
     muscleState:Array.from(context.body.muscleState),wingDrive:[context.body.wingDriveLeft??0,context.body.wingDriveRight??0],
     wingPower:Array.from(context.body.wings.power),wingPhase:context.body.wings.phase,internal:{...context.body.internal}});
   }
   if(job.captureTegulaFeedback){
    const {body,fly}=context,rates=fly.sensory?.body?.rates;
    record.tegulaFeedback.push({bodyTimeSeconds:body.time,inputBodyTimeSeconds:fly.sensory?.bodyTime??null,
     inputRatesHz:rates&&Object.hasOwn(rates,'tegula_left')?{left:rates.tegula_left,right:rates.tegula_right}:null,
     loadAfterPhysics:body.readWingLoadFeedback?.()??null});
   }
   if(job.captureHaltereFeedback){
    const {body,fly}=context;
    record.haltereFeedback.push({bodyTimeSeconds:body.time,
     currentInput:structuredClone(fly.sensory?.haltereCurrent??null),
     powerAfterPhysics:Array.from(body.halterePower),omegaAfterPhysicsRadS:Array.from(body.data.qvel.slice(3,6)),
     wingPhaseAfterPhysicsRadians:body.wings.phase,wingFrequencyHz:body.wings.frequencyHz});
   }
   // Explicit external sensor-motion control, after recording actual native
   // state. This changes the next sensory sample only, never native actuation.
   if(job.prescribedLeftHaltereMotion)record.prescribedLeftHaltereMotion.push(prescribeLeftHaltereSensorMotion(context,job.prescribedLeftHaltereMotion));
  }:undefined;
  const file=path.join(output,`${String(index).padStart(3,'0')}-${job.name}-${runId}.json`);
  let lastProgress=0,neuralObserver;
  try{
   await verifyCode();
   if(job.captureHaltereNeuralState)neuralObserver=installHaltereNeuralObserver({...haltereNeuralSelection,onSample:sample=>{
    record.haltereNeuralState.samples.push({timeMs:sample.timeMs,state:Array.from(sample.state)});
   }});
   record.evaluation=await environment.evaluate(job,{previewHz:.2,dutyCycle:1,
    checkpoint:async()=>{await new Promise(resolve=>setTimeout(resolve,0));if(await stopping()){const error=new Error('Operator stopped evaluation');error.name='AbortError';throw error;}},
    onFrame:frame=>{if(frames.length<1000)frames.push(structuredClone(frame));},
    onInitialState:captureBody,onPhysicsStep:captureBody,
    // Recording only. The pinned environment config independently determines
    // whether the body consumes events; this hook never enables that path.
    onMotorEvents:job.captureMotorEvents?packet=>{record.motorEvents.push({
     ...packet,indices:Array.from(packet.indices),ratesHz:Array.from(packet.ratesHz),counts:Array.from(packet.counts),
     events:packet.events.map(event=>({...event})),
    });}:undefined,
    onProgress:value=>{if(performance.now()-lastProgress>15000){lastProgress=performance.now();console.log(JSON.stringify({kind:'evaluation-progress',name:job.name,simSeconds:value.simSeconds,score:value.return}));}}
   });
   await verifyCode();
   const value=record.evaluation;
   if(value.configHash!==configHash||value.modelFingerprint!==config.modelFingerprint||JSON.stringify(value.parameters)!==JSON.stringify(job.parameters)||value.seed!==job.seed||value.stage!==job.stage||value.backend!==expectedBackend)throw new Error('Evaluation differs from planned assignment');
   if(value.cancelled){run.stopped=true;}
   else {
    if(value.reason==='simulation_error')throw new Error(value.metrics?.error||'Native evaluation error');
    const full=Math.abs(value.simSeconds-job.durationSeconds)<1e-7;
    const physicalFailure=!value.success&&value.terminated&&['outside_habitat','excessive_rotation','overturned'].includes(value.reason);
    if(!Number.isFinite(value.return)||!Number.isFinite(value.simSeconds)||value.simSeconds<=0||!Number.isInteger(value.steps)||value.steps<1||
      (!full&&!physicalFailure)||value.simSeconds>job.durationSeconds+1e-7||Math.abs(value.simSeconds-value.steps*config.bodyBlockMs/1000)>1e-7)throw new Error('Evaluation did not complete its full horizon or native physical failure');
    record.complete=true;
   }
  }catch(error){record.error=error.stack||error.message;run.error=record.error;}
  finally{
   neuralObserver?.restore();
   if(flightHistory)record.flightHistory=flightHistory.finish();
   if(motorReplay&&record.evaluation)record.motorReplay=motorReplay.finish(record.evaluation);
   if(physicsHash)record.physicsDigest={sha256:physicsHash.digest('hex'),rows:physicsRows,valuesPerRow:physicsValuesPerRow,
    encoding:'Native-endian float64 on the recorded Node platform, at initialization and every completed 2 ms body block.',
    fields:['body.time','qpos','qvel','act','ctrl','muscleState','wingPhase','wingDeployment','wingPower','wingTarget']};
   await fs.writeFile(file,JSON.stringify(record)+'\n',{flag:'wx'});
   run.jobs.push({name:job.name,file,complete:record.complete,error:record.error,cancelled:record.evaluation?.cancelled??false});
   await fs.writeFile(runFile,JSON.stringify(run)+'\n');
  }
  console.log(JSON.stringify({kind:'evaluated',name:job.name,complete:record.complete,reason:record.evaluation?.reason,simSeconds:record.evaluation?.simSeconds,score:record.evaluation?.return,takeoff:record.evaluation?.metrics?.hasTakenOff,bestFlightSeconds:record.evaluation?.metrics?.bestFlightSeconds,error:record.error,file}));
  if(record.error||run.stopped)break;
 }
}catch(error){run.error=error.stack||error.message;}
finally{try{environment?.dispose();}finally{nativeWebGPU?.uninstall();}run.finishedAt=new Date().toISOString();await fs.writeFile(runFile,JSON.stringify(run)+'\n');}
console.log(JSON.stringify({kind:'evaluation-finished',runFile,completed:run.jobs.filter(job=>job.complete).length,planned:plan.jobs.length,stopped:run.stopped,error:run.error}));
if(run.error||run.stopped||run.jobs.length!==plan.jobs.length)process.exitCode=1;
