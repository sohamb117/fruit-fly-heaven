// Native Node contributor (WASM by default, optionally pinned Dawn/Metal) to the same durable coordinator protocol as the
// browser. HTTP loopback or an explicitly pinned HTTPS coordinator; no local optimizer.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {registerHooks} from 'node:module';
import {createHash,randomUUID} from 'node:crypto';
import {validateTrainingConfigSchema} from '../web/training/config-schema.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const digest=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
export function validateContributorBackendPlan(value){
 if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Backend plan must be an object');
 const allowed=['schemaVersion','backend','configHash','modelFingerprint','nativeWebGPU','scope','stopAfterGeneration'];
 if(Object.keys(value).some(key=>!allowed.includes(key))||value.schemaVersion!==1||!['wasm','dawn-metal'].includes(value.backend)||!digest(value.configHash)||!digest(value.modelFingerprint)||(value.scope!==undefined&&(typeof value.scope!=='string'||value.scope.length>4096)))throw new Error('Invalid contributor backend plan schema');
 if(value.stopAfterGeneration!==undefined&&(!Number.isSafeInteger(value.stopAfterGeneration)||value.stopAfterGeneration<1))throw new Error('Invalid contributor generation limit');
 if(value.backend==='dawn-metal'){
  const pins=value.nativeWebGPU;
  if(!pins||typeof pins!=='object'||Array.isArray(pins)||Object.keys(pins).length!==2||!digest(pins.moduleSha256)||!digest(pins.packageLockSha256))throw new Error('Dawn plan requires exact loader and package-lock hashes');
 }else if(value.nativeWebGPU!==undefined)throw new Error('WASM plan must not claim native WebGPU provenance');
 const plan=structuredClone(value);if(plan.nativeWebGPU)Object.freeze(plan.nativeWebGPU);return Object.freeze(plan);
}

export function validateContributorOrigin(endpoint,backendPlan=null){
 const url=new URL(endpoint);
 if(url.username||url.password||url.search||url.hash||url.pathname!=='/')throw new Error('Use a coordinator origin without credentials, path, query or fragment');
 if(url.protocol==='https:'){
  if(!backendPlan)throw new Error('HTTPS contribution requires an explicit pinned backend plan');
  validateContributorBackendPlan(backendPlan);
 }else if(url.protocol!=='http:'||!['localhost','127.0.0.1'].includes(url.hostname))throw new Error('Use HTTPS or an HTTP loopback coordinator origin');
 return url;
}

// Every asset/API request stays on the selected scheme, host and port. The only
// file-URL exceptions are the two known local shader imports, mapped to that
// same origin. Callers cannot enable redirects or embed URL credentials.
export function createContributorFetch(endpoint,networkFetch){
 const url=new URL(endpoint);
 return (input,options={})=>{
  let requested=new URL(typeof input==='string'?input:input.url||String(input),url);
  for(const shader of ['neural.wgsl','neural-dlm.wgsl'])if(requested.href===pathToFileURL(path.join(root,'packages/banc-runtime/src',shader)).href)requested=new URL('/banc-engine/src/'+shader,url);
  if(requested.origin!==url.origin||requested.username||requested.password)throw new Error('External request or URL credentials denied');
  return networkFetch(requested,{...options,redirect:'error',signal:options.signal||AbortSignal.timeout(30000)});
 };
}

// Observer-only allowlist matching the coordinator's bounded preview schema.
// Invalid telemetry is omitted; it must not prevent a plain lease renewal.
export function makeContributorPreviewFrame(frame){
 try{
  const number=(v,low=-1e6,high=1e6)=>{if(typeof v!=='number'||!Number.isFinite(v)||v<low||v>high)throw new Error('number');return v;};
  const vector=(v,n,low=-1e6)=>{if(!Array.isArray(v)||v.length!==n)throw new Error('vector');return v.map(x=>number(x,low));};
  const items=(v,n,map)=>{if(!Array.isArray(v)||v.length>n)throw new Error('list');return v.map(map);};
  const shape=v=>({position:vector(v.position,3),rotation:vector(v.rotation,9),size:vector(v.size,3,0)});
  const out={time:number(frame.time,0,1e12),simSeconds:number(frame.simSeconds,0,1e12),
   position:vector(frame.position,3),quaternion:vector(frame.quaternion,4)};
  if(Math.abs(Math.hypot(...out.quaternion)-1)>.01)throw new Error('quaternion');
  for(const key of ['stage','phase','motion'])if(frame[key]!==undefined){
   if(typeof frame[key]!=='string'||!/^[a-z_]{1,32}$/.test(frame[key]))throw new Error('label');out[key]=frame[key];
  }
  for(const key of ['neuralMs','neuralSpikes','episodeTimeSeconds','nativeTimeSeconds','releaseNativeTime','releaseNeuralTimeMs','warmupSeconds','scoredSteps'])
   if(frame[key]!==undefined&&frame[key]!==null)out[key]=number(frame[key],0,1e12);
  for(const key of ['warmup','vision'])if(frame[key]!==undefined){if(typeof frame[key]!=='boolean')throw new Error('flag');out[key]=frame[key];}
  if(frame.feet!==undefined)out.feet=items(frame.feet,6,v=>vector(v,3));
  if(frame.legs!==undefined)out.legs=items(frame.legs,6,v=>items(v,6,p=>vector(p,3)));
  if(frame.wings!==undefined)out.wings=items(frame.wings,2,v=>{
   if(!['left','right'].includes(v.side))throw new Error('wing');return {...shape(v),side:v.side,anchor:vector(v.anchor,3)};
  });
  if(frame.mouth!==undefined)out.mouth={anchors:items(frame.mouth.anchors??[],4,v=>vector(v,3)),ellipsoids:items(frame.mouth.ellipsoids??[],4,shape)};
  if(frame.contacts!==undefined){
   out.contacts={};for(const key of ['environment','food'])if(frame.contacts[key]!==undefined)out.contacts[key]=number(frame.contacts[key],0);
   for(const [key,count]of [['legs',6],['mouth',2],['wings',2]])if(frame.contacts[key]!==undefined)out.contacts[key]=items(frame.contacts[key],count,v=>number(v,0));
  }
  if(frame.bowl!==undefined){
   out.bowl={radiusCm:number(frame.bowl.radiusCm,0)};
   if(frame.bowl.ceilingCm!==undefined)out.bowl.ceilingCm=number(frame.bowl.ceilingCm,0);
   if(frame.bowl.floor!==undefined){const floor=frame.bowl.floor;out.bowl.floor={baseCm:number(floor.baseCm),radialCoefficientPerCm:number(floor.radialCoefficientPerCm)};
    if(floor.capRadiusCm!==undefined)out.bowl.floor.capRadiusCm=number(floor.capRadiusCm,0);}
  }
  return Buffer.byteLength(JSON.stringify(out),'utf8')<=32768?out:null;
 }catch{return null;}
}

// A fixed lease identity, one in-flight renewal, and an owned latest-frame
// snapshot keep observer requests from leaking into the following assignment.
export function createContributorHeartbeat({identity,send,getPreview,intervalMs=5000,
 setTimer=setInterval,clearTimer=clearInterval}){
 const lease={...identity};let inFlight=null,failure=null,stopped=false;
 function tick(){
  if(stopped||failure)return Promise.resolve();if(inFlight)return inFlight;
  const frame=getPreview(),payload={...lease,...(frame?{previewFrame:structuredClone(frame)}:{})};
  inFlight=Promise.resolve().then(()=>send(payload)).catch(error=>{failure=error;}).finally(()=>{inFlight=null;});
  return inFlight;
 }
 const timer=setTimer(()=>{void tick();},intervalMs);
 const check=()=>{if(failure)throw failure;};
 return {check,async flush(){if(inFlight)await inFlight;check();await tick();check();},
  async stop(){stopped=true;clearTimer(timer);if(inFlight)await inFlight;}};
}

// Keep installation, all setup failures, and environment cleanup inside the
// same native-provider lifetime. Injected operations are for small unit tests.
export async function withContributorBackend(plan,run,{read=fs.readFile,load=async file=>import(pathToFileURL(file))}={}){
 if(plan!==null&&plan!==undefined)plan=validateContributorBackendPlan(plan);
 let native=null;
 try{
  if(plan?.backend==='dawn-metal'){
   const directory=path.join(root,'reports/native-webgpu-tooling'),module=path.join(directory,'backend.mjs');
   if(sha(await read(module))!==plan.nativeWebGPU.moduleSha256||sha(await read(path.join(directory,'package-lock.json')))!==plan.nativeWebGPU.packageLockSha256)throw new Error('Native WebGPU tooling differs from backend plan');
   const {installNativeWebGPU}=await load(module);native=await installNativeWebGPU();
   const p=native.provenance;
   if(p.moduleSha256!==plan.nativeWebGPU.moduleSha256||p.packageLockSha256!==plan.nativeWebGPU.packageLockSha256||p.backend!=='dawn-metal'||p.adapter?.isFallbackAdapter!==false)throw new Error('Native WebGPU installation differs from backend plan');
  }
  return await run(native);
 }finally{native?.uninstall();}
}
export function assertContributorBackendEvidence(evaluation,expectedBackend){
 if(!['wasm','webgpu'].includes(expectedBackend)||evaluation.backend!==expectedBackend||evaluation.provenance?.backend!==expectedBackend||evaluation.metrics?.actualNeuralBackend!==expectedBackend||evaluation.bodyBackend!=='mujoco-wasm'||evaluation.provenance?.bodyBackend!=='mujoco-wasm')throw new Error('Evaluation backend evidence differs from the planned engines');
}

export function assertGuardedContributorPlan(config,plan){
 validateTrainingConfigSchema(config);
 if(config.schemaVersion!==2)return;
 const pin=config.optimizer.acceptance.nativeExecution;
 if(plan?.backend!==pin.backend||plan.nativeWebGPU?.moduleSha256!==pin.moduleSha256||plan.nativeWebGPU?.packageLockSha256!==pin.packageLockSha256)
  throw new Error('Guarded training requires its pinned native Dawn backend plan');
}

export function guardedResultMetrics(evaluation,config){
 if(config.schemaVersion!==2)return {};
 if(evaluation.cancelled===true||typeof evaluation.terminated!=='boolean'||['cancelled','simulation_error','invalid_observation','unexpected_external_force'].includes(evaluation.reason))
  throw new Error('Invalid evaluation cannot enter guarded checkpoint comparison');
 return {terminated:evaluation.terminated,cancelled:false};
}

export async function runContributor(args=process.argv.slice(2)){
if(args.length<2||args.length>4)throw new Error('Usage: contribute-training-native.mjs origin reports/output maxJobs [backend-plan.json]');
const [endpoint,outputArg,maxJobsArg='8',backendPlanArg]=args;
const output=path.resolve(outputArg),maxJobs=Number(maxJobsArg);
if(!output.startsWith(path.join(root,'reports')+path.sep)||!Number.isInteger(maxJobs)||maxJobs<1||maxJobs>10000)throw new Error('Invalid report path or job count');
const backendPlanBytes=backendPlanArg?await fs.readFile(path.resolve(backendPlanArg)):null;
const backendPlan=backendPlanBytes?validateContributorBackendPlan(JSON.parse(backendPlanBytes)):null;
const url=validateContributorOrigin(endpoint,backendPlan);
await fs.mkdir(output,{recursive:true});
if((await fs.readdir(output)).some(name=>name.endsWith('.pending.json')))throw new Error('Unresolved result in output directory; reconcile it before starting another contributor');
const networkFetch=globalThis.fetch,locationDescriptor=Object.getOwnPropertyDescriptor(globalThis,'location');
const backendLabel=backendPlan?.backend??'wasm',expectedBackend=backendLabel==='dawn-metal'?'webgpu':'wasm';
const backendPlanSha256=backendPlanBytes?sha(backendPlanBytes):null,contributorSourceSha256=sha(await fs.readFile(fileURLToPath(import.meta.url)));
await withContributorBackend(backendPlan,async nativeWebGPU=>{
 let environment=null,active=null,heartbeat=null,stop=false,api=null,moduleHooks=null;
 const onSignal=()=>{stop=true;};process.on('SIGINT',onSignal);process.on('SIGTERM',onSignal);
 try{
const prefixes=[['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/'],['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
function localAsset(asset){const entry=prefixes.find(([prefix])=>asset.startsWith(prefix));const file=path.resolve(root,entry?entry[1]+asset.slice(entry[0].length):'web/'+asset.slice(1));if(!file.startsWith(root+path.sep))throw new Error('Asset escaped checkout');return file;}
moduleHooks=registerHooks({resolve(specifier,context,next){if(prefixes.some(([prefix])=>specifier.startsWith(prefix)))return {url:pathToFileURL(localAsset(specifier)).href,shortCircuit:true};return next(specifier,context);}});
globalThis.location={href:url.href};
globalThis.fetch=createContributorFetch(url,networkFetch);
api=async function(route,body){const response=await fetch('/api/training'+route,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined,signal:AbortSignal.timeout(15000)});const value=await response.json();if(!response.ok)throw new Error(`Coordinator ${response.status}: ${value.error||value.code}`);return value;};
const response=await fetch('/training/config.json');if(!response.ok)throw new Error('Cannot load coordinator configuration');
const raw=await response.text(),config=JSON.parse(raw),configHash=sha(raw);
if(backendPlan&&(backendPlan.configHash!==configHash||backendPlan.modelFingerprint!==config.modelFingerprint))throw new Error('Backend plan does not match coordinator configuration');
assertGuardedContributorPlan(config,backendPlan);
// Imported code comes from this checkout, so verify it against the served
// contract as well as the environment's checks of downloaded asset bytes.
async function verifyCode(){
 if(sha(await fs.readFile(fileURLToPath(import.meta.url)))!==contributorSourceSha256)throw new Error('Contributor source changed during contribution');
 if(nativeWebGPU&&(sha(await fs.readFile(path.join(root,'reports/native-webgpu-tooling/backend.mjs')))!==backendPlan.nativeWebGPU.moduleSha256||sha(await fs.readFile(path.join(root,'reports/native-webgpu-tooling/package-lock.json')))!==backendPlan.nativeWebGPU.packageLockSha256))throw new Error('Native WebGPU tooling changed during contribution');
 for(const [asset,digest]of Object.entries(config.assets))if(!asset.startsWith('/body-model/')&&sha(await fs.readFile(localAsset(asset)))!==digest)throw new Error('Local/served asset mismatch: '+asset);
}
await verifyCode();
const status=await api('/status');if(status.configHash!==configHash||status.modelFingerprint!==config.modelFingerprint)throw new Error('Coordinator contract mismatch');
let previousCheckpoint=status.checkpoint;
if(backendPlan?.stopAfterGeneration!==undefined&&previousCheckpoint.generation>=backendPlan.stopAfterGeneration){
 console.log(JSON.stringify({kind:'native-contributor-generation-limit',generation:previousCheckpoint.generation}));return;
}
const contributorId='native-'+randomUUID(),identity={contributorId,configHash,modelFingerprint:config.modelFingerprint};
const {createTrainingEnvironment}=await import('../web/training/environment.js');
environment=await createTrainingEnvironment(config,{onProgress:value=>console.log(value.message)});
async function stopping(){if(stop)return true;try{await fs.access(path.join(output,'STOP'));return true;}catch(error){if(error.code!=='ENOENT')throw error;return false;}}
 const ready=await environment.ready();if(ready.backend!==expectedBackend)throw new Error('Contributor requires its planned neural backend; fallback is not accepted');
 console.log(JSON.stringify({kind:'native-contributor-ready',...identity,backend:ready.backend,backendLabel,maxJobs}));
 for(let completed=0;completed<maxJobs&&!stop;){
  if(await stopping())break;
  if(backendPlan?.stopAfterGeneration!==undefined&&previousCheckpoint.generation>=backendPlan.stopAfterGeneration)break;
  await verifyCode();
  const leased=await api('/lease',identity),job=leased.job;
  if(!job){await new Promise(resolve=>setTimeout(resolve,Math.min(leased.waitMs||2000,10000)));continue;}
  if(job.configHash!==configHash||job.modelFingerprint!==config.modelFingerprint)throw new Error('Incompatible leased job');
  if(!/^[a-zA-Z0-9._:-]+$/.test(job.jobId))throw new Error('Invalid job identifier');
  if(backendPlan?.stopAfterGeneration!==undefined&&job.generation>=backendPlan.stopAfterGeneration){
   await api('/release',{...identity,jobId:job.jobId,leaseToken:job.leaseToken});break;
  }
  active={...identity,jobId:job.jobId,leaseToken:job.leaseToken};
  let latestPreview=null;
  heartbeat=createContributorHeartbeat({identity:active,send:payload=>api('/heartbeat',payload),getPreview:()=>latestPreview});
  let lastProgress=0;
  const frames=[];
  const evaluation=await environment.evaluate(job,{previewHz:.2,dutyCycle:1,
   checkpoint:async()=>{await new Promise(resolve=>setTimeout(resolve,0));heartbeat.check();if(await stopping()){const error=new Error('Operator stopped contributor');error.name='AbortError';throw error;}},
   onFrame:frame=>{latestPreview=makeContributorPreviewFrame(frame);if(frames.length<1000)frames.push(structuredClone(frame));},
   onProgress:value=>{if(performance.now()-lastProgress>15000){lastProgress=performance.now();console.log(JSON.stringify({kind:'progress',jobId:job.jobId,generation:job.generation,simSeconds:value.simSeconds,score:value.return}));}}
  });
  const {leaseToken:_token,...assignment}=job;
  const attempt=job.jobId+'-'+randomUUID();
  let verificationError=null;try{await verifyCode();}catch(error){verificationError=error;}
  await fs.writeFile(path.join(output,attempt+'.json'),JSON.stringify({scope:'Actual native contributor. Reduced model if specified by the coordinator; not Safari or a validation result. Distinct neural backends are not assumed numerically identical.',backendLabel,backendPlanSha256,contributorSourceSha256,nativeWebGPU:nativeWebGPU?.provenance??null,sourceVerificationError:verificationError?.message??null,assignment,evaluation,frames})+'\n',{flag:'wx'});
  if(verificationError)throw verificationError;
  assertContributorBackendEvidence(evaluation,expectedBackend);
  if(['cancelled','simulation_error'].includes(evaluation.reason))throw new Error('Incomplete evaluation: '+evaluation.reason);
  if(evaluation.backend!==expectedBackend||JSON.stringify(evaluation.parameters)!==JSON.stringify(job.parameters)||evaluation.seed!==job.seed||evaluation.stage!==job.stage||!Number.isFinite(evaluation.return)||!Number.isFinite(evaluation.simSeconds)||evaluation.simSeconds<=0||!Number.isInteger(evaluation.steps)||evaluation.steps<1)throw new Error('Evaluation differs from assignment');
  const full=Math.abs(evaluation.simSeconds-job.durationSeconds)<1e-7;
  const physicalFailure=!evaluation.success&&evaluation.terminated&&['invalid_observation','unexpected_external_force','outside_habitat','excessive_rotation','overturned'].includes(evaluation.reason);
  if(!full&&!physicalFailure||evaluation.simSeconds>job.durationSeconds+1e-7||Math.abs(evaluation.simSeconds-evaluation.steps*config.bodyBlockMs/1000)>1e-7)throw new Error('Evaluation did not complete its assigned horizon or a physical failure');
  const evidence={};for(const key of ['hasTakenOff','takeoffTime','flightSeconds','bestFlightSeconds','landingSeconds','landingTime','diagnostics','initialCondition','finalObservation','wallSeconds','setupWallSeconds','executionWallSeconds'])if(evaluation.metrics?.[key]!==undefined)evidence[key]=evaluation.metrics[key];
  const assignmentFields=Object.fromEntries(['parametersHash','durationSeconds','generation','sign','pairId'].map(key=>[key,job[key]]));
  const payload={...active,...assignmentFields,parameters:evaluation.parameters,objective:evaluation.return,metrics:{...evidence,success:evaluation.success,simSeconds:evaluation.simSeconds,steps:evaluation.steps,reason:evaluation.reason,...guardedResultMetrics(evaluation,config)},provenance:{...evaluation.provenance,...assignmentFields,parameters:evaluation.parameters,neuralEngine:backendLabel,contributorSourceSha256,...(backendPlanSha256?{backendPlanSha256}:{}),...(nativeWebGPU?{nativeWebGPU:nativeWebGPU.provenance}:{})}};
  await heartbeat.flush();await heartbeat.stop();heartbeat=null;
  const pending=path.join(output,attempt+'.pending.json');await fs.writeFile(pending,JSON.stringify(payload)+'\n',{flag:'wx',mode:0o600});
  let accepted;for(let attempt=0;attempt<3;attempt++){try{accepted=await api('/result',payload);break;}catch(error){if(attempt===2)throw error;await new Promise(resolve=>setTimeout(resolve,1000*(attempt+1)));}}
  if(accepted.accepted!==true)throw new Error('Coordinator did not accept result');
  active=null;
  await fs.unlink(pending);completed++;
  const current=await api('/status');await fs.writeFile(path.join(output,'checkpoint.json'),JSON.stringify(current.checkpoint,null,2)+'\n');
  if(current.checkpoint.generation!==previousCheckpoint.generation){
   const deltas=current.checkpoint.parameters.map((value,index)=>value-previousCheckpoint.parameters[index]);
   const update={kind:'parameter-update',fromGeneration:previousCheckpoint.generation,toGeneration:current.checkpoint.generation,
    changedParameterCount:deltas.filter(value=>value!==0).length,l2:Math.hypot(...deltas),maxAbs:Math.max(...deltas.map(Math.abs)),
    deltaByParameter:Object.fromEntries(config.parameters.map((parameter,index)=>[parameter.name,deltas[index]]))};
   await fs.appendFile(path.join(output,'parameter-updates.jsonl'),JSON.stringify(update)+'\n');console.log(JSON.stringify(update));
   previousCheckpoint=current.checkpoint;
  }
  console.log(JSON.stringify({kind:'accepted',jobId:job.jobId,generation:job.generation,completed,score:evaluation.return,takeoff:evaluation.metrics.hasTakenOff,bestFlightSeconds:evaluation.metrics.bestFlightSeconds,reason:evaluation.reason,nextGeneration:current.generation}));
 }
 }finally{
  if(heartbeat)await heartbeat.stop();if(active&&api)await api('/release',active).catch(()=>{});
  try{environment?.dispose();}finally{
   try{moduleHooks?.deregister();}finally{
    globalThis.fetch=networkFetch;
    if(locationDescriptor)Object.defineProperty(globalThis,'location',locationDescriptor);else delete globalThis.location;
    process.off('SIGINT',onSignal);process.off('SIGTERM',onSignal);
   }
  }
 }
});
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))await runContributor();
