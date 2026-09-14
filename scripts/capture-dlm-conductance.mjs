// Development-only neural observation. No body, input encoder or model mutation.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {loadBancModel} from '../packages/banc-runtime/src/model.js';
import {WebGPUBrain} from '../packages/banc-runtime/src/webgpu.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [endpoint,outputArg,flag]=process.argv.slice(2);
assert(endpoint&&outputArg&&(!flag||flag==='--prepare-only')&&process.argv.length<=5,
 'Usage: node scripts/capture-dlm-conductance.mjs http://127.0.0.1:PORT/ reports/output [--prepare-only]');
const origin=new URL(endpoint),output=path.resolve(outputArg),prepareOnly=flag==='--prepare-only';
assert(origin.protocol==='http:'&&['127.0.0.1','localhost'].includes(origin.hostname)&&origin.pathname==='/'&&!origin.username&&!origin.password&&!origin.search&&!origin.hash);
assert(output.startsWith(path.join(root,'reports')+path.sep));await fs.mkdir(output,{recursive:true});
assert((await fs.realpath(output)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const pack=array=>{const bytes=Buffer.alloc(array.length*4);array.forEach((v,i)=>bytes.writeFloatLE(v,i*4));return {encoding:'float32-little-endian-base64',data:bytes.toString('base64')};};
const unpack=value=>{const bytes=Buffer.from(typeof value==='string'?value:value.data,'base64');assert.equal(bytes.length%4,0);return Float32Array.from({length:bytes.length/4},(_,i)=>bytes.readFloatLE(i*4));};
const bitEqual=(a,b,label)=>{assert.equal(a.length,b.length,label+' length');for(let k=0;k<a.length;k++)assert(Object.is(a[k],b[k]),label+' field '+k);};
const sourceUrls=['/banc-engine/src/model.js','/banc-engine/src/webgpu.js','/banc-engine/src/neural.wgsl'];
function localAsset(url){
 const [prefix,directory]=url.startsWith('/banc-engine/')?['/banc-engine/','packages/banc-runtime/']:['/banc-data/','data/prepared/banc888/'];
 assert(url.startsWith(prefix));const file=path.resolve(root,directory+url.slice(prefix.length));
 assert(file.startsWith(path.resolve(root,directory)+path.sep));return file;
}
const artifactFiles={familyPlan:'reports/flight-sensory-families/run/plan.json',familyResult:'reports/flight-sensory-families/run/result.json',
 familyAll:'reports/flight-sensory-families/run/all.json',prefix:'reports/flight-afferent-identification/robustness/prefix300-pulse1p25/result.json'};
const artifactBytes=Object.fromEntries(await Promise.all(Object.entries(artifactFiles).map(async([key,file])=>[key,await fs.readFile(path.join(root,file))])));
const familyPlan=JSON.parse(artifactBytes.familyPlan),familyResult=JSON.parse(artifactBytes.familyResult),familyAll=JSON.parse(artifactBytes.familyAll),prefix=JSON.parse(artifactBytes.prefix);
assert(familyResult.interpretationAllowed&&!familyResult.error&&familyResult.allInputReferenceGate.passed&&familyResult.zeroTasteGate.passed);
assert.equal(familyResult.planSha256,sha(artifactBytes.familyPlan));assert(familyAll.completed&&!familyAll.error&&familyAll.trace.length===150);
assert(prefix.interpretationAllowed&&!prefix.error&&prefix.baselineGate.passed&&prefix.protocol.prefixMs===300);
assert.equal(familyPlan.artifactHashes[artifactFiles.prefix],sha(artifactBytes.prefix));
assert.deepEqual(familyAll.finalForwardStateHashes,prefix.prefixForwardStateHashes);
const context=structuredClone(familyPlan.frozenContext),currents=unpack(context.currentsPa),internal=structuredClone(familyPlan.internal);
assert.deepEqual(internal,context.internal);assert(currents.every(Number.isFinite));
const manifestBytes=await fs.readFile(localAsset('/banc-data/manifest.json')),manifest=JSON.parse(manifestBytes);
assert.equal(sha(manifestBytes),familyResult.sourceHashes['/banc-data/manifest.json']);assert.equal(manifest.dt_ms,.5);
assert.equal(currents.length,manifest.neuron_count);
const neuralHashes=Object.fromEntries(await Promise.all(sourceUrls.map(async url=>{
 const bytes=await fs.readFile(localAsset(url)),digest=sha(bytes);assert.equal(digest,familyResult.sourceHashes[url],'Neural source changed: '+url);return [url,digest];
})));
const dataBytes=Object.fromEntries(await Promise.all(['offsets.bin','edges.bin','params.bin','io.json','ids.bin'].map(async name=>{
 const bytes=await fs.readFile(localAsset('/banc-data/'+name));assert.equal(bytes.length,manifest.files[name].bytes);assert.equal(sha(bytes),manifest.files[name].sha256);return [name,bytes];
})));
const io=JSON.parse(dataBytes['io.json']),params=unpack(dataBytes['params.bin'].toString('base64'));
const offsets=Uint32Array.from({length:dataBytes['offsets.bin'].length/4},(_,i)=>dataBytes['offsets.bin'].readUInt32LE(i*4));
const mappings=io.muscles.filter(row=>row.kind==='asynchronous_wing'&&row.target==='dorsal_longitudinal_muscle');
const indices=Uint32Array.from([...new Set(mappings.flatMap(row=>row.indices))].sort((a,b)=>a-b));
assert.equal(indices.length,10);assert.equal(mappings.length,2);assert(mappings.every(row=>row.indices.length===5));
const gapInputs=[];
for(const index of indices){
 assert.equal(currents[index],0);assert.equal(params[index*16+11],0);assert.equal(params[index*16+15],0);
 for(let e=offsets[manifest.neuron_count+1+index];e<offsets[manifest.neuron_count+2+index];e++)gapInputs.push({post:index,source:dataBytes['edges.bin'].readUInt32LE(e*16),conductanceNs:dataBytes['edges.bin'].readFloatLE(e*16+4)});
}
assert.equal(gapInputs.length,0,'Primary replay assumes no native DLM electrical edges');
const parameterRows=new Float32Array(indices.length*16);indices.forEach((id,k)=>parameterRows.set(params.subarray(id*16,id*16+16),k*16));
const receptorRows=Float32Array.from(manifest.receptors.flatMap(row=>[row.rise_ms,row.decay_ms,row.reversal_mv]));
const backendFile='reports/native-webgpu-tooling/backend.mjs',backendLockFile='reports/native-webgpu-tooling/package-lock.json';
assert.equal(sha(await fs.readFile(path.join(root,backendFile))),familyResult.backend.moduleSha256);
assert.equal(sha(await fs.readFile(path.join(root,backendLockFile))),familyResult.backend.packageLockSha256);
const source=await fs.readFile(fileURLToPath(import.meta.url)),scriptSha256=sha(source);
const contextBytes=Buffer.from(JSON.stringify(context)+'\n');
const layout={strideFloats:36,oldState8:[0,8],newState8:[8,16],postReceptorRiseDecay18:[16,34],lastSpikeTimeMs:34,externalCurrentPa:35};
const plan={schemaVersion:1,kind:'dlm-frozen-conductance-capture-plan',scriptSha256,
 scope:'One fresh full-BANC native-Dawn brain,300ms fixed saved external currents. No body, sensory re-encoding, muscles, controller, optimizer or neural-model change.',
 sourceHashes:neuralHashes,preparedManifestSha256:sha(manifestBytes),preparedFiles:manifest.files,
 artifactHashes:Object.fromEntries(Object.entries(artifactFiles).map(([key,file])=>[file,sha(artifactBytes[key])])),
 contextFile:'frozen-context.used.json',contextSha256:sha(contextBytes),backendModuleSha256:familyResult.backend.moduleSha256,backendLockSha256:familyResult.backend.packageLockSha256,
 sourceExclusions:'The old encoder, console sensory mapping, body and muscle code are not executed. Their original hashes remain in the pinned historical family plan; current unrelated mapping changes are intentionally not required to match.',
 protocol:{durationMs:300,dtMs:.5,steps:600,readoutEveryStep:true,baselineCheckEveryMs:2,gaps:true,internal,inputSha256:sha(Buffer.from(currents.buffer)),sameCurrentAcrossOriginal2msBlocks:true},
 indices:Array.from(indices),mappings,parameterRows:pack(parameterRows),receptorRows:pack(receptorRows),receptors:manifest.receptors,gapInputs,
 units:{time:'ms',voltage:'mV',conductance:'nS',current:'pA',capacitance:'pF',release:'ms^-1'},
 packedLayout:layout,
 timing:'Each post-kinetics decay gate was consumed by the native update over(timeBeforeMs,timeAfterMs]; it must drive that same replay interval. Row oldState8 is the unchanged previous-state buffer after step1, row newState8 is the updated buffer. Initial row has both initial buffers.',
 driveSeparation:'Captured externalCurrentPa is the raw neural input array. Intrinsic/hormone/monoamine terms remain separately recoverable from params,internal and gates. For these10DLM external+tonic=0 and hormone coefficient=0, so current before adaptation is exactly0. State7 includes negative legacy adaptation and excludes chemical driving current. Never use state7 as the ionic replay stimulus.',
 gates:['Initial full forward-state hashes equal historical fresh state.','Every600 oldState8 row equals the preceding captured newState8.','All10 DLM8-state fields and exact last-spike time match150 saved2ms all-input rows.','Both full state buffers,history,kinetics and inputCurrents hashes match the historical300ms final state.','No relevant source/model/context bytes change during capture.'],
 expectedInitialForwardStateHashes:familyResult.initialForwardStateHashes,expectedFinalForwardStateHashes:familyAll.finalForwardStateHashes};
async function writeSameOrNew(file,bytes){try{assert((await fs.readFile(file)).equals(bytes),'Existing immutable artifact differs: '+file);}catch(error){if(error.code!=='ENOENT')throw error;await fs.writeFile(file,bytes,{flag:'wx'});}}
await writeSameOrNew(path.join(output,'plan.json'),Buffer.from(JSON.stringify(plan,null,2)+'\n'));
await writeSameOrNew(path.join(output,'frozen-context.used.json'),contextBytes);
await writeSameOrNew(path.join(output,'source.used.mjs'),source);
await fs.mkdir(path.join(output,'neural-source'),{recursive:true});
for(const url of sourceUrls)await writeSameOrNew(path.join(output,'neural-source',path.basename(url)),await fs.readFile(localAsset(url)));
if(prepareOnly){console.log(JSON.stringify({prepared:true,neuralExecution:false,plan:path.join(output,'plan.json'),neurons:indices.length,steps:600,scriptSha256}));process.exit(0);}
assert(!(await fs.readdir(output)).some(name=>['result.json','trace.f32','initial.f32'].includes(name)),'Use an output without prior capture results');

const networkFetch=globalThis.fetch;globalThis.location={href:origin.href};
globalThis.fetch=(input,options={})=>{
 let url=new URL(typeof input==='string'?input:input.url||String(input),origin);
 if(url.href===pathToFileURL(path.join(root,'packages/banc-runtime/src/neural.wgsl')).href)url=new URL('/banc-engine/src/neural.wgsl',origin);
 const method=String(options.method||input?.method||'GET').toUpperCase();
 assert(url.origin===origin.origin&&['GET','HEAD'].includes(method));assert(sourceUrls.includes(url.pathname)||url.pathname.startsWith('/banc-data/'));
 return networkFetch(url,{...options,method,redirect:'error',signal:options.signal||AbortSignal.timeout(30000)});
};
for(const [url,digest]of Object.entries(neuralHashes)){const response=await fetch(url);assert(response.ok);assert.equal(sha(Buffer.from(await response.arrayBuffer())),digest);}
const model=await loadBancModel();assert.deepEqual(model.manifest,manifest);assert.deepEqual(model.io,io);
assert.equal(sha(Buffer.from(model.params.buffer,model.params.byteOffset,model.params.byteLength)),familyResult.paramsHash);
let brain,provider,staging,stop=false;process.on('SIGINT',()=>{stop=true;});process.on('SIGTERM',()=>{stop=true;});
const report={schemaVersion:1,kind:'dlm-frozen-conductance-capture',startedAt:new Date().toISOString(),planSha256:sha(await fs.readFile(path.join(output,'plan.json'))),
 scriptSha256,sourceHashes:neuralHashes,indices:Array.from(indices),layout,stepsCaptured:0,gates:{initialState:false,oldNewContinuity:false,baseline2ms:false,finalState:false,sourcePins:false},completed:false,interpretationAllowed:false,error:null};
const frames=[];
async function checkpoint(){await new Promise(resolve=>setTimeout(resolve,0));let marker=false;try{await fs.access(path.join(output,'STOP'));marker=true;}catch(error){if(error.code!=='ENOENT')throw error;}if(stop||marker){const error=new Error('Operator stopped capture');error.name='AbortError';throw error;}}
async function verifyPins(){
 for(const [url,digest]of Object.entries(neuralHashes))assert.equal(sha(await fs.readFile(localAsset(url))),digest);
 for(const [key,file]of Object.entries(artifactFiles))assert.equal(sha(await fs.readFile(path.join(root,file))),sha(artifactBytes[key]));
 assert.equal(sha(await fs.readFile(fileURLToPath(import.meta.url))),scriptSha256);
 assert.equal(sha(await fs.readFile(localAsset('/banc-data/manifest.json'))),sha(manifestBytes));
 assert.equal(sha(Buffer.from(model.params.buffer,model.params.byteOffset,model.params.byteLength)),familyResult.paramsHash);
}
async function forwardStateHashes(){
 const hashes={};for(const [name,buffer,length]of [['state0',brain.states[0]],['state1',brain.states[1]],['history',brain.history],['kinetics',brain.kinetics],['inputCurrents',brain.inputs,brain.n*4]]){
  const size=length??buffer.size,b=brain.device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const encoder=brain.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,b,0,size);brain.device.queue.submit([encoder.finish()]);await b.mapAsync(GPUMapMode.READ);hashes[name]=sha(new Uint8Array(b.getMappedRange()));}
  finally{if(b.mapState==='mapped')b.unmap();b.destroy();}
 }return hashes;
}
async function packedRead(){
 brain.live();assert(!brain.busy);const encoder=brain.device.createCommandEncoder(),parity=brain.tick%2;
 indices.forEach((id,k)=>{
  const start=k*layout.strideFloats*4;
  encoder.copyBufferToBuffer(brain.states[brain.tick?1-parity:parity],id*32,staging,start,32);
  encoder.copyBufferToBuffer(brain.states[parity],id*32,staging,start+32,32);
  encoder.copyBufferToBuffer(brain.kinetics,id*72,staging,start+64,72);
  encoder.copyBufferToBuffer(brain.kinetics,(brain.n*18+id)*4,staging,start+136,4);
  encoder.copyBufferToBuffer(brain.inputs,id*4,staging,start+140,4);
 });
 brain.device.queue.submit([encoder.finish()]);
 try{await staging.mapAsync(GPUMapMode.READ);brain.live();const a=new Float32Array(staging.getMappedRange()).slice();assert(a.every(Number.isFinite));return a;}
 finally{if(staging.mapState==='mapped')staging.unmap();}
}
try{
 await checkpoint();await verifyPins();
 const {installNativeWebGPU}=await import(pathToFileURL(path.join(root,backendFile)));provider=await installNativeWebGPU();report.backend=provider.provenance;
 assert.deepEqual(provider.provenance,familyResult.backend,'Backend changed from saved reference');
 brain=await WebGPUBrain.create(model,{gpu:provider.gpu});assert.equal(brain.timeMs,0);
 report.initialForwardStateHashes=await forwardStateHashes();assert.deepEqual(report.initialForwardStateHashes,plan.expectedInitialForwardStateHashes);report.gates.initialState=true;
 staging=brain.device.createBuffer({size:indices.length*layout.strideFloats*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 let previous=await packedRead();const initial=Buffer.from(pack(previous).data,'base64');await fs.writeFile(path.join(output,'initial.f32'),initial,{flag:'wx'});report.initial={file:'initial.f32',sha256:sha(initial),bytes:initial.length};
 const savedOffset=new Map(familyResult.wingIndices.map((id,k)=>[id,k]));let checked=0;
 for(let step=0;step<600;step++){
  await checkpoint();await brain.step(1,currents,internal,true);const state=await packedRead();
  indices.forEach((id,k)=>{
   const s=k*layout.strideFloats;bitEqual(state.subarray(s,s+8),previous.subarray(s+8,s+16),'Old/new continuity step '+step+',cell '+id);
   assert.equal(state[s+35],currents[id]);assert.equal(state[s+35],0);
   if((step+1)%4===0){const saved=unpack(familyAll.trace[(step+1)/4-1].state9),pos=savedOffset.get(id)*9;
    bitEqual(state.subarray(s+8,s+16),saved.subarray(pos,pos+8),'Saved2ms baseline step '+step+',cell '+id);
    assert(Object.is(state[s+34],saved[pos+8]),'Saved last-spike time mismatch');
   }
  });
  if((step+1)%4===0)checked++;
  frames.push(Buffer.from(pack(state).data,'base64'));report.stepsCaptured=step+1;previous=state;
  if((step+1)%100===0){await verifyPins();console.log(JSON.stringify({kind:'capture-progress',steps:step+1,neuralMs:brain.timeMs,checked2msRows:checked}));}
 }
 assert.equal(checked,150);report.gates.oldNewContinuity=true;report.gates.baseline2ms=true;report.checked2msRows=checked;
 report.finalForwardStateHashes=await forwardStateHashes();assert.deepEqual(report.finalForwardStateHashes,plan.expectedFinalForwardStateHashes);report.gates.finalState=true;
 await verifyPins();report.gates.sourcePins=true;report.completed=true;report.interpretationAllowed=true;
}catch(error){report.error=error.stack||error.message;process.exitCode=1;}
finally{
 staging?.destroy();brain?.dispose();provider?.uninstall();globalThis.fetch=networkFetch;
 const trace=Buffer.concat(frames);await fs.writeFile(path.join(output,'trace.f32'),trace,{flag:'wx'});
 report.trace={file:'trace.f32',sha256:sha(trace),bytes:trace.length,encoding:'float32-little-endian',rows:frames.length,neuronsPerRow:indices.length,strideFloats:layout.strideFloats,
  timeConvention:'Rowk applies over(k*.5,(k+1)*.5]ms. See plan.packedLayout; initial.f32 holds the time0 row.'};
 report.finishedAt=new Date().toISOString();await fs.writeFile(path.join(output,'result.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
}
console.log(JSON.stringify({output,stepsCaptured:report.stepsCaptured,interpretationAllowed:report.interpretationAllowed,error:report.error}));
