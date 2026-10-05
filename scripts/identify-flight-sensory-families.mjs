// Seven frozen-current conditions. No body, muscle, controller or optimizer.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {loadBancModel} from '../packages/banc-runtime/src/model.js';
import {WebGPUBrain} from '../packages/banc-runtime/src/webgpu.js';
import {createWingMotorEventReader} from '../packages/banc-runtime/src/motor-events.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [endpoint,outputArg,flag]=process.argv.slice(2);
assert(endpoint&&outputArg&&(!flag||flag==='--prepare-only')&&process.argv.length<=5,'Usage: node scripts/identify-flight-sensory-families.mjs http://127.0.0.1:PORT/ reports/output [--prepare-only]');
const origin=new URL(endpoint),output=path.resolve(outputArg),prepareOnly=flag==='--prepare-only';
assert(origin.protocol==='http:'&&['127.0.0.1','localhost'].includes(origin.hostname)&&origin.pathname==='/'&&!origin.username&&!origin.password&&!origin.search&&!origin.hash);
assert(output.startsWith(path.join(root,'reports')+path.sep));await fs.mkdir(output,{recursive:true});
assert((await fs.realpath(output)).startsWith((await fs.realpath(path.join(root,'reports')))+path.sep));
const sha=b=>createHash('sha256').update(b).digest('hex');
const pack=a=>{const b=Buffer.alloc(a.length*4);a.forEach((v,i)=>b.writeFloatLE(v,i*4));return b.toString('base64');};
const unpack=s=>{const b=Buffer.from(s,'base64');return Float32Array.from({length:b.length/4},(_,i)=>b.readFloatLE(i*4));};
const prefixes=[['/banc-engine/','packages/banc-runtime/'],['/banc-data/','data/prepared/banc888/'],['/body-model/','models/']];
function localAsset(url){const entry=prefixes.find(([prefix])=>url.startsWith(prefix));const file=path.resolve(root,entry?entry[1]+url.slice(entry[0].length):'web/'+url.slice(1));assert(file.startsWith(root+path.sep));return file;}
const files={reference:'reports/flight-afferent-identification/robustness/prefix300-pulse1p25/result.json',
 annotations:'reports/flight-sensory-families/fallback-annotations.json',backend:'reports/native-webgpu-tooling/backend.mjs',backendLock:'reports/native-webgpu-tooling/package-lock.json'};
const bytes=Object.fromEntries(await Promise.all(Object.entries(files).map(async([key,file])=>[key,await fs.readFile(path.join(root,file))])));
const reference=JSON.parse(bytes.reference),annotations=JSON.parse(bytes.annotations);
assert(reference.baselineGate.passed&&reference.interpretationAllowed&&reference.protocol.prefixMs===300&&reference.prefixTrace.length===150);
const sourceHashes={...reference.sourceHashes};
sourceHashes['/banc-engine/src/motor-events.js']=sha(await fs.readFile(localAsset('/banc-engine/src/motor-events.js')));
for(const [url,digest]of Object.entries(sourceHashes))assert.equal(sha(await fs.readFile(localAsset(url))),digest,'Local pinned dependency changed: '+url);
for(const [file,digest]of Object.entries(reference.artifactHashes))assert.equal(sha(await fs.readFile(path.join(root,file))),digest,'Frozen source artifact changed: '+file);
for(const [file,digest]of Object.entries(annotations.sourceHashes))assert.equal(sha(await fs.readFile(path.join(root,file))),digest,'Fallback annotation source changed: '+file);
const groups=JSON.parse(await fs.readFile(localAsset('/banc-data/console/groups.json'))),sensory=JSON.parse(await fs.readFile(localAsset('/banc-data/console/sensory-inputs.json')));
const manifest=JSON.parse(await fs.readFile(localAsset('/banc-data/manifest.json'))),io=JSON.parse(await fs.readFile(localAsset('/banc-data/io.json')));
const currents=unpack(reference.context.currentsPa),encodedRates=unpack(reference.context.ratesHz),rateById=new Map(reference.context.indices.map((id,k)=>[id,encodedRates[k]]));
assert.equal(currents.length,manifest.neuron_count);assert(currents.every(Number.isFinite));
const masks={odor:[...groups.odor_left,...groups.odor_right],taste:[...groups.sweet],vision:sensory.vision.receptors.map(row=>row.index),body:sensory.channels.flatMap(row=>row.indices)};
assert.deepEqual([...masks.odor,...masks.taste,...masks.vision,...masks.body],reference.context.indices);
assert.equal(new Set(Object.values(masks).flat()).size,reference.context.indices.length,'Sensory families overlap');
const transducers=new Set(sensory.body_transducers.map(row=>row.index)),excluded=new Set((sensory.body_transducer_exclusions||[]).map(row=>row.index));
assert(masks.body.every(id=>!excluded.has(id)),'Excluded modality occurs inside the declared body mask');
masks.nativeTransducers=masks.body.filter(id=>transducers.has(id));masks.broadFallback=masks.body.filter(id=>!transducers.has(id));
assert.equal(masks.nativeTransducers.length,4430);assert.equal(masks.broadFallback.length,629);
assert.deepEqual([...masks.nativeTransducers,...masks.broadFallback].sort((a,b)=>a-b),[...masks.body].sort((a,b)=>a-b));
assert.deepEqual(annotations.rows.map(row=>row.index).sort((a,b)=>a-b),[...masks.broadFallback].sort((a,b)=>a-b));
const sensoryIds=new Set(reference.context.indices);assert(currents.every((v,id)=>v===0||sensoryIds.has(id)));
assert(masks.vision.every(id=>currents[id]===0));assert(masks.taste.every(id=>currents[id]===0));
const stats=ids=>{const values=ids.map(id=>currents[id]),hz=ids.map(id=>rateById.get(id)),active=values.filter(v=>v!==0).sort((a,b)=>a-b);return {cells:ids.length,nonzeroCurrentCells:active.length,
 requestedRateHz:{minimum:Math.min(...hz),maximum:Math.max(...hz),mean:hz.reduce((a,b)=>a+b,0)/hz.length},
 currentPa:{minimum:Math.min(...values),maximum:Math.max(...values),sum:values.reduce((a,b)=>a+b,0),activeMedian:active[Math.floor(active.length/2)]??null}};};
const familyStats=Object.fromEntries(Object.entries(masks).map(([key,ids])=>[key,stats(ids)]));
assert.equal(familyStats.body.nonzeroCurrentCells,1630);assert.equal(familyStats.odor.nonzeroCurrentCells,146);
const definitions=[['all',reference.context.indices],['no-external',[]],['body-only',masks.body],['odor-only',masks.odor],['taste-only',masks.taste],['native-transducer-only',masks.nativeTransducers],['broad-fallback-only',masks.broadFallback]];
const inputs=new Map(definitions.map(([name,indices])=>{const input=new Float32Array(currents.length);indices.forEach(id=>{input[id]=currents[id];});return [name,input];}));
assert.deepEqual(inputs.get('all'),currents);assert.deepEqual(inputs.get('no-external'),inputs.get('taste-only'));
for(let i=0;i<currents.length;i++){
 assert.equal(currents[i],inputs.get('body-only')[i]+inputs.get('odor-only')[i]+inputs.get('taste-only')[i]);
 assert.equal(inputs.get('body-only')[i],inputs.get('native-transducer-only')[i]+inputs.get('broad-fallback-only')[i]);
}
const wingMappings=io.muscles.filter(row=>['asynchronous_wing','wing_steering_assumption'].includes(row.kind));
const indices=Uint32Array.from([...new Set(wingMappings.flatMap(row=>row.indices))].sort((a,b)=>a-b));assert.equal(indices.length,48);assert.equal(wingMappings.length,28);
const internal=structuredClone(reference.context.internal),scriptSha256=sha(await fs.readFile(fileURLToPath(import.meta.url)));
const plan={schemaVersion:1,kind:'predeclared-frozen-sensory-family-identification',scriptSha256,neuralFingerprint:reference.neuralFingerprint,
 scope:'Seven fresh full-BANC mutable states on one shared immutable graph.300ms constant external current per arm; no body, muscle, controller, optimizer, axis assignment or neural parameter change.',
 sourceHashes,artifactHashes:Object.fromEntries(Object.entries(files).map(([key,file])=>[file,sha(bytes[key])])),
 frozenContext:reference.context,internal,retainedCurrents:'Only the external input array is masked. params[11] intrinsic current, neuromodulator gain, params[15] internal-state response, recurrent inputs and gap junctions remain unchanged.',
 sampling:{durationMs:300,dtMs:.5,readoutMs:2,readStride:9,wingNeuronCount:48,gaps:true},
 masks,familyStats,fallbackAnatomy:annotations.summary,
 gates:['Disjoint complete food/body/vision masks and native-transducer/fallback body partition.','Fresh initial full forward-state hashes equal across all arms.','All-input48×8 trajectory and full final forward-state hashes equal the saved300ms reference prefix.','Zero-external and taste-only traces and full final forward-state bytes equal exactly.'],
 arms:definitions.map(([name,ids])=>({name,selectedCells:ids.length,nonzeroCurrentCells:inputs.get(name).filter(v=>v!==0).length,inputSha256:sha(Buffer.from(inputs.get(name).buffer))})),
 limitations:['This frozen200ms feedback context is artificial and never updates from the body.','Requested input rates are current-map targets, not guaranteed afferent firing rates inside the graph.','Summed currents across cells do not measure equivalent circuit or physiological drive.','Taste is inactive here, so taste-only duplicates zero-external and cannot test active taste.','No DLM recruitment difference establishes flight or proves the underlying sensory/receptor assumptions correct.']};
const planPath=path.join(output,'plan.json');
try{const existing=JSON.parse(await fs.readFile(planPath));assert.deepEqual(existing,plan,'Predeclared plan changed');}catch(error){if(error.code!=='ENOENT')throw error;await fs.writeFile(planPath,JSON.stringify(plan,null,2)+'\n',{flag:'wx'});}
if(prepareOnly){console.log(JSON.stringify({prepared:true,neuralExecution:false,arms:plan.arms,familyStats,plan:planPath}));process.exit(0);}
assert(!(await fs.readdir(output)).some(name=>name==='result.json'||definitions.some(([arm])=>name===arm+'.json')),'Use an output without prior experiment results');
const networkFetch=globalThis.fetch;globalThis.location={href:origin.href};
globalThis.fetch=(input,options={})=>{let url=new URL(typeof input==='string'?input:input.url||String(input),origin);
 if(url.href===pathToFileURL(path.join(root,'packages/banc-runtime/src/neural.wgsl')).href)url=new URL('/banc-engine/src/neural.wgsl',origin);
 const method=String(options.method||input?.method||'GET').toUpperCase();assert(url.origin===origin.origin&&['GET','HEAD'].includes(method));
 return networkFetch(url,{...options,method,redirect:'error',signal:options.signal||AbortSignal.timeout(30000)});};
for(const [url,digest]of Object.entries(sourceHashes)){const response=await fetch(url);assert(response.ok,'Missing '+url);assert.equal(sha(Buffer.from(await response.arrayBuffer())),digest,'Served dependency changed: '+url);}
const model=await loadBancModel();assert.deepEqual(model.manifest,manifest);assert.deepEqual(model.io,io);
const paramsHash=sha(Buffer.from(model.params.buffer,model.params.byteOffset,model.params.byteLength));
const report={schemaVersion:1,kind:'frozen-sensory-family-identification',startedAt:new Date().toISOString(),planSha256:sha(await fs.readFile(planPath)),scriptSha256,paramsHash,
 sourceHashes,artifactHashes:plan.artifactHashes,wingIndices:Array.from(indices),wingMappings,internal,arms:[],allInputReferenceGate:{passed:false},zeroTasteGate:{passed:false},interpretationAllowed:false,error:null};
let provider,anchor,stop=false;process.on('SIGINT',()=>{stop=true;});process.on('SIGTERM',()=>{stop=true;});
async function checkpoint(){await new Promise(resolve=>setTimeout(resolve,0));let marker=false;try{await fs.access(path.join(output,'STOP'));marker=true;}catch(error){if(error.code!=='ENOENT')throw error;}if(stop||marker){const error=new Error('Operator stopped experiment');error.name='AbortError';throw error;}}
async function verifyPins(){for(const [url,digest]of Object.entries(sourceHashes))assert.equal(sha(await fs.readFile(localAsset(url))),digest);assert.equal(sha(Buffer.from(model.params.buffer,model.params.byteOffset,model.params.byteLength)),paramsHash);}
async function forwardState(brain){
 const output={};for(const [name,buffer,length]of [['state0',brain.states[0]],['state1',brain.states[1]],['history',brain.history],['kinetics',brain.kinetics],['inputCurrents',brain.inputs,brain.n*4]]){
  const size=length??buffer.size,staging=brain.device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{const encoder=brain.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,staging,0,size);brain.device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);output[name]=Buffer.from(new Uint8Array(staging.getMappedRange()).slice());staging.unmap();}finally{staging.destroy();}
 }return output;
}
const hashes=state=>Object.fromEntries(Object.entries(state).map(([key,value])=>[key,sha(value)]));
const referenceOffset=new Map(reference.readIds.map((id,k)=>[id,k]));
let initialHashes,zeroState,zeroTrace;
try{
 const {installNativeWebGPU}=await import(pathToFileURL(path.join(root,files.backend)));provider=await installNativeWebGPU();report.backend=provider.provenance;
 assert.equal(provider.provenance.moduleSha256,sha(bytes.backend));assert.equal(provider.provenance.packageLockSha256,sha(bytes.backendLock));
 anchor=await WebGPUBrain.create(model,{gpu:provider.gpu});initialHashes=hashes(await forwardState(anchor));report.initialForwardStateHashes=initialHashes;
 for(const [name]of definitions){
  await checkpoint();await verifyPins();const brain=await WebGPUBrain.create(model,{shared:anchor});
  const arm={name,inputSha256:plan.arms.find(row=>row.name===name).inputSha256,trace:[],completed:false,error:null};let finalState;
  try{
   assert.equal(brain.timeMs,0);assert.deepEqual(hashes(await forwardState(brain)),initialHashes);
   const reader=createWingMotorEventReader({indices,params:model.params,dtMs:.5,bodyBlockMs:2});
   const serialize=packet=>({...packet,indices:Array.from(packet.indices),ratesHz:Array.from(packet.ratesHz),counts:Array.from(packet.counts)});
   arm.initial=serialize(reader.read(await brain.readState(indices,{includeSpikeTime:true}),0));
   for(let block=0;block<150;block++){
    await checkpoint();await brain.step(4,inputs.get(name),internal,true);const state=await brain.readState(indices,{includeSpikeTime:true});
    const packet=serialize(reader.read(state,brain.timeMs));arm.trace.push({...packet,state9:pack(state),globalSpikes:state.totalSpikes});
    if(name==='all'){
     const expected=unpack(reference.prefixTrace[block].neuralState);
     for(let k=0;k<indices.length;k++)for(let field=0;field<8;field++)assert(Object.is(state[k*9+field],expected[referenceOffset.get(indices[k])*8+field]),`Reference mismatch block${block},cell${indices[k]},field${field}`);
     assert.equal(state.totalSpikes,reference.prefixTrace[block].globalSpikes);
    }
   }
   finalState=await forwardState(brain);arm.finalForwardStateHashes=hashes(finalState);
   if(name==='all'){assert.deepEqual(arm.finalForwardStateHashes,reference.prefixForwardStateHashes);report.allInputReferenceGate={passed:true,checkedBlocks:150,checkedNeurons:48,checkedFields:8,finalForwardStateHashes:arm.finalForwardStateHashes};}
   if(name==='no-external'){zeroState=finalState;zeroTrace=arm.trace;}
   if(name==='taste-only'){
    assert.deepEqual(arm.trace,zeroTrace);for(const [key,value]of Object.entries(zeroState))assert(value.equals(finalState[key]),'Zero/taste forward state mismatch: '+key);
    report.zeroTasteGate={passed:true,checkedBlocks:150,checkedForwardBytes:Object.values(zeroState).reduce((sum,value)=>sum+value.length,0)};zeroState=null;zeroTrace=null;
   }
   await verifyPins();arm.completed=true;
  }catch(error){arm.error=error.stack||error.message;throw error;}
  finally{brain.dispose();await fs.writeFile(path.join(output,name+'.json'),JSON.stringify(arm)+'\n',{flag:'wx'});report.arms.push({name,completed:arm.completed,error:arm.error,file:name+'.json'});}
  console.log(JSON.stringify({kind:'arm-complete',name,neuralMs:300,allInputGate:report.allInputReferenceGate.passed,zeroTasteGate:report.zeroTasteGate.passed}));
 }
 assert.deepEqual(hashes(await forwardState(anchor)),initialHashes);assert.equal(report.arms.length,7);assert(report.allInputReferenceGate.passed&&report.zeroTasteGate.passed);report.interpretationAllowed=true;
}catch(error){report.error=error.stack||error.message;process.exitCode=1;}
finally{anchor?.dispose();provider?.uninstall();report.finishedAt=new Date().toISOString();await fs.writeFile(path.join(output,'result.json'),JSON.stringify(report)+'\n',{flag:'wx'});}
console.log(JSON.stringify({output,completed:report.arms.filter(arm=>arm.completed).length,interpretationAllowed:report.interpretationAllowed,error:report.error}));
