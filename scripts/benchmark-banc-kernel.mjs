// Sequential A/B on the complete prepared BANC graph. No body, coordinator,
// training results or optimizer writes. Pause other heavy work before timing.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {WasmBrain} from '../packages/banc-runtime/src/wasm.js';
import {validateModel,historySlots,SPIKE_CAPACITY} from '../packages/banc-runtime/src/model.js';
import {createWingMotorEventReader} from '../packages/banc-runtime/src/motor-events.js';

const [baselineDirectory,optimizedDirectory,outputFile,...flags]=process.argv.slice(2);
if(!baselineDirectory||!optimizedDirectory||!outputFile)throw new Error('Usage: node scripts/benchmark-banc-kernel.mjs BASELINE_DIR OPTIMIZED_DIR OUTPUT.json [--blocks=40 --warmup=10 --repeats=3]');
const options={blocks:40,warmup:10,repeats:3};
for(const flag of flags){const match=/^--(blocks|warmup|repeats)=(\d+)$/.exec(flag);if(!match)throw new Error('Unknown option: '+flag);options[match[1]]=Number(match[2]);}
if(options.blocks<1||options.blocks>1000||options.warmup>1000||options.repeats<1||options.repeats>10)throw new Error('Invalid benchmark bounds');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const readJson=async file=>JSON.parse(await fs.readFile(file,'utf8'));
const configBytes=await fs.readFile('reports/motor-decoder-v1/browser-wasm-001/config.json'),config=JSON.parse(configBytes);
const baseDirectory='data/prepared/banc888';
const manifest=await readJson(path.join(baseDirectory,'manifest.json')),fileHashes={};
async function readModel(name,Type){
  const data=await fs.readFile(path.join(baseDirectory,name));
  const expected=manifest.files[name];assert.equal(data.length,expected.bytes);assert.equal(sha(data),expected.sha256);
  fileHashes[name]=sha(data);
  return Type?new Type(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength)):JSON.parse(data);
}
const model=validateModel({manifest:{...manifest,intrinsic_models:structuredClone(config.intrinsicModels)},
  offsets:await readModel('offsets.bin',Uint32Array),edges:await readModel('edges.bin',Uint32Array),params:await readModel('params.bin',Float32Array),io:await readModel('io.json')});
const motorIds=Uint32Array.from(model.io.motor_neurons,m=>m.index),wingIds=Uint32Array.from(config.motorDecoderContract.indices);
const sensoryIds=[...new Set(model.io.sensory.map(s=>s.index))].sort((a,b)=>a-b);
const patterns=Array.from({length:4},(_,phase)=>{
  const input=new Float32Array(manifest.neuron_count);
  for(const index of sensoryIds){
    const p=index*16,scale=[0,.65,1.2,1.8][(index+phase)%4];
    input[index]=scale*model.params[p+1]*(model.params[p+3]-model.params[p+2]);
  }
  return input;
});
const variants=[];
for(const [name,directory]of [['baseline',baselineDirectory],['optimized',optimizedDirectory]]){
  const js=path.resolve(directory,'core.js'),bytes=await fs.readFile(path.resolve(directory,'core.wasm'));
  const create=(await import(pathToFileURL(js))).default;
  variants.push({name,create,bytes,sha256:sha(bytes)});
}
assert.equal(variants[0].sha256,config.optimizer.acceptance.nativeExecution.moduleSha256,'Baseline must be the pinned browser experiment binary');
assert.notEqual(variants[0].sha256,variants[1].sha256,'Baseline and candidate must be distinct binaries');
function digestBrain(brain){
  const hash=createHash('sha256'),n=brain.n;
  for(const [pointer,length]of [[brain.params,brain.intrinsic.packedLength*4],...brain.states.map(p=>[p,n*8*4]),
    [brain.history,n*historySlots(model)*4],[brain.kinetics,brain.intrinsic.kineticsLength*4],[brain.events,(2+SPIKE_CAPACITY*2)*4]])
    hash.update(brain.core.HEAPU8.subarray(pointer,pointer+length));
  return hash.digest('hex');
}
const references=[],samples=[];
const packetDigest=packet=>sha(JSON.stringify(packet));
async function run(variant,repetition){
  const core=await variant.create({wasmBinary:variant.bytes});
  let kernelMs=0,kernelCalls=0;
  const original=core._br_step_dlm;
  core._br_step_dlm=(...args)=>{const start=performance.now();try{return original(...args);}finally{kernelMs+=performance.now()-start;kernelCalls++;}};
  const brain=new WasmBrain(core,model);
  const oldReader=createWingMotorEventReader({indices:wingIds,params:model.params,dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,eventContract:brain.eventContract});
  const fastReader=createWingMotorEventReader({indices:wingIds,params:model.params,dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,eventContract:brain.eventContract});
  const readWing=fast=>brain.readState(wingIds,{includeSpikeTime:true,...(fast?{includeStatistics:false,includeSpikeHistory:false}:{})});
  let stepMs=0,motorReadMs=0,legacyWingMs=0,selectedWingMs=0,checksumMs=0,totalSpikes=0,selectedEvents=0;
  const initialOld=readWing(false),initialFast=readWing(true);
  assert(Buffer.from(initialOld.buffer).equals(Buffer.from(initialFast.buffer)));
  assert.equal(packetDigest(oldReader.read(initialOld,0)),packetDigest(fastReader.read(initialFast,0)));
  try{
    for(let block=0;block<options.warmup+options.blocks;block++){
      const measured=block>=options.warmup;
      if(block===options.warmup){kernelMs=0;kernelCalls=0;}
      const input=patterns[Math.floor(block/5)%patterns.length],internal={hunger:.65,insulin:(block%3)*.1,akh:.4};
      let start=performance.now();brain.step(4,input,internal,true);if(measured)stepMs+=performance.now()-start;
      start=performance.now();const motors=brain.readState(motorIds);if(measured)motorReadMs+=performance.now()-start;
      totalSpikes=motors.totalSpikes;
      // Alternate readout order to avoid giving either path a consistent cache advantage.
      let legacy,selected;
      const oldRead=()=>{start=performance.now();legacy=readWing(false);if(measured)legacyWingMs+=performance.now()-start;};
      const fastRead=()=>{start=performance.now();selected=readWing(true);if(measured)selectedWingMs+=performance.now()-start;};
      if(block%2){fastRead();oldRead();}else{oldRead();fastRead();}
      assert(Buffer.from(legacy.buffer).equals(Buffer.from(selected.buffer)),'Selected motor values changed');
      const oldPacket=oldReader.read(legacy,brain.timeMs),fastPacket=fastReader.read(selected,brain.timeMs);
      assert.equal(packetDigest(oldPacket),packetDigest(fastPacket),'Motor event packet changed');selectedEvents+=oldPacket.events.length;
      start=performance.now();const digest=digestBrain(brain);checksumMs+=performance.now()-start;
      if(variant.name==='baseline'&&repetition===0)references.push(digest);else assert.equal(digest,references[block],`State differs at block ${block}, ${variant.name} repetition ${repetition}`);
    }
    const sample={variant:variant.name,repetition,simulatedMs:options.blocks*config.bodyBlockMs,stepMs,kernelMs,kernelCalls,motorReadMs,legacyWingMs,selectedWingMs,
      pathMs:stepMs+motorReadMs+(variant.name==='baseline'?legacyWingMs:selectedWingMs),checksumMs,totalSpikes,selectedEvents,finalStateSha256:references.at(-1)};
    samples.push(sample);console.log(JSON.stringify(sample));
  }finally{brain.dispose();}
}
for(let repetition=0;repetition<options.repeats;repetition++)for(const variant of repetition%2?[...variants].reverse():variants)await run(variant,repetition);
const median=values=>{const sorted=values.toSorted((a,b)=>a-b),mid=Math.floor(sorted.length/2);return sorted.length%2?sorted[mid]:(sorted[mid-1]+sorted[mid])/2;};
const medians=Object.fromEntries(variants.map(v=>[v.name,Object.fromEntries(['stepMs','kernelMs','motorReadMs','legacyWingMs','selectedWingMs','pathMs'].map(key=>[key,median(samples.filter(s=>s.variant===v.name).map(s=>s[key]))]))]));
const report={date:new Date().toISOString(),runtime:process.version,platform:process.platform,arch:process.arch,...options,
  scope:'Complete BANC v888 graph and ten DLM intrinsic cells, deterministic synthetic sensory currents. One isolated neural instance at a time. No body, browser scheduling, rendering, coordinator or training result writes. Timings exclude integrity checks.',
  correctness:'Exact full bytes at every 2 ms boundary for parameters, both neural state buffers, all delay history, receptor kinetics, DLM ionic state and complete event ring. Selected motor state and decoded motor events also match the legacy readout.',
  configHash:sha(configBytes),modelFingerprint:config.modelFingerprint,neurons:manifest.neuron_count,chemicalEdges:manifest.chemical_edges,fileHashes,
  wasmSha256:Object.fromEntries(variants.map(v=>[v.name,v.sha256])),inputSha256:patterns.map(input=>sha(new Uint8Array(input.buffer))),samples,medians,
  speedup:{neuralStep:medians.baseline.stepMs/medians.optimized.stepMs,kernel:medians.baseline.kernelMs/medians.optimized.kernelMs,
    wingReadout:medians.optimized.legacyWingMs/medians.optimized.selectedWingMs,neuralAndReadout:medians.baseline.pathMs/medians.optimized.pathMs}};
await fs.mkdir(path.dirname(path.resolve(outputFile)),{recursive:true});await fs.writeFile(outputFile,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({speedup:report.speedup,medians,output:outputFile}));
