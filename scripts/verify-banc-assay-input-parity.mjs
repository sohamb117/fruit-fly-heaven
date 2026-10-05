// Verify provenance-only exclusion of the formerly zero-driven conflicted
// sugar candidate does not change any input from the completed GPU assays.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {createBancTasteMapper} from '../web/banc-taste.js';
import {createBancSensoryCurrentMapper} from '../web/banc-sensory-current.js';
import {createWasmCore} from '../packages/banc-runtime/src/wasm.js';
const directory='reports/banc-sensory-recruitment',data='data/prepared/banc888',read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const [manifest,io,after,before,paramBytes,reports]=await Promise.all([read(data+'/manifest.json'),read(data+'/io.json'),
 Promise.all([read(data+'/console/groups.json'),read(data+'/console/sensory-inputs.json'),read('models/banc-taste-peg-annotations.json')]),
 Promise.all([read(directory+'/before-groups.json'),read(directory+'/before-sensory-inputs.json'),read(directory+'/before-taste-supplement.json')]),
 fs.readFile(data+'/params.bin'),Promise.all(['full-neural-assay.json','actual-posture-neural-assay.json'].map(async file=>({file,report:await read(directory+'/'+file)})))]);
const make=([groups,sensory,supplement])=>new SensoryEncoder(sensory,groups,{odor:()=>0,surface:()=>({y:0,contact:false})},
 {tasteMapper:createBancTasteMapper([...io.sensory,...supplement.annotations],groups.sweet)});
const core=await createWasmCore(),params=new Float32Array(paramBytes.buffer.slice(paramBytes.byteOffset,paramBytes.byteOffset+paramBytes.byteLength));
const mapper=await createBancSensoryCurrentMapper(core,{manifest,params},{indices:Uint32Array.from(new Set([...make(before).indices,...make(after).indices]))});
const matches=[];
for(const {file,report}of reports){
 assert(report.passed);assert.deepEqual(report.manifest,manifest,'physiology and connectivity manifest unchanged');
 for(const condition of report.results){
  const config=condition.version==='before'?before:after,encoder=make(config),sensory=config[1];
  const bodyKinds=new Map(sensory.body_transducers.map(row=>[row.index,row.kind]));
  for(const channel of sensory.channels)if(channel.key.startsWith('antenna_'))for(const index of channel.indices)bodyKinds.set(index,'antenna');
  const encoded=encoder.update({x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback:condition.feedback},null,
   {odor:false,vision:false,taste:condition.taste,bodySense:condition.body});
  const inputs=[];
  for(let k=0;k<encoded.indices.length;k++){
   const index=encoded.indices[k],kind=bodyKinds.get(index),requestedHz=encoded.ratesHz[k];
   const excluded=kind&&(condition.excludeBodyKind===kind||(condition.onlyBodyKind&&condition.onlyBodyKind!==kind));
   // The production worker/GPU harness writes the mapper output into a
   // Float32Array before neural dispatch; compare that actual stored current.
   const currentPa=Math.fround(mapper.current(index,excluded?0:requestedHz));
   if(currentPa!==0)inputs.push({index,requestedHz,currentPa});
  }
  assert.equal(inputs.length,condition.inputs.length,condition.name+': active input count');
  inputs.forEach((row,k)=>assert.deepEqual(row,condition.inputs[k],condition.name+': exact current at input '+k));
  matches.push({file,condition:condition.name,inputCount:inputs.length,exact:true});
 }
}
const paths=['web/sensory-encoder.js','web/banc-world-worker.js','models/banc-sensory-annotation-evidence.json',
 'models/banc-taste-peg-annotations.json',data+'/console/groups.json',data+'/console/sensory-inputs.json'];
const report={date:new Date().toISOString(),passed:true,scope:'No GPU or physical simulation; exact production-encoder added-current parity against every stored input-only trial after the final provenance exclusion.',
 removedIndex:19757,removedPreviousCurrentPa:0,conditions:matches,
 note:'The excluded candidate previously received zero added current in every case. Reported mean sugar-group Hz denominator changes from540 to539; neural input currents, graph and physiology do not change.',
 sourceSha256:Object.fromEntries(await Promise.all(paths.map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')])))};
await fs.writeFile(directory+'/final-input-parity.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,conditions:matches.length,scope:report.scope}));
