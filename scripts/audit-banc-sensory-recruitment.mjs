// Bounded production-encoder / isolated configured-cell assay. This does not
// run a full brain, body, browser or controller, or claim behavioral success.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {createBancTasteMapper} from '../web/banc-taste.js';
import {createBancSensoryCurrentMapper,measureIsolatedSensoryRates} from '../web/banc-sensory-current.js';
import {createWasmCore} from '../packages/banc-runtime/src/wasm.js';

const dir='reports/banc-sensory-recruitment',data='data/prepared/banc888';
const json=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const [raw,oldSensory,oldGroups,oldSupplement,sensory,groups,supplement,io,manifest,paramBytes,idBytes]=await Promise.all([
 json(dir+'/raw-annotations.json'),json(dir+'/before-sensory-inputs.json'),json(dir+'/before-groups.json'),json(dir+'/before-taste-supplement.json'),
 json(data+'/console/sensory-inputs.json'),json(data+'/console/groups.json'),json('models/banc-taste-peg-annotations.json'),json(data+'/io.json'),
 json(data+'/manifest.json'),fs.readFile(data+'/params.bin'),fs.readFile(data+'/ids.bin')]);
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.equal(supplement.prepared_ids_sha256,sha(idBytes));
assert.equal(supplement.prepared_io_sha256,sha(await fs.readFile(data+'/io.json')));
assert.equal(supplement.source.sha256,sha(await fs.readFile(supplement.source.path)));
assert.equal(supplement.prepared_ids_sha256,oldSupplement.prepared_ids_sha256);
assert.equal(supplement.prepared_io_sha256,oldSupplement.prepared_io_sha256);
const ids=new DataView(idBytes.buffer,idBytes.byteOffset,idBytes.byteLength);
for(const row of supplement.annotations)assert.equal(ids.getBigUint64(row.index*8,true).toString(),row.root_id);
const addedSugar=groups.sweet.filter(i=>!oldGroups.sweet.includes(i));
assert.deepEqual(addedSugar,[7230,60605,63482,68890,69032,128708,156561]);
assert.equal(oldGroups.sweet.filter(i=>!groups.sweet.includes(i)).length,0);
assert.deepEqual(addedSugar,raw.missingSugar.filter(row=>row.index!==19757).map(row=>row.index));
assert.equal(supplement.annotations.length,16);
assert.equal(sensory.sugar_annotation_exclusions.length,1);assert.equal(sensory.sugar_annotation_exclusions[0].index,19757);
assert.match(sensory.sugar_annotation_exclusions[0].reason,/mechanosensory/);
const identityEvidence=await json(supplement.identity_evidence_file);
assert.equal(sha(await fs.readFile(supplement.identity_evidence_file)),supplement.identity_evidence_sha256);
for(const [path,evidence]of Object.entries(identityEvidence.sources))assert.equal(sha(await fs.readFile(path)),evidence.sha256);
assert.deepEqual(supplement.annotations.filter(row=>row.identity_evidence).map(row=>row.index),[68890,69032]);
assert(supplement.annotations.filter(row=>row.identity_evidence).every(row=>row.identity_evidence.supports_external_sugar));
const auditory=raw.antenna.filter(row=>['auditory_low_frequency','auditory_high_frequency'].includes(row.cell_function_detailed));
const auditoryIds=new Set(auditory.map(row=>row.index));assert.equal(auditory.length,360);
for(const row of auditory){
 assert.equal(ids.getBigUint64(row.index*8,true).toString(),row.root_id);
 assert(!sensory.channels.some(c=>c.indices.includes(row.index)));
 const exclusion=sensory.body_transducer_exclusions.find(c=>c.index===row.index);
 assert(exclusion);assert.equal(exclusion.function,row.cell_function_detailed);
 assert.equal(exclusion.cell_type,row.cell_type);assert.equal(exclusion.root_id,row.root_id);
}
const retained=raw.antenna.filter(row=>!auditoryIds.has(row.index));assert.equal(retained.length,579);
assert(retained.every(row=>sensory.channels.some(c=>c.indices.includes(row.index))));
const make=(s,g,a)=>{const taste=createBancTasteMapper([...io.sensory,...a.annotations],g.sweet);
 return {taste,encoder:new SensoryEncoder(s,g,{odor:()=>0,surface:()=>({y:0,contact:false})},{tasteMapper:taste})};};
const before=make(oldSensory,oldGroups,oldSupplement),after=make(sensory,groups,supplement);
assert.equal(before.taste.coverage.mapped,509);assert.equal(after.taste.coverage.mapped,516);
assert.equal(after.taste.coverage.unmapped.length,23);
assert(after.taste.coverage.unmapped.every(row=>row.reason==='missing or unsupported laterality'));
const feedback={legs:Array.from({length:6},()=>({loadBodyWeights:0,collision:1,tibiaAngle:.3,coxaAngle:.2,tibiaVelocity:0,vibration:0})),
 antennae:[{angle:.4,speed:0},{angle:-.3,speed:0}],speed:0,tilt:.7,angularVelocity:[0,0,0],wingPowerLeft:0,wingPowerRight:0,halterePower:[0,0],
 legFoodContact:Array(6).fill(0),mouthFoodContact:[0,0],wingFoodContact:[0,0]};
let time=0;
const encode=(instance,f,extra={})=>{
 const out=instance.encoder.update({x:0,y:0,z:0,heading:0,bodyTime:time+=.002,contact:false,feedback:f},null,{odor:false,vision:false,...extra});
 return new Map(Array.from(out.indices,(index,k)=>[index,out.ratesHz[k]]));
};
const pre=encode(before,feedback),post=encode(after,feedback);
for(const row of auditory){assert.equal(pre.get(row.index),row.side==='left'?21:17.5);assert.equal(post.get(row.index)??0,0);}
for(const row of retained)assert.equal(post.get(row.index),pre.get(row.index));
for(const [index,rate]of pre)if(!auditoryIds.has(index))assert.equal(post.get(index),rate,'every previous nonauditory input unchanged for identical feedback');
assert(groups.sweet.every(index=>post.get(index)===0),'dry mechanical collision is not sugar');
// A stale channel list with the new explicit exclusions still cannot fall
// back to the generic antenna drive. It retains all cells at zero added Hz.
const guarded=make({...sensory,channels:oldSensory.channels},groups,supplement);
const guardedRates=encode(guarded,feedback);
assert(auditory.every(row=>guardedRates.get(row.index)===0));
const organs=[];
for(const [field,slots]of [['legFoodContact',6],['mouthFoodContact',2],['wingFoodContact',2]])for(let slot=0;slot<slots;slot++){
 const f=structuredClone(feedback);f[field][slot]=1;
 const old=encode(before,f),now=encode(after,f),active=groups.sweet.filter(index=>now.get(index)>0);
 for(const index of oldGroups.sweet)assert.equal(now.get(index),old.get(index),'all previous sugar inputs retain exact organ/side rates');
 for(const index of active)assert.equal(now.get(index),150);
 const extra=addedSugar.filter(index=>now.get(index)>0);
 for(const index of extra){
  const annotation=raw.missingSugar.find(row=>row.index===index);
  const expectedField=annotation.body_part_sensory==='labellum'?'mouthFoodContact':'legFoodContact';
  const expectedSlot=expectedField==='mouthFoodContact'?(annotation.side==='left'?0:1):['front_leg','middle_leg','hind_leg'].indexOf(annotation.body_part_sensory)+(annotation.side==='right'?3:0);
  assert.equal(field,expectedField);assert.equal(slot,expectedSlot);assert(annotation.side);
 }
 const disconnected=encode(after,f,{taste:false});assert(groups.sweet.every(index=>disconnected.get(index)===0));
 organs.push({field,slot,previousCells:oldGroups.sweet.filter(index=>old.get(index)>0).length,currentCells:active.length,addedIndices:extra});
}
assert.equal(new Set(organs.flatMap(row=>row.addedIndices)).size,7);
assert(!organs.some(row=>row.addedIndices.includes(19757)),'null side is not inferred from nerve text');
const params=new Float32Array(paramBytes.buffer.slice(paramBytes.byteOffset,paramBytes.byteOffset+paramBytes.byteLength));
const core=await createWasmCore(),current=await createBancSensoryCurrentMapper(core,{manifest,params},{indices:[...auditoryIds,...addedSugar]});
const referenceIndex=auditory[0].index,profile=params.subarray(referenceIndex*16,referenceIndex*16+16);
const currents=[0,17.5,21,150].map(hz=>({requestedHz:hz,currentPa:current.current(referenceIndex,hz)}));
const measured=measureIsolatedSensoryRates(core,profile,manifest.dt_ms,currents.map(row=>row.currentPa));
const sourcePaths=['scripts/prepare-banc-console.py','web/sensory-encoder.js','web/banc-world-worker.js','web/banc-taste.js','web/banc-ground-sense.js',
 'models/banc-taste-peg-annotations.json',data+'/console/groups.json',data+'/console/sensory-inputs.json',data+'/params.bin',data+'/edges.bin',data+'/manifest.json'];
const report={date:new Date().toISOString(),passed:true,scope:'Production encoder and isolated production WASM profile only; no full-brain, body, flight, feeding, or behavioral validation.',
 sourceSha256:Object.fromEntries(await Promise.all(sourcePaths.map(async path=>[path,sha(await fs.readFile(path))]))),
 identitySha256:{ids:supplement.prepared_ids_sha256,io:supplement.prepared_io_sha256,raw:supplement.source.sha256},
 auditory:{excludedCells:auditory.length,retainedOtherAntennaCells:retained.length,staticDeflectionBeforeHz:{left:21,right:17.5},afterAddedHz:0,
  source:'https://doi.org/10.3389/fphys.2014.00179',note:'Fig2 experimentally separates A/B vibration responses from static deflection; D/mixed and C/E groups are retained. BANC frequency labels are preserved verbatim, no frequency tuning is assigned.',cells:auditory},
 sugar:{rawAddedCandidates:raw.missingSugar,routedAddedCells:7,excludedConflictedCells:sensory.sugar_annotation_exclusions,
  identityEvidence,beforeCoverage:before.taste.coverage,afterCoverage:after.taste.coverage,organContactChecks:organs},
 isolatedCell:{index:referenceIndex,profile:Array.from(profile),dtMs:manifest.dt_ms,currents,measured},
 unchangedOtherInputs:true,staleAuditoryChannelGuardPassed:true,connectivityAndPhysiologyUnchanged:true};
await fs.writeFile(dir+'/encoder-assay.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,scope:report.scope,auditoryExcluded:auditory.length,otherAntennaRetained:retained.length,
 rawAddedSugarCandidates:raw.missingSugar.length,addedRouted:7,addedExcluded:1,coverage:after.taste.coverage.mapped,isolatedCell:report.isolatedCell}));
