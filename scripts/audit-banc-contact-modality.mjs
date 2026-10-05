// Isolated production-encoder assay. No whole-brain/body state, observer, or
// sensory preparation is changed. `before` captures the actual pre-fix code.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {createBancTasteMapper} from '../web/banc-taste.js';
import {createBancSensoryCurrentMapper,measureIsolatedSensoryRates} from '../web/banc-sensory-current.js';
import {createWasmCore} from '../packages/banc-runtime/src/wasm.js';

const mode=process.argv[2]??'after';assert(['before','before-position','after'].includes(mode));
const directory='data/prepared/banc888',readJson=async path=>JSON.parse(await fs.readFile(path,'utf8'));
const [manifest,sensory,groups,io,supplement,parameters,idBytes]=await Promise.all([
 readJson(directory+'/manifest.json'),readJson(directory+'/console/sensory-inputs.json'),readJson(directory+'/console/groups.json'),
 readJson(directory+'/io.json'),readJson('models/banc-taste-peg-annotations.json'),fs.readFile(directory+'/params.bin'),fs.readFile(directory+'/ids.bin')
]);
const params=new Float32Array(parameters.buffer.slice(parameters.byteOffset,parameters.byteOffset+parameters.byteLength));
const ids=new DataView(idBytes.buffer,idBytes.byteOffset,idBytes.byteLength);
const chemicalTokens=new Set(['sugar','low_salt','contact_pheromone']);
const activeChemical=sensory.body_transducers.filter(s=>s.kind==='touch'&&s.function.split(',').some(token=>chemicalTokens.has(token.trim())));
const exclusions=(sensory.body_transducer_exclusions??[]).filter(s=>s.function.split(',').some(token=>chemicalTokens.has(token.trim())));
const targets=[...activeChemical,...exclusions];
assert.equal(targets.length,10,'pinned BANC annotation set contains four sugar/salt and six pheromone cells');
assert.equal(new Set(targets.map(s=>s.index)).size,10,'excluded cells cannot retain active body transducers');
if(mode==='before')assert.equal(activeChemical.length,10,'before capture requires the original prepared collision mapping');
for(const sensor of exclusions)assert(!sensory.channels.some(channel=>channel.indices.includes(sensor.index)),'excluded cells must leave all aggregate body channels');
assert(targets.every(s=>!groups.sweet.includes(s.index)||s.function.split(',').some(token=>token.trim()==='sugar')),
 'only explicitly sugar-labelled cells may enter the separate contact taste group');
const chemicalSet=new Set(targets.map(s=>s.index));
const unassignedPosition=[...sensory.body_transducers.filter(s=>s.kind==='touch'&&s.function==='joint_angle'),
 ...(sensory.body_transducer_exclusions??[]).filter(s=>s.function==='joint_angle')];
assert.equal(unassignedPosition.length,1);assert.equal(unassignedPosition[0].index,42986);
assert.equal(ids.getBigUint64(42986*8,true).toString(),'720575941480808867');
const mechanical=sensory.body_transducers.filter(s=>s.kind==='touch'&&!chemicalSet.has(s.index)&&s.index!==42986);
assert.equal(mechanical.length,2965);
const supportedPosition=sensory.body_transducers.filter(s=>s.kind==='position');
assert.equal(supportedPosition.length,403,'properly mapped position receptors remain present');
const mapper=createBancTasteMapper([...io.sensory,...supplement.annotations],groups.sweet);
const feedback=()=>({legs:Array.from({length:6},()=>({loadBodyWeights:0,collision:1,tibiaAngle:0,tibiaVelocity:0,coxaAngle:0,vibration:0})),
 legFoodContact:new Uint8Array(6),mouthFoodContact:new Uint8Array(2),wingFoodContact:new Uint8Array(2),
 angularVelocity:[0,0,0],wingPowerLeft:0,wingPowerRight:0,halterePower:[0,0]});
const encoder=new SensoryEncoder(sensory,groups,{odor:()=>0,surface:()=>({y:0,contact:false})},{tasteMapper:mapper});
const pose={x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback:feedback()};
const encode=options=>{
 pose.bodyTime+=.002;const output=encoder.update(pose,null,{odor:false,vision:false,...options});
 return new Map(Array.from(output.indices,(index,k)=>[index,output.ratesHz[k]]));
};
const off=encode({taste:false}),on=encode({taste:true}),disconnected=encode({taste:true,bodySense:false});
const requested=(rates,index)=>rates.get(index)??0; // Excluded cells have no sensory-current slot.
for(const sensor of targets){assert.equal(requested(off,sensor.index),mode==='before'?60:0);assert.equal(requested(on,sensor.index),mode==='before'?60:0);assert.equal(requested(disconnected,sensor.index),0);}
assert.equal(requested(off,42986),mode==='after'?0:60);assert.equal(requested(on,42986),mode==='after'?0:60);
assert.equal(requested(disconnected,42986),0);
assert(mechanical.every(s=>off.get(s.index)===60&&on.get(s.index)===60),'ordinary mechanical bristles retain native collision drive');
assert(groups.sweet.every(index=>off.get(index)===0&&on.get(index)===0),'dry collision is not sugar contact');
assert(supportedPosition.every(s=>off.get(s.index)===5),'properly mapped position receptors retain neutral-angle input');
pose.feedback.legs.forEach(leg=>{leg.tibiaAngle=.3;leg.coxaAngle=.2;});const moved=encode({taste:false});
assert(supportedPosition.every(s=>moved.get(s.index)===10.5),'properly mapped position receptors retain their configured angle response');
assert.equal(requested(moved,42986),mode==='after'?0:60,'unassigned bristle is not given an invented position response');
pose.feedback.legs.forEach(leg=>{leg.tibiaAngle=0;leg.coxaAngle=0;});
pose.feedback.legFoodContact[0]=1;const food=encode({taste:true});
const expectedSugar=groups.sweet.filter(index=>mapper.rate(index,pose.feedback)>0);
assert.equal(expectedSugar.length,73);assert(expectedSugar.every(index=>food.get(index)===150));
assert(groups.sweet.filter(index=>!expectedSugar.includes(index)).every(index=>food.get(index)===0));
const core=await createWasmCore(),current=await createBancSensoryCurrentMapper(core,{manifest,params},{indices:[...targets.map(s=>s.index),42986]});
const report={date:new Date().toISOString(),mode,scope:'Production SensoryEncoder and calibrated production-WASM isolated cell; synthetic native-format dry/contact feedback only, not observed behavior or a whole-brain test',
 sourceSha256:Object.fromEntries(await Promise.all(['web/banc-ground-sense.js','web/sensory-encoder.js','web/banc-taste.js','scripts/prepare-banc-console.py'].map(async path=>[path,createHash('sha256').update(await fs.readFile(path)).digest('hex')]))),
 preparedChemicalTransducers:activeChemical.length,preparedChemicalExclusions:exclusions.length,
 unassignedPosition:{...unassignedPosition[0],rootId:ids.getBigUint64(42986*8,true).toString(),dryCollisionTasteOffHz:requested(off,42986),dryCollisionTasteOnHz:requested(on,42986),movedJointHz:requested(moved,42986),bodySenseOffHz:requested(disconnected,42986),dryCollisionAddedCurrentPa:current.current(42986,requested(off,42986)),profile:Array.from(params.subarray(42986*16,42986*16+16))},
 supportedPosition:{cells:supportedPosition.length,indices:supportedPosition.map(s=>s.index),neutralHz:5,movedJointHz:10.5},
 mappedTasteCoverage:mapper.coverage,mechanicalBristlesUnchanged:mechanical.length,sugarLeftFrontContactCells:expectedSugar.length,
 cells:targets.map(sensor=>({...sensor,rootId:ids.getBigUint64(sensor.index*8,true).toString(),
  dryCollisionTasteOffHz:requested(off,sensor.index),dryCollisionTasteOnHz:requested(on,sensor.index),bodySenseOffHz:requested(disconnected,sensor.index),
  dryCollisionAddedCurrentPa:current.current(sensor.index,requested(off,sensor.index)),profile:Array.from(params.subarray(sensor.index*16,sensor.index*16+16))})),
 isolatedDefaultProfileRateCheck:measureIsolatedSensoryRates(core,params.subarray(targets[0].index*16,targets[0].index*16+16),manifest.dt_ms,[0,current.current(targets[0].index,60),current.current(targets[0].index,150)]),
 passed:true};
const output=`reports/observation-60min-20260913/contact-modality-${mode}.json`;
await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({mode,passed:true,output,chemicalCells:targets.length,ordinaryMechanicalCells:mechanical.length,unassignedPositionHz:report.unassignedPosition.dryCollisionTasteOffHz,supportedPositionCells:supportedPosition.length,addedCurrentPa:report.cells[0].dryCollisionAddedCurrentPa,isolatedRates:report.isolatedDefaultProfileRateCheck}));
