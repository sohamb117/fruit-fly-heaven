// Static native-state sensory probes. No BANC steps, model changes or flight
// simulation: only captured qpos/qvel, native forward caches, and real encoder.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {registerHooks} from 'node:module';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore} from '../packages/banc-runtime/src/wasm.js';
import {SensoryEncoder} from '../web/sensory-encoder.js';
import {createBancTasteMapper} from '../web/banc-taste.js';
import {bancBodyRate} from '../web/banc-ground-sense.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const output='reports/flight-sensory-observability';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const paths={capture:'reports/flight-live-variants/no-adhesion-power15-steering005.json',
 replay:'reports/steering-recruitment-calibration/legacy-replay/result.json',profiles:'reports/steering-recruitment-calibration/legacy-replay/muscle-profiles.json',
 sensory:'data/prepared/banc888/console/sensory-inputs.json',groups:'data/prepared/banc888/console/groups.json',taste:'models/banc-taste-peg-annotations.json',
 habitat:'web/habitat.json',params:'data/prepared/banc888/params.bin'};
const bytes=Object.fromEntries(await Promise.all(Object.entries(paths).map(async([key,file])=>[key,await fs.readFile(file)])));
const read=key=>JSON.parse(bytes[key]);
const original=read('capture'),capture=original.evaluation.motorReplay,replay=read('replay'),profiles=read('profiles');
assert.equal(replay.baselineGate.passed,true);assert.equal(profiles.baselineGate.passed,true);
const manifest=read('sensory'),groups=read('groups'),taste=read('taste');
const prefixes=[['/banc-engine/','packages/banc-runtime/'],['/body-engine/','packages/flybody-runtime/node_modules/@mujoco/mujoco/']];
registerHooks({resolve(specifier,context,next){const entry=prefixes.find(([prefix])=>specifier.startsWith(prefix));return entry?{url:pathToFileURL(path.join(root,entry[1]+specifier.slice(entry[0].length))).href,shortCircuit:true}:next(specifier,context);}});
const {FlyBodyWorld}=await import('../web/flybody-world.js');
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const fly={...structuredClone(read('habitat').flies[0]),id:1,brain:{time_ms:0,motorNeuronRates:Array(805).fill(0)}};
const world=new FlyBodyWorld(capture.scene.fruit,[fly],capture.scene.options,{mj,core,
 xml:original.virtualAssets['/body-model/flybody-mujoco.xml'],metadata:capture.scene.metadata,io:capture.scene.io});
const body=world.bodies.get(1),tasteMapper=createBancTasteMapper([...capture.scene.io.sensory,...taste.annotations],groups.sweet);
const decode=(text,bits)=>{const b=Buffer.from(text,'base64');return Array.from({length:b.length/(bits/8)},(_,i)=>bits===32?b.readFloatLE(i*4):b.readDoubleLE(i*8));};
const indexToChannel=new Map();for(const channel of manifest.channels)for(const index of channel.indices)indexToChannel.set(index,channel.key);
const transducers=new Map(manifest.body_transducers.map(row=>[row.index,row])),exclusions=new Set(manifest.body_transducer_exclusions.map(row=>row.index));
function stateAt(index,{angular,linear,tibiaVelocity}={}){
 const row=capture.steps[index],sample=replay.cases[0].samples[index],state=decode(profiles.profiles[(index+1)*2-1].state,32);
 body.data.qpos.set(decode(row.postQpos,64));body.data.qvel.set(decode(row.postQvel,64));body.data.time=row.timeAfter;
 if(angular)body.data.qvel.set(angular,3);if(linear)body.data.qvel.set(linear,0);
 if(tibiaVelocity!==undefined)body.data.qvel[body.byJoint.get('tibia_T1_left').dof]=tibiaVelocity;
 body.muscleState=Float32Array.from(state);body.wingPhase=sample.wing.phaseRad;
 [body.wingPowerLeft,body.wingPowerRight]=sample.wing.sides.rows.map(side=>side[5]);body.wingPower=(body.wingPowerLeft+body.wingPowerRight)/2;
 body.halterePower=['left','right'].map(side=>{const values=body.mappings.flatMap((m,k)=>m.kind==='asynchronous_haltere'&&m.joint.endsWith(side)?[state[k*3+2]]:[]);return values.reduce((a,b)=>a+b,0)/Math.max(1,values.length);});
 body.haltereSteering={left:{},right:{}};
 body.mappings.forEach((m,k)=>{if(m.kind==='haltere_steering_assumption')body.haltereSteering[m.joint.endsWith('left')?'left':'right'][m.target]=state[k*3+2];});
 mj.mj_forward(world.model,body.data);body.refresh();
 const pose=structuredClone(fly);pose.brain={time_ms:row.neuralMs,motorNeuronRates:decode(row.ratesHz,32)};
 // A static probe uses dt=0: no finite-difference visual-joint motion is
 // invented from the preceding probe. Explicit native joint velocities remain.
 world.copyPose(pose,body,0);return pose;
}
function encode(pose){const encoder=new SensoryEncoder(manifest,groups,world.habitat,{tasteMapper});const result=encoder.update(pose,null,{odor:true,taste:true,vision:false,bodySense:true});return {indices:Array.from(result.indices),rates:result.ratesHz.slice(),pose};}
function compare(a,b){assert.deepEqual(a.indices,b.indices);const changed=[];for(let k=0;k<a.rates.length;k++)if(!Object.is(a.rates[k],b.rates[k]))changed.push({index:a.indices[k],a:a.rates[k],b:b.rates[k],channel:indexToChannel.get(a.indices[k])||'food/vision'});return {identical:changed.length===0,changedCount:changed.length,maximumDifference:Math.max(0,...changed.map(row=>Math.abs(row.a-row.b))),changedByChannel:changed.reduce((counts,row)=>{counts[row.channel]=(counts[row.channel]||0)+1;return counts;},{}),examples:changed.slice(0,8)};}
const report={schemaVersion:1,kind:'static-native-sensory-observability',createdAt:new Date().toISOString(),
 scope:'Captured airborne pose/velocity and native muscle gates, evaluated through FlyBodyWorld.copyPose and the production SensoryEncoder. Native forward caches are rebuilt, but no neural or integration steps run after initialization. Probes do not simulate future contact/sensory changes.',
 sourceHashes:Object.fromEntries(Object.entries(bytes).map(([key,value])=>[paths[key],hash(value)])),
 codeHashes:Object.fromEntries(await Promise.all(['web/flybody-world.js','web/banc-ground-sense.js','web/sensory-encoder.js','web/banc-sensory-current.js','web/body-world.js','web/banc-taste.js',
  'scripts/audit-flight-sensory-observability.mjs'].map(async file=>[file,hash(await fs.readFile(file))]))),
 capturedConfigHash:capture.provenance.configHash,probeTimeSeconds:.2,comparisons:[],saturation:[]};
try{
 const baseline=encode(stateAt(99));assert.equal(baseline.pose.airborne,true);report.inputCount=baseline.indices.length;
 report.baselineFeedback=baseline.pose.feedback;
 for(const magnitude of [1,10,25,50])for(const [axis,name]of ['roll','pitch','yaw'].entries()){
  const plus=[0,0,0],minus=[0,0,0];plus[axis]=magnitude;minus[axis]=-magnitude;
  const result=compare(encode(stateAt(99,{angular:plus})),encode(stateAt(99,{angular:minus})));
  assert.equal(result.identical,true);report.comparisons.push({probe:`${name} velocity + versus - ${magnitude} rad/s`,...result});
 }
 for(const [a,b,label]of [[[10,0,0],[0,10,0],'roll versus pitch magnitude10'],[[10,0,0],[0,0,10],'roll versus yaw magnitude10'],[[1,2,3],[-1,-2,-3],'mixed-axis full sign reversal']])report.comparisons.push({probe:label,...compare(encode(stateAt(99,{angular:a})),encode(stateAt(99,{angular:b})))});
 for(const [axis,name]of ['x','y','z'].entries()){
  const plus=[0,0,0],minus=[0,0,0];plus[axis]=2;minus[axis]=-2;
  const result=compare(encode(stateAt(99,{linear:plus})),encode(stateAt(99,{linear:minus})));assert(result.identical);
  report.comparisons.push({probe:`world linear ${name} velocity + versus - 2 cm/s`,...result});
 }
 report.comparisons.push({probe:'T1-left tibia velocity + versus - 10 rad/s',...compare(encode(stateAt(99,{tibiaVelocity:10})),encode(stateAt(99,{tibiaVelocity:-10})))});
 for(const [label,edit]of [
  ['wing phase',pose=>{pose.feedback.wingPhase+=1;}],['wing frequency',pose=>{pose.feedback.wingFrequency*=1.2;}],
  ['native signed wing joint angles',pose=>{pose.joints.wings=pose.joints.wings.map(row=>row.map(value=>-value));}],
  ['signed reported pitch/bank',pose=>{pose.pitch=-pose.pitch;pose.bank=-pose.bank;}],
  ['reported altitude',pose=>{pose.altitude+=10;}],
 ]){const changedPose=structuredClone(baseline.pose);edit(changedPose);const result=compare(baseline,encode(changedPose));assert(result.identical);report.comparisons.push({probe:label+' field only (downstream dependency probe)',...result});}
 const single=stateAt(99,{angular:[10,0,0]});single.feedback.halterePower=[0,0];single.feedback.haltereSteering={left:{},right:{}};
 single.feedback.wingPowerLeft=1;single.feedback.wingPowerRight=0;const left=encode(single);
 const other=structuredClone(single);other.feedback.wingPowerLeft=0;other.feedback.wingPowerRight=1;
 report.comparisons.push({probe:'left-only versus right-only wing gate, same rotation',...compare(left,encode(other))});
 for(const time of [.10,.15,.20,.25,.28]){
  const encoded=encode(stateAt(Math.round(time/.002)-1)),byId=new Map(encoded.indices.map((id,k)=>[id,encoded.rates[k]]));
  const values=manifest.body_transducers.filter(s=>s.kind==='rotation').map(s=>({organ:s.organ,side:s.side,rate:byId.get(s.index)}));
  report.saturation.push({time,angularSpeed:Math.hypot(...encoded.pose.feedback.angularVelocity),halterePower:encoded.pose.feedback.halterePower,
   wingPower:[encoded.pose.feedback.wingPowerLeft,encoded.pose.feedback.wingPowerRight],
   groups:Object.fromEntries(['haltere/left','haltere/right','wing_base/left','wing_base/right'].map(key=>{const rows=values.filter(v=>v.organ+'/'+v.side===key);return [key,{count:rows.length,rate:rows[0].rate,saturated:rows.filter(row=>row.rate===100).length}];}))});
 }
 report.recordedRotationWindow={fromSeconds:.1,toSeconds:.28,samples:91,groups:{}};
 for(let index=49;index<140;index++){
  const feedback=stateAt(index).feedback;
  for(const organ of ['haltere','wing_base'])for(const side of ['left','right']){
   const key=organ+'/'+side,group=report.recordedRotationWindow.groups[key]??={samples:0,clippedSamples:0,float32CeilingSamples:0,zeroAsyncPowerSamples:0,minimumHz:Infinity,maximumHz:0};
   const rate=bancBodyRate({kind:'rotation',organ,side},feedback);
   group.samples++;group.clippedSamples+=rate===100;group.float32CeilingSamples+=Math.fround(rate)===100;
   group.zeroAsyncPowerSamples+=organ==='haltere'&&feedback.halterePower[side==='left'?0:1]===0;
   group.minimumHz=Math.min(group.minimumHz,rate);group.maximumHz=Math.max(group.maximumHz,rate);
  }
 }
 report.rotationMagnitudeCeiling={formula:'(100 - 5*activity)/(4*power) rad/s when power>0; activity=max(power, haltere steering) for haltere and power for wing-base.',
  fullPowerThresholdRadPerSecond:23.75,zeroPower:'No rotation-dependent term; only the haltere steering-gated basal term can remain.'};
 const currentProfiles={};for(const id of baseline.indices){const p=Array.from({length:16},(_,k)=>bytes.params.readFloatLE((id*16+k)*4));const key=JSON.stringify({graded:p[8]>.5,refractoryMs:p[5]});currentProfiles[key]=(currentProfiles[key]||0)+1;}
 report.currentConversion={profiles:currentProfiles,proof:'For a fixed neuron index the production current mapper is a deterministic function of requested rate. Every byte-identical rate-vector comparison therefore has identical added currents, regardless of differing cell physiology. No isolated-cell calibration or BANC steps were run.',
  bodyLimits:'All 5059 body-channel cells are spiking with2ms refractory. The mapper caps its profile at200Hz; body encoders already cap at100Hz. Positive requested rates below2Hz map to the2Hz calibration floor; zero maps to zero added current.'};
 report.channels=manifest.channels.map(channel=>({key:channel.key,count:channel.indices.length,transducers:channel.indices.filter(id=>transducers.has(id)).length,
  fallback:channel.indices.filter(id=>!transducers.has(id)&&!exclusions.has(id)).length}));
 report.rotationGroups=manifest.body_transducers.filter(s=>s.kind==='rotation').reduce((groups,s)=>{const key=s.organ+'/'+s.side;const g=groups[key]??={count:0,cellTypes:new Set(),fields:new Set(),functions:new Set()};g.count++;g.cellTypes.add(s.cell_type);Object.keys(s).forEach(key=>g.fields.add(key));g.functions.add(s.function);return groups;},{});
 for(const group of Object.values(report.rotationGroups)){group.cellTypes=[...group.cellTypes].sort();group.cellTypeCount=group.cellTypes.length;group.fields=[...group.fields].sort();group.functions=[...group.functions].sort();}
}finally{world.dispose();}
await fs.mkdir(output,{recursive:true});await fs.writeFile(path.join(output,'result.json'),JSON.stringify(report)+'\n');
const rows=report.comparisons.map(row=>`| ${row.probe} | ${row.changedCount} | ${row.maximumDifference.toFixed(3)} |`);
await fs.writeFile(path.join(output,'README.md'),`# Flight sensory observability\n\nThe full ${report.inputCount}-entry encoder produces identical requested-rate vectors for every tested positive/negative native angular-velocity pair at fixed recorded airborne posture. The production current mapper cannot recover a distinction absent from those rates. This is a static native-state capability test, not a neural or flight simulation.\n\n| Probe | Changed inputs | Maximum rate difference (Hz) |\n|---|---:|---:|\n${rows.join('\n')}\n\nDetailed native feedback, organ/side gates, saturation measurements, anatomical grouping fields and source hashes are in result.json. Angular sign is discarded; organ/side identity is retained. Magnitude saturation can additionally erase changes in rotation strength. Future motion, contacts or odor sampling can still change sensory input and are not tested by static sign pairs.\n`);
console.log(JSON.stringify({output,inputCount:report.inputCount,comparisons:report.comparisons,saturation:report.saturation,rotationGroups:report.rotationGroups,currentConversion:report.currentConversion}));
