import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {LEG_PROPRIOCEPTION_PRIOR,LEG_PROPRIOCEPTION_ANNOTATION_SOURCE,
 createLegProprioceptionPopulation,createLegJointDescriptors,createLegProprioceptionMapper,
 validateLegProprioceptionConfig} from '../banc-leg-proprioception.js';

const read=async path=>JSON.parse(await readFile(new URL('../../'+path,import.meta.url)));
const [manifest,sensory,catalog,metadata,bytes]=await Promise.all([
 read('data/prepared/banc888/manifest.json'),read('data/prepared/banc888/console/sensory-inputs.json'),
 read('models/banc-leg-proprioception-v1.json'),read('models/flybody-mujoco.json'),
 readFile(new URL('../../data/prepared/banc888/ids.bin',import.meta.url))]);
const ids=new BigUint64Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),base={manifest,ids};
const population=await createLegProprioceptionPopulation(base,sensory,catalog),descriptors=createLegJointDescriptors(metadata);
const mapper=config=>createLegProprioceptionMapper(population,config,descriptors);
const feedback=(angle=0,velocity=0)=>({legs:Array.from({length:6},()=>({tibiaAngle:angle,tibiaVelocity:velocity}))});
const rates=sample=>new Map(Array.from(sample.indices,(index,k)=>[index,sample.ratesHz[k]]));
const selected=family=>population.selected.filter(c=>c.family===family);
const close=(a,b,tolerance=1e-6)=>assert(Math.abs(a-b)<tolerance,`${a} != ${b}`);

test('pinned raw annotations join all1066 exact identities without replacing another sensory route',()=>{
 assert.deepEqual(population.selection,{claw:160,hook:137,abstained:434,club:335});
 assert.equal(population.selected.length,632);assert.equal(population.abstained.length,434);
 assert.deepEqual(population.replacementIndices,sensory.channels.filter(c=>/^self_motion_(left|right)$/.test(c.key)).flatMap(c=>c.indices));
 assert(population.selected.every(c=>c.root_id===String(ids[c.index])&&c.anatomical_polarity===null));
 const unrelated=new Set(sensory.channels.filter(c=>!/^self_motion_(left|right)$/.test(c.key)).flatMap(c=>c.indices));
 assert(population.replacementIndices.every(i=>!unrelated.has(i)));
 assert.equal(population.source.rawMetadataSha256,LEG_PROPRIOCEPTION_ANNOTATION_SOURCE.rawMetadataSha256);
 assert(Object.isFrozen(population.selected[0]));assert(Object.isFrozen(population.replacementIndices));
 assert.throws(()=>{population.selected[0].leg=5;},TypeError);
 assert.throws(()=>createLegProprioceptionMapper(structuredClone(population),undefined,descriptors),/validated leg population/);
});

test('changed source bytes, root identities, routing, catalog and anatomical claims fail closed',async()=>{
 for(const change of [b=>b.ids[961]=1n,b=>b.manifest.materialization=889,b=>b.manifest.files['ids.bin'].sha256='0'.repeat(64)]){
  const b=structuredClone(base);change(b);await assert.rejects(createLegProprioceptionPopulation(b,sensory,catalog),/Leg proprioception/);
 }
 for(const change of [s=>s.channels.find(c=>c.key==='self_motion_left').indices.reverse(),
  s=>s.channels.find(c=>c.key==='self_motion_right').indices.pop(),
  s=>s.channels.find(c=>c.key==='antenna_left').indices.push(961),
  s=>s.vision.receptors.push({index:961})]){
  const s=structuredClone(sensory);change(s);await assert.rejects(createLegProprioceptionPopulation(base,s,catalog),/Leg proprioception/);
 }
 for(const change of [c=>c.cells[0].root_id='1',c=>c.cells[0].anatomical_polarity=1,
  c=>c.cells[0].family='hook',c=>c.cells[0].side='right']){
  const c=structuredClone(catalog);change(c);await assert.rejects(createLegProprioceptionPopulation(base,sensory,c),/catalog hash/);
 }
});

test('native descriptor preserves leg ordering, signed radian reference and bounds',()=>{
 assert.deepEqual(descriptors.joints.map(j=>j.joint),['tibia_T1_left','tibia_T2_left','tibia_T3_left','tibia_T1_right','tibia_T2_right','tibia_T3_right']);
 assert(descriptors.joints.every(j=>j.referenceAngleRadians===0&&j.rangeRadians[0]===-1.35&&j.rangeRadians[1]===1.3));
 assert.equal(descriptors.angleUnit,'radian');assert.equal(descriptors.velocityUnit,'radian/second');
 for(const change of [m=>m.joints=m.joints.filter(j=>j.name!=='tibia_T1_left'),
  m=>m.joints.push({...m.joints.find(j=>j.name==='tibia_T1_left')}),
  m=>m.joints.find(j=>j.name==='tibia_T1_left').range=[-90,90],
  m=>m.joints.find(j=>j.name==='tibia_T1_left').neutral=NaN,
  m=>m.joints.find(j=>j.name==='tibia_T1_left').qpos=m.joints.find(j=>j.name==='tibia_T2_left').qpos]){
  const m=structuredClone(metadata);change(m);assert.throws(()=>createLegJointDescriptors(m),/Leg proprioception/);
 }
 assert.throws(()=>createLegProprioceptionMapper(population,undefined,structuredClone(descriptors)),/validated native joint/);
});

test('claw position and hook direction retain sign around tonic baseline; club motion is bidirectional',()=>{
 const model=mapper(),zero=rates(model.sample(feedback())),plus=rates(model.sample(feedback(.5,5))),minus=rates(model.sample(feedback(-.5,-5)));
 for(const cell of selected('claw')){close(zero.get(cell.index),20);close(plus.get(cell.index),26);close(minus.get(cell.index),14);}
 for(const cell of selected('hook')){close(zero.get(cell.index),20);close(plus.get(cell.index),25);close(minus.get(cell.index),15);}
 for(const cell of selected('club')){close(zero.get(cell.index),5);close(plus.get(cell.index),10);close(minus.get(cell.index),10);}
 assert(model.metadata.cells.filter(c=>c.family!=='club').every(c=>c.configuredSign===1&&c.anatomical_polarity===null));
 // Within-family cells are not randomly divided into invented anatomical subtypes.
 assert.equal(new Set(model.metadata.cells.filter(c=>c.family==='hook').map(c=>c.configuredSign)).size,1);
 for(const sample of [model.sample(feedback(.7,1000)),model.sample(feedback(-.7,-1000))]){
  assert(Array.from(sample.ratesHz).every(x=>Number.isFinite(x)&&x>=0&&x<=100));assert(sample.diagnostics.rateClips>0);
 }
});

test('receptor family depends only on its stated joint signal and never broad body state',()=>{
 const model=mapper(),zero=rates(model.sample(feedback())),moving=rates(model.sample(feedback(0,5))),angled=rates(model.sample(feedback(.5,0)));
 for(const cell of selected('claw'))assert.equal(moving.get(cell.index),zero.get(cell.index));
 for(const cell of [...selected('hook'),...selected('club')])assert.equal(angled.get(cell.index),zero.get(cell.index));
 const poisoned=feedback();
 for(const key of ['speed','tilt','yaw','support','angularVelocity','height','verticalSpeed'])Object.defineProperty(poisoned,key,{get(){throw new Error('forbidden '+key);}});
 for(const leg of poisoned.legs)for(const key of ['coxaAngle','angle','speed','loadBodyWeights','collision','vibration','support'])Object.defineProperty(leg,key,{get(){throw new Error('forbidden '+key);}});
 assert.deepEqual(model.sample(poisoned),model.sample(feedback()));
});

test('one joint moves only its anatomically assigned leg, while uncertain families always abstain',()=>{
 const model=mapper(),baseRates=rates(model.sample(feedback())),f=feedback();f.legs[0]={tibiaAngle:.5,tibiaVelocity:5};
 const changed=rates(model.sample(f));
 for(const cell of population.selected)assert.equal(changed.get(cell.index)!==baseRates.get(cell.index),cell.leg===0);
 const extreme=rates(model.sample(feedback(2,1000)));
 assert(population.abstained.every(c=>extreme.get(c.index)===0));
 assert.equal(population.abstained.filter(c=>c.cell_class==='hair_plate_neuron').length,262);
 assert.equal(population.abstained.filter(c=>c.cell_class==='campaniform_sensillum_neuron').length,61);
 const high=model.sample(feedback(2,0)),atLimit=model.sample(feedback(1.3,0));
 assert.deepEqual(high.ratesHz,atLimit.ratesHz);assert.equal(high.diagnostics.angleRangeClips,6);
});

test('root-verified sign and reference priors are explicit; crossed axes, legs and invented measured labels fail',()=>{
 const cell=selected('claw')[0],joint=descriptors.joints[cell.leg],override={index:cell.index,root_id:cell.root_id,joint:joint.joint,axis:joint.axis,sign:-1};
 const config={cellOverrides:[override]},model=mapper(config),baseRates=rates(mapper().sample(feedback(.5,0))),changed=rates(model.sample(feedback(.5,0)));
 assert.equal(baseRates.get(cell.index),26);assert.equal(changed.get(cell.index),14);
 for(const c of population.selected)if(c.index!==cell.index)assert.equal(changed.get(c.index),baseRates.get(c.index));
 const reference=mapper({cellOverrides:[{...override,referenceAngleRadians:.5}]}).sample(feedback(.5));
 close(rates(reference).get(cell.index),20);
 override.sign=1;assert.equal(rates(model.sample(feedback(.5))).get(cell.index),14);
 for(const change of [o=>o.axis='coxa_roll',o=>o.sign=0,o=>o.sign=NaN,o=>o.root_id='1',
  o=>o.joint=joint.side==='left'?'tibia_T1_right':'tibia_T1_left',o=>o.referenceAngleRadians=3,
  o=>o.measured=true,o=>o.index=population.abstained[0].index]){
  const o={...override};change(o);assert.throws(()=>mapper({cellOverrides:[o]}),/Leg proprioception/);
 }
 const hook=selected('hook')[0],hookJoint=descriptors.joints[hook.leg];
 assert.throws(()=>mapper({cellOverrides:[{index:hook.index,root_id:hook.root_id,joint:hookJoint.joint,axis:hookJoint.axis,sign:1,referenceAngleRadians:0}]}),/reference angle requires claw/);
 assert.throws(()=>mapper({cellOverrides:[override,override]}),/duplicate cell override/);
});

test('require-measured polarity mode abstains on all unknown claw/hook signs, retaining unsigned club motion',()=>{
 const cell=selected('hook')[0],joint=descriptors.joints[cell.leg];
 const model=mapper({polarityMode:'require-measured',cellOverrides:[{index:cell.index,root_id:cell.root_id,joint:joint.joint,axis:joint.axis,sign:-1}]});
 const sample=model.sample(feedback(.5,5)),byIndex=rates(sample);
 assert.equal(sample.diagnostics.polarityAbstentions,297);assert.equal(sample.diagnostics.drivenCells,335);
 assert([...selected('claw'),...selected('hook'),...population.abstained].every(c=>byIndex.get(c.index)===0));
 assert(selected('club').every(c=>byIndex.get(c.index)===10));
 assert.throws(()=>validateLegProprioceptionConfig({polarityMode:'measured'}),/unsupported polarity mode/);
});

test('disabled sampling is exact pass-through; stateless reads and owned outputs cannot affect future samples',()=>{
 const model=mapper(),f=feedback(.2,3),saved=structuredClone(f),expected=model.sample(f);
 assert.equal(model.sample(null,{enabled:false}),null);assert.equal(model.sample(undefined,{enabled:false}),null);
 for(let k=0;k<4;k++)assert.deepEqual(model.sample(f),expected);
 assert.deepEqual(f,saved);
 const owned=model.sample(f);owned.indices.fill(0);owned.ratesHz.fill(999);owned.diagnostics.familyMeanRateHz.claw=999;
 assert.deepEqual(model.sample(f),expected);assert.equal(model.sample(f).diagnostics.stateless,true);
 assert.equal(typeof model.advance,'undefined');assert.equal(typeof model.reset,'undefined');
});

test('missing signed fields and invalid units/config fail instead of silently substituting zero',()=>{
 const model=mapper();
 for(const f of [null,{legs:[]},feedback(NaN),feedback(0,Infinity),feedback(90),feedback(0,1e7)])assert.throws(()=>model.sample(f),/Leg proprioception/);
 for(const key of ['tibiaAngle','tibiaVelocity']){const f=feedback();delete f.legs[2][key];assert.throws(()=>model.sample(f),/native leg 2/);}
 for(const config of [{unexpected:true},{maxRateHz:201},{baselineRateHz:101},{velocityGainHzPerRadPerSecond:-1},{cellOverrides:null}])assert.throws(()=>mapper(config),/Leg proprioception/);
 assert.throws(()=>model.sample(feedback(),{enabled:1}),/enabled must be boolean/);
 assert(Object.isFrozen(LEG_PROPRIOCEPTION_PRIOR));
});
