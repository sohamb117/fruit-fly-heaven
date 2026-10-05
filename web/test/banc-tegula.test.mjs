import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {TEGULA_CELLS,createTegulaSensoryManifest,createTegulaInputMapper,validateTegulaConfig} from '../banc-tegula.js';
import {SensoryEncoder,validateSensoryManifest} from '../sensory-encoder.js';

const json=async name=>JSON.parse(await readFile(new URL('../../data/prepared/banc888/'+name,import.meta.url)));
const [manifest,io,actualSensory,idsBytes]=await Promise.all([json('manifest.json'),json('io.json'),json('console/sensory-inputs.json'),readFile(new URL('../../data/prepared/banc888/ids.bin',import.meta.url))]);
const ids=new BigUint64Array(idsBytes.buffer.slice(idsBytes.byteOffset,idsBytes.byteOffset+idsBytes.byteLength));
const base={manifest,io,ids},config={schema:1,profile:'tegula-fluid-load-v1',maxRateHz:100,halfLoadNative:.02};
const fixture={schema_version:1,neuron_count:manifest.neuron_count,
 vision:{width:32,height:16,receptors:[{index:3,side:'left',u:0,v:0},{index:4,side:'right',u:0,v:0}]},
 channels:[{key:'self_motion_left',indices:[5,6]},{key:'antenna_right',indices:[7]}],
 body_transducers:[{index:5,kind:'load',side:'left',leg:0}],body_transducer_exclusions:[{index:6}]};
const groups={odor_left:[0],odor_right:[1],sweet:[2]},environment={odor:()=>0,surface:()=>({y:0,contact:false})};
const feedback=(left=.02,right=.06,time=0)=>({kind:'native-wing-aerodynamic-moment-v1',units:'g cm^2/s^2',left,right,bodyTimeSeconds:time,forceTimeSeconds:time});
const pose=(wingLoad=feedback())=>({x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,
 feedback:{speed:12,tilt:.3,antennae:[{angle:0,speed:0},{angle:.2,speed:1}],
  legs:Array.from({length:6},()=>({loadBodyWeights:.25,angle:.1,speed:2,support:1})),wingLoad}});
const flags={odor:false,taste:false,vision:false};
const mapRates=out=>new Map(Array.from(out.indices,(index,k)=>[index,out.ratesHz[k]]));

test('all 26 actual IO/index/root/type/side joins pass and augment an owned sensory clone',()=>{
 const before=structuredClone(actualSensory),beforeIO=structuredClone(io),beforeConfig=structuredClone(config);
 const result=createTegulaSensoryManifest(base,actualSensory,config);
 assert.doesNotThrow(()=>validateSensoryManifest(result,manifest.neuron_count));
 assert.deepEqual(actualSensory,before);assert.deepEqual(io,beforeIO);assert.deepEqual(config,beforeConfig);
 assert.deepEqual(result.channels.slice(0,-2),actualSensory.channels);
 assert.deepEqual(result.body_transducers.slice(0,-26),actualSensory.body_transducers);
 assert.deepEqual(result.channels.slice(-2).map(channel=>channel.indices.length),[14,12]);
 assert.equal(result.body_transducers.filter(row=>row.kind==='wing_strain').length,26);
 assert(TEGULA_CELLS.every(cell=>String(ids[cell.index])===cell.root_id));
 result.channels[0].indices[0]=-1;result.tegula_model.halfLoadNative=2;
 assert.deepEqual(actualSensory,before);assert.equal(config.halfLoadNative,.02);
});

test('strict identity validation rejects mismatched data, duplicates, overlaps and unsupported priors',()=>{
 for(const alter of [
  b=>b.ids[TEGULA_CELLS[0].index]=1n,
  b=>b.io.sensory.find(row=>row.index===TEGULA_CELLS[0].index).side='left',
  b=>b.io.sensory.find(row=>row.index===TEGULA_CELLS[0].index).body_part='wing_base',
  b=>b.io.sensory.find(row=>row.index===TEGULA_CELLS[0].index).cell_type='SNpp37',
  b=>b.io.sensory.push({...b.io.sensory.find(row=>row.index===TEGULA_CELLS[0].index)}),
  b=>delete b.ids,
 ]){const changed=structuredClone(base);alter(changed);assert.throws(()=>createTegulaSensoryManifest(changed,fixture,config),/Tegula/);}
 for(const alter of [
  s=>s.channels[0].indices.push(TEGULA_CELLS[0].index),
  s=>s.body_transducer_exclusions.push({index:TEGULA_CELLS[0].index}),
  s=>s.channels.push({key:'tegula_left',indices:[8]}),
 ]){const changed=structuredClone(fixture);alter(changed);assert.throws(()=>createTegulaSensoryManifest(base,changed,config),/overlap/);}
 for(const change of [{extra:1},{schema:2},{profile:'other'},{maxRateHz:101},{halfLoadNative:0},{halfLoadNative:-1},{halfLoadNative:Infinity}])
  assert.throws(()=>validateTegulaConfig({...config,...change}),/Tegula/);
});

test('finite per-side load gives monotone bounded rates without axis, phase or opposite-side borrowing',()=>{
 const mapper=createTegulaInputMapper(config),loads=[0,.002,.01,.02,.06,1,Number.MAX_VALUE];
 const rates=loads.map(load=>mapper.rates({wingLoad:feedback(load,.02)},true,0));
 assert.equal(rates[0].tegula_left,0);assert.equal(rates[3].tegula_left,50);assert(Math.abs(rates[4].tegula_left-75)<1e-12);
 assert(rates.every((rate,k)=>rate.tegula_left>=0&&rate.tegula_left<=100&&rate.tegula_right===50&&(k===0||rate.tegula_left>rates[k-1].tegula_left)));
 const input={...config},owned=createTegulaInputMapper(input);input.halfLoadNative=50;
 assert.equal(owned.rates({wingLoad:feedback()},true,0).tegula_left,50);
 assert.equal(mapper.rates({wingLoad:{...feedback(.02,.06,.002),forceTimeSeconds:.00195}},true,.002).tegula_left,50);
 for(const forceTimeSeconds of [0,.001,.002])assert.throws(()=>mapper.rates({wingLoad:{...feedback(.02,.06,.002),forceTimeSeconds}},true,.002),/force time/);
 assert.throws(()=>mapper.rates({wingLoad:{...feedback(),forceTimeSeconds:1e-10}},true,0),/force time/);
});

test('encoder sends the side-specific proxy to exactly the added cells and preserves other rate bytes',()=>{
 const augmented=createTegulaSensoryManifest(base,fixture,config);
 for(const native of [true,false])for(const bodySense of [true,false]){
  const current=pose();if(!native)current.feedback.legs=current.feedback.legs.map(({loadBodyWeights,...leg})=>leg);
  const old=new SensoryEncoder(fixture,groups,environment).update(current,null,{...flags,bodySense});
  const out=new SensoryEncoder(augmented,groups,environment).update(current,null,{...flags,bodySense});
  assert.deepEqual(new Uint8Array(out.ratesHz.buffer,0,old.ratesHz.byteLength),new Uint8Array(old.ratesHz.buffer));
  const rates=mapRates(out);
  for(const cell of TEGULA_CELLS)assert.equal(rates.get(cell.index),bodySense?(cell.side==='left'?50:75):0);
 }
 const old=new SensoryEncoder(fixture,groups,environment).update(pose(undefined),null,flags);
 assert.deepEqual(Array.from(old.ratesHz),[0,0,0,0,0,20,0,36]);
});

test('missing, invalid and future native feedback fail visibly; bodySense off needs no feedback',()=>{
 const augmented=createTegulaSensoryManifest(base,fixture,config);
 for(const bad of [undefined,{...feedback(),left:-1},{...feedback(),right:NaN},{...feedback(),kind:'other'},
  {...feedback(),forceTimeSeconds:.01},{...feedback(),bodyTimeSeconds:1}]){
  const encoder=new SensoryEncoder(augmented,groups,environment),current=pose();current.feedback.wingLoad=bad;
  assert.throws(()=>encoder.update(current,null,flags),/Tegula/);
  current.feedback.wingLoad=feedback();assert(encoder.update(current,null,flags),'failed samples must not poison the cached key');
 }
 const current=pose();delete current.feedback.wingLoad;
 const out=new SensoryEncoder(augmented,groups,environment).update(current,null,{...flags,bodySense:false});
 assert(TEGULA_CELLS.every(cell=>mapRates(out).get(cell.index)===0));
 const plain=new SensoryEncoder(fixture,groups,environment);assert.doesNotThrow(()=>plain.update(current,null,flags));
});

test('malformed augmented manifests cannot put strain sensors into the broad fallback',()=>{
 const augmented=createTegulaSensoryManifest(base,fixture,config);
 for(const alter of [
  s=>delete s.tegula_model,
  s=>s.body_transducers.find(row=>row.kind==='wing_strain').side='both',
  s=>s.body_transducers.find(row=>row.kind==='wing_strain').root_id='1',
  s=>s.channels.at(-1).key='self_motion_right',
  s=>s.channels.at(-1).indices.reverse(),
 ]){const changed=structuredClone(augmented);alter(changed);assert.throws(()=>new SensoryEncoder(changed,groups,environment),/Tegula/);}
});
