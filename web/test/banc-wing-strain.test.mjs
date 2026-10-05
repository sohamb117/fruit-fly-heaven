import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {TEGULA_CELLS,createTegulaStrainPrior,createTegulaInputMapper,createTegulaSensoryManifest,validateTegulaConfig} from '../banc-tegula.js';
import {SensoryEncoder} from '../sensory-encoder.js';
const prior=createTegulaStrainPrior({halfLoadNative:.02});
const feedback=(time,left=[.02,0,0],right=[-.02,0,0])=>({wingLoad:{kind:'native-wing-aerodynamic-moment-v1',
 units:'g cm^2/s^2',localFrame:'native-thorax',momentThorax:{left,right},bodyTimeSeconds:time,
 forceTimeSeconds:time===0?0:time-.00005,localFrameTimeSeconds:time===0?0:time-.00005}});
const run=(input,config=prior)=>{const m=createTegulaInputMapper(config);m.rates(feedback(0,...input),true,0);return m.rates(feedback(.002,...input),true,.002);};

test('local deformation retains signed direction, axis and bilateral anatomical identities',()=>{
 const compressed=run([[.02,0,0],[-.02,0,0]]),reverse=run([[-.02,0,0],[.02,0,0]]),otherAxis=run([[0,.02,0],[0,-.02,0]]);
 assert(compressed.tegula_left>0);assert.equal(compressed.tegula_left,compressed.tegula_right);
 assert.equal(reverse.tegula_left,0);assert.equal(reverse.tegula_right,0);assert.equal(otherAxis.tegula_left,0);
 assert.deepEqual([...compressed.cellRates.keys()],TEGULA_CELLS.map(c=>c.index));
 assert(compressed.diagnostics.deformationRadians[0][0]>0);assert(compressed.diagnostics.deformationRadians[1][0]<0);
 assert(prior.fields.every(f=>f.status==='engineering-prior'));
});
test('receptive fields are per cell, explicitly unassignable and owned by the validated config',()=>{
 const config=structuredClone(prior),a=config.fields.find(f=>f.side==='left'),b=config.fields.find(f=>f.side==='left'&&f!==a);
 a.projection=null;a.status='unassigned';a.evidence='No registered direction';b.projection=[0,1,0];
 const result=run([[0,.02,0],[0,0,0]],config);
 assert.equal(result.cellRates.get(a.index),0);assert(result.cellRates.get(b.index)>0);assert.equal(result.diagnostics.unassigned,1);
 const owned=validateTegulaConfig(config);config.fields[0].projection=[0,0,1];assert.notDeepEqual(owned.fields[0],config.fields[0]);
 for(const mutate of [c=>c.fields[0].root_id='1',c=>c.fields[0].projection=[2,0,0],c=>c.fields[0].projection=null,
  c=>c.fields[0].status='anatomy',c=>c.fields.push(c.fields[0]),c=>c.relaxationSeconds=0,c=>c.complianceRadiansPerNativeMoment[0]=-1]){
  const bad=structuredClone(prior);mutate(bad);assert.throws(()=>validateTegulaConfig(bad),/Tegula/);
 }
});
test('causal held-load dynamics, repeat calls, reset and snapshots do not leak future or previous-episode load',()=>{
 const m=createTegulaInputMapper(prior);
 assert.equal(m.rates(feedback(0),true,0).tegula_left,0);
 const at2=m.rates(feedback(.002),true,.002),state=m.snapshot();
 assert.deepEqual(m.rates(feedback(.002),true,.002),at2);
 const at4=m.rates(feedback(.004,[0,0,0],[0,0,0]),true,.004);
 assert(at4.tegula_left>at2.tegula_left,'new zero load cannot retroactively change the prior interval');
 const copy=createTegulaInputMapper(prior);copy.restore(state);
 for(const mutate of [s=>s.held[0][0]+=1,s=>s.lastSource=null,s=>s.lastSource.identity='{}']){
  const bad=structuredClone(state);mutate(bad);const before=copy.snapshot();assert.throws(()=>copy.restore(bad),/snapshot/);assert.deepEqual(copy.snapshot(),before);
 }
 assert.deepEqual(copy.rates(feedback(.004,[0,0,0],[0,0,0]),true,.004),at4);
 const before=m.snapshot();assert.throws(()=>m.rates(feedback(.004),true,.004),/conflicting/);assert.deepEqual(m.snapshot(),before);
 assert.throws(()=>m.rates(feedback(.003),true,.003),/clock/);
 m.reset();assert.equal(m.rates(feedback(0,[0,0,0],[0,0,0]),true,0).tegula_left,0);
 assert.equal(m.rates(feedback(.002,[0,0,0],[0,0,0]),true,.002).tegula_left,0);
 assert.equal(m.rates(null,false,.004).tegula_left,0);
});
test('missing local frames, stale force caches and invalid vectors fail without silently borrowing magnitude',()=>{
 const m=createTegulaInputMapper(prior);m.rates(feedback(0),true,0);const before=m.snapshot();
 for(const change of [f=>delete f.wingLoad.momentThorax,f=>f.wingLoad.localFrame='world',f=>f.wingLoad.forceTimeSeconds=.001,
  f=>f.wingLoad.localFrameTimeSeconds=0,f=>f.wingLoad.momentThorax.left[0]=NaN]){
  const f=feedback(.002);change(f);assert.throws(()=>m.rates(f,true,.002),/Tegula/);assert.deepEqual(m.snapshot(),before);
 }
});
test('actual sensory manifest routes distinct cell fields without changing other input bytes',async()=>{
 const root=new URL('../../data/prepared/banc888/',import.meta.url),json=async file=>JSON.parse(await fs.readFile(new URL(file,root)));
 const [manifest,io,sensory,groups,bytes]=await Promise.all([json('manifest.json'),json('io.json'),json('console/sensory-inputs.json'),json('console/groups.json'),fs.readFile(new URL('ids.bin',root))]);
 const ids=new BigUint64Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
 const config=structuredClone(prior);config.fields[0].projection=null;config.fields[0].status='unassigned';
 const augmented=createTegulaSensoryManifest({manifest,io,ids},sensory,config),env={odor:()=>0};
 const encoder=new SensoryEncoder(augmented,groups,env),old=new SensoryEncoder(sensory,groups,env);
 const pose=t=>({x:0,y:3,z:0,heading:0,bodyTime:t,contact:false,feedback:feedback(t)}),flags={vision:false};
 encoder.update(pose(0),null,flags);old.update(pose(0),null,flags);
 const actual=encoder.update(pose(.002),null,flags),previous=old.update(pose(.002),null,flags);
 assert.deepEqual(actual.ratesHz.slice(0,previous.ratesHz.length),previous.ratesHz);
 const values=new Map(Array.from(actual.indices,(id,k)=>[id,actual.ratesHz[k]]));assert.equal(values.get(TEGULA_CELLS[0].index),0);
 assert(values.get(TEGULA_CELLS[1].index)>0);assert.equal(actual.sample.wingStrain.unassigned,1);
 assert(!Object.hasOwn(actual.sample.body.rates,'cellRates'));
});
