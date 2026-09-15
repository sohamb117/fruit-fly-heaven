import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createCoupledVirtualHalteres,validateCoupledHaltereModel} from '../virtual-haltere.js';
import {createHaltereCurrentMapper,validateHaltereCurrentProfile} from '../banc-haltere.js';
import {createHaltereCurrentMapper as createLegacy} from './fixtures/haltere-v1/banc-haltere.js';
const read=async path=>JSON.parse(await fs.readFile(new URL(path,import.meta.url)));
const identities=await read('./fixtures/haltere-identities-v888.json'),
  directional=await read('../../models/banc-haltere-directional-v2.json'),
  marginalized=await read('../../models/banc-haltere-marginalized-v2.json');
const preparedIds=new BigUint64Array(identities.neuronCount);
for(const cell of identities.cells)preparedIds[cell.index]=BigInt(cell.root_id);
const inputs={sensoryManifest:{schema_version:1,neuron_count:identities.neuronCount,
  body_transducers:identities.cells.map(({root_id,...cell})=>cell)},preparedIds,geometries:directional.geometries};
const mechanicalModel=directional.mechanicalModel,profile=marginalized.profile;
const makePopulationProfile=(_manifest,_ids,{directional:useDirectional=false}={})=>structuredClone(useDirectional?directional.profile:marginalized.profile);
const options={enabled:true,profile,...inputs,mechanicalModel,maxCurrentPa:800};
const create=extra=>createHaltereCurrentMapper({...options,...extra});
const model=extra=>createCoupledVirtualHalteres(inputs.geometries,{...mechanicalModel,...extra});
const sample=(k=0,{left=1,right=1,power=[0,0],omega=[0,0,0]}={})=>{
  const t=Math.floor(k/4)*.002,phase=2*Math.PI*236*t;
  return {bodyTimeSeconds:t,elapsedSeconds:k%4*.0005,wingPhaseRadians:phase,wingFrequencyHz:236,
    omegaRootRadS:omega,halterePower:power,wingPower:[left,right],
    wingMotion:Object.fromEntries(['left','right'].map((side,i)=>[side,{joint:`wing_roll_${side}`,source:'native-joint',frame:'native-joint-coordinate',
      angleRadians:mechanicalModel.sides[side].centerRadians+[left,right][i]*.6*Math.sin(phase),angularVelocityRadS:[left,right][i]*.6*2*Math.PI*236*Math.cos(phase)}]))};
};
const near=(a,b,eps=1e-8)=>assert(Math.abs(a-b)<=eps,`${a} != ${b}`);

test('v1 currents and diagnostics remain exactly equal to archived code',async()=>{
  const legacyProfile=await read('./fixtures/haltere-v1/orientation-prior.json');
  const old=createLegacy({...options,profile:legacyProfile}),now=create({...options,profile:legacyProfile,mechanicalModel:undefined});
  for(const halterePower of [[0,0],[0,.6],[.8,.5],undefined])for(const elapsedSeconds of [0,.0005,.001,.0015]){
    const s={halterePower,omegaRootRadS:[3,-5,7],wingPhaseRadians:.713,wingFrequencyHz:236,elapsedSeconds};
    const a=old.currents(s),b=now.currents(s);assert.deepEqual(a,b);assert.deepEqual(a.diagnostics,b.diagnostics);
  }
  assert.throws(()=>create({...options,profile:legacyProfile}),/version 2 sensory profile/);
});
test('actual 328-cell identity join, unknown fields and owned population tuning',()=>{
  assert.equal(profile.cells.length,328);assert.equal(profile.cells.filter(c=>c.side==='left').length,171);
  assert.equal(profile.cells.filter(c=>c.side==='right').length,157);assert.equal(validateHaltereCurrentProfile(profile,inputs).length,328);
  assert(profile.tuningPopulations.every(p=>p.field===null&&p.orientationRadians===null));
  for(const change of [p=>p.cells[0].root_id='1',p=>p.cells[0].side='right',p=>p.cells[0].cell_type='invented',
    p=>p.cells[0].orientationRadians=0,p=>p.tuningPopulations[0].field='dF2',p=>p.tuningPopulations[0].orientationRadians=0,
    p=>p.tuningPopulations[0].orientationStatus='measured']){
    const bad=structuredClone(profile);change(bad);assert.throws(()=>create({profile:bad}));
  }
  const p=structuredClone(profile),m=create({profile:p});p.cells[0].root_id='2';p.tuningPopulations[0].evidence[0].note='changed';
  assert.notEqual(m.cells[0].root_id,'2');assert(Object.isFrozen(m.cells[0].tuning));
});
test('zero own power can have ipsilateral modeled motion without changing reported muscle activity',()=>{
  const m=create();let result;
  for(let k=0;k<40;k++)result=m.currents(sample(k,{right:0}));
  assert.equal(result.diagnostics.sides.left.actualMusclePower,0);
  assert(result.diagnostics.sides.left.amplitudeRadians>0);
  assert(result.some((v,i)=>profile.cells[i].side==='left'&&v>0));
  assert(result.every((v,i)=>profile.cells[i].side!=='right'||v===0));
  assert.equal(result.diagnostics.sides.right.amplitudeRadians,0);
});
test('own drive and opposite-side changes are isolated; cold zero drive is exact zero',()=>{
  const a=model(),b=model(),zero=model();
  for(let k=0;k<32;k++){
    const x=a.sample(sample(k,{left:0,right:0,power:[.5,0]})),y=b.sample(sample(k,{left:0,right:1,power:[.5,.9]}));
    assert.deepEqual(x.sides.left,y.sides.left);
    const z=zero.sample(sample(k,{left:0,right:0}));
    for(const s of Object.values(z.sides)){assert.equal(s.theta,0);assert.equal(s.thetaDot,0);assert(s.totalMomentRoot.every(v=>v===0));}
  }
  assert(a.snapshot().states[0].angleRadians!==0);
});
test('signed root rotation reverses only the Coriolis term at matched motion',()=>{
  const a=model(),b=model(),z=model();let x,y,n;
  for(let k=0;k<20;k++){x=a.sample(sample(k,{omega:[0,5,0]}));y=b.sample(sample(k,{omega:[0,-5,0]}));n=z.sample(sample(k));}
  for(const side of ['left','right']){
    assert.deepEqual(x.sides[side].baselineMomentRoot,y.sides[side].baselineMomentRoot);
    x.sides[side].coriolisMomentRoot.forEach((v,i)=>near(v,-y.sides[side].coriolisMomentRoot[i],1e-12));
    x.sides[side].totalMomentRoot.forEach((v,i)=>near((v+y.sides[side].totalMomentRoot[i])/2,n.sides[side].totalMomentRoot[i],1e-12));
  }
});
test('actual motion determines coupling; a phase label alone cannot move silent sensors',()=>{
  const a=model(),b=model();
  for(let k=0;k<12;k++){
    const input=sample(k,{left:0,right:0}),other={...input,wingPhaseRadians:input.wingPhaseRadians+Math.PI};
    const x=a.sample(input),y=b.sample(other);
    for(const side of ['left','right']){assert.equal(x.sides[side].theta,0);assert.equal(y.sides[side].theta,0);}
  }
  const c=model(),d=model();let x,y;
  for(let k=0;k<20;k++){
    const input=sample(k),opposite=structuredClone(input);
    for(const [side,m]of Object.entries(opposite.wingMotion)){m.angleRadians=2*mechanicalModel.sides[side].centerRadians-m.angleRadians;m.angularVelocityRadS*=-1;}
    x=c.sample(input);y=d.sample(opposite);
  }
  for(const side of ['left','right']){near(x.sides[side].theta,-y.sides[side].theta);near(x.sides[side].thetaDot,-y.sides[side].thetaDot);}
});
test('0.5 ms causal schedule, duplicate idempotence, snapshot and reset preserve exact state',()=>{
  const a=create();const first=a.currents(sample(0));assert.deepEqual(a.currents(sample(0)),first);
  for(let k=1;k<13;k++)a.currents(sample(k));
  const snapshot=a.snapshot(),b=create();b.restore(snapshot);snapshot.motion.states[0].angleRadians=99;
  for(let k=13;k<24;k++){assert.deepEqual(a.currents(sample(k)),b.currents(sample(k)));assert.deepEqual(a.snapshot(),b.snapshot());}
  a.reset();const fresh=create();for(let k=0;k<8;k++)assert.deepEqual(a.currents(sample(k)),fresh.currents(sample(k)));
  const changed=create({maxCurrentPa:400});assert.throws(()=>changed.restore(b.snapshot()),/contract mismatch/);
});
test('new boundary samples cannot retroactively alter integrated angle and velocity',()=>{
  const a=model(),b=model();for(let k=0;k<4;k++){a.sample(sample(k));b.sample(sample(k));}
  const input=sample(4),changed=structuredClone(input);changed.wingMotion.left.angularVelocityRadS*=2;
  const x=a.sample(input).sides.left,y=b.sample(changed).sides.left;
  assert.equal(x.theta,y.theta);assert.equal(x.thetaDot,y.thetaDot);assert.notEqual(x.thetaDDot,y.thetaDDot);
});
test('invalid time, frame, side and power fail atomically before state/current writes',()=>{
  const m=create(),target=new Float32Array(m.neuronCount).fill(7);m.writeInto(target,sample(0));
  const before=target.slice(),state=m.snapshot();
  const changes=[s=>s.bodyTimeSeconds=.1,s=>s.elapsedSeconds=.00025,s=>s.halterePower=[null,0],s=>s.wingPower=[1],
    s=>s.wingMotion.left.joint='wing_roll_right',s=>s.wingMotion.right.frame='world',s=>s.wingMotion.right.angleRadians=NaN,
    s=>s.omegaRootRadS=[0,Infinity,0],s=>s.wingFrequencyHz=1001];
  for(const change of changes){const input=sample(1);change(input);assert.throws(()=>m.writeInto(target,input));assert.deepEqual(target,before);assert.deepEqual(m.snapshot(),state);}
  assert.throws(()=>m.currents({...sample(0),wingPhaseRadians:1}),/Conflicting repeated/);
  assert.throws(()=>create().currents(sample(1)),/time zero/);
  m.currents(sample(1));assert.throws(()=>m.currents(sample(0)),/advance exactly/);
});
test('unknown orientation has no hidden preferred direction; declared priors must carry evidence',()=>{
  const unknown=create(),p=structuredClone(profile);
  for(const group of p.tuningPopulations){group.orientationStatus='declared-prior';group.orientationRadians=.3;group.evidence.push({source:'fixture-only',note:'Unmeasured explicit test direction'});}
  const tuned=create({profile:p});let a,b;
  for(let k=0;k<20;k++){a=unknown.currents(sample(k,{omega:[0,5,0]}));b=tuned.currents(sample(k,{omega:[0,5,0]}));}
  assert.notDeepEqual(a,b);
  for(const side of ['left','right']){const values=Array.from(a).filter((_,i)=>profile.cells[i].side===side);assert(values.every(v=>v===values[0]));}
  const group=p.tuningPopulations[0];group.evidence=[];assert.throws(()=>create({profile:p}),/declared prior/);
});
test('root-frame rotation preserves projected currents and native mirror sign is explicit',()=>{
  const rot=([x,y,z])=>[-y,x,z],gs=inputs.geometries.map(g=>({...g,pivotRoot:rot(g.pivotRoot),neutralComFromPivotRoot:rot(g.neutralComFromPivotRoot),
    oscillationAxisRoot:rot(g.oscillationAxisRoot),beamTangentRoot:rot(g.beamTangentRoot),beamNormalsRoot:g.beamNormalsRoot.map(rot)}));
  const a=create(),b=create({geometries:gs});
  for(let k=0;k<20;k++){const input=sample(k,{omega:[3,-5,7]}),x=a.currents(input),y=b.currents({...input,omegaRootRadS:rot(input.omegaRootRadS)});assert.deepEqual(x,y);}
  const config=structuredClone(mechanicalModel);config.sides.right.nativeSign=0;assert.throws(()=>validateCoupledHaltereModel(config),/sign/);
  config.sides.right.nativeSign=1;config.sides.right.joint='wing_roll_left';assert.throws(()=>validateCoupledHaltereModel(config),/ipsilateral/);
});
test('directional population candidate retains signed phase-dependent pitch contrast with silent own muscles',()=>{
  const p=makePopulationProfile(inputs.sensoryManifest,inputs.preparedIds,{directional:true}),a=create({profile:p}),b=create({profile:p});
  assert(p.tuningPopulations.every(g=>g.orientationStatus==='declared-prior'&&g.orientationRadians===(g.side==='left'?0:Math.PI)));
  const contrast=[];
  for(let k=0;k<40;k++){
    const plus=a.currents(sample(k,{omega:[0,5,0]})),minus=b.currents(sample(k,{omega:[0,-5,0]}));
    contrast.push(Math.max(...plus.map((v,i)=>Math.abs(v-minus[i]))));
    for(const s of Object.values(plus.diagnostics.sides))assert.equal(s.actualMusclePower,0);
  }
  assert(Math.max(...contrast)>0);assert(new Set(contrast.map(x=>x.toFixed(6))).size>4);
});
