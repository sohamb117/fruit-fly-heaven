import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import loadMujoco from '../../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {ANTENNA_PRIOR,ANTENNA_ANNOTATION_SOURCE,createAntennaPopulation,createAntennaAirflowModel,validateAntennaConfig,
 ANTENNA_FAMILY_PRIOR,ANTENNA_FAMILY_PROFILE,createNativeAntennaKinematics} from '../banc-antenna.js';

const read=async path=>JSON.parse(await readFile(new URL('../../data/prepared/banc888/'+path,import.meta.url)));
const [manifest,io,sensory,bytes]=await Promise.all([read('manifest.json'),read('io.json'),read('console/sensory-inputs.json'),readFile(new URL('../../data/prepared/banc888/ids.bin',import.meta.url))]);
const ids=new BigUint64Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),base={manifest,io,ids};
const population=await createAntennaPopulation(base,sensory);
const observer=config=>createAntennaAirflowModel(population,config);
const flow=(t,v=[0,0,0],wind=[0,0,0])=>({bodyTimeSeconds:t,velocityRootCmPerSecond:v,windRootCmPerSecond:wind});
function drive(model,velocity,wind=[0,0,0],duration=.1,dt=.002){model.advance(flow(0,velocity,wind));for(let k=1;k<=Math.round(duration/dt);k++)model.advance(flow(k*dt,velocity,wind));return model;}
const close=(a,b,tolerance=1e-11)=>assert(Math.abs(a-b)<tolerance,`${a} vs ${b}`);

test('joins exact BANC identities and replaces only the579 existing antenna inputs',()=>{
 assert.deepEqual(population.selection,{left:239,right:323,driven:562,zeroed:17});
 assert.deepEqual(population.replacementIndices,sensory.channels.filter(c=>/^antenna_(left|right)$/.test(c.key)).flatMap(c=>c.indices));
 assert(population.selected.every(c=>c.root_id===String(ids[c.index])&&/^JO-[CDEF]/.test(c.cell_type)));
 assert.equal(population.selected.filter(c=>c.function==='direction').length,47);
 assert.equal(population.selected.filter(c=>c.function==='position').length,515);
 assert.equal(population.abstained.filter(c=>c.cell_type==='JO-mz').length,13);
 assert(population.abstained.every(c=>['JO','JO-mz'].includes(c.cell_type)));
 const sample=observer().sample(),rates=new Map(Array.from(sample.indices,(id,k)=>[id,sample.ratesHz[k]]));
 assert(population.abstained.every(c=>rates.get(c.index)===0));
 assert(population.selected.every(c=>rates.get(c.index)===ANTENNA_PRIOR.baselineRateHz));
 assert(Object.isFrozen(population.selected[0]));
});

test('source/root/side/family changes and routing overlap fail before constructing a sensor',async()=>{
 for(const change of [b=>b.ids[population.selected[0].index]=1n,b=>b.io.sensory.find(c=>c.index===population.selected[0].index).side='right',b=>b.manifest.files['io.json'].sha256='0'.repeat(64)]){
  const changed=structuredClone(base);change(changed);await assert.rejects(createAntennaPopulation(changed,sensory),/Antenna airflow/);
 }
 for(const change of [s=>s.channels.find(c=>c.key==='antenna_left').indices[0]=population.replacementIndices[1],
  s=>s.channels[0].indices.push(population.replacementIndices[0]),s=>s.body_transducer_exclusions.push({index:population.replacementIndices[0]}),
  s=>s.channels.find(c=>c.key==='antenna_left').indices.pop()]){
  const changed=structuredClone(sensory);change(changed);await assert.rejects(createAntennaPopulation(base,changed),/Antenna airflow/);
 }
 assert.equal(population.source.rawMetadataSha256,ANTENNA_ANNOTATION_SOURCE.rawMetadataSha256);
});

test('signed airflow produces opposing deflections and distinguishable population patterns',()=>{
 for(const axis of [[100,0,0],[0,100,0],[0,0,100]]){
  const a=drive(observer(),axis),b=drive(observer(),axis.map(x=>-x));
  a.snapshot().anglesRadians.flat().forEach((x,i)=>close(x,-b.snapshot().anglesRadians.flat()[i]));
  assert(a.sample().ratesHz.some((rate,k)=>Math.abs(rate-b.sample().ratesHz[k])>1));
  for(const model of [a,b])assert(Array.from(model.sample().ratesHz).every(x=>Number.isFinite(x)&&x>=0&&x<=100));
 }
 // Equal root velocity and wind imply zero relative airflow, even at high speed.
 const a=drive(observer(),[100,-25,40],[100,-25,40]),b=drive(observer(),[0,0,0]);
 assert.deepEqual(a.sample(),b.sample());
 const c=drive(observer(),[100,40,-10],[30,10,5]),d=drive(observer(),[70,30,-15]);
 assert.deepEqual(c.sample(),d.sample());
});

test('new measured airflow affects only the following interval, with no same-time drift',()=>{
 const a=observer();a.advance(flow(0));a.advance(flow(.002,[100,0,0]));
 assert(a.snapshot().anglesRadians.flat().every(x=>x===0));
 const before=a.snapshot();for(let k=0;k<10;k++)a.advance(flow(.002,[100,0,0]));assert.deepEqual(a.snapshot(),before);
 a.advance(flow(.004,[100,0,0]));assert(a.snapshot().anglesRadians.flat().some(x=>Math.abs(x)>0));
});

test('exact passive dynamics are timestep-consistent and dissipate without airflow',()=>{
 const a=drive(observer(),[80,40,20],[0,0,0],.1,.002),b=drive(observer(),[80,40,20],[0,0,0],.1,.001);
 a.snapshot().anglesRadians.flat().forEach((x,i)=>close(x,b.snapshot().anglesRadians.flat()[i]));
 a.snapshot().velocitiesRadiansPerSecond.flat().forEach((x,i)=>close(x,b.snapshot().velocitiesRadiansPerSecond.flat()[i]));
 a.advance(flow(.1));
 const energy=()=>{const s=a.snapshot(),w=1/ANTENNA_PRIOR.timeConstantSeconds;return s.anglesRadians.flat().reduce((sum,x,i)=>sum+w*w*x*x+s.velocitiesRadiansPerSecond.flat()[i]**2,0);};
 let last=energy();for(let k=1;k<=100;k++){a.advance(flow(.1+k*.002));const next=energy();assert(next<=last+1e-10);last=next;}
 assert(a.snapshot().anglesRadians.flat().every(x=>Math.abs(x)<1e-4));
});

test('snapshot branching is exact; preview reads and disabled output cannot mutate state',()=>{
 const a=drive(observer(),[80,20,40]),saved=a.snapshot(),b=observer();b.restore(saved);
 for(let k=0;k<100;k++){a.sample();a.snapshot();assert.equal(a.sample({enabled:false}),null);}
 assert.deepEqual(a.snapshot(),saved);
 for(let k=1;k<=20;k++){const input=flow(.1+k*.002,[20,30,40]);a.advance(input);b.advance(input);assert.deepEqual(a.sample(),b.sample());}
 const owned=a.sample();owned.ratesHz.fill(999);owned.indices.fill(0);owned.state.anglesRadians[0][0]=999;
 assert.deepEqual(a.sample(),b.sample());
 assert.notStrictEqual(a.sample().state.anglesRadians[0],a.sample().state.anglesRadians[0]);
 a.reset();assert.deepEqual(a.sample(),observer().sample());
});

test('invalid clocks, units/vectors and snapshots are rejected transactionally',()=>{
 const a=drive(observer(),[20,30,40]),before=a.snapshot();
 for(const input of [flow(.09),flow(.151),{...flow(.102),velocityRootCmPerSecond:[NaN,0,0]},
  {...flow(.102),windRootCmPerSecond:undefined},flow(.102,[1e9,0,0]),{...flow(.102),bodyTimeSeconds:Infinity}]){
  assert.throws(()=>a.advance(input),/Antenna airflow/);assert.deepEqual(a.snapshot(),before);
 }
 for(const change of [s=>s.config.orientationSeed++,s=>s.anglesRadians[0][0]=NaN,s=>s.heldAirflowRootCmPerSecond=[1e9,0,0],s=>s.populationSource='wrong']){
  const bad=structuredClone(before);change(bad);assert.throws(()=>a.restore(bad),/Antenna airflow/);assert.deepEqual(a.snapshot(),before);
 }
 for(const config of [{foo:1},{timeConstantSeconds:0},{maxRateHz:201},{orientationSeed:-1},{baselineRateHz:101}])assert.throws(()=>validateAntennaConfig(config),/Antenna airflow/);
});

test('unknown directional tuning is reproducible, seed-controlled and never anatomical',()=>{
 const a=observer(),b=observer(),c=observer({orientationSeed:999});
 assert.deepEqual(a.metadata,b.metadata);assert.notDeepEqual(a.metadata.cells,c.metadata.cells);
 assert.equal(a.metadata.orientationAssignment.anatomicalKnowledge,false);
 assert(a.metadata.cells.every(cell=>Math.abs(Math.hypot(...cell.preferredDirection)-1)<1e-12));
 assert.equal(a.metadata.cells.length,562);assert(Object.isFrozen(a.metadata.cells[0]));
 assert.equal(a.sample().diagnostics.nativeActuatorsChanged,false);
});

test('the original v1 requested-rate bytes remain reproducible after adding the opt-in profile',()=>{
 const model=drive(observer(),[80,20,40]),rates=model.sample().ratesHz;
 assert.equal(createHash('sha256').update(Buffer.from(rates.buffer,rates.byteOffset,rates.byteLength)).digest('hex'),
  '3a5206cc9efa0931f1968cfa0980f2c0158a98496771a21500364e3fbe1b300b');
});

let nativeAssets;
async function nativeFixture(){
 nativeAssets??=Promise.all([loadMujoco(),readFile(new URL('../../models/flybody-mujoco.xml',import.meta.url),'utf8'),
  readFile(new URL('../../models/flybody-mujoco.json',import.meta.url),'utf8').then(JSON.parse)]);
 const [mj,xml,metadata]=await nativeAssets,model=mj.MjModel.from_xml_string(xml),data=new mj.MjData(model);
 data.qpos.set([.3,-.2,1.5,Math.SQRT1_2,0,0,Math.SQRT1_2]);
 const kinematics=createNativeAntennaKinematics({mj,model,metadata});
 return {mj,model,data,metadata,kinematics,geometry:kinematics.geometry,dispose(){data.delete();model.delete();}};
}
const matrixVector=(m,v)=>[0,1,2].map(i=>m[i*3]*v[0]+m[i*3+1]*v[1]+m[i*3+2]*v[2]);
const familyInput=(geometry,t,strengths=[0,0])=>({schema:2,profile:ANTENNA_FAMILY_PROFILE,geometryKey:geometry.key,bodyTimeSeconds:t,
 sides:geometry.sides.map((g,s)=>({side:g.side,airflowLocalCmPerSecond:g.anteriorTangentLocal.map(x=>x*strengths[s])}))});
function driveFamily(model,geometry,strengths,duration=.1,dt=.002){
 model.advance(familyInput(geometry,0,strengths));for(let k=1;k<=Math.round(duration/dt);k++)model.advance(familyInput(geometry,k*dt,strengths));return model;
}
function familyRate(model,family,side='left'){
 const sample=model.sample(),rates=new Map(Array.from(sample.indices,(id,k)=>[id,sample.ratesHz[k]])),selected=model.metadata.cells.filter(c=>c.family===family&&c.side===side);
 return selected.reduce((sum,c)=>sum+rates.get(c.index),0)/selected.length;
}

test('native fixed-frame geometry agrees with MuJoCo and includes rotational velocity at the receiver',async()=>{
 const f=await nativeFixture();try{
  f.data.qvel.set([4,5,6,1,2,3]);f.mj.mj_forward(f.model,f.data);
  const nativeBefore={qpos:Array.from(f.data.qpos),qvel:Array.from(f.data.qvel),ctrl:Array.from(f.data.ctrl)},
    sample=f.kinematics.sample(f.data,{bodyTimeSeconds:0,windWorldCmPerSecond:[0,0,0]}),rootMatrix=Array.from(f.data.xmat.slice(f.geometry.rootBody*9,f.geometry.rootBody*9+9));
  for(let s=0;s<2;s++){
   const g=f.geometry.sides[s],actual=sample.sides[s];
   actual.attachmentPositionWorldCm.forEach((x,k)=>close(x,f.data.xpos[g.body*3+k]));
   actual.receiverPositionWorldCm.forEach((x,k)=>close(x,f.data.geom_xpos[g.geom*3+k]));
   const localFlowRoot=matrixVector(g.rotationRoot,actual.airflowLocalCmPerSecond);
   localFlowRoot.forEach((x,k)=>close(x,-actual.receiverVelocityRootCmPerSecond[k]));
   assert.notDeepEqual(actual.attachmentVelocityRootCmPerSecond,actual.receiverVelocityRootCmPerSecond);
  }
  assert.deepEqual({qpos:Array.from(f.data.qpos),qvel:Array.from(f.data.qvel),ctrl:Array.from(f.data.ctrl)},nativeBefore);
  // Central finite differences use actual native integration of the free
  // joint. This checks angular-coordinate convention and omega-cross-r sign.
  const positions=[];for(const sign of [-1,1]){
   f.data.qpos.set(nativeBefore.qpos);f.mj.mj_integratePos(f.model,f.data.qpos,f.data.qvel,sign*1e-6);f.mj.mj_forward(f.model,f.data);
   positions.push(f.geometry.sides.map(g=>Array.from(f.data.geom_xpos.slice(g.geom*3,g.geom*3+3))));
  }
  for(let s=0;s<2;s++){
   const expected=matrixVector(rootMatrix,sample.sides[s].receiverVelocityRootCmPerSecond);
   expected.forEach((v,k)=>close(v,(positions[1][s][k]-positions[0][s][k])/(2e-6),1e-8));
  }
  // Deliberately leave native xpos stale: sampling current qpos still moves.
  const previous=sample.sides[0].receiverPositionWorldCm;f.data.qpos.set(nativeBefore.qpos);f.data.qpos[0]+=1;
  const moved=f.kinematics.sample(f.data,{windWorldCmPerSecond:[0,0,0]});close(moved.sides[0].receiverPositionWorldCm[0]-previous[0],1);
 }finally{f.dispose();}
});

test('native flow is frame-covariant and rejects unsupported moving joints or mismatched physical wind',async()=>{
 const f=await nativeFixture();try{
  f.data.qpos.set([0,0,0,1,0,0,0]);f.data.qvel.set([12,-7,4,3,1,-2]);
  const a=f.kinematics.sample(f.data,{windWorldCmPerSecond:[0,0,0]});
  // Rotate the whole physical scenario +90 degrees about world Z.
  f.data.qpos.set([0,0,0,Math.SQRT1_2,0,0,Math.SQRT1_2]);f.data.qvel.set([7,12,4,3,1,-2]);
  const b=f.kinematics.sample(f.data,{windWorldCmPerSecond:[0,0,0]});
  a.sides.forEach((side,s)=>side.airflowLocalCmPerSecond.forEach((x,k)=>close(x,b.sides[s].airflowLocalCmPerSecond[k])));
  assert.throws(()=>f.kinematics.sample(f.data,{windWorldCmPerSecond:[1,0,0]}),/physical wind/);
  assert.throws(()=>f.kinematics.sample(f.data,{bodyTimeSeconds:1,windWorldCmPerSecond:[0,0,0]}),/time/);
  assert.throws(()=>createNativeAntennaKinematics({...f,metadata:{...f.metadata,joints:[...f.metadata.joints,{name:'antenna_left'}]}}),/frozen/);
  const head=f.geometry.headBody,previous=f.model.body_jntnum[head];f.model.body_jntnum[head]=1;
  assert.throws(()=>createNativeAntennaKinematics(f),/moving head/);f.model.body_jntnum[head]=previous;
 }finally{f.dispose();}
});

test('family v2 uses opposing C/E tuning, separates D dynamics, and abstains on unresolved F cells',async()=>{
 const f=await nativeFixture();try{
  const make=()=>createAntennaAirflowModel(population,ANTENNA_FAMILY_PRIOR,f.geometry),a=driveFamily(make(),f.geometry,[100,100]),b=driveFamily(make(),f.geometry,[-100,-100]);
  assert.deepEqual(a.metadata.selection,{driven:388,zeroed:191,left:191,right:197});
  assert(a.metadata.cells.every(c=>!Object.hasOwn(c,'preferredDirection')));assert(!Object.hasOwn(a.metadata.config,'orientationSeed'));
  for(const side of ['left','right']){
   assert(familyRate(a,'C',side)>ANTENNA_FAMILY_PRIOR.baselineRateHz);assert(familyRate(a,'E',side)<ANTENNA_FAMILY_PRIOR.baselineRateHz);
   assert(familyRate(b,'C',side)<ANTENNA_FAMILY_PRIOR.baselineRateHz);assert(familyRate(b,'E',side)>ANTENNA_FAMILY_PRIOR.baselineRateHz);
  }
  for(const model of [a,b]){
   const sample=model.sample(),rates=new Map(Array.from(sample.indices,(id,k)=>[id,sample.ratesHz[k]]));
   assert(model.metadata.abstained.every(c=>rates.get(c.index)===0));assert(model.metadata.abstained.filter(c=>/^JO-F/.test(c.cell_type)).length===174);
   assert(sample.ratesHz.every(x=>Number.isFinite(x)&&x>=0&&x<=100));
  }
  const onset=make();onset.advance(familyInput(f.geometry,0,[100,100]));onset.advance(familyInput(f.geometry,.002,[100,100]));
  assert(familyRate(onset,'D')>familyRate(onset,'C'),'D must retain a separate phasic term');
  const noMotion=onset.snapshot();noMotion.velocitiesRadiansPerSecond=[0,0];onset.restore(noMotion);
  close(familyRate(onset,'D'),familyRate(onset,'C'));
 }finally{f.dispose();}
});

test('family dynamics retain causal timing, cadence independence, owned snapshots and exact replay',async()=>{
 const f=await nativeFixture();try{
  const make=()=>createAntennaAirflowModel(population,ANTENNA_FAMILY_PRIOR,f.geometry),a=make();
  a.advance(familyInput(f.geometry,.002,[100,-100]));assert.deepEqual(a.snapshot().anglesRadians,[0,0]);
  const initial=a.snapshot();for(let k=0;k<10;k++){a.sample();a.snapshot();a.advance(familyInput(f.geometry,.002,[100,-100]));}assert.deepEqual(a.snapshot(),initial);
  a.advance(familyInput(f.geometry,.004,[100,-100]));assert(a.snapshot().anglesRadians[0]>0);assert(a.snapshot().anglesRadians[1]<0);
  const coarse=driveFamily(make(),f.geometry,[80,-50],.1,.002),fine=driveFamily(make(),f.geometry,[80,-50],.1,.001);
  coarse.snapshot().anglesRadians.forEach((x,s)=>close(x,fine.snapshot().anglesRadians[s]));
  coarse.snapshot().velocitiesRadiansPerSecond.forEach((x,s)=>close(x,fine.snapshot().velocitiesRadiansPerSecond[s]));
  const saved=coarse.snapshot(),branch=make();branch.restore(saved);
  for(let k=1;k<=20;k++){const input=familyInput(f.geometry,.1+k*.002,[30,-20]);coarse.advance(input);branch.advance(input);assert.deepEqual(coarse.sample(),branch.sample());}
  const untouched=coarse.snapshot(),owned=coarse.sample();owned.ratesHz.fill(999);owned.state.anglesRadians[0]=999;assert.deepEqual(coarse.snapshot(),untouched);
  assert.equal(coarse.sample({enabled:false}),null);assert.deepEqual(coarse.snapshot(),untouched);
  coarse.reset();assert.deepEqual(coarse.sample(),make().sample());
 }finally{f.dispose();}
});

test('family invalid samples, geometry, configuration and snapshots fail without mutating state',async()=>{
 const f=await nativeFixture();try{
  const a=driveFamily(createAntennaAirflowModel(population,ANTENNA_FAMILY_PRIOR,f.geometry),f.geometry,[50,50]),before=a.snapshot();
  for(const mutate of [x=>x.bodyTimeSeconds=.09,x=>x.bodyTimeSeconds=.151,x=>x.geometryKey+='x',x=>x.sides[0].airflowLocalCmPerSecond[0]=NaN,x=>x.sides[1].side='left']){
   const invalid=familyInput(f.geometry,.102,[20,20]);mutate(invalid);assert.throws(()=>a.advance(invalid),/Antenna airflow/);assert.deepEqual(a.snapshot(),before);
  }
  for(const mutate of [x=>x.geometryKey+='x',x=>x.config.baselineRateHz++,x=>x.anglesRadians[0]=NaN,x=>x.heldAirflowLocalCmPerSecond[1]=[1e9,0,0]]){
   const invalid=structuredClone(before);mutate(invalid);assert.throws(()=>a.restore(invalid),/Antenna airflow/);assert.deepEqual(a.snapshot(),before);
  }
  assert.throws(()=>a.restore(observer().snapshot()),/snapshot identity/);
  assert.throws(()=>createAntennaAirflowModel(population,ANTENNA_FAMILY_PRIOR),/geometry/);
  assert.throws(()=>validateAntennaConfig({...ANTENNA_FAMILY_PRIOR,orientationSeed:888}),/unknown/);
  const geometry=structuredClone(f.geometry);geometry.sides[0].receiverRootCm[0]+=.01;
  assert.throws(()=>createAntennaAirflowModel(population,ANTENNA_FAMILY_PRIOR,geometry),/identity/);
 }finally{f.dispose();}
});
