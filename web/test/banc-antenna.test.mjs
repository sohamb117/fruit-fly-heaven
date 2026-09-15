import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {ANTENNA_PRIOR,ANTENNA_ANNOTATION_SOURCE,createAntennaPopulation,createAntennaAirflowModel,validateAntennaConfig} from '../banc-antenna.js';

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
