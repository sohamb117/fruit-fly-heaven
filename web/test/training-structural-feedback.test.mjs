// Integration-only sensory tests: actual prepared BANC identities and native
// model geometry, with prescribed kinematic samples. No neural/physics steps.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import loadMujoco from '../../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createTrainingSensoryResources} from '../training/sensory-feedback.js';
import {SensoryEncoder} from '../sensory-encoder.js';
import {LEG_PROPRIOCEPTION_PRIOR} from '../banc-leg-proprioception.js';
import {ANTENNA_FAMILY_PRIOR} from '../banc-antenna.js';
import {createTegulaStrainPrior,createTegulaSensoryManifest,createTegulaInputMapper,TEGULA_CELLS} from '../banc-tegula.js';

const json=async path=>JSON.parse(await readFile(new URL('../../'+path,import.meta.url)));
const [manifest,io,sensory,groups,legCatalog,metadata,idBytes]=await Promise.all([
 json('data/prepared/banc888/manifest.json'),json('data/prepared/banc888/io.json'),
 json('data/prepared/banc888/console/sensory-inputs.json'),json('data/prepared/banc888/console/groups.json'),
 json('models/banc-leg-proprioception-v1.json'),json('models/flybody-mujoco.json'),
 readFile(new URL('../../data/prepared/banc888/ids.bin',import.meta.url))]);
const ids=new BigUint64Array(idBytes.buffer.slice(idBytes.byteOffset,idBytes.byteOffset+idBytes.byteLength));
const base={manifest,io},options={base,sensory,groups,ids,legCatalog},flags={vision:false};
const legIds=new Set(sensory.channels.filter(c=>/^self_motion_/.test(c.key)).flatMap(c=>c.indices));
const antennaIds=new Set(sensory.channels.filter(c=>/^antenna_/.test(c.key)).flatMap(c=>c.indices));
const wingIds=new Set(TEGULA_CELLS.map(c=>c.index));
const defaultConfig={bodyBlockMs:2,vision:false};
const antennaFeedback={schema:1,windWorldCmPerSecond:[0,0,0],mechanics:ANTENNA_FAMILY_PRIOR};
const wingPrior=createTegulaStrainPrior({halfLoadNative:.02});
const wingManifest=createTegulaSensoryManifest({...base,ids},sensory,wingPrior);
const rates=encoded=>new Map(Array.from(encoded.indices,(index,k)=>[index,encoded.ratesHz[k]]));
const close=(a,b,tol=1e-7)=>assert(Math.abs(a-b)<tol,`${a} != ${b}`);
const snapshot=system=>({antenna:system.antenna?.snapshot(),wing:system.encoder.tegula?.snapshot()});

function feedback(time){
 const forceTime=time===0?0:time-.00005;
 return {speed:7,tilt:.4,yaw:-.3,touch:[.25,.5],impact:2,
  antennae:[{angle:.3,speed:2},{angle:-.2,speed:1}],
  legs:Array.from({length:6},(_,k)=>({tibiaAngle:(k-2.5)*.1,tibiaVelocity:(k-2.5)*2,
   coxaAngle:.3,angle:.4,speed:4,support:.2,loadBodyWeights:.05,collision:1,vibration:6})),
  wingLoad:{kind:'native-wing-aerodynamic-moment-v1',units:'g cm^2/s^2',localFrame:'native-thorax',
   momentThorax:{left:[.02,0,0],right:[-.02,0,0]},bodyTimeSeconds:time,forceTimeSeconds:forceTime,localFrameTimeSeconds:forceTime}};
}
function fixture(){
 const data={time:0,qpos:Float64Array.from([0,0,3,1,0,0,0]),qvel:new Float64Array(6)},
  body={get time(){return data.time;},data},
  fly={id:1,bodyTime:0,x:0,y:3,z:0,heading:0,contact:false,feedback:feedback(0)},
  world={metadata,model:{opt:{wind:[0,0,0]}},habitat:{odor:()=>.2,ceiling:100,fruit:[]}};
 return {body,fly,world,setTime(t){data.time=t;fly.bodyTime=t;fly.feedback=feedback(t);},dispose(){}};
}
let nativeAssets;
async function nativeFixture(){
 nativeAssets??=Promise.all([loadMujoco(),readFile(new URL('../../models/flybody-mujoco.xml',import.meta.url),'utf8')]);
 const [mj,xml]=await nativeAssets,model=mj.MjModel.from_xml_string(xml),data=new mj.MjData(model),f=fixture();
 data.qpos.set([0,0,3,1,0,0,0]);
 f.world={...f.world,mj,model};f.body={get time(){return data.time;},data};
 f.setTime=t=>{data.time=t;f.fly.bodyTime=t;f.fly.feedback=feedback(t);};
 f.dispose=()=>{data.delete();model.delete();};
 return f;
}
async function system(f,config={},inputSensory=sensory){
 const resources=await createTrainingSensoryResources({...options,sensory:inputSensory,config:{...defaultConfig,...config}});
 return resources.create(f);
}
function compareExcept(actual,expected,excluded){
 const a=rates(actual),b=rates(expected);
 for(const [index,value]of b)if(!excluded.has(index))assert.equal(a.get(index),value,`unrelated input ${index}`);
}
function checkInjected(actual,sample){
 const byIndex=rates(actual);
 for(let k=0;k<sample.indices.length;k++)assert.equal(byIndex.get(sample.indices[k]),sample.ratesHz[k],`replacement ${sample.indices[k]}`);
}

test('absent structural configuration preserves every legacy input and cached-update behavior',async()=>{
 const f=fixture(),s=await system(f),legacy=new SensoryEncoder(sensory,groups,f.world.habitat);
 try{for(const t of [0,.002,.004]){
  f.setTime(t);const actual=s.update(),expected=legacy.update(f.fly,null,flags);
  assert.deepEqual(actual.indices,expected.indices);assert.deepEqual(actual.ratesHz,expected.ratesHz);assert.deepEqual(actual.sample,expected.sample);
  assert.equal(s.update(),null);assert.equal(legacy.update(f.fly,null,flags),null);
 }}finally{s.dispose();}
});

test('leg integration replaces all1066 cells exactly and preserves every other input byte',async()=>{
 const f=fixture(),s=await system(f,{legProprioception:LEG_PROPRIOCEPTION_PRIOR}),legacy=new SensoryEncoder(sensory,groups,f.world.habitat);
 try{for(const t of [0,.002]){
  f.setTime(t);if(t)f.fly.feedback.legs.forEach(l=>{l.tibiaAngle*=-1;l.tibiaVelocity*=-1;});
  const actual=s.update(),expected=legacy.update(f.fly,null,flags),replacement=s.legs.sample(f.fly.feedback);
  assert.equal(replacement.indices.length,1066);assert.equal(actual.indices.length,new Set(actual.indices).size);
  assert.deepEqual(actual.indices,expected.indices);compareExcept(actual,expected,legIds);checkInjected(actual,replacement);
  assert.equal(actual.sample.legProprioception.annotationAbstentions,434);
  const byIndex=rates(actual);assert(legCatalog.cells.filter(c=>c.abstention).every(c=>byIndex.get(c.index)===0));
  for(const c of sensory.channels.filter(c=>/^self_motion_/.test(c.key)))close(actual.sample.body.rates[c.key],c.indices.reduce((sum,i)=>sum+byIndex.get(i),0)/c.indices.length);
 }}finally{s.dispose();}
});

test('native antenna integration uses signed local airflow, is frame-covariant and zeros unresolved families',async()=>{
 const fixtures=await Promise.all([nativeFixture(),nativeFixture(),nativeFixture()]),systems=[];
 try{
  for(const f of fixtures)systems.push(await system(f,{antennaFeedback}));
  // Same local motion under a +90-degree world rotation, then exact reversal.
  fixtures[0].body.data.qvel.set([80,30,10,1,2,3]);
  fixtures[1].body.data.qpos.set([0,0,3,Math.SQRT1_2,0,0,Math.SQRT1_2]);
  fixtures[1].body.data.qvel.set([-30,80,10,1,2,3]);
  fixtures[2].body.data.qvel.set([-80,-30,-10,-1,-2,-3]);
  let actual;
  for(const t of [0,.002,.004,.006]){actual=systems.map((s,k)=>{fixtures[k].setTime(t);return s.update();});}
  const [a,b,c]=actual.map(rates);let reversed=0;
  for(const index of antennaIds){close(a.get(index),b.get(index),1e-6);reversed+=a.get(index)!==c.get(index);}
  assert(reversed>100,'signed airflow must change the antennal population');
  for(let k=0;k<3;k++){
   const s=systems[k],expected=s.antenna.sample();checkInjected(actual[k],expected);
   assert.equal(s.antenna.metadata.selection.zeroed,191);
   const active=new Set(s.antenna.metadata.cells.map(c=>c.index)),byIndex=rates(actual[k]);
   assert(Array.from(antennaIds).filter(i=>!active.has(i)).every(i=>byIndex.get(i)===0));
   const legacy=new SensoryEncoder(sensory,groups,fixtures[k].world.habitat).update(fixtures[k].fly,null,flags);
   compareExcept(actual[k],legacy,antennaIds);
   assert.equal(actual[k].sample.antennaFeedback.drivenCells,388);
  }
  const nativeBefore={qpos:Array.from(fixtures[0].body.data.qpos),qvel:Array.from(fixtures[0].body.data.qvel),ctrl:Array.from(fixtures[0].body.data.ctrl)};
  const held=snapshot(systems[0]);assert.equal(systems[0].update(),null);systems[0].summary();assert.deepEqual(snapshot(systems[0]),held);
  assert.deepEqual({qpos:Array.from(fixtures[0].body.data.qpos),qvel:Array.from(fixtures[0].body.data.qvel),ctrl:Array.from(fixtures[0].body.data.ctrl)},nativeBefore);
 }finally{systems.forEach(s=>s.dispose());fixtures.forEach(f=>f.dispose());}
});

test('leg plus schema2 tegula routes stay disjoint and inject each cell once at its own predicted rate',async()=>{
 const f=fixture(),s=await system(f,{legProprioception:LEG_PROPRIOCEPTION_PRIOR,tegulaFeedback:wingPrior},wingManifest),
  legacy=new SensoryEncoder(sensory,groups,f.world.habitat),wing=createTegulaInputMapper(wingPrior);
 try{
  assert([...legIds].every(i=>!wingIds.has(i)));assert.equal(wingManifest.channels.length,sensory.channels.length+2);
  for(const t of [0,.002,.004]){
   f.setTime(t);const actual=s.update(),expected=legacy.update(f.fly,null,flags),wingExpected=wing.rates(f.fly.feedback,true,t),byIndex=rates(actual);
   assert.equal(actual.indices.length,expected.indices.length+26);assert.equal(new Set(actual.indices).size,actual.indices.length);
   checkInjected(actual,s.legs.sample(f.fly.feedback));
   for(const cell of TEGULA_CELLS)assert.equal(byIndex.get(cell.index),Math.fround(wingExpected.cellRates.get(cell.index)));
   compareExcept(actual,expected,new Set([...legIds,...wingIds]));
   assert.equal(actual.sample.wingStrain.profile,wingPrior.profile);assert.equal(actual.sample.legProprioception.drivenCells,632);
  }
  assert.equal(sensory.tegula_model,undefined,'prepared input manifest must remain unchanged');
 }finally{s.dispose();}
});

test('invalid leg samples can be corrected at the same timestamp without leaving legacy rates cached',async()=>{
 const a=fixture(),b=fixture(),s=await system(a,{legProprioception:LEG_PROPRIOCEPTION_PRIOR}),clean=await system(b,{legProprioception:LEG_PROPRIOCEPTION_PRIOR});
 try{
  s.update();clean.update();a.setTime(.002);b.setTime(.002);
  const saved=a.fly.feedback.legs[2].tibiaVelocity;a.fly.feedback.legs[2].tibiaVelocity=NaN;
  assert.throws(()=>s.update(),/Leg proprioception/);a.fly.feedback.legs[2].tibiaVelocity=saved;
  const corrected=s.update(),expected=clean.update();assert(corrected,'corrected sample must be delivered');
  assert.deepEqual(corrected.ratesHz,expected.ratesHz);assert.deepEqual(corrected.sample,expected.sample);
 }finally{s.dispose();clean.dispose();}
});

test('invalid native antenna samples can be corrected at the same timestamp without poisoning the encoder cache',async()=>{
 const a=await nativeFixture(),b=await nativeFixture(),s=await system(a,{antennaFeedback}),clean=await system(b,{antennaFeedback});
 try{
  a.body.data.qvel.set([80,30,10,1,2,3]);b.body.data.qvel.set([80,30,10,1,2,3]);s.update();clean.update();
  a.setTime(.002);b.setTime(.002);const before=snapshot(s);a.body.data.qvel[0]=NaN;
  assert.throws(()=>s.update(),/Antenna airflow/);assert.deepEqual(snapshot(s),before);a.body.data.qvel[0]=80;
  const corrected=s.update(),expected=clean.update();assert(corrected,'corrected sample must be delivered');
  assert.deepEqual(corrected.ratesHz,expected.ratesHz);assert.deepEqual(corrected.sample,expected.sample);assert.deepEqual(snapshot(s),snapshot(clean));
 }finally{s.dispose();clean.dispose();a.dispose();b.dispose();}
});

test('a rejected combined sample cannot advance another observer or poison a corrected retry',async()=>{
 const a=await nativeFixture(),b=await nativeFixture(),config={antennaFeedback,legProprioception:LEG_PROPRIOCEPTION_PRIOR,tegulaFeedback:wingPrior},
  s=await system(a,config,wingManifest),clean=await system(b,config,wingManifest);
 try{
  a.body.data.qvel.set([80,30,10,1,2,3]);b.body.data.qvel.set([80,30,10,1,2,3]);s.update();clean.update();
  for(const [k,mutate,restore,pattern]of [
   [1,f=>{f.fly.feedback.legs[0].tibiaAngle=NaN;},f=>{f.fly.feedback.legs[0].tibiaAngle=-.25;},/Leg proprioception/],
   [2,f=>{f.body.data.qvel[0]=NaN;},f=>{f.body.data.qvel[0]=80;},/Antenna airflow/],
   [3,f=>{f.fly.feedback.wingLoad.localFrame='world';},f=>{f.fly.feedback.wingLoad.localFrame='native-thorax';},/Tegula/]
  ]){
   a.setTime(k*.002);b.setTime(k*.002);const before=snapshot(s);mutate(a);
   assert.throws(()=>s.update(),pattern);assert.deepEqual(snapshot(s),before,'rejected combined input changed an observer');restore(a);
   const corrected=s.update(),expected=clean.update();assert(corrected,'corrected combined sample must be delivered');
   assert.deepEqual(corrected.ratesHz,expected.ratesHz);assert.deepEqual(corrected.sample,expected.sample);assert.deepEqual(snapshot(s),snapshot(clean));
  }
 }finally{s.dispose();clean.dispose();a.dispose();b.dispose();}
});
