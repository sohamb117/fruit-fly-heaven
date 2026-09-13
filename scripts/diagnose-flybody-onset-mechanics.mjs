// Matched, test-only mechanics ablations of the captured original-UI onset.
// Run: node scripts/diagnose-flybody-onset-mechanics.mjs
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
import {onsetState,restoreOnsetState} from './flybody-onset-capture-hooks.mjs';

const options=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const directory=options.directory||'reports/flybody-onset-causal';
const hash=x=>createHash('sha256').update(x).digest('hex');
const [capture,prior,mj,core]=await Promise.all([
 fs.readFile(`${directory}/capture.json`,'utf8').then(JSON.parse),
 fs.readFile(`${directory}/replay.json`,'utf8').then(JSON.parse),loadMujoco(),createWasmCore()]);
assert(prior.passed&&prior.sourceMatch&&prior.interpretationAllowed,'Require formal source-matched baseline replay first');
assert.equal(hash(JSON.stringify(capture)),prior.captureSha256,'Capture changed since baseline replay');
const paths={'/banc-engine/':'packages/banc-runtime/','/body-engine/':'packages/flybody-runtime/node_modules/@mujoco/mujoco/',
 '/body-model/':'models/','/banc-data/':'data/prepared/banc888/'};
for(const [url,expected] of Object.entries(prior.sourceHashes)){
 const prefix=Object.keys(paths).find(p=>url.startsWith(p));
 const file=prefix?paths[prefix]+url.slice(prefix.length):'web'+url;
 assert.equal(hash(await fs.readFile(file)),expected,`Source changed: ${file}`);
}
const template=mj.MjModel.from_xml_string(capture.scene.xml);
const wingBodies=new Set(capture.scene.metadata.wing_bodies),records=[];
for(let id=0;id<template.ngeom;id++)records.push({id,name:mj.mj_id2name(template,5,id),body:template.geom_bodyid[id],
 contype:template.geom_contype[id],conaffinity:template.geom_conaffinity[id],fluidInteraction:template.geom_fluid[id*12]});
const wingCollision=records.filter(g=>wingBodies.has(g.body)&&g.contype!==0);
const wingFluid=records.filter(g=>wingBodies.has(g.body)&&g.fluidInteraction>0);
const environment=records.filter(g=>g.body===0&&g.contype!==0);
assert.equal(wingCollision.length,4);assert.equal(wingFluid.length,2);
const wingIds=new Set(wingCollision.map(g=>g.id));
const wingJoints=capture.scene.metadata.joints.filter(j=>j.name.startsWith('wing_'));
const wingActuators=capture.scene.metadata.actuators.filter(a=>a.name.startsWith('wing_'));
assert.equal(wingJoints.length,6);assert.equal(wingActuators.length,6);
const allow=(a,b)=>!!((a.contype&b.conaffinity)||(b.contype&a.conaffinity));
const maxAbs=a=>Array.from(a).reduce((m,x)=>Math.max(m,Math.abs(x)),0);
const diff=(a,b)=>Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);
template.delete();

function makeModel(mode){
 let xml=capture.scene.xml,changedPairs=[];
 const noEnvironment=mode==='no_wing_environment'||mode==='no_wing_environment_or_fluid';
 if(noEnvironment||mode==='no_wing_collisions'){
  const masks=new Map();
  for(const g of records){
   if(g.contype===0&&g.conaffinity===0)continue;
   assert(g.name,`Active unnamed geom ${g.id} needs an explicit XML identifier`);
   assert.equal(g.contype,1);assert.equal(g.conaffinity,1);
   const pair=mode==='no_wing_collisions'?(wingIds.has(g.id)?[0,0]:[1,1]):
    wingIds.has(g.id)?[2,2]:g.body===0?[1,1]:[1,3];
   masks.set(g.name,pair);
  }
  xml=xml.replace(/<geom\b[^>]*>/g,tag=>{
   const name=tag.match(/\bname="([^"]+)"/)?.[1],mask=masks.get(name);if(!mask)return tag;
   return tag.replace(/\s+contype="[^"]*"/g,'').replace(/\s+conaffinity="[^"]*"/g,'')
    .replace(/\s*\/?>$/,` contype="${mask[0]}" conaffinity="${mask[1]}"/>`);
  });
 }
 const model=mj.MjModel.from_xml_string(xml);model.hfield_data.set(capture.scene.heights);
 assert.equal(model.ngeom,records.length);
 for(let i=0;i<records.length;i++)for(let j=i+1;j<records.length;j++){
  const a=records[i],b=records[j],nextA={contype:model.geom_contype[i],conaffinity:model.geom_conaffinity[i]},
   nextB={contype:model.geom_contype[j],conaffinity:model.geom_conaffinity[j]},before=allow(a,b),after=allow(nextA,nextB);
  if(before===after)continue;
  assert(before&&!after,'Diagnostic must not create new contact pair eligibility');
  const wingEnvironment=(wingIds.has(i)&&b.body===0)||(wingIds.has(j)&&a.body===0);
  assert(mode==='no_wing_collisions'?(wingIds.has(i)||wingIds.has(j)):wingEnvironment,'Unexpected contact pair disabled');
  changedPairs.push([i,j]);
 }
 if(noEnvironment)assert.equal(changedPairs.length,wingCollision.length*environment.length);
 // Current Euler integration does not use the fluid Jacobian. In MuJoCo 3.13
 // implicit fluid derivatives are not scaled by interaction, so do not silently
 // reuse this force-only suppression if the model's integrator changes.
 assert.equal(Number(model.opt.integrator?.value??model.opt.integrator),0,'Wing-fluid diagnostic requires Euler');
 if(mode==='no_wing_fluid'||mode==='no_wing_environment_or_fluid')for(const g of wingFluid)model.geom_fluid[g.id*12]=1e-300;
 return {model,xmlSha256:hash(xml),changedPairs};
}

const report={date:new Date().toISOString(),scope:'Matched recorded motor rates and identical initial body/muscle state; original 18-gain wing servo retained. Native MuJoCo WASM, no browser, root forces, pose correction or neural rerun.',
 captureSha256:prior.captureSha256,sourceHashes:prior.sourceHashes,scriptSha256:hash(await fs.readFile(new URL(import.meta.url))),
 originalSceneXmlSha256:hash(capture.scene.xml),nativeVersion:mj.mj_versionString(),
 wingCollisionGeoms:wingCollision,wingFluidGeoms:wingFluid,environmentGeomIds:environment.map(g=>g.id),wingJoints,wingActuators,
 fineTraceIntervalSeconds:[.075,.13],sampleTiming:'Post mj_step integrated qpos/qvel; contact and force caches describe the just-completed 50 us step (its pre-integration state). No additional forward/step calls.',
 fluidAblation:'Only active wing geom interaction coefficients become 1e-300. A positive coefficient keeps the ellipsoid model selected; all force terms including viscosity are scaled. Exactly zero would enable the inertia-model fallback. This is negligible numerical suppression, not exact zero.',
 integrator:'Euler (0), verified for every compiled condition',
 fluidSource:'https://raw.githubusercontent.com/google-deepmind/mujoco/3.13.0/src/engine/engine_passive.c',
 fluidJacobianCaveat:'MuJoCo 3.13 implicit fluid derivatives are not interaction-scaled. This diagnostic asserts Euler; it must not be reused unchanged with implicit integration.',
 fluidJacobianSource:'https://raw.githubusercontent.com/google-deepmind/mujoco/3.13.0/src/engine/engine_derivative.c',
 cases:[],baselineGate:{tolerances:{qpos:1e-9,qvel:1e-7,ctrl:1e-9,muscleState:1e-6}}};

function runCase(mode){
 const started=performance.now(),{model,xmlSha256,changedPairs}=makeModel(mode);
 const habitat=createHabitat(structuredClone(capture.scene.fruit));
 const force=new mj.DoubleBuffer(6),adapter=Object.create(mj);
 let observing=false,body,firstWingEnv=null,firstWingSelf=null,firstOverturned=null,firstAirborne=null;
 let maxOmega=0,maxRise=-Infinity,minUp=1,maxApplied=0,maxFluidRoot=0,maxFluidWing=0;
 const errors={qpos:0,qvel:0,ctrl:0,muscleState:0},trace=[],samples=[];
 const contactStats={wing_environment:{steps:0,contacts:0,positiveForceContacts:0,maxNormalForce:0,normalImpulse:0},
  wing_self:{steps:0,contacts:0,positiveForceContacts:0,maxNormalForce:0,normalImpulse:0},other:{steps:0,contacts:0,positiveForceContacts:0,maxNormalForce:0,normalImpulse:0}};
 const peaks={angularSpeed:null,wingEnvironmentNormalForce:null,wingSelfNormalForce:null,fluidRootForce:null};
 const firstContacts={wingEnvironment:null,wingSelf:null};
 adapter.mj_step=(m,d)=>{
  mj.mj_step(m,d);if(!observing)return;
  const q=Array.from(d.qpos),v=Array.from(d.qvel),time=d.time,up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6));
  maxRise=Math.max(maxRise,q[2]-capture.initial.native.qpos[2]);minUp=Math.min(minUp,up);
  if(up<0&&firstOverturned===null)firstOverturned=time;
  maxApplied=Math.max(maxApplied,maxAbs(d.qfrc_applied),maxAbs(d.xfrc_applied));
  const contactRows=[],categories=new Set(),contacts=d.ncon?d.contact:null;
  try{for(let i=0;i<d.ncon;i++){
   const c=contacts.get(i);
   try{
    const geoms=Array.from(c.geom),[a,b]=geoms.map(id=>records[id]);
    const hasWing=wingIds.has(a.id)||wingIds.has(b.id);
    const category=hasWing?((a.body===0||b.body===0)?'wing_environment':'wing_self'):'other';
    mj.mj_contactForce(m,d,i,force);const f=Array.from(force.GetView()),normal=Math.max(0,f[0]);
    const frame=Array.from(c.frame);
    const row={geoms,category,distance:c.dist,position:Array.from(c.pos),normal:frame.slice(0,3),forceContactFrame:f,
     forceWorld:Array.from({length:3},(_,axis)=>f[0]*frame[axis]+f[1]*frame[3+axis]+f[2]*frame[6+axis])};
    contactRows.push(row);categories.add(category);
    const stats=contactStats[category];stats.contacts++;stats.positiveForceContacts+=normal>0?1:0;stats.normalImpulse+=normal*capture.scene.metadata.timestep;
    if(normal>stats.maxNormalForce){stats.maxNormalForce=normal;
     if(category!=='other')peaks[category==='wing_environment'?'wingEnvironmentNormalForce':'wingSelfNormalForce']={time,...row,geomNames:[a.name,b.name],geomBodies:[a.body,b.body],root:q.slice(0,7),omega};}
    if(normal>0&&category==='wing_environment'&&firstWingEnv===null){firstWingEnv=time;
     firstContacts.wingEnvironment={time,...row,geomNames:[a.name,b.name],geomBodies:[a.body,b.body],root:q.slice(0,7),omega};}
    if(normal>0&&category==='wing_self'&&firstWingSelf===null){firstWingSelf=time;
     firstContacts.wingSelf={time,...row,geomNames:[a.name,b.name],geomBodies:[a.body,b.body],root:q.slice(0,7),omega};}
   }finally{c.delete();}
  }}finally{contacts?.delete();}
  for(const key of categories)contactStats[key].steps++;
  const forceNames=['qfrc_actuator','qfrc_passive','qfrc_fluid','qfrc_constraint','qfrc_bias'];
  const forces=Object.fromEntries(forceNames.map(key=>[key,{root:Array.from(d[key]).slice(0,6),wing:wingJoints.map(j=>d[key][j.dof])}]));
  const fluidRoot=Math.hypot(...forces.qfrc_fluid.root.slice(0,3));
  maxFluidWing=Math.max(maxFluidWing,maxAbs(forces.qfrc_fluid.wing));
  const sample={time,root:q.slice(0,7),rootVelocity:v.slice(0,6),up,omega,wingQpos:wingJoints.map(j=>q[j.qpos]),
   wingQvel:wingJoints.map(j=>v[j.dof]),wingCtrl:wingActuators.map(a=>d.ctrl[a.id]),
   wingActuatorForce:wingActuators.map(a=>d.actuator_force[a.id]),forces,contacts:contactRows};
  if(omega>maxOmega){maxOmega=omega;peaks.angularSpeed=sample;}
  if(fluidRoot>maxFluidRoot){maxFluidRoot=fluidRoot;peaks.fluidRootForce=sample;}
  if(time>=.075-1e-10&&time<=.13+1e-10)trace.push(sample);
 };
 try{
  body=new FlyBodyPhysics(adapter,model,capture.scene.metadata,capture.scene.io,n=>new WasmMuscles(core,n),{
   surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
   foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});
  restoreOnsetState(body,capture.initial);observing=true;
  for(const row of capture.frames){
   body.food=structuredClone(row.before.food);body.step(new Map(row.rates),row.duration,row.options);
   const state=onsetState(body);
   if(mode==='baseline'){
    for(const key of ['qpos','qvel','ctrl'])errors[key]=Math.max(errors[key],diff(state.native[key],row.after.native[key]));
    errors.muscleState=Math.max(errors.muscleState,diff(state.muscleState,row.after.muscleState));
   }
   if(body.airborne&&firstAirborne===null)firstAirborne=body.time;
   const q=state.native.qpos,v=state.native.qvel;
   samples.push({time:body.time,root:q.slice(0,7),rootVelocity:v.slice(0,6),up:1-2*(q[4]**2+q[5]**2),omega:Math.hypot(...v.slice(3,6)),
    wing:state.wing,wingPower:body.wingPower,environmentContacts:body.environmentContactCount,airborne:body.airborne,
    wingCtrl:wingActuators.map(a=>state.native.ctrl[a.id])});
  }
  return {mode,seconds:body.time,wallSeconds:(performance.now()-started)/1000,sceneXmlSha256:xmlSha256,
   changedContactPairEligibility:changedPairs,fluidInteraction:wingFluid.map(g=>[g.id,model.geom_fluid[g.id*12]]),
   errors:mode==='baseline'?errors:null,metrics:{maximumAngularSpeedRadPerSecond:maxOmega,maximumRootRiseCm:maxRise,minimumUpZ:minUp,
    firstOverturnedSeconds:firstOverturned,firstAirborneSeconds:firstAirborne,firstWingEnvironmentForceSeconds:firstWingEnv,
    firstWingSelfForceSeconds:firstWingSelf,maximumFluidRootForce:maxFluidRoot,maximumWingFluidGeneralizedTorque:maxFluidWing,
    maximumAppliedForce:maxApplied,finalRoot:samples.at(-1).root},
   contactStats,firstContacts,peaks,samples,trace};
 }finally{body?.dispose();force.delete();model.delete();}
}

const modes=options.modes?.split(',')||['baseline','no_wing_environment','no_wing_fluid','no_wing_collisions','no_wing_environment_or_fluid'];
assert.equal(modes[0],'baseline');
for(const mode of modes){
 assert(['baseline','no_wing_environment','no_wing_fluid','no_wing_collisions','no_wing_environment_or_fluid'].includes(mode));
 const result=runCase(mode);report.cases.push(result);
 console.log(JSON.stringify({mode,errors:result.errors,metrics:result.metrics,contactStats:result.contactStats,wallSeconds:result.wallSeconds}));
 if(mode==='baseline'){
  report.baselineGate.passed=Object.entries(result.errors).every(([key,value])=>value<report.baselineGate.tolerances[key]);
  assert(report.baselineGate.passed,'Instrumented baseline diverged; do not interpret ablation');
 }
 assert.equal(result.metrics.maximumAppliedForce,0);
 await fs.writeFile(`${directory}/mechanics.json`,JSON.stringify(report)+'\n');
}
report.passed=report.baselineGate.passed&&report.cases.every(c=>c.metrics.maximumAppliedForce===0);
report.caveats=['Held recorded MN rates isolate immediate body mechanics; alternative movement would generate different sensory input in a new closed-loop brain run.',
 'Pair masks alter collision eligibility only. Native exclusions and contact parameters remain intact; every changed pair is enumerated.',
 'Angular speed can peak well after first airborne/overturned onset. Peaks and the onset trace are reported separately.',
 'The wing-fluid condition retains negligible positive interaction to prevent fallback; non-wing aerodynamic forces remain present.'];
await fs.writeFile(`${directory}/mechanics.json`,JSON.stringify(report)+'\n');
console.log(JSON.stringify({passed:report.passed,output:`${directory}/mechanics.json`}));
