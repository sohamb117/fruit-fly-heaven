// Current capture baseline parity, then isolated wing-contact diagnostics.
// No browser, controller change, root correction or external applied force.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {restoreOnsetState,onsetState} from './flybody-onset-capture-hooks.mjs';
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const directory=args.directory||'reports/flybody-solid-wing-repair/after-com';
const capture=JSON.parse(await fs.readFile(`${directory}/onset/capture.json`,'utf8'));
const output=args.output||`${directory}/wing-contact.json`;
const hash=x=>createHash('sha256').update(x).digest('hex');
const sources={'/flybody-physics.js':'web/flybody-physics.js','/flybody-wings.js':'web/flybody-wings.js',
 '/flybody-stance.js':'web/flybody-stance.js','/flybody-leg-actuation.js':'web/flybody-leg-actuation.js',
 '/banc-proboscis.js':'web/banc-proboscis.js','/banc/embodiment.js':'web/banc/embodiment.js','/body-world.js':'web/body-world.js',
 '/banc-engine/src/wasm.js':'packages/banc-runtime/src/wasm.js','/banc-engine/dist/core.js':'packages/banc-runtime/dist/core.js',
 '/banc-engine/dist/core.wasm':'packages/banc-runtime/dist/core.wasm',
 '/body-engine/mujoco.js':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
 '/body-engine/mujoco.wasm':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm',
 '/body-model/flybody-mujoco.xml':'models/flybody-mujoco.xml','/body-model/flybody-mujoco.json':'models/flybody-mujoco.json',
 '/banc-data/io.json':'data/prepared/banc888/io.json'};
const sourceHashes={};for(const [url,file] of Object.entries(sources)){
 const local=hash(await fs.readFile(file)),expected=capture.sourceHashes[url]?.original??capture.sourceHashes[url]?.served;
 assert.equal(local,expected,`Executed mechanics source changed: ${file}`);sourceHashes[url]=local;
}
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const report={date:new Date().toISOString(),captureSha256:hash(JSON.stringify(capture)),sourceHashes,
 provenance:'All imported/executed mechanics sources match capture; wrapper world/app are not imported or executed. Exact native state parity is required before intervention.',
 scope:'Recorded MN rates held fixed across immediate mechanics interventions. Muscle/servo feedback evolves with each body; not a closed-loop neural rerun.',
 sourceScriptSha256:hash(await fs.readFile(new URL(import.meta.url))),cases:[]};
const difference=(a,b)=>Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);
const maxAbs=a=>Array.from(a).reduce((m,x)=>Math.max(m,Math.abs(x)),0);
function replay(mode){
 let xml=capture.scene.xml;
 const template=mj.MjModel.from_xml_string(xml),records=[];
 const wingBodies=new Set(capture.scene.metadata.wing_bodies);
 for(let id=0;id<template.ngeom;id++)records.push({id,name:mj.mj_id2name(template,5,id),body:template.geom_bodyid[id],
  contype:template.geom_contype[id],conaffinity:template.geom_conaffinity[id]});
 template.delete();
 const activeWingIds=new Set(records.filter(g=>wingBodies.has(g.body)&&(g.contype||g.conaffinity)).map(g=>g.id));
 if(mode==='no_wing_environment'){
  const masks=new Map();
  for(const g of records){
   if(!g.contype&&!g.conaffinity)continue;
   assert(g.name);assert.equal(g.contype,1);assert.equal(g.conaffinity,1);
   masks.set(g.name,activeWingIds.has(g.id)?[2,2]:g.body===0?[1,1]:[1,3]);
  }
  xml=xml.replace(/<geom\b[^>]*>/g,tag=>{
   const mask=masks.get(tag.match(/\bname="([^"]+)"/)?.[1]);if(!mask)return tag;
   return tag.replace(/\s+contype="[^"]*"/g,'').replace(/\s+conaffinity="[^"]*"/g,'')
    .replace(/\s*\/?>$/,` contype="${mask[0]}" conaffinity="${mask[1]}"/>`);
  });
 }
 const m=mj.MjModel.from_xml_string(xml);m.hfield_data.set(capture.scene.heights);
 const allowed=(a,b)=>!!((a.contype&b.conaffinity)||(b.contype&a.conaffinity)),changedPairs=[];
 for(let i=0;i<records.length;i++)for(let j=i+1;j<records.length;j++){
  const before=allowed(records[i],records[j]),after=allowed({contype:m.geom_contype[i],conaffinity:m.geom_conaffinity[i]},
   {contype:m.geom_contype[j],conaffinity:m.geom_conaffinity[j]});
  if(before===after)continue;
  assert(before&&!after);
  assert(mode==='no_wing_environment'&&((activeWingIds.has(i)&&records[j].body===0)||(activeWingIds.has(j)&&records[i].body===0)),
   'Only wing-environment pair eligibility may change');
  changedPairs.push([i,j]);
 }
 if(mode==='no_wing_environment')assert.equal(changedPairs.length,activeWingIds.size*records.filter(g=>g.body===0&&(g.contype||g.conaffinity)).length);
 const wingGeoms=[];
 for(let id=0;id<m.ngeom;id++)if(wingBodies.has(m.geom_bodyid[id])&&(m.geom_contype[id]||m.geom_conaffinity[id])){
  wingGeoms.push({id,name:mj.mj_id2name(m,5,id),originalSolref:Array.from(m.geom_solref.slice(id*2,id*2+2)),
   originalSolimp:Array.from(m.geom_solimp.slice(id*5,id*5+5)),priority:m.geom_priority[id]});
  if(mode!=='baseline'&&mode!=='no_wing_environment')m.geom_solref[id*2]=Number(mode)/1000;
 }
 assert.equal(wingGeoms.length,4);
 const ids=new Set(wingGeoms.map(g=>g.id)),habitat=createHabitat(structuredClone(capture.scene.fruit)),force=new mj.DoubleBuffer(6),facade=Object.create(mj);
 const wingJoints=capture.scene.metadata.joints.filter(j=>j.name.startsWith('wing_'));
 const wingActuators=capture.scene.metadata.actuators.filter(a=>a.name.startsWith('wing_'));
 let body,observing=false,firstImpact=null,firstOverturned=null,maximumOmega=0,maximumRise=-Infinity,minimumUp=1,maximumApplied=0;
 let maximumWingNormalForce=0,normalImpulse=0,peakImpact=null,peakRotation=null;
 const trace=[],samples=[],errors={qpos:0,qvel:0,ctrl:0,muscleState:0};
 facade.mj_step=(model,d)=>{
  mj.mj_step(model,d);if(!observing)return;
  const q=Array.from(d.qpos),v=Array.from(d.qvel),up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6)),time=d.time;
  minimumUp=Math.min(minimumUp,up);maximumRise=Math.max(maximumRise,q[2]-capture.initial.native.qpos[2]);
  if(up<0&&firstOverturned===null)firstOverturned=time;
  maximumApplied=Math.max(maximumApplied,maxAbs(d.qfrc_applied),maxAbs(d.xfrc_applied));
  const wingContacts=[],contacts=d.ncon?d.contact:null;
  try{for(let i=0;i<d.ncon;i++){
   const c=contacts.get(i);try{
    const pair=Array.from(c.geom);
    if(!pair.some(id=>ids.has(id))||!pair.some(id=>m.geom_bodyid[id]===0))continue;
    mj.mj_contactForce(m,d,i,force);const values=Array.from(force.GetView());if(values[0]<=0)continue;
    const row={time,geomIds:pair,geomNames:pair.map(id=>mj.mj_id2name(m,5,id)),position:Array.from(c.pos),normal:Array.from(c.frame).slice(0,3),
     forceContactFrame:values,penetration:c.dist,solref:Array.from(c.solref),solimp:Array.from(c.solimp),up,omega};
    wingContacts.push(row);normalImpulse+=values[0]*m.opt.timestep;
    if(firstImpact===null)firstImpact=row;
    if(values[0]>maximumWingNormalForce){maximumWingNormalForce=values[0];peakImpact=row;}
   }finally{c.delete();}
  }}finally{contacts?.delete();}
  const row={time,root:q.slice(0,7),velocity:v.slice(0,6),up,omega,wingContacts,
   wingQpos:wingJoints.map(j=>q[j.qpos]),wingQvel:wingJoints.map(j=>v[j.dof]),wingControls:wingActuators.map(a=>d.ctrl[a.id]),
   rootFluid:Array.from(d.qfrc_fluid.slice(0,6)),rootConstraint:Array.from(d.qfrc_constraint.slice(0,6))};
  if(omega>maximumOmega){maximumOmega=omega;peakRotation=row;}
  if(time>=.05&&time<=.2||wingContacts.length)trace.push(row);
 };
 try{
  body=new FlyBodyPhysics(facade,m,capture.scene.metadata,capture.scene.io,n=>new WasmMuscles(core,n),{
   surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
   foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});
  restoreOnsetState(body,capture.initial);observing=true;
  for(const frame of capture.frames){
   body.food=structuredClone(frame.before.food);body.step(new Map(frame.rates),frame.duration,frame.options);const state=onsetState(body);
   if(mode==='baseline'){
    for(const key of ['qpos','qvel','ctrl'])errors[key]=Math.max(errors[key],difference(state.native[key],frame.after.native[key]));
    errors.muscleState=Math.max(errors.muscleState,difference(state.muscleState,frame.after.muscleState));
   }
   samples.push({time:body.time,root:state.native.qpos.slice(0,7),velocity:state.native.qvel.slice(0,6),power:body.wingPower,
    wing:state.wing,environmentContacts:body.environmentContactCount,airborne:body.airborne});
  }
  return {mode:['baseline','no_wing_environment'].includes(mode)?mode:`wing_geom_solref_${mode}ms`,
   xmlSha256:hash(xml),changedPairs,wingGeoms,errors:mode==='baseline'?errors:null,
   metrics:{firstWingImpactSeconds:firstImpact?.time??null,firstOverturnedSeconds:firstOverturned,maximumOmega,maximumRise,minimumUp,maximumApplied,
    maximumWingNormalForce,wingNormalImpulse:normalImpulse},firstImpact,peakImpact,peakRotation,samples,trace};
 }finally{body?.dispose();force.delete();m.delete();}
}
for(const mode of (args.modes||'baseline,5,10').split(',')){
 const result=replay(mode);report.cases.push(result);console.log(JSON.stringify({mode,metrics:result.metrics,errors:result.errors,firstImpact:result.firstImpact}));
 if(mode==='baseline')assert.deepEqual(result.errors,{qpos:0,qvel:0,ctrl:0,muscleState:0},'Stop: baseline is not exact');
 assert.equal(result.metrics.maximumApplied,0);
 await fs.writeFile(output,JSON.stringify(report)+'\n');
}
const baseline=report.cases.find(c=>c.mode==='baseline'),noEnvironment=report.cases.find(c=>c.mode==='no_wing_environment');
if(baseline&&noEnvironment){
 let rootBeforeImpact=0,velocityBeforeImpact=0,count=0;
 for(let i=0;i<baseline.trace.length;i++){
  const a=baseline.trace[i],b=noEnvironment.trace[i];
  if(a.time>=baseline.firstImpact.time-1e-10)break;
  assert.equal(a.time,b.time);count++;rootBeforeImpact=Math.max(rootBeforeImpact,difference(a.root,b.root));
  velocityBeforeImpact=Math.max(velocityBeforeImpact,difference(a.velocity,b.velocity));
 }
 report.preImpactParity={nativeStepsCompared:count,maximumRootQposError:rootBeforeImpact,maximumRootQvelError:velocityBeforeImpact};
 assert.equal(rootBeforeImpact,0);assert.equal(velocityBeforeImpact,0);
}
report.caveat='Wing-environment suppression is a causal diagnostic only; its full pair eligibility audit preserves all other contact pairs. 5/10ms time constants are explicit diagnostic priors, not measured material properties or production changes. Native pair mixing is recorded per contact; feet/body response parameters are preserved.';
report.passed=true;await fs.writeFile(output,JSON.stringify(report)+'\n');
