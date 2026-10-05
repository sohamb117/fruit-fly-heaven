// Isolated causal claw-adhesion ablation; never modifies production files.
// node scripts/experiment-flybody-claw-release.mjs
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
import {createContactFoodResolver} from '../web/flybody-contact-environment.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {restoreOnsetState,onsetState} from './flybody-onset-capture-hooks.mjs';
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const out='reports/flybody-claw-release',capturePath=args.capture||'reports/flybody-solid-wing-repair/after-com/onset/capture.json';
await fs.mkdir(out,{recursive:true});
const capture=JSON.parse(await fs.readFile(capturePath,'utf8')),hash=x=>createHash('sha256').update(x).digest('hex');
const sources={'/flybody-physics.js':'web/flybody-physics.js','/flybody-wings.js':'web/flybody-wings.js',
 '/flybody-stance.js':'web/flybody-stance.js','/flybody-leg-actuation.js':'web/flybody-leg-actuation.js',
 '/banc-proboscis.js':'web/banc-proboscis.js','/banc/embodiment.js':'web/banc/embodiment.js','/body-world.js':'web/body-world.js',
 '/flybody-habitat-collision.js':'web/flybody-habitat-collision.js','/flybody-contact-environment.js':'web/flybody-contact-environment.js',
 '/banc-engine/src/wasm.js':'packages/banc-runtime/src/wasm.js','/banc-engine/dist/core.js':'packages/banc-runtime/dist/core.js',
 '/banc-engine/dist/core.wasm':'packages/banc-runtime/dist/core.wasm',
 '/body-engine/mujoco.js':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
 '/body-engine/mujoco.wasm':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm',
 '/body-model/flybody-mujoco.xml':'models/flybody-mujoco.xml','/body-model/flybody-mujoco.json':'models/flybody-mujoco.json',
 '/banc-data/io.json':'data/prepared/banc888/io.json'};
const sourceHashes={};
for(const [url,file]of Object.entries(sources)){
 const expected=capture.sourceHashes[url]?.original??capture.sourceHashes[url]?.served;
 assert(expected,`Missing captured source: ${url}`);sourceHashes[url]=hash(await fs.readFile(file));
 assert.equal(sourceHashes[url],expected,`Source changed: ${file}`);
}
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const maxAbs=a=>Array.from(a).reduce((m,x)=>Math.max(m,Math.abs(x)),0);
const diff=(a,b)=>{assert.equal(a.length,b.length);return Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);};
const clamp=x=>Math.max(-1,Math.min(1,x));
const report={date:new Date().toISOString(),capturePath,captureSha256:hash(JSON.stringify(capture)),sceneSha256:hash(capture.scene.xml),
 sourceHashes,scriptSha256:hash(await fs.readFile(new URL(import.meta.url))),nativeVersion:mj.mj_versionString(),
 scope:'Matched captured 805-MN sequence through real WASM muscles and native wing updates. Only selected claw-adhesion controls are zeroed at each native step; other controls follow the original equations and body feedback. No neural rerun, geometry/model change, root force or pose correction.',
 nativeStepSeconds:capture.scene.metadata.timestep,legOrder:['left_front','left_middle','left_hind','right_front','right_middle','right_hind'],
 forceUnits:'g cm/s² (multiply by 10 for microNewton); contact normal force excludes separately applied actuator adhesion.',cases:[]};

function run(mode){
 const m=mj.MjModel.from_xml_string(capture.scene.xml);m.hfield_data.set(capture.scene.heights);
 const meta=capture.scene.metadata,habitat=createHabitat(structuredClone(capture.scene.fruit)),force=new mj.DoubleBuffer(6),facade=Object.create(mj);
 const claws=[];
 for(const side of ['left','right'])for(let segment=1;segment<=3;segment++){
  const name=`adhere_claw_T${segment}_${side}`,id=mj.mj_name2id(m,19,name);assert(id>=0);
  assert.equal(m.actuator_trntype[id],5,'Claw adhesion must transmit to a body');
  const body=m.actuator_trnid[id*2],leg=meta.body_to_leg[body];assert(leg>=0);
  claws.push({name,id,body,leg,segment,gain:m.actuator_gainprm[id*10],range:Array.from(m.actuator_ctrlrange.slice(id*2,id*2+2))});
 }
 assert.equal(claws.length,6);
 const zeroIds=new Set(claws.filter(c=>mode==='all_claws_zero'||mode==='hind_claws_zero'&&c.segment===3).map(c=>c.id));
 const wingBodies=new Set(meta.wing_bodies),errors={qpos:0,qvel:0,ctrl:0,muscleState:0},trace=[],samples=[],impacts=[];
 const metrics={firstTilt30Seconds:null,firstTilt60Seconds:null,firstOverturnedSeconds:null,firstWingImpactSeconds:null,
  maximumOmega:0,minimumUp:1,maximumRise:-Infinity,maximumApplied:0,maximumWingNormalForce:0,wingNormalImpulse:0,
  maximumPositivePitchUpDegrees:0,lastPositiveSupportBeforeWingImpact:Array(6).fill(null),lastClawContactBeforeWingImpact:Array(6).fill(null)};
 let body,observing=false,step=0;
 facade.mj_step=(model,d)=>{
  if(observing)for(const id of zeroIds)d.ctrl[id]=0;
  mj.mj_step(model,d);if(!observing)return;step++;
  const q=Array.from(d.qpos),v=Array.from(d.qvel),up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6)),time=d.time;
  const tiltDegrees=Math.acos(clamp(up))*180/Math.PI,pitchUpDegrees=Math.asin(clamp(2*(q[4]*q[6]-q[3]*q[5])))*180/Math.PI;
  metrics.minimumUp=Math.min(metrics.minimumUp,up);metrics.maximumOmega=Math.max(metrics.maximumOmega,omega);
  metrics.maximumRise=Math.max(metrics.maximumRise,q[2]-capture.initial.native.qpos[2]);
  metrics.maximumApplied=Math.max(metrics.maximumApplied,maxAbs(d.qfrc_applied),maxAbs(d.xfrc_applied));
  metrics.maximumPositivePitchUpDegrees=Math.max(metrics.maximumPositivePitchUpDegrees,pitchUpDegrees);
  if(tiltDegrees>=30&&metrics.firstTilt30Seconds===null)metrics.firstTilt30Seconds=time;
  if(tiltDegrees>=60&&metrics.firstTilt60Seconds===null)metrics.firstTilt60Seconds=time;
  if(up<0&&metrics.firstOverturnedSeconds===null)metrics.firstOverturnedSeconds=time;
  const legs=Array.from({length:6},()=>({contacts:0,clawContacts:0,normalForce:0,clawNormalForce:0})),wingContacts=[];
  const contacts=d.ncon?d.contact:null;
  try{for(let i=0;i<d.ncon;i++){
   const c=contacts.get(i);try{
    const pair=Array.from(c.geom),bodies=pair.map(g=>m.geom_bodyid[g]);
    if(!bodies.includes(0)||bodies[0]===bodies[1])continue;
    const physicalBody=bodies.find(b=>b!==0),leg=meta.body_to_leg[physicalBody]??-1;
    mj.mj_contactForce(m,d,i,force);const f=Array.from(force.GetView()),normal=Math.max(0,f[0]);
    if(leg>=0){
     if(c.dist<.002)legs[leg].contacts++;
     legs[leg].normalForce+=normal;
     if(meta.claw_bodies[leg]===physicalBody){if(c.dist<.002)legs[leg].clawContacts++;legs[leg].clawNormalForce+=normal;}
    }
    if(wingBodies.has(physicalBody)&&normal>0){
     const row={step,time,geomNames:pair.map(g=>mj.mj_id2name(m,5,g)),position:Array.from(c.pos),normal:Array.from(c.frame).slice(0,3),
      penetration:c.dist,forceContactFrame:f,up,tiltDegrees,pitchUpDegrees,omega};
     wingContacts.push(row);impacts.push(row);metrics.wingNormalImpulse+=normal*m.opt.timestep;
     metrics.maximumWingNormalForce=Math.max(metrics.maximumWingNormalForce,normal);
     if(metrics.firstWingImpactSeconds===null)metrics.firstWingImpactSeconds=time;
    }
   }finally{c.delete();}
  }}finally{contacts?.delete();}
  if(metrics.firstWingImpactSeconds===null)legs.forEach((l,i)=>{
   if(l.normalForce>1e-12)metrics.lastPositiveSupportBeforeWingImpact[i]=time;
   if(l.clawContacts>0)metrics.lastClawContactBeforeWingImpact[i]=time;
  });
  const clawControls=claws.map(c=>d.ctrl[c.id]),clawForces=claws.map(c=>d.actuator_force[c.id]);
  if(time<=.18||step%20===0||wingContacts.length)trace.push({step,time,root:q.slice(0,7),rootVelocity:v.slice(0,6),up,tiltDegrees,pitchUpDegrees,omega,
   legs,clawControls,clawForces,wingPower:Array.from(body.wings.power),wingOpening:Array.from(body.wings.opening),
   rootFluid:Array.from(d.qfrc_fluid.slice(0,6)),rootActuator:Array.from(d.qfrc_actuator.slice(0,6)),rootConstraint:Array.from(d.qfrc_constraint.slice(0,6)),
   wingContactCount:wingContacts.length});
 };
 try{
  body=new FlyBodyPhysics(facade,m,meta,capture.scene.io,n=>new WasmMuscles(core,n),{
   surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
   foodForContact:createContactFoodResolver(mj,m,habitat.fruit,capture.scene.fruitGeomNames),
   foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});
  restoreOnsetState(body,capture.initial);observing=true;
  for(const frame of capture.frames){
   body.food=structuredClone(frame.before.food);body.step(new Map(frame.rates),frame.duration,frame.options);
   const state=onsetState(body);
   if(mode==='baseline'){
    for(const k of ['qpos','qvel','ctrl'])errors[k]=Math.max(errors[k],diff(state.native[k],frame.after.native[k]));
    errors.muscleState=Math.max(errors.muscleState,diff(state.muscleState,frame.after.muscleState));
   }
   for(const id of zeroIds)assert.equal(state.native.ctrl[id],0);
   samples.push({time:body.time,root:state.native.qpos.slice(0,7),rootVelocity:state.native.qvel.slice(0,6),ctrl:state.native.ctrl,
    muscleState:state.muscleState,wing:state.wing,contacts:body.environmentContactCount,legLoads:Array.from(body.legLoads)});
  }
  assert.equal(metrics.maximumApplied,0);
  if(mode==='baseline')assert.deepEqual(errors,{qpos:0,qvel:0,ctrl:0,muscleState:0},'Stop: baseline is not bitexact');
  const timeline=[.02,.04,.048,.06,.08,.09,.1,.12,.136,.138,.14,.16,.18].map(time=>{
   const row=trace.reduce((best,r)=>Math.abs(r.time-time)<Math.abs(best.time-time)?r:best,trace[0]);return row;
  });
  return {mode,zeroedControls:claws.filter(c=>zeroIds.has(c.id)),claws,metrics,errors:mode==='baseline'?errors:null,
   firstImpact:impacts[0]??null,peakImpact:impacts.reduce((best,r)=>!best||r.forceContactFrame[0]>best.forceContactFrame[0]?r:best,null),timeline,samples,trace};
 }finally{body?.dispose();force.delete();m.delete();}
}
for(const mode of ['baseline','all_claws_zero','hind_claws_zero']){
 const result=run(mode);report.cases.push(result);console.log(JSON.stringify({mode,metrics:result.metrics,errors:result.errors}));
 await fs.writeFile(`${out}/result.json`,JSON.stringify(report)+'\n');
}
report.passed=true;report.passedMeaning='Source-matched baseline and isolated control/no-root-force gates pass; this does not mean flight succeeds.';
await fs.writeFile(`${out}/result.json`,JSON.stringify(report)+'\n');
