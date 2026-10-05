// Recorded-input identification only. Never changes production or extends MN input.
// node scripts/audit-takeoff-coordination.mjs [--output=reports/takeoff-calibration]
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
const output=args.output||'reports/takeoff-calibration';
const capturePath=args.capture||'reports/flybody-solid-wing-repair/after-com/onset/capture.json';
const captureBytes=await fs.readFile(capturePath),capture=JSON.parse(captureBytes);
const hash=x=>createHash('sha256').update(x).digest('hex');
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
 assert(expected,`Missing source provenance: ${url}`);
 sourceHashes[url]=hash(await fs.readFile(file));assert.equal(sourceHashes[url],expected,`Source changed: ${file}`);
}
assert(capture.done&&capture.frames.length>0,'A complete actual capture is required');
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const diff=(a,b)=>{assert.equal(a.length,b.length);return Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);};
const maxAbs=a=>Array.from(a).reduce((m,x)=>Math.max(m,Math.abs(x)),0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function rotate(q,v){const t=cross(q.slice(1),v).map(x=>2*x),u=cross(q.slice(1),t);return v.map((x,i)=>x+q[0]*t[i]+u[i]);}
function wrenchAtCom(generalized,q,com){
 const force=Array.from(generalized.slice(0,3)),moment=rotate(q.slice(3,7),Array.from(generalized.slice(3,6)));
 const shift=cross(com.map((x,i)=>x-q[i]),force);
 return {forceWorld:force,momentComWorld:moment.map((x,i)=>x-shift[i])};
}
const specs=[
 {name:'baseline',hindScale:1},
 ...[0,.25,.5,.75].map(hindScale=>({name:`hind_${hindScale}`,hindScale})),
 ...[0,.5].map(allScale=>({name:`all_${allScale}`,allScale})),
 ...[0,.25,.5,.75].map(powerRateScale=>({name:`power_rate_${powerRateScale}`,powerRateScale})),
 ...[0,.5].map(hindScale=>({name:`hind_${hindScale}_power_rate_0.5`,hindScale,powerRateScale:.5})),
 ...[0,.02,.04,.06,.08].map(start=>({name:`hind_release_${Math.round(start*1000)}ms`,release:{start,duration:.04}})),
];
const report={schema:1,date:new Date().toISOString(),capturePath,captureSha256:hash(captureBytes),
 sceneSha256:hash(capture.scene.xml),sourceHashes,scriptSha256:hash(await fs.readFile(new URL(import.meta.url))),
 nativeVersion:mj.mj_versionString(),nativeStepSeconds:capture.scene.metadata.timestep,
 horizonSeconds:capture.frames.reduce((s,f)=>s+f.duration,0),motorInput:'Exactly the 200 captured 2 ms BANC MN updates. No input beyond the capture and no neural rerun.',
 interventions:'Native adhesion-control multiplication; alternatively DLM/DVM MN-rate multiplication before the unchanged WASM muscle decoder. Release diagnostics apply a smoothstep multiplier only to hind adhesion, independent of root state. All other controls use original equations and feedback.',
 measurement:'Native events checked each 50 microseconds; 2 ms snapshots plus first-impact and fixed 136 ms records. COM and wrenches use the native force-evaluation configuration; root poses describe the following integrated state. Contact normal force excludes separately applied adhesion. No mj_forward or physical-state writes during observation.',
 units:{position:'cm',time:'s',force:'g cm/s^2',moment:'g cm^2/s^2',work:'g cm^2/s^2',angle:'rad unless named degrees'},
 legOrder:['left_front','left_middle','left_hind','right_front','right_middle','right_hind'],
 powerMuscles:capture.scene.io.muscles.map((m,index)=>({...m,index})).filter(m=>m.kind==='asynchronous_wing').map(({index,target,joint,indices})=>({index,target,joint,motorCount:indices.length})),
 limitations:['Single initial body pose and 0.4 s recorded motor trajectory; delayed failures and stable flight are not tested.',
 'Adhesion multipliers and release timing are identification interventions, not physiological estimates or deployed policies.',
 'Rate attenuation changes motor input before saturation, activation and fatigue; it is not a torque gain.',
 'The replay follows intervened body mechanics but reuses original neural inputs; it cannot predict closed-loop BANC adaptation.'],cases:[]};

function run(spec){
 const m=mj.MjModel.from_xml_string(capture.scene.xml);m.hfield_data.set(capture.scene.heights);
 const meta=capture.scene.metadata,habitat=createHabitat(structuredClone(capture.scene.fruit));
 const forceBuffer=new mj.DoubleBuffer(6),facade=Object.create(mj),rootBody=m.jnt_bodyid[0];
 const claws=[];
 for(const side of ['left','right'])for(let segment=1;segment<=3;segment++){
  const name=`adhere_claw_T${segment}_${side}`,id=mj.mj_name2id(m,19,name);assert(id>=0);
  assert.equal(m.actuator_trntype[id],5);
  claws.push({name,id,segment,leg:meta.body_to_leg[m.actuator_trnid[id*2]],gain:m.actuator_gainprm[id*10]});
 }
 const wingJoints=meta.joints.filter(j=>j.name.startsWith('wing_')),wingBodies=new Set(meta.wing_bodies);
 const powerIds=new Set(capture.scene.io.muscles.filter(x=>x.kind==='asynchronous_wing').flatMap(x=>x.indices));
 const errors={qpos:0,qvel:0,ctrl:0,muscleState:0};
 const metrics={firstTilt30Seconds:null,firstTilt60Seconds:null,firstInversionSeconds:null,firstWingImpactSeconds:null,
  firstDetached10msSeconds:null,firstDetachedUpright10msSeconds:null,maximumOmega:0,minimumUp:1,maximumRise:-Infinity,
  maximumPreImpactTiltDegrees:0,maximumPreImpactOmega:0,maximumWingNormalForce:0,wingNormalImpulse:0,
  maximumAppliedForce:0,maximumAppliedGeneralized:0,maximumWingJointTorque:0,absoluteWingWork:0,
  lastPositiveSupportBeforeWingImpact:Array(6).fill(null),lastClawContactBeforeWingImpact:Array(6).fill(null)};
 let body,observing=false,step=0,detachedStart=null,uprightDetachedStart=null,latest,firstImpact=null,at136ms=null;
 const samples=[];
 facade.mj_step=(model,d)=>{
  if(!observing){mj.mj_step(model,d);return;}
  const qBefore=Array.from(d.qpos.slice(0,7)),vBefore=Array.from(d.qvel),rawClawControls=claws.map(c=>d.ctrl[c.id]);
  const release=spec.release,u=release?clamp((d.time-release.start)/release.duration):0;
  const releaseMultiplier=1-u*u*(3-2*u);
  const multipliers=claws.map(c=>spec.allScale??(c.segment===3?(release?releaseMultiplier:spec.hindScale??1):1));
  claws.forEach((c,i)=>{d.ctrl[c.id]=rawClawControls[i]*multipliers[i];});
  mj.mj_step(model,d);step++;
  const q=Array.from(d.qpos.slice(0,7)),v=Array.from(d.qvel.slice(0,6)),time=d.time;
  const up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3));
  const tiltDegrees=Math.acos(clamp(up,-1,1))*180/Math.PI;
  metrics.minimumUp=Math.min(metrics.minimumUp,up);metrics.maximumOmega=Math.max(metrics.maximumOmega,omega);
  metrics.maximumRise=Math.max(metrics.maximumRise,q[2]-capture.initial.native.qpos[2]);
  metrics.maximumAppliedForce=Math.max(metrics.maximumAppliedForce,maxAbs(d.xfrc_applied));
  metrics.maximumAppliedGeneralized=Math.max(metrics.maximumAppliedGeneralized,maxAbs(d.qfrc_applied));
  if(tiltDegrees>=30&&metrics.firstTilt30Seconds===null)metrics.firstTilt30Seconds=time;
  if(tiltDegrees>=60&&metrics.firstTilt60Seconds===null)metrics.firstTilt60Seconds=time;
  if(up<0&&metrics.firstInversionSeconds===null)metrics.firstInversionSeconds=time;
  if(metrics.firstWingImpactSeconds===null){metrics.maximumPreImpactTiltDegrees=Math.max(metrics.maximumPreImpactTiltDegrees,tiltDegrees);metrics.maximumPreImpactOmega=Math.max(metrics.maximumPreImpactOmega,omega);}
  const legs=Array.from({length:6},()=>({normalForce:0,clawNormalForce:0,contacts:0,clawContacts:0}));
  const contacts=d.ncon?d.contact:null;let environmentContacts=0,wingNormalForce=0,newImpact=null;
  try{for(let i=0;i<d.ncon;i++){
   const c=contacts.get(i);try{
    const pair=Array.from(c.geom),bodies=pair.map(g=>m.geom_bodyid[g]);
    if((bodies[0]===0)===(bodies[1]===0))continue;
    if(c.dist<.002)environmentContacts++;
    const physicalBody=bodies[0]||bodies[1],leg=meta.body_to_leg[physicalBody]??-1;
    mj.mj_contactForce(m,d,i,forceBuffer);const f=Array.from(forceBuffer.GetView()),normal=Math.max(0,f[0]);
    if(leg>=0){
     legs[leg].normalForce+=normal;if(c.dist<.002)legs[leg].contacts++;
     if(meta.claw_bodies[leg]===physicalBody){legs[leg].clawNormalForce+=normal;if(c.dist<.002)legs[leg].clawContacts++;}
    }
    if(wingBodies.has(physicalBody)&&normal>0){
     wingNormalForce+=normal;metrics.wingNormalImpulse+=normal*m.opt.timestep;
     metrics.maximumWingNormalForce=Math.max(metrics.maximumWingNormalForce,normal);
     if(metrics.firstWingImpactSeconds===null){metrics.firstWingImpactSeconds=time;newImpact={time,geomNames:pair.map(g=>mj.mj_id2name(m,5,g)),position:Array.from(c.pos),normal:Array.from(c.frame).slice(0,3),penetration:c.dist,forceContactFrame:f};}
    }
   }finally{c.delete();}
  }}finally{contacts?.delete();}
  if(metrics.firstWingImpactSeconds===null)legs.forEach((l,i)=>{
   if(l.normalForce>1e-12)metrics.lastPositiveSupportBeforeWingImpact[i]=time;
   if(l.clawContacts)metrics.lastClawContactBeforeWingImpact[i]=time;
  });
  detachedStart=environmentContacts?null:(detachedStart??time);
  uprightDetachedStart=environmentContacts||tiltDegrees>=60?null:(uprightDetachedStart??time);
  if(detachedStart!==null&&time-detachedStart>=.01&&metrics.firstDetached10msSeconds===null)metrics.firstDetached10msSeconds=detachedStart;
  if(uprightDetachedStart!==null&&time-uprightDetachedStart>=.01&&metrics.firstDetachedUpright10msSeconds===null)metrics.firstDetachedUpright10msSeconds=uprightDetachedStart;
  const torques=wingJoints.map(j=>d.qfrc_actuator[j.dof]);
  metrics.maximumWingJointTorque=Math.max(metrics.maximumWingJointTorque,maxAbs(torques));
  metrics.absoluteWingWork+=torques.reduce((s,x,i)=>s+Math.abs(x*vBefore[wingJoints[i].dof]),0)*m.opt.timestep;
  if(step%40===0||newImpact||Math.abs(time-.136)<m.opt.timestep/3){
   const com=Array.from(d.subtree_com.slice(rootBody*3,rootBody*3+3));
   const fluid=wrenchAtCom(d.qfrc_fluid,qBefore,com),constraint=wrenchAtCom(d.qfrc_constraint,qBefore,com),actuator=wrenchAtCom(d.qfrc_actuator,qBefore,com);
   const gravity=Array.from(m.opt.gravity).map(x=>x*meta.mass_g);
   latest={step,time,root:q,rootVelocity:v,rootAcceleration:Array.from(d.qacc.slice(0,6)),comForceEvaluation:com,
    up,tiltDegrees,omega,environmentContacts,legs,clawControlsRaw:rawClawControls,clawControlsApplied:claws.map(c=>d.ctrl[c.id]),
    clawMultipliers:multipliers,clawForces:claws.map(c=>d.actuator_force[c.id]),
    wingPower:Array.from(body.wings.power),wingOpening:Array.from(body.wings.opening),wingDeployment:Array.from(body.wings.deployment),
    wingTarget:Array.from(body.wings.target),wingActual:wingJoints.map(j=>d.qpos[j.qpos]),wingActuatorTorque:torques,wingNormalForce,
    externalWrenches:{fluid,constraint,actuator,gravityForceWorld:gravity,
     sumForceWorld:gravity.map((x,i)=>x+fluid.forceWorld[i]+constraint.forceWorld[i]+actuator.forceWorld[i]),
     sumMomentComWorld:fluid.momentComWorld.map((x,i)=>x+constraint.momentComWorld[i]+actuator.momentComWorld[i])}};
   if(newImpact)firstImpact={...newImpact,sample:latest};
   if(Math.abs(time-.136)<m.opt.timestep/3)at136ms=latest;
  }
  // A fractional multiplier must not compound across the 20 native steps of
  // one muscle-control update. Restore only its input between solver calls.
  claws.forEach((c,i)=>{d.ctrl[c.id]=rawClawControls[i];});
 };
 try{
  body=new FlyBodyPhysics(facade,m,meta,capture.scene.io,n=>new WasmMuscles(core,n),{
   surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
   foodForContact:createContactFoodResolver(mj,m,habitat.fruit,capture.scene.fruitGeomNames),
   foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});
  restoreOnsetState(body,capture.initial);observing=true;
  for(const frame of capture.frames){
   const rates=new Map(frame.rates);
   if(spec.powerRateScale!==undefined)for(const id of powerIds)rates.set(id,(rates.get(id)||0)*spec.powerRateScale);
   body.food=structuredClone(frame.before.food);body.step(rates,frame.duration,frame.options);
   const state=onsetState(body);
   if(spec.name==='baseline'){
    for(const key of ['qpos','qvel','ctrl'])errors[key]=Math.max(errors[key],diff(state.native[key],frame.after.native[key]));
    errors.muscleState=Math.max(errors.muscleState,diff(state.muscleState,frame.after.muscleState));
   }
   const muscles=report.powerMuscles.map(p=>({target:p.target,side:p.joint.endsWith('left')?'left':'right',
    requestedHz:capture.scene.io.muscles[p.index].indices.reduce((s,id)=>s+(rates.get(id)||0),0)/p.motorCount,
    excitation:body.activation[p.index],state:Array.from(body.muscleState.slice(p.index*3,p.index*3+3))}));
   samples.push({...latest,muscles});
  }
  assert.equal(metrics.maximumAppliedForce,0);assert.equal(metrics.maximumAppliedGeneralized,0);
  assert.equal(step,Math.round(report.horizonSeconds/meta.timestep));
  if(spec.name==='baseline')assert.deepEqual(errors,{qpos:0,qvel:0,ctrl:0,muscleState:0},'Stop: baseline differs from capture');
  return {spec,metrics,at136ms,firstImpact,errors:spec.name==='baseline'?errors:null,claws,samples};
 }finally{body?.dispose();forceBuffer.delete();m.delete();}
}

await fs.mkdir(output,{recursive:true});
for(const spec of specs){
 const result=run(spec);report.cases.push(result);
 console.log(JSON.stringify({case:spec.name,metrics:result.metrics,tilt136:result.at136ms.tiltDegrees,errors:result.errors}));
 await fs.writeFile(`${output}/result.json`,JSON.stringify(report)+'\n');
}
report.passed=true;report.passedMeaning='Source provenance, bitexact baseline and no applied-force gates pass; no flight-success claim.';
await fs.writeFile(`${output}/result.json`,JSON.stringify(report)+'\n');
await fs.writeFile(`${output}/summary.json`,JSON.stringify({...report,cases:report.cases.map(({samples,at136ms,firstImpact,...c})=>({...c,tilt136:at136ms.tiltDegrees,power136:at136ms.wingPower,firstImpact:firstImpact&&{time:firstImpact.time,tiltDegrees:firstImpact.sample.tiltDegrees}}))},null,2)+'\n');
