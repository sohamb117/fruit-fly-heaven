// Offline, source-matched MN/muscle replay. Never writes production files.
// Run only after benchmark completion and the native prototype probe passes:
// node scripts/experiment-flybody-wing-flex-replay.mjs
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
import {createFlybodyHabitatCollision} from '../web/flybody-habitat-collision.js';
import {createContactFoodResolver} from '../web/flybody-contact-environment.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {restoreOnsetState,onsetState} from './flybody-onset-capture-hooks.mjs';

const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const out='reports/flybody-wing-flex-prototype';
const capturePath=args.capture||'reports/flybody-solid-wing-repair/after-com/onset/capture.json';
const hash=x=>createHash('sha256').update(x).digest('hex');
const [capture,prototype,baseXml]=await Promise.all([
 fs.readFile(capturePath,'utf8').then(JSON.parse),fs.readFile(`${out}/result.json`,'utf8').then(JSON.parse),
 fs.readFile('models/flybody-mujoco.xml','utf8')]);
assert.equal(prototype.passed,true,'Native conservation and force-probe gate must pass first');
for(const [file,expected]of Object.entries(prototype.sourceHashes))assert.equal(hash(await fs.readFile(file)),expected,`Prototype source changed: ${file}`);
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
 assert(expected,`Missing captured source ${url}`);
 const actual=hash(await fs.readFile(file));assert.equal(actual,expected,`Mechanics source changed: ${file}`);sourceHashes[url]=actual;
}
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const original=mj.MjModel.from_xml_string(capture.scene.xml);
const name=(m,type,id)=>mj.mj_id2name(m,type,id);
const id=(m,type,n)=>{const value=mj.mj_name2id(m,type,n);assert(value>=0,`Missing ${type} name ${n}`);return value;};
const maxAbs=a=>Array.from(a).reduce((m,x)=>Math.max(m,Math.abs(x)),0);
const diff=(a,b)=>{assert.equal(a.length,b.length);return Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);};
const typeWidth=t=>t===0?[7,6]:t===1?[4,3]:[1,1];

function sceneFor(xml,habitat){
 const scene=createFlybodyHabitatCollision(habitat);
 assert.deepEqual(Array.from(scene.heights),capture.scene.heights,'Habitat height samples changed');
 assert.deepEqual(scene.fruitGeomNames,capture.scene.fruitGeomNames,'Geom-to-food identity changed');
 // Browser and Node Math.sin/cos can differ in their final bit (raw wall
 // coordinates are not rounded by the helper). Reuse captured helper output
 // exactly, while checking its generated height and food-identity contracts.
 const assets=capture.scene.xml.match(/<asset>([\s\S]*?)<\/asset>/)?.[1];
 const geoms=capture.scene.xml.match(/<worldbody>([\s\S]*?)(?=\s*<body\b)/)?.[1];
 assert(assets&&geoms);
 if(xml.includes('<asset />'))xml=xml.replace('<asset />',`<asset>${assets}</asset>`);
 else {assert(xml.includes('</asset>'));xml=xml.replace('</asset>',`${assets}</asset>`);}
 assert(xml.includes('<worldbody>'));xml=xml.replace('<worldbody>',`<worldbody>${geoms}`);
 return {...scene,xml};
}

function topology(model){
 const joints=[];
 for(let old=0;old<original.njnt;old++){
  const n=name(original,3,old),next=id(model,3,n),t=original.jnt_type[old];assert.equal(model.jnt_type[next],t);
  const [nq,nv]=typeWidth(t);
  joints.push({name:n,old,next,nq,nv,oldQ:original.jnt_qposadr[old],q:model.jnt_qposadr[next],oldV:original.jnt_dofadr[old],v:model.jnt_dofadr[next]});
 }
 const bodies=Array.from({length:original.nbody},(_,old)=>({name:name(original,1,old),old,next:id(model,1,name(original,1,old))}));
 const sites=Array.from({length:original.nsite},(_,old)=>({name:name(original,6,old),old,next:id(model,6,name(original,6,old))}));
 const targetNames=(m,a)=>{
  const tr=m.actuator_trntype[a],target=Array.from(m.actuator_trnid.slice(a*2,a*2+2));
  const kind=tr<=1?3:tr===3?18:tr===4?6:tr===5?1:null;
  assert(kind!==null,`Unsupported transmission type ${tr}`);
  return [tr,...target.map((x,k)=>x<0||k===1&&tr!==4?null:name(m,kind,x))];
 };
 const actuators=Array.from({length:original.nu},(_,old)=>{
  const n=name(original,19,old),next=id(model,19,n);assert.deepEqual(targetNames(original,old),targetNames(model,next),`Actuator target changed: ${n}`);
  for(const [field,stride]of [['actuator_gainprm',10],['actuator_biasprm',10],['actuator_dynprm',10],['actuator_gear',6],['actuator_ctrlrange',2]])
   assert.deepEqual(Array.from(original[field].slice(old*stride,(old+1)*stride)),Array.from(model[field].slice(next*stride,(next+1)*stride)),`${n} ${field}`);
  const na=original.actuator_actnum[old];assert.equal(na,model.actuator_actnum[next]);
  return {name:n,old,next,target:targetNames(model,next),oldAct:original.actuator_actadr[old],act:model.actuator_actadr[next],na};
 });
 const newJoints=[];const originalNames=new Set(joints.map(j=>j.name));
 for(let j=0;j<model.njnt;j++)if(!originalNames.has(name(model,3,j)))newJoints.push({name:name(model,3,j),id:j,q:model.jnt_qposadr[j],v:model.jnt_dofadr[j]});
 assert(newJoints.every(j=>['wing_left_bend','wing_right_bend'].includes(j.name)));
 return {joints,bodies,sites,actuators,newJoints};
}

function metadataFor(model,map){
 const meta=structuredClone(capture.scene.metadata);
 const bj=new Map(map.joints.map(j=>[j.old,j])),bb=new Map(map.bodies.map(b=>[b.old,b.next])),bs=new Map(map.sites.map(s=>[s.old,s.next]));
 meta.joints=meta.joints.map(j=>{const x=bj.get(j.id);assert.equal(x.name,j.name);return {...j,id:x.next,qpos:x.q,dof:x.v};});
 meta.joints.push(...map.newJoints.map(j=>({name:j.name,id:j.id,qpos:j.q,dof:j.v,neutral:0,range:Array.from(model.jnt_range.slice(j.id*2,j.id*2+2)),passive:true,limited:false})));
 meta.actuators=meta.actuators.map(a=>{const x=map.actuators.find(x=>x.name===a.name);return {...a,id:x.next,joint:a.joint===null?null:bj.get(a.joint).next};});
 meta.feet=meta.feet.map(x=>bs.get(x));
 if(meta.mouth_site!==null)meta.mouth_site=bs.get(meta.mouth_site);
 for(const field of ['claw_bodies','wing_bodies','mouth_bodies'])meta[field]=meta[field].map(x=>bb.get(x));
 meta.leg_bodies=meta.leg_bodies.map(row=>row.map(x=>bb.get(x)));
 const old=meta.body_to_leg;meta.body_to_leg=Array(model.nbody).fill(-1);old.forEach((value,i)=>{meta.body_to_leg[bb.get(i)]=value;});
 return meta;
}

function mappedState(model,map){
 const state=structuredClone(capture.initial),q=Array.from(model.qpos0),v=Array(model.nv).fill(0),warm=v.slice(),applied=v.slice(),rest=q.slice();
 for(const j of map.joints){
  for(let k=0;k<j.nq;k++){q[j.q+k]=state.native.qpos[j.oldQ+k];rest[j.q+k]=state.restPose[j.oldQ+k];}
  for(let k=0;k<j.nv;k++){v[j.v+k]=state.native.qvel[j.oldV+k];warm[j.v+k]=state.native.qacc_warmstart[j.oldV+k];applied[j.v+k]=state.native.qfrc_applied[j.oldV+k];}
 }
 const ctrl=Array(model.nu).fill(0),act=Array(model.na).fill(0),external=Array(model.nbody*6).fill(0);
 for(const a of map.actuators){ctrl[a.next]=state.native.ctrl[a.old];for(let k=0;k<a.na;k++)act[a.act+k]=state.native.act[a.oldAct+k];}
 for(const b of map.bodies)for(let k=0;k<6;k++)external[b.next*6+k]=state.native.xfrc_applied[b.old*6+k];
 state.native={time:state.native.time,qpos:q,qvel:v,qacc_warmstart:warm,qfrc_applied:applied,xfrc_applied:external,ctrl,act};state.restPose=rest;
 assert(map.newJoints.every(j=>q[j.q]===0&&v[j.v]===0));return state;
}

function canonicalState(d,map){
 const q=Array(original.nq),v=Array(original.nv),ctrl=Array(original.nu);
 for(const j of map.joints){for(let k=0;k<j.nq;k++)q[j.oldQ+k]=d.qpos[j.q+k];for(let k=0;k<j.nv;k++)v[j.oldV+k]=d.qvel[j.v+k];}
 for(const a of map.actuators)ctrl[a.old]=d.ctrl[a.next];return {qpos:q,qvel:v,ctrl};
}

async function run(mode){
 const habitat=createHabitat(structuredClone(capture.scene.fruit));habitat.ceiling=capture.scene.ceiling;
 const xml=mode==='baseline'?baseXml:await fs.readFile(`${out}/${mode==='rigid_split'?'rigid-split-control':'flex'}.xml`,'utf8');
 const scene=sceneFor(xml,habitat);if(mode==='baseline')assert.equal(hash(scene.xml),hash(capture.scene.xml),'Baseline scene must be byte identical');
 await fs.writeFile(`${out}/${mode}-scene.xml`,scene.xml);
 const model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
 const nativeOverrides=[];
 if(mode!=='baseline'){
  const required=prototype.requiredNativeOverrides;assert(required,'Require corrected distal-fluid construction metadata');
  assert.equal(Number(model.opt.integrator?.value??model.opt.integrator),required.integratorId,'Fluid suppression supports the audited Euler integrator only');
  for(const patch of required.fluidGeometryOverrides){
   assert.equal(patch.array,'geom_fluid');assert.equal(patch.component,0);assert.equal(patch.stride,12);
   const geom=id(model,5,patch.geom),offset=geom*patch.stride+patch.component;
   assert.equal(model.geom_fluid[offset],patch.expectedCompiledValue);
   model.geom_fluid[offset]=patch.value;nativeOverrides.push({...patch,geomId:geom});
  }
 }
 const map=topology(model),metadata=metadataFor(model,map),state=mappedState(model,map);
 await fs.writeFile(`${out}/${mode}-metadata.json`,JSON.stringify(metadata)+'\n');
 const wingBodies=new Set();
 for(let b=1;b<model.nbody;b++)for(let p=b;p;p=model.body_parentid[p])if(metadata.wing_bodies.includes(p)){wingBodies.add(b);break;}
 const bend=map.newJoints.map(j=>({...j,spec:prototype.wings.find(s=>j.name===s.originalBody+'_bend')}));
 const force=new mj.DoubleBuffer(6),facade=Object.create(mj),trace=[],samples=[],impacts=[];
 const metrics={firstWingImpactSeconds:null,firstOverturnedSeconds:null,minimumUp:1,maximumOmega:0,maximumRise:-Infinity,maximumApplied:0,
  maximumWingNormalForce:0,wingNormalImpulse:0,maximumBendAngle:0,maximumTipNormalDeflectionMm:0,nativeStepsOutsideLinearRange:0};
 const errors={qpos:0,qvel:0,ctrl:0,muscleState:0};let nativeStep=0,observing=false,body;
 facade.mj_step=(m,d)=>{
  mj.mj_step(m,d);if(!observing)return;nativeStep++;
  const q=Array.from(d.qpos),v=Array.from(d.qvel),up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6)),time=d.time;
  metrics.minimumUp=Math.min(metrics.minimumUp,up);metrics.maximumOmega=Math.max(metrics.maximumOmega,omega);
  metrics.maximumRise=Math.max(metrics.maximumRise,q[2]-capture.initial.native.qpos[2]);
  if(up<0&&metrics.firstOverturnedSeconds===null)metrics.firstOverturnedSeconds=time;
  metrics.maximumApplied=Math.max(metrics.maximumApplied,maxAbs(d.qfrc_applied),maxAbs(d.xfrc_applied));
  const bends=bend.map(j=>{
   const angle=q[j.q],lever=j.spec.probeLeverCm,limit=3.3e-6/j.spec.tipStiffnessNPerM/(lever*.01);
   const displacementMm=lever*10*Math.sin(angle),outside=Math.abs(angle)>Math.asin(Math.min(1,limit));
   metrics.maximumBendAngle=Math.max(metrics.maximumBendAngle,Math.abs(angle));
   metrics.maximumTipNormalDeflectionMm=Math.max(metrics.maximumTipNormalDeflectionMm,Math.abs(displacementMm));
   return {name:j.name,angle,velocity:v[j.v],tipNormalDeflectionMm:displacementMm,outsideStaticCalibrationRange:outside,
    springMomentNative:j.spec.rotationalSpringNative*angle};
  });
  if(bends.some(b=>b.outsideStaticCalibrationRange))metrics.nativeStepsOutsideLinearRange++;
  const contacts=d.ncon?d.contact:null,wingContacts=[];
  try{for(let k=0;k<d.ncon;k++){
   const c=contacts.get(k);try{
    const pair=Array.from(c.geom),bodies=pair.map(g=>model.geom_bodyid[g]);
    if(!bodies.includes(0)||!bodies.some(b=>wingBodies.has(b)))continue;
    mj.mj_contactForce(model,d,k,force);const f=Array.from(force.GetView());if(f[0]<=0)continue;
    const row={time,step:nativeStep,geomNames:pair.map(g=>name(model,5,g)),position:Array.from(c.pos),normal:Array.from(c.frame).slice(0,3),
     penetration:c.dist,forceContactFrame:f,up,omega,bends};
    wingContacts.push(row);impacts.push(row);metrics.wingNormalImpulse+=f[0]*model.opt.timestep;
    metrics.maximumWingNormalForce=Math.max(metrics.maximumWingNormalForce,f[0]);
    if(metrics.firstWingImpactSeconds===null)metrics.firstWingImpactSeconds=time;
   }finally{c.delete();}
  }}finally{contacts?.delete();}
  if(time<=.2||nativeStep%20===0||wingContacts.length)trace.push({step:nativeStep,time,root:q.slice(0,7),rootVelocity:v.slice(0,6),up,omega,bends,wingContactCount:wingContacts.length});
 };
 try{
  body=new FlyBodyPhysics(facade,model,metadata,capture.scene.io,n=>new WasmMuscles(core,n),{
   surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
   foodForContact:createContactFoodResolver(mj,model,habitat.fruit,scene.fruitGeomNames),
   foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});
  restoreOnsetState(body,state);observing=true;
  for(const frame of capture.frames){
   body.food=structuredClone(frame.before.food);body.step(new Map(frame.rates),frame.duration,frame.options);
   const current=canonicalState(body.data,map),muscleState=onsetState(body).muscleState;
   if(mode==='baseline'){
    for(const key of ['qpos','qvel','ctrl'])errors[key]=Math.max(errors[key],diff(current[key],frame.after.native[key]));
    errors.muscleState=Math.max(errors.muscleState,diff(muscleState,frame.after.muscleState));
   }
   samples.push({time:body.time,...current,muscleState,power:body.wingPower,contacts:body.environmentContactCount,
    internal:structuredClone(body.internal),bends:bend.map(j=>[body.data.qpos[j.q],body.data.qvel[j.v]])});
  }
  assert.equal(metrics.maximumApplied,0,'No external applied force is allowed in onset replay');
  if(mode==='baseline')assert.deepEqual(errors,{qpos:0,qvel:0,ctrl:0,muscleState:0},'Stop: original baseline is not bitexact');
  return {mode,sceneSha256:hash(scene.xml),nativeOverrides,topology:map,metrics,errors:mode==='baseline'?errors:null,firstImpact:impacts[0]??null,
   peakImpact:impacts.reduce((best,row)=>!best||row.forceContactFrame[0]>best.forceContactFrame[0]?row:best,null),samples,trace,impacts};
 }finally{body?.dispose();force.delete();model.delete();}
}

const report={date:new Date().toISOString(),capturePath,captureSha256:hash(JSON.stringify(capture)),prototypeReportSha256:hash(await fs.readFile(`${out}/result.json`)),
 sourceHashes,scriptSha256:hash(await fs.readFile(new URL(import.meta.url))),nativeVersion:mj.mj_versionString(),
 scope:'Offline captured 805-MN sequence with actual WASM muscle feedback and native wing stepping. No neural rerun, fixed-control replay, pose correction, or root force.',
 calibrationCaveat:'Static load-point evidence extends only to about 3.3 microNewton. Range flags use its equivalent linear tip displacement, not a dynamic material failure law. Fluid geometry remains proximal.',cases:[]};
try{
 for(const mode of ['baseline','rigid_split','flex']){
  const result=await run(mode);report.cases.push(result);console.log(JSON.stringify({mode,metrics:result.metrics,errors:result.errors}));
  await fs.writeFile(`${out}/mn-replay.json`,JSON.stringify(report)+'\n');
 }
 const baseline=report.cases[0];report.preImpactComparisons=[];
 for(const candidate of report.cases.slice(1)){
  const stop=Math.min(baseline.metrics.firstWingImpactSeconds??Infinity,candidate.metrics.firstWingImpactSeconds??Infinity);
  const byStep=new Map(candidate.trace.map(row=>[row.step,row]));let pose=0,velocity=0,count=0;
  for(const a of baseline.trace){if(a.time>=stop-1e-10)break;const b=byStep.get(a.step);assert(b);count++;
   pose=Math.max(pose,diff(a.root,b.root));velocity=Math.max(velocity,diff(a.rootVelocity,b.rootVelocity));}
  report.preImpactComparisons.push({candidate:candidate.mode,cutoffSeconds:stop,nativeSteps:count,maximumRootQposError:pose,maximumRootQvelError:velocity,
   interpretation:candidate.mode==='rigid_split'?'Shape/control comparison: tessellation changes contact geometry.':'Flex may respond to wing inertia before impact; divergence is not solely contact softness.'});
 }
 report.passed=true;report.passedMeaning='All source, baseline, naming and no-root-force gates passed; behavioral outcomes are reported separately.';
 await fs.writeFile(`${out}/mn-replay.json`,JSON.stringify(report)+'\n');
}finally{original.delete();}
