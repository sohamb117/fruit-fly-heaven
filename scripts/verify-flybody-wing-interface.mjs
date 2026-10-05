// Counterfactual wing-interface candidates. Original captured MN rates remain
// fixed; free-body probes use stated balanced muscle-force inputs, not BANC.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {createHabitat} from '../web/body-world.js';
import {onsetState,restoreOnsetState} from './flybody-onset-capture-hooks.mjs';
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const out=args.output||'reports/flybody-wing-correction';await fs.mkdir(out,{recursive:true});
const freeSeconds=Number(args['free-seconds']||1),powers=(args.powers||'.75,1').split(',').map(Number);
const [capture,baseXml,baseMeta,candidate,mj,core]=await Promise.all([
 fs.readFile(args.capture||'reports/flybody-onset-causal/capture.json','utf8').then(JSON.parse),fs.readFile('models/flybody-mujoco.xml','utf8'),
 fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile(args.candidate||'reports/flybody-wing-correction/candidate.json','utf8').then(JSON.parse),loadMujoco(),createWasmCore()]);
const hash=x=>createHash('sha256').update(x).digest('hex');
const original=capture.scene.metadata.wing_actuation;
const center=baseMeta.wing_pattern[0].map((_,k)=>baseMeta.wing_pattern.reduce((s,r)=>s+r[k],0)/baseMeta.wing_pattern.length);
const raw={...original,frequency_hz:218,targets:original.powers.map(p=>baseMeta.wing_pattern.map(row=>{
 const a=row.map((x,k)=>center[k]+(x-center[k])*(k===0?Math.sqrt(p):Math.min(1,p/.1)));return [...a,...a];}))};
const report={date:new Date().toISOString(),source:{capture:hash(JSON.stringify(capture)),candidate:hash(JSON.stringify(candidate)),
 physics:hash(await fs.readFile('web/flybody-physics.js')),wings:hash(await fs.readFile('web/flybody-wings.js'))},cases:[]};
const difference=(a,b)=>Array.from(a).reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function recorded(name,config){
 const metadata={...capture.scene.metadata,wing_actuation:config},habitat=createHabitat(structuredClone(capture.scene.fruit));
 const m=mj.MjModel.from_xml_string(capture.scene.xml);m.hfield_data.set(capture.scene.heights);
 const body=new FlyBodyPhysics(mj,m,metadata,capture.scene.io,n=>new WasmMuscles(core,n),{
  surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
  foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});
 restoreOnsetState(body,capture.initial);let maxOmega=0,minUp=1,maxRise=-Infinity,firstOverturn=null,firstAir=null,maxError=0;
 const samples=[];
 try{for(const row of capture.frames){
  body.food=structuredClone(row.before.food);body.step(new Map(row.rates),row.duration,row.options);
  const q=Array.from(body.data.qpos),v=Array.from(body.data.qvel),up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6));
  maxOmega=Math.max(maxOmega,omega);minUp=Math.min(minUp,up);maxRise=Math.max(maxRise,q[2]-capture.initial.native.qpos[2]);
  if(up<0&&firstOverturn===null)firstOverturn=body.time;if(body.airborne&&firstAir===null)firstAir=body.time;
  if(name==='original')maxError=Math.max(maxError,difference(q,row.after.native.qpos),difference(v,row.after.native.qvel));
  samples.push({time:body.time,root:q.slice(0,7),velocity:v.slice(0,6),up,omega,power:body.wingPower,contacts:body.environmentContactCount});
  assert(body.data.qfrc_applied.every(x=>x===0));assert(body.data.xfrc_applied.every(x=>x===0));
 }return {mode:'recorded_mn',name,metrics:{maxOmega,minUp,maxRise,firstOverturn,firstAir},baselineError:name==='original'?maxError:null,samples};
 }finally{body.dispose();m.delete();}
}
function free(name,config,power){
 const m=mj.MjModel.from_xml_string(baseXml),d=new mj.MjData(m),meta={...baseMeta,wing_actuation:config},w=new FlyBodyWings(meta);
 for(const j of meta.joints)d.qpos[j.qpos]=j.neutral;d.qpos.set([0,0,10000,1,0,0,0]);
 const fixed=Array.from(d.qpos),nonwing=meta.joints.filter(j=>!j.name.startsWith('wing_'));
 for(const a of meta.actuators){const j=meta.joints.find(j=>j.id===a.joint);if(j&&!j.name.startsWith('wing_'))d.ctrl[a.id]=clamp(j.neutral,...a.range);}
 const steer={left:{},right:{}},h=meta.timestep;
 // Only this diagnostic warms periodic wing state in a restrained rig.
 // There is no root/other-joint correction after the release below.
 for(let step=0;step<2000;step++){
  d.qpos.set(fixed.slice(0,7));d.qvel.fill(0,0,6);for(const j of nonwing){d.qpos[j.qpos]=j.neutral;d.qvel[j.dof]=0;}
  if(step%4===0)w.step(d.qpos,d.ctrl,power,power,steer,h*4);mj.mj_step(m,d);
 }
 d.qpos.set(fixed.slice(0,7));d.qvel.fill(0,0,6);for(const j of nonwing){d.qpos[j.qpos]=j.neutral;d.qvel[j.dof]=0;}mj.mj_forward(m,d);
 const release=d.time,samples=[];let maxOmega=0,minUp=1,firstOverturn=null;
 for(let step=0;step<Math.round(freeSeconds/h);step++){
  if(step%4===0)w.step(d.qpos,d.ctrl,power,power,steer,h*4);mj.mj_step(m,d);
  const q=d.qpos,v=d.qvel,up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6));
  maxOmega=Math.max(maxOmega,omega);minUp=Math.min(minUp,up);if(up<0&&firstOverturn===null)firstOverturn=d.time-release;
  if(step%40===39)samples.push({time:d.time-release,root:Array.from(q.slice(0,7)),velocity:Array.from(v.slice(0,6)),up,omega});
  assert(d.qfrc_applied.every(x=>x===0));assert(d.xfrc_applied.every(x=>x===0));
 }
 const result={mode:'balanced_free_body',name,power,releaseWarmupSeconds:release,freeSeconds:d.time-release,
  metrics:{maxOmega,minUp,firstOverturn,displacement:samples.at(-1).root.slice(0,3).map((x,i)=>x-fixed[i])},samples};d.delete();m.delete();return result;
}
function coldMotorRamp(name,config,context){
 const groundXml=baseXml.replace('<worldbody>','<worldbody><geom name="wing_test_ground" type="plane" size="100 100 .1" contype="1" conaffinity="1" condim="3" friction=".6 .005 .0001" solref=".002 1" solimp=".95 .99 .01"/>');
 const m=mj.MjModel.from_xml_string(groundXml),metadata={...baseMeta,wing_actuation:config};
 const body=new FlyBodyPhysics(mj,m,metadata,capture.scene.io,n=>new WasmMuscles(core,n),{surface:()=>0,odor:()=>0,foodAt:()=>null});
 body.place(0,0,0);
 if(context==='air'){body.data.qpos[2]=10000;mj.mj_forward(m,body.data);body.refresh();}
 const initial=Array.from(body.data.qpos.slice(0,3)),rates=new Map(capture.scene.io.motor_neurons.map(n=>[n.index,0]));
 for(const muscle of capture.scene.io.muscles)if(muscle.kind==='asynchronous_wing')for(const id of muscle.indices)rates.set(id,80);
 let maxOmega=0,minUp=1,maxRise=-Infinity,firstOverturn=null;const samples=[];
 try{for(let step=0;step<Math.round(freeSeconds/.002);step++){
  body.step(rates,.002,{coupling:true,flight:true});
  const q=Array.from(body.data.qpos),v=Array.from(body.data.qvel),up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...v.slice(3,6));
  maxOmega=Math.max(maxOmega,omega);minUp=Math.min(minUp,up);maxRise=Math.max(maxRise,q[2]-initial[2]);
  if(up<0&&firstOverturn===null)firstOverturn=body.time;
  samples.push({time:body.time,root:q.slice(0,7),velocity:v.slice(0,6),up,omega,power:body.wingPower,contacts:body.environmentContactCount});
  assert(body.data.qfrc_applied.every(x=>x===0));assert(body.data.xfrc_applied.every(x=>x===0));
 }return {mode:'actual_wasm_muscle_cold_ramp',name,context,rateHz:80,seconds:body.time,metrics:{maxOmega,minUp,maxRise,firstOverturn},samples};
 }finally{body.dispose();m.delete();}
}
for(const [name,config]of [['original',original],['power_specific_fit',candidate],['raw_upstream',raw]].filter(([name])=>(args.modes||'original,power_specific_fit').split(',').includes(name))){
 const r=recorded(name,config);report.cases.push(r);console.log(JSON.stringify({name,mode:r.mode,metrics:r.metrics,baselineError:r.baselineError}));
 if(name==='original')assert.equal(r.baselineError,0);
 for(const power of powers){const f=free(name,config,power);report.cases.push(f);console.log(JSON.stringify({name,mode:f.mode,power,metrics:f.metrics}));}
 if(args.cold==='true')for(const context of ['air','flat_floor']){
  const f=coldMotorRamp(name,config,context);report.cases.push(f);console.log(JSON.stringify({name,mode:f.mode,context,metrics:f.metrics}));
 }
 await fs.writeFile(`${out}/candidate-validation.json`,JSON.stringify(report)+'\n');
}
