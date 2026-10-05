// Actual production BANC-rate -> WASM muscle -> native body pulses. One body
// at a time, root freely falls; no forces, clamps, pose writes after creation.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';

const files=['models/flybody-mujoco.xml','models/flybody-mujoco.json','data/prepared/banc888/io.json',
 'web/flybody-physics.js','web/flybody-leg-actuation.js','packages/banc-runtime/dist/core.wasm'];
const bytes=Object.fromEntries(await Promise.all(files.map(async path=>[path,await fs.readFile(path)])));
const meta=JSON.parse(bytes[files[1]]),io=JSON.parse(bytes[files[2]]),[mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const model=mj.MjModel.from_xml_string(String(bytes[files[0]]));
const joints=new Map(meta.joints.map(j=>[j.name,j])),actuators=new Map(meta.actuators.map(a=>[a.name,a]));
const names=meta.joints.filter(j=>/^(femur|tibia)_T[123]_(left|right)$/.test(j.name)).map(j=>j.name);
const duration=.02;
const report={createdAt:new Date().toISOString(),command:'node scripts/verify-flybody-leg-direction-pulses.mjs',
 scope:'Actual production muscle/physics path, isolated 80 Hz muscle-family pulses from zero activation for 20 ms. One native body at a time, no body state writes after creation; root remains free under gravity. Relative to an independently integrated zero-input body, not a standing/walking success test.',
 sourceHashes:Object.fromEntries(files.map(path=>[path,createHash('sha256').update(bytes[path]).digest('hex')])),
 cases:[],passed:false};
function angle(body,name){
 const [kind,segment,side]=name.split('_'),suffix=segment+'_'+side;
 const seq=kind==='femur'?['coxa','femur','tibia']:['femur','tibia','tarsus'];
 const points=seq.map(n=>Array.from(body.data.xanchor.slice(joints.get(n+'_'+suffix).id*3,joints.get(n+'_'+suffix).id*3+3)));
 const a=points[0].map((v,i)=>v-points[1][i]),b=points[2].map((v,i)=>v-points[1][i]);
 return Math.acos(Math.max(-1,Math.min(1,a.reduce((s,v,i)=>s+v*b[i],0)/(Math.hypot(...a)*Math.hypot(...b)))));
}
function trial(name=null,ioSign=0){
 const body=new FlyBodyPhysics(mj,model,meta,io,n=>new WasmMuscles(core,n),{surface:()=>0,odor:()=>0,foodAt:()=>null});
 const rates=new Map(),groups=io.muscles.filter(m=>m.kind==='leg'&&m.joint===name&&m.sign===ioSign);
 for(const group of groups)for(const index of group.indices)rates.set(index,80);
 const initialRoot=Array.from(body.data.qpos.slice(0,7));let maximumAppliedForce=0,maximumContacts=0;
 for(let step=0;step<20;step++){
  body.step(rates,.001,{flight:false});
  maximumAppliedForce=Math.max(maximumAppliedForce,...body.data.qfrc_applied.map(Math.abs),...body.data.xfrc_applied.map(Math.abs));
  maximumContacts=Math.max(maximumContacts,body.data.ncon);
 }
 mj.mj_forward(model,body.data);
 const result={name,abstractSign:ioSign,groups:groups.map(m=>({target:m.target,indices:m.indices})),
  initialRoot,finalRoot:Array.from(body.data.qpos.slice(0,7)),duration:body.time,maximumAppliedForce,maximumContacts,
  angles:Object.fromEntries(names.map(n=>[n,angle(body,n)])),
  qpos:Object.fromEntries(names.map(n=>[n,body.data.qpos[joints.get(n).qpos]])),
  target:name?body.data.ctrl[actuators.get(name).id]:null,
  appliedTorque:name?body.data.qfrc_actuator[joints.get(name).dof]:null,
  finite:body.data.qpos.every(Number.isFinite)&&body.data.qvel.every(Number.isFinite)};
 body.dispose();return result;
}
const baseline=trial();report.baseline=baseline;
for(const name of names)for(const [functionName,ioSign,expected] of [['flex',1,-1],['extend',-1,1]]){
 const result=trial(name,ioSign),deltaAngle=result.angles[name]-baseline.angles[name],deltaQ=result.qpos[name]-baseline.qpos[name];
 const passed=result.groups.length>0&&expected*deltaAngle>1e-6&&expected*deltaQ>1e-6&&
  result.maximumAppliedForce===0&&result.finite&&Math.abs(result.duration-duration)<1e-10;
 report.cases.push({...result,functionName,expectedNativeDirection:expected,deltaInternalAngleRadiansVsBaseline:deltaAngle,
  deltaInternalAngleDegreesVsBaseline:deltaAngle*180/Math.PI,deltaQposVsBaseline:deltaQ,passed});
}
report.passed=report.cases.every(c=>c.passed)&&baseline.maximumAppliedForce===0;
report.summary={pulses:report.cases.length,passed:report.cases.filter(c=>c.passed).length,
 smallestAbsoluteAngleChangeDegrees:Math.min(...report.cases.map(c=>Math.abs(c.deltaInternalAngleDegreesVsBaseline))),
 largestAbsoluteAngleChangeDegrees:Math.max(...report.cases.map(c=>Math.abs(c.deltaInternalAngleDegreesVsBaseline))),
 baselineRootDropCm:baseline.initialRoot[2]-baseline.finalRoot[2],maximumAppliedForce:Math.max(...report.cases.map(c=>c.maximumAppliedForce))};
model.delete();
await fs.writeFile('reports/flybody-leg-direction-pulses.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({summary:report.summary,failed:report.cases.filter(c=>!c.passed).map(c=>({name:c.name,function:c.functionName,delta:c.deltaInternalAngleDegreesVsBaseline}))},null,2));
assert(report.passed,'Production pulse direction did not match anatomical action');
