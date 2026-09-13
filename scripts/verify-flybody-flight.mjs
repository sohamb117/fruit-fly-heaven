// Native mechanical regressions. These controlled inputs are not autonomous behavior.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
const [mj,xml,meta]=await Promise.all([loadMujoco(),fs.readFile('models/flybody-mujoco.xml','utf8'),fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse)]);
const args=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
if(args.config)meta.wing_actuation=JSON.parse(await fs.readFile(args.config,'utf8'));
const model=mj.MjModel.from_xml_string(xml),weight=meta.mass_g*981,report={date:new Date().toISOString(),scope:'Restrained-root native aerodynamic measurements and actuator contract tests; not a free-flight success test.',cases:[],checks:[]};
report.sourceSha256=Object.fromEntries(await Promise.all(['web/flybody-wings.js','web/flybody-physics.js','web/flybody-stance.js','models/flybody-mujoco.xml','models/flybody-mujoco.json','scripts/verify-flybody-flight.mjs'].map(async f=>[f,createHash('sha256').update(await fs.readFile(f)).digest('hex')])));
for(const table of meta.wing_actuation.targets)for(let i=0;i<table.length;i++)assert(Math.max(...table[i].map((x,k)=>Math.abs(x-table[(i+1)%table.length][k])))<.1,'Wing trajectory must be continuous at every phase, including wraparound');
report.checks.push('All power levels have continuous periodic joint targets; no Euler-angle wrap impulses');
const cases=[... [0,.05,.1,.25,.5,.75,1].map(power=>({name:`power_${power}`,power,steering:{left:{},right:{}}})),
 {name:'retracted',power:1,steering:{left:{iii1_muscle:1},right:{iii1_muscle:1}}},
 {name:'i1_steering',power:1,steering:{left:{i1_muscle:1},right:{i1_muscle:1}}},
 {name:'b2_left',power:1,steering:{left:{b2_muscle:.8},right:{}}},
 {name:'b2_right',power:1,steering:{left:{},right:{b2_muscle:.8}}},
 ...[.25,.5,.75].flatMap(power=>['left','right'].map(side=>({name:`b2_${side}_${power}`,power,steering:{left:{},right:{},[side]:{b2_muscle:.8}}})))];
for(const c of cases){
 const data=new mj.MjData(model),wings=new FlyBodyWings(meta);
 for(const j of meta.joints)data.qpos[j.qpos]=j.neutral;
 data.qpos.set([0,0,2,1,0,0,0]);const fixed=Float64Array.from(data.qpos),nonwing=meta.joints.filter(j=>!j.name.startsWith('wing_'));
 const force=new Float64Array(6),originTorque=new Float64Array(3);let samples=0;
 const steps=Math.round((.12+20/meta.wing_actuation.frequency_hz)/meta.timestep);
 for(let step=0;step<steps;step++){
  data.qpos.set(fixed.subarray(0,7));data.qvel.fill(0,0,6);
  for(const j of nonwing){data.qpos[j.qpos]=fixed[j.qpos];data.qvel[j.dof]=0;}
  if(step%4===0)wings.step(data.qpos,data.ctrl,c.power,c.power,c.steering,meta.timestep*4);
  mj.mj_step(model,data);
  if(step*meta.timestep>=.12){
   const f=data.qfrc_fluid,com=data.subtree_com,rootBody=model.jnt_bodyid[0];
   const rx=com[rootBody*3]-fixed[0],ry=com[rootBody*3+1]-fixed[1],rz=com[rootBody*3+2]-fixed[2];
   for(let k=0;k<3;k++){force[k]+=f[k];originTorque[k]+=f[k+3];}
   force[3]+=f[3]-(ry*f[2]-rz*f[1]);force[4]+=f[4]-(rz*f[0]-rx*f[2]);force[5]+=f[5]-(rx*f[1]-ry*f[0]);samples++;
  }
 }
 const row={name:c.name,inputPower:c.power,steering:c.steering,meanForceBodyWeights:Array.from(force.slice(0,3),x=>x/samples/weight),meanTorqueAboutCOM:Array.from(force.slice(3),x=>x/samples),meanTorqueAboutThorax:Array.from(originTorque,x=>x/samples),effectivePower:Array.from(wings.power),deployment:Array.from(wings.deployment)};
 assert(data.qpos.every(Number.isFinite));assert(data.qfrc_applied.every(x=>x===0));assert(data.xfrc_applied.every(x=>x===0));
 report.cases.push(row);data.delete();console.log(JSON.stringify(row));
}
const byName=Object.fromEntries(report.cases.map(c=>[c.name,c]));
assert(Math.abs(byName.power_0.meanForceBodyWeights[2])<.001);
assert(byName.power_1.meanForceBodyWeights[2]>1.15&&byName.power_1.meanForceBodyWeights[2]<1.4);
assert(Math.abs(byName.power_1.meanForceBodyWeights[0])<.1);
for(const power of [.05,.1,.25,.5,.75,1])assert(Math.max(...byName[`power_${power}`].meanTorqueAboutCOM.map(Math.abs))<.0003,'Balanced power must not inject a COM pitching moment');
assert(byName['power_0.75'].meanForceBodyWeights[2]>.8);
assert(byName.retracted.effectivePower.every(x=>x===0));assert(Math.abs(byName.retracted.meanForceBodyWeights[2])<.001);
assert(byName.i1_steering.effectivePower.every(x=>x>.99),'I1 steering must not be treated as the III1 retraction muscle');
assert(byName.b2_left.meanTorqueAboutCOM[0]*byName.b2_right.meanTorqueAboutCOM[0]<0);
assert(Math.abs(byName.b2_left.meanTorqueAboutCOM[0])>.0001);
for(const power of [.25,.5,.75])assert(byName[`b2_left_${power}`].meanTorqueAboutCOM[0]*byName[`b2_right_${power}`].meanTorqueAboutCOM[0]<0,'Bilateral b2 responses should oppose at partial power');
report.checks.push('Open hinge supports more than body weight with little forward bias or mean pitch torque',
 'Retractor muscle disengages the hinge despite active power muscles; removing retraction restores lift',
 'Left and right b2 activation produce opposite roll moments through native wing joints',
 'No external body forces or root torques are applied');
if(args.config)report.sourceSha256[args.config]=createHash('sha256').update(await fs.readFile(args.config)).digest('hex');
report.momentReference='Whole-body subtree COM; the root/thorax offset is subtracted from every instantaneous aerodynamic wrench before averaging.';
report.passed=true;model.delete();await fs.writeFile(args.output||'reports/flybody-flight-mechanics-validation.json',JSON.stringify(report,null,2)+'\n');
