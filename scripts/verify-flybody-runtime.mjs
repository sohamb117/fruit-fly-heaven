import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {bancBodyRate} from '../web/banc-ground-sense.js';
import {PROBOSCIS_SOURCE} from '../web/banc-proboscis.js';

const [mj,core,xml,meta,io,reference]=await Promise.all([loadMujoco(),createWasmCore(),fs.readFile('models/flybody-mujoco.xml','utf8'),
  fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse),
  fs.readFile('models/flybody-native-reference.json','utf8').then(JSON.parse)]);
const report={date:new Date().toISOString(),checks:[],maxNativeError:0,xmlSha256:meta.xml_sha256};
assert.equal(meta.xml_sha256,reference.xml_sha256);assert.equal(meta.mass_g,meta.original_mass_g);
let model=mj.MjModel.from_xml_string(xml),data=new mj.MjData(model);
data.qpos[2]=2;for(const j of meta.joints)data.qpos[j.qpos]=j.neutral;
data.ctrl[meta.actuators.find(a=>a.name==='femur_T1_left').id]=.2;mj.mj_forward(model,data);
for(let step=1;step<=200;step++){
  mj.mj_step(model,data);
  if(step%20===0){const frame=reference.frames[step/20-1];for(const field of ['qpos','qvel'])report.maxNativeError=Math.max(report.maxNativeError,...frame[field].map((v,i)=>Math.abs(v-data[field][i])));}
}
assert(report.maxNativeError<1e-7);data.delete();model.delete();report.checks.push('WASM trajectory matches native MuJoCo, with original mass and coupled inertia');
const habitat={surface:()=>({y:.1}),ceiling:50},scene=flybodyScene(xml,habitat),environment={surface:()=>.01,odor:()=>0,foodAt:()=>null};
model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
const make=()=>{const b=new FlyBodyPhysics(mj,model,meta,io,n=>new WasmMuscles(core,n),environment);b.place(0,0,0);return b;};
let b=make();const start=performance.now();for(let k=0;k<10;k++)b.step(new Map(),.02);
report.quiet={x:b.x,y:b.y,z:b.z,airborne:b.airborne,wingPower:b.wingPower};report.bodySecondsPerWallSecond=.2/((performance.now()-start)/1000);
assert.equal(b.wingPower,0);assert.equal(b.airborne,false);assert(Math.hypot(b.x,b.y)<.05);
report.groundFeedback={supportedLoads:Array.from(b.legLoads),supportedContacts:b.contactCount};
assert(b.legLoads.every(Number.isFinite));assert(b.legLoads.reduce((s,v)=>s+v,0)>.5);
const feedback=()=>({legs:Array.from(b.legLoads,(loadBodyWeights,i)=>({loadBodyWeights,collision:b.legCollisions[i]}))});
const supported=Array.from(b.legLoads,(_,leg)=>bancBodyRate({kind:'load',leg},feedback()));assert(supported.some(v=>v>0));
b.data.qpos[2]+=3;b.data.qvel[b.byJoint.get('tibia_T1_left').dof]=12;mj.mj_forward(model,b.data);b.refresh();
assert.equal(b.airborne,true);assert.equal(b.contactCount,0);assert(b.legLoads.every(v=>v===0));assert(b.legCollisions.every(v=>v===0));
assert(Array.from(b.legLoads,(_,leg)=>bancBodyRate({kind:'load',leg},feedback())).every(v=>v===0));
report.groundFeedback.liftedLoads=Array.from(b.legLoads);b.dispose();
report.checks.push('Quiet body is supported; lifting it clears native per-leg load and touch despite moving joints');
const food={remaining:1};environment.foodAt=()=>food;b=make();b.step(new Map(),.02);
assert.equal(b.onFood,true);assert.equal(b.mouthContact,false);assert.equal(b.internal.ingested,0);
assert(b.legFoodContact.some(v=>v===1));assert(b.wingFoodContact.every(v=>v===0));assert(b.mouthFoodContact.every(v=>v===0));
report.foodContacts={legs:Array.from(b.legFoodContact),wings:Array.from(b.wingFoodContact),mouth:Array.from(b.mouthFoodContact)};
// Lower this isolated test body until the actual native mouth geoms touch.
// Foot/body contact alone must never set the mouth flag.
for(let k=0;k<100&&!b.mouthContact;k++){b.data.qpos[2]-=.001;mj.mj_forward(model,b.data);b.refresh();}
assert.equal(b.mouthContact,true);b.data.qpos[2]+=3;mj.mj_forward(model,b.data);b.refresh();assert.equal(b.mouthContact,false);
assert(b.legFoodContact.every(v=>v===0));assert(b.wingFoodContact.every(v=>v===0));assert(b.mouthFoodContact.every(v=>v===0));
report.checks.push('Food taste follows actual tarsal, wing and mouth contacts; standing on food does not activate noncontacting organs');
// Stimulate documented protractors/extensors and the separately annotated pump.
// Stimulating every proboscis MN also activates antagonists and is not a probe.
const feedTargets=['proboscis_m9_muscle','proboscis_m4a_muscle','proboscis_m4b_muscle'];
const feedRates=new Map(io.muscles.filter(m=>feedTargets.includes(m.target)||m.joint==='pump').flatMap(m=>m.indices.map(i=>[i,80])));
b.step(feedRates,.05);assert(b.proboscis>.05&&b.pump>.05);assert.equal(b.internal.ingested,0);assert.equal(food.remaining,1);
report.airborneFeedingPulse={targets:feedTargets,pumpTargets:io.muscles.filter(m=>m.joint==='pump').map(m=>m.target),proboscis:b.proboscis,pump:b.pump,ingested:b.internal.ingested};
report.checks.push('Only native mouth contact sets feeding contact; an airborne probing and pumping body cannot ingest food');
b.dispose();environment.foodAt=()=>null;
// Fresh, freely falling bodies isolate joint actuation from food or floor
// contact. Root coordinates are changed only for the test's initial condition.
// Production motor decoder receives only the annotated MN rates below.
report.proboscisPulses={source:PROBOSCIS_SOURCE,durationSeconds:.05,motorRateHz:80,samples:[]};
const reachNames=['rostrum','haustellum'];
const pulse=({label,targets,expected,coupling=true})=>{
 const body=make();body.data.qpos[2]+=3;mj.mj_forward(model,body.data);body.refresh();
 const muscles=io.muscles.filter(m=>targets.includes(m.target));
 assert.equal(new Set(muscles.map(m=>m.target)).size,targets.length,'every pulse target has annotated BANC motor neurons');
 const pulseRates=new Map(muscles.flatMap(m=>m.indices.map(id=>[id,80])));
 try{
  body.step(pulseRates,.05,{coupling});
  const sample={label,targets,rootIds:muscles.flatMap(m=>m.root_ids),coupling,
   controls:Object.fromEntries(reachNames.map(name=>[name,body.data.ctrl[body.actuators.get(name).id]])),
   positions:Object.fromEntries(reachNames.map(name=>[name,body.data.qpos[body.byJoint.get(name).qpos]])),
   channels:{...body.proboscisChannels},mouthContact:body.mouthContact,proboscis:body.proboscis};
  for(const name of reachNames){
   const direction=expected[name]??0,neutral=body.byJoint.get(name).neutral;
   if(direction===0)assert.equal(sample.controls[name],neutral,`${label}: no unsupported ${name} command`);
   else{
    assert.ok(direction*(sample.controls[name]-neutral)>.01,`${label}: correct native ${name} actuator sign`);
    assert.ok(direction*(sample.positions[name]-neutral)>.001,`${label}: real native ${name} joint moves in that direction`);
   }
  }
  assert.equal(body.mouthContact,false);assert.equal(body.internal.ingested,0);
  assert(body.data.qfrc_applied.every(v=>v===0));assert(body.data.xfrc_applied.every(v=>v===0));
  report.proboscisPulses.samples.push(sample);
 }finally{body.dispose();}
};
for(const assay of [
 {label:'rostrum extension m9',targets:['proboscis_m9_muscle'],expected:{rostrum:-1}},
 {label:'rostrum retraction m1',targets:['proboscis_m1_muscle'],expected:{rostrum:1}},
 {label:'haustellum extension m4',targets:['proboscis_m4a_muscle','proboscis_m4b_muscle'],expected:{haustellum:-1}},
 {label:'haustellum retraction m3',targets:['proboscis_m3l_muscle','proboscis_m3m_muscle'],expected:{haustellum:1}},
 {label:'unsupported m8',targets:['proboscis_m8_muscle'],expected:{}},
 {label:'unrepresented labellar m6 and m7',targets:['proboscis_m6_muscle','proboscis_m7_muscle'],expected:{}},
 {label:'disconnected reach extensors',targets:feedTargets,expected:{},coupling:false}
])pulse(assay);
report.checks.push('Documented proboscis motor pulses reach opposite native joint directions; m8 and frozen labellar channels do not invent reach extension; coupling-off removes commands');
const muscle=io.muscles.find(m=>m.kind==='leg'&&m.joint==='femur_T1_left'&&m.sign===1),rates=new Map(muscle.indices.map(i=>[i,80]));
b=make();b.step(rates,.05);const j=b.byJoint.get(muscle.joint),driven=b.data.qpos[j.qpos];assert(b.muscleState.some((v,i)=>i%3===2&&v>0));b.dispose();
b=make();b.step(rates,.05,{coupling:false});const disconnected=b.data.qpos[j.qpos];assert(b.muscleState.every(v=>v===0));assert(Math.abs(driven-disconnected)>.01);b.dispose();
report.jointPulse={driven,disconnected};report.checks.push('Identified motor-unit stimulation changes a real joint; disconnection removes muscle actuation');
const grip=io.muscles.find(m=>m.kind==='claw_grip_assumption');assert(grip);
b=make();b.step(new Map(grip.indices.map(i=>[i,80])),.03);const gripControl=b.data.ctrl[b.actuators.get(grip.joint).id];assert(gripControl>0);b.dispose();report.gripControl=gripControl;report.checks.push('Long-tendon motor output reaches the native claw adhesion actuator');
b=make();b.data.qpos[2]+=3;mj.mj_forward(model,b.data);b.refresh();
const wings=io.muscles.filter(m=>m.kind==='asynchronous_wing').flatMap(m=>m.indices);b.step(new Map(wings.map(i=>[i,80])),.04);
assert(b.wingPower>.5);assert(b.data.qfrc_applied.every(v=>v===0));assert(b.data.xfrc_applied.every(v=>v===0));
report.wingPulse={power:b.wingPower,position:[b.x,b.y,b.z],velocity:[b.vx,b.vy,b.vz]};b.dispose();model.delete();
report.checks.push('Wing muscles operate native wing joints and fluid forces with zero externally applied root force');report.passed=true;
await fs.writeFile('reports/flybody-runtime-validation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
