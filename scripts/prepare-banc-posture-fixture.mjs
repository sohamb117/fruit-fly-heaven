// Read-only native forward-kinematics fixture from the actual captured start.
// No physical trajectory is stepped and no production file is modified.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {createContactFoodResolver} from '../web/flybody-contact-environment.js';
import {restoreOnsetState} from './flybody-onset-capture-hooks.mjs';
const path=process.argv[2]||'reports/flybody-solid-wing-repair/before-wing/onset/capture.json';
const bytes=await fs.readFile(path),capture=JSON.parse(bytes);
assert(capture.capturePassed&&capture.initial.native.time===0);
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const {scene,initial}=capture,model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
const names=Object.fromEntries([...scene.xml.matchAll(/<geom name="(habitat_fruit_(\d+)_[^"]+)"/g)].map(m=>[m[1],Number(m[2])]));
assert(Object.keys(names).length>0);
const environment={surface:()=>0,odor:()=>0,foodForContact:createContactFoodResolver(mj,model,scene.fruit,names),foodAt:()=>null};
const body=new FlyBodyPhysics(mj,model,scene.metadata,scene.io,n=>new WasmMuscles(core,n),environment);
try{
 restoreOnsetState(body,initial);
 const q=body.data.qpos,meta=scene.metadata,quat=initial.diagnostics.quaternion;
 const feedback={legs:Array.from({length:6},(_,i)=>{
  const side=i<3?'left':'right',segment=`T${i%3+1}`,joint=name=>meta.joints.find(j=>j.name===`${name}_${segment}_${side}`);
  return {loadBodyWeights:initial.diagnostics.legLoads[i],collision:body.legCollisions[i],tibiaAngle:q[joint('tibia').qpos],coxaAngle:q[joint('coxa').qpos],tibiaVelocity:0,vibration:0};
 }),antennae:[{angle:0,speed:0},{angle:0,speed:0}],speed:0,tilt:Math.acos(Math.max(-1,Math.min(1,1-2*(quat[1]**2+quat[2]**2)))),
 angularVelocity:[0,0,0],wingPowerLeft:0,wingPowerRight:0,halterePower:[0,0],
 legFoodContact:Array.from(body.legFoodContact),mouthFoodContact:Array.from(body.mouthFoodContact),wingFoodContact:Array.from(body.wingFoodContact)};
 const geometry=[];
 for(const [name,rostrum,haustellum]of [['neutral',0,0],['rostrum_only',-1.24,0],['haustellum_only',0,-1.59],['both_extensors',-1.24,-1.59]]){
  body.data.qpos.set(initial.native.qpos);body.data.qpos[body.byJoint.get('rostrum').qpos]=rostrum;body.data.qpos[body.byJoint.get('haustellum').qpos]=haustellum;
  mj.mj_forward(model,body.data);body.refresh();
  geometry.push({name,rostrum,haustellum,mouthFoodContact:Array.from(body.mouthFoodContact),mouthContact:body.mouthContact,
   legFoodContact:Array.from(body.legFoodContact),nativeTime:body.data.time});
 }
 const reachable=[];
 for(let r=0;r<=24;r++)for(let h=0;h<=24;h++){
  body.data.qpos.set(initial.native.qpos);
  const rostrum=-1.24*r/24,haustellum=-1.59*h/24;
  body.data.qpos[body.byJoint.get('rostrum').qpos]=rostrum;body.data.qpos[body.byJoint.get('haustellum').qpos]=haustellum;
  mj.mj_forward(model,body.data);body.refresh();
  if(body.mouthContact)reachable.push({rostrum,haustellum,mouthFoodContact:Array.from(body.mouthFoodContact)});
 }
 const report={path,captureSha256:createHash('sha256').update(bytes).digest('hex'),feedback,
  scope:'Held initial posture from the actual solid-fruit browser capture. Native contact flags reconstructed by mj_forward at exact captured qpos/qvel; recorded initial leg loads retained. No integration. Antennae and body velocities are zero at initial state.',
  assumptions:['Holding this body feedback constant for200ms is an input-only diagnostic, not a naturally evolving contact trajectory.',
   'The reach sweep writes only isolated-test joint positions at fixed root and does not show that neural activity can produce those positions.'],
  initialRoot:initial.native.qpos.slice(0,7),initialTime:initial.native.time,initialRecordedContacts:initial.diagnostics.environmentContacts,geometry,
  reachGrid:{samples:625,stepRostrum:1.24/24,stepHaustellum:1.59/24,reachableConfigurations:reachable}};
 await fs.writeFile('reports/banc-sensory-recruitment/actual-posture-fixture.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
}finally{body.dispose();model.delete();}
