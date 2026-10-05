import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {createContactFoodResolver} from '../web/flybody-contact-environment.js';

// Real native contacts, with a raised food solid and traversable air below.
// foodAt deliberately returns food everywhere: production must use geom IDs.
const [mj,core,xml,meta,io]=await Promise.all([loadMujoco(),createWasmCore(),
  fs.readFile('models/flybody-mujoco.xml','utf8'),fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),
  fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse)]);
const scene=flybodyScene(xml,{surface:()=>({y:.1}),ceiling:50});
const fixture=scene.xml.replace('</worldbody>','<geom name="food_box" type="box" pos="1 0 1.55" size=".6 .6 .05" friction=".6 .005 .0001" condim="3"/></worldbody>');
const model=mj.MjModel.from_xml_string(fixture);model.hfield_data.set(scene.heights);
const food={remaining:1},foodForContact=createContactFoodResolver(mj,model,[food],{food_box:0});
const foodId=mj.mj_name2id(model,5,'food_box'),groundId=mj.mj_name2id(model,5,'ground');
assert(foodId>0);assert.equal(groundId,0);assert.equal(foodForContact(foodId),food);assert.equal(foodForContact(groundId),null);
assert.throws(()=>createContactFoodResolver(mj,model,[food],{missing:0}),/Invalid fruit/);
const environment={surface:()=>.01,odor:()=>0,foodAt:()=>food,foodForContact};
const body=new FlyBodyPhysics(mj,model,meta,io,n=>new WasmMuscles(core,n),environment);
const report={date:new Date().toISOString(),checks:[],foodGeomId:foodId,groundGeomId:groundId};
try{
  body.place(1,0,0);body.step(new Map(),.1);
  assert(!body.airborne);assert(body.legLoads.reduce((s,v)=>s+v,0)>.5);
  assert(!body.onFood);assert(!body.mouthContact);assert(body.legFoodContact.every(v=>v===0));
  report.floorBelowFood={loads:Array.from(body.legLoads),foodContact:body.onFood};
  report.checks.push('Floor support below food does not become taste or feeding contact');
  body.data.qpos[2]=.7;body.data.qvel.fill(0);mj.mj_forward(model,body.data);
  // Deliberately project the top above this unsupported body, as the rendered
  // habitat does. A surface-height test would falsely classify it as grounded.
  environment.surface=()=>1.6;body.refresh();
  assert.equal(body.contactCount,0);assert(body.airborne);assert(!body.onFood);
  assert(body.legLoads.every(v=>v===0));assert(body.legFoodContact.every(v=>v===0));
  report.checks.push('Contact-free space under food is airborne despite a top surface above the body');
  body.place(1,0,0);body.step(new Map(),.1);
  assert(!body.airborne);assert(body.onFood);assert(!body.mouthContact);
  assert(body.legLoads.reduce((s,v)=>s+v,0)>.5);assert(body.legFoodContact.some(v=>v===1));
  assert(body.wingFoodContact.every(v=>v===0));assert(body.mouthFoodContact.every(v=>v===0));
  report.foodSupport={loads:Array.from(body.legLoads),legs:Array.from(body.legFoodContact)};
  report.checks.push('A nonzero static food geom supports native legs and stimulates only touching tarsi');
  for(let i=0;i<150&&!body.mouthContact;i++){
    body.data.qpos[2]-=.001;mj.mj_forward(model,body.data);body.refresh();
  }
  assert(body.mouthContact);assert(body.mouthFoodContact.some(v=>v===1));assert.equal(body.internal.ingested,0);
  report.checks.push('Actual mouth contact with the same food geom activates oral taste without inventing intake');
  body.data.qpos[2]+=2;mj.mj_forward(model,body.data);body.refresh();
  assert(body.airborne);assert(!body.onFood);assert(!body.mouthContact);
  assert(body.legFoodContact.every(v=>v===0));assert(body.mouthFoodContact.every(v=>v===0));
  assert(body.data.qfrc_applied.every(v=>v===0));assert(body.data.xfrc_applied.every(v=>v===0));
  report.checks.push('Leaving the solid clears all support and taste with no applied root force');
  report.passed=true;
}finally{body.dispose();model.delete();}
await fs.writeFile('reports/flybody-contact-environment-validation.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
