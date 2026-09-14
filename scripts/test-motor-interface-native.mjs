import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {CalibratedWasmMuscles} from '../packages/banc-runtime/src/calibrated-muscles.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {configureMotorInterface} from '../web/motor-interface.js';
import {measureFootSupport} from '../web/training/contact-observation.js';
const [mj,core,xml,meta,io]=await Promise.all([loadMujoco(),createWasmCore(),fs.readFile('models/flybody-mujoco.xml','utf8'),fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse)]);
function make(profile){
 const model=mj.MjModel.from_xml_string(xml.replace('<worldbody>','<worldbody><geom name="test_floor" type="plane" size="10 10 .1"/>'));
 const body=new FlyBodyPhysics(mj,model,meta,io,n=>new WasmMuscles(core,n),{surface:()=>0,odor:()=>0,foodAt:()=>null});body.place(0,0,0);
 if(profile)configureMotorInterface(body,profile,p=>new CalibratedWasmMuscles(core,p));
 return {body,model,dispose(){body.dispose();model.delete();}};
}
test('optional configured defaults reproduce legacy muscle and native-body trajectory exactly',()=>{
 const a=make(),b=make({});
 try{
  const rates=new Map(io.motor_neurons.map(m=>[m.index,0]));
  for(let step=0;step<150;step++){
   for(const mapping of io.muscles)for(const index of mapping.indices)rates.set(index,mapping.kind==='asynchronous_wing'?(step<60?12:40):mapping.kind==='claw_grip_assumption'?20:0);
   a.body.step(rates,.002);b.body.step(rates,.002);
   for(const field of ['qpos','qvel','ctrl','act'])assert.deepEqual([...a.body.data[field]],[...b.body.data[field]],field+' step '+step);
   assert.deepEqual([...a.body.muscleState],[...b.body.muscleState]);
  }
 }finally{a.dispose();b.dispose();}
});
test('upward foot support measurement is read-only and excludes unsupported free air',()=>{
 const fixture=make();try{
  const {body}=fixture;body.step(new Map(),.05);const q=[...body.data.qpos],v=[...body.data.qvel],support=measureFootSupport(body);
  assert(support.footSupportCount>=3);assert(support.footSupportFraction>0);
  assert.deepEqual([...body.data.qpos],q);assert.deepEqual([...body.data.qvel],v);
  body.data.qpos[2]+=2;mj.mj_forward(fixture.model,body.data);assert.deepEqual(measureFootSupport(body),{footSupportCount:0,footSupportFraction:0,environmentContacts:0});
 }finally{fixture.dispose();}
});
