import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
const [mj,core,xml,meta,io,hd]=await Promise.all([loadMujoco(),createWasmCore(),fs.readFile('models/flybody-mujoco.xml','utf8'),fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse),fs.readFile('web/habitat.json','utf8').then(JSON.parse)]);
const report={date:new Date().toISOString(),scope:'One native body at a time, no neural or injected motor drive, all six root DoFs free.',cases:[]};
report.sourceSha256=Object.fromEntries(await Promise.all(['web/flybody-physics.js','web/flybody-habitat-collision.js','web/flybody-stance.js','web/flybody-wings.js','models/flybody-mujoco.json','models/flybody-mujoco.xml'].map(async f=>[f,createHash('sha256').update(await fs.readFile(f)).digest('hex')])));
for(const c of [{name:'original fruit',x:-3.285577942512126,y:-1.2281421463553833,heading:1.428317065961191},{name:'fruit opposite heading',x:-3.285577942512126,y:-1.2281421463553833,heading:1.428317065961191+Math.PI},{name:'flat',x:0,y:0,heading:0,flat:true}]){
 const habitat=c.flat?{surface:()=>({y:1,fruitIndex:-1}),ceiling:50}:createHabitat(hd.fruit);
 const scene=flybodyScene(xml,habitat),model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
 const b=new FlyBodyPhysics(mj,model,meta,io,n=>new WasmMuscles(core,n),{surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:()=>0,foodAt:()=>null});b.place(c.x,c.y,c.heading);
 const initialLegs=new Set();
 const contacts=b.data.ncon?b.data.contact:null;
 try{for(let i=0;i<b.data.ncon;i++){
  const contact=contacts.get(i);
  try{const pair=contact.geom,body0=model.geom_bodyid[pair[0]],body1=model.geom_bodyid[pair[1]];
   const body=body0===0?body1:body1===0?body0:0,leg=meta.body_to_leg[body];if(leg>=0)initialLegs.add(leg);
  }finally{contact.delete();}
 }}finally{contacts?.delete();}
 assert.equal(initialLegs.size,6,`${c.name}: all six feet must reach the native surface`);
 const start=[b.x,b.y,b.z];let maxTilt=0,airborne=0,maxSpeed=0;
 for(let k=0;k<500;k++){b.step(new Map(),.002);maxTilt=Math.max(maxTilt,Math.acos(Math.max(-1,Math.min(1,1-2*(b.quaternion[1]**2+b.quaternion[2]**2)))));airborne+=b.airborne;maxSpeed=Math.max(maxSpeed,Math.hypot(b.vx,b.vy,b.vz));}
 const displacement=Math.hypot(b.x-start[0],b.y-start[1],b.z-start[2]),endSpeed=Math.hypot(b.vx,b.vy,b.vz);
 const row={...c,seconds:b.time,initialLegs:[...initialLegs],displacementMm:displacement*10,maxTilt,maxSpeed,endSpeed,airborneSamples:airborne,loads:Array.from(b.legLoads)};
 report.cases.push(row);console.log(JSON.stringify(row));
 assert.equal(airborne,0);assert(maxTilt<.7);assert(displacement<.06);assert(endSpeed<.1);assert.equal(b.wingPower,0);
 assert(b.data.qfrc_applied.every(x=>x===0));assert(b.data.xfrc_applied.every(x=>x===0));b.dispose();model.delete();
}
report.passed=true;await fs.writeFile('reports/flybody-support-validation.json',JSON.stringify(report,null,2)+'\n');
