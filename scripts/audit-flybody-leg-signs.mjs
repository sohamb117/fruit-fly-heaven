// Export exact production placement poses without advancing the simulation,
// then independently measure native segment geometry in Python MuJoCo.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
const files=['web/flybody-physics.js','web/flybody-stance.js','models/flybody-mujoco.xml','models/flybody-mujoco.json',
 'data/prepared/banc888/io.json','scripts/prepare-banc.py','web/habitat.json'];
const bytes=Object.fromEntries(await Promise.all(files.map(async path=>[path,await fs.readFile(path)])));
const hashes=Object.fromEntries(files.map(path=>[path,createHash('sha256').update(bytes[path]).digest('hex')]));
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const xml=String(bytes[files[2]]),meta=JSON.parse(bytes[files[3]]),io=JSON.parse(bytes[files[4]]),hd=JSON.parse(bytes[files[6]]);
const report={scope:'Kinematic sign audit. Production placement only; no integration, controller changes or applied forces.',sourceHashes:hashes,cases:[]};
const muscleProbe=new WasmMuscles(core,3),input=new Float32Array([.5,1,-1,1,1,.5,1,0,1,1,.5,1,1,1,1]);
const output=muscleProbe.step(input,.001),isometric=output[5];
report.compiledMuscleVelocityProbe={inputVelocities:[-1,0,1],forces:[output[2],output[5],output[8]],
 forceRatiosToZeroVelocity:[output[2]/isometric,1,output[8]/isometric],
 interpretation:'The current kernel decreases force for positive velocity input. This input must mean positive shortening speed to match the intended concentric force-velocity effect, not positive lengthening.'};
muscleProbe.dispose();
for(const placement of [{name:'original_fruit',x:-3.285577942512126,y:-1.2281421463553833,heading:1.428317065961191},
 {name:'fruit_opposite_heading',x:-3.285577942512126,y:-1.2281421463553833,heading:1.428317065961191+Math.PI},
 {name:'flat',x:0,y:0,heading:0,flat:true}]){
 const habitat=createHabitat(hd.fruit);if(placement.flat)habitat.surface=()=>({y:1,fruitIndex:-1});
 const scene=flybodyScene(xml,habitat),model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
 const body=new FlyBodyPhysics(mj,model,meta,io,n=>new WasmMuscles(core,n),{
  surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:()=>0,foodAt:()=>null});
 body.place(placement.x,placement.y,placement.heading);
 report.cases.push({...placement,qpos:Array.from(body.data.qpos),time:body.data.time,contactCount:body.contactCount,
  externalForceMaximum:Math.max(...body.data.qfrc_applied.map(Math.abs),...body.data.xfrc_applied.map(Math.abs))});
 body.dispose();model.delete();
}
await fs.writeFile('reports/flybody-leg-sign-placements.json',JSON.stringify(report,null,2)+'\n');
console.log(execFileSync('.venv/bin/python',['scripts/audit-flybody-leg-signs.py'],{encoding:'utf8'}));
