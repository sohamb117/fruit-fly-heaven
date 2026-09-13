// Mechanical replay ablations, one body at a time. This does not validate neural behavior.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
const args=Object.fromEntries(process.argv.slice(2).map(s=>s.replace(/^--/,'').split('=')));
function readContacts(data,details=false){
 const result=[],vector=data.ncon?data.contact:null;
 try{for(let i=0;i<data.ncon;i++){
  const c=vector.get(i);
  try{result.push({geoms:Array.from(c.geom),distance:c.dist,...(details?{dim:c.dim,friction:Array.from(c.friction)}:{})});}finally{c.delete();}
 }}finally{vector?.delete();}
 return result;
}
const source=args.source||'reports/flybody-one-fly-final.json',duration=Number(args.seconds||.4);
const [mj,core,xml,meta,io,sceneData,replay]=await Promise.all([loadMujoco(),createWasmCore(),fs.readFile('models/flybody-mujoco.xml','utf8'),fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse),fs.readFile('web/habitat.json','utf8').then(JSON.parse),fs.readFile(source,'utf8').then(JSON.parse)]);
const habitat=createHabitat(sceneData.fruit.map(f=>({...f,remaining:10})));if(args.terrain==='flat')habitat.surface=()=>({y:1,fruitIndex:-1});const scene=flybodyScene(args.integrator?xml.replace('<option ',`<option integrator="${args.integrator}" `):xml,habitat),model=mj.MjModel.from_xml_string(args.friction?scene.xml.replace('friction=".6 .005 .0001"',`friction="${args.friction} .005 .0001"`):scene.xml);model.hfield_data.set(scene.heights);
const environment={surface:(x,y)=>habitat.surface(x*10,y*10).y/10,odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}};
const kind=new Map(io.muscles.flatMap(m=>m.indices.map(id=>[id,m.kind]))),samples=replay.samples.filter(s=>s.coupled);
const cases={all:{},quiet:{quiet:true},no_wings:{flight:false},power_only:{kinds:['asynchronous_wing']},no_steering:{exclude:['wing_steering_assumption']},wings_only:{kinds:['asynchronous_wing','wing_steering_assumption']},legs_only:{kinds:['leg']},no_grip:{exclude:['claw_grip_assumption']},legs_quarter:{flight:false,range:.25},all_quarter:{range:.25}};
const report={date:new Date().toISOString(),options:args,runtimeSha256:Object.fromEntries(await Promise.all(["web/flybody-physics.js","web/flybody-stance.js","models/flybody-mujoco.json","models/flybody-mujoco.xml"].map(async file=>[file,createHash("sha256").update(await fs.readFile(file)).digest("hex")]))),terrain:args.terrain||'habitat',scope:'Fixed recorded BANC motor-rate replay; sequential isolated native bodies; no new brain simulation.',source,sourceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex'),seconds:duration,cases:[]};
for(const name of (args.cases||Object.keys(cases).join(',')).split(',')){
 const settings=cases[name];if(!settings)throw new Error(name);
 const mm=structuredClone(meta);if(args.stance)mm.initializeStance=args.stance==='true';if(args.range)mm.maxJointExcursion=Number(args.range);if(settings.range)for(const j of mm.joints)if(/^(coxa|femur|tibia|tarsus)/.test(j.name))j.range=j.range.map(v=>j.neutral+(v-j.neutral)*settings.range);
 habitat.fruit.forEach(f=>f.remaining=10);
 const b=new FlyBodyPhysics(mj,model,mm,io,n=>new WasmMuscles(core,n),environment),first=samples[0];b.place(first.position[0]/10,first.position[2]/10,first.heading);
 const initialContacts=readContacts(b.data,true);const row={name,settings,initialContacts,maxTilt:0,maxAngularSpeed:0,maxSpeed:0,uprightSamples:0,airborneSamples:0,samples:0,trace:[]};let index=0;
 for(let t=0;t<duration;t+=.002){
  while(index<samples.length-2&&samples[index+1].neuralMs<t*1000)index++;
  const a=samples[index],c=samples[Math.min(index+1,samples.length-1)],u=Math.max(0,Math.min(1,(t*1000-a.neuralMs)/(c.neuralMs-a.neuralMs||1)));
  const rates=new Map(io.motor_neurons.map((n,k)=>{const type=kind.get(n.index),on=!settings.quiet&&(!settings.kinds||settings.kinds.includes(type))&&(!settings.exclude||!settings.exclude.includes(type));return[n.index,on?(a.motorRates[k]||0)*(1-u)+(c.motorRates[k]||0)*u:0];}));
  b.step(rates,.002,{flight:settings.flight!==false});
  const q=b.quaternion,tilt=Math.acos(Math.max(-1,Math.min(1,1-2*(q[1]*q[1]+q[2]*q[2])))),omega=Math.hypot(...b.data.qvel.slice(3,6)),speed=Math.hypot(b.vx,b.vy,b.vz);
  if(omega>row.maxAngularSpeed)row.peakContacts=readContacts(b.data);row.maxTilt=Math.max(row.maxTilt,tilt);row.maxAngularSpeed=Math.max(row.maxAngularSpeed,omega);row.maxSpeed=Math.max(row.maxSpeed,speed);row.uprightSamples+=tilt<Math.PI/2;row.airborneSamples+=b.airborne;row.samples++;
  if(row.samples%25===0)row.trace.push({t:b.time,position:[b.x,b.y,b.z],tilt,omega,wing:b.wingPower});
 }
 row.endPosition=[b.x,b.y,b.z];row.ingested=b.internal.ingested;row.uprightFraction=row.uprightSamples/row.samples;row.airborneFraction=row.airborneSamples/row.samples;b.dispose();report.cases.push(row);console.log(JSON.stringify({...row,trace:undefined}));
}
model.delete();await fs.writeFile(args.output||'reports/flybody-stability-ablation.json',JSON.stringify(report,null,2)+'\n');
