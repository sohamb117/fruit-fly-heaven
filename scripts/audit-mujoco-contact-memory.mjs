import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';

// Run modes in separate Node processes: native allocations in the deliberate
// regression fixture must not contaminate the corrected production run.
const option=(name,fallback)=>process.argv.find(x=>x.startsWith(`--${name}=`))?.slice(name.length+3)??fallback;
const mode=option('mode','fixed'),seconds=Number(option('seconds','5'));
assert(['fixed','leaky'].includes(mode));assert(seconds>0&&seconds<=60);
const output=option('output',`reports/mujoco-contact-memory-${mode}.json`);
const counters={vectorsCreated:0,vectorsDeleted:0,contactsCreated:0,contactsDeleted:0};
const messages=[];
const recordNative=(channel,args)=>{
 const message=args.map(String).join(' ');messages.push({channel,message});
 if(messages.length>40)messages.shift();console.error(`[MuJoCo ${channel}] ${message}`);
};
const [mj,core,xml,metadata,io]=await Promise.all([
 loadMujoco({print:(...args)=>recordNative('stdout',args),printErr:(...args)=>recordNative('stderr',args)}),
 createWasmCore(),fs.readFile('models/flybody-mujoco.xml','utf8'),
 fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),
 fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse)
]);

// Instrument documented Embind lifetimes without reading unexported runtime
// properties (their Emscripten getter traps can themselves abort the module).
let contactOwner=mj.MjData.prototype;
while(contactOwner&&!Object.getOwnPropertyDescriptor(contactOwner,'contact'))contactOwner=Object.getPrototypeOf(contactOwner);
const descriptor=Object.getOwnPropertyDescriptor(contactOwner,'contact');
Object.defineProperty(contactOwner,'contact',{...descriptor,get(){
 const vector=descriptor.get.call(this);counters.vectorsCreated++;
 const get=vector.get,deleteVector=vector.delete;
 vector.get=function(index){
  const contact=get.call(this,index);if(!contact)return contact;
  counters.contactsCreated++;const deleteContact=contact.delete;
  contact.delete=function(){const result=deleteContact.call(this);counters.contactsDeleted++;return result;};
  return contact;
 };
 vector.delete=function(){const result=deleteVector.call(this);counters.vectorsDeleted++;return result;};
 return vector;
}});

// Explicit reconstruction of the former refresh() contact loop. The corrected
// stance initializer is shared by both modes; this isolates steady-state
// contact ownership, with identical neural inputs, actuator and body dynamics.
// This is not a source snapshot captured before the production correction.
class LeakyContactBody extends FlyBodyPhysics{
 refresh(){
  const d=this.data,q=d.qpos.slice();this.x=q[0];this.y=q[1];this.z=q[2];this.vx=d.qvel[0];this.vy=d.qvel[1];this.vz=d.qvel[2];this.time=d.time;
  this.heading=Math.atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]*q[5]+q[6]*q[6]));this.quaternion=Array.from(q.slice(3,7));
  this.contactCount=0;this.legLoads.fill(0);this.legCollisions.fill(0);this.contactFood=null;
  for(let i=0;i<d.ncon;i++){
   const contact=d.contact.get(i),geom=contact.geom[0]===0?contact.geom[1]:contact.geom[1]===0?contact.geom[0]:-1;
   if(geom<0||contact.dist>=.002)continue;
   this.contactCount++;const body=this.model.geom_bodyid[geom],leg=this.metadata.body_to_leg?.[body]??-1;
   if(leg>=0){
    this.mj.mj_contactForce(this.model,d,i,this.contactForce);
    this.legLoads[leg]+=Math.max(0,this.contactForce.GetView()[0])/(this.metadata.mass_g*981);
    if(!this.metadata.claw_bodies.includes(body))this.legCollisions[leg]=1;
   }
   if(this.metadata.mouth_bodies?.includes(body)&&contact.dist<=0)this.contactFood=this.environment.foodAt(contact.pos[0],contact.pos[1]);
  }
  this.metadata.joints.forEach((j,i)=>this.jointState.set([q[j.qpos],d.qvel[j.dof]],i*2));
  const clamp=v=>Math.max(0,Math.min(1,v));
  this.proboscis=['rostrum','haustellum'].reduce((sum,name)=>{const j=this.byJoint.get(name);return sum+clamp((j.neutral-q[j.qpos])/(j.neutral-j.range[0]));},0)/2;
  this.feet=this.metadata.feet.map(i=>Array.from(d.site_xpos.slice(i*3,i*3+3)));
  this.airborne=this.contactCount===0&&Math.min(...this.feet.map(p=>p[2]-this.surface(p[0],p[1])))>.01;
  this.mouthContact=!!this.contactFood&&this.contactFood.remaining>0;this.onFood=!!this.environment.foodAt(this.x,this.y)&&!this.airborne;
  if(!q.every(Number.isFinite)||Math.max(...d.qvel.map(Math.abs))>1e5)throw new Error('MuJoCo body became unstable');
 }
}

const report={date:new Date().toISOString(),mode,requestedSeconds:seconds,xmlSha256:metadata.xml_sha256,
 fixture:mode==='leaky'?'Reconstructed unfreed refresh contact loop; corrected stance initializer':'Current production FlyBodyPhysics and stance initializer',
 nativeMessages:messages,samples:[],passed:false};
const scene=flybodyScene(xml,{surface:()=>({y:.1}),ceiling:50});
const environment={surface:()=>.01,odor:()=>0,foodAt:()=>null};
let model,body;const started=performance.now();
const sample=()=>{
 const sample={simSeconds:body.data.time,wallSeconds:(performance.now()-started)/1000,
  wasmHeapCapacityBytes:body.data.qpos.buffer.byteLength,...counters,
  vectorsLive:counters.vectorsCreated-counters.vectorsDeleted,contactsLive:counters.contactsCreated-counters.contactsDeleted,
  ncon:body.data.ncon,position:[body.x,body.y,body.z],quaternion:body.quaternion,
  qpos:Array.from(body.data.qpos),qvel:Array.from(body.data.qvel),rssBytes:process.memoryUsage().rss};
 report.samples.push(sample);
 console.log(JSON.stringify({mode,simSeconds:sample.simSeconds,wallSeconds:sample.wallSeconds,heapBytes:sample.wasmHeapCapacityBytes,
  vectorsLive:sample.vectorsLive,contactsLive:sample.contactsLive}));
 return sample;
};
try{
 model=mj.MjModel.from_xml_string(scene.xml);model.hfield_data.set(scene.heights);
 const Body=mode==='leaky'?LeakyContactBody:FlyBodyPhysics;
 body=new Body(mj,model,metadata,io,n=>new WasmMuscles(core,n),environment);body.place(0,0,0);sample();
 const rates=new Map();let requested=0,nextSample=.25;
 while(requested+1e-12<seconds){
  const duration=Math.min(.01,seconds-requested);body.step(rates,duration);requested+=duration;
  if(requested+1e-12>=nextSample||requested+1e-12>=seconds){sample();nextSample+=.25;}
 }
 const last=report.samples.at(-1);
 assert(Math.abs(last.simSeconds-seconds)<.0011);
 if(mode==='fixed'){
  assert.equal(last.vectorsLive,0,'all copied contact vectors released');
  assert.equal(last.contactsLive,0,'all copied contacts released');
  assert.equal(last.wasmHeapCapacityBytes,report.samples[0].wasmHeapCapacityBytes,'native heap capacity stays bounded after initialization');
 }else assert(last.vectorsLive>0&&last.contactsLive>0,'regression fixture actually exercises ownership leak');
 report.passed=true;
}catch(error){report.error={name:error.name,message:error.message,stack:error.stack};process.exitCode=1;}
finally{
 report.finalCounters={...counters};report.wallSeconds=(performance.now()-started)/1000;
 try{body?.dispose();model?.delete();}catch(error){report.cleanupError={message:error.message};}
 Object.defineProperty(contactOwner,'contact',descriptor);
 await fs.mkdir('reports',{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({mode,passed:report.passed,output,error:report.error,wallSeconds:report.wallSeconds}));
}
