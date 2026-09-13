// Exact onset replay gate, then paired DLM/DVM and wing-servo interventions.
// No live browser, brain rerun, root forces or state corrections during replay.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
import {createContactFoodResolver} from '../web/flybody-contact-environment.js';
import {onsetState,onsetScene,installOnsetCapture,exportOnsetCapture,restoreOnsetState} from './flybody-onset-capture-hooks.mjs';

const options=Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
const directory=options.output||'reports/flybody-onset-causal';
const hash=x=>createHash('sha256').update(x).digest('hex');
const sourcePaths={
 '/flybody-physics.js':'web/flybody-physics.js','/flybody-wings.js':'web/flybody-wings.js',
 '/flybody-habitat-collision.js':'web/flybody-habitat-collision.js','/flybody-contact-environment.js':'web/flybody-contact-environment.js',
 '/vendor/three.module.js':'web/vendor/three.module.js',
 '/flybody-stance.js':'web/flybody-stance.js','/flybody-leg-actuation.js':'web/flybody-leg-actuation.js',
 '/flybody-world.js':'web/flybody-world.js','/banc-proboscis.js':'web/banc-proboscis.js',
 '/banc/embodiment.js':'web/banc/embodiment.js','/body-world.js':'web/body-world.js',
 '/banc-engine/src/wasm.js':'packages/banc-runtime/src/wasm.js','/banc-engine/dist/core.js':'packages/banc-runtime/dist/core.js',
 '/banc-engine/dist/core.wasm':'packages/banc-runtime/dist/core.wasm',
 '/body-engine/mujoco.js':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
 '/body-engine/mujoco.wasm':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm',
 '/body-model/flybody-mujoco.xml':'models/flybody-mujoco.xml','/body-model/flybody-mujoco.json':'models/flybody-mujoco.json',
 '/banc-data/io.json':'data/prepared/banc888/io.json'};
const localHashes=Object.fromEntries(await Promise.all(Object.entries(sourcePaths).map(async([url,file])=>[url,hash(await fs.readFile(file))])));
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const environment=(habitat,model,fruitGeomNames)=>({surface:(x,y)=>habitat.surface(x*10,y*10).y/10,
 odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),
 foodForContact:fruitGeomNames?createContactFoodResolver(mj,model,habitat.fruit,fruitGeomNames):undefined,
 foodAt:(x,y)=>{const i=habitat.surface(x*10,y*10).fruitIndex;return i>=0?habitat.fruit[i]:null;}});

async function fixture(){
 const [xml,metadata,io,hd]=await Promise.all([fs.readFile('models/flybody-mujoco.xml','utf8'),
  fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse),
  fs.readFile('web/habitat.json','utf8').then(JSON.parse)]);
 const habitat=createHabitat(hd.fruit.map(f=>({...f,remaining:10}))),scene=flybodyScene(xml,habitat),model=mj.MjModel.from_xml_string(scene.xml);
 model.hfield_data.set(scene.heights);
 globalThis.__flybodyOnset={enabled:true,kind:'synthetic-rate-native-fixture-only',seconds:.04,frames:[]};
 onsetScene({habitat,flightEnabled:true,motorCoupling:true,movementMode:'direct'},scene,metadata,io);
 installOnsetCapture(FlyBodyPhysics);
 const body=new FlyBodyPhysics(mj,model,metadata,io,n=>new WasmMuscles(core,n),environment(habitat,model,scene.fruitGeomNames));
 body.place(-3.285577942512126,-1.2281421463553833,1.428317065961191);
 const rates=new Map(io.motor_neurons.map(m=>[m.index,0]));
 for(const m of io.muscles)if(m.kind==='asynchronous_wing'||m.kind==='leg'&&m.joint==='tibia_T1_left'&&m.sign===-1)for(const index of m.indices)rates.set(index,80);
 for(let i=0;i<20;i++)body.step(rates,.002,{coupling:true,flight:true});
 const capture={...exportOnsetCapture(),capturePassed:true,sourceHashes:Object.fromEntries(Object.entries(localHashes).map(([url,served])=>[url,{served}])),
  scope:'Synthetic constant-rate plumbing fixture, not original UI or BANC behavior. Tests capture hooks, state restoration and continuous replay.'};
 globalThis.__flybodyOnset.enabled=false;body.dispose();model.delete();return capture;
}

await fs.mkdir(directory,{recursive:true});
const capture=options['fixture-only']==='true'?await fixture():JSON.parse(await fs.readFile(options.capture||path.join(directory,'capture.json'),'utf8'));
if(options['fixture-only']==='true')await fs.writeFile(path.join(directory,'capture.json'),JSON.stringify(capture)+'\n');
assert(capture.capturePassed&&capture.done&&capture.frames.length,'Capture did not complete successfully');
assert(capture.initial.native.time===0,'Expected a true onset, before any physical integration');
const mismatches=Object.entries(localHashes).flatMap(([url,local])=>{
 const original=capture.sourceHashes[url]?.original??capture.sourceHashes[url]?.served;
 return original===local?[]:[{url,expected:original??'missing from capture',local}];
});
assert.deepEqual(mismatches,[],'Body/runtime source differs from the capture; use the captured source version before replay');
const report={date:new Date().toISOString(),scope:'Sequential native MuJoCo WASM causal replay. Recorded MN rates are held fixed across interventions, so this measures immediate mechanics, not a new closed-loop BANC behavior.',
 captureKind:capture.kind,captureSha256:hash(JSON.stringify(capture)),sceneXmlSha256:hash(capture.scene.xml),
 sourceHashes:localHashes,sourceMatch:true,frames:capture.frames.length,seconds:capture.frames.at(-1).after.native.time,
 baselineGate:{required:true,tolerance:{qpos:1e-9,qvel:1e-7,ctrl:1e-9,muscleState:1e-6}},cases:[]};
const difference=(a,b)=>{assert.equal(a.length,b.length);return b.reduce((max,value,i)=>Math.max(max,Math.abs(a[i]-value)),0);};

function newBody(){
 const habitat=createHabitat(structuredClone(capture.scene.fruit)),model=mj.MjModel.from_xml_string(capture.scene.xml);
 model.hfield_data.set(capture.scene.heights);
 const body=new FlyBodyPhysics(mj,model,capture.scene.metadata,capture.scene.io,n=>new WasmMuscles(core,n),environment(habitat,model,capture.scene.fruitGeomNames));
 restoreOnsetState(body,capture.initial);
 for(const actuator of capture.scene.metadata.actuators.filter(a=>a.name.startsWith('wing_')))assert.equal(model.actuator_gainprm[actuator.id*10],18);
 return {body,model};
}

function fullControlReplay(){
 const {body,model}=newBody(),errors={qpos:0,qvel:0},perFrame=[];
 try{
  for(const row of capture.frames){
   for(const step of row.wingSteps){
    assert(Math.abs(body.data.time-step.time)<1e-9);body.data.ctrl.set(step.ctrl);
    const substeps=Math.round(step.dt/capture.scene.metadata.timestep);assert.equal(substeps,4);
    for(let i=0;i<substeps;i++)mj.mj_step(model,body.data);
   }
   const e={qpos:difference(body.data.qpos,row.after.native.qpos),qvel:difference(body.data.qvel,row.after.native.qvel)};
   for(const key of Object.keys(errors))errors[key]=Math.max(errors[key],e[key]);perFrame.push({time:body.data.time,...e});
  }
  return {kind:'recorded_full_controls',errors,passed:errors.qpos<1e-9&&errors.qvel<1e-7,perFrame};
 }finally{body.dispose();model.delete();}
}

function rateReplay(name){
 const {body,model}=newBody(),powerIds=new Set(capture.scene.io.muscles.filter(m=>m.kind==='asynchronous_wing').flatMap(m=>m.indices));
 const errors={qpos:0,qvel:0,ctrl:0,muscleState:0},samples=[];
 const zeroPower=name!=='baseline',zeroWingControls=name==='dlm_dvm_zero_and_wing_controls_zero';
 if(zeroWingControls){
  const original=body.wings.step;
  body.wings.step=function(...args){const result=original.apply(this,args);for(const a of this.actuators)args[1][a.id]=0;return result;};
 }
 let maxApplied=0,maxRise=-Infinity,minUp=1,maxOmega=0,firstAirborne=null,firstOverturned=null;
 const initialRoot=capture.initial.native.qpos.slice(0,3);
 try{
  for(const row of capture.frames){
   const rates=new Map(row.rates.map(([id,value])=>[id,zeroPower&&powerIds.has(id)?0:value]));
   // This nearest-food object is an exogenous per-step input from BodyWorld.
   // It changes reporting, not the motor decoder; actual food contact and
   // muscle/internal state continue through the replay without corrections.
   body.food=structuredClone(row.before.food);
   body.step(rates,row.duration,row.options);
   const state=onsetState(body),q=state.native.qpos,up=1-2*(q[4]**2+q[5]**2),omega=Math.hypot(...state.native.qvel.slice(3,6));
   const force=Math.max(...state.native.qfrc_applied.map(Math.abs),...state.native.xfrc_applied.map(Math.abs));
   maxApplied=Math.max(maxApplied,force);maxRise=Math.max(maxRise,q[2]-initialRoot[2]);minUp=Math.min(minUp,up);maxOmega=Math.max(maxOmega,omega);
   if(body.airborne&&firstAirborne===null)firstAirborne=body.time;
   if(up<0&&firstOverturned===null)firstOverturned=body.time;
   if(name==='baseline'){
    for(const key of ['qpos','qvel','ctrl'])errors[key]=Math.max(errors[key],difference(state.native[key],row.after.native[key]));
    errors.muscleState=Math.max(errors.muscleState,difference(state.muscleState,row.after.muscleState));
   }
   samples.push({time:body.time,qpos:q,qvel:state.native.qvel,ctrl:state.native.ctrl,wing:state.wing,
    muscleState:state.muscleState,wingDriveLeft:body.wingDriveLeft,wingDriveRight:body.wingDriveRight,
    wingPower:body.wingPower,airborne:body.airborne,environmentContacts:body.environmentContactCount,up,omega,
    internal:state.internal,appliedForceMaximum:force});
  }
  return {name,intervention:zeroWingControls?'Zero only DLM/DVM rates, then set only the six wing actuator controls to zero; passive springs/damping retained':
   zeroPower?'Zero only annotated DLM/DVM MN rates; all other recorded rates, original gain18, steering and restoring servo retained':'All recorded MN rates, original muscles and gain18; no intervention',
   errors:name==='baseline'?errors:null,baselinePassed:name==='baseline'?Object.entries(errors).every(([key,value])=>value<report.baselineGate.tolerance[key]):null,
   metrics:{maximumRootRiseCm:maxRise,minimumUpZ:minUp,maximumAngularSpeedRadPerSecond:maxOmega,firstAirborneSeconds:firstAirborne,
    firstOverturnedSeconds:firstOverturned,maximumAppliedForce:maxApplied,finalRoot:samples.at(-1).qpos.slice(0,7)},samples};
 }finally{body.dispose();model.delete();}
}

report.fullControlParity=fullControlReplay();
if(report.fullControlParity.passed){
 const baseline=rateReplay('baseline');report.cases.push(baseline);report.baselineGate.passed=baseline.baselinePassed;
 if(baseline.baselinePassed){
  report.cases.push(rateReplay('dlm_dvm_zero'));
  if(options['passive-wings']!=='false')report.cases.push(rateReplay('dlm_dvm_zero_and_wing_controls_zero'));
 }
}else report.baselineGate.passed=false;
report.passed=report.baselineGate.passed&&report.cases.every(c=>c.metrics.maximumAppliedForce===0);
report.interpretationAllowed=report.passed;
report.caveats=['Compare baseline to DLM/DVM-zero to test whether power recruitment is necessary for this recorded mechanical onset.',
 'Compare DLM/DVM-zero with its zero-wing-control counterpart to test the remaining active wing servo contribution. This does not identify a biologically correct gain.',
 'Counterfactual body motion would change sensory input in a live brain. That neural response is intentionally not simulated in this paired replay.',
 'A synthetic fixture validates recording/restoration only and is never evidence for the actual BANC launch.'];
await fs.writeFile(path.join(directory,'replay.json'),JSON.stringify(report)+'\n');
console.log(JSON.stringify({passed:report.passed,kind:capture.kind,fullControlParity:report.fullControlParity.errors,
 baselineGate:report.baselineGate,cases:report.cases.map(c=>({name:c.name,errors:c.errors,metrics:c.metrics})),
 output:path.join(directory,'replay.json')},null,2));
if(!report.passed)process.exitCode=1;
