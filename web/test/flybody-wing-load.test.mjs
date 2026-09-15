// Pure data and mechanical-clock fixtures: no MuJoCo or neural simulation.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createWingLoadSampler} from '../flybody-wing-load.js';
import {FlyBodyPhysics} from '../flybody-physics.js';
import {createMotorExcitation} from '../flybody-motor-excitation.js';

const close=(a,b,tolerance=1e-11)=>assert(Math.abs(a-b)<=tolerance,`${a} != ${b}`);
const names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
const defaultJoints=names.map((name,k)=>({name,id:[5,2,6,4,1,3][k],dof:[10,7,11,9,6,8][k],body:k<3?2:3}));
function nativeDataFixture(joints=defaultJoints){
  const njnt=Math.max(...joints.map(j=>j.id))+1,nv=Math.max(...joints.map(j=>j.dof))+1,nbody=Math.max(...joints.map(j=>j.body))+1;
  const model={njnt,nv,nbody,opt:{timestep:.00005,integrator:0},jnt_type:new Int32Array(njnt),
    jnt_bodyid:new Int32Array(njnt),jnt_dofadr:new Int32Array(njnt),body_parentid:new Int32Array(nbody).fill(1),body_jntnum:new Int32Array(nbody)};
  model.body_parentid[0]=0;model.body_parentid[1]=0;
  for(const j of joints){model.jnt_type[j.id]=3;model.jnt_bodyid[j.id]=j.body;model.jnt_dofadr[j.id]=j.dof;model.body_jntnum[j.body]=3;}
  const mj={mjtObj:{mjOBJ_JOINT:{value:3}},mj_name2id(_model,_kind,name){return joints.find(j=>j.name===name)?.id??-1;}};
  const arrays={xaxis:new Float64Array(njnt*3),xanchor:new Float64Array(njnt*3),qfrc_fluid:new Float64Array(nv)};
  const data={time:0},reads={xaxis:0,xanchor:0,qfrc_fluid:0};
  for(const key of Object.keys(arrays))Object.defineProperty(data,key,{get(){reads[key]++;return arrays[key];}});
  const setObservation=(side,axes,anchors,tau)=>{
    for(let k=0;k<3;k++){const j=joints[side*3+k];arrays.xaxis.set(axes[k],j.id*3);arrays.xanchor.set(anchors[k],j.id*3);arrays.qfrc_fluid[j.dof]=tau[k];}
  };
  const setKnown=(time=0)=>{
    const theta=time*100,axes=[[Math.cos(theta),Math.sin(theta),0],[-Math.sin(theta),Math.cos(theta),0],[0,0,1]];
    for(let side=0;side<2;side++){
      const moment=side?[4,5-time*100,6]:[1+time*1000,2,3],anchor=[side?.1:-.1,.2,.3];
      setObservation(side,axes,[anchor,anchor,anchor],axes.map(axis=>axis.reduce((s,x,k)=>s+x*moment[k],0)));
    }
  };
  setKnown();return {model,mj,data,arrays,reads,joints,setObservation,setKnown};
}

test('resolves actual IDs/DOFs; world moments and source times are owned observations',()=>{
  const f=nativeDataFixture(),s=createWingLoadSampler(f);assert.equal(s.read(),null);
  const before=Object.fromEntries(Object.entries(f.arrays).map(([k,v])=>[k,v.slice()]));
  s.capture(f.data,0);const sample=s.read();
  assert.deepEqual(sample.momentWorld,{left:[1,2,3],right:[4,5,6]});
  assert.equal(sample.kind,'native-wing-aerodynamic-moment-v1');assert.equal(sample.units,'g cm^2/s^2');
  close(sample.left,Math.sqrt(14));close(sample.right,Math.sqrt(77));
  close(sample.diagnostics.left.conditionUpperBound,3);
  for(const [key,array]of Object.entries(before))assert.deepEqual(f.arrays[key],array,'observation mutated native '+key);
  sample.momentWorld.left.fill(99);sample.diagnostics.left.conditionUpperBound=0;
  f.arrays.xaxis.fill(NaN);f.arrays.qfrc_fluid.fill(NaN);
  assert.deepEqual(s.read().momentWorld.left,[1,2,3]);close(s.read().diagnostics.left.conditionUpperBound,3);
});

const nativeEvidence=new URL('../../reports/flight-tegula-inputs/load-observability/result.json',import.meta.url);
test('optional thorax frame uses the matching force cache and is invariant under world rotation',()=>{
 const f=nativeDataFixture(),r=[0,-1,0,1,0,0,0,0,1];
 f.data.xmat=new Float64Array(f.model.nbody*9);f.data.xmat.set(r,9);
 const axes=[[0,1,0],[-1,0,0],[0,0,1]];
 for(let side=0;side<2;side++)f.setObservation(side,axes,[[0,0,0],[0,0,0],[0,0,0]],[1,2,3]);
 const sampler=createWingLoadSampler({...f,localFrame:true});sampler.capture(f.data,0);
 const sample=sampler.read();assert.deepEqual(sample.momentWorld.left,[-2,1,3]);
 assert.deepEqual(sample.momentThorax.left,[1,2,3]);assert.equal(sample.localFrameTimeSeconds,0);
 sample.momentThorax.left[0]=99;assert.equal(sampler.read().momentThorax.left[0],1);
 f.data.xmat.fill(0);assert.throws(()=>sampler.capture(f.data,0),/rotation/);
 assert.equal(sampler.read().momentThorax.left[0],1);
});
test('all112 saved native observations match the independent diagnostic reconstruction',
  {skip:fs.existsSync(nativeEvidence)?false:'Local native assay artifact is not present'},()=>{
  // Existing native evidence only; this test never loads or steps MuJoCo.
  const r=JSON.parse(fs.readFileSync(nativeEvidence));assert.equal(r.completed,true);assert.equal(r.explicitDiagnosticForwardEvaluations,112);
  const f=nativeDataFixture(r.resolvedJoints),s=createWingLoadSampler(f);let checked=0;
  for(const state of r.states)for(const observation of state.evaluations){
    f.data.time=state.nativeTimeSeconds;
    observation.wings.forEach((wing,side)=>f.setObservation(side,wing.axesWorld,wing.anchorsWorldCm,wing.fluidGeneralizedTorques));
    s.capture(f.data,f.data.time);const actual=s.read();
    for(const [side,k]of [['left',0],['right',1]]){
      close(actual[side],observation.wings[k].magnitude,1e-12);
      actual.momentWorld[side].forEach((x,j)=>close(x,observation.wings[k].momentWorld[j],1e-12));
      assert(actual.diagnostics[side].conditionUpperBound+1e-12>=observation.wings[k].condition);
    }
    checked++;
  }
  assert.equal(checked,112);
});

test('malformed topology, geometry, loads and clocks reject without replacing the last good sample',()=>{
  for(const mutate of [f=>f.model.opt.integrator=1,f=>f.model.jnt_type[5]=2,f=>f.model.body_jntnum[2]=4,
    f=>f.model.body_parentid[3]=2,f=>f.model.jnt_dofadr[5]=7,f=>f.mj.mj_name2id=()=>-1]){
    const f=nativeDataFixture();mutate(f);assert.throws(()=>createWingLoadSampler(f));
  }
  for(const mutate of [f=>f.arrays.xanchor[4*3]+=1e-7,f=>f.arrays.xaxis[4*3]*=2,
    f=>f.arrays.xaxis.set(f.arrays.xaxis.slice(4*3,4*3+3),1*3),f=>f.arrays.qfrc_fluid[9]=NaN]){
    const f=nativeDataFixture(),s=createWingLoadSampler(f);s.capture(f.data,0);const before=s.read();
    f.data.time=.001;mutate(f);assert.throws(()=>s.capture(f.data,.00095));assert.deepEqual(s.read(),before);
  }
  const f=nativeDataFixture(),s=createWingLoadSampler(f);s.capture(f.data,0);f.data.time=.002;
  for(const time of [NaN,Infinity,-1,.003,.001])assert.throws(()=>s.capture(f.data,time));
  s.capture(f.data,.00195);assert.throws(()=>{f.data.time=.001;s.capture(f.data,.00095);},/regressed/);
});

function bodyFixture(){
  const f=nativeDataFixture(),b=Object.create(FlyBodyPhysics.prototype),commands=[],refreshSamples=[],calls={native:0,muscle:0};
  Object.assign(f.data,{qpos:new Float64Array(15),qvel:new Float64Array(14),ctrl:new Float64Array(6)});f.data.qpos[3]=1;
  const byJoint=new Map(['rostrum','haustellum'].map((name,k)=>[name,{qpos:13+k,dof:12+k,neutral:0,range:[-1,1]}]));
  f.mj.mj_step=(_model,data)=>{f.setKnown(data.time);data.time+=.00005;calls.native++;};
  Object.assign(b,{mj:f.mj,model:f.model,data:f.data,metadata:{timestep:.00005},_wingLoadFeedback:null,_wingMotorEvents:null,
    time:0,remainder:0,mappings:[{joint:'pump',kind:'pump',sign:1,indices:[100]}],activation:new Float32Array(1),
    muscleState:new Float32Array(3),input:new Float32Array(5),muscleDirections:new Float64Array([1]),byJoint,
    actuators:new Map([['rostrum',{id:0,range:[-1,1]}],['haustellum',{id:1,range:[-1,1]}]]),restPose:f.data.qpos.slice(),
    motorExcitation:createMotorExcitation(),proboscisDecoder:{read(){return {rostrumExtend:0,rostrumRetract:0,haustellumExtend:0,haustellumRetract:0};}},
    muscles:{step(input){calls.muscle++;return Float32Array.from([input[0],0,input[0]*input[4]]);}},
    wings:{phase:0,power:[0,0],step(_q,ctrl,left,right,steering,dt){commands.push({ctrl:Array.from(ctrl),left,right,steering:structuredClone(steering),dt});this.phase+=dt;}},
    internal:{energy:1,crop:0,step(){}},heading:0,x:0,y:0,z:0,proboscis:0,contactFood:null,smell:()=>0,monitor:{step(){}},
    refresh(){this.time=this.data.time;refreshSamples.push(this.readWingLoadFeedback());
      // Simulate invalidation/heap replacement after the observer boundary.
      for(const array of Object.values(f.arrays))array.fill(NaN);},
  });
  return {...f,body:b,commands,refreshSamples,calls};
}

test('body opt-in captures the initial forward and final Euler cache before each1ms refresh, leaving outputs unchanged',()=>{
  const plain=bodyFixture(),observed=bodyFixture();
  const initial=observed.body.enableWingLoadFeedback();assert.equal(initial.forceTimeSeconds,0);assert.equal(initial.bodyTimeSeconds,0);
  plain.body.step(new Map([[100,40]]),.002);observed.body.step(new Map([[100,40]]),.002);
  assert.deepEqual(plain.commands,observed.commands);assert.deepEqual(plain.calls,observed.calls);
  for(const key of ['qpos','qvel','ctrl'])assert.deepEqual(plain.data[key],observed.data[key]);
  for(const key of ['activation','input','muscleState'])assert.deepEqual(plain.body[key],observed.body[key]);
  assert.equal(plain.body.time,observed.body.time);assert.equal(plain.body.remainder,observed.body.remainder);
  assert.equal(plain.body.readWingLoadFeedback(),null);assert.deepEqual(plain.reads,{xaxis:0,xanchor:0,qfrc_fluid:0});
  assert.deepEqual(observed.reads,{xaxis:3,xanchor:3,qfrc_fluid:3});
  for(const [i,sample]of observed.refreshSamples.entries()){
    close(sample.bodyTimeSeconds,(i+1)*.001);close(sample.forceTimeSeconds,(i+1)*.001-.00005);
    close(sample.momentWorld.left[0],1+sample.forceTimeSeconds*1000);
    close(sample.bodyTimeSeconds-sample.forceTimeSeconds,.00005);
  }
  assert.deepEqual(observed.body.readWingLoadFeedback(),observed.refreshSamples.at(-1));
});

test('body enabling is fresh-only, rejects unsupported initial caches transactionally, and does no native work',()=>{
  const f=bodyFixture();
  for(const [object,key,value]of [[f.body,'time',.001],[f.data,'time',.001],[f.body,'remainder',.0005],[f.body,'_disposed',true]]){
    const old=object[key];object[key]=value;assert.throws(()=>f.body.enableWingLoadFeedback());object[key]=old;
    assert.equal(f.body._wingLoadFeedback,null);assert.equal(f.calls.native,0);
  }
  f.arrays.qfrc_fluid[9]=NaN;assert.throws(()=>f.body.enableWingLoadFeedback());assert.equal(f.body._wingLoadFeedback,null);
  f.setKnown();f.body.enableWingLoadFeedback();assert.equal(f.calls.native,0);
  assert.throws(()=>f.body.enableWingLoadFeedback(),/already enabled/);
  f.body.data.time=.002;assert.throws(()=>f.body.place(0,0,0),/running wing load/);
});

test('fresh placement recaptures its final forward while running placement rejects before mutation',()=>{
  const f=bodyFixture(),b=f.body;let forwards=0,resets=0;
  b.metadata.feet=[0];b.metadata.initializeStance=false;b.surface=()=>0;
  b.data.site_xpos=new Float64Array(3);b.monitor.resetContinuity=()=>{resets++;};
  b.mj.mj_forward=()=>{forwards++;f.setKnown(b.data.qpos[0]*.001);};
  b.enableWingLoadFeedback();b.place(2,0,0);
  assert.equal(forwards,2);assert.equal(resets,1);assert.equal(b.readWingLoadFeedback().forceTimeSeconds,0);
  close(b.readWingLoadFeedback().momentWorld.left[0],3);
  b.data.time=.002;b.time=.002;const q=b.data.qpos.slice(),before=b.readWingLoadFeedback();
  assert.throws(()=>b.place(9,0,0),/running wing load/);
  assert.equal(forwards,2);assert.equal(resets,1);assert.deepEqual(b.data.qpos,q);assert.deepEqual(b.readWingLoadFeedback(),before);
});

// Import the actual world source with only external/environment dependencies
// stubbed. Its copyPose implementation is unchanged and executed below.
const stubs={
  './flight-scene-profile.js':'export const matchMaintainedScene=()=>null;',
  './body-world.js':'export class BodyWorld{};export const jointPose=()=>({}),senseBody=()=>({legs:[]}),decodeMotorOutput=()=>({antennaLeft:0,antennaRight:0});',
  './flybody-physics.js':'export class FlyBodyPhysics{};export const flybodyScene=()=>{};',
  '/banc-engine/src/index.js':'export class WasmMuscles{};export const createWasmCore=()=>{throw Error("not used")};',
  '/body-engine/mujoco.js':'export default ()=>{throw Error("not used")};',
  './flybody-mouth-pose.js':'export const createMouthLandmarks=()=>{},sampleMouthPose=()=>null;',
  './flybody-wing-pose.js':'export const createWingLandmarks=()=>{},sampleWingPose=()=>null;',
  './flybody-contact-environment.js':'export const createContactFoodResolver=()=>{};',
};
const worldSource=fs.readFileSync(new URL('../flybody-world.js',import.meta.url),'utf8').replace(/from '([^']+)'/g,(text,name)=>{
  assert(Object.hasOwn(stubs,name),'Unexpected world dependency '+name);
  return `from 'data:text/javascript;base64,${Buffer.from(stubs[name]).toString('base64')}'`;
});
const {FlyBodyWorld}=await import('data:text/javascript;base64,'+Buffer.from(worldSource).toString('base64'));
test('actual copyPose exports owned wing loads only under opt-in with matching body time',()=>{
  const f=bodyFixture(),b=f.body,world=Object.create(FlyBodyWorld.prototype);
  Object.assign(world,{metadata:{leg_bodies:[]},habitat:{surface(){return {y:0,normal:[0,1,0]};}},backend:'fixture'});
  Object.assign(b,{quaternion:[1,0,0,0],restHeight:0,vx:0,vy:0,vz:0,byJoint:{get(){return {qpos:0,dof:0,neutral:0};}},
    monitor:{events:[],flightEvidence:[],landingCount:0},wings:{config:{frequency_hz:235.8}},wingPower:0,wingPhase:0,
    halterePower:[0,0],haltereSteering:{left:{},right:{}},legFoodContact:[],wingFoodContact:[],mouthFoodContact:[]});
  const plain={brain:{},antennaPhase:0};world.copyPose(plain,b,0);assert(!Object.hasOwn(plain.feedback,'wingLoad'));
  b.enableWingLoadFeedback();const enabled={brain:{},antennaPhase:0};world.copyPose(enabled,b,0);
  assert.equal(enabled.feedback.wingLoad.bodyTimeSeconds,enabled.bodyTime);assert.equal(enabled.feedback.wingLoad.forceTimeSeconds,0);
  const copy={...enabled,feedback:{...enabled.feedback}};delete copy.feedback.wingLoad;assert.deepEqual(copy,plain);
  enabled.feedback.wingLoad.momentWorld.left[0]=123;assert.equal(b.readWingLoadFeedback().momentWorld.left[0],1);
});
