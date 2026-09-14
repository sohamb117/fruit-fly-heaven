import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {FlyBodyWings} from '../flybody-wings.js';
import loadMujoco from '../../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {STEERING_MUSCLE_TYPES,FLIGHT_PARAMETER_NAMES,DEFAULT_FLIGHT_INTERPRETER,
 flightParametersToInterpreter,validateFlightInterpreter} from '../training/flight-parameters.js';

const realBytes=fs.readFileSync(new URL('../../models/flybody-mujoco.json',import.meta.url));
const realMetadata=JSON.parse(realBytes);
const coefficients=()=>structuredClone(DEFAULT_FLIGHT_INTERPRETER);

function fixture(){
 const joints=['left','right'].flatMap((side,s)=>['yaw','roll','pitch'].map((axis,a)=>({name:`wing_${axis}_${side}`,qpos:s*3+a,neutral:0,range:[-2,2]})));
 const actuators=joints.map((j,id)=>({name:j.name,id,range:[-1,1]}));
 const zero=[0,0,0,0,0,0],full=[1,1,1,1,1,1];
 const steering=Object.fromEntries(STEERING_MUSCLE_TYPES.map(name=>[name,[0,0,0,0,0,0]]));
 steering.iii1_muscle=[.1,0,0,0,0,0];steering.b1_muscle=[0,.1,0,0,0,0];
 const metadata={joints,actuators,wing_actuation:{steering,targets:[[zero,zero],[full,full]],powers:[0,1],frequency_hz:200,deployment_tau_s:.01,deployment_before_beating:0}};
 return {wings:new FlyBodyWings(metadata),q:new Float64Array(6),ctrl:new Float64Array(6)};
}

test('flight interpreter preserves independent left/right BANC power',()=>{
 const {wings,q,ctrl}=fixture();wings.step(q,ctrl,1,0,{left:{},right:{}},.05);
 assert.ok(wings.power[0]>.95);assert.equal(wings.power[1],0);assert.ok(ctrl[0]>.9);assert.equal(ctrl[3],0);
});

test('steering muscles perturb the fixed wingbeat basis but cannot gate deployment',()=>{
 const a=fixture(),b=fixture();
 a.wings.step(a.q,a.ctrl,1,1,{left:{iii1_muscle:1},right:{}},.05);
 b.wings.step(b.q,b.ctrl,1,1,{left:{},right:{}},.05);
 assert.ok(a.wings.deployment[0]>.95);assert.ok(a.wings.power[0]>.95);
 assert.notEqual(a.ctrl[0],b.ctrl[0]);
});

test('the three common scales and explicit muscle coefficients stay inside the flight interface',()=>{
 const {wings,q,ctrl}=fixture();
 const profile=coefficients();profile.powerGain=.5;profile.deploymentTauScale=2;profile.frequencyScale=.9;
 profile.steering.b1_muscle={biasGain:.5,amplitudeGain:2};wings.setInterpreterParameters(profile);
 wings.step(q,ctrl,1,0,{left:{},right:{}},.05);
 assert.ok(wings.power[0]<=.5);assert.equal(wings.frequencyHz,180);assert.equal(wings.config.frequency_hz,180);
 assert.deepEqual(wings.controlState().parameters,profile);
 assert.throws(()=>wings.setInterpreterParameters({frequencyScale:0}));
 assert.throws(()=>wings.setInterpreterParameters({unknown:1}));
});

test('the complete log-vector contract has exactly 27 independent named coefficients',()=>{
 assert.equal(STEERING_MUSCLE_TYPES.length,12);assert.equal(FLIGHT_PARAMETER_NAMES.length,27);
 assert.equal(new Set(FLIGHT_PARAMETER_NAMES).size,27);
 assert.deepEqual(new Set(STEERING_MUSCLE_TYPES),new Set(Object.keys(realMetadata.wing_actuation.steering)));
 assert.deepEqual(flightParametersToInterpreter(new Float64Array(27)),DEFAULT_FLIGHT_INTERPRETER);
 const flatten=value=>[value.powerGain,value.deploymentTauScale,value.frequencyScale,
  ...STEERING_MUSCLE_TYPES.flatMap(name=>[value.steering[name].biasGain,value.steering[name].amplitudeGain])];
 for(let i=0;i<27;i++){
  const vector=new Float64Array(27);vector[i]=Math.log(1.5);const physical=flatten(flightParametersToInterpreter(vector));
  physical.forEach((value,j)=>assert.equal(value,i===j?1.5:1,FLIGHT_PARAMETER_NAMES[i]+' changed '+FLIGHT_PARAMETER_NAMES[j]));
 }
 for(const invalid of [[],new Float64Array(26),new Float64Array(28),{},null])assert.throws(()=>flightParametersToInterpreter(invalid),/27/);
 for(const value of [NaN,Infinity,-Infinity,1000,-1000,undefined,'0']){
  const vector=Array(27).fill(0);vector[7]=value;assert.throws(()=>flightParametersToInterpreter(vector),/Invalid/);
 }
});

test('missing, unknown and invalid steering coefficients fail atomically, including old global gains',()=>{
 const {wings}=fixture(),before=wings.controlState();
 const invalid=[];
 let p=coefficients();delete p.steering.b1_muscle;invalid.push(p);
 p=coefficients();delete p.steering.b1_muscle.amplitudeGain;invalid.push(p);
 p=coefficients();p.steering.not_a_muscle={biasGain:1,amplitudeGain:1};invalid.push(p);
 p=coefficients();p.steering.b1_muscle.phase=0;invalid.push(p);
 for(const bad of [NaN,Infinity,-1,0,null,undefined,'1']){
  p=coefficients();p.steering.iv4_muscle.biasGain=bad;invalid.push(p);
 }
 invalid.push({steering:null},{steering:{}},{steeringBiasGain:1},{steeringAmplitudeGain:1},{frequencyScale:2,steering:{}});
 for(const value of invalid){assert.throws(()=>wings.setInterpreterParameters(value));assert.deepEqual(wings.controlState(),before);}
 for(const value of [null,[],new Float64Array(27),1])assert.throws(()=>validateFlightInterpreter(value));
});

test('each instance owns its coefficients, and exported state cannot mutate the interpreter',()=>{
 const a=fixture(),b=fixture(),profile=coefficients();profile.steering.b1_muscle.biasGain=1.5;
 a.wings.setInterpreterParameters(profile);profile.steering.b1_muscle.biasGain=9;
 assert.equal(a.wings.interpreter.steering.b1_muscle.biasGain,1.5);
 assert.equal(b.wings.interpreter.steering.b1_muscle.biasGain,1);
 const exported=a.wings.controlState();exported.parameters.steering.b1_muscle.biasGain=8;
 assert.equal(a.wings.interpreter.steering.b1_muscle.biasGain,1.5);
 a.wings.setInterpreterParameters({frequencyScale:.9});
 assert.equal(a.wings.interpreter.steering.b1_muscle.biasGain,1.5);
 assert(Object.isFrozen(a.wings.interpreter.steering.b1_muscle));
 assert.equal(DEFAULT_FLIGHT_INTERPRETER.steering.b1_muscle.biasGain,1);
});

function realFixture(phase=1.17){
 const wings=new FlyBodyWings(realMetadata),q=new Float64Array(1+Math.max(...realMetadata.joints.map(j=>j.qpos))),
  ctrl=new Float64Array(1+Math.max(...realMetadata.actuators.map(a=>a.id)));
 wings.deployment.fill(1);wings.phase=phase;
 // Place wing joints on the zero-steering native target so modest response
 // differences reach actuator controls instead of being hidden by saturation.
 wings.step(q,ctrl,.6,.6,{left:{},right:{}},realMetadata.timestep*4);
 for(let k=0;k<6;k++)q[wings.joints[k].qpos]=wings.target[k];
 wings.phase=phase;ctrl.fill(0);
 return {wings,q,ctrl};
}

for(const muscle of STEERING_MUSCLE_TYPES)for(const gain of ['biasGain','amplitudeGain']){
 test(`${muscle} ${gain} independently changes native-basis targets and controls`,()=>{
  let greatestTargetChange=0,greatestControlChange=0;
  for(const phase of [.37,1.17,2.31]){
   const base=realFixture(phase),changed=realFixture(phase),profile=coefficients();
   profile.steering[muscle][gain]=1.4;changed.wings.setInterpreterParameters(profile);
   const force=.5,drive={left:{[muscle]:force},right:{}};
   for(const f of [base,changed])f.wings.step(f.q,f.ctrl,.6,.6,drive,realMetadata.timestep*4);
   for(let k=0;k<3;k++){
    greatestTargetChange=Math.max(greatestTargetChange,Math.abs(changed.wings.target[k]-base.wings.target[k]));
    const id=changed.wings.actuators[k].id;
    greatestControlChange=Math.max(greatestControlChange,Math.abs(changed.ctrl[id]-base.ctrl[id]));
    assert.equal(changed.wings.target[k+3],base.wings.target[k+3], 'Inactive opposite-side muscles cannot acquire drive');
    assert.equal(changed.ctrl[changed.wings.actuators[k+3].id],base.ctrl[base.wings.actuators[k+3].id]);
   }
   for(let k=0;k<6;k++){
    const scaled=gain==='biasGain'?k<3:k>=3;
    assert.equal(changed.wings.residuals[0][k],force*realMetadata.wing_actuation.steering[muscle][k]*(scaled?1.4:1));
    assert.equal(changed.wings.residuals[1][k],0);
    if(!scaled)assert.equal(changed.wings.residuals[0][k],base.wings.residuals[0][k]);
   }
   assert.deepEqual(changed.wings.power,base.wings.power);assert.deepEqual(changed.wings.deployment,base.wings.deployment);
   assert.equal(changed.wings.phase,base.wings.phase);
  }
  assert(greatestTargetChange>1e-6,`No native target sensitivity for ${muscle}.${gain}`);
  assert(greatestControlChange>1e-6,`No native control sensitivity for ${muscle}.${gain}`);
 });
}

test('tying bilateral coefficients preserves unequal left/right muscle forces and leaves other types unchanged',()=>{
 const a=realFixture(),b=realFixture(),profile=coefficients();profile.steering.b2_muscle={biasGain:1.3,amplitudeGain:.7};
 a.wings.setInterpreterParameters(profile);
 const drive={left:{b2_muscle:.2},right:{b2_muscle:.6}};
 a.wings.step(a.q,a.ctrl,.6,.6,drive,realMetadata.timestep*4);
 for(let k=0;k<6;k++){
  const g=k<3?1.3:.7,basis=realMetadata.wing_actuation.steering.b2_muscle[k];
  assert.equal(a.wings.residuals[0][k],.2*basis*g);
  assert.equal(a.wings.residuals[1][k],.6*basis*g);
 }
 const changedInactive=realFixture();changedInactive.wings.setInterpreterParameters(profile);
 const another={left:{b1_muscle:.4},right:{b1_muscle:.7}};
 for(const f of [b,changedInactive])f.wings.step(f.q,f.ctrl,.6,.6,another,realMetadata.timestep*4);
 assert.deepEqual(changedInactive.wings.target,b.wings.target);assert.deepEqual(changedInactive.ctrl,b.ctrl);
});

test('all-one 27-parameter defaults exactly reproduce the prechange 2048-step native-metadata replay',()=>{
 // Captured from commit 7507c7c before replacing the five-scale interpreter.
 // This is command/target parity, not proof of native free-flight behavior.
 assert.equal(createHash('sha256').update(realBytes).digest('hex'),'c1348adf19decee75abbbb1d2be8a5ef6ac8cfe4fe4850083a502b62877c4a48');
 const w=new FlyBodyWings(realMetadata),q=new Float64Array(1+Math.max(...realMetadata.joints.map(j=>j.qpos))),
  ctrl=new Float64Array(1+Math.max(...realMetadata.actuators.map(a=>a.id))),names=Object.keys(realMetadata.wing_actuation.steering),hash=createHash('sha256');
 w.setInterpreterParameters(flightParametersToInterpreter(new Float64Array(27)));
 for(let i=0;i<2048;i++){
  for(let k=0;k<6;k++)q[w.joints[k].qpos]=w.joints[k].neutral+.35*Math.sin(i*.011+k*.7);
  const steering={left:{},right:{}};
  for(let k=0;k<names.length;k++){
   steering.left[names[k]]=.5+.45*Math.sin(i*.007+k*.6);
   steering.right[names[k]]=.5+.45*Math.cos(i*.005+k*.4);
  }
  w.step(q,ctrl,.03+.96*(.5+.5*Math.sin(i*.003)),.03+.96*(.5+.5*Math.cos(i*.002)),steering,realMetadata.timestep*4);
  const row=new Float64Array([...w.target,...ctrl,...w.power,...w.deployment,w.phase]);hash.update(Buffer.from(row.buffer));
 }
 assert.equal(hash.digest('hex'),'6d55618caaef1d812961d26383eb7d07dbed0a408c570ca82b88523e58eb583e');
});

const referenceValues=()=>Object.fromEntries(STEERING_MUSCLE_TYPES.map((name,i)=>[name,.25+i*.025]));
const referenceMetadata=reference=>({...realMetadata,wing_actuation:{...realMetadata.wing_actuation,steering_force_reference:reference}});

test('force references are complete, bounded, immutable local priors and add no learned parameters',()=>{
 const source=referenceValues(),metadata=referenceMetadata(source),wings=new FlyBodyWings(metadata);
 const original=source.b1_muscle;source.b1_muscle=.9;
 assert.equal(wings.steeringForceReference.b1_muscle,original);
 assert(Object.isFrozen(wings.steeringForceReference));
 assert.equal(wings.config.steering_force_reference,wings.steeringForceReference);
 assert.throws(()=>{wings.steeringForceReference.b1_muscle=1;},TypeError);
 assert.equal(FLIGHT_PARAMETER_NAMES.length,27);
 const valid=referenceValues(),before=structuredClone(valid),control=wings.controlState();
 const missing=referenceValues();delete missing.i1_muscle;
 const invalid=[null,undefined,[],{},false,missing,{...valid,other:0},Object.assign({...valid},{[Symbol('other')]:0})];
 for(const value of [NaN,Infinity,-Infinity,-.01,1.01,'0',null,undefined])invalid.push({...valid,b1_muscle:value});
 for(const candidate of invalid){
  assert.throws(()=>new FlyBodyWings(referenceMetadata(candidate)),/force reference/);
  assert.deepEqual(valid,before);assert.deepEqual(wings.controlState(),control);
 }
 for(const boundary of [0,1])assert.doesNotThrow(()=>new FlyBodyWings(referenceMetadata(
  Object.fromEntries(STEERING_MUSCLE_TYPES.map(name=>[name,boundary])))));
});

test('missing force means zero physical force, including withdrawal below the declared reference',()=>{
 const refs=referenceValues(),missing=new FlyBodyWings(referenceMetadata(refs)),explicit=new FlyBodyWings(referenceMetadata(refs));
 const q=new Float64Array(realMetadata.nq??57),a=new Float64Array(56),b=new Float64Array(56);
 for(const wings of [missing,explicit])wings.deployment.fill(1);
 missing.step(q,a,.6,.6,{},.0002);
 const zeros=Object.fromEntries(STEERING_MUSCLE_TYPES.map(name=>[name,0]));
 explicit.step(q,b,.6,.6,{left:zeros,right:zeros},.0002);
 assert.deepEqual(missing.residuals,explicit.residuals);assert.deepEqual(a,b);
 assert(missing.residuals[0].some(value=>Math.abs(value)>1e-6));
 for(const side of missing.residuals)for(let axis=0;axis<6;axis++){
  const expected=STEERING_MUSCLE_TYPES.reduce((sum,name)=>sum-refs[name]*realMetadata.wing_actuation.steering[name][axis],0);
  assert(Math.abs(side[axis]-expected)<1e-15);
 }
 const off=new Float64Array(56);missing.step(q,off,0,0,{},.0002);
 assert.deepEqual(Array.from(missing.power),[0,0]);
});

let nativeMujoco;
async function nativeModel(){
 nativeMujoco??=loadMujoco();const mj=await nativeMujoco;
 return {mj,model:mj.MjModel.from_xml_string(fs.readFileSync(new URL('../../models/flybody-mujoco.xml',import.meta.url),'utf8'))};
}
function nativeRig(mj,model,reference,phase=.37){
 const data=new mj.MjData(model),wings=new FlyBodyWings(reference?referenceMetadata(reference):realMetadata);
 for(const j of realMetadata.joints)data.qpos[j.qpos]=j.neutral;
 data.qpos[2]=3;wings.deployment.fill(1);wings.phase=phase;
 const drive=reference?{left:{...reference},right:{...reference}}:{left:{},right:{}};
 wings.step(data.qpos,data.ctrl,.6,.6,drive,.0002);
 for(let k=0;k<6;k++)data.qpos[wings.joints[k].qpos]=wings.target[k];
 data.ctrl.fill(0);wings.phase=phase;mj.mj_forward(model,data);
 return {data,wings,drive};
}
const nativeBytes=array=>Buffer.from(array.buffer,array.byteOffset,array.byteLength);

test('declared force reference reproduces the zero-residual native trajectory byte for byte with independent gains',async()=>{
 const {mj,model}=await nativeModel(),reference=referenceValues(),plain=nativeRig(mj,model),centered=nativeRig(mj,model,reference);
 try{
  const gains=coefficients();
  for(const [i,name]of STEERING_MUSCLE_TYPES.entries())gains.steering[name]={biasGain:.5+i*.07,amplitudeGain:1.3-i*.03};
  centered.wings.setInterpreterParameters(gains);
  for(let block=0;block<64;block++){
   for(const rig of [plain,centered]){
    rig.wings.step(rig.data.qpos,rig.data.ctrl,.6,.6,rig.drive,.0002);
    for(let k=0;k<4;k++)mj.mj_step(model,rig.data);
   }
   for(const field of ['qpos','qvel','act','ctrl'])
    assert.deepEqual(nativeBytes(centered.data[field]),nativeBytes(plain.data[field]),`${field} block ${block}`);
   assert(centered.wings.residuals.every(values=>values.every(value=>value===0)));
  }
 }finally{plain.data.delete();centered.data.delete();model.delete();}
});

test('centered per-muscle perturbations preserve opposite native response signs, slopes and side independence',async()=>{
 const {mj,model}=await nativeModel(),reference=referenceValues();
 try{
  for(const muscle of STEERING_MUSCLE_TYPES)for(const side of ['left','right']){
   let response=0,rootResponse=0;
   for(const phase of [.37,1.17,2.31]){
    const rigs=[0,.1,-.1].map(delta=>{
     const rig=nativeRig(mj,model,reference,phase),gains=coefficients();
     gains.steering[muscle]={biasGain:1.3,amplitudeGain:.8};rig.wings.setInterpreterParameters(gains);
     rig.drive[side][muscle]+=delta;rig.wings.step(rig.data.qpos,rig.data.ctrl,.6,.6,rig.drive,.0002);
     mj.mj_forward(model,rig.data);return rig;
    });
    try{
     const [base,positive,negative]=rigs;
     for(let k=0;k<6;k++){
      const actuator=base.wings.actuators[k].id,active=k<3?side==='left':side==='right';
      const plus=positive.data.actuator_force[actuator]-base.data.actuator_force[actuator];
      const minus=negative.data.actuator_force[actuator]-base.data.actuator_force[actuator];
      const controlDelta=positive.data.ctrl[actuator]-base.data.ctrl[actuator];
      if(active){
       assert(Math.abs(plus+minus)<1e-10,`${muscle} ${side} lost symmetric native slope`);
       response=Math.max(response,Math.abs(plus));
       if(Math.abs(controlDelta)>1e-10)assert(plus*controlDelta>0,'Native actuator response reversed its control sign');
      }else{
       assert.equal(plus,0);assert.equal(minus,0);
       assert.equal(positive.wings.target[k],base.wings.target[k]);assert.equal(negative.wings.target[k],base.wings.target[k]);
      }
     }
     for(let axis=0;axis<6;axis++)rootResponse=Math.max(rootResponse,Math.abs(positive.data.qacc[axis]-base.data.qacc[axis]));
    }finally{for(const rig of rigs)rig.data.delete();}
   }
   assert(response>1e-5,`${muscle} ${side} had no native actuator response`);
   assert(rootResponse>1e-5,`${muscle} ${side} had no native root reaction`);
  }
 }finally{model.delete();}
});
