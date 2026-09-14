// DIAGNOSTIC ONLY. Frozen classical controller through native and event output paths.
// No BANC execution, fitting, production changes, or fresh response calibration.
// node scripts/diagnose-flight-event-control.mjs --run --output=reports/flight-event-control
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {createMotorExcitation} from '../web/flybody-motor-excitation.js';
import {createWingEventExcitation,DEFAULT_WING_EVENT_PRIORS} from '../web/flybody-wing-event-excitation.js';
import {measureFlightKinematics} from '../web/training/flight-observation.js';
import {worldComWrench,readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),run=args.includes('--run');
assert(args.every(arg=>arg==='--run'||arg==='--check-only'||/^--(output|reference)=/.test(arg)),'Unknown argument');
assert(!(run&&args.includes('--check-only')),'Choose --run or --check-only');
const option=(key,fallback)=>args.find(arg=>arg.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const reports=path.join(root,'reports'),output=path.resolve(root,option('output','reports/flight-event-control'));
const referencePath=path.resolve(root,option('reference','reports/flight-classical-control/result.json'));
assert(output.startsWith(reports+path.sep)&&referencePath.startsWith(reports+path.sep),'Reports paths required');
assert((await fs.realpath(referencePath)).startsWith((await fs.realpath(reports))+path.sep),'Reference symlink escapes reports');
const sha=value=>createHash('sha256').update(value).digest('hex'),clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const referenceBytes=await fs.readFile(referencePath),reference=JSON.parse(referenceBytes);
assert(reference.completed&&reference.sourceUnchanged&&reference.pairedInitialStateIdentical,'Incomplete or unpaired reference');
const historical=reference.cases.find(row=>row.mode==='closed_loop');
assert(historical?.meetsDeclaredPositiveControl,'Reference needs a successful direct-force closed loop');
const reducedPath=path.join(path.dirname(referencePath),'fixed-nonwing.xml'),reducedBytes=await fs.readFile(reducedPath);
assert.equal(sha(reducedBytes),reference.modelHash,'Historical reduced plant changed');
const historicalScript=path.join(path.dirname(referencePath),'diagnostic-source.used.mjs');
assert.equal(sha(await fs.readFile(historicalScript)),reference.sourceHashes['scripts/diagnose-flight-classical-control.mjs']);
const files=['scripts/diagnose-flight-event-control.mjs','scripts/motor-wing-calibration-helpers.mjs',
 'web/flybody-wings.js','web/training/flight-parameters.js','web/training/flight-observation.js','web/flybody-motor-excitation.js','web/flybody-wing-event-excitation.js',
 'models/flybody-mujoco.xml','models/flybody-mujoco.json','models/flybody-wing-actuation.json','data/prepared/banc888/io.json',
 'packages/banc-runtime/src/wasm.js','packages/banc-runtime/src/model.js','packages/banc-runtime/native/core.cpp',
 'packages/banc-runtime/dist/core.js','packages/banc-runtime/dist/core.wasm',
 'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js','packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
const buffers=await Promise.all(files.map(file=>fs.readFile(path.join(root,file))));
const bytes=Object.fromEntries(files.map((file,i)=>[file,buffers[i]])),hashes=Object.fromEntries(files.map(file=>[file,sha(bytes[file])]));
for(const [file,expected]of Object.entries(reference.sourceHashes)){
 if(file==='scripts/diagnose-flight-classical-control.mjs'||file==='web/flybody-wings.js')continue;
 assert.equal(hashes[file],expected,'Reference dependency changed: '+file);
}
// The optional force-reference branch was added after the historical assay.
// Canonical metadata must retain its old behavior, then direct trajectory
// equality below gates the change before either delayed arm may execute.
const metadata=JSON.parse(bytes['models/flybody-mujoco.json']),io=JSON.parse(bytes['data/prepared/banc888/io.json']);
assert.equal(sha(bytes['models/flybody-mujoco.xml']),metadata.xml_sha256);
assert.deepEqual(metadata.wing_actuation,JSON.parse(bytes['models/flybody-wing-actuation.json']));
assert(!Object.hasOwn(metadata.wing_actuation,'steering_force_reference'),'Reference uses absolute-force legacy basis');
assert(!Object.hasOwn(metadata,'motor_excitation'),'This latency isolation uses canonical legacy recruitment');
const mappings=io.muscles.flatMap((mapping,mappingIndex)=>['wing_steering_assumption','asynchronous_wing'].includes(mapping.kind)?[{...mapping,mappingIndex}]:[]);
assert.equal(mappings.length,28);
const muscleNames=Object.keys(metadata.wing_actuation.steering);
const controlNames=['left','right'].flatMap(side=>muscleNames.map(name=>`${side}:${name}`)).concat('common_power');
assert.deepEqual(controlNames,reference.controlNames);
const mappingControls=mappings.map(mapping=>{
 assert(!metadata.joints.some(joint=>joint.name===mapping.joint),'Frozen unit length/velocity requires a virtual wing muscle');
 const side=mapping.joint.endsWith('_left')?'left':mapping.joint.endsWith('_right')?'right':null;assert(side);
 if(mapping.kind==='asynchronous_wing')return 24;
 const index=controlNames.indexOf(`${side}:${mapping.target}`);assert(index>=0&&index<24);return index;
});
for(let k=0;k<24;k++)assert.equal(mappingControls.filter(index=>index===k).length,1,'Missing/duplicate steering mapping');
for(const side of ['left','right'])assert.equal(mappings.filter(m=>m.kind==='asynchronous_wing'&&m.joint.endsWith('_'+side)).length,2);
const trim=reference.trimControls,B=reference.responseJacobian,trimY=reference.trimResponse;
assert.equal(trim.length,25);assert(B.length===4&&B.every(row=>row.length===25&&row.every(Number.isFinite)));
const lower=reference.assumptions.lower,upper=reference.assumptions.upper,inertia=reference.conventions.inertia;
const {naturalFrequencyRadPerSecond:naturalFrequency,dampingRatio,angularRateFilterTauSeconds:filterTau,heightKp,heightKd}=reference.assumptions.gains;
assert.deepEqual([filterTau,heightKp,heightKd],[.006,16,8]);
const warmup=reference.assumptions.warmupSeconds,releasePose=reference.assumptions.releasePose;
assert.equal(warmup,.1);

const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const model=mj.MjModel.from_xml_string(String(reducedBytes)),data=new mj.MjData(model);
const id=(kind,name)=>{const result=mj.mj_name2id(model,mj.mjtObj[kind].value,name);assert(result>=0,name);return result;};
const names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
const joints=names.map(name=>{const i=id('mjOBJ_JOINT',name);return {...metadata.joints.find(j=>j.name===name),id:i,
 qpos:model.jnt_qposadr[i],dof:model.jnt_dofadr[i],range:Array.from(model.jnt_range.slice(i*2,i*2+2))};});
const actuators=names.map(name=>{const i=id('mjOBJ_ACTUATOR',name);return {name,id:i,range:Array.from(model.actuator_ctrlrange.slice(i*2,i*2+2))};});
const meta={...metadata,joints,actuators},rig={mj,model,data,metadata:meta},rootBody=model.jnt_bodyid[0];
assert.deepEqual([model.nq,model.nv,model.njnt,model.nu,model.neq],[13,12,7,6,0]);
assert.equal(model.jnt_type[0],0);assert.equal(mj.mj_versionString(),reference.nativeVersion);
const h=model.opt.timestep,wingStride=4,controlStride=Math.round(.002/h),muscleStride=Math.round(.001/h);
assert.equal(h,metadata.timestep);assert.equal(controlStride*h,.002);assert.equal(muscleStride*h,.001);
const count=historical.nativeSteps,weight=reference.conventions.weight,length=reference.conventions.lengthCm;
assert(count*h>=1&&count*h<=2);assert.equal(weight,metadata.mass_g*981);
const steering=controls=>({left:Object.fromEntries(muscleNames.map((name,i)=>[name,controls[i]])),right:Object.fromEntries(muscleNames.map((name,i)=>[name,controls[12+i]]))});
const zeroApplied=()=>{for(const array of [data.qfrc_applied,data.xfrc_applied])for(const value of array)assert.equal(value,0,'External applied force');};
let released=false,restraintWrites=0;
function restrain(pose){assert(!released,'Root write after release');data.qpos.set(pose,0);data.qvel.fill(0,0,6);restraintWrites++;}
function drive(wings,controls,power){wings.step(data.qpos,data.ctrl,power?.left??controls[24],power?.right??controls[24],steering(controls),h*wingStride);}
function warm(){
 released=false;restraintWrites=0;mj.mj_resetData(model,data);data.qpos.set(releasePose);
 for(const j of joints)data.qpos[j.qpos]=j.neutral;mj.mj_forward(model,data);
 const wings=new FlyBodyWings(meta),steps=Math.round(warmup/h);
 wings.phase=(((-2*Math.PI*wings.frequencyHz*warmup)%(2*Math.PI))+2*Math.PI)%(2*Math.PI);
 for(let s=0;s<steps;s++){restrain(releasePose);if(s%wingStride===0)drive(wings,trim);mj.mj_step(model,data);}
 restrain(releasePose);mj.mj_forward(model,data);return wings;
}
function wrench(){
 const r=Array.from(data.xmat.slice(rootBody*9,rootBody*9+9));
 return worldComWrench(data.qfrc_fluid.slice(0,6),r,data.xpos.slice(rootBody*3,rootBody*3+3),data.subtree_com.slice(rootBody*3,rootBody*3+3));
}
function allocate(target,start){
 const scale=B.map(row=>1/Math.max(1e-6,Math.hypot(...row))),A=B.map((row,i)=>row.map(v=>v*scale[i]));
 const rhs=target.map((v,i)=>(v-trimY[i]+B[i].reduce((s,b,j)=>s+b*trim[j],0))*scale[i]);
 const u=start.map((v,i)=>clamp(v,lower[i],upper[i])),residual=rhs.map((v,i)=>A[i].reduce((s,a,j)=>s+a*u[j],-v)),regularization=1e-4;
 for(let pass=0;pass<20;pass++)for(let j=0;j<25;j++){
  const gradient=A.reduce((s,row,i)=>s+row[j]*residual[i],regularization*(u[j]-trim[j]));
  const curvature=A.reduce((s,row)=>s+row[j]*row[j],regularization);
  const next=clamp(u[j]-gradient/curvature,lower[j],upper[j]),delta=next-u[j];u[j]=next;
  for(let i=0;i<4;i++)residual[i]+=A[i][j]*delta;
 }
 return u;
}
function state(wings){const k=measureFlightKinematics(rig);return {...k,qpos:Array.from(data.qpos),qvel:Array.from(data.qvel),act:Array.from(data.act),wingPhase:wings.phase,wingState:wings.controlState()};}

// This diagnostic uses the actual native muscle kernel. Its other force
// multipliers are frozen explicitly, just as for virtual wing muscles at full
// energy in FlyBodyPhysics. No fatigue or activation state is reset in flight.
const frozen={normalizedLength:1,positiveShorteningVelocity:0,Fmax:1,energy:1};
const contractileGain=frozen.Fmax*Math.exp(-(((frozen.normalizedLength-1)/.45)**2))*clamp(1-.25*frozen.positiveShorteningVelocity,.1,1.8)*clamp(frozen.energy,0,1);
const recruitment=createMotorExcitation();
function inverseForce(force,fatigue){
 const gain=contractileGain*(1-fatigue);assert(gain>0);
 const unbounded=force/gain;return {excitation:clamp(unbounded,0,1),unbounded,gain};
}
class OutputPath{
 constructor(mode){
  this.mode=mode;this.muscles=mode==='direct_force'?null:new WasmMuscles(core,28);
  this.input=new Float32Array(28*5);this.muscleState=new Float32Array(28*3);
  this.rates=new Float32Array(28);this.targetRates=new Float32Array(28);this.excitation=new Float32Array(28);
  this.commanded=[...trim];this.delivered=[...trim];this.inverseClipped=0;this.commandCount=0;
  this.forceErrorSumSquares=0;this.forceErrorMaximum=0;this.forceErrorCount=0;this.maximumFatigue=0;this.maximumActivation=0;
  for(let i=0;i<28;i++){
   const f=trim[mappingControls[i]],e=inverseForce(f,0).excitation;
   assert(Math.abs(e*contractileGain-f)<1e-12,'Initial trim force is unattainable');
   this.muscleState.set([e,0,f],i*3);this.rates[i]=80*e;this.targetRates[i]=80*e;this.excitation[i]=e;
   this.input.set([e,frozen.normalizedLength,frozen.positiveShorteningVelocity,frozen.Fmax,frozen.energy],i*5);
  }
  if(this.muscles)core.HEAPF32.set(this.muscleState,this.muscles.state/4);
  this.initial={muscleState:Array.from(this.muscleState),rateHz:Array.from(this.rates),targetRateHz:Array.from(this.targetRates),
   maximumTrimForceRoundingError:Math.max(...mappingControls.map((k,i)=>Math.abs(this.muscleState[i*3+2]-trim[k])))};
 }
 command(controls){
  this.commanded=[...controls];if(!this.muscles){this.delivered=[...controls];return;}
  for(let i=0;i<28;i++){
   const inverse=inverseForce(controls[mappingControls[i]],this.muscleState[i*3+1]);
   this.inverseClipped+=Number(inverse.unbounded>1);this.commandCount++;
   this.targetRates[i]=80*inverse.excitation;
   if(this.mode==='muscle_only')this.rates[i]=this.targetRates[i];
   else{
    // Synthetic held rate, NOT spikes or BANC. Match the existing nominal
    // 50 ms EMA at four 0.5 ms ticks before the next 2 ms body block.
    const alpha=1-Math.exp(-.5/50);
    for(let tick=0;tick<4;tick++)this.rates[i]+=(this.targetRates[i]-this.rates[i])*alpha;
   }
   this.excitation[i]=recruitment.fromRate(mappings[i].kind,this.rates[i]);
   this.input[i*5]=this.excitation[i];
  }
 }
 step(){
  if(!this.muscles)return;
  this.muscleState=this.muscles.step(this.input,muscleStride*h);
  const power=[];
  for(let i=0;i<28;i++){
   const force=this.muscleState[i*3+2],index=mappingControls[i];
   if(index===24)power.push(force);else this.delivered[index]=force;
   const error=force-this.commanded[index];this.forceErrorSumSquares+=error*error;this.forceErrorMaximum=Math.max(this.forceErrorMaximum,Math.abs(error));this.forceErrorCount++;
   this.maximumFatigue=Math.max(this.maximumFatigue,this.muscleState[i*3+1]);this.maximumActivation=Math.max(this.maximumActivation,this.muscleState[i*3]);
  }
  // Two equal-power muscles per side, averaged in the production body path.
  // Their identical inputs/initial states must preserve exact equality here.
  assert(power.length===4&&power.every(value=>value===power[0]),'Unexpected bilateral power asymmetry');this.delivered[24]=power[0];
 }
 readout(){return {deliveredControls:[...this.delivered],targetRatesHz:this.muscles?Array.from(this.targetRates):null,filteredRatesHz:this.muscles?Array.from(this.rates):null,
  excitation:this.muscles?Array.from(this.excitation):null,muscleState:this.muscles?Array.from(this.muscleState):null};}
 summary(){return {inverseClippedCommandFraction:this.commandCount?this.inverseClipped/this.commandCount:0,inverseClippedCommands:this.inverseClipped,
  muscleForceRmsTrackingError:this.forceErrorCount?Math.sqrt(this.forceErrorSumSquares/this.forceErrorCount):0,
  muscleForceMaximumTrackingError:this.forceErrorMaximum,maximumFatigue:this.maximumFatigue,maximumActivation:this.maximumActivation};}
 dispose(){this.muscles?.dispose();}
}
// Fixed native activation lookup, measured without a body or feedback control.
// It uses exactly the event generator and within-muscle unit phases below.
const rateGrid=[0,2,5,10,20,40,80,120,160,200,300,400];let tables=null;
function nextEvents(generator){
 const from=generator.eventTimeMs,to=from+2,events=[];
 for(let tick=1;tick<=4;tick++)for(let k=0;k<48;k++){
  const t=from+tick*.5;generator.phase[k]+=generator.unitRates[k]*.0005;
  // Arithmetic tolerance prevents a mathematically exact threshold from
  // drifting one neural tick across otherwise periodic integer-Hz cycles.
  if(generator.phase[k]>=1-1e-12&&t-generator.last[k]>=2.5){
   generator.phase[k]=Math.max(0,generator.phase[k]-1);generator.last[k]=t;generator.counts[k]++;
   events.push({index:generator.ids[k],timeMs:t});
  }else generator.phase[k]=Math.min(1,generator.phase[k]);
 }
 generator.eventTimeMs=to;generator.events+=events.length;
 return {initialized:false,fromTimeMs:from,timeMs:to,indices:generator.ids,counts:generator.counts,ratesHz:generator.unitRates.map(Math.fround),events};
}
function generatorState(ids){
 return {ids,phase:ids.map(id=>{const m=mappings.find(m=>m.indices.includes(id));return m.indices.indexOf(id)/m.indices.length;}),
  counts:Array(48).fill(0),last:Array(48).fill(-Infinity),unitRates:Array(48).fill(0),eventTimeMs:0,events:0};
}
async function buildTables(){
 const table=Object.fromEntries(['steering','dlm','dvm'].map(family=>[family,[{rateHz:0,activation:0,excitation:0}]]));
 const family=mappings.map(m=>m.kind==='wing_steering_assumption'?'steering':m.target==='dorsal_longitudinal_muscle'?'dlm':'dvm');
 for(const rateHz of rateGrid.slice(1)){
  const adapter=createWingEventExcitation({io}),ids=Array.from(adapter.readState().indices),generator=generatorState(ids),muscles=new WasmMuscles(core,28);
  generator.unitRates.fill(rateHz);const input=new Float32Array(140);for(let m=0;m<28;m++)input.set([0,1,0,1,1],m*5);
  const activationSum=Array(28).fill(0),excitationSum=Array(28).fill(0),ring=Array(1000);let settleError=0;
  const packetHash=createHash('sha256');
  adapter.accept({initialized:true,fromTimeMs:null,timeMs:0,indices:ids,counts:generator.counts,ratesHz:Array(48).fill(0),events:[]});
  try{
   for(let t=2;t<=4000;t+=2){
    const packet=nextEvents(generator);packetHash.update(JSON.stringify(packet)+'\n');adapter.accept(packet);
    for(const end of [t-1,t]){
     const interval=adapter.finishInterval(end);for(let m=0;m<28;m++)input[m*5]=interval.excitation[m];
     const state=muscles.step(input,.001),activation=Array.from({length:28},(_,m)=>state[m*3]);
     if(end>3000){
      const previous=ring[(end-1)%1000];assert(previous);
      for(let m=0;m<28;m++){activationSum[m]+=activation[m];excitationSum[m]+=interval.excitation[m];settleError=Math.max(settleError,Math.abs(activation[m]-previous[m]));}
     }
     ring[(end-1)%1000]=activation;
    }
   }
   assert(settleError<=1e-6,'Native activation lookup did not settle to1s periodicity');
   for(const f of ['steering','dlm','dvm']){
    const which=family.flatMap((value,m)=>value===f?[m]:[]),values=which.map(m=>activationSum[m]/1000);
    assert(Math.max(...values)-Math.min(...values)<1e-12,'Identical family input gave differing mean activations');
    table[f].push({rateHz,activation:values[0],excitation:excitationSum[which[0]]/1000,settleError});
   }
   console.log(JSON.stringify({kind:'activation-lookup',rateHz,settleError,packetSha256:packetHash.digest('hex')}));
  }finally{muscles.dispose();}
  await new Promise(resolve=>setTimeout(resolve,0));
 }
 for(const rows of Object.values(table))for(let k=1;k<rows.length;k++)assert(rows[k].activation>=rows[k-1].activation,'Nonmonotone activation lookup');
 return table;
}
function inverseEventRate(family,activation){
 const rows=tables[family];if(activation<=0)return {rateHz:0,clipped:false};
 if(activation>rows.at(-1).activation)return {rateHz:rows.at(-1).rateHz,clipped:activation>rows.at(-1).activation};
 const hi=rows.findIndex(row=>row.activation>=activation),a=rows[hi-1],b=rows[hi];
 return {rateHz:a.rateHz+(b.rateHz-a.rateHz)*(activation-a.activation)/(b.activation-a.activation),clipped:false};
}
class EventOutputPath extends OutputPath{
 constructor(){
  super('muscle_only');this.adapter=createWingEventExcitation({io});const contract=this.adapter.readState();
  assert.deepEqual(Array.from(contract.mappingIndices),mappings.map(m=>m.mappingIndex));
  this.ids=Array.from(contract.indices);this.slot=new Map(this.ids.map((id,k)=>[id,k]));
  this.unitMapping=this.ids.map(id=>mappings.findIndex(m=>m.indices.includes(id)));
  this.families=this.unitMapping.map(m=>mappings[m].kind==='wing_steering_assumption'?'steering':mappings[m].target==='dorsal_longitudinal_muscle'?'dlm':'dvm');
  Object.assign(this,generatorState(this.ids));
  // Matching bilateral power phases, evenly staggered within each muscle.
  this.rateClipped=0;this.rateCommands=0;this.quadratureMax=0;
  this.adapter.accept({initialized:true,fromTimeMs:null,timeMs:0,indices:this.ids,counts:this.counts,ratesHz:Array(48).fill(0),events:[]});
  this.command(trim);
  // Only kernel history is prepared. Native activation is the declared trim
  // state with fatigue zero, identical to the direct-excitation reference.
  for(let t=2;t<=2000;t+=2){this.acceptNext();this.adapter.finishInterval(t-1);this.adapter.finishInterval(t);}
  this.initial={...this.initial,eventKernel:this.adapter.snapshot(),generator:{timeMs:this.eventTimeMs,phase:[...this.phase],last:[...this.last],unitRates:[...this.unitRates]},
   preparation:'2 seconds of explicit periodic synthetic events; native state remains prepared trim/zero fatigue'};
  this.rateClipped=0;this.rateCommands=0;this.commandCount=0;this.inverseClipped=0;this.events=0;
 }
 command(controls){
  this.commanded=[...controls];
  for(let m=0;m<28;m++){
   const inverse=inverseForce(controls[mappingControls[m]],this.muscleState[m*3+1]);this.inverseClipped+=Number(inverse.unbounded>1);this.commandCount++;
   const family=mappings[m].kind==='wing_steering_assumption'?'steering':mappings[m].target==='dorsal_longitudinal_muscle'?'dlm':'dvm';
   const mapped=inverseEventRate(family,inverse.excitation);this.rateClipped+=Number(mapped.clipped);this.rateCommands++;
   this.targetRates[m]=mapped.rateHz;
   for(const id of mappings[m].indices)this.unitRates[this.slot.get(id)]=mapped.rateHz;
  }
 }
 acceptNext(){this.adapter.accept(nextEvents(this));}
 step(){
  const end=this.adapter.readState().integratedThroughMs+1,interval=this.adapter.finishInterval(end);
  this.quadratureMax=Math.max(this.quadratureMax,interval.quadratureDiscrepancy);
  for(let m=0;m<28;m++){this.excitation[m]=interval.excitation[m];this.input[m*5]=interval.excitation[m];}
  this.muscleState=this.muscles.step(this.input,muscleStride*h);
  const power={left:[],right:[]};
  for(let m=0;m<28;m++){
   const force=this.muscleState[m*3+2],index=mappingControls[m];
   if(index===24)power[mappings[m].joint.endsWith('_left')?'left':'right'].push(force);else this.delivered[index]=force;
   const error=force-this.commanded[index];this.forceErrorSumSquares+=error*error;this.forceErrorMaximum=Math.max(this.forceErrorMaximum,Math.abs(error));this.forceErrorCount++;
   this.maximumFatigue=Math.max(this.maximumFatigue,this.muscleState[m*3+1]);this.maximumActivation=Math.max(this.maximumActivation,this.muscleState[m*3]);
  }
  this.power=Object.fromEntries(Object.entries(power).map(([side,values])=>[side,values.reduce((a,b)=>a+b,0)/values.length]));
  this.delivered[24]=(this.power.left+this.power.right)/2;
 }
 readout(){const r=super.readout();delete r.filteredRatesHz;return {...r,power:this.power,eventCounts:[...this.counts]};}
 summary(){return {...super.summary(),eventsAfterRelease:this.events,eventRateClippedCommandFraction:this.rateClipped/Math.max(1,this.rateCommands),quadratureDiscrepancyMaximum:this.quadratureMax};}
}
const latencyFile=path.join(root,'reports/flight-muscle-latency/result.json'),latencyBytes=await fs.readFile(latencyFile),latency=JSON.parse(latencyBytes);
assert(latency.completed&&latency.historicalDirectGate.passed&&latency.sourceUnchanged);
const nativeReference=latency.cases.find(c=>c.mode==='muscle_only');assert(nativeReference.meetsDeclaredPositiveControl);
for(const [file,expected]of Object.entries(latency.sourceHashes))if(file!=='scripts/diagnose-flight-muscle-latency.mjs')assert.equal(hashes[file],expected,'Native reference dependency changed: '+file);
function simulate(mode){
 const wings=warm(),initial=state(wings),initialHash=sha(JSON.stringify(initial)),z0=initial.height,start=data.time;
 const eventMode=mode.startsWith('event_'),causal=mode!=='historical_native';
 const outputPath=eventMode?new EventOutputPath():new OutputPath('muscle_only'),filteredOmega=[0,0,0],samples=[],forceSum=[0,0,0],torqueSum=[0,0,0];
 const kp=naturalFrequency**2,kd=2*dampingRatio*naturalFrequency;
 let controls=[...trim],minimumUp=1,maxAngle=0,maxOmega=0,sumOmega2=0,maximumDrift=0,minimumHeight=z0,maximumHeight=z0,clippedSteps=0,contacts=0,maxAllocationSaturation=0;
 try{
  released=true;
  for(let step=0;step<count;step++){
   if(step%controlStride===0){
    const k=measureFlightKinematics(rig),q=data.qpos.slice(3,7),omega=Array.from(data.qvel.slice(3,6)),alpha=-Math.expm1(-controlStride*h/filterTau);
    for(let i=0;i<3;i++)filteredOmega[i]+=alpha*(omega[i]-filteredOmega[i]);
    const sign=q[0]<0?-1:1,n=Math.hypot(q[1],q[2],q[3]),angle=2*Math.atan2(n,Math.abs(q[0]));
    const error=[q[1],q[2],q[3]].map(v=>n>1e-12?sign*v*angle/n:0);
    const acceleration=error.map((v,i)=>-kp*v-kd*filteredOmega[i]);
    const torque=inertia.map(row=>row.reduce((s,v,i)=>s+v*acceleration[i],0));
    const az=clamp(16*(z0-k.height)-8*k.verticalSpeed,-.4*981,.4*981),up=1-2*(q[1]*q[1]+q[2]*q[2]);
    const target=[(1+az/981)/Math.max(.5,up),...torque.map(v=>v/(weight*length))];
    controls=mode==='event_open_loop'?[...trim]:allocate(target,controls);outputPath.command(controls);
    if(eventMode)outputPath.acceptNext();
    maxAllocationSaturation=Math.max(maxAllocationSaturation,controls.filter((v,i)=>v<lower[i]+1e-6||v>upper[i]-1e-6).length);
   }
   if(!causal&&step%muscleStride===0)outputPath.step();
   if(step%wingStride===0)drive(wings,outputPath.delivered,outputPath.power);
   mj.mj_step(model,data);if(causal&&(step+1)%muscleStride===0)outputPath.step();const w=wrench();
   for(let i=0;i<3;i++){forceSum[i]+=w[i];torqueSum[i]+=w[3+i];}
   zeroApplied();assert(data.qpos.every(Number.isFinite)&&data.qvel.every(Number.isFinite),'Nonfinite free trajectory');
   assert(data.time>start,'Native time reset after release');
   const q=data.qpos,v=data.qvel,up=1-2*(q[4]*q[4]+q[5]*q[5]),omega=Math.hypot(v[3],v[4],v[5]);
   minimumUp=Math.min(minimumUp,up);maxAngle=Math.max(maxAngle,2*Math.acos(clamp(Math.abs(q[3]),0,1)));maxOmega=Math.max(maxOmega,omega);sumOmega2+=omega*omega;
   contacts+=Number(data.ncon>0);clippedSteps+=Number(actuators.some(a=>Math.abs(data.ctrl[a.id]-a.range[0])<1e-10||Math.abs(data.ctrl[a.id]-a.range[1])<1e-10));
   if((step+1)%controlStride===0){
    const k=measureFlightKinematics(rig);minimumHeight=Math.min(minimumHeight,k.height);maximumHeight=Math.max(maximumHeight,k.height);
    maximumDrift=Math.max(maximumDrift,Math.hypot(k.position[0]-initial.position[0],k.position[1]-initial.position[1]));
    samples.push({t:data.time-start,position:k.position,verticalSpeed:k.verticalSpeed,q:Array.from(data.qpos.slice(3,7)),omega:Array.from(data.qvel.slice(3,6)),filteredOmega:[...filteredOmega],controls:[...controls],...outputPath.readout()});
   }
  }
  const final=state(wings),warnings=readInstabilityWarnings(data.warning,mj.mjtWarning);
  const metrics={maximumTiltDegrees:Math.acos(clamp(minimumUp,-1,1))*180/Math.PI,maximumAttitudeErrorDegrees:maxAngle*180/Math.PI,
   finalTiltDegrees:Math.acos(clamp(1-2*(final.qpos[4]**2+final.qpos[5]**2),-1,1))*180/Math.PI,
   maximumAngularSpeed:maxOmega,rmsAngularSpeed:Math.sqrt(sumOmega2/count),finalFilteredAngularSpeed:Math.hypot(...filteredOmega),
   COMriseCm:final.height-z0,minimumHeightCm:minimumHeight,maximumHeightCm:maximumHeight,maximumHorizontalDriftCm:maximumDrift,
   wingControlClippedStepFraction:clippedSteps/count,contactStepFraction:contacts/count,maximumSaturatedAllocationChannels:maxAllocationSaturation,
   meanWorldFluidForce:forceSum.map(v=>v/count),meanWorldFluidTorqueAtCOM:torqueSum.map(v=>v/count)};
  return {mode,nativeMusclesActive:true,initial,initialHash,initialOutputState:outputPath.initial,final,nativeSteps:count,freeSeconds:data.time-start,warmupSeconds:warmup,restraintWrites,
   poseWritesAfterRelease:0,maximumAppliedExternalForce:0,warnings,metrics,outputMetrics:outputPath.summary(),
   meetsDeclaredPositiveControl:metrics.maximumTiltDegrees<20&&metrics.finalFilteredAngularSpeed<3&&Math.abs(metrics.COMriseCm)<10&&contacts===0&&warnings.length===0,samples};
 }finally{outputPath.dispose();}
}

let report,ownsOutput=false;
try{
 report={schemaVersion:1,kind:'diagnostic-classical-wing-event-control',createdAt:new Date().toISOString(),completed:false,
  nativeVersion:mj.mj_versionString(),sourceHashes:hashes,reference:{file:path.relative(root,referencePath),sha256:sha(referenceBytes),modelSha256:sha(reducedBytes),
   historicalScriptSha256:reference.sourceHashes['scripts/diagnose-flight-classical-control.mjs'],historicalWingSourceSha256:reference.sourceHashes['web/flybody-wings.js']},
  dimensions:{nq:model.nq,nv:model.nv,njnt:model.njnt,nu:model.nu},controlNames,wingMappings:mappings.map((mapping,i)=>({...mapping,controlIndex:mappingControls[i]})),
  trimControls:trim,trimResponse:trimY,responseJacobian:B,conventions:reference.conventions,
  protocol:{arms:['historical_native','causal_native','event_closed_loop','event_open_loop'],freeSeconds:count*h,warmupSeconds:warmup,
   neuralExecution:false,calibrationProbes:0,gainSearches:0,controllerGains:reference.assumptions.gains,
   controlIntervalSeconds:.002,muscleIntervalSeconds:.001,wingIntervalSeconds:h*wingStride,nativeTimestepSeconds:h,
   frozenMuscleInput:frozen,nativeMuscleTimeConstantsSeconds:{activation:.015,deactivation:.04},
   eventPriors:DEFAULT_WING_EVENT_PRIORS,rateGrid,eventInverseTables:tables,
   lookupProtocol:'Same0.5ms generator and within-muscle phases;28native muscle groups;3s warmup+1s mean activation, periodic agreement<=1e-6. Fatigue evolves but table inverts activation only; force inverse accounts for current fatigue independently.',
   eventGenerator:'Per-unit phase accumulator on0.5ms grid, held rates per2ms control interval, minimum2.5ms between events; no packet rates are used by the adapter.',
   eventInverse:'Current-fatigue force inverse followed by linear interpolation of fixed periodic-mean native activation table; unattainable rates clip at400Hz and are reported. No dynamics inverse or fitting.',
   startup:'Identical100ms physical warmup under direct trim; native activation initialized to trim and fatigue0. Event arms additionally prepare2s kernel history at trim rates without evolving native fatigue. This is a declared initial condition, not a steady-state biological fly.',
   ordering:'Historical_native reproduces old pre-boundary update exactly. Other arms integrate1ms physics with previous force, then update native muscle at the completed interval right boundary. Event average uses only events at/before that boundary.',
   constraints:'Existing6wing actuators only; no root writes or applied forces after release. Synthetic classical controller is a positive control, not BANC or learning. No gain searches, horizontal controller, takeoff or landing.',
   inference:'Failure of this fixed controller does not prove no controller can stabilize the event path. Inspect rate clipping, force error and changed scheduling.',
   nativeReference:{file:path.relative(root,latencyFile),sha256:sha(latencyBytes)},passCriterion:reference.assumptions.passCriterion},cases:[]};
 console.log(JSON.stringify({kind:'event-control-preflight',execute:run,dimensions:report.dimensions,arms:report.protocol.arms,freeSeconds:count*h,muscles:mappings.length,calibrationProbes:0}));
 // A check-only invocation loads/validates the real modules and mappings but
 // makes no call to mj_step or the native muscle kernel and writes no report.
 if(!run)console.log('No simulation performed. Add --run for four sequential fixed-controller arms.');
 else{
  await fs.mkdir(path.dirname(output),{recursive:true});
  assert((await fs.realpath(path.dirname(output))).startsWith((await fs.realpath(reports))+path.sep)||await fs.realpath(path.dirname(output))===await fs.realpath(reports),'Output symlink escapes reports');
  await fs.mkdir(output);ownsOutput=true;
  await fs.writeFile(path.join(output,'diagnostic-source.used.mjs'),bytes['scripts/diagnose-flight-event-control.mjs'],{flag:'wx'});
  await fs.writeFile(path.join(output,'fixed-nonwing.xml'),reducedBytes,{flag:'wx'});
  tables=await buildTables();report.protocol.eventInverseTables=tables;
  const save=async()=>{await fs.writeFile(path.join(output,'result.tmp'),JSON.stringify(report)+'\n');await fs.rename(path.join(output,'result.tmp'),path.join(output,'result.json'));};
  await save();
  for(const mode of report.protocol.arms){
   console.log(JSON.stringify({kind:'arm-start',mode}));
   const result=simulate(mode);report.cases.push(result);await save();
   assert.equal(result.initialHash,historical.initialHash,'Historical plant/wing warm state changed');
   if(mode==='historical_native'){
    assert.deepEqual(result.samples,nativeReference.samples,'Historical native sampled trajectory differs');
    assert.deepEqual(result.final,nativeReference.final,'Historical native final state differs');
    assert.deepEqual(result.metrics,nativeReference.metrics,'Historical native metrics differ');
    assert.deepEqual(result.outputMetrics,nativeReference.outputMetrics,'Historical native muscle metrics differ');
    assert(result.meetsDeclaredPositiveControl,'Historical native positive control failed');
    report.historicalNativeGate={passed:true,exactSampledTrajectory:true,samples:result.samples.length,exactFinalState:true,exactMetrics:true};
   }else if(mode==='causal_native')assert.deepEqual(result.initialOutputState,report.cases[0].initialOutputState);
   else if(mode==='event_open_loop')assert.deepEqual(result.initialOutputState,report.cases.find(c=>c.mode==='event_closed_loop').initialOutputState);
   await save();console.log(JSON.stringify({kind:'arm-complete',mode,pass:result.meetsDeclaredPositiveControl,...result.metrics,...result.outputMetrics,warnings:result.warnings}));
  }
  assert(report.cases.every(row=>row.initialHash===report.cases[0].initialHash));report.pairedInitialStateIdentical=true;
  for(const [file,expected]of Object.entries(hashes))assert.equal(sha(await fs.readFile(path.join(root,file))),expected,'Source changed during assay: '+file);
  assert.equal(sha(await fs.readFile(referencePath)),sha(referenceBytes),'Historical report changed during assay');
  assert.equal(sha(await fs.readFile(reducedPath)),sha(reducedBytes),'Historical reduced XML changed during assay');
  report.sourceUnchanged=true;report.completed=true;
  report.classification=Object.fromEntries(report.cases.map(row=>[row.mode,row.meetsDeclaredPositiveControl?'passes_fixed_positive_control':'fails_fixed_positive_control']));
  await save();console.log(JSON.stringify({output,completed:true,historicalNativeGate:report.historicalNativeGate,classification:report.classification}));
 }
}catch(error){
 if(run&&report&&ownsOutput){report.failure=String(error.stack||error);try{await fs.writeFile(path.join(output,'failure.json'),JSON.stringify(report)+'\n',{flag:'wx'});}catch{}}
 throw error;
}finally{data.delete();model.delete();}
