// DIAGNOSTIC ONLY. A frozen classical wing controller with three output paths.
// No BANC execution, fitting, production changes, or fresh response calibration.
// node scripts/diagnose-flight-muscle-latency.mjs --run --output=reports/flight-muscle-latency
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {createMotorExcitation} from '../web/flybody-motor-excitation.js';
import {measureFlightKinematics} from '../web/training/flight-observation.js';
import {worldComWrench,readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),run=args.includes('--run');
assert(args.every(arg=>arg==='--run'||arg==='--check-only'||/^--(output|reference)=/.test(arg)),'Unknown argument');
assert(!(run&&args.includes('--check-only')),'Choose --run or --check-only');
const option=(key,fallback)=>args.find(arg=>arg.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const reports=path.join(root,'reports'),output=path.resolve(root,option('output','reports/flight-muscle-latency'));
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
const files=['scripts/diagnose-flight-muscle-latency.mjs','scripts/motor-wing-calibration-helpers.mjs',
 'web/flybody-wings.js','web/training/flight-parameters.js','web/training/flight-observation.js','web/flybody-motor-excitation.js',
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
function drive(wings,controls){wings.step(data.qpos,data.ctrl,controls[24],controls[24],steering(controls),h*wingStride);}
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
function simulate(mode){
 const wings=warm(),initial=state(wings),initialHash=sha(JSON.stringify(initial)),z0=initial.height,start=data.time;
 const outputPath=new OutputPath(mode),filteredOmega=[0,0,0],samples=[],forceSum=[0,0,0],torqueSum=[0,0,0];
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
    controls=allocate(target,controls);outputPath.command(controls);
    maxAllocationSaturation=Math.max(maxAllocationSaturation,controls.filter((v,i)=>v<lower[i]+1e-6||v>upper[i]-1e-6).length);
   }
   if(step%muscleStride===0)outputPath.step();
   if(step%wingStride===0)drive(wings,outputPath.delivered);
   mj.mj_step(model,data);const w=wrench();
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
  return {mode,nativeMusclesActive:mode!=='direct_force',initial,initialHash,initialOutputState:outputPath.initial,final,nativeSteps:count,freeSeconds:data.time-start,warmupSeconds:warmup,restraintWrites,
   poseWritesAfterRelease:0,maximumAppliedExternalForce:0,warnings,metrics,outputMetrics:outputPath.summary(),
   meetsDeclaredPositiveControl:metrics.maximumTiltDegrees<20&&metrics.finalFilteredAngularSpeed<3&&Math.abs(metrics.COMriseCm)<10&&contacts===0&&warnings.length===0,samples};
 }finally{outputPath.dispose();}
}

let report,ownsOutput=false;
try{
 report={schemaVersion:1,kind:'diagnostic-classical-wing-muscle-latency',createdAt:new Date().toISOString(),completed:false,
  nativeVersion:mj.mj_versionString(),sourceHashes:hashes,reference:{file:path.relative(root,referencePath),sha256:sha(referenceBytes),modelSha256:sha(reducedBytes),
   historicalScriptSha256:reference.sourceHashes['scripts/diagnose-flight-classical-control.mjs'],historicalWingSourceSha256:reference.sourceHashes['web/flybody-wings.js']},
  dimensions:{nq:model.nq,nv:model.nv,njnt:model.njnt,nu:model.nu},controlNames,wingMappings:mappings.map((mapping,i)=>({...mapping,controlIndex:mappingControls[i]})),
  trimControls:trim,trimResponse:trimY,responseJacobian:B,conventions:reference.conventions,
  protocol:{arms:['direct_force','muscle_only','rate50ms_muscle'],freeSeconds:count*h,warmupSeconds:warmup,neuralExecution:false,calibrationProbes:0,gainSearches:0,
   controllerGains:reference.assumptions.gains,controlIntervalSeconds:.002,muscleIntervalSeconds:.001,wingIntervalSeconds:h*wingStride,nativeTimestepSeconds:h,
   nativeMuscleTimeConstantsSeconds:{activation:.015,deactivation:.04},syntheticRateFilter:{tauMs:50,timestepMs:.5,ticksPerCommand:4,float32State:true},
   frozenMuscleInput:frozen,contractileGain,rateDecoder:'Canonical legacy excitation=clamp(rateHz/80,0,1), including steering. Hill recruitment is not used.',
   inverse:'excitation=clamp(desiredForce/[Fmax*(1-fatigue)*exp(-((length-1)/.45)^2)*clamp(1-.25*velocity,.1,1.8)*clamp(energy,0,1)],0,1); desiredRateHz=80*excitation. Recomputed at each 2 ms command from current native fatigue; not activation-lag cancellation.',
   startup:'Identical 100 ms plant/wing warm-up under direct trim. Before release native muscle activation is initialized to the exact instantaneous trim-force inverse, fatigue to zero, and rate-filter state to the corresponding tonic rate. This prepared state removes arbitrary zero-activation startup; it is not a simulated neural or fatigue equilibrium. Native state and delivered force use float32 rounding.',
   ordering:'Controller at 2 ms boundaries; optional synthetic held-rate EMA advanced four 0.5 ms ticks; excitation held for two actual native 1 ms muscle steps, each before its 1 ms block of physical integration. Mirrors the body block ordering without generating spikes.',
   fatigue:'Actual native fatigue evolves after release. The controller is given modeled fatigue for algebraic force scaling. This diagnostic compensation is an explicit controller assumption, not a BANC mechanism.',
   constraints:'All arms use unchanged classical PD/allocation, current legacy absolute-force basis, existing six wing controls and reduced free-root plant. No root write after release, xfrc/qfrc drive, BANC, learning, gain tuning, horizontal control, takeoff or landing.',
   inference:'A failed delayed arm shows this fixed controller is not robust to the added modeled output dynamics; it does not prove no latency-aware controller can stabilize the plant. Native gating also introduces actual fatigue and reduced available force; inspect saturation and tracking error before attributing failure solely to delay.',
   passCriterion:reference.assumptions.passCriterion},cases:[]};
 console.log(JSON.stringify({kind:'muscle-latency-preflight',execute:run,dimensions:report.dimensions,arms:report.protocol.arms,freeSeconds:count*h,muscles:mappings.length,calibrationProbes:0}));
 // A check-only invocation loads/validates the real modules and mappings but
 // makes no call to mj_step or the native muscle kernel and writes no report.
 if(!run)console.log('No simulation performed. Add --run for three sequential fixed-controller arms.');
 else{
  await fs.mkdir(path.dirname(output),{recursive:true});
  assert((await fs.realpath(path.dirname(output))).startsWith((await fs.realpath(reports))+path.sep)||await fs.realpath(path.dirname(output))===await fs.realpath(reports),'Output symlink escapes reports');
  await fs.mkdir(output);ownsOutput=true;
  await fs.writeFile(path.join(output,'diagnostic-source.used.mjs'),bytes['scripts/diagnose-flight-muscle-latency.mjs'],{flag:'wx'});
  await fs.writeFile(path.join(output,'fixed-nonwing.xml'),reducedBytes,{flag:'wx'});
  const save=async()=>{await fs.writeFile(path.join(output,'result.tmp'),JSON.stringify(report)+'\n');await fs.rename(path.join(output,'result.tmp'),path.join(output,'result.json'));};
  await save();
  for(const mode of report.protocol.arms){
   console.log(JSON.stringify({kind:'arm-start',mode}));
   const result=simulate(mode);report.cases.push(result);await save();
   assert.equal(result.initialHash,historical.initialHash,'Historical plant/wing warm state changed');
   if(mode==='direct_force'){
    const comparison=result.samples.map(({t,position,verticalSpeed,q,omega,filteredOmega,controls})=>({t,position,verticalSpeed,q,omega,filteredOmega,controls}));
    assert.deepEqual(comparison,historical.samples,'Direct sampled trajectory differs from historical positive control');
    assert.deepEqual(result.final,historical.final,'Direct final state differs from historical positive control');
    assert.deepEqual(result.metrics,historical.metrics,'Direct metrics differ from historical positive control');
    assert(result.meetsDeclaredPositiveControl,'Direct positive control failed');
    report.historicalDirectGate={passed:true,exactSampledTrajectory:true,samples:comparison.length,exactFinalState:true,exactMetrics:true};
   }else assert.deepEqual(result.initialOutputState,report.cases[0].initialOutputState,'Prepared muscle/filter states differ');
   await save();console.log(JSON.stringify({kind:'arm-complete',mode,pass:result.meetsDeclaredPositiveControl,...result.metrics,...result.outputMetrics,warnings:result.warnings}));
  }
  assert(report.cases.every(row=>row.initialHash===report.cases[0].initialHash));report.pairedInitialStateIdentical=true;
  for(const [file,expected]of Object.entries(hashes))assert.equal(sha(await fs.readFile(path.join(root,file))),expected,'Source changed during assay: '+file);
  assert.equal(sha(await fs.readFile(referencePath)),sha(referenceBytes),'Historical report changed during assay');
  assert.equal(sha(await fs.readFile(reducedPath)),sha(reducedBytes),'Historical reduced XML changed during assay');
  report.sourceUnchanged=true;report.completed=true;
  report.classification=Object.fromEntries(report.cases.map(row=>[row.mode,row.meetsDeclaredPositiveControl?'passes_fixed_positive_control':'fails_fixed_positive_control']));
  await save();console.log(JSON.stringify({output,completed:true,historicalDirectGate:report.historicalDirectGate,classification:report.classification}));
 }
}catch(error){
 if(run&&report&&ownsOutput){report.failure=String(error.stack||error);try{await fs.writeFile(path.join(output,'failure.json'),JSON.stringify(report)+'\n',{flag:'wx'});}catch{}}
 throw error;
}finally{data.delete();model.delete();}
