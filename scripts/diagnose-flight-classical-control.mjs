// DIAGNOSTIC ONLY. Synthetic bounded muscle-force inputs through the existing
// FlyBodyWings basis; no BANC, production controller, or changed model assets.
// node scripts/diagnose-flight-classical-control.mjs --run --output=reports/flight-classical-control
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {measureFlightKinematics} from '../web/training/flight-observation.js';
import {worldComWrench,readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),run=args.includes('--run');
assert(args.every(arg=>arg==='--run'||arg==='--check-only'||/^--(output|seconds|frequency|damping)=/.test(arg)),'Unknown argument');
assert(!(run&&args.includes('--check-only')),'Choose --run or --check-only');
const option=(key,fallback)=>args.find(arg=>arg.startsWith(`--${key}=`))?.slice(key.length+3)??fallback;
const duration=Number(option('seconds',1.5)),naturalFrequency=Number(option('frequency',24)),dampingRatio=Number(option('damping',.9));
assert(Number.isFinite(duration)&&duration>=1&&duration<=2,'Free duration must be 1–2 seconds');
assert(Number.isFinite(naturalFrequency)&&naturalFrequency>=4&&naturalFrequency<=60,'PD natural frequency must be 4–60 rad/s');
assert(Number.isFinite(dampingRatio)&&dampingRatio>=.3&&dampingRatio<=2,'Damping ratio must be .3–2');
const output=path.resolve(root,option('output','reports/flight-classical-control'));
assert(output.startsWith(path.join(root,'reports')+path.sep),'Output must be under reports');
const sha=value=>createHash('sha256').update(value).digest('hex'),clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const files=['scripts/diagnose-flight-classical-control.mjs','scripts/motor-wing-calibration-helpers.mjs',
 'web/flybody-wings.js','web/training/flight-parameters.js','web/training/flight-observation.js',
 'models/flybody-mujoco.xml','models/flybody-mujoco.json','models/flybody-wing-actuation.json',
 'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js','packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
const buffers=await Promise.all(files.map(file=>fs.readFile(path.join(root,file))));
const hashes=Object.fromEntries(files.map((file,i)=>[file,sha(buffers[i])])),xml=String(buffers[5]),metadata=JSON.parse(buffers[6]);
assert.equal(sha(xml),metadata.xml_sha256);assert.deepEqual(metadata.wing_actuation,JSON.parse(buffers[7]));
const names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
const nonwing=metadata.joints.filter(j=>!names.includes(j.name)),remove=new Set(nonwing.map(j=>j.name));
let removedJointCount=0;
let reducedXml=xml.replace(/<joint\b[^>]*\/>/g,tag=>{
 if(!remove.has(/\bname="([^"]+)"/.exec(tag)?.[1]))return tag;
 removedJointCount++;return '';
});
assert.equal(removedJointCount,nonwing.length);
let retainedActuatorCount=0;
reducedXml=reducedXml.replace(/<actuator>([\s\S]*?)<\/actuator>/,(_,section)=>'<actuator>'+section.replace(/<([a-z]+)\b[^>]*\/>/g,tag=>{
 if(!names.includes(/\bname="([^"]+)"/.exec(tag)?.[1]))return '';
 retainedActuatorCount++;return tag;
})+'</actuator>');
assert.equal(retainedActuatorCount,6);

const mj=await loadMujoco(),model=mj.MjModel.from_xml_string(reducedXml),data=new mj.MjData(model);
const id=(kind,name)=>{const result=mj.mj_name2id(model,mj.mjtObj[kind].value,name);assert(result>=0,name);return result;};
const joints=names.map(name=>{const i=id('mjOBJ_JOINT',name);return {...metadata.joints.find(j=>j.name===name),id:i,
 qpos:model.jnt_qposadr[i],dof:model.jnt_dofadr[i],range:Array.from(model.jnt_range.slice(i*2,i*2+2))};});
const actuators=names.map(name=>{const i=id('mjOBJ_ACTUATOR',name);return {name,id:i,range:Array.from(model.actuator_ctrlrange.slice(i*2,i*2+2))};});
const meta={...metadata,joints,actuators},rig={mj,model,data,metadata:meta},rootBody=model.jnt_bodyid[0];
assert.deepEqual([model.nq,model.nv,model.njnt,model.nu,model.neq],[13,12,7,6,0]);
assert.equal(model.jnt_type[0],0);
const h=model.opt.timestep,wingStride=4,controlStride=Math.round(.002/h),warmup=.1,weight=metadata.mass_g*981,length=.27;
assert.equal(h,metadata.timestep);assert.equal(controlStride*h,.002);
const muscleNames=Object.keys(metadata.wing_actuation.steering),controlNames=['left','right'].flatMap(side=>muscleNames.map(name=>`${side}:${name}`)).concat('common_power');
assert.equal(muscleNames.length,12);
const lower=new Array(25).fill(0),upper=new Array(25).fill(1);lower[24]=.25;
const center=new Array(25).fill(.35);center[24]=.85;
const levelPose=[0,0,10,1,0,0,0];
const releaseVector=[3,3,3].map(v=>v*Math.PI/180),releaseAngle=Math.hypot(...releaseVector);
const releasePose=[0,0,10,Math.cos(releaseAngle/2),...releaseVector.map(v=>v/releaseAngle*Math.sin(releaseAngle/2))];
const steering=controls=>({left:Object.fromEntries(muscleNames.map((name,i)=>[name,controls[i]])),right:Object.fromEntries(muscleNames.map((name,i)=>[name,controls[12+i]]))});
const zeroApplied=()=>{for(const array of [data.qfrc_applied,data.xfrc_applied])for(const value of array)assert.equal(value,0,'External applied force');};
let released=false,restraintWrites=0;
function restrain(pose){assert(!released,'Root write after release');data.qpos.set(pose,0);data.qvel.fill(0,0,6);restraintWrites++;}
function reset(pose=levelPose){released=false;restraintWrites=0;mj.mj_resetData(model,data);data.qpos.set(pose);for(const j of joints)data.qpos[j.qpos]=j.neutral;mj.mj_forward(model,data);return new FlyBodyWings(meta);}
function drive(wings,controls){wings.step(data.qpos,data.ctrl,controls[24],controls[24],steering(controls),h*wingStride);}
function wrench(){
 const r=Array.from(data.xmat.slice(rootBody*9,rootBody*9+9));
 const world=worldComWrench(data.qfrc_fluid.slice(0,6),r,data.xpos.slice(rootBody*3,rootBody*3+3),data.subtree_com.slice(rootBody*3,rootBody*3+3));
 const bodyTorque=[0,1,2].map(i=>r[i]*world[3]+r[3+i]*world[4]+r[6+i]*world[5]);
 return {world,bodyTorque};
}
function inertiaAtCOM(){
 mj.mj_kinematics(model,data);mj.mj_comPos(model,data);
 const com=data.subtree_com.slice(rootBody*3,rootBody*3+3),inertia=Array.from({length:3},()=>[0,0,0]);
 for(let b=rootBody;b<model.nbody;b++){
  const mass=model.body_mass[b],r=data.ximat.slice(b*9,b*9+9),d=Array.from(data.xipos.slice(b*3,b*3+3),(v,i)=>v-com[i]),d2=d.reduce((s,v)=>s+v*v,0);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++){
   for(let k=0;k<3;k++)inertia[i][j]+=r[i*3+k]*model.body_inertia[b*3+k]*r[j*3+k];
   inertia[i][j]+=mass*((i===j?d2:0)-d[i]*d[j]);
  }
 }
 return inertia;
}
function warm(controls,pose){
 const wings=reset(pose),steps=Math.round(warmup/h);
 // All calibration and release warmups end at the same wing phase.
 wings.phase=(((-2*Math.PI*wings.frequencyHz*warmup)%(2*Math.PI))+2*Math.PI)%(2*Math.PI);
 for(let s=0;s<steps;s++){restrain(pose);if(s%wingStride===0)drive(wings,controls);mj.mj_step(model,data);}
 restrain(pose);mj.mj_forward(model,data);return wings;
}
function measure(controls,label){
 const wings=warm(controls,levelPose),steps=Math.round(16/metadata.wing_actuation.frequency_hz/h),sum=[0,0,0,0],full=[0,0,0,0,0,0];
 for(let s=0;s<steps;s++){
  restrain(levelPose);if(s%wingStride===0)drive(wings,controls);mj.mj_step(model,data);
  const w=wrench(),value=[w.world[2]/weight,...w.bodyTorque.map(v=>v/(weight*length))];
  for(let i=0;i<4;i++)sum[i]+=value[i];for(let i=0;i<6;i++)full[i]+=w.world[i];zeroApplied();
 }
 assert(data.qpos.every(Number.isFinite)&&data.qvel.every(Number.isFinite),'Nonfinite calibration');
 return {label,controls:[...controls],response:sum.map(v=>v/steps),meanWorldCOMWrench:full.map(v=>v/steps),nativeSteps:Math.round(warmup/h)+steps,
  measuredSeconds:steps*h,restraintWrites,contacts:data.ncon};
}
function allocate(B,baseU,baseY,target,reference,start,passes=30){
 // Convex box-constrained least squares, coordinate descent. Scale each output
 // by its measured row norm so torque is not lost to force-unit magnitudes.
 const scale=B.map(row=>1/Math.max(1e-6,Math.hypot(...row))),A=B.map((row,i)=>row.map(v=>v*scale[i]));
 const rhs=target.map((v,i)=>(v-baseY[i]+B[i].reduce((s,b,j)=>s+b*baseU[j],0))*scale[i]);
 const u=start.map((v,i)=>clamp(v,lower[i],upper[i])),residual=rhs.map((v,i)=>A[i].reduce((s,a,j)=>s+a*u[j],-v)),regularization=1e-4;
 for(let pass=0;pass<passes;pass++)for(let j=0;j<25;j++){
  const gradient=A.reduce((s,row,i)=>s+row[j]*residual[i],regularization*(u[j]-reference[j]));
  const curvature=A.reduce((s,row)=>s+row[j]*row[j],regularization);
  const next=clamp(u[j]-gradient/curvature,lower[j],upper[j]),delta=next-u[j];u[j]=next;
  for(let i=0;i<4;i++)residual[i]+=A[i][j]*delta;
 }
 return u;
}
function state(wings){const k=measureFlightKinematics(rig);return {...k,qpos:Array.from(data.qpos),qvel:Array.from(data.qvel),act:Array.from(data.act),wingPhase:wings.phase,wingState:wings.controlState()};}
function simulate(mode,trim,B,trimY,inertia){
 const wings=warm(trim,releasePose),initial=state(wings),initialHash=sha(JSON.stringify(initial)),z0=initial.height,start=data.time;
 const count=Math.round(duration/h),filteredOmega=[0,0,0],samples=[],forceSum=[0,0,0],torqueSum=[0,0,0];
 const filterTau=.006,kp=naturalFrequency**2,kd=2*dampingRatio*naturalFrequency;
 let controls=[...trim],minimumUp=1,maxAngle=0,maxOmega=0,sumOmega2=0,maximumDrift=0,minimumHeight=z0,maximumHeight=z0,clippedSteps=0,contacts=0,maxAllocationSaturation=0;
 released=true;
 for(let step=0;step<count;step++){
  if(step%controlStride===0){
   const k=measureFlightKinematics(rig),q=data.qpos.slice(3,7),omega=Array.from(data.qvel.slice(3,6)),alpha=-Math.expm1(-controlStride*h/filterTau);
   for(let i=0;i<3;i++)filteredOmega[i]+=alpha*(omega[i]-filteredOmega[i]);
   if(mode==='closed_loop'){
    const sign=q[0]<0?-1:1,n=Math.hypot(q[1],q[2],q[3]),angle=2*Math.atan2(n,Math.abs(q[0]));
    const error=[q[1],q[2],q[3]].map(v=>n>1e-12?sign*v*angle/n:0);
    const acceleration=error.map((v,i)=>-kp*v-kd*filteredOmega[i]);
    const torque=inertia.map(row=>row.reduce((s,v,i)=>s+v*acceleration[i],0));
    // Common power only: conservative height PD, no horizontal controller.
    const az=clamp(16*(z0-k.height)-8*k.verticalSpeed,-.4*981,.4*981),up=1-2*(q[1]*q[1]+q[2]*q[2]);
    const target=[(1+az/981)/Math.max(.5,up),...torque.map(v=>v/(weight*length))];
    controls=allocate(B,trim,trimY,target,trim,controls,20);
   }
   maxAllocationSaturation=Math.max(maxAllocationSaturation,controls.filter((v,i)=>v<lower[i]+1e-6||v>upper[i]-1e-6).length);
  }
  if(step%wingStride===0)drive(wings,controls);
  mj.mj_step(model,data);const w=wrench();
  for(let i=0;i<3;i++){forceSum[i]+=w.world[i];torqueSum[i]+=w.world[3+i];}
  zeroApplied();assert(data.qpos.every(Number.isFinite)&&data.qvel.every(Number.isFinite),'Nonfinite free trajectory');
  assert(data.time>start,'Native time reset after release');
  const q=data.qpos,v=data.qvel,up=1-2*(q[4]*q[4]+q[5]*q[5]),omega=Math.hypot(v[3],v[4],v[5]);
  minimumUp=Math.min(minimumUp,up);maxAngle=Math.max(maxAngle,2*Math.acos(clamp(Math.abs(q[3]),0,1)));maxOmega=Math.max(maxOmega,omega);sumOmega2+=omega*omega;
  contacts+=Number(data.ncon>0);clippedSteps+=Number(actuators.some(a=>Math.abs(data.ctrl[a.id]-a.range[0])<1e-10||Math.abs(data.ctrl[a.id]-a.range[1])<1e-10));
  if((step+1)%controlStride===0){
   const k=measureFlightKinematics(rig);minimumHeight=Math.min(minimumHeight,k.height);maximumHeight=Math.max(maximumHeight,k.height);
   maximumDrift=Math.max(maximumDrift,Math.hypot(k.position[0]-initial.position[0],k.position[1]-initial.position[1]));
   samples.push({t:data.time-start,position:k.position,verticalSpeed:k.verticalSpeed,q:Array.from(data.qpos.slice(3,7)),omega:Array.from(data.qvel.slice(3,6)),filteredOmega:[...filteredOmega],controls:[...controls]});
  }
 }
 const final=state(wings),warnings=readInstabilityWarnings(data.warning,mj.mjtWarning);
 const metrics={maximumTiltDegrees:Math.acos(clamp(minimumUp,-1,1))*180/Math.PI,maximumAttitudeErrorDegrees:maxAngle*180/Math.PI,
  finalTiltDegrees:Math.acos(clamp(1-2*(final.qpos[4]**2+final.qpos[5]**2),-1,1))*180/Math.PI,
  maximumAngularSpeed:maxOmega,rmsAngularSpeed:Math.sqrt(sumOmega2/count),finalFilteredAngularSpeed:Math.hypot(...filteredOmega),
  COMriseCm:final.height-z0,minimumHeightCm:minimumHeight,maximumHeightCm:maximumHeight,maximumHorizontalDriftCm:maximumDrift,
  wingControlClippedStepFraction:clippedSteps/count,contactStepFraction:contacts/count,maximumSaturatedAllocationChannels:maxAllocationSaturation,
  meanWorldFluidForce:forceSum.map(v=>v/count),meanWorldFluidTorqueAtCOM:torqueSum.map(v=>v/count)};
 return {mode,initial,initialHash,final,nativeSteps:count,freeSeconds:data.time-start,warmupSeconds:warmup,restraintWrites,
  poseWritesAfterRelease:0,maximumAppliedExternalForce:0,warnings,metrics,
  meetsDeclaredPositiveControl:metrics.maximumTiltDegrees<20&&metrics.finalFilteredAngularSpeed<3&&Math.abs(metrics.COMriseCm)<10&&contacts===0&&warnings.length===0,samples};
}

let report,ownsOutput=false;
try{
 reset();const inertia=inertiaAtCOM();
 report={schemaVersion:1,kind:'diagnostic-classical-wing-basis-positive-control',createdAt:new Date().toISOString(),nativeVersion:mj.mj_versionString(),sourceHashes:hashes,
  modelHash:sha(reducedXml),dimensions:{nq:model.nq,nv:model.nv,njnt:model.njnt,nu:model.nu},controlNames,
  conventions:{massG:metadata.mass_g,lengthCm:length,weight,angularVelocity:'root-local rad/s',inertia,inertiaUnits:'g cm^2',
   inertiaReference:'Complete modeled rigid bodies about whole-body COM at level neutral pose, expressed in root axes. Joint armature is excluded from this controller approximation but remains in the native plant.',
   calibrationResponse:['vertical fluid force / weight','body COM torque x / (weight*0.27cm)','body COM torque y / (weight*0.27cm)','body COM torque z / (weight*0.27cm)']},
  assumptions:{inputs:'Synthetic normalized muscle-force outputs, independently bounded [0,1], passed directly to the unchanged FlyBodyWings steering basis. Neural spiking and muscle activation dynamics are bypassed; this is a plant/control-authority test.',
   affineOffset:'The current FlyBodyWings implementation applies its derivative steering basis to absolute nonnegative force, whereas the cited MPC source uses deviations from a muscle-mean operating point. This diagnostic preserves the current implementation and measures its own tonic trim; it does not resolve or validate that source-to-runtime affine-offset transfer.',
   plant:'Nonwing joints removed at qpos0; rigid geometry, masses/inertias, wing armatures, wing actuators and fluid model preserved. Seven joints, twelve velocity DoFs. No environment contacts.',
   controller:'Classical quaternion PD with filtered root-local angular velocity, static rigid-body COM inertia approximation and measured constrained wing-basis allocation. Common-power height PD; no horizontal position controller.',
   afterRelease:'Only existing six wing actuator controls are written. No root resets, applied forces, root actuator or neural route.',
   warmupSeconds:warmup,calibrationCycles:16,calibrationProbes:51,maximumTrimUpdates:3,center,lower,upper,releasePose,
   gains:{naturalFrequencyRadPerSecond:naturalFrequency,dampingRatio,angularRateFilterTauSeconds:.006,heightKp:16,heightKd:8},
   passCriterion:'Full requested 1–2 seconds; tilt <20 degrees throughout, final filtered angular speed <3 rad/s, absolute COM rise <10 cm, no contacts or instability warnings. This is not takeoff/landing or biological validation.'},
  calibration:[],cases:[],completed:false};
 console.log(JSON.stringify({kind:'classical-control-preflight',dimensions:report.dimensions,inertia,execute:run}));
 if(!run){console.log('No simulation performed. Add --run for 51 finite-difference probes, at most 3 trim updates and one paired free-control comparison.');}
 else{
  await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);ownsOutput=true;
  await fs.writeFile(path.join(output,'fixed-nonwing.xml'),reducedXml,{flag:'wx'});
  const save=async()=>{await fs.writeFile(path.join(output,'result.tmp'),JSON.stringify(report)+'\n');await fs.rename(path.join(output,'result.tmp'),path.join(output,'result.json'));};
  await save();
  const base=measure(center,'center');report.calibration.push(base);const B=Array.from({length:4},()=>new Array(25).fill(0));
  for(let j=0;j<25;j++){
   const eps=j===24?.05:.15,minus=[...center],plus=[...center];minus[j]-=eps;plus[j]+=eps;
   const low=measure(minus,`${controlNames[j]} minus`),high=measure(plus,`${controlNames[j]} plus`);report.calibration.push(low,high);
   for(let i=0;i<4;i++)B[i][j]=(high.response[i]-low.response[i])/(2*eps);
   if(j%6===0){await save();console.log(JSON.stringify({kind:'calibration-progress',completed:report.calibration.length,total:51}));}
  }
  report.responseJacobian=B;let trim=[...center],trimY=[...base.response];report.trim=[];
  for(let iteration=0;iteration<3;iteration++){
   trim=allocate(B,trim,trimY,[1,0,0,0],center,trim,120);
   const observation=measure(trim,`trim ${iteration+1}`);trimY=observation.response;report.trim.push(observation);await save();
   console.log(JSON.stringify({kind:'hover-trim',iteration:iteration+1,power:trim[24],response:trimY}));
  }
  report.trimControls=trim;report.trimResponse=trimY;await save();
  for(const mode of ['open_loop','closed_loop']){
   const result=simulate(mode,trim,B,trimY,inertia);report.cases.push(result);await save();
   console.log(JSON.stringify({kind:mode,pass:result.meetsDeclaredPositiveControl,...result.metrics,warnings:result.warnings}));
  }
  assert.equal(report.cases[0].initialHash,report.cases[1].initialHash,'Paired warm states differ');
  report.pairedInitialStateIdentical=true;
  for(const [file,expected]of Object.entries(hashes))assert.equal(sha(await fs.readFile(path.join(root,file))),expected,`Source changed during assay: ${file}`);
  report.sourceUnchanged=true;report.completed=true;await save();
  console.log(JSON.stringify({output,completed:true,positiveControl:report.cases[1].meetsDeclaredPositiveControl}));
 }
}catch(error){
 if(run&&report&&ownsOutput){report.failure=String(error.stack||error);try{await fs.writeFile(path.join(output,'failure.json'),JSON.stringify(report)+'\n',{flag:'wx'});}catch{}}
 throw error;
}finally{data.delete();model.delete();}
