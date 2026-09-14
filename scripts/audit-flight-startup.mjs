// Diagnostic native mechanics only. No BANC simulation or runtime modification.
// node scripts/audit-flight-startup.mjs [--output=reports/flight-startup]
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {measureFlightKinematics} from '../web/training/flight-observation.js';

const args=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split('=')));
const output=args.output||'reports/flight-startup',duration=1;
const files=['scripts/audit-flight-startup.mjs','web/flybody-wings.js','web/training/flight-parameters.js',
  'web/training/flight-observation.js','models/flybody-mujoco.xml','models/flybody-mujoco.json',
  'models/flybody-wing-actuation.json','packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
const buffers=await Promise.all(files.map(file=>fs.readFile(file)));
const metadata=JSON.parse(buffers[5]),mj=await loadMujoco(),model=mj.MjModel.from_xml_string(String(buffers[4])),data=new mj.MjData(model);
const body={mj,model,data,metadata},root=model.jnt_bodyid[0],offset=root*3,h=metadata.timestep,wingDt=h*4,TAU=2*Math.PI;
const template=new FlyBodyWings(metadata),joints=template.joints,actuators=template.actuators,weight=metadata.mass_g*981;
const neutral=Float64Array.from(data.qpos),restCtrl=new Float64Array(model.nu),nonWing=metadata.joints.filter(j=>!j.name.startsWith('wing_'));
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),norm=v=>Math.hypot(...v),subtract=(a,b)=>a.map((x,i)=>x-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const add=(a,b)=>a.map((x,i)=>x+b[i]),scale=(a,k)=>a.map(x=>x*k),zero3=()=>[0,0,0];
const rotate=(matrix,v)=>[0,1,2].map(i=>matrix[i*3]*v[0]+matrix[i*3+1]*v[1]+matrix[i*3+2]*v[2]);
for(const joint of metadata.joints)neutral[joint.qpos]=joint.neutral;
neutral.set([0,0,3,1,0,0,0]);
for(const actuator of metadata.actuators){
  const joint=nonWing.find(j=>j.id===actuator.joint);
  if(joint)restCtrl[actuator.id]=clamp(joint.neutral,...actuator.range);
}
function reset(){
  mj.mj_resetData(model,data);data.qpos.set(neutral);data.ctrl.set(restCtrl);
  for(const a of metadata.actuators)if(model.actuator_actadr[a.id]>=0)data.act[model.actuator_actadr[a.id]]=restCtrl[a.id];
  mj.mj_forward(model,data);
}
function kinematics(){
  const com=measureFlightKinematics(body),momentum=Array.from(data.subtree_angmom.slice(offset,offset+3));
  return {...com,angularMomentum:momentum,angularMomentumMagnitude:norm(momentum)};
}
function cachedFluidWrench(){
  // qfrc_fluid has world translation and root-local rotational coordinates.
  // Use pre-integration native geometry with its matching fluid force, then
  // shift the moment from the thorax/free-joint origin to the whole-body COM.
  const force=Array.from(data.qfrc_fluid.slice(0,3)),localMoment=Array.from(data.qfrc_fluid.slice(3,6));
  const rotation=data.xmat.slice(root*9,root*9+9),origin=Array.from(data.xpos.slice(offset,offset+3));
  const lever=subtract(Array.from(data.subtree_com.slice(offset,offset+3)),origin);
  return {force,torque:subtract(rotate(rotation,localMoment),cross(lever,force))};
}
function bucket(start){return {start,steps:0,upSum:0,minimumUp:1,omegaSquared:0,maximumOmega:0,omegaSum:zero3(),
  forceSum:zero3(),torqueSum:zero3(),momentumSum:zero3(),maximumMomentum:0,controlClipped:0,targetClipped:0};}
function accumulate(b,value){
  b.steps++;b.upSum+=value.up;b.minimumUp=Math.min(b.minimumUp,value.up);
  b.omegaSquared+=value.omega*value.omega;b.maximumOmega=Math.max(b.maximumOmega,value.omega);
  b.omegaSum=add(b.omegaSum,value.angular);b.forceSum=add(b.forceSum,value.wrench.force);b.torqueSum=add(b.torqueSum,value.wrench.torque);
  b.momentumSum=add(b.momentumSum,value.momentum);b.maximumMomentum=Math.max(b.maximumMomentum,norm(value.momentum));
  b.controlClipped+=Number(value.controlClipped);b.targetClipped+=Number(value.targetClipped);
}
function complete(b,end){
  const mean=key=>scale(b[key],1/b.steps),omega=mean('omegaSum');
  return {startSeconds:b.start,endSeconds:end,steps:b.steps,minimumUp:b.minimumUp,meanUp:b.upSum/b.steps,
    maximumAngularSpeed:b.maximumOmega,rmsAngularSpeed:Math.sqrt(b.omegaSquared/b.steps),meanAngularVelocity:omega,meanAngularVectorMagnitude:norm(omega),
    meanFluidForce:mean('forceSum'),meanFluidLiftBodyweights:b.forceSum[2]/b.steps/weight,meanFluidTorqueAtCOM:mean('torqueSum'),
    meanAngularMomentum:mean('momentumSum'),maximumAngularMomentum:b.maximumMomentum,
    controlClippedStepFraction:b.controlClipped/b.steps,targetClippedStepFraction:b.targetClipped/b.steps};
}
function run({startPhase,deploymentTauScale,warmupSeconds=0}){
  reset();const wing=new FlyBodyWings(metadata),steering={left:{},right:{}};
  wing.setInterpreterParameters({deploymentTauScale});
  // Warm trials select the release phase; cold trials select the first phase.
  wing.phase=((startPhase-TAU*wing.frequencyHz*warmupSeconds)%TAU+TAU)%TAU;
  const hold=()=>{data.qpos.set(neutral.subarray(0,7));data.qvel.fill(0,0,6);
    for(const joint of nonWing){data.qpos[joint.qpos]=neutral[joint.qpos];data.qvel[joint.dof]=0;}};
  let warmMomentumSum=zero3(),warmSamples=0,warmMaxMomentum=0;
  const warmCount=Math.round(warmupSeconds/h),warmTailSeconds=10/wing.frequencyHz;
  for(let step=0;step<warmCount;step++){
    hold();
    if(step*h>=warmupSeconds-warmTailSeconds){
      const current=kinematics();warmMomentumSum=add(warmMomentumSum,current.angularMomentum);warmSamples++;
      warmMaxMomentum=Math.max(warmMaxMomentum,current.angularMomentumMagnitude);
    }
    if(step%4===0)wing.step(data.qpos,data.ctrl,1,1,steering,wingDt);
    mj.mj_step(model,data);
  }
  if(warmupSeconds){hold();mj.mj_forward(model,data);}
  const releasePhase=wing.phase,initial=kinematics(),start=data.time,total=bucket(0),samples=[],cycles=[],windows=[];
  let cycle=bucket(0),window=bucket(0),minimumHeight=initial.height,firstTilt45=null,firstInversion=null,angularWithin20=0;
  let maximumHeight=initial.height,minimumUp=1,lastWrench=null,previousWindow=0;
  const totalSteps=Math.round(duration/h);
  for(let step=0;step<totalSteps;step++){
    if(step%4===0)wing.step(data.qpos,data.ctrl,1,1,steering,wingDt);
    mj.mj_step(model,data);
    const wrench=cachedFluidWrench();lastWrench=wrench;
    const time=data.time-start,current=kinematics(),q=data.qpos,up=1-2*(q[4]*q[4]+q[5]*q[5]);
    const angular=Array.from(data.qvel.slice(3,6)),omega=norm(angular);
    const controlClipped=actuators.some(a=>Math.abs(data.ctrl[a.id]-a.range[0])<1e-10||Math.abs(data.ctrl[a.id]-a.range[1])<1e-10);
    const targetClipped=joints.some((j,i)=>Math.abs(wing.target[i]-j.range[0])<1e-10||Math.abs(wing.target[i]-j.range[1])<1e-10);
    const value={up,angular,omega,wrench,momentum:current.angularMomentum,controlClipped,targetClipped};
    for(const row of [total,cycle,window])accumulate(row,value);
    minimumHeight=Math.min(minimumHeight,current.height);maximumHeight=Math.max(maximumHeight,current.height);minimumUp=Math.min(minimumUp,up);
    if(up<Math.SQRT1_2&&firstTilt45===null)firstTilt45=time;
    if(up<0&&firstInversion===null)firstInversion=time;
    angularWithin20+=Number(omega<=20);
    if(time-cycle.start>=1/wing.frequencyHz){cycles.push(complete(cycle,time));cycle=bucket(time);}
    const nextWindow=time<.1?0:time<.5?1:time<.75?2:3;
    if(nextWindow!==previousWindow){windows.push(complete(window,time));window=bucket(time);previousWindow=nextWindow;}
    if(step%200===199)samples.push([time,current.height,current.verticalSpeed,up,omega,...current.angularMomentum]);
    assert(data.qpos.every(Number.isFinite)&&data.qvel.every(Number.isFinite));
    assert(data.xfrc_applied.every(x=>x===0)&&data.qfrc_applied.every(x=>x===0));
  }
  if(window.steps)windows.push(complete(window,duration));
  const final=kinematics(),stats=complete(total,duration),tail=cycles.slice(-10);
  const tailMean=key=>scale(tail.reduce((sum,row)=>add(sum,row[key]),zero3()),1/tail.length);
  return {startPhase,releasePhase,deploymentTauScale,warmupSeconds,initial,final,
    releaseMomentumComparedWithWarmCycle:warmSamples?{lastWarmCycles:10,meanAngularMomentum:scale(warmMomentumSum,1/warmSamples),
      maximumAngularMomentum:warmMaxMomentum,releaseMinusMean:subtract(initial.angularMomentum,scale(warmMomentumSum,1/warmSamples))}:null,
    metrics:{...stats,comRiseCm:final.height-initial.height,minimumHeightCm:minimumHeight,maximumHeightCm:maximumHeight,
      maximumTiltDegrees:Math.acos(clamp(minimumUp,-1,1))*180/Math.PI,firstTilt45Seconds:firstTilt45,firstInversionSeconds:firstInversion,
      angularSpeedWithin20Fraction:angularWithin20/totalSteps,maxCycleMeanAngularSpeed:Math.max(...cycles.map(c=>c.meanAngularVectorMagnitude))},
    terminal:{lastInstantaneousFluidWrench:lastWrench,lastTenCycleMeanFluidTorqueAtCOM:tailMean('meanFluidTorqueAtCOM'),
      lastTenCycleMeanAngularMomentum:tailMean('meanAngularMomentum'),lastTenCycleMeanAngularVelocity:tailMean('meanAngularVelocity'),
      lastTenCycleMeanFluidForce:tailMean('meanFluidForce')},windows,lastTenCycles:tail,samples};
}

const report={createdAt:new Date().toISOString(),nativeVersion:mj.mj_versionString(),nativeStepSeconds:h,freeSeconds:duration,
  sourceSha256:Object.fromEntries(files.map((file,i)=>[file,createHash('sha256').update(buffers[i]).digest('hex')])),
  units:{length:'cm',mass:'g',time:'s',angularVelocity:'rad/s',angularMomentum:'g cm^2/s',torque:'g cm^2/s^2'},
  scope:'Fixed wing table, equal normalized muscle force 1, no steering, no neural simulation. Fully free root after diagnostic release.',
  methodology:{phases:'8 equally spaced starting phases including zero; warm trials select release phase after 100ms restrained warmup.',
    torque:'Native fluid generalized force is transformed from root-local moment to world axes and shifted from root origin to whole-body COM, using the matching pre-integration geometry.',
    angularMomentum:'Native subtree_angmom at the complete fly subtree, measured after refreshing kinematics and subtree velocity; world axes about whole-body COM.',
    angularVelocity:'Root-local angular vectors; cycle means cover approximately one nominal wing period rounded to 50us.',
    restraint:'Warmup resets the root and nonwing joints during 100ms. There are no pose resets, force injections, or hidden stabilization after release.',
    samples:'Rows: [time, COM height, COM vertical speed, up, angular speed, angular momentum x/y/z], every 10ms.',
    limits:['This model has no habitat contacts, so no takeoff or landing is evaluated.','Cold airborne onset is a mechanical diagnostic, not the training initial state.','Subtree angular momentum describes native rigid bodies; generalized armature treatment must be considered when interpreting momentum conservation.']},
  wingJointArmature:joints.map(j=>({name:j.name,armature:model.dof_armature[j.dof]})),cold:[],warm:[]};
await fs.mkdir(output,{recursive:true});
try{
  for(const deploymentTauScale of [.5,1,2])for(let i=0;i<8;i++){
    const result=run({startPhase:i*TAU/8,deploymentTauScale});report.cold.push(result);
    console.log(JSON.stringify({type:'cold',tau:deploymentTauScale,phase:i,...Object.fromEntries(['comRiseCm','maximumTiltDegrees','maximumAngularSpeed','controlClippedStepFraction'].map(key=>[key,result.metrics[key]]))}));
  }
  for(let i=0;i<8;i++){
    const result=run({startPhase:i*TAU/8,deploymentTauScale:1,warmupSeconds:.1});report.warm.push(result);
    console.log(JSON.stringify({type:'warm',phase:i,releaseAngularMomentum:result.initial.angularMomentum,
      comRiseCm:result.metrics.comRiseCm,maximumTiltDegrees:result.metrics.maximumTiltDegrees}));
  }
  const criteria={maximumTiltDegrees:45,minimumCOMRiseCm:0,minimumLastQuarterFluidLiftBodyweights:.8};
  report.diagnosticScreen={criteria,meaning:'Screen for a promising fixed mechanical startup, not a training reward or learned flight claim.',
    coldPassing:report.cold.filter(r=>r.metrics.maximumTiltDegrees<=criteria.maximumTiltDegrees&&r.metrics.comRiseCm>=0&&r.windows.at(-1).meanFluidLiftBodyweights>=.8).map(r=>({phase:r.startPhase,tau:r.deploymentTauScale})),
    warmPassing:report.warm.filter(r=>r.metrics.maximumTiltDegrees<=criteria.maximumTiltDegrees&&r.metrics.comRiseCm>=0&&r.windows.at(-1).meanFluidLiftBodyweights>=.8).map(r=>({phase:r.startPhase,tau:r.deploymentTauScale}))};
  await fs.writeFile(`${output}/result.json`,JSON.stringify(report)+'\n');
  const rows=[...report.cold,...report.warm].map(r=>`| ${r.warmupSeconds?'Warm':'Cold'} | ${r.deploymentTauScale} | ${(r.startPhase*180/Math.PI).toFixed(0)} | ${r.metrics.comRiseCm.toFixed(2)} | ${r.metrics.maximumTiltDegrees.toFixed(1)} | ${r.metrics.maximumAngularSpeed.toFixed(1)} | ${(100*r.metrics.controlClippedStepFraction).toFixed(1)}% | ${norm(r.terminal.lastTenCycleMeanFluidTorqueAtCOM).toExponential(2)} |`).join('\n');
  await fs.writeFile(`${output}/README.md`,`# Fixed-table flight startup sweep\n\nGenerated ${report.createdAt}. Run \`node scripts/audit-flight-startup.mjs\` from the repository root. [Source-pinned JSON](result.json) includes all 32 outcomes, sampled trajectories, time-window statistics, terminal cycles, and release angular momentum.\n\nEach trial drives both wings with constant normalized muscle force 1, without steering, for 1s in free native dynamics. Twenty-four cold trials sweep eight phases and deployment time scales 0.5/1/2. Eight warm trials use scale 1 and sweep release phase after 100ms of root/nonwing restraint. No restraint or applied root forces remain after release. These are airborne mechanics diagnostics with no habitat contacts, not training or takeoff/landing trials.\n\n${report.diagnosticScreen.coldPassing.length}/24 cold and ${report.diagnosticScreen.warmPassing.length}/8 warm trials pass the descriptive screen: maximum tilt ≤45°, nonnegative COM rise, and last-quarter mean vertical fluid force ≥0.8 bodyweights. This screen does not substitute for the training flight criterion.\n\n| Onset | Deployment time scale | Phase (degrees) | COM rise (cm) | Maximum tilt (degrees) | Maximum angular speed (rad/s) | Clipped control steps | Final 10-cycle mean COM torque magnitude |\n|---|---:|---:|---:|---:|---:|---:|---:|\n${rows}\n\nCOM torque is in g·cm²/s² and refers to fluid forces only. It is transformed using the native root orientation and shifted to whole-body COM; terminal values are means across the final ten wing cycles. Angular momentum is the native rigid-body subtree result, about COM in world axes. Joint armature values are retained in JSON; this quantity should not be described as including unmodeled rotor angular momentum.\n`);
}finally{data.delete();model.delete();}
