// Native mechanical assays, never a training controller or biological success claim.
// node scripts/audit-flight-parameter-sensitivity.mjs [--output=reports/flight-parameter-sensitivity]
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {FLIGHT_PARAMETER_NAMES,STEERING_MUSCLE_TYPES,flightParametersToInterpreter} from '../web/training/flight-parameters.js';
import {measureFlightKinematics} from '../web/training/flight-observation.js';

const args=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split('=')));
const output=args.output||'reports/flight-parameter-sensitivity';
const files=['scripts/audit-flight-parameter-sensitivity.mjs','web/flybody-wings.js','web/training/flight-parameters.js',
  'web/training/flight-observation.js','web/training/config.json','models/flybody-mujoco.xml','models/flybody-mujoco.json',
  'models/flybody-wing-actuation.json','packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
const buffers=await Promise.all(files.map(file=>fs.readFile(file)));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const metadata=JSON.parse(buffers[6]),config=JSON.parse(buffers[4]),mj=await loadMujoco();
assert.equal(FLIGHT_PARAMETER_NAMES.length,27);
assert.deepEqual(config.parameters.map(p=>p.name),FLIGHT_PARAMETER_NAMES);
const zero=Array(27).fill(0),h=metadata.timestep,wingDt=h*4,phases=Array.from({length:8},(_,i)=>.17+i*2*Math.PI/8);
const model=mj.MjModel.from_xml_string(String(buffers[5])),data=new mj.MjData(model);
const body={mj,model,data,metadata},weight=metadata.mass_g*981,wingTemplate=new FlyBodyWings(metadata);
const joints=wingTemplate.joints,actuators=wingTemplate.actuators,neutral=Float64Array.from(data.qpos),restCtrl=new Float64Array(model.nu);
const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
const maxAbs=values=>Math.max(0,...values.map(Math.abs));
const subtract=(a,b)=>a.map((value,i)=>value-b[i]);
const pick=(values,records,key)=>records.map(record=>values[record[key]]);
for(const joint of metadata.joints)neutral[joint.qpos]=joint.neutral;
neutral.set([0,0,3,1,0,0,0]);
for(const actuator of metadata.actuators){
  const joint=metadata.joints.find(j=>j.id===actuator.joint);
  if(joint&&!joint.name.startsWith('wing_'))restCtrl[actuator.id]=clamp(joint.neutral,...actuator.range);
}
function reset(q=neutral,v=new Float64Array(model.nv)){
  mj.mj_resetData(model,data);data.qpos.set(q);data.qvel.set(v);data.ctrl.set(restCtrl);
  for(const actuator of metadata.actuators){
    const address=model.actuator_actadr[actuator.id];
    if(address>=0)data.act[address]=restCtrl[actuator.id];
  }
}
const noSteering=()=>({left:{},right:{}});
function command(parameters,phase,deployment,drive,steering,q){
  const wing=new FlyBodyWings(metadata),ctrl=Float64Array.from(restCtrl);
  wing.setInterpreterParameters(flightParametersToInterpreter(parameters));
  wing.phase=phase;wing.deployment.fill(deployment);
  // A frequency change first changes phase, then the next command. Compare at
  // equal elapsed time rather than artificially forcing equal final phases.
  for(let step=0;step<2;step++)wing.step(q,ctrl,drive[0],drive[1],steering,wingDt);
  return {wing,ctrl};
}
function snapshot(phase,deployment,drive){
  const q=Float64Array.from(neutral),v=new Float64Array(model.nv);
  const center=command(zero,phase,deployment,drive,noSteering(),q).wing.target;
  const epsilon=1e-5;
  const before=command(zero,phase-epsilon,deployment,drive,noSteering(),q).wing.target;
  const after=command(zero,phase+epsilon,deployment,drive,noSteering(),q).wing.target;
  for(const [i,joint] of joints.entries()){
    q[joint.qpos]=center[i];
    v[joint.dof]=(after[i]-before[i])/(2*epsilon)*2*Math.PI*metadata.wing_actuation.frequency_hz;
  }
  return {q,v};
}
function mechanicalSample(parameters,phase,deployment,drive,steering,state){
  reset(state.q,state.v);
  const {wing,ctrl}=command(parameters,phase,deployment,drive,steering,state.q);
  data.ctrl.set(ctrl);mj.mj_forward(model,data);
  const sample={ctrl:pick(data.ctrl,actuators,'id'),wingActuatorForce:pick(data.actuator_force,actuators,'id'),
    rootAcceleration:Array.from(data.qacc.slice(0,6)),wingAcceleration:pick(data.qacc,joints,'dof'),
    deployment:Array.from(wing.deployment),power:Array.from(wing.power),frequencyHz:wing.frequencyHz,
    controlClipped:actuators.map(a=>Math.abs(data.ctrl[a.id]-a.range[0])<1e-10||Math.abs(data.ctrl[a.id]-a.range[1])<1e-10),
    targetClipped:joints.map((j,i)=>Math.abs(wing.target[i]-j.range[0])<1e-10||Math.abs(wing.target[i]-j.range[1])<1e-10)};
  const rootBefore=Array.from(data.qvel.slice(0,6)),fluid=new Float64Array(3);
  // One millisecond of genuinely free native response from the identical state.
  // No root/limb reset, corrective force, or desired pose is applied here.
  for(let step=0;step<20;step++){
    if(step&&step%4===0)wing.step(data.qpos,data.ctrl,drive[0],drive[1],steering,wingDt);
    mj.mj_step(model,data);
    for(let axis=0;axis<3;axis++)fluid[axis]+=data.qfrc_fluid[axis]/20;
  }
  sample.realizedRootAcceleration=Array.from(data.qvel.slice(0,6),(value,i)=>(value-rootBefore[i])/(20*h));
  sample.responseMeanFluidForce=Array.from(fluid);
  sample.responseWingAngles=pick(data.qpos,joints,'qpos');
  assert(data.qpos.every(Number.isFinite)&&data.qvel.every(Number.isFinite));
  assert(data.xfrc_applied.every(value=>value===0)&&data.qfrc_applied.every(value=>value===0));
  return sample;
}
function sensitivity(index){
  const parameter=config.parameters[index],muscle=index>=3?STEERING_MUSCLE_TYPES[Math.floor((index-3)/2)]:null;
  const minus=zero.slice(),plus=zero.slice();
  minus[index]=Math.max(parameter.min/2,-.15);plus[index]=Math.min(parameter.max/2,.15);
  assert(minus[index]<0&&plus[index]>0);
  const summary={name:parameter.name,muscle,logPerturbations:[minus[index],plus[index]],coefficientMultipliers:[Math.exp(minus[index]),Math.exp(plus[index])],
    sides:[],responsive:false};
  for(const side of muscle?['left','right']:['bilateral']){
    const maxima={ctrl:0,wingActuatorForce:0,wingAcceleration:0,rootAcceleration:0,realizedRootAcceleration:0,responseMeanFluidForce:0,responseWingAngles:0};
    const metrics={side,samples:0,unresponsiveSamples:0,controlClippedSamples:0,targetClippedSamples:0,contralateralCtrlDelta:0,maxima,witness:null};
    const drive=[.6,.4],deployment=index===1?.94:1;
    const steering=noSteering();if(muscle)steering[side][muscle]=.35;
    for(const phase of phases){
      const state=snapshot(phase,deployment,drive),baseline=mechanicalSample(zero,phase,deployment,drive,steering,state);
      for(const [sign,vector] of [[-1,minus],[1,plus]]){
        const changed=mechanicalSample(vector,phase,deployment,drive,steering,state),deltas={};
        for(const key of Object.keys(maxima)){deltas[key]=subtract(changed[key],baseline[key]);maxima[key]=Math.max(maxima[key],maxAbs(deltas[key]));}
        metrics.samples++;
        if(maxAbs(deltas.ctrl)<=1e-9)metrics.unresponsiveSamples++;
        if(baseline.controlClipped.some(Boolean)||changed.controlClipped.some(Boolean))metrics.controlClippedSamples++;
        if(baseline.targetClipped.some(Boolean)||changed.targetClipped.some(Boolean))metrics.targetClippedSamples++;
        if(muscle){
          const other=side==='left'?deltas.ctrl.slice(3,6):deltas.ctrl.slice(0,3);
          metrics.contralateralCtrlDelta=Math.max(metrics.contralateralCtrlDelta,maxAbs(other));
        }
        if(!metrics.witness||maxAbs(deltas.ctrl)>metrics.witness.maxCtrlDelta)
          metrics.witness={phase,sign,drive,steering,initialDeployment:deployment,maxCtrlDelta:maxAbs(deltas.ctrl),baseline,changed,deltas};
      }
    }
    metrics.responsive=maxima.ctrl>1e-8&&maxima.wingActuatorForce>1e-8&&maxima.realizedRootAcceleration>1e-6;
    if(muscle)assert(metrics.contralateralCtrlDelta<1e-12,'A unilateral muscle signal altered the opposite side command');
    summary.sides.push(metrics);
  }
  summary.responsive=summary.sides.every(row=>row.responsive);
  return summary;
}
function freeControl(warmupSeconds){
  reset();const wing=new FlyBodyWings(metadata),steering=noSteering(),root=Array.from(neutral.slice(0,7));
  const nonWing=metadata.joints.filter(j=>!j.name.startsWith('wing_'));
  const hold=()=>{data.qpos.set(root);data.qvel.fill(0,0,6);for(const j of nonWing){data.qpos[j.qpos]=neutral[j.qpos];data.qvel[j.dof]=0;}};
  mj.mj_forward(model,data);
  for(let step=0;step<Math.round(warmupSeconds/h);step++){
    hold();if(step%4===0)wing.step(data.qpos,data.ctrl,1,1,steering,wingDt);mj.mj_step(model,data);
  }
  if(warmupSeconds){hold();mj.mj_forward(model,data);}
  const initial=measureFlightKinematics(body),start=data.time,samples=[],cycles=[];
  let maxOmega=0,minUp=1,firstInversion=null,firstTilt45=null,integratedLift=0,maxApplied=0;
  let withinAngularLimit=0,controlClippedSteps=0,targetClippedSteps=0,cycleStart=0,cycleOmega=[0,0,0],cycleSquared=0,cycleMax=0,cycleCount=0;
  const frequency=wing.frequencyHz,seconds=.5,totalSteps=Math.round(seconds/h);
  for(let step=0;step<totalSteps;step++){
    if(step%4===0)wing.step(data.qpos,data.ctrl,1,1,steering,wingDt);
    mj.mj_step(model,data);
    const q=data.qpos,v=data.qvel,time=data.time-start,up=1-2*(q[4]*q[4]+q[5]*q[5]);
    const angular=Array.from(v.slice(3,6)),omega=Math.hypot(...angular);
    maxOmega=Math.max(maxOmega,omega);minUp=Math.min(minUp,up);withinAngularLimit+=Number(omega<=20);
    controlClippedSteps+=Number(actuators.some(a=>Math.abs(data.ctrl[a.id]-a.range[0])<1e-10||Math.abs(data.ctrl[a.id]-a.range[1])<1e-10));
    targetClippedSteps+=Number(joints.some((j,i)=>Math.abs(wing.target[i]-j.range[0])<1e-10||Math.abs(wing.target[i]-j.range[1])<1e-10));
    if(up<0&&firstInversion===null)firstInversion=time;
    if(up<Math.SQRT1_2&&firstTilt45===null)firstTilt45=time;
    integratedLift+=data.qfrc_fluid[2]*h;
    cycleSquared+=omega*omega;cycleMax=Math.max(cycleMax,omega);cycleCount++;
    for(let axis=0;axis<3;axis++)cycleOmega[axis]+=angular[axis];
    if(time-cycleStart>=1/frequency){
      const mean=cycleOmega.map(value=>value/cycleCount);
      cycles.push({time,meanAngularVelocity:mean,meanVectorMagnitude:Math.hypot(...mean),rmsAngularSpeed:Math.sqrt(cycleSquared/cycleCount),maximumAngularSpeed:cycleMax,up});
      cycleStart=time;cycleOmega=[0,0,0];cycleSquared=0;cycleCount=0;cycleMax=0;
    }
    if(step%40===39){
      const com=measureFlightKinematics(body);
      samples.push({time,...com,up,angularVelocity:angular,angularSpeed:omega,power:Array.from(wing.power),
        fluidLiftBodyweights:data.qfrc_fluid[2]/weight,wingActuatorForce:pick(data.actuator_force,actuators,'id')});
    }
    maxApplied=Math.max(maxApplied,maxAbs(Array.from(data.xfrc_applied)),maxAbs(Array.from(data.qfrc_applied)));
    assert(q.every(Number.isFinite)&&v.every(Number.isFinite));
  }
  assert.equal(maxApplied,0);
  const final=measureFlightKinematics(body);
  return {warmupSeconds,freeSeconds:data.time-start,drive:[1,1],steering,initial,final,
    metrics:{maximumAngularSpeed:maxOmega,minimumUp:minUp,maximumTiltDegrees:Math.acos(clamp(minUp,-1,1))*180/Math.PI,
      firstInversionSeconds:firstInversion,firstTilt45Seconds:firstTilt45,angularSpeedWithin20Fraction:withinAngularLimit/totalSteps,
      controlClippedStepFraction:controlClippedSteps/totalSteps,targetClippedStepFraction:targetClippedSteps/totalSteps,
      meanFluidLiftBodyweights:integratedLift/seconds/weight,comRiseCm:final.height-initial.height,
      maxCycleMeanAngularSpeed:Math.max(...cycles.map(c=>c.meanVectorMagnitude)),maxCycleRmsAngularSpeed:Math.max(...cycles.map(c=>c.rmsAngularSpeed)),maximumAppliedForce:maxApplied},
    samples,cycles};
}

const report={createdAt:new Date().toISOString(),nativeVersion:mj.mj_versionString(),
  sourceSha256:Object.fromEntries(files.map((file,i)=>[file,sha(buffers[i])])),parameterCount:27,phaseCount:phases.length,nativeStepSeconds:h,
  scope:'Controlled muscle-force/native-body assays; no BANC simulation, learned controller, or flight-success claim.',
  units:{length:'cm',mass:'g',time:'s',angularVelocity:'rad/s',actuatorForce:'g cm^2/s^2 for wing joint torques',fluidForce:'g cm/s^2'},
  method:{sensitivity:'One coefficient at a time, paired common frozen qpos/qvel across 8 phases; side-specific named force 0.35, left/right power 0.6/0.4. Two interpreter ticks expose frequency phase advance. Deployment tau uses 0.94 deployment; other coefficients use 1. Native forward dynamics and 1ms free response follow.',
    freeControl:'Constant equal wing force 1, no steering, airborne level initial condition diagnostic only. Cold onset and 100ms root/nonwing kinematic restraint warmup followed by 500ms fully free native dynamics. There are no habitat contacts in this model.',
    forceTiming:'Instantaneous qacc/actuator forces are native forward dynamics at common state. Response mean fluid forces follow 20 actual native steps. Free-control fluid force is pre-integration while COM/orientation are end-step.',
    limits:['The sensitivity assay establishes parameter influence, not learnability or stable closed-loop flight.','Root qacc is a generalized-coordinate measurement; COM trajectories are reported separately.','The warmup restraint and airborne diagnostic reset are not part of training.','Cycle metrics average native root-local angular vectors over approximately one nominal wing period; sample bins are rounded to 50us.','Claw capacity is the nominal maximum force per actuator, not the observed loaded force during grounded takeoff.']},
  freeControls:[],sensitivity:[],clawCapacity:[]};
await fs.mkdir(output,{recursive:true});
try{
  for(const actuator of metadata.actuators.filter(a=>a.name.startsWith('adhere_claw_'))){
    const gain=model.actuator_gainprm[actuator.id*10];
    report.clawCapacity.push({name:actuator.name,controlRange:actuator.range,gain,maximumForceBodyweights:gain*actuator.range[1]/weight});
  }
  for(const warmup of [0,.1]){
    const result=freeControl(warmup);report.freeControls.push(result);
    console.log(JSON.stringify({assay:'equal-wing-free-control',warmupSeconds:warmup,...result.metrics}));
    await fs.writeFile(`${output}/result.json`,JSON.stringify(report)+'\n');
  }
  for(let index=0;index<27;index++){
    const result=sensitivity(index);report.sensitivity.push(result);
    console.log(JSON.stringify({parameter:result.name,responsive:result.responsive,
      maxCtrlDelta:Math.max(...result.sides.map(s=>s.maxima.ctrl)),maxRootResponseAccelerationDelta:Math.max(...result.sides.map(s=>s.maxima.realizedRootAcceleration))}));
  }
  report.allParametersMechanicallyResponsive=report.sensitivity.length===27&&report.sensitivity.every(row=>row.responsive);
  report.unresponsiveParameters=report.sensitivity.filter(row=>!row.responsive).map(row=>row.name);
  report.maximumContralateralCommandLeak=Math.max(...report.sensitivity.flatMap(row=>row.sides.map(side=>side.contralateralCtrlDelta)));
  await fs.writeFile(`${output}/result.json`,JSON.stringify(report)+'\n');
  const rows=report.freeControls.map(({warmupSeconds,metrics:m})=>`| ${warmupSeconds?'100ms restrained warmup':'Cold onset'} | ${m.comRiseCm.toFixed(3)} | ${m.maximumTiltDegrees.toFixed(1)} | ${m.maximumAngularSpeed.toFixed(2)} | ${m.maxCycleMeanAngularSpeed.toFixed(2)} | ${m.meanFluidLiftBodyweights.toFixed(3)} | ${(m.controlClippedStepFraction*100).toFixed(2)}% |`).join('\n');
  const markdown=`# Native flight interpreter assay\n\nGenerated ${report.createdAt}. Reproduce from the repository root with \`node scripts/audit-flight-parameter-sensitivity.mjs\`. The [JSON report](result.json) pins source/WASM hashes and records paired mechanical witnesses, COM trajectories, and cycle summaries.\n\nAll **${report.sensitivity.filter(p=>p.responsive).length}/27** coefficients affect controls, native actuator forces, and the body's free response at the tested states. Every named muscle is tested separately on each side. Maximum contralateral command change is ${report.maximumContralateralCommandLeak}; no tested sensitivity command or target hits a bound. This demonstrates usable mechanical influence, not successful learning or measured biological coefficients.\n\nBoth positive controls use equal normalized wing muscle force 1 with no steering for 500ms. The cold control starts from neutral wings. The warm control restrains the root and nonwing joints during 100ms of native wing motion, then releases everything. After release there are no pose resets or applied root forces. The model has no habitat contacts; descent below the initial height is possible.\n\n| Control | COM rise (cm) | Maximum tilt (degrees) | Maximum angular speed (rad/s) | Maximum cycle mean angular vector (rad/s) | Mean fluid lift / weight | Control clipped steps |\n|---|---:|---:|---:|---:|---:|---:|\n${rows}\n\nThe current equal-wing command produces aerodynamic lift near bodyweight but does not establish stable level flight. The orientation excursions and large cycle mean angular velocity indicate accumulated rotation in addition to wingbeat recoil. These trials do not evaluate takeoff from support or landing.\n\nSensitivity uses eight phases, common qpos/qvel across paired perturbations, left/right force 0.6/0.4, and a named steering force of 0.35 on one side at a time. Commands are measured after two 200µs interpreter ticks; native forward dynamics and a 1ms free response follow. Deployment starts at 0.94 for its time-constant assay and 1 otherwise. Bounds and full witness arrays are in the JSON.\n\nEach claw's nominal maximum adhesion is ${report.clawCapacity[0].maximumForceBodyweights.toFixed(3)} bodyweights. This is actuator capacity only: actual claw loading and its causal effect on grounded takeoff were not tested.\n`;
  await fs.writeFile(`${output}/README.md`,markdown);
  if(!report.allParametersMechanicallyResponsive)process.exitCode=1;
}finally{data.delete();model.delete();}
