// Read-only command/actuator contract audit. No brain run or body trajectory.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {muscleFuel} from '../web/banc/embodiment.js';

const paths={xml:'models/flybody-mujoco.xml',metadata:'models/flybody-mujoco.json',io:'data/prepared/banc888/io.json',
  reference:'models/flybody-flight-reference.json',wing:'web/flybody-wings.js',physics:'web/flybody-physics.js',
  muscles:'packages/banc-runtime/native/core.cpp',core:'packages/banc-runtime/dist/core.wasm',
  embodiment:'web/banc/embodiment.js',generator:'scripts/distill-flybody-wings.py',
  upstreamTask:'references/flybody/flybody/tasks/flight_imitation.py',
  documentation:'reports/flybody-flight-repair.md',readme:'README.md',
  snapshot:'reports/observation-60min-20260913/latest.json'};
const buffers=Object.fromEntries(await Promise.all(Object.entries(paths).map(async([key,path])=>[key,await fs.readFile(path)])));
const parse=key=>JSON.parse(buffers[key]),metadata=parse('metadata'),io=parse('io'),reference=parse('reference'),snapshot=parse('snapshot');
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const model=mj.MjModel.from_xml_string(String(buffers.xml)),data=new mj.MjData(model);
const clamp=x=>Math.max(0,Math.min(1,x)),mean=a=>a.reduce((s,x)=>s+x,0)/Math.max(1,a.length);
const hash=x=>createHash('sha256').update(x).digest('hex');
const h=metadata.timestep,wingDt=h*4,muscleDt=h*20,c=metadata.wing_actuation;
const q=Float64Array.from(model.qpos0);q.set([0,0,2,1,0,0,0],0);
for(const joint of metadata.joints)q[joint.qpos]=joint.neutral;
const neutral=()=>({left:{},right:{}}),controls=()=>new Float64Array(model.nu);
const report={createdAt:new Date().toISOString(),command:'node scripts/audit-flybody-wing-contract.mjs',
  scope:'Read-only native MuJoCo WASM instantaneous torque plus actual WASM muscle / JS wing command probes. No body integration, posture clamp, root force, trained policy or BANC behavior experiment.',
  sourceHashes:Object.fromEntries(Object.entries(paths).map(([key,path])=>[path,hash(buffers[key])])),
  sourceRevision:metadata.source.revision,model:{nq:model.nq,nv:model.nv,nu:model.nu,na:model.na},
  units:{time:'seconds',jointPosition:'radians',jointVelocity:'radians/second',nativeTorque:'g cm^2 / s^2',
    wingForceInput:'normalized reduced-muscle effort; Fmax=1 and fictive length=1, velocity=0, not a native torque'},
  confirmedIndexSignUnitOrTimebaseBugs:[],limitations:[]};

const fields=a=>({name:a.name,id:a.id,joint:a.joint,transmissionType:model.actuator_trntype[a.id],
  nativeJoint:model.actuator_trnid[a.id*2],gain:model.actuator_gainprm[a.id*10],
  bias:Array.from(model.actuator_biasprm.slice(a.id*10,a.id*10+3)),
  dynamicsType:model.actuator_dyntype[a.id],dynamicsParameter:model.actuator_dynprm[a.id*10],
  controlRange:Array.from(model.actuator_ctrlrange.slice(a.id*2,a.id*2+2)),
  gear:Array.from(model.actuator_gear.slice(a.id*6,a.id*6+6))});
const names=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
report.wingActuators=names.map(name=>{
  const joint=metadata.joints.find(j=>j.name===name),actuator=metadata.actuators.find(a=>a.name===name);
  const native=fields(actuator);
  assert.equal(native.nativeJoint,joint.id);
  assert.equal(model.jnt_qposadr[joint.id],joint.qpos);
  assert.equal(model.jnt_dofadr[joint.id],joint.dof);
  assert.equal(native.gain,18);
  assert.deepEqual(native.controlRange,[-1,1]);
  return {...native,qpos:joint.qpos,dof:joint.dof,axis:Array.from(model.jnt_axis.slice(joint.id*3,joint.id*3+3)),
    neutral:joint.neutral,springReference:model.qpos_spring[joint.qpos],passiveStiffness:model.jnt_stiffness[joint.id],
    passiveDamping:model.dof_damping[joint.dof]};
});

// An instantaneous displaced joint state, not a restrained-root simulation.
// qfrc_actuator is measured after mj_forward, independently of the JS formula.
const wing=new FlyBodyWings(metadata),ctrl=controls(),yaw=wing.joints[0],yawActuator=wing.actuators[0];
data.qpos.set(q);data.qpos[yaw.qpos]-=.01;data.qvel.fill(0);
wing.step(data.qpos,data.ctrl,0,0,neutral(),wingDt);mj.mj_forward(model,data);
report.zeroDriveDisplacedWing={joint:yaw.name,displacementRad:data.qpos[yaw.qpos]-yaw.neutral,
  targetRad:wing.target[0],control:data.ctrl[yawActuator.id],
  actuatorForce:data.actuator_force[yawActuator.id],actuatorJointTorque:data.qfrc_actuator[yaw.dof],
  passiveJointTorque:data.qfrc_passive[yaw.dof],
  actuatorToPassiveRatio:data.qfrc_actuator[yaw.dof]/data.qfrc_passive[yaw.dof],
  externalAppliedForceMaximum:Math.max(...data.qfrc_applied.map(Math.abs),...data.xfrc_applied.map(Math.abs)),
  interpretation:'Native force actuators plus a position-error command produce a tonic target-restoring servo even when normalized power-muscle force is zero. This is an adapter assumption, not passive-only muscle mechanics.'};
assert(Math.abs(report.zeroDriveDisplacedWing.actuatorJointTorque-.18)<1e-12);
assert(Math.abs(report.zeroDriveDisplacedWing.passiveJointTorque-.0001)<1e-12);
assert.equal(report.zeroDriveDisplacedWing.externalAppliedForceMaximum,0);

const cold=new FlyBodyWings(metadata),coldSteering=neutral();let maximumColdControl=0,maximumColdPower=0;
const coldSteps=1250;
for(let i=0;i<coldSteps;i++){
  cold.step(q,ctrl,0,0,coldSteering,wingDt);
  maximumColdControl=Math.max(maximumColdControl,...cold.actuators.map(a=>Math.abs(ctrl[a.id])));
  maximumColdPower=Math.max(maximumColdPower,...cold.power);
}
const expectedPhase=(coldSteps*wingDt*c.frequency_hz*2*Math.PI)%(2*Math.PI);
report.coldZeroDrive={commandDurationSeconds:coldSteps*wingDt,maximumControl:maximumColdControl,
  maximumEffectivePower:maximumColdPower,phase:cold.phase,expectedPhase,
  phaseAbsoluteError:Math.abs(cold.phase-expectedPhase),
  interpretation:'The phase clock runs, but neutral wings receive zero targets/errors and no oscillatory command. There is no cold-start autonomous flapping in this module.'};
assert.equal(maximumColdControl,0);assert.equal(maximumColdPower,0);assert(Math.abs(cold.phase-expectedPhase)<1e-10);

report.openingDoseResponse=[];
for(const opening of [1,.99,.95,.9,.85,.84,.5,0]){
  const w=new FlyBodyWings(metadata),s={left:{iii1_muscle:1-opening},right:{iii1_muscle:1-opening}};
  for(let i=0;i<2500;i++)w.step(q,ctrl,1,1,s,wingDt);
  const expected=Math.min(opening*opening,clamp((opening-c.deployment_before_beating)/(1-c.deployment_before_beating)));
  assert(Math.abs(w.power[0]-expected)<1e-10);
  report.openingDoseResponse.push({powerMuscleInput:1,opening,settledDeployment:w.deployment[0],
    effectivePower:w.power[0],openingSquared:opening*opening,absoluteDeploymentGate:clamp((w.deployment[0]-c.deployment_before_beating)/(1-c.deployment_before_beating)),
    formulaSteadyLimit:expected});
}
report.deploymentSemantics={tauSeconds:c.deployment_tau_s,absoluteThreshold:c.deployment_before_beating,
  firstBeatAtFullOpeningSeconds:-c.deployment_tau_s*Math.log(1-c.deployment_before_beating),
  codeFormula:'min(mean(leftForce,rightForce)*opening^2, clamp((deployment-0.85)/0.15))',
  steadyFormula:'min(power*opening^2, clamp((opening-0.85)/0.15))',
  documentationQuote:'Deployment takes 12 ms and the beat engages above 85% deployment; opening also scales transmission.',
  classification:'Consistent with explicitly documented absolute deployment prior. Double attenuation is real, but not a proven semantic error.',
  alternativeNotAFix:'Normalizing deployment by requested opening would gate on percent of requested deployment. That changes the model semantics and needs an independent biological/mechanical justification; it is not a correction established here.',
  anatomicalIdentity:'Current code uses III1 for retraction and retains I1 as a separate steering channel. The historical report flags and corrects its previous I1 attribution.'};
const unilateral=new FlyBodyWings(metadata),unilateralSteering=neutral();
for(let i=0;i<2500;i++)unilateral.step(q,ctrl,1,0,unilateralSteering,wingDt);
report.unilateralDrive={inputLeft:1,inputRight:0,effectiveLeft:unilateral.power[0],effectiveRight:unilateral.power[1],
  classification:'Documented shared thoracic power oscillator; bilateral steering/opening remain separate. Not an actuator-index swap.'};
assert.deepEqual(Array.from(unilateral.power),[.5,.5]);

const legs=metadata.actuators.filter(a=>/_T[123]_(left|right)$/.test(a.name)&&a.joint!==null);
const byType=new Map();
for(const a of legs){
  const type=a.name.replace(/_T[123]_(left|right)$/,''),j=metadata.joints.find(j=>j.id===a.joint),native=fields(a);
  const values={gain:native.gain,bias:native.bias,dynamicsType:native.dynamicsType,dynamicsParameter:native.dynamicsParameter,
    passiveStiffness:model.jnt_stiffness[j.id],passiveDamping:model.dof_damping[j.dof]};
  const group=byType.get(type)??{type,names:[],values};
  assert.deepEqual(values,group.values,`Leg type ${type} differs across limbs`);
  group.names.push(a.name);byType.set(type,group);
}
report.legActuatorGroups=Array.from(byType.values());
assert.equal(legs.length,42);assert([...byType.values()].every(x=>x.names.length===6));
report.gainComparison={wing:18,proximalLeg:.8,distalLeg:.4,wingToProximal:18/.8,wingToDistal:18/.4,
  sameAcrossAllSixLegsByType:true,sameAcrossWingAndLegs:false,
  caveat:'Comparing raw gains alone does not establish an appropriate muscle strength ratio: wings use clipped radian errors with no actuator bias, legs use filtered position commands and affine restoring bias. Both retain native passive springs.'};

const lookup=new Map(io.motor_neurons.map((m,i)=>[m.index,i]));
assert.equal(snapshot.state.rates.length,io.motor_neurons.length);
const wingGroups=io.muscles.filter(m=>m.kind==='asynchronous_wing'||m.kind==='wing_steering_assumption');
const observed=wingGroups.map(m=>{
  const rates=m.indices.map(index=>{assert(lookup.has(index));return snapshot.state.rates[lookup.get(index)];});
  return {joint:m.joint,target:m.target,kind:m.kind,indices:m.indices,meanRateHz:mean(rates),excitation:clamp(mean(rates)/80)};
});
const get=(side,target)=>observed.find(m=>m.joint.endsWith(side)&&m.target===target).excitation;
report.observedSnapshot={file:paths.snapshot,sha256:hash(buffers.snapshot),frame:snapshot.index,at:snapshot.at,version:snapshot.version,
  bodyTime:snapshot.state.bodyTime,observedWingPower:snapshot.state.wingPower,energy:snapshot.state.internal.energy,
  fuel:muscleFuel(snapshot.state.internal.energy),airborne:snapshot.state.airborne,upZ:snapshot.state.upZ,
  groups:observed,sides:['left','right'].map(side=>({side,dlmExcitation:get(side,'dorsal_longitudinal_muscle'),
    dvmExcitation:get(side,'dorsoventral_muscle'),iii1Excitation:get(side,'iii1_muscle'),b1Excitation:get(side,'b1_muscle'),
    meanPowerExcitation:(get(side,'dorsal_longitudinal_muscle')+get(side,'dorsoventral_muscle'))/2,
    openingExcitationProxy:1-clamp(get(side,'iii1_muscle')-get(side,'b1_muscle'))})),
  caveat:'Snapshot rates establish current excitations only. Actual muscle forces, fatigue, prior activation and deployment are not present; they are not reconstructed or inferred to equal these proxies.'};

// Actual compiled WASM muscle dynamics, with normalized wing lengths/velocities
// matching production. This is a held-input command assay, not behavior replay.
report.heldRateCommandProbes=[];
for(const kind of ['all_power_80Hz','both_dlm_only_80Hz','left_dlm_only_80Hz','snapshot_rates_held']){
  const muscles=new WasmMuscles(core,wingGroups.length),w=new FlyBodyWings(metadata),input=new Float32Array(wingGroups.length*5);
  const samples=[];let state;
  for(let step=0;step<200;step++){
    wingGroups.forEach((group,index)=>{
      let excitation=0;
      if(kind==='snapshot_rates_held')excitation=observed[index].excitation;
      else if(group.kind==='asynchronous_wing'){
        if(kind==='all_power_80Hz')excitation=1;
        if(kind==='both_dlm_only_80Hz'&&group.target==='dorsal_longitudinal_muscle')excitation=1;
        if(kind==='left_dlm_only_80Hz'&&group.target==='dorsal_longitudinal_muscle'&&group.joint.endsWith('left'))excitation=1;
      }
      input.set([excitation,1,0,1,kind==='snapshot_rates_held'?muscleFuel(snapshot.state.internal.energy):1],index*5);
    });
    state=muscles.step(input,muscleDt);
    const power={left:[],right:[]},steering=neutral();
    wingGroups.forEach((group,index)=>{
      const side=group.joint.endsWith('left')?'left':'right';
      if(group.kind==='asynchronous_wing')power[side].push(state[index*3+2]);else steering[side][group.target]=state[index*3+2];
    });
    const driveLeft=mean(power.left),driveRight=mean(power.right);
    for(let sub=0;sub<5;sub++)w.step(q,ctrl,driveLeft,driveRight,steering,wingDt);
    if([9,49,99,199].includes(step))samples.push({seconds:(step+1)*muscleDt,driveLeft,driveRight,
      opening:Array.from(w.opening),deployment:Array.from(w.deployment),effectivePower:Array.from(w.power)});
  }
  report.heldRateCommandProbes.push({kind,durationSeconds:200*muscleDt,samples,
    finalPowerMuscles:wingGroups.map((group,index)=>({group,index})).filter(x=>x.group.kind==='asynchronous_wing').map(({group,index})=>({
      joint:group.joint,target:group.target,activation:state[index*3],fatigue:state[index*3+1],force:state[index*3+2]}))});
  muscles.dispose();
}

report.timebaseAndOfficialContract={physicsStepSeconds:h,muscleUpdateSeconds:muscleDt,wingUpdateSeconds:wingDt,
  physicsStepsPerWingUpdate:4,wingUpdatesPerMuscleUpdate:5,
  directFrequencyHz:c.frequency_hz,official:Object.fromEntries(['base_beat_freq','rel_freq_range','ctrl_filter','rate'].map(key=>[key,reference.wingbeat_generator[key]])),
  officialFrequencyRangeHz:[218*(1-.05),218*(1+.05)],
  officialActionMapping:reference.action_mapping,
  officialFormula:'Wing command = learned residual + measured WPG target (radians) - current wing qpos (radians); native force actuator gain 18. Last user action changes WPG frequency.',
  directFormula:'Wing command = clipped reduced-adapter target (radians) - current qpos (radians), clipped to native [-1,1]; same gain 18. Target tables and fixed frequency replace the original WPG/residual contract.',
  classification:'Correct second/radian units, positive target-minus-current sign, exact name-to-joint mapping and 50/200 microsecond physics/wing cadence. Fixed 235.8 Hz is outside the released 207.1–228.9 Hz policy frequency range, but is an explicitly fitted alternative, not a milliseconds conversion bug.',
  stabilization:'The native body supplies inertia, passive springs/damping, fluids and contacts. It does not supply a posture stabilizer. The released policy provides state-dependent residuals/frequency; the direct wing adapter has no root-state inputs.'};
report.limitations.push('No free-flight or takeoff stability conclusion follows from these instantaneous/command-only probes.',
  'The pinned source and current production model have different retained DoFs and observation contracts. A common 65-actuator model exists as a separate reuse experiment; this report audits the active 56-actuator direct model.',
  'The 80 Hz rate scale, 15/40 ms muscle filters, 0.08/s fatigue term, shared DLM/DVM arithmetic means, opening competition and target-restoring torque are reduced-model priors, not identified BANC muscle physiology.');
report.passed=true;
data.delete();model.delete();
await fs.writeFile('reports/flybody-wing-contract-audit.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({passed:report.passed,model:report.model,zeroDrive:report.zeroDriveDisplacedWing,
  cold:report.coldZeroDrive,opening:report.openingDoseResponse,gain:report.gainComparison,
  snapshot:{frame:report.observedSnapshot.frame,bodyTime:report.observedSnapshot.bodyTime,wingPower:report.observedSnapshot.observedWingPower,sides:report.observedSnapshot.sides},
  held:report.heldRateCommandProbes.map(p=>({kind:p.kind,last:p.samples.at(-1)})),
  output:'reports/flybody-wing-contract-audit.json'},null,2));
