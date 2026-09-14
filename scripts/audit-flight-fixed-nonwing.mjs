// Native mechanical control, deliberately separate from training/runtime.
// Default: compile + check invariants, no mj_step. Execute only with --run.
// node scripts/audit-flight-fixed-nonwing.mjs --run --output=reports/flight-fixed-nonwing
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {measureFlightKinematics} from '../web/training/flight-observation.js';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),execute=args.includes('--run');
assert(args.every(arg=>arg==='--run'||arg==='--check-only'||arg.startsWith('--output=')),'Unknown argument');
assert(!(execute&&args.includes('--check-only')),'Choose --run or --check-only');
const output=path.resolve(root,args.find(arg=>arg.startsWith('--output='))?.slice(9)||'reports/flight-fixed-nonwing');
assert(output.startsWith(path.join(root,'reports')+path.sep),'Output must be inside reports');
const duration=1,warmup=.1,force=1,phase=0,sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const files=['scripts/audit-flight-fixed-nonwing.mjs','web/flybody-wings.js','web/training/flight-parameters.js',
 'web/training/flight-observation.js','models/flybody-mujoco.xml','models/flybody-mujoco.json',
 'models/flybody-wing-actuation.json','packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
 'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
const buffers=await Promise.all(files.map(file=>fs.readFile(path.join(root,file))));
const sourceHashes=Object.fromEntries(files.map((file,i)=>[file,sha(buffers[i])])),xml=String(buffers[4]),metadata=JSON.parse(buffers[5]);
assert.equal(sha(buffers[4]),metadata.xml_sha256,'XML and metadata must describe the same model');
assert.deepEqual(metadata.wing_actuation,JSON.parse(buffers[6]),'Embedded wing table differs from the standalone calibration');
const wingNames=['left','right'].flatMap(side=>['yaw','roll','pitch'].map(axis=>`wing_${axis}_${side}`));
const nonwing=metadata.joints.filter(joint=>!wingNames.includes(joint.name));
const removedNames=new Set(nonwing.map(joint=>joint.name)),removedJoints=[];
// Exported model has named, self-closing joints; defaults have no name and are
// left untouched. Removing a hinge at qpos0 preserves its body's rest frame.
let reducedXml=xml.replace(/<joint\b[^>]*\/>/g,tag=>{
 const name=/\bname="([^"]+)"/.exec(tag)?.[1];
 if(!removedNames.has(name))return tag;
 removedJoints.push(name);return '';
});
assert.deepEqual([...removedJoints].sort(),[...removedNames].sort(),'Each nonwing hinge must be removed once');
const removedActuators=[],retainedActuators=[];
assert.equal((xml.match(/<actuator>/g)||[]).length,1,'Expected one actuator section');
reducedXml=reducedXml.replace(/<actuator>([\s\S]*?)<\/actuator>/,(_,section)=>'<actuator>'+section.replace(/<([a-z]+)\b[^>]*\/>/g,tag=>{
 const name=/\bname="([^"]+)"/.exec(tag)?.[1],joint=/\bjoint="([^"]+)"/.exec(tag)?.[1];
 assert(name&&metadata.actuators.some(actuator=>actuator.name===name),'Unknown actuator in exported model');
 if(wingNames.includes(name)){assert.equal(joint,name);retainedActuators.push(name);return tag;}
 removedActuators.push(name);return '';
})+'</actuator>');
assert.deepEqual([...retainedActuators].sort(),[...wingNames].sort());
assert.equal(removedActuators.length,metadata.actuators.length-6);

const mj=await loadMujoco(),models=[],rigs=[];
const clamp=(x,a=-1,b=1)=>Math.max(a,Math.min(b,x));
const maxDifference=(a,b)=>{assert.equal(a.length,b.length);let result=0;for(let i=0;i<a.length;i++)result=Math.max(result,Math.abs(a[i]-b[i]));return result;};
const nameId=(model,kind,name)=>{const id=mj.mj_name2id(model,mj.mjtObj[kind].value,name);assert(id>=0,`Missing ${kind}: ${name}`);return id;};
const enumName=(model,kind,index)=>mj.mj_id2name(model,mj.mjtObj[kind].value,index);
const rootPose=[0,0,3,1,0,0,0],noSteering={left:{},right:{}};
function makeRig(kind,source){
 const model=mj.MjModel.from_xml_string(source);models.push(model);
 const joints=wingNames.map(name=>{const id=nameId(model,'mjOBJ_JOINT',name),old=metadata.joints.find(j=>j.name===name);
  return {...old,id,qpos:model.jnt_qposadr[id],dof:model.jnt_dofadr[id],range:Array.from(model.jnt_range.slice(id*2,id*2+2))};});
 const actuators=wingNames.map(name=>{const id=nameId(model,'mjOBJ_ACTUATOR',name);return {name,id,
  joint:model.actuator_trnid[id*2],range:Array.from(model.actuator_ctrlrange.slice(id*2,id*2+2)),gain:model.actuator_gainprm[id*10]};});
 const meta={...metadata,joints,actuators},data=new mj.MjData(model),rest=Float64Array.from(model.qpos0),ctrl=new Float64Array(model.nu);
 const movableNonwing=kind==='dynamic'?nonwing.map(j=>({...j,id:nameId(model,'mjOBJ_JOINT',j.name)})):[];
 for(const joint of joints)rest[joint.qpos]=joint.neutral;
 for(const joint of movableNonwing){
  assert.equal(joint.neutral,model.qpos0[joint.qpos],'Calibration nonwing pose must be qpos0');
  const actuator=nameId(model,'mjOBJ_ACTUATOR',joint.name),range=model.actuator_ctrlrange;
  ctrl[actuator]=clamp(joint.neutral,range[actuator*2],range[actuator*2+1]);
 }
 rest.set(rootPose);
 const rig={kind,mj,model,data,metadata:meta,joints,actuators,movableNonwing,rest,ctrl};rigs.push(rig);return rig;
}
function reset(rig){
 const {model,data,rest,ctrl}=rig;mj.mj_resetData(model,data);data.qpos.set(rest);data.ctrl.set(ctrl);
 for(let id=0;id<model.nu;id++){const address=model.actuator_actadr[id];if(address>=0)data.act[address]=ctrl[id];}
 mj.mj_forward(model,data);
}
function invariantReport(dynamic,fixed){
 const a=dynamic.model,b=fixed.model;reset(dynamic);reset(fixed);
 assert.equal(a.nbody,b.nbody);assert.equal(a.ngeom,b.ngeom);assert.equal(a.nsite,b.nsite);
 assert.equal(b.nq,13);assert.equal(b.nv,12);assert.equal(b.njnt,7);assert.equal(b.nu,6);assert.equal(b.neq,0);
 assert.equal(b.jnt_type[0],0);assert.equal(a.nq-b.nq,nonwing.length);
 const errors={};
 for(const [count,kind]of [[a.nbody,'mjOBJ_BODY'],[a.ngeom,'mjOBJ_GEOM'],[a.nsite,'mjOBJ_SITE']])for(let i=0;i<count;i++)
  assert.equal(enumName(a,kind,i),enumName(b,kind,i),'Structural reduction reordered named geometry');
 for(const field of ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','geom_size','geom_pos','geom_quat']){
  errors[field]=maxDifference(a[field],b[field]);assert.equal(errors[field],0,`${field} changed`);
 }
 for(const field of ['xpos','xquat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat']){
  errors['neutral_'+field]=maxDifference(dynamic.data[field],fixed.data[field]);assert(errors['neutral_'+field]<1e-12,`${field}: neutral geometry changed`);
 }
 const optionFields=['timestep','density','viscosity'];for(const field of optionFields)assert.equal(a.opt[field],b.opt[field]);
 assert.equal(maxDifference(a.opt.gravity,b.opt.gravity),0);
 for(let i=0;i<6;i++){
  const ja=dynamic.joints[i],jb=fixed.joints[i],aa=dynamic.actuators[i].id,ab=fixed.actuators[i].id;
  for(const field of ['dof_armature','dof_damping'])assert.equal(a[field][ja.dof],b[field][jb.dof]);
  assert.equal(a.jnt_stiffness[ja.id],b.jnt_stiffness[jb.id]);
  for(const [field,stride]of [['actuator_gainprm',10],['actuator_biasprm',10],['actuator_dynprm',10],['actuator_gear',6],['actuator_ctrlrange',2]])
   assert.equal(maxDifference(a[field].slice(aa*stride,aa*stride+stride),b[field].slice(ab*stride,ab*stride+stride)),0,`Wing ${field} changed`);
 }
 return {passed:true,maximumNeutralGeometryError:Math.max(...Object.values(errors)),errors,
  rigidBodyMassG:Array.from(b.body_mass).reduce((sum,value)=>sum+value,0),
  dynamic:{nq:a.nq,nv:a.nv,nu:a.nu},fixed:{nq:b.nq,nv:b.nv,nu:b.nu},
  neutralContacts:{dynamic:dynamic.data.ncon,fixed:fixed.data.ncon},
  removedJoints:nonwing.map(j=>({name:j.name,qpos0:a.qpos0[j.qpos],armature:a.dof_armature[j.dof]})),removedActuators,
  interpretation:'Rigid-body masses, inertias, body frames, collision/visual geometry and six wing mechanics are preserved. Removed hinge coordinates, their generalized armatures and nonwing actuators are absent by design; the generalized mass matrices need not be equal.'};
}
function zeroApplied(data){for(const values of [data.xfrc_applied,data.qfrc_applied])for(const value of values)assert.equal(value,0,'External applied force detected');}
function fluidWrench(rig){
 const {model,data}=rig,r=model.jnt_bodyid[0],rotation=data.xmat,offset=r*9;
 const force=Array.from(data.qfrc_fluid.slice(0,3)),local=data.qfrc_fluid.slice(3,6);
 const torque=[0,1,2].map(i=>rotation[offset+i*3]*local[0]+rotation[offset+i*3+1]*local[1]+rotation[offset+i*3+2]*local[2]);
 const lever=[0,1,2].map(i=>data.subtree_com[r*3+i]-data.xpos[r*3+i]);
 const cross=[lever[1]*force[2]-lever[2]*force[1],lever[2]*force[0]-lever[0]*force[2],lever[0]*force[1]-lever[1]*force[0]];
 return {force,torque:torque.map((value,i)=>value-cross[i])};
}
function kinematics(rig){return measureFlightKinematics(rig);}
function snapshot(rig,wings){const {data,joints,actuators}=rig;return {time:data.time,...kinematics(rig),rootQpos:Array.from(data.qpos.slice(0,7)),
 rootQvel:Array.from(data.qvel.slice(0,6)),wingQpos:joints.map(j=>data.qpos[j.qpos]),wingQvel:joints.map(j=>data.qvel[j.dof]),
 wingCtrl:actuators.map(a=>data.ctrl[a.id]),phase:wings.phase,...wings.controlState()};}
function simulate(rig,onset){
 reset(rig);const {model,data,joints,actuators}=rig,wings=new FlyBodyWings(rig.metadata),h=metadata.timestep;
 const warmSeconds=onset==='warm'?warmup:0,totalSteps=Math.round(duration/h),warmSteps=Math.round(warmSeconds/h);
 wings.phase=((phase-2*Math.PI*wings.frequencyHz*warmSeconds)%(2*Math.PI)+2*Math.PI)%(2*Math.PI);
 let released=false,warmResetCount=0;
 const holdBeforeRelease=()=>{assert.equal(released,false,'A restrained pose reset attempted after release');
  data.qpos.set(rig.rest.subarray(0,7));data.qvel.fill(0,0,6);
  for(const joint of rig.movableNonwing){data.qpos[joint.qpos]=rig.rest[joint.qpos];data.qvel[joint.dof]=0;}warmResetCount++;};
 for(let step=0;step<warmSteps;step++){
  holdBeforeRelease();if(step%4===0)wings.step(data.qpos,data.ctrl,force,force,noSteering,h*4);mj.mj_step(model,data);
 }
 if(warmSteps){holdBeforeRelease();mj.mj_forward(model,data);}
 const initial=snapshot(rig,wings),start=data.time;released=true;
 let minimumUp=1,minimumHeight=initial.height,maximumHeight=initial.height,maximumOmega=0,omegaSquared=0;
 let firstTilt45=null,firstInversion=null,maximumNonwingDeviation=0,maximumNonwingSpeed=0,controlClipped=0;
 let contactSteps=0,maximumNativeContacts=0;
 let windowSteps=0,windowOmegaSquared=0,maximum20msRms=0,tailSteps=0;
 const forceSum=[0,0,0],torqueSum=[0,0,0],tailForce=[0,0,0],tailTorque=[0,0,0],samples=[];
 const sampleStride=Math.round(.002/h),rmsStride=Math.round(.02/h),fixedControlIds=Array.from({length:model.nu},(_,i)=>i).filter(i=>!actuators.some(a=>a.id===i));
 for(let step=0;step<totalSteps;step++){
  // Only six wing controls change here. The free root and all retained joints
  // integrate natively; the fixed plant has no nonwing joint coordinates.
  if(step%4===0)wings.step(data.qpos,data.ctrl,force,force,noSteering,h*4);
  mj.mj_step(model,data);const wrench=fluidWrench(rig),time=data.time-start;
  contactSteps+=Number(data.ncon>0);maximumNativeContacts=Math.max(maximumNativeContacts,data.ncon);
  const current=kinematics(rig),q=data.qpos,v=data.qvel,up=1-2*(q[4]*q[4]+q[5]*q[5]),omega=Math.hypot(v[3],v[4],v[5]);
  assert(q.every(Number.isFinite)&&v.every(Number.isFinite),'Nonfinite native trajectory');zeroApplied(data);
  for(const id of fixedControlIds)assert.equal(data.ctrl[id],rig.ctrl[id],'Nonwing control changed after release');
  minimumUp=Math.min(minimumUp,up);maximumOmega=Math.max(maximumOmega,omega);omegaSquared+=omega*omega;
  minimumHeight=Math.min(minimumHeight,current.height);maximumHeight=Math.max(maximumHeight,current.height);
  if(up<Math.SQRT1_2&&firstTilt45===null)firstTilt45=time;if(up<0&&firstInversion===null)firstInversion=time;
  for(const joint of rig.movableNonwing){maximumNonwingDeviation=Math.max(maximumNonwingDeviation,Math.abs(q[joint.qpos]-joint.neutral));maximumNonwingSpeed=Math.max(maximumNonwingSpeed,Math.abs(v[joint.dof]));}
  controlClipped+=Number(actuators.some(a=>Math.abs(data.ctrl[a.id]-a.range[0])<1e-10||Math.abs(data.ctrl[a.id]-a.range[1])<1e-10));
  windowOmegaSquared+=omega*omega;windowSteps++;
  if(windowSteps===rmsStride){maximum20msRms=Math.max(maximum20msRms,Math.sqrt(windowOmegaSquared/windowSteps));windowOmegaSquared=0;windowSteps=0;}
  for(let i=0;i<3;i++){forceSum[i]+=wrench.force[i];torqueSum[i]+=wrench.torque[i];if(time>=.75){tailForce[i]+=wrench.force[i];tailTorque[i]+=wrench.torque[i];}}
  if(time>=.75)tailSteps++;
  if((step+1)%sampleStride===0)samples.push([time,...current.position,current.verticalSpeed,up,omega,...Array.from(q.slice(3,7)),...Array.from(v.slice(3,6))]);
 }
 const final=snapshot(rig,wings),weight=metadata.mass_g*981;
 return {kind:rig.kind,onset,freeSeconds:data.time-start,nativeSteps:totalSteps,warmupSeconds:warmSeconds,warmResetCount,
  poseResetsAfterRelease:0,appliedExternalForceMaximum:0,initial,final,
  metrics:{comRiseCm:final.height-initial.height,minimumHeightCm:minimumHeight,maximumHeightCm:maximumHeight,
   maximumTiltDegrees:Math.acos(clamp(minimumUp))*180/Math.PI,firstTilt45Seconds:firstTilt45,firstInversionSeconds:firstInversion,
   maximumAngularSpeed:maximumOmega,rmsAngularSpeed:Math.sqrt(omegaSquared/totalSteps),maximumNonoverlapping20msRmsAngularSpeed:maximum20msRms,
   controlClippedStepFraction:controlClipped/totalSteps,maximumNonwingDeviationRad:maximumNonwingDeviation,maximumNonwingSpeed,
   selfContactStepFraction:contactSteps/totalSteps,maximumNativeContacts,
   meanFluidForce:forceSum.map(x=>x/totalSteps),meanFluidTorqueAtCOM:torqueSum.map(x=>x/totalSteps),
   lastQuarterMeanFluidLiftBodyweights:tailForce[2]/tailSteps/weight,lastQuarterMeanFluidTorqueAtCOM:tailTorque.map(x=>x/tailSteps)},
  sampleColumns:['timeSeconds','COMxCm','COMyCm','COMzCm','COMverticalSpeedCmPerSecond','up','angularSpeedRadPerSecond','qw','qx','qy','qz','omegaX','omegaY','omegaZ'],samples};
}

let report;
try{
 const dynamic=makeRig('dynamic',xml),fixed=makeRig('fixed_nonwing',reducedXml),invariants=invariantReport(dynamic,fixed);
 report={schemaVersion:1,createdAt:new Date().toISOString(),kind:'fixed-nonwing-native-mechanical-control',nativeVersion:mj.mj_versionString(),sourceHashes,
  modelHashes:{dynamic:sha(xml),fixed_nonwing:sha(reducedXml)},invariants,
  scope:'Four bounded airborne mechanics controls; no BANC, optimizer, food task, takeoff/landing test, or stability claim from compilation checks.',
  inputs:{leftMuscleForce:force,rightMuscleForce:force,steering:{left:{},right:{}},interpreter:'All 27 multipliers equal 1',releasePhaseRad:phase,
   frequencyHz:metadata.wing_actuation.frequency_hz,freeSeconds:duration,warmupSeconds:warmup,rootPose},
  methodology:{fixedPlant:`Remove ${nonwing.length} nonwing hinge coordinates at calibration qpos0 and all ${removedActuators.length} nonwing/adhesion actuators; retain rigid body frames, masses and inertias. No equality constraints added.`,
   dynamicPlant:'Original articulated plant with constant neutral nonwing actuator targets; no BANC leg signals.',
   warmup:'Each plant develops its own periodic wing state for 100ms with root and any nonwing joints restrained. Initial wing states are recorded, not assumed equal across plants.',
   afterRelease:'Root is completely free. No qpos/qvel writes, pose resets, root actuators, or applied external forces. Only the six wing controls update.',
   forces:'Native fluid force/moment from the retained pre-integration cache, rotated to world axes and shifted to its matching complete-fly COM; then refresh kinematics for poststep COM/attitude.',
   limitations:['Airborne start and no habitat contacts; this is not grounded takeoff or landing training.','The fixed pose is calibration-neutral, not the upstream springref-retracted flight pose.','Removing hinge DoFs removes their generalized armature terms; all modeled rigid-body masses and inertias remain.','Structurally welded bodies also acquire native welded-body collision filtering; self-contact counts are reported so contact differences cannot be mistaken for a pure inertial effect.','One phase and one constant input level per onset; no parameter fitting or automatic winner selection.']},cases:[]};
 console.log(JSON.stringify({kind:'fixed-nonwing-invariants',...invariants,execute}));
 if(execute){
  // Exclusive directory creation prevents an assay rerun from overwriting
  // prior observations, source snapshots, or partially completed results.
  await fs.mkdir(path.dirname(output),{recursive:true});await fs.mkdir(output);
  await fs.writeFile(path.join(output,'fixed-nonwing.xml'),reducedXml,{flag:'wx'});
  await fs.writeFile(path.join(output,'fixed-wing-metadata.json'),JSON.stringify(fixed.metadata)+'\n',{flag:'wx'});
  const save=async()=>{const temporary=path.join(output,'result.tmp');await fs.writeFile(temporary,JSON.stringify(report)+'\n');await fs.rename(temporary,path.join(output,'result.json'));};
  await save();
  for(const onset of ['cold','warm'])for(const rig of [dynamic,fixed]){
   try{
    const result=simulate(rig,onset);report.cases.push(result);await save();
    console.log(JSON.stringify({kind:rig.kind,onset,...result.metrics}));
   }catch(error){
    report.completed=false;report.failure={kind:rig.kind,onset,message:String(error?.message||error),nativeTime:rig.data.time};
    await save();throw error;
   }
  }
  for(const [file,expected]of Object.entries(sourceHashes))assert.equal(sha(await fs.readFile(path.join(root,file))),expected,'Source changed during assay: '+file);
  report.sourceUnchanged=true;report.completed=true;await save();
  await fs.writeFile(path.join(output,'README.md'),`# Fixed nonwing mechanics control\n\nAll four one-second native runs are retained in result.json. Inputs are identical; the root is free after release. This is not training or a takeoff/landing validation.\n\nThe fixed plant removes nonwing joint coordinates at the calibration pose; the dynamic plant retains them under neutral actuator commands. Warm cases have a 100ms restrained startup, with independently recorded initial wing states. No pose resets occur after release.\n\nSee the model/source hashes, neutral-geometry invariants, removed joint/armature list, whole-body COM fluid moments, and 2ms samples in result.json.\n`,{flag:'wx'});
 }else console.log('No simulation performed. Add --run to execute four bounded mechanical controls.');
}finally{for(const rig of rigs)rig.data.delete();for(const model of models)model.delete();}
