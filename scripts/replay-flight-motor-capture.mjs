// Exact recorded-MN native replay, followed by mechanical counterfactuals.
// node scripts/replay-flight-motor-capture.mjs --capture=path.json
// --fixture-only=true validates plumbing; it is never evidence of BANC behavior.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore,WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics,flybodyScene} from '../web/flybody-physics.js';
import {createHabitat} from '../web/body-world.js';
import {createContactFoodResolver} from '../web/flybody-contact-environment.js';
import {measureFlightObservation,measureFlightKinematics,upwardContactForce} from '../web/training/flight-observation.js';
import {makeFlightTelemetry} from '../web/training/flight-telemetry.js';

const args=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split('=')));
const output=args.output||'reports/flight-motor-replay',fixtureOnly=args['fixture-only']==='true';
const baselineOnly=args['baseline-only']==='true';
const exportMuscleProfiles=args['export-muscle-profiles']==='true';
const holdLastSeconds=Number(args['hold-last-seconds']||0);
assert(Number.isFinite(holdLastSeconds)&&holdLastSeconds>=0&&holdLastSeconds<=1&&Math.abs(holdLastSeconds/.002-Math.round(holdLastSeconds/.002))<1e-9);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const inputBytes=fixtureOnly?null:await fs.readFile(args.capture||path.join(output,'capture.json'));
const inputDocument=inputBytes?JSON.parse(inputBytes):null;
const virtualAssets=inputDocument?.virtualAssets||{};
assert(Object.keys(virtualAssets).every(url=>['/body-model/flybody-mujoco.xml','/body-model/flybody-mujoco.json','/training/config.json'].includes(url)),
  'Only captured model XML, metadata and manifest may override local data; executable runtime assets cannot be replaced');
const canonicalConfigBytes=await fs.readFile('web/training/config.json');
const configBytes=virtualAssets['/training/config.json']?Buffer.from(virtualAssets['/training/config.json']):canonicalConfigBytes;
const config=JSON.parse(configBytes);
const assetBytes=async(url,file)=>Object.hasOwn(virtualAssets,url)?Buffer.from(virtualAssets[url]):fs.readFile(file);
const sourcePaths={
  '/flybody-physics.js':'web/flybody-physics.js','/flybody-wings.js':'web/flybody-wings.js',
  '/flybody-habitat-collision.js':'web/flybody-habitat-collision.js','/flybody-contact-environment.js':'web/flybody-contact-environment.js',
  '/flybody-stance.js':'web/flybody-stance.js','/flybody-leg-actuation.js':'web/flybody-leg-actuation.js',
  '/banc-proboscis.js':'web/banc-proboscis.js','/banc/embodiment.js':'web/banc/embodiment.js',
  '/body-world.js':'web/body-world.js','/vendor/three.module.js':'web/vendor/three.module.js','/vendor/three.core.js':'web/vendor/three.core.js',
  '/training/flight-parameters.js':'web/training/flight-parameters.js','/training/flight-observation.js':'web/training/flight-observation.js',
  '/training/flight-telemetry.js':'web/training/flight-telemetry.js',
  '/banc-engine/src/wasm.js':'packages/banc-runtime/src/wasm.js','/banc-engine/src/model.js':'packages/banc-runtime/src/model.js',
  '/banc-engine/dist/core.js':'packages/banc-runtime/dist/core.js','/banc-engine/dist/core.wasm':'packages/banc-runtime/dist/core.wasm',
  '/body-engine/mujoco.js':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
  '/body-engine/mujoco.wasm':'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm',
  '/body-model/flybody-mujoco.xml':'models/flybody-mujoco.xml','/body-model/flybody-mujoco.json':'models/flybody-mujoco.json',
  '/banc-data/io.json':'data/prepared/banc888/io.json'};
const sourceHashes=Object.fromEntries(await Promise.all(Object.entries(sourcePaths).map(async([url,file])=>[url,hash(await assetBytes(url,file))])));
const manifestMismatches=Object.entries(sourceHashes).filter(([url,value])=>config.assets[url]!==value).map(([url,value])=>({url,manifest:config.assets[url]??null,local:value}));
assert.deepEqual(manifestMismatches,[],'Replay mechanics differ from the captured training manifest');
const [mj,core]=await Promise.all([loadMujoco(),createWasmCore()]);
const decode=(text,bits)=>{
  const bytes=Buffer.from(text,'base64');assert.equal(bytes.length%(bits/8),0);
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),array=bits===32?new Float32Array(bytes.length/4):new Float64Array(bytes.length/8);
  for(let i=0;i<array.length;i++)array[i]=bits===32?view.getFloat32(i*4,true):view.getFloat64(i*8,true);
  assert(array.every(Number.isFinite));return array;
};
const unpack=(record,bits)=>{assert.equal(record.encoding,`float${bits}-little-endian-base64`);return decode(record.data,bits);};
const pack64=array=>{const bytes=Buffer.alloc(array.length*8);for(let i=0;i<array.length;i++)bytes.writeDoubleLE(array[i],i*8);return bytes;};
const pack32=array=>{const bytes=Buffer.alloc(array.length*4);for(let i=0;i<array.length;i++)bytes.writeFloatLE(array[i],i*4);return bytes.toString('base64');};
const compareBits=(actual,expectedBase64)=>{
  const expected=Buffer.from(expectedBase64,'base64'),actualBytes=pack64(actual);assert.equal(actualBytes.length,expected.length);
  if(actualBytes.equals(expected))return {exact:true,maximumAbsoluteError:0};
  let first=-1,maximumAbsoluteError=0,mismatchedValues=0;
  for(let i=0;i<actual.length;i++){
    maximumAbsoluteError=Math.max(maximumAbsoluteError,Math.abs(actual[i]-expected.readDoubleLE(i*8)));
    if(!actualBytes.subarray(i*8,i*8+8).equals(expected.subarray(i*8,i*8+8))){if(first<0)first=i;mismatchedValues++;}
  }
  return {exact:false,maximumAbsoluteError,mismatchedValues,firstIndex:first,actual:actual[first],expected:expected.readDoubleLE(first*8),
    actualBits:actualBytes.subarray(first*8,first*8+8).toString('hex'),expectedBits:expected.subarray(first*8,first*8+8).toString('hex')};
};
function environment(habitat,model,names){return {surface:(x,y)=>habitat.surface(x*10,y*10).y/10,
  odor:(x,y,z)=>habitat.odor(x*10,z*10,y*10),foodForContact:createContactFoodResolver(mj,model,habitat.fruit,names),
  foodAt:(x,y)=>{const index=habitat.surface(x*10,y*10).fruitIndex;return index>=0?habitat.fruit[index]:null;}};}
function restore(body,initial){
  const state=unpack(initial.packed.nativeIntegration,64),spec=initial.nativeIntegration.spec;
  assert.equal(spec,mj.mjtState.mjSTATE_INTEGRATION.value);assert.equal(state.length,mj.mj_stateSize(body.model,spec));
  // This binding takes a NumberArray for setState, while getState uses an
  // owned DoubleBuffer. Inputs are copied synchronously by the native wrapper.
  mj.mj_setState(body.model,body.data,state,spec);mj.mj_forward(body.model,body.data);
  // Rebuild derived caches, then restore authoritative integration state,
  // including warmstart and equality flags changed by the native forward.
  mj.mj_setState(body.model,body.data,state,spec);
  const packed=initial.packed;
  body.muscles.core.HEAPF32.set(unpack(packed.muscleWasmState,32),body.muscles.state/4);
  body.muscleState=unpack(packed.muscleState,32);body.input.set(unpack(packed.muscleInput,32));body.activation.set(unpack(packed.activation,32));
  body.restPose=unpack(packed.restPose,64);body.restHeight=initial.restHeight;body.remainder=initial.remainder;
  Object.assign(body.internal,structuredClone(initial.internal));Object.assign(body.monitor,structuredClone(initial.monitor));
  body.food=structuredClone(initial.food);body.wings.setInterpreterParameters(initial.wing.interpreter);
  body.wings.phase=initial.wing.phase;body.wings.frequencyHz=initial.wing.frequencyHz;
  for(const field of ['deployment','power','target'])body.wings[field].set(unpack(packed.wing[field],64));
  packed.wing.residuals.forEach((row,i)=>body.wings.residuals[i].set(unpack(row,64)));
  body.refresh();
  for(const [key,value]of Object.entries(initial.wrapper))body[key]=value===null?undefined:structuredClone(value);
}
function newBody(capture){
  const scene=capture.scene,habitat=createHabitat(structuredClone(scene.fruit)),model=mj.MjModel.from_xml_string(scene.xml);
  model.hfield_data.set(decode(scene.heights,32));
  const body=new FlyBodyPhysics(mj,model,scene.metadata,scene.io,n=>new WasmMuscles(core,n),environment(habitat,model,scene.fruitGeomNames));
  restore(body,capture.initial);return {body,model,habitat};
}
function footLoads(body){
  const out=Array(6).fill(0),contacts=body.data.ncon?body.data.contact:null,weight=body.metadata.mass_g*981;
  try{for(let i=0;i<body.data.ncon;i++){
    const contact=contacts.get(i);
    try{
      const ids=contact.geom,a=body.geomBodyIds[ids[0]],b=body.geomBodyIds[ids[1]];
      if((a===0)===(b===0))continue;
      const foot=body.tasteBodyToLeg[a===0?b:a];if(foot<0)continue;
      mj.mj_contactForce(body.model,body.data,i,body.contactForce);
      out[foot]+=upwardContactForce(contact.frame,body.contactForce.GetView(),a===0)/weight;
    }finally{contact.delete();}
  }}finally{contacts?.delete();}
  return out;
}
function clawForces(body){
  const d=body.data,weight=body.metadata.mass_g*981;
  return body.metadata.actuators.filter(a=>a.name.startsWith('adhere_claw_')).map(a=>{
    const generalized=Array(6).fill(0),force=d.actuator_force[a.id],start=d.moment_rowadr[a.id],count=d.moment_rownnz[a.id];
    for(let k=start;k<start+count;k++){const column=d.moment_colind[k];if(column<6)generalized[column]+=d.actuator_moment[k]*force;}
    return {name:a.name,control:d.ctrl[a.id],gain:body.model.actuator_gainprm[a.id*10],scalarActuatorForce:force,
      scalarBodyweights:force/weight,rootGeneralizedForce:generalized,worldForceBodyweights:generalized.slice(0,3).map(x=>x/weight)};
  });
}
function observe(body,rates,ceiling=5.6){
  // Match training cadence: retained last native force/contact evaluation,
  // then refreshed COM. These observations never write integration state.
  const support=measureFlightObservation(body),feet=footLoads(body),claws=clawForces(body),telemetry=makeFlightTelemetry(body,rates);
  const com=measureFlightKinematics(body),q=body.data.qpos,angular=Array.from(body.data.qvel.slice(3,6));
  const up=1-2*(q[4]*q[4]+q[5]*q[5]),angularSpeed=Math.hypot(...angular);
  const appliedForceMaximum=Math.max(0,...body.data.qfrc_applied.map(Math.abs),...body.data.xfrc_applied.map(Math.abs));
  // Same fields and arithmetic as environment.js observe(), exported at every
  // native feedback boundary for pure offline rescoring. This does not run a
  // scorer or change controls, contacts, neural inputs, or integration state.
  const trainingObservation={finite:[body.time,...com.position,com.verticalSpeed,com.speedCmPerSecond,up,angularSpeed,...body.quaternion].every(Number.isFinite),
    up,angularSpeed,...com,radius:Math.hypot(...com.position.slice(0,2)),ceiling,
    wingPower:body.wingPower,...support,externalForce:appliedForceMaximum!==0};
  return {time:body.data.time,...com,up,angularVelocity:angular,angularSpeed,trainingObservation,
    ...support,footSupportBodyweights:feet,claws,wing:telemetry.wing,wingJoints:telemetry.wingJoints.rows,
    rootQpos:Array.from(q.slice(0,7)),rootQvel:Array.from(body.data.qvel.slice(0,6)),
    rootActuatorGeneralizedForce:Array.from(body.data.qfrc_actuator.slice(0,6)),
    appliedForceMaximum};
}
async function fixture(){
  const {createMotorReplayCapture}=await import('../web/test/training-motor-capture.js');
  const [xml,metadata,io,definition]=await Promise.all([fs.readFile('models/flybody-mujoco.xml','utf8'),
    fs.readFile('models/flybody-mujoco.json','utf8').then(JSON.parse),fs.readFile('data/prepared/banc888/io.json','utf8').then(JSON.parse),
    fs.readFile('web/habitat.json','utf8').then(JSON.parse)]);
  const habitat=createHabitat(definition.fruit.map(f=>({...f,remaining:10}))),scene=flybodyScene(xml,habitat),model=mj.MjModel.from_xml_string(scene.xml);
  model.hfield_data.set(scene.heights);const body=new FlyBodyPhysics(mj,model,metadata,io,n=>new WasmMuscles(core,n),environment(habitat,model,scene.fruitGeomNames));
  const fly=structuredClone(definition.flies[0]);fly.id=1;fly.brain={motorNeuronRates:Array(io.motor_neurons.length).fill(0)};
  body.place(fly.x/10,fly.z/10,fly.heading);
  const world={habitat,metadata,io,movementMode:'direct',motorCoupling:true,flightEnabled:true};
  const recorder=createMotorReplayCapture({sourceXml:xml,seconds:.04});
  const indices=io.motor_neurons.map(m=>m.index),rates=new Map(indices.map(i=>[i,0]));
  observe(body,rates);
  recorder.onInitialState({body,world,fly,motorIndices:indices,job:{seed:0,stage:'takeoff',parameters:Array(27).fill(0)},
    provenance:{configHash:hash(configBytes),modelFingerprint:config.modelFingerprint,backend:'synthetic-no-neural-model'},initialCondition:{fixture:true},initialObservation:{fixture:true}});
  try{for(let index=0;index<20;index++){
    fly.brain.motorNeuronRates=indices.map((id,k)=>Math.fround(20+(k%11)*3+index*.125));indices.forEach((id,k)=>rates.set(id,fly.brain.motorNeuronRates[k]));
    const timeBefore=body.time;body.step(rates,.002,{coupling:true,flight:true});
    recorder.onPhysicsStep({body,world,fly,index,durationSeconds:.002,timeBefore,neuralMs:(index+1)*2});observe(body,rates);
  }
  return {...recorder.finish({reason:'fixture',simSeconds:.04,steps:20,cancelled:false,success:false}),kind:'synthetic-native-motor-replay-fixture'};
  }finally{body.dispose();model.delete();}
}

await fs.mkdir(output,{recursive:true});
const input=fixtureOnly?await fixture():inputDocument;
const capture=input.kind?.includes('motor-replay')?input:input.evaluation?.motorReplay||input.trials?.[0]?.evaluation?.motorReplay;
assert(capture?.schemaVersion===1&&capture.steps.length&&capture.initial.packed&&capture.scene.nativeHeightfieldExact,'Incomplete motor capture');
const synthetic=capture.kind==='synthetic-native-motor-replay-fixture'||capture.provenance.backend==='synthetic-no-neural-model';
assert.equal(capture.physicalStateWritesByRecorder,0);assert.equal(capture.bodyBlockSeconds,.002);assert.equal(capture.initial.native.time,0);
assert.equal(capture.provenance.configHash,hash(configBytes),'Replay config differs from captured config');
assert.equal(capture.provenance.modelFingerprint,config.modelFingerprint);
assert.deepEqual(capture.motorNeuronIndices,capture.scene.io.motor_neurons.map(m=>m.index));
assert.equal(capture.dimensions.motorCount,capture.motorNeuronIndices.length);
assert.equal(hash(await assetBytes('/body-model/flybody-mujoco.xml','models/flybody-mujoco.xml')),capture.scene.sourceXmlSha256);
assert.deepEqual(capture.scene.metadata,JSON.parse(await assetBytes('/body-model/flybody-mujoco.json','models/flybody-mujoco.json')));
assert.deepEqual(capture.scene.io,JSON.parse(await fs.readFile('data/prepared/banc888/io.json','utf8')));
if(fixtureOnly)await fs.writeFile(path.join(output,'fixture-capture.json'),JSON.stringify(capture)+'\n');
const frames=capture.steps.map((step,index)=>{
  assert.equal(step.index,index);assert.equal(step.durationSeconds,.002);const rates=decode(step.ratesHz,32);
  assert.equal(rates.length,capture.dimensions.motorCount);return {...step,decodedRates:rates};
});
const steeringIds=new Set(capture.scene.io.muscles.filter(m=>m.kind==='wing_steering_assumption').flatMap(m=>m.indices));
const steeringSharedWithOtherMappings=capture.scene.io.muscles.filter(m=>m.kind!=='wing_steering_assumption'&&m.indices.some(i=>steeringIds.has(i))).map(m=>({kind:m.kind,target:m.target}));
assert.deepEqual(steeringSharedWithOtherMappings,[],'Steering-rate ablation would also alter a nonsteering mapping');
const report={createdAt:new Date().toISOString(),captureKind:capture.kind,captureFile:fixtureOnly?'fixture-capture.json':args.capture||path.join(output,'capture.json'),
  captureProvenance:structuredClone(capture.provenance),captureScope:input.scope??null,
  captureSha256:hash(inputBytes||JSON.stringify(capture)),scriptSha256:hash(await fs.readFile('scripts/replay-flight-motor-capture.mjs')),
  nativeVersion:mj.mj_versionString(),sourceHashes,configHash:hash(configBytes),modelFingerprint:config.modelFingerprint,
  canonicalConfigHash:hash(canonicalConfigBytes),virtualAssetHashes:Object.fromEntries(Object.entries(virtualAssets).map(([url,value])=>[url,hash(value)])),
  diagnosticVariant:input.diagnostic??null,
  observationSourceHashes:Object.fromEntries(await Promise.all(['web/training/environment.js','web/training/flight-objective.js'].map(async file=>[file,{local:hash(await fs.readFile(file)),captured:config.assets['/'+file.slice(4)]??null}]))),
  frames:frames.length,recordedSeconds:frames.length*.002,holdLastSeconds,baselineOnly,exportMuscleProfiles,
  baselineGate:{required:'Every recorded qpos and qvel value must have identical float64 bytes at every 2ms boundary; no numerical tolerance.',passed:false},
  scope:'Recorded motor vectors held fixed; native mechanics and muscle/internal state evolve continuously. Counterfactual sensory responses and live BANC adaptation are not simulated.',
  units:{length:'cm',time:'s',angularSpeed:'rad/s',clawForce:'g cm/s^2',wingTorque:'g cm^2/s^2'},
  measurement:{footOrder:['T1_left','T2_left','T3_left','T1_right','T2_right','T3_right'],
    clawForce:'Scalar actuator_force alone is not transmitted adhesion. Sparse native actuator_moment times actuator_force supplies each claw contribution to free-root generalized force; its first three components are world force.',
    cadence:'Contacts, inversion and force events are sampled every 2ms. COM refresh follows retained native force/contact reads.',
    wingDrive:'muscleDrive is the pre-intervention muscle-force pool; requestedPower uses the actual drive passed to wings.step after any symmetric-drive intervention.',
    wingJoints:makeFlightTelemetry?['name','actuatorId','targetAngleRad','actualAngleRad','velocityRadPerSecond','control','nativeActuatorForce']:[]},cases:[]};
function replay(intervention){
  const {body,model}=newBody(capture),baseline=intervention.name==='baseline';
  const samples=[],gate={exact:true,checkedSteps:0,firstMismatch:null};
  const muscleProfiles=[];let currentFrameIndex=-1;
  if(baseline&&exportMuscleProfiles){
    const muscleStep=body.muscles.step;
    body.muscles.step=function(input,dt){
      const inputBase64=pack32(input),timeBefore=body.data.time;
      const result=muscleStep.call(this,input,dt);
      muscleProfiles.push({index:muscleProfiles.length,frameIndex:currentFrameIndex,timeBefore,dt,
        input:inputBase64,state:pack32(result)});
      return result;
    };
  }
  const wingStep=body.wings.step;let appliedDrive=null;
  if(intervention.noClaws)for(const a of body.metadata.actuators.filter(a=>a.name.startsWith('adhere_claw_')))model.actuator_gainprm[a.id*10]=0;
  if(intervention.powerMultiplier)body.wings.setInterpreterParameters({...body.wings.interpreter,powerGain:body.wings.interpreter.powerGain*intervention.powerMultiplier});
  body.wings.step=function(q,ctrl,left,right,steer,dt){
    if(intervention.symmetricDrive)left=right=(left+right)/2;
    if(intervention.zeroSteeringForces)steer={left:{},right:{}};
    appliedDrive=[left,right];
    return wingStep.call(this,q,ctrl,left,right,steer,dt);
  };
  const zeroRates=new Map(capture.motorNeuronIndices.map(i=>[i,0])),initial=observe(body,zeroRates,capture.scene.ceiling/10);
  let error=null;
  try{
    const total=frames.length+(baseline?0:Math.round(holdLastSeconds/.002));
    for(let index=0;index<total;index++){
      const row=frames[Math.min(index,frames.length-1)],heldLast=index>=frames.length;
      currentFrameIndex=index;
      const rates=new Map(capture.motorNeuronIndices.map((id,k)=>[id,intervention.zeroSteeringRates&&steeringIds.has(id)?0:row.decodedRates[k]]));
      body.food=structuredClone(row.food);body.step(rates,row.durationSeconds,row.bodyOptions);
      if(baseline){
        const qpos=compareBits(body.data.qpos,row.postQpos),qvel=compareBits(body.data.qvel,row.postQvel);gate.checkedSteps++;
        if(!qpos.exact||!qvel.exact){gate.exact=false;gate.firstMismatch={index,time:body.data.time,qpos,qvel};break;}
      }
      const sample=observe(body,rates,capture.scene.ceiling/10);sample.phase=heldLast?'held-last-motor-vector':'recorded-motor-vector';
      const requestedPower=appliedDrive.map(value=>Math.max(0,Math.min(1,value*body.wings.interpreter.powerGain)));
      sample.wing.sides.rows.forEach((side,k)=>{side[2]=requestedPower[k];});
      sample.interpreterInput={muscleDrive:[body.wingDriveLeft,body.wingDriveRight],appliedDrive,
        requestedPower,powerGain:body.wings.interpreter.powerGain,steeringForcesSuppressed:!!intervention.zeroSteeringForces};samples.push(sample);
      assert.equal(sample.appliedForceMaximum,0,'Replay contains a direct external force');
    }
  }catch(e){error=e.message;gate.exact=false;}
  finally{body.dispose();model.delete();}
  const summary=rows=>{
    const final=rows.at(-1)||initial;
    return {minimumUp:Math.min(initial.up,...rows.map(s=>s.up)),maximumAngularSpeed:Math.max(initial.angularSpeed,...rows.map(s=>s.angularSpeed)),
      maximumCOMRiseCm:Math.max(initial.height,...rows.map(s=>s.height))-initial.height,finalCOMRiseCm:final.height-initial.height,
      firstInversionSeconds:rows.find(s=>s.up<0)?.time??null,firstNoEnvironmentContactSeconds:rows.find(s=>s.environmentContacts===0)?.time??null,
      firstNoFootSupportSeconds:rows.find(s=>s.footSupportCount===0)?.time??null,finalUp:final.up,finalAngularSpeed:final.angularSpeed,
      completedRecordedHorizon:rows.length>=frames.length,sampledSeconds:rows.length*.002,finalFootSupportCount:final.footSupportCount,
      maximumAppliedForce:Math.max(0,...rows.map(s=>s.appliedForceMaximum))};
  };
  const metrics=summary(samples.slice(0,frames.length));metrics.recordedCOMRiseCm=metrics.finalCOMRiseCm;
  return {name:intervention.name,intervention,baselineGate:baseline?gate:null,error,initial,metrics,
    diagnosticTotalMetrics:samples.length>frames.length?summary(samples):null,samples,muscleProfiles:exportMuscleProfiles&&baseline?muscleProfiles:undefined};
}
const baseline=replay({name:'baseline'});report.cases.push(baseline);
report.baselineGate.passed=baseline.baselineGate.exact&&baseline.baselineGate.checkedSteps===frames.length&&!baseline.error;
report.baselineGate.details=baseline.baselineGate;
report.observationGate={passed:false,checkedInitial:false,checkedPreviewFrames:0,error:null};
if(report.baselineGate.passed){
  try{
    if(!synthetic){
      assert.deepEqual(baseline.initial.trainingObservation,capture.initialObservation,'Reconstructed initial training observation differs');
      report.observationGate.checkedInitial=true;
      for(const frame of input.frames||[]){
        if(!frame.metrics?.observation)continue;
        const index=Math.round(frame.time/.002)-1;
        const observed=index<0?baseline.initial:baseline.samples[index];
        if(!observed||Math.abs(observed.time-frame.time)>1e-8)continue;
        assert.deepEqual(observed.trainingObservation,frame.metrics.observation,`Reconstructed training observation differs at ${frame.time}s`);
        report.observationGate.checkedPreviewFrames++;
      }
    }
    report.observationGate.passed=true;
    if(exportMuscleProfiles)await fs.writeFile(path.join(output,'muscle-profiles.json'),JSON.stringify({schemaVersion:1,
      kind:'exact-native-muscle-input-replay',captureFile:report.captureFile,captureSha256:report.captureSha256,
      captureProvenance:report.captureProvenance,sourceHashes,configHash:report.configHash,baselineGate:report.baselineGate,
      observationGate:report.observationGate,dimensions:capture.dimensions,mappings:capture.scene.io.muscles,
      motorNeuronIndices:capture.motorNeuronIndices,initial:capture.initial,
      frames:frames.map(({decodedRates,...row})=>row),profiles:baseline.muscleProfiles,
      encoding:{input:'float32-little-endian-base64; stride5 excitation,length,shortening velocity,Fmax,energy',
        state:'float32-little-endian-base64; stride3 activation,fatigue,force'}})+'\n');
    await fs.writeFile(path.join(output,'training-observations.json'),JSON.stringify({schemaVersion:1,
      captureFile:report.captureFile,captureSha256:report.captureSha256,captureProvenance:report.captureProvenance,
      captureScope:report.captureScope,diagnosticVariant:report.diagnosticVariant,sourceHashes:report.sourceHashes,
      observationSourceHashes:report.observationSourceHashes,baselineGate:report.baselineGate,observationGate:report.observationGate,
      initial:baseline.initial.trainingObservation,steps:baseline.samples.map((s,index)=>({index,time:s.time,dt:.002,observation:s.trainingObservation}))})+'\n');
  }catch(error){report.observationGate.error=error.message;}
}
if(report.baselineGate.passed&&!baselineOnly){
  // Factorial mechanics controls leave every captured neural rate unchanged.
  for(const powerMultiplier of [1,1.5])for(const noClaws of [false,true])for(const zeroSteeringForces of [false,true])for(const symmetricDrive of [false,true]){
    if(powerMultiplier===1&&!noClaws&&!zeroSteeringForces&&!symmetricDrive)continue;
    const name=[noClaws?'no_claw_gain':null,zeroSteeringForces?'zero_steering_force':null,symmetricDrive?'symmetric_wing_drive':null,powerMultiplier===1.5?'power_x1.5':null].filter(Boolean).join('+');
    const result=replay({name,noClaws,zeroSteeringForces,symmetricDrive,powerMultiplier});report.cases.push(result);
    console.log(JSON.stringify({name,error:result.error,...result.metrics}));
  }
  const result=replay({name:'zero_steering_rates',zeroSteeringRates:true});report.cases.push(result);
}
report.interpretationAllowed=report.baselineGate.passed&&report.observationGate.passed&&!synthetic;
report.caveats=['Synthetic fixtures only validate capture/restoration and ablation plumbing.',
  'Zero claw gain removes native grip while keeping claw motor rates, muscle state and all other raw motor inputs.',
  'Zero steering force suppresses only the named force supplied to the wing interpreter; zero steering rates separately tests upstream recruitment.',
  'Symmetric drive replaces left/right asynchronous muscle-force means with their bilateral mean; it is a test intervention, not a proposed hidden controller.',
  'Power multiplication acts on interpreter powerGain and retains native saturation.',
  'Any hold-last extension is a diagnostic continuation after the recorded neural sequence ended, not normal replay or closed-loop neural behavior.'];
await fs.writeFile(path.join(output,'result.json'),JSON.stringify(report)+'\n');
const rows=report.cases.map(c=>`| ${c.name} | ${c.metrics.recordedCOMRiseCm.toFixed(4)} | ${c.metrics.minimumUp.toFixed(4)} | ${c.metrics.maximumAngularSpeed.toFixed(2)} | ${c.metrics.firstInversionSeconds?.toFixed(3)??'none'} | ${c.metrics.finalFootSupportCount} | ${c.error||'none'} |`).join('\n');
await fs.writeFile(path.join(output,'README.md'),`# Recorded motor mechanical replay\n\nCapture: ${report.captureKind}; neural backend ${capture.provenance.backend}. Native MuJoCo ${report.nativeVersion}; ${report.frames} recorded 2ms blocks (${report.recordedSeconds}s). [Full source-pinned results](result.json). ${input.scope||''}\n\nBit-exact qpos/qvel baseline gate: **${report.baselineGate.passed?'PASS':'FAIL'}**. Behavioral interpretation allowed: **${report.interpretationAllowed}**. ${synthetic?'This is a synthetic plumbing fixture, not actual BANC behavior.':''}\n\nNo direct qfrc_applied/xfrc_applied injection or pose resets occur during replay. Every case starts from the same complete state. The factorial mechanical interventions retain recorded motor rates; the separately identified steering-rate ablation zeros only the annotated steering rates. Native contacts, muscle length/velocity effects, internal state and biomechanics evolve freely. Claw scalar actuator output and transmitted world force are reported separately. Events are sampled every 2ms.\n\nAll table metrics use only the recorded horizon.\n\n| Case | Recorded COM rise (cm) | Minimum up | Maximum angular speed (rad/s) | First inversion (s) | Final loaded feet | Error |\n|---|---:|---:|---:|---:|---:|---|\n${rows}\n\nHold-last extension: ${holdLastSeconds}s, applied only to counterfactual diagnostic runs and explicitly labeled in each sample. Its total-horizon summaries are separate diagnosticTotalMetrics fields. Counterfactual motion would alter sensory input in a live brain; that neural response is not simulated.\n`);
console.log(JSON.stringify({output,baselineGate:report.baselineGate,observationGate:report.observationGate,interpretationAllowed:report.interpretationAllowed,cases:report.cases.length}));
if(!report.baselineGate.passed||!report.observationGate.passed||report.cases.some(c=>c.error))process.exitCode=1;
