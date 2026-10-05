import {validateMaintainedScene} from '../flight-scene-profile.js';
import {parameterValues,seededRandom,createEpisodeScore,CRITERIA,STAGES,clip} from './episode.js';
import {measureFlightObservation,measureFlightKinematics} from './flight-observation.js';
import {makeFlightTelemetry} from './flight-telemetry.js';
import {createTrainingBrainResources} from './brain-resources.js';
import {createTrainingSensoryResources} from './sensory-feedback.js';
import {createHaltereCurrentMapper} from '../banc-haltere.js';
import {createWingMotorEventReader} from '/banc-engine/src/motor-events.js';
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const sha=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
const clone=x=>structuredClone(x);
const externalForceApplied=data=>{
 for(const values of [data.xfrc_applied,data.qfrc_applied])for(let i=0;i<values.length;i++)if(values[i]!==0)return true;
 return false;
};
function haltereSample(body,fly,structural=false){
 const sample={halterePower:fly.feedback?.halterePower,omegaRootRadS:Array.from(body.data.qvel.slice(3,6)),
  wingPhaseRadians:body.wings.phase,wingFrequencyHz:body.wings.frequencyHz};
 if(!structural)return sample;
 Object.assign(sample,{bodyTimeSeconds:body.time,wingPower:[body.wingPowerLeft,body.wingPowerRight]});
 sample.wingMotion=Object.fromEntries(['left','right'].map(side=>{
  const name='wing_roll_'+side,joint=body.byJoint.get(name);
  if(!joint)throw new Error('Missing native haltere driver joint '+name);
  return [side,{joint:name,angleRadians:body.data.qpos[joint.qpos],angularVelocityRadS:body.data.qvel[joint.dof],
   source:'native-joint',frame:'native-joint-coordinate'}];
 }));
 return sample;
}
async function fetchText(url){const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw new Error(`Training asset unavailable: ${url}`);return response.text();}
async function fetchJson(url){return JSON.parse(await fetchText(url));}
function assetRecords(assets){return Object.entries(assets||{}).map(([key,value])=>typeof value==='string'?{url:key,sha256:value}:{...value,url:value.url||key});}
async function checkAssets(config,onProgress,verifiedWasm=null){
 const records=assetRecords(config.assets);if(!records.length)throw new Error('Training asset manifest has not been built');
 for(let i=0;i<records.length;i++){
  const entry=records[i];if(!/^[a-f0-9]{64}$/.test(entry.sha256))throw new Error('Invalid training asset digest: '+entry.url);
  const response=await fetch(entry.url,{cache:'no-store'});if(!response.ok)throw new Error('Missing training asset: '+entry.url);const bytes=await response.arrayBuffer();
  if(entry.bytes!==undefined&&entry.bytes!==bytes.byteLength||await sha(bytes)!==entry.sha256)throw new Error('Training asset checksum mismatch: '+entry.url);
  if(verifiedWasm&&entry.url==='/banc-engine/dist/core.wasm'){
   if(entry.sha256!==verifiedWasm.moduleSha256)throw new Error('Conflicting training WASM execution pin');
   verifiedWasm.bytes=new Uint8Array(bytes);
  }
  onProgress?.({message:'Checking training assets',completed:i+1,total:records.length});
 }
 return records;
}

/** One real BANC network and native FlyBody, with an explicitly bounded policy
 * vector. Reward/targets are observations only; no root control is introduced. */
export async function createTrainingEnvironment(requestedConfig,{configUrl='/training/config.json',onProgress}={}){
 const text=await fetchText(configUrl),config=JSON.parse(text),configHash=await sha(new TextEncoder().encode(text));
 if(requestedConfig&&JSON.stringify(requestedConfig)!==JSON.stringify(config))throw new Error('Initialization config differs from canonical training config');
 const initialParameters=parameterValues(config,config.parameters.map(p=>p.initial)),motorDecoderEnabled=initialParameters.motorDecoder!==undefined;
 const sensorimotorEnabled=initialParameters.sensoryConfig!==undefined;
 const sceneProfile=validateMaintainedScene(config.maintainedScene);
 if(sceneProfile&&!sensorimotorEnabled&&(config.stage!=='maintained_flight'||config.stages.some(s=>s.id!=='maintained_flight')))throw new Error('Spacious curriculum is exclusive to maintained flight');
 if(!/^[a-f0-9]{64}$/.test(config.modelFingerprint))throw new Error('Training model fingerprint is not finalized');
 if(config.neuralInputSequence!==undefined&&typeof config.neuralInputSequence!=='boolean')throw new Error('neuralInputSequence must be boolean');
 const useInputSequence=config.neuralInputSequence===true||config.haltereFeedback!==undefined;
 const wasmExecution=config.optimizer?.acceptance?.nativeExecution?.backend==='wasm'?clone(config.optimizer.acceptance.nativeExecution):null;
 const verifiedWasm=wasmExecution?{moduleSha256:wasmExecution.moduleSha256,bytes:null}:null;
 const assets=await checkAssets(config,onProgress,verifiedWasm);
 if(verifiedWasm&&!verifiedWasm.bytes)throw new Error('Training WASM execution requires its verified binary');
 if(motorDecoderEnabled&&!assets.some(asset=>asset.url==='/motor-decoder.js'))throw new Error('Motor decoder requires its pinned executable asset');
 const [{loadBancModel,WebGPUBrain,WasmBrain,createWasmCore,intrinsicLayout},{createBancBodyFactory},{SensoryEncoder},{createBancTasteMapper},{createBancSensoryCurrentMapper}]=await Promise.all([
  import('/banc-engine/src/index.js'),import('../flybody-world.js'),import('../sensory-encoder.js'),import('../banc-taste.js'),import('../banc-sensory-current.js')]);
 const [base,core,bodyFactory,habitat,groups,sensory,supplement]=await Promise.all([loadBancModel(),createWasmCore(verifiedWasm?{wasmBinary:verifiedWasm.bytes}:undefined),createBancBodyFactory(),fetchJson('/habitat.json'),fetchJson('/banc-data/console/groups.json'),fetchJson('/banc-data/console/sensory-inputs.json'),fetchJson('/body-model/banc-taste-peg-annotations.json')]);
 if(base.manifest.dt_ms!==config.dtMs||supplement.prepared_ids_sha256!==base.manifest.files['ids.bin'].sha256||supplement.prepared_io_sha256!==base.manifest.files['io.json'].sha256)throw new Error('Training graph or sensory identity mismatch');
 // The declared decoder contract must describe this actual prepared graph,
 // before allocating a native body or neural state.
 if(motorDecoderEnabled)parameterValues(config,config.parameters.map(p=>p.initial),base.io);
 const brainModel=config.intrinsicModels===undefined?base:{...base,manifest:{...base.manifest,intrinsic_models:clone(config.intrinsicModels)}};
 const eventContract=brainModel.manifest.intrinsic_models===undefined?undefined:intrinsicLayout(brainModel).eventContract;
 let sensoryModel=sensory,haltereMapper=null,preparedIds=null;
 if(config.tegulaFeedback!==undefined||config.haltereFeedback!==undefined||config.antennaFeedback!==undefined||config.legProprioception!==undefined){
  const response=await fetch('/banc-data/ids.bin',{cache:'no-store'});
  if(!response.ok)throw new Error('Prepared sensory identities unavailable');
  const bytes=await response.arrayBuffer(),entry=base.manifest.files['ids.bin'];
  if(bytes.byteLength!==entry.bytes||await sha(bytes)!==entry.sha256)throw new Error('Prepared sensory identity checksum mismatch');
  preparedIds=new BigUint64Array(bytes);
 }
 if(config.tegulaFeedback!==undefined){
  const {createTegulaSensoryManifest}=await import('../banc-tegula.js');
  sensoryModel=createTegulaSensoryManifest({...base,ids:preparedIds},sensory,config.tegulaFeedback);
 }
 if(config.haltereFeedback!==undefined){
  const settings=config.haltereFeedback;
  if(!settings||typeof settings!=='object'||Array.isArray(settings)||Object.keys(settings).some(key=>!['profile','geometries','maxCurrentPa','mechanicalModel'].includes(key)))throw new Error('Invalid haltere feedback declaration');
  haltereMapper=createHaltereCurrentMapper({enabled:true,...settings,sensoryManifest:sensory,preparedIds});
 }
 const tasteMapper=createBancTasteMapper([...base.io.sensory,...supplement.annotations],groups.sweet);
 if(tasteMapper.coverage.unmapped.some(row=>row.reason==='missing sensory annotation'))throw new Error('Training taste annotations incomplete');
 let projection=null;
 if(config.visionFeedback!==undefined){
  const url='/banc-data/console/visual-projections.json';
  if(!assets.some(asset=>asset.url===url))throw new Error('Unpinned retinal projection');
  projection=await fetchJson(url);
 }
 let legCatalog=null;
 if(config.legProprioception!==undefined){
  const url='/body-model/banc-leg-proprioception-v1.json';
  if(!assets.some(asset=>asset.url===url))throw new Error('Unpinned leg proprioceptor catalog');
  legCatalog=await fetchJson(url);
 }
 const sensoryResources=await createTrainingSensoryResources({config,base,sensory:sensoryModel,groups,tasteMapper,projection,ids:preparedIds,sceneProfile,legCatalog});
 const externalIndices=sensoryResources.indices,visionEnabled=sensoryResources.visionEnabled;
 const mapper=await createBancSensoryCurrentMapper(core,base,{indices:externalIndices,onProfile:(_,i,n)=>onProgress?.({message:'Calibrating isolated sensory interface',completed:i+1,total:n})});
 // New sequential candidates rebuild their own observers, never the prepared
 // graph or native body factory. Their index layout must remain identical to
 // the already calibrated sensory-current mapping.
 async function candidateSensors(effectiveConfig){
  if(!sensorimotorEnabled)return {sensoryResources,haltereMapper,effectiveConfig:config};
  if(!effectiveConfig)throw new Error('Missing candidate sensory configuration');
  let candidateManifest=sensory,candidateHaltere=null;
  if(effectiveConfig.tegulaFeedback!==undefined){
   const {createTegulaSensoryManifest}=await import('../banc-tegula.js');
   candidateManifest=createTegulaSensoryManifest({...base,ids:preparedIds},sensory,effectiveConfig.tegulaFeedback);
  }
  if(effectiveConfig.haltereFeedback!==undefined){
   const settings=effectiveConfig.haltereFeedback;
   if(!settings||typeof settings!=='object'||Array.isArray(settings)||Object.keys(settings).some(key=>!['profile','geometries','maxCurrentPa','mechanicalModel'].includes(key)))throw new Error('Invalid candidate haltere declaration');
   candidateHaltere=createHaltereCurrentMapper({enabled:true,...settings,sensoryManifest:sensory,preparedIds});
  }
  const resources=await createTrainingSensoryResources({config:effectiveConfig,base,sensory:candidateManifest,groups,tasteMapper,projection,ids:preparedIds,sceneProfile,legCatalog});
  if(resources.visionEnabled!==visionEnabled||resources.indices.length!==externalIndices.length||!resources.indices.every((id,k)=>id===externalIndices[k]))
   throw new Error('Candidate sensory tuning changed the fixed input identity layout');
  return {sensoryResources:resources,haltereMapper:candidateHaltere,effectiveConfig};
 }
 const motorIndices=Uint32Array.from(base.io.motor_neurons,m=>m.index);
 let backend=null,active=null,disposed=false,lastSensorySummary=null;
 const brainResources=createTrainingBrainResources({model:brainModel,backend:wasmExecution?'wasm':'auto',
  createGPU:(model,options)=>WebGPUBrain.create(model,options),createWasm:(model,options)=>new WasmBrain(core,model,options),
  onFallback:error=>onProgress?.({message:'WebGPU unavailable; using real WASM BANC',detail:error.message})});
 const makeBrain=async()=>{const brain=await brainResources.create();backend=brainResources.backend;
  if(wasmExecution&&(backend!=='wasm'||brain.backend!=='wasm')){brain.dispose();throw new Error('Training WASM execution backend mismatch');}
  return brain;};
 const executionEvidence=()=>wasmExecution?{neuralEngine:'wasm',wasmExecution:clone(wasmExecution)}:{};
 const cleanup=()=>{if(active){lastSensorySummary=active.sensoryFeedback?.summary()??null;try{active.sensoryFeedback?.dispose();active.brain?.dispose();}finally{active.world?.dispose();active=null;}}};
 const shaders=assets.filter(a=>/\/neural(?:-dlm)?\.wgsl$/.test(a.url));
 const checkShader=async()=>{for(const shader of shaders)if(await sha(new TextEncoder().encode(await fetchText(shader.url)))!==shader.sha256)throw new Error('Neural shader changed after model fingerprint');};
 function frame(stage,metrics={}){
  const {world,body:b}=active,d=b.data,point=(buffer,id)=>Array.from(buffer.slice(id*3,id*3+3));
  const wings=world.wingLandmarks.map(l=>({side:l.side,position:point(d.geom_xpos,l.geom),rotation:Array.from(d.geom_xmat.slice(l.geom*9,l.geom*9+9)),size:[...l.size],anchor:point(d.xpos,l.body)}));
  const mouth={anchors:world.mouthLandmarks.joints.map(id=>point(d.xanchor,id)),ellipsoids:world.mouthLandmarks.geoms.map(l=>({position:point(d.geom_xpos,l.id),rotation:Array.from(d.geom_xmat.slice(l.id*9,l.id*9+9)),size:[...l.size]}))};
  return {time:b.time,simSeconds:b.time,stage,position:[b.x,b.y,b.z],quaternion:[...b.quaternion],feet:b.feet.map(p=>[...p]),legs:world.metadata.leg_bodies.map((ids,i)=>ids.map((id,k)=>k===2?[...b.feet[i]]:point(d.xpos,id))),wings,mouth,
   food:world.habitat.fruit.map(f=>({kind:f.kind,position:[f.x/10,f.z/10,f.y/10],radiusCm:f.radius/10,lengthCm:(f.length||0)/10,rotation:f.angle||0,path:f.path?.map(([x,y])=>[x/10,y/10,f.y/10]),remaining:f.remaining})),
   bowl:sceneProfile?{radiusCm:sceneProfile.radiusCm,floor:{baseCm:.15,radialCoefficientPerCm:.037,capRadiusCm:sceneProfile.floorCapRadiusCm},ceilingCm:sceneProfile.ceilingCm,curriculumScene:sceneProfile}:{radiusCm:6.5,floor:{baseCm:.15,radialCoefficientPerCm:.037},ceilingCm:world.habitat.ceiling/10},
   contacts:{environment:b.environmentContactCount,food:b.foodContactCount,legs:Array.from(b.legFoodContact),mouth:Array.from(b.mouthFoodContact),wings:Array.from(b.wingFoodContact)},
   ...(active.haltereMapper?{haltereCurrent:clone(active.fly.sensory?.haltereCurrent??null)}:{}),
   phase:metrics.phase||'grounded',motion:active.fly.motion,metrics:{...metrics,observation:active.lastObservation,mechanics:makeFlightTelemetry(b,world.rates(active.fly))},neuralMs:active.brain?.timeMs||0,neuralSpikes:active.lastSpikes||0,vision:visionEnabled};
 }
 function createWorld(stage,seed,interpreter,motorDecoder,observers={sensoryResources,haltereMapper,effectiveConfig:config}){
  observers.haltereMapper?.reset?.();
  const random=seededRandom(seed),fly={...clone(habitat.flies[0]),id:1,brain:{time_ms:0,motor:{},motorNeuronRates:Array(motorIndices.length).fill(0)}};
  // All objectives begin in native stance. Landing must follow the fly's own
  // takeoff and flight; an airborne reset would also reward merely falling.
  fly.x+=(random()-.5)*.6;fly.z+=(random()-.5)*.6;fly.heading+=(random()-.5)*.3;
  const world=bodyFactory(habitat.fruit,[fly],{movementMode:'direct',motorCoupling:true,flightEnabled:true,...(sceneProfile?{maintainedScene:sceneProfile}:{})}),body=world.bodies.get(1);
  try{
  body.wings.setInterpreterParameters(interpreter);
  if(config.wingEventExcitation!==undefined)body.enableWingMotorEvents(config.wingEventExcitation,eventContract);
  if(motorDecoder!==undefined)body.enableMotorDecoder(motorDecoder);
  body.data.qvel.fill(0);
  body.data.qvel[3]=(random()-.5)*.8;body.data.qvel[4]=(random()-.5)*.8;
  // Initial conditions only. No reset or direct body force occurs thereafter.
  world.mj.mj_forward(world.model,body.data);
  if(config.tegulaFeedback!==undefined)body.enableWingLoadFeedback({localFrame:config.tegulaFeedback.schema===2});
  body.refresh();body.monitor.resetContinuity('training episode initial condition');world.copyPose(fly,body,0);
  return {world,body,fly,...observers,initialCondition:{stage,seed,position:[body.x,body.y,body.z],quaternion:[...body.quaternion],velocity:Array.from(body.data.qvel.slice(0,6)),airborne:body.airborne,placement:'Native grounded stance with a seeded small angular disturbance; no airborne offset or launch impulse.'}};
  }catch(error){world.dispose();throw error;}
 }
 function observe(){
  const {body:b,world}=active,up=1-2*(b.quaternion[1]**2+b.quaternion[2]**2),angularSpeed=Math.hypot(...b.data.qvel.slice(3,6));
  const support=measureFlightObservation(b),kinematics=measureFlightKinematics(b);
  const finite=[b.time,...kinematics.position,kinematics.verticalSpeed,kinematics.speedCmPerSecond,up,angularSpeed,...b.quaternion].every(Number.isFinite);
  const observation={finite,up,angularSpeed,...kinematics,radius:Math.hypot(...kinematics.position.slice(0,2)),ceiling:world.habitat.ceiling/10,
   wingPower:b.wingPower,...support,
   externalForce:externalForceApplied(b.data),...(sceneProfile?{curriculumScene:sceneProfile}:{})};
  active.lastObservation=observation;return observation;
 }
 const ready=async()=>{
  if(disposed)throw new Error('Training environment disposed');
  cleanup();const {interpreter,motorDecoder,sensoryConfig}=parameterValues(config,config.parameters.map(p=>p.initial));
  try{const observers=await candidateSensors(sensoryConfig);active={...createWorld(config.stage,config.optimizer.seed,interpreter,motorDecoder,observers)};active.brain=await makeBrain();await checkShader();
   return {environmentVersion:config.environmentVersion,modelFingerprint:config.modelFingerprint,configHash,backend,...executionEvidence(),bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,vision:visionEnabled,parameterCount:config.parameters.length,criteria:CRITERIA,trainedCellCounts:{},parameterScope:sensorimotorEnabled?'sensory-and-individual-mn-decoder':motorDecoderEnabled?'individual-mn-motor-decoder-only':'flight-interpreter-only',frame:frame(config.stage),limitations:[visionEnabled?'Experimental retinal motion input at mapped T4/T5 cells; receptive fields and gains are modeling priors.':'Vision is disabled; the observer preview is not retinal input.','Reward thresholds and fitted parameters are modeling assumptions, not measured biological physiology.',sensorimotorEnabled?'Declared sensory scales and individual-MN decoder coefficients are tuned against simulation tasks. Anatomy, neural graph, native physics and receptive directions stay fixed; this is engineering calibration, not measured physiology.':motorDecoderEnabled?'Only the declared individual-MN motor-decoder coefficients are trainable. Neural physiology, synapses, sensory gains, muscle dynamics, native body physics and wingbeat tables stay fixed.':'Only 27 wing-interpreter coefficients are trainable. Neural physiology, synapses, sensory gains, muscle dynamics, native body physics and wingbeat tables stay fixed.']};
  }finally{cleanup();}
 };
 async function evaluate(job,{checkpoint=async()=>delay(0),previewHz=6,dutyCycle=.65,getBudget,onFrame,onProgress,onInitialState,onBeforePhysicsStep,onPhysicsStep,onMotorEvents}={}){
  if(disposed)throw new Error('Training environment disposed');if(active)throw new Error('Concurrent training evaluation');
  if(sensorimotorEnabled)lastSensorySummary=null;
  if(onMotorEvents!==undefined&&typeof onMotorEvents!=='function')throw new Error('onMotorEvents must be a function');
  if(onBeforePhysicsStep!==undefined&&typeof onBeforePhysicsStep!=='function')throw new Error('onBeforePhysicsStep must be a function');
  const {vector,interpreter,motorDecoder,sensoryConfig,sensoryCount}=parameterValues(config,job.parameters);
  if(!Number.isInteger(job.seed)||job.seed<0||job.seed>0xffffffff||!STAGES.includes(job.stage))throw new Error('Invalid episode seed or stage');
  const stageConfig=config.stages.find(s=>s.id===job.stage);
  if(!stageConfig)throw new Error('Episode stage is not configured');
  const duration=job.durationSeconds??stageConfig.durationSeconds;
  if(!Number.isFinite(duration)||duration!==stageConfig.durationSeconds||duration!==CRITERIA.minimumDurationSeconds[job.stage])throw new Error('Invalid episode duration; full objective horizon is required');
  const started=performance.now();let score,steps=0,termination=null,cancelled=false,error=null,lastFrameWall=-Infinity,lastProgressWall=-Infinity,lastFiniteFrame=null;
  // Active setup and feedback-block work exclude checkpoint pauses and budget
  // rests. wallSeconds below remains total elapsed evaluation time.
  let setupStarted=null,blockStarted=null,setupWallSeconds=0,executionWallSeconds=0;
  const provenance={environmentVersion:config.environmentVersion,backend:null,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,modelFingerprint:config.modelFingerprint,configHash,seed:job.seed,stage:job.stage,vision:visionEnabled};
  if(sensorimotorEnabled)Object.assign(provenance,{parameterContract:config.parameterContract,sensoryParameterCount:sensoryCount,motorParameterCount:motorDecoder.length,calibrationInterpretation:'simulation-based engineering calibration'});
  for(const name of ['generation','pair','pairId','sign','parametersHash'])if(job[name]!==undefined)provenance[name]=job[name];
  let initialCondition,initialObservation,finalObservation;
  try{
   await checkpoint();setupStarted=performance.now();const observers=await candidateSensors(sensoryConfig);active={...createWorld(job.stage,job.seed,interpreter,motorDecoder,observers)};initialCondition=active.initialCondition;active.brain=await makeBrain();provenance.backend=backend;Object.assign(provenance,executionEvidence());await checkShader();
   const {brain,world,body,fly,haltereMapper,effectiveConfig,sensoryResources:episodeSensoryResources}=active,sensoryFeedback=episodeSensoryResources.create({world,body,fly}),encoder=sensoryFeedback.encoder,input=new Float32Array(base.manifest.neuron_count);
   active.sensoryFeedback=sensoryFeedback;
   if(useInputSequence&&typeof brain.stepSequence!=='function')throw new Error('Per-tick current delivery is unavailable');
   const inputSequence=useInputSequence?Array.from({length:config.bodyBlockMs/config.dtMs},()=>new Float32Array(input.length)):null;
   let emitMotorEvents;
   if(onMotorEvents||config.wingEventExcitation!==undefined){
    const wingIndices=Uint32Array.from([...new Set(base.io.muscles.filter(mapping=>mapping.kind==='asynchronous_wing'||mapping.kind==='wing_steering_assumption').flatMap(mapping=>mapping.indices))].sort((a,b)=>a-b));
    if(wingIndices.length!==48)throw new Error('BANC wing event observation requires the48 mapped wing motor neurons');
    if(brain.timeMs!==0)throw new Error('Wing event observation requires a fresh neural baseline');
    const reader=createWingMotorEventReader({indices:wingIndices,params:base.params,dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,eventContract});
    emitMotorEvents=async()=>{
     const state9=await brain.readState(wingIndices,{includeSpikeTime:true,includeStatistics:false,includeSpikeHistory:false}),packet=reader.read(state9,brain.timeMs);
     // The opt-in body consumes owned event state before an optional observer
     // receives its packet. Observation is never required to drive the body.
     if(config.wingEventExcitation!==undefined)body.acceptWingMotorEvents(packet);
     if(!onMotorEvents)return;
     const bodyTimeSeconds=body.time,wingPhaseRadians=body.wings.phase,wingFrequencyHz=body.wings.frequencyHz;
     if(![bodyTimeSeconds,wingPhaseRadians,wingFrequencyHz].every(Number.isFinite))throw new Error('Nonfinite wing event observation context');
     onMotorEvents({...packet,bodyTimeSeconds,wingPhaseRadians,wingFrequencyHz});
    };
    await emitMotorEvents();
   }
   initialObservation=observe();score=createEpisodeScore(job.stage,duration,initialObservation);
   // Development observers copy state synchronously. No snapshots, rate-map
   // conversions or callback objects are created when these hooks are absent.
   onInitialState?.({body,world,fly,brain,input,inputSequence,encoder,sensoryFeedback,motorIndices,job,provenance,initialCondition,initialObservation});
   lastFiniteFrame=frame(job.stage,score.state);onFrame?.(lastFiniteFrame);lastFrameWall=performance.now();
   setupWallSeconds=(performance.now()-setupStarted)/1000;setupStarted=null;
   while(steps*config.bodyBlockMs/1000+1e-10<duration){
    await checkpoint();blockStarted=performance.now();
    const encoded=sensoryFeedback.update();
    if(encoded)for(let k=0;k<encoded.indices.length;k++){const index=encoded.indices[k];input[index]=mapper.current(index,encoded.ratesHz[k]);}
    if(inputSequence){
     const sample=haltereMapper?haltereSample(body,fly,effectiveConfig.haltereFeedback.mechanicalModel!==undefined):null;
     for(let k=0;k<inputSequence.length;k++){
      inputSequence[k].set(input);
      if(haltereMapper){
       const elapsedSeconds=k*config.dtMs/1000;
       const diagnostics=haltereMapper.writeInto(inputSequence[k],{...sample,elapsedSeconds});
       encoder.sample.haltereCurrent={...diagnostics,sourceBodyTimeSeconds:body.time,sourcePhaseRadians:sample.wingPhaseRadians,
        mostRecentInputOffsetSeconds:elapsedSeconds,samplingIntervalMs:config.dtMs,unit:'pA',
        replacesLegacyHaltereRateCurrent:true};
      }
     }
     await brain.stepSequence(inputSequence,body.internal,true);
    }else await brain.step(config.bodyBlockMs/config.dtMs,input,body.internal,true);
    const state=await brain.readState(motorIndices);
    if(!state.every(Number.isFinite))throw new Error('Nonfinite BANC motor state');
    fly.brain={time_ms:brain.timeMs,motor:{},motorNeuronRates:Array.from(motorIndices,(_,k)=>state[k*8+4])};fly.senses=encoder.sample.food;fly.sensory=encoder.sample;active.lastSpikes=state.totalSpikes;
    if(emitMotorEvents)await emitMotorEvents();
    const physicsTimeBefore=onPhysicsStep||onBeforePhysicsStep?body.time:0;
    onBeforePhysicsStep?.({body,world,fly,brain,input,inputSequence,encoder,sensoryFeedback,index:steps,episodeIndex:steps,phase:'scored',episodeTimeSeconds:steps*config.bodyBlockMs/1000,releaseNativeTime:0,durationSeconds:config.bodyBlockMs/1000,timeBefore:physicsTimeBefore,neuralMs:brain.timeMs});
    world.tick(config.bodyBlockMs/1000);steps++;
    onPhysicsStep?.({body,world,fly,brain,input,inputSequence,encoder,sensoryFeedback,index:steps-1,durationSeconds:config.bodyBlockMs/1000,timeBefore:physicsTimeBefore,neuralMs:brain.timeMs});
    if(Math.abs(body.time-brain.timeMs/1000)>1e-8)throw new Error('Neural/native feedback clocks diverged');
    finalObservation=observe();termination=score.step(finalObservation,config.bodyBlockMs/1000);
    const wall=performance.now(),budget=getBudget?.()||{previewHz,dutyCycle},hz=clip(Number(budget.previewHz)||0,0,10),duty=clip(Number(budget.dutyCycle)||.65,.05,1);
    if(hz&&wall-lastFrameWall>=1000/hz){lastFiniteFrame=frame(job.stage,score.state);onFrame?.(lastFiniteFrame);lastFrameWall=wall;}
    if(wall-lastProgressWall>=500){onProgress?.({stage:job.stage,simSeconds:body.time,steps,return:score.state.return,success:score.state.success,backend,bodyBackend:'mujoco-wasm'});lastProgressWall=wall;}
    const blockWallMs=performance.now()-blockStarted;executionWallSeconds+=blockWallMs/1000;blockStarted=null;
    if(termination.terminated)break;
    const idle=blockWallMs*(1/duty-1);for(let remaining=idle;remaining>0;remaining-=20){await delay(Math.min(20,remaining));await checkpoint();}
   }
   await checkShader();lastFiniteFrame=frame(job.stage,score.state);onFrame?.(lastFiniteFrame);
  }catch(e){if(e.name==='AbortError')cancelled=true;else error=e.message;}
  finally{
   if(setupStarted!==null)setupWallSeconds+=(performance.now()-setupStarted)/1000;
   if(blockStarted!==null)executionWallSeconds+=(performance.now()-blockStarted)/1000;
   cleanup();
  }
  const simSeconds=steps*config.bodyBlockMs/1000,success=!error&&!cancelled&&score?.state.success===true;
  const reason=cancelled?'cancelled':error?'simulation_error':termination?.reason||'time_limit';
  return {return:error?-10:score?.state.return??0,success,terminated:!!error||!!termination?.terminated,truncated:cancelled||(!error&&!termination?.terminated),reason,cancelled,simSeconds,steps,parameters:vector,seed:job.seed,stage:job.stage,
   environmentVersion:config.environmentVersion,backend:provenance.backend||backend,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,modelFingerprint:config.modelFingerprint,configHash,provenance,
   metrics:{...(score?.state||{}),error,wallSeconds:(performance.now()-started)/1000,setupWallSeconds,executionWallSeconds,initialCondition,initialObservation,finalObservation,actualNeuralBackend:provenance.backend||backend,sensoryFeedback:lastSensorySummary,vision:visionEnabled,frameCountIsNotPhysicsSteps:true,criteria:CRITERIA,finalPosition:lastFiniteFrame?.position,finalNeuralSpikes:lastFiniteFrame?.neuralSpikes||0}};
 }
 // Staged-only maintained-flight path. Grounded evaluate() below is unchanged.
 let maintainedDependencies;
 const maintainedUrls=['/training/maintained-flight-objective.js','/training/airborne-reset-contract.js','/training/airborne-reset.js'];
 async function evaluateMaintained(job,{checkpoint=async()=>delay(0),previewHz=6,dutyCycle=.65,getBudget,onFrame,onProgress,onInitialState,onBeforePhysicsStep,onPhysicsStep,onMotorEvents}={}){
  if(disposed)throw new Error('Training environment disposed');if(active)throw new Error('Concurrent training evaluation');
  if(sensorimotorEnabled)lastSensorySummary=null;
  if(onMotorEvents!==undefined&&typeof onMotorEvents!=='function')throw new Error('onMotorEvents must be a function');
  if(onBeforePhysicsStep!==undefined&&typeof onBeforePhysicsStep!=='function')throw new Error('onBeforePhysicsStep must be a function');
  if(job.captureMotorReplay===true)throw new Error('Time-zero motor replay capture is incompatible with maintained-flight release observers');
  const {vector,interpreter,motorDecoder,sensoryConfig,sensoryCount}=parameterValues(config,job.parameters),isRecovery=job.stage==='recovery';
  if(!Number.isInteger(job.seed)||job.seed<0||job.seed>0xffffffff||!['maintained_flight','recovery'].includes(job.stage))throw new Error('Invalid airborne seed or stage');
  const stageConfig=config.stages.find(s=>s.id===job.stage),duration=job.durationSeconds??stageConfig?.durationSeconds;
  if(!stageConfig||duration!==5||stageConfig.durationSeconds!==5)throw new Error('Maintained flight requires five scored seconds');
  if(config.wingEventExcitation===undefined)throw new Error('Maintained flight requires the explicit wing event interface');
  for(const url of maintainedUrls)if(!assets.some(asset=>asset.url===url))throw new Error('Unpinned maintained-flight dependency: '+url);
  const metadata=assets.find(asset=>asset.url==='/body-model/flybody-mujoco.json');
  if(!metadata)throw new Error('Maintained flight requires pinned native body metadata');
  maintainedDependencies??=Promise.all([import('./maintained-flight-objective.js'),import('./airborne-reset-contract.js'),import('./airborne-reset.js')]);
  const [{createMaintainedFlightScore,MAINTAINED_FLIGHT_CRITERIA,maintainedFlightCriteria},{selectFlightInitialCondition,createMaintainedFlightClock},{beginAirborneWarmup}]=await maintainedDependencies;
  let recoveryModule=null,disturbance=null;
  if(isRecovery){
   if(!assets.some(asset=>asset.url==='/training/recovery-objective.js'))throw new Error('Unpinned recovery objective');
   recoveryModule=await import('./recovery-objective.js');
   disturbance=recoveryModule.createRecoveryDisturbance(config.recoveryDisturbance,job.seed);
  }
  const context={bodyMetadataSha256:metadata.sha256},settings=selectFlightInitialCondition('maintained_flight',config.initialCondition,context);
  if(settings.warmupSeconds!==.5)throw new Error('This maintained-flight variant requires an explicit 0.5 second warmup');
  const clock=createMaintainedFlightClock(settings,context),blockSeconds=config.bodyBlockMs/1000;
  const started=performance.now();let score,steps=0,warmupSteps=0,totalBlocks=0,termination=null,cancelled=false,error=null;
  let setupStarted=null,blockStarted=null,setupWallSeconds=0,executionWallSeconds=0,warmupExecutionWallSeconds=0,scoredExecutionWallSeconds=0;
  let lastFrameWall=-Infinity,lastProgressWall=-Infinity,lastFiniteFrame=null,warmup=null,warmupAudit=null;
  let initialCondition,initialObservation,finalObservation,releaseNativeTime=null,releaseNeuralTimeMs=null;
  let lastPacketTimeMs=null,lastAbsoluteNativeTime=0,lastAbsoluteNeuralTimeMs=0,blockPhase=null;
  const provenance={environmentVersion:config.environmentVersion,backend:null,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,
   modelFingerprint:config.modelFingerprint,configHash,seed:job.seed,stage:job.stage,vision:visionEnabled,
   ...(sensorimotorEnabled?{parameterContract:config.parameterContract,sensoryParameterCount:sensoryCount,motorParameterCount:motorDecoder.length,calibrationInterpretation:'simulation-based engineering calibration'}:{}),
   ...(disturbance?{recoveryDisturbance:clone(disturbance)}:{}),
   initialConditionProfile:clone(settings),...(sceneProfile?{curriculumScene:clone(sceneProfile)}:{}),clockSemantics:'Brain, native and event clocks stay absolute; simSeconds and steps count only the scored interval.'};
  for(const name of ['generation','pair','pairId','sign','parametersHash'])if(job[name]!==undefined)provenance[name]=job[name];
  function maintainedFrame(phase){
   const metrics=score?.state||{phase:'warmup',elapsed:0,return:0,success:false};
   return {...frame(job.stage,metrics),simSeconds:steps*blockSeconds,episodeTimeSeconds:steps*blockSeconds,
    nativeTimeSeconds:active.body.time,releaseNativeTime,releaseNeuralTimeMs,warmup:phase==='warmup',
    warmupSeconds:warmupSteps*blockSeconds,totalBlocks,scoredSteps:steps};
  }
  function snapshot({release=false}={}){
   const {body,brain}=active,event=body._wingMotorEvents,audit=warmup.audit();
   if(!event?.initialized)throw new Error('Maintained-flight event state is not initialized');
   // The native adapter sets pending=null iff integratedThroughMs equals
   // observedThroughMs. Check its actual state once at release; after every
   // scored block use that invariant and the body's integer event clocks.
   const pendingEvents=release?event.adapter.snapshot().pending!==null:event.elapsedMs!==event.observedMs;
   return {nativeTimeSeconds:body.time,neuralTimeMs:brain.timeMs,bodyEventElapsedMs:event.elapsedMs,
    bodyEventObservedMs:event.observedMs,lastPacketTimeMs,remainderSeconds:body.remainder,pendingEvents,
    rootRestraintActive:audit.rootRestraintActive,externalForceApplied:externalForceApplied(body.data),rootWriteCount:audit.rootWriteCount};
  }
  function accountBlock(){
   const blockWallMs=performance.now()-blockStarted;executionWallSeconds+=blockWallMs/1000;
   if(blockPhase==='warmup')warmupExecutionWallSeconds+=blockWallMs/1000;else scoredExecutionWallSeconds+=blockWallMs/1000;
   blockStarted=null;return blockWallMs;
  }
  async function budgetAfterBlock(phase){
   const wall=performance.now(),budget=getBudget?.()||{previewHz,dutyCycle},hz=clip(Number(budget.previewHz)||0,0,10),duty=clip(Number(budget.dutyCycle)||.65,.05,1);
   if(hz&&wall-lastFrameWall>=1000/hz){lastFiniteFrame=maintainedFrame(phase);onFrame?.(lastFiniteFrame);lastFrameWall=wall;}
   if(wall-lastProgressWall>=500){onProgress?.({stage:job.stage,phase,simSeconds:steps*blockSeconds,steps,warmupSteps,
    warmupSeconds:warmupSteps*blockSeconds,nativeTimeSeconds:active.body.time,neuralMs:active.brain.timeMs,releaseNativeTime,
    return:score?.state.return??0,success:score?.state.success??false,backend,bodyBackend:'mujoco-wasm'});lastProgressWall=wall;}
   const blockWallMs=accountBlock();
   if(termination?.terminated)return;
   const idle=blockWallMs*(1/duty-1);for(let remaining=idle;remaining>0;remaining-=20){await delay(Math.min(20,remaining));await checkpoint();}
  }
  try{
   await checkpoint();setupStarted=performance.now();const observers=await candidateSensors(sensoryConfig);active={...createWorld(job.stage,job.seed,interpreter,motorDecoder,observers)};
   const preWarmupCondition=clone(active.initialCondition);active.brain=await makeBrain();provenance.backend=backend;Object.assign(provenance,executionEvidence());await checkShader();
   const {brain,world,body,fly,haltereMapper,effectiveConfig,sensoryResources:episodeSensoryResources}=active,sensoryFeedback=episodeSensoryResources.create({world,body,fly}),encoder=sensoryFeedback.encoder,input=new Float32Array(base.manifest.neuron_count);
   active.sensoryFeedback=sensoryFeedback;
   if(useInputSequence&&typeof brain.stepSequence!=='function')throw new Error('Per-tick current delivery is unavailable');
   const inputSequence=useInputSequence?Array.from({length:config.bodyBlockMs/config.dtMs},()=>new Float32Array(input.length)):null;
   let emitMotorEvents;
   if(onMotorEvents||config.wingEventExcitation!==undefined){
    const wingIndices=Uint32Array.from([...new Set(base.io.muscles.filter(mapping=>mapping.kind==='asynchronous_wing'||mapping.kind==='wing_steering_assumption').flatMap(mapping=>mapping.indices))].sort((a,b)=>a-b));
    if(wingIndices.length!==48)throw new Error('BANC wing event observation requires the48 mapped wing motor neurons');
    if(brain.timeMs!==0)throw new Error('Wing event observation requires a fresh neural baseline');
    const reader=createWingMotorEventReader({indices:wingIndices,params:base.params,dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,eventContract});
    emitMotorEvents=async()=>{
     const state9=await brain.readState(wingIndices,{includeSpikeTime:true,includeStatistics:false,includeSpikeHistory:false}),packet=reader.read(state9,brain.timeMs);
     // The opt-in body consumes owned event state before an optional observer
     // receives its packet. Observation is never required to drive the body.
     if(config.wingEventExcitation!==undefined)body.acceptWingMotorEvents(packet);
     lastPacketTimeMs=packet.timeMs;
     if(!onMotorEvents)return;
     const bodyTimeSeconds=body.time,wingPhaseRadians=body.wings.phase,wingFrequencyHz=body.wings.frequencyHz;
     if(![bodyTimeSeconds,wingPhaseRadians,wingFrequencyHz].every(Number.isFinite))throw new Error('Nonfinite wing event observation context');
     onMotorEvents({...packet,bodyTimeSeconds,wingPhaseRadians,wingFrequencyHz});
    };
    await emitMotorEvents();
   }
   // Packet zero establishes event history before the one explicit airborne
   // placement. The same reader, native kernels and BANC instance survive it.
   warmup=beginAirborneWarmup({body,world,fly,settings});
   active.lastObservation=observe();if(!active.lastObservation.finite)throw new Error('Nonfinite airborne warmup state');
   lastFiniteFrame=maintainedFrame('warmup');onFrame?.(lastFiniteFrame);lastFrameWall=performance.now();
   setupWallSeconds=(performance.now()-setupStarted)/1000;setupStarted=null;
   async function advanceFeedbackBlock(phase){
    const encoded=sensoryFeedback.update();
    if(encoded)for(let k=0;k<encoded.indices.length;k++){const index=encoded.indices[k];input[index]=mapper.current(index,encoded.ratesHz[k]);}
    if(inputSequence){
     const sample=haltereMapper?haltereSample(body,fly,effectiveConfig.haltereFeedback.mechanicalModel!==undefined):null;
     for(let k=0;k<inputSequence.length;k++){
      inputSequence[k].set(input);
      if(haltereMapper){
       const elapsedSeconds=k*config.dtMs/1000;
       const diagnostics=haltereMapper.writeInto(inputSequence[k],{...sample,elapsedSeconds});
       encoder.sample.haltereCurrent={...diagnostics,sourceBodyTimeSeconds:body.time,sourcePhaseRadians:sample.wingPhaseRadians,
        mostRecentInputOffsetSeconds:elapsedSeconds,samplingIntervalMs:config.dtMs,unit:'pA',
        replacesLegacyHaltereRateCurrent:true};
      }
     }
     await brain.stepSequence(inputSequence,body.internal,true);
    }else await brain.step(config.bodyBlockMs/config.dtMs,input,body.internal,true);
    const state=await brain.readState(motorIndices);
    if(!state.every(Number.isFinite))throw new Error('Nonfinite BANC motor state');
    fly.brain={time_ms:brain.timeMs,motor:{},motorNeuronRates:Array.from(motorIndices,(_,k)=>state[k*8+4])};fly.senses=encoder.sample.food;fly.sensory=encoder.sample;active.lastSpikes=state.totalSpikes;
    if(emitMotorEvents)await emitMotorEvents();
    const physicsTimeBefore=onPhysicsStep||onBeforePhysicsStep?body.time:0;
    onBeforePhysicsStep?.({body,world,fly,brain,input,inputSequence,encoder,sensoryFeedback,index:totalBlocks,episodeIndex:phase==='scored'?steps:null,phase,episodeTimeSeconds:steps*blockSeconds,releaseNativeTime,durationSeconds:config.bodyBlockMs/1000,timeBefore:physicsTimeBefore,neuralMs:brain.timeMs});
    world.tick(config.bodyBlockMs/1000);totalBlocks++;if(phase==='scored')steps++;else warmupSteps++;
    onPhysicsStep?.({body,world,fly,brain,input,inputSequence,encoder,sensoryFeedback,index:totalBlocks-1,episodeIndex:phase==='scored'?steps-1:null,phase,episodeTimeSeconds:steps*blockSeconds,releaseNativeTime,durationSeconds:config.bodyBlockMs/1000,timeBefore:physicsTimeBefore,neuralMs:brain.timeMs});
    if(Math.abs(body.time-brain.timeMs/1000)>1e-8)throw new Error('Neural/native feedback clocks diverged');
    lastAbsoluteNativeTime=body.time;lastAbsoluteNeuralTimeMs=brain.timeMs;
    const observation=observe();if(!observation.finite)throw new Error('Nonfinite maintained-flight body state');
    return observation;
   }
   while(warmupSteps<clock.warmupBlocks){
    await checkpoint();blockStarted=performance.now();blockPhase='warmup';
    await advanceFeedbackBlock('warmup');
    await budgetAfterBlock('warmup');
   }
   // Release occurs at the already consumed 500 ms boundary. Neither clock,
   // the event reader, native muscle states nor wing phase is reinitialized.
   await checkpoint();setupStarted=performance.now();warmup.finish({neuralTimeMs:brain.timeMs,...(disturbance?{releaseVelocity:disturbance.releaseVelocity}:{})});
   const release=clock.release(snapshot({release:true}));releaseNativeTime=release.releaseNativeTimeSeconds;releaseNeuralTimeMs=release.releaseNeuralTimeMs;
   initialObservation=observe();score=isRecovery?recoveryModule.createRecoveryScore(duration,initialObservation,sceneProfile??undefined):createMaintainedFlightScore(duration,initialObservation,sceneProfile??undefined);
   warmupAudit=warmup.audit();
   initialCondition={stage:job.stage,seed:job.seed,kind:isRecovery?'recovery-scored-release':'maintained-flight-scored-release',initialization:clone(settings),preWarmupCondition,
    position:[body.x,body.y,body.z],quaternion:[...body.quaternion],velocity:Array.from(body.data.qvel.slice(0,6)),airborne:body.airborne,
    releaseNativeTime,releaseNeuralTimeMs,episodeTimeSeconds:0,warmupSteps,warmupAudit:clone(warmupAudit),
    ...(disturbance?{recoveryDisturbance:clone(disturbance)}:{}),
    placement:disturbance?'Explicit airborne placement and live BANC warmup, followed by one declared seeded velocity disturbance at release. No sustaining forces, takeoff or landing credit.':'Explicit airborne placement before live BANC warmup; root restraint removed at release. No takeoff or landing credit.'};
   provenance.releaseNativeTime=releaseNativeTime;provenance.releaseNeuralTimeMs=releaseNeuralTimeMs;
   onInitialState?.({body,world,fly,brain,input,inputSequence,encoder,sensoryFeedback,motorIndices,job,provenance,initialCondition,initialObservation,
    phase:'scored-release',releaseNativeTime,releaseNeuralTimeMs,episodeTimeSeconds:0,warmupSteps});
   lastFiniteFrame=maintainedFrame('scored');onFrame?.(lastFiniteFrame);lastFrameWall=performance.now();
   setupWallSeconds+=(performance.now()-setupStarted)/1000;setupStarted=null;
   if(score.state.reason&&score.state.phase==='failed')termination={terminated:true,success:false,reason:score.state.reason};
   while(!termination?.terminated&&steps*blockSeconds+1e-10<duration){
    await checkpoint();blockStarted=performance.now();blockPhase='scored';
    finalObservation=await advanceFeedbackBlock('scored');
    const time=clock.advance(snapshot());if(time.scoredSteps!==steps)throw new Error('Maintained-flight scored counter mismatch');
    termination=score.step(finalObservation,blockSeconds);
    await budgetAfterBlock('scored');
   }
   await checkShader();lastFiniteFrame=maintainedFrame('scored');onFrame?.(lastFiniteFrame);
  }catch(e){if(e.name==='AbortError')cancelled=true;else error=e.message;}
  finally{
   // Completed physics can precede a throwing observer or clock check. Report
   // actual final clocks rather than the last successfully observed block.
   if(active?.body)lastAbsoluteNativeTime=active.body.time;
   if(active?.brain)lastAbsoluteNeuralTimeMs=active.brain.timeMs;
   if(setupStarted!==null)setupWallSeconds+=(performance.now()-setupStarted)/1000;
   if(blockStarted!==null)accountBlock();
   try{if(warmup){warmupAudit=warmup.audit();warmup.abort();}}catch(e){error??=e.message;}
   cleanup();
  }
  const simSeconds=steps*blockSeconds,success=!error&&!cancelled&&score?.state.success===true;
  const reason=cancelled?'cancelled':error?'simulation_error':termination?.reason||'time_limit';
  return {return:error?-10:score?.state.return??0,success,terminated:!!error||!!termination?.terminated,
   truncated:cancelled||(!error&&!termination?.terminated),reason,cancelled,simSeconds,steps,
   warmupSteps,warmupSeconds:warmupSteps*blockSeconds,totalBlocks,totalSimSeconds:totalBlocks*blockSeconds,
   nativeTimeSeconds:lastAbsoluteNativeTime,neuralMs:lastAbsoluteNeuralTimeMs,releaseNativeTime,releaseNeuralTimeMs,
   parameters:vector,seed:job.seed,stage:job.stage,environmentVersion:config.environmentVersion,backend:provenance.backend||backend,
   bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,modelFingerprint:config.modelFingerprint,configHash,provenance,
   metrics:{...(score?.state||{}),error,wallSeconds:(performance.now()-started)/1000,setupWallSeconds,executionWallSeconds,warmupExecutionWallSeconds,scoredExecutionWallSeconds,
    initialCondition,initialObservation,finalObservation,warmupAudit,releaseNativeTime,releaseNeuralTimeMs,
    actualNeuralBackend:provenance.backend||backend,sensoryFeedback:lastSensorySummary,vision:visionEnabled,frameCountIsNotPhysicsSteps:true,criteria:isRecovery?recoveryModule.recoveryFlightCriteria(sceneProfile??undefined):maintainedFlightCriteria(sceneProfile??undefined),
    finalPosition:lastFiniteFrame?.position,finalNeuralSpikes:lastFiniteFrame?.neuralSpikes||0}};
 }
 function evaluateWithMaintained(job,options){return ['maintained_flight','recovery'].includes(job?.stage)?evaluateMaintained(job,options):evaluate(job,options);}

 return {config,configHash,modelFingerprint:config.modelFingerprint,ready,evaluate:evaluateWithMaintained,dispose(){if(disposed)return;disposed=true;try{cleanup();}finally{brainResources.dispose();}}};
}
