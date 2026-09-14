import {parameterValues,parameterizedModel,seededRandom,createEpisodeScore,CRITERIA,STAGES,clip} from './episode.js';
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const sha=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
const clone=x=>structuredClone(x);
async function fetchText(url){const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw new Error(`Training asset unavailable: ${url}`);return response.text();}
async function fetchJson(url){return JSON.parse(await fetchText(url));}
function assetRecords(assets){return Object.entries(assets||{}).map(([key,value])=>typeof value==='string'?{url:key,sha256:value}:{...value,url:value.url||key});}
async function checkAssets(config,onProgress){
 const records=assetRecords(config.assets);if(!records.length)throw new Error('Training asset manifest has not been built');
 for(let i=0;i<records.length;i++){
  const entry=records[i];if(!/^[a-f0-9]{64}$/.test(entry.sha256))throw new Error('Invalid training asset digest: '+entry.url);
  const response=await fetch(entry.url,{cache:'no-store'});if(!response.ok)throw new Error('Missing training asset: '+entry.url);const bytes=await response.arrayBuffer();
  if(entry.bytes!==undefined&&entry.bytes!==bytes.byteLength||await sha(bytes)!==entry.sha256)throw new Error('Training asset checksum mismatch: '+entry.url);
  onProgress?.({message:'Checking training assets',completed:i+1,total:records.length});
 }
 return records;
}

/** One real BANC network and native FlyBody, with an explicitly bounded policy
 * vector. Reward/targets are observations only; no root control is introduced. */
export async function createTrainingEnvironment(requestedConfig,{configUrl='/training/config.json',onProgress}={}){
 const text=await fetchText(configUrl),config=JSON.parse(text),configHash=await sha(new TextEncoder().encode(text));
 if(requestedConfig&&JSON.stringify(requestedConfig)!==JSON.stringify(config))throw new Error('Initialization config differs from canonical training config');
 parameterValues(config,config.parameters.map(p=>p.initial));
 if(!/^[a-f0-9]{64}$/.test(config.modelFingerprint))throw new Error('Training model fingerprint is not finalized');
 const assets=await checkAssets(config,onProgress);
 const [{loadBancModel,WebGPUBrain,WasmBrain,createWasmCore},{createBancBodyFactory},{SensoryEncoder},{createBancTasteMapper},{createBancSensoryCurrentMapper}]=await Promise.all([
  import('/banc-engine/src/index.js'),import('../flybody-world.js'),import('../sensory-encoder.js'),import('../banc-taste.js'),import('../banc-sensory-current.js')]);
 const [base,core,bodyFactory,habitat,groups,sensory,supplement]=await Promise.all([loadBancModel(),createWasmCore(),createBancBodyFactory(),fetchJson('/habitat.json'),fetchJson('/banc-data/console/groups.json'),fetchJson('/banc-data/console/sensory-inputs.json'),fetchJson('/body-model/banc-taste-peg-annotations.json')]);
 if(base.manifest.dt_ms!==config.dtMs||supplement.prepared_ids_sha256!==base.manifest.files['ids.bin'].sha256||supplement.prepared_io_sha256!==base.manifest.files['io.json'].sha256)throw new Error('Training graph or sensory identity mismatch');
 const tasteMapper=createBancTasteMapper([...base.io.sensory,...supplement.annotations],groups.sweet);
 if(tasteMapper.coverage.unmapped.some(row=>row.reason==='missing sensory annotation'))throw new Error('Training taste annotations incomplete');
 const makeEncoder=environment=>new SensoryEncoder(sensory,groups,environment,{tasteMapper});
 const calibrationEncoder=makeEncoder({odor:()=>0}),externalIndices=calibrationEncoder.indices;
 const mapper=await createBancSensoryCurrentMapper(core,base,{indices:externalIndices,onProfile:(_,i,n)=>onProgress?.({message:'Calibrating isolated sensory interface',completed:i+1,total:n})});
 // Leak-scaled cells must never include any externally driven input profile.
 parameterizedModel(base,parameterValues(config,config.parameters.map(p=>p.initial)).gains,externalIndices);
 const modality=new Map();for(const i of [...groups.odor_left,...groups.odor_right])modality.set(i,'odor');for(const i of groups.sweet)modality.set(i,'taste');for(const c of sensory.channels)for(const i of c.indices)modality.set(i,'body');
 const motorIndices=Uint32Array.from(base.io.motor_neurons,m=>m.index);
 let backend=null,active=null,disposed=false;
 const makeBrain=async model=>{
  if(backend==='wasm')return new WasmBrain(core,model);
  try{const brain=await WebGPUBrain.create(model);backend='webgpu';return brain;}
  catch(error){if(backend==='webgpu')throw error;onProgress?.({message:'WebGPU unavailable; using real WASM BANC',detail:error.message});backend='wasm';return new WasmBrain(core,model);}
 };
 const cleanup=()=>{if(active){try{active.brain?.dispose();}finally{active.world?.dispose();active=null;}}};
 const shader=assets.find(a=>a.url.endsWith('/neural.wgsl'));
 const checkShader=async()=>{if(shader&&await sha(new TextEncoder().encode(await fetchText(shader.url)))!==shader.sha256)throw new Error('Neural shader changed after model fingerprint');};
 function foodTarget(world,body){return world.habitat.fruit.reduce((best,f)=>Math.hypot(f.x/10-body.x,f.z/10-body.y)<Math.hypot(best.x/10-body.x,best.z/10-body.y)?f:best);}
 function distance(body,food){return Math.max(0,Math.hypot(body.x-food.x/10,body.y-food.z/10)-food.radius/10);}
 function frame(stage,metrics={}){
  const {world,body:b}=active,d=b.data,point=(buffer,id)=>Array.from(buffer.slice(id*3,id*3+3));
  const wings=world.wingLandmarks.map(l=>({side:l.side,position:point(d.geom_xpos,l.geom),rotation:Array.from(d.geom_xmat.slice(l.geom*9,l.geom*9+9)),size:[...l.size],anchor:point(d.xpos,l.body)}));
  const mouth={anchors:world.mouthLandmarks.joints.map(id=>point(d.xanchor,id)),ellipsoids:world.mouthLandmarks.geoms.map(l=>({position:point(d.geom_xpos,l.id),rotation:Array.from(d.geom_xmat.slice(l.id*9,l.id*9+9)),size:[...l.size]}))};
  return {time:b.time,simSeconds:b.time,stage,position:[b.x,b.y,b.z],quaternion:[...b.quaternion],feet:b.feet.map(p=>[...p]),legs:world.metadata.leg_bodies.map((ids,i)=>ids.map((id,k)=>k===2?[...b.feet[i]]:point(d.xpos,id))),wings,mouth,
   food:world.habitat.fruit.map(f=>({kind:f.kind,position:[f.x/10,f.z/10,f.y/10],radiusCm:f.radius/10,lengthCm:(f.length||0)/10,rotation:f.angle||0,path:f.path?.map(([x,y])=>[x/10,y/10,f.y/10]),remaining:f.remaining})),
   bowl:{radiusCm:6.5,floor:{baseCm:.15,radialCoefficientPerCm:.037},ceilingCm:world.habitat.ceiling/10},
   contacts:{environment:b.environmentContactCount,food:b.foodContactCount,legs:Array.from(b.legFoodContact),mouth:Array.from(b.mouthFoodContact),wings:Array.from(b.wingFoodContact)},
   phase:b.monitor.phase,motion:active.fly.motion,metrics:{...metrics},neuralMs:active.brain?.timeMs||0,neuralSpikes:active.lastSpikes||0,vision:false};
 }
 function createWorld(stage,seed,gains,families){
  const random=seededRandom(seed),fly={...clone(habitat.flies[0]),id:1,brain:{time_ms:0,motor:{},motorNeuronRates:Array(motorIndices.length).fill(0)}};
  if(['approach','landing','sequence'].includes(stage)){
   const angle=random()*Math.PI*2,radius=48+random()*1.5;fly.x=Math.cos(angle)*radius;fly.z=Math.sin(angle)*radius;
   const food=habitat.fruit.reduce((best,f)=>Math.hypot(f.x-fly.x,f.z-fly.z)<Math.hypot(best.x-fly.x,best.z-fly.z)?f:best);
   fly.heading=Math.atan2(food.z-fly.z,food.x-fly.x)+.4+(random()-.5)*.2;
  }else{fly.x+=(random()-.5)*.6;fly.z+=(random()-.5)*.6;fly.heading+=(random()-.5)*.3;}
  const world=bodyFactory(habitat.fruit,[fly],{movementMode:'direct',motorCoupling:true,flightEnabled:true}),body=world.bodies.get(1);
  body.wings.setInterpreterParameters({
   powerGain:gains.flight_power_log_gain,
   deploymentTauScale:gains.flight_deployment_tau_log_scale,
   frequencyScale:gains.flight_frequency_log_scale,
   steeringBiasGain:gains.flight_steering_bias_log_gain,
   steeringAmplitudeGain:gains.flight_steering_amplitude_log_gain
  });
  const originalRates=world.rates.bind(world);world.rates=f=>new Map(Array.from(originalRates(f),([index,rate])=>{
   const family=families.get(index),scale=family&&family!=='wing'?gains[`muscle_${family}_log_gain`]:1;return [index,rate*scale];
  }));
  body.data.qvel.fill(0);
  if(stage==='posture'){body.data.qvel[3]=(random()-.5)*.8;body.data.qvel[4]=(random()-.5)*.8;}
  if(stage==='landing'||stage==='sequence'){
   body.data.qpos[2]+=.8;const target=foodTarget(world,body),heading=Math.atan2(target.z/10-body.y,target.x/10-body.x);
   body.data.qvel[0]=Math.cos(heading)*.5;body.data.qvel[1]=Math.sin(heading)*.5;body.data.qvel[2]=-.2;
  }
  // Initial conditions only. No reset or direct body force occurs thereafter.
  world.mj.mj_forward(world.model,body.data);body.refresh();body.monitor.resetContinuity('training episode initial condition');world.copyPose(fly,body,0);
  return {world,body,fly,initialCondition:{stage,seed,position:[body.x,body.y,body.z],quaternion:[...body.quaternion],velocity:Array.from(body.data.qvel.slice(0,6)),airborne:body.airborne,placement:'Native stance solver followed by the explicitly recorded initial velocity/airborne offset only.'}};
 }
 function observe(score){
  const {body:b,world,food}=active,up=1-2*(b.quaternion[1]**2+b.quaternion[2]**2),angularSpeed=Math.hypot(...b.data.qvel.slice(3,6));
  const finite=[b.time,b.x,b.y,b.z,b.vz,up,angularSpeed,b.internal.ingested,...b.quaternion].every(Number.isFinite);
  return {finite,up,angularSpeed,radius:Math.hypot(b.x,b.y),height:b.z,ceiling:world.habitat.ceiling/10,distance:distance(b,food),intake:b.internal.ingested,contacts:b.environmentContactCount,airborne:b.airborne,wingPower:b.wingPower,verticalSpeed:b.vz,footFoodContacts:Array.from(b.legFoodContact).reduce((s,x)=>s+x,0),probing:b.mouthContact&&b.proboscis>.05,landings:b.monitor.landingCount,flightQualified:b.monitor.flightEvidence.qualified===true,events:b.monitor.events,externalForce:[b.data.xfrc_applied,b.data.qfrc_applied].some(a=>Array.from(a).some(x=>x!==0)),score};
 }
 const ready=async()=>{
  if(disposed)throw new Error('Training environment disposed');
  cleanup();const {gains}=parameterValues(config,config.parameters.map(p=>p.initial)),applied=parameterizedModel(base,gains,externalIndices);
  try{active={...createWorld('posture',config.optimizer.seed,gains,applied.families)};active.brain=await makeBrain(applied.model);active.food=foodTarget(active.world,active.body);await checkShader();
   return {environmentVersion:config.environmentVersion,modelFingerprint:config.modelFingerprint,configHash,backend,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,vision:false,parameterCount:config.parameters.length,criteria:CRITERIA,trainedCellCounts:applied.counts,frame:frame('posture'),limitations:['Vision is disabled in v2; the observer preview is not retinal input.','Model parameters are learned hypotheses, not measured biological physiology.','Flight training changes only the thin BANC-to-FlyBody interpreter; native wingbeat tables, rigid-wing collision and fluid forces remain fixed.']};
  }finally{cleanup();}
 };
 async function evaluate(job,{checkpoint=async()=>delay(0),previewHz=6,dutyCycle=.65,getBudget,onFrame,onProgress}={}){
  if(disposed)throw new Error('Training environment disposed');if(active)throw new Error('Concurrent training evaluation');
  const {vector,gains}=parameterValues(config,job.parameters);
  if(!Number.isInteger(job.seed)||job.seed<0||job.seed>0xffffffff||!STAGES.includes(job.stage))throw new Error('Invalid episode seed or stage');
  const stageConfig=config.stages.find(s=>s.id===job.stage),duration=job.durationSeconds??stageConfig.durationSeconds;
  if(!Number.isFinite(duration)||duration<=0||duration>stageConfig.durationSeconds||Math.abs(duration*1000/config.bodyBlockMs-Math.round(duration*1000/config.bodyBlockMs))>1e-8)throw new Error('Invalid episode duration');
  const started=performance.now(),applied=parameterizedModel(base,gains,externalIndices);let score,steps=0,termination=null,cancelled=false,error=null,lastFrameWall=-Infinity,lastProgressWall=-Infinity,lastFiniteFrame=null;
  const provenance={environmentVersion:config.environmentVersion,backend:null,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,modelFingerprint:config.modelFingerprint,configHash,seed:job.seed,stage:job.stage,vision:false};
  for(const name of ['generation','pair','pairId','sign','parametersHash'])if(job[name]!==undefined)provenance[name]=job[name];
  let initialCondition;
  try{
   await checkpoint();active={...createWorld(job.stage,job.seed,gains,applied.families)};initialCondition=active.initialCondition;active.brain=await makeBrain(applied.model);provenance.backend=backend;await checkShader();
   const {brain,world,body,fly}=active;active.food=foodTarget(world,body);const encoder=makeEncoder(world.habitat),input=new Float32Array(base.manifest.neuron_count);
   score=createEpisodeScore(job.stage,duration,{distance:distance(body,active.food),height:body.z,intake:body.internal.ingested,airborne:body.airborne});
   lastFiniteFrame=frame(job.stage,score.state);onFrame?.(lastFiniteFrame);lastFrameWall=performance.now();
   while(steps*config.bodyBlockMs/1000+1e-10<duration){
    await checkpoint();const blockStarted=performance.now();
    const encoded=encoder.update(fly,null,{odor:true,taste:true,vision:false,bodySense:true});
    if(encoded)for(let k=0;k<encoded.indices.length;k++){const index=encoded.indices[k],kind=modality.get(index),rate=encoded.ratesHz[k]*(kind?gains[`sensory_${kind}_log_gain`]:1);input[index]=mapper.current(index,rate);}
    await brain.step(config.bodyBlockMs/config.dtMs,input,body.internal,true);const state=await brain.readState(motorIndices);
    if(!state.every(Number.isFinite))throw new Error('Nonfinite BANC motor state');
    fly.brain={time_ms:brain.timeMs,motor:{},motorNeuronRates:Array.from(motorIndices,(_,k)=>state[k*8+4])};fly.senses=encoder.sample.food;fly.sensory=encoder.sample;active.lastSpikes=state.totalSpikes;
    world.tick(config.bodyBlockMs/1000);steps++;
    if(Math.abs(body.time-brain.timeMs/1000)>1e-8)throw new Error('Neural/native feedback clocks diverged');
    termination=score.step(observe(),config.bodyBlockMs/1000);
    const wall=performance.now(),budget=getBudget?.()||{previewHz,dutyCycle},hz=clip(Number(budget.previewHz)||0,0,10),duty=clip(Number(budget.dutyCycle)||.65,.05,1);
    if(hz&&wall-lastFrameWall>=1000/hz){lastFiniteFrame=frame(job.stage,score.state);onFrame?.(lastFiniteFrame);lastFrameWall=wall;}
    if(wall-lastProgressWall>=500){onProgress?.({stage:job.stage,simSeconds:body.time,steps,return:score.state.return,success:score.state.success,backend,bodyBackend:'mujoco-wasm'});lastProgressWall=wall;}
    if(termination.terminated)break;
    const idle=(performance.now()-blockStarted)*(1/duty-1);for(let remaining=idle;remaining>0;remaining-=20){await delay(Math.min(20,remaining));await checkpoint();}
   }
   await checkShader();lastFiniteFrame=frame(job.stage,score.state);onFrame?.(lastFiniteFrame);
  }catch(e){if(e.name==='AbortError')cancelled=true;else error=e.message;}
  finally{cleanup();}
  const simSeconds=steps*config.bodyBlockMs/1000,success=!error&&!cancelled&&score?.state.success===true;
  const reason=cancelled?'cancelled':error?'simulation_error':termination?.reason||'time_limit';
  return {return:error?-10:score?.state.return??0,success,terminated:!!error||!!termination?.terminated,truncated:cancelled||(!error&&!termination?.terminated),reason,cancelled,simSeconds,steps,parameters:vector,seed:job.seed,stage:job.stage,
   environmentVersion:config.environmentVersion,backend:provenance.backend||backend,bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,modelFingerprint:config.modelFingerprint,configHash,provenance,
   metrics:{...(score?.state||{}),error,wallSeconds:(performance.now()-started)/1000,initialCondition,actualNeuralBackend:provenance.backend||backend,vision:false,frameCountIsNotPhysicsSteps:true,criteria:CRITERIA,finalPosition:lastFiniteFrame?.position,finalNeuralSpikes:lastFiniteFrame?.neuralSpikes||0}};
 }
 return {config,configHash,modelFingerprint:config.modelFingerprint,ready,evaluate,dispose(){if(disposed)return;disposed=true;cleanup();}};
}
