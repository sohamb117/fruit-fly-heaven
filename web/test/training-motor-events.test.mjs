import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
import {createHash} from 'node:crypto';

// Exercise the real environment control flow and real event reader, using a
// tiny deterministic neural/body fixture instead of a full BANC/native run.
const key='__trainingMotorEventFixture';
const get=`const f=()=>globalThis.${key};`;
const modules=new Map([
 ['./episode.js',`${get} export const clip=(x,a,b)=>Math.max(a,Math.min(b,x)); export const STAGES=['takeoff']; export const CRITERIA={minimumDurationSeconds:{takeoff:1}};
  export const seededRandom=()=>()=>.5; export const parameterValues=(config,vector,io)=>{
   if(io){f().checkedDecoderIo=io;if(f().decoderContractError)throw new Error(f().decoderContractError);}
   return {vector:[...vector],interpreter:{},...(config.parameterContract==='fixture-motor-decoder'?{motorDecoder:[...vector]}:{})};};
  export function createEpisodeScore(){return {state:{return:0,success:false},step(){f().scores++;return {terminated:f().scores===2,reason:'fixture_complete'};}}}`],
 ['./flight-observation.js',`export const measureFlightObservation=()=>({environmentContacts:3,nonFootEnvironmentContacts:0,footSupportCount:3,footSupportFraction:1});
  export const measureFlightKinematics=()=>({height:0,verticalSpeed:0,speedCmPerSecond:0,position:[0,0,0]});`],
 ['./flight-telemetry.js',`export const makeFlightTelemetry=()=>({});`],
 ['./brain-resources.js',`${get} export function createTrainingBrainResources({model,backend}){f().brainModel=model;f().requestedBackend=backend;return {get backend(){return f().reportedBackend??(backend==='wasm'?'wasm':'fixture');},async create(){const b=f().makeBrain();f().brains.push(b);return b;},dispose(){f().resourcesDisposed=true;}};}`],
 ['/banc-engine/src/index.js',`${get} export const loadBancModel=async()=>f().base,createWasmCore=async options=>{f().wasmOptions=options;return {};}; export const intrinsicLayout=model=>{f().intrinsicModel=model;return {eventContract:f().intrinsicContract};}; export class WebGPUBrain{} export class WasmBrain{}`],
 ['../flybody-world.js',`${get} export const createBancBodyFactory=async()=>f().makeWorld;`],
 ['../sensory-encoder.js',`export class SensoryEncoder{constructor(){this.indices=new Uint32Array();this.sample={food:null};}update(){return null;}}`],
 ['../banc-tegula.js',`${get} export const createTegulaSensoryManifest=(model,manifest,config)=>{f().tegulaModel=model;f().tegulaConfig=config;return manifest;};`],
 ['../banc-taste.js',`export const createBancTasteMapper=()=>({coverage:{unmapped:[]}});`],
 ['../banc-sensory-current.js',`export const createBancSensoryCurrentMapper=async()=>({current(){throw new Error('Empty sensory fixture');}});`],
 ['./maintained-flight-objective.js',`${get} export const MAINTAINED_FLIGHT_CRITERIA={}; export const maintainedFlightCriteria=()=>({});
  export function createMaintainedFlightScore(){return {state:{return:0,success:false},step(){f().scores++;return {terminated:f().scores===2,reason:'fixture_complete'};}};}`],
 ['./airborne-reset-contract.js',`${get} export const selectFlightInitialCondition=(_stage,value)=>value;
  export function createMaintainedFlightClock(){let steps=0;return {warmupBlocks:250,release(snapshot){f().releaseSnapshot=snapshot;return {releaseNativeTimeSeconds:snapshot.nativeTimeSeconds,releaseNeuralTimeMs:snapshot.neuralTimeMs};},advance(){return {scoredSteps:++steps};}};}`],
 ['./airborne-reset.js',`${get} export function beginAirborneWarmup({body}){let held=true;f().warmupBrain=f().brains.at(-1);return {finish(){held=false;},audit(){return {rootRestraintActive:held,rootWriteCount:0};},abort(){f().warmupAborted=true;}};}`],
]);
const environmentURL=new URL('../training/environment.js',import.meta.url).href;
const helperURL=new URL('../../packages/banc-runtime/src/motor-events.js',import.meta.url).href;
const hooks=registerHooks({resolve(specifier,context,next){
 if(specifier==='/banc-engine/src/motor-events.js')return {url:helperURL,shortCircuit:true};
 if(context.parentURL===environmentURL&&modules.has(specifier))return {url:'data:text/javascript,'+encodeURIComponent(modules.get(specifier)),shortCircuit:true};
 return next(specifier,context);
}});
const {createTrainingEnvironment}=await import(environmentURL);
const previousFetch=globalThis.fetch;
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');

function fixture(){
 const fixture={scores:0,reads:[],order:[],brains:[],worlds:[],ticks:[],resourcesDisposed:false};
 const params=new Float32Array(805*16);for(let i=0;i<805;i++)params[i*16+5]=2;
 fixture.base={manifest:{dt_ms:.5,neuron_count:805,files:{'ids.bin':{sha256:'a'.repeat(64)},'io.json':{sha256:'b'.repeat(64)}}},params,
  io:{motor_neurons:Array.from({length:805},(_,index)=>({index})),sensory:[],muscles:[
   {kind:'asynchronous_wing',indices:Array.from({length:24},(_,i)=>24-i)},
   {kind:'wing_steering_assumption',indices:Array.from({length:24},(_,i)=>48-i)},
   {kind:'leg',indices:[0]},
  ]}};
 const asset='event-fixture';
 fixture.config={parameters:[{initial:0}],modelFingerprint:'c'.repeat(64),assets:{'/fixture.bin':digest(asset)},dtMs:.5,bodyBlockMs:2,
  stage:'takeoff',optimizer:{seed:1},stages:[{id:'takeoff',durationSeconds:1}],environmentVersion:'event-fixture'};
 const habitat={flies:[{x:0,z:0,heading:0,motion:'standing'}],fruit:[]};
 const json={'/training/config.json':fixture.config,'/habitat.json':habitat,'/banc-data/console/groups.json':{sweet:[]},'/banc-data/console/sensory-inputs.json':{},
  '/body-model/banc-taste-peg-annotations.json':{annotations:[],prepared_ids_sha256:'a'.repeat(64),prepared_io_sha256:'b'.repeat(64)}};
 fixture.supplement=json['/body-model/banc-taste-peg-annotations.json'];
 globalThis.fetch=async url=>{const name=String(url);if(name==='/banc-engine/dist/core.wasm'&&fixture.wasmBytes){fixture.wasmFetches=(fixture.wasmFetches||0)+1;return new Response(fixture.wasmBytes);}if(name==='/fixture.bin'||name==='/motor-decoder.js'||fixture.extraAssetUrls?.includes(name))return new Response(asset);if(name==='/banc-data/ids.bin'&&fixture.idsBytes)return new Response(fixture.idsBytes);if(name in json)return new Response(JSON.stringify(json[name]));throw new Error('Unexpected fixture fetch: '+name);};
 fixture.makeBrain=()=>({timeMs:0,disposed:false,backend:fixture.actualBackend??(fixture.requestedBackend==='wasm'?'wasm':'fixture'),
  async step(count){assert.equal(count,4);this.timeMs+=2;fixture.order.push('step:'+this.timeMs);},
  async readState(indices,options){
   const stride=options?.includeSpikeTime?9:8;fixture.reads.push({time:this.timeMs,stride,indices:Array.from(indices)});fixture.order.push(`read${stride}:${this.timeMs}`);
   const result=new Float32Array(indices.length*stride);
   indices.forEach((id,k)=>{const spiked=this.timeMs>=2&&(id===1||id===2);result[k*stride+3]=+spiked;result[k*stride+4]=id+this.timeMs/8;
    if(stride===9)result[k*stride+8]=spiked?(id===1?(fixture.dlmTime??1.5):2):-1e30;});
   result.totalSpikes=this.timeMs>=2?2:0;return result;
  },dispose(){this.disposed=true;fixture.order.push('dispose-brain');}});
 fixture.makeWorld=(_fruit,[fly])=>{
  const data={qvel:new Float64Array(6),qfrc_applied:new Float64Array(6),xfrc_applied:new Float64Array(6),geom_xpos:[],geom_xmat:[],xpos:[],xanchor:[]};
  const body={data,time:0,x:0,y:0,z:0,quaternion:[1,0,0,0],feet:Array.from({length:6},()=>[0,0,0]),internal:{hunger:0,insulin:0,akh:0},airborne:false,
   enableWingMotorEvents(config,eventContract){if(fixture.eventError)throw new Error(fixture.eventError);fixture.order.push('enable-events');this.eventConfig=structuredClone(config);this.eventContract=eventContract;this.eventPackets=[];this._wingMotorEvents={initialized:true,observedMs:0,elapsedMs:0,adapter:{snapshot:()=>({pending:null})}};this.remainder=0;},
   enableMotorDecoder(vector){assert(this.eventConfig);assert.equal(this.time,0);assert.equal(this.eventPackets.length,0);fixture.order.push('enable-decoder');if(fixture.decoderError)throw new Error(fixture.decoderError);this.decoderVector=[...vector];},
   enableWingLoadFeedback(){fixture.order.push('enable-wing-load');this.wingLoadEnabled=true;},
   acceptWingMotorEvents(packet){assert(this.eventConfig);fixture.order.push('consume:'+packet.timeMs);this.eventPackets.push(structuredClone(packet));this._wingMotorEvents.observedMs=packet.timeMs;},
   wings:{phase:.25,frequencyHz:235.8,setInterpreterParameters(){}},wingPower:0,environmentContactCount:3,foodContactCount:0,
   legFoodContact:[],mouthFoodContact:[],wingFoodContact:[],monitor:{resetContinuity(){}},refresh(){}};
  const world={bodies:new Map([[1,body]]),mj:{mj_forward(){fixture.order.push('initial-forward');}},model:{},habitat:{fruit:[],ceiling:65},metadata:{leg_bodies:[]},
   wingLandmarks:[],mouthLandmarks:{joints:[],geoms:[]},rates:()=>new Map(),copyPose(){fixture.order.push('initial-pose');},disposed:false,
   tick(dt){if(body.eventConfig)assert(Math.abs(body.eventPackets.at(-1).timeMs-(body.time+dt)*1000)<1e-8);fixture.order.push('tick:'+body.time);fixture.ticks.push({dt,time:body.time,rates:[...fly.brain.motorNeuronRates],phase:body.wings.phase});body.time+=dt;body.wings.phase+=.5;if(body._wingMotorEvents)body._wingMotorEvents.elapsedMs=body._wingMotorEvents.observedMs;},
   dispose(){this.disposed=true;fixture.order.push('dispose-world');}};
  fixture.worlds.push(world);return world;
 };
 globalThis[key]=fixture;return fixture;
}
const job={parameters:[0],seed:1,stage:'takeoff',durationSeconds:1};
const options={checkpoint:async()=>{},previewHz:0,dutyCycle:1};
async function run(hook,eventMode=false){const f=fixture();if(eventMode)f.config.wingEventExcitation={test:'explicit-event-mode'};
 const environment=await createTrainingEnvironment();
 try{const result=await environment.evaluate(job,hook===undefined?options:{...options,onMotorEvents:hook(f)});return {f,result};}
 finally{environment.dispose();}}

test('no hook preserves the sole805×8 motor read and has no initial read',async()=>{
 const {f,result}=await run();assert.equal(result.metrics.error,null);assert.equal(result.steps,2);
 assert.deepEqual(f.reads.map(row=>[row.time,row.stride,row.indices.length]),[[2,8,805],[4,8,805]]);
 assert(f.brains.every(brain=>brain.disposed));assert(f.worlds.every(world=>world.disposed));
});

test('observer reads48 sorted actual wing IDs, initializes at0 and emits owned packets before physics',async()=>{
 const baseline=await run(),packets=[];
 const observed=await run(f=>packet=>{
  f.order.push('callback:'+packet.timeMs);packets.push(structuredClone(packet));
  assert.equal(f.ticks.length,packet.initialized?0:packet.timeMs/2-1);
  assert(!('brain' in packet)&&!('body' in packet)&&!('params' in packet));
  packet.indices.fill(800);packet.ratesHz.fill(9000);packet.counts.fill(999);if(packet.events[0])packet.events[0].index=799;
 });
 assert.equal(observed.result.metrics.error,null);assert.deepEqual(observed.f.ticks,baseline.f.ticks);
 assert.deepEqual(observed.f.reads.map(row=>[row.time,row.stride,row.indices.length]),[[0,9,48],[2,8,805],[2,9,48],[4,8,805],[4,9,48]]);
 assert.deepEqual(packets.map(packet=>[packet.initialized,packet.fromTimeMs,packet.timeMs,packet.bodyTimeSeconds,packet.wingPhaseRadians,packet.wingFrequencyHz]),[
  [true,null,0,0,.25,235.8],[false,0,2,0,.25,235.8],[false,2,4,.002,.75,235.8]]);
 packets.forEach(packet=>assert.deepEqual(Array.from(packet.indices),Array.from({length:48},(_,i)=>i+1)));
 assert.deepEqual(packets[0].events,[]);assert.deepEqual(packets[1].events,[{index:1,timeMs:1.5},{index:2,timeMs:2}]);assert.deepEqual(packets[2].events,[]);
 assert.deepEqual(Array.from(packets[1].ratesHz),baseline.f.ticks[0].rates.slice(1,49));
});

test('invalid hooks fail before creating a brain or body',async()=>{
 const f=fixture(),environment=await createTrainingEnvironment();
 try{await assert.rejects(environment.evaluate(job,{...options,onMotorEvents:3}),/must be a function/);assert.equal(f.brains.length,0);assert.equal(f.worlds.length,0);}
 finally{environment.dispose();}
});

test('a failing observer cleans up and never advances native physics',async()=>{
 const {f,result}=await run(()=>()=>{throw new Error('observer failed');});
 assert.equal(result.metrics.error,'observer failed');assert.equal(result.steps,0);assert.equal(f.ticks.length,0);
 assert(f.brains.every(brain=>brain.disposed));assert(f.worlds.every(world=>world.disposed));
});

test('event mode consumes baseline and every packet without an observer',async()=>{
 const {f,result}=await run(undefined,true);assert.equal(result.metrics.error,null);assert.equal(result.steps,2);
 const body=f.worlds[0].bodies.get(1);
 assert.deepEqual(body.eventConfig,{test:'explicit-event-mode'});
 assert.deepEqual(body.eventPackets.map(p=>[p.initialized,p.timeMs]),[[true,0],[false,2],[false,4]]);
 assert.deepEqual(body.eventPackets[1].events,[{index:1,timeMs:1.5},{index:2,timeMs:2}]);
 assert(f.order.indexOf('consume:2')<f.order.indexOf('tick:0'));
 assert(f.brains.every(brain=>brain.disposed)&&f.worlds.every(world=>world.disposed));
});

test('observer mutation cannot alter events already consumed by event-mode body',async()=>{
 const {f,result}=await run(f=>packet=>{
  assert.equal(f.worlds[0].bodies.get(1).eventPackets.at(-1).timeMs,packet.timeMs);
  packet.counts.fill(900);packet.indices.fill(800);packet.events.splice(0);
 },true);
 assert.equal(result.metrics.error,null);
 const packets=f.worlds[0].bodies.get(1).eventPackets;
 assert.equal(packets[1].counts[0],1);assert.equal(packets[1].indices[0],1);
 assert.deepEqual(packets[1].events,[{index:1,timeMs:1.5},{index:2,timeMs:2}]);
});

test('event setup failure releases the newly allocated body before creating a brain',async()=>{
 const f=fixture();f.config.wingEventExcitation={test:'invalid'};f.eventError='invalid event configuration';
 const environment=await createTrainingEnvironment();
 try{
  const result=await environment.evaluate(job,options);
  assert.equal(result.metrics.error,'invalid event configuration');assert.equal(result.steps,0);
  assert.equal(f.brains.length,0);assert.equal(f.worlds.length,1);assert(f.worlds[0].disposed);
 }finally{environment.dispose();}
});

test('intrinsic profile reaches brain and event consumers without mutating the base model',async()=>{
 const f=fixture();f.config.wingEventExcitation={test:'event-mode'};
 f.config.intrinsicModels={test:'model-declaration'};
 f.intrinsicContract={schema:1,profile:'dlm-snl-2023-v1',indices:[1]};f.dlmTime=1.7;
 const environment=await createTrainingEnvironment();
 try{
  assert.notEqual(f.brainModel,f.base);assert.equal(f.brainModel.params,f.base.params);
  assert.deepEqual(f.brainModel.manifest.intrinsic_models,f.config.intrinsicModels);
  assert.equal(f.intrinsicModel,f.brainModel);assert.equal(f.base.manifest.intrinsic_models,undefined);
  const result=await environment.evaluate(job,options);assert.equal(result.metrics.error,null);
  const body=f.worlds[0].bodies.get(1);assert.equal(body.eventContract,f.intrinsicContract);
  assert.deepEqual(body.eventPackets[1].events,[{index:1,timeMs:1.7},{index:2,timeMs:2}]);
 }finally{environment.dispose();}
});

test('tegula input verifies raw identities and enables the native observer after initial forward',async()=>{
 const f=fixture();f.config.tegulaFeedback={schema:1,profile:'tegula-fluid-load-v1',maxRateHz:100,halfLoadNative:.1};
 f.idsBytes=new Uint8Array(new BigUint64Array(805).buffer);f.base.manifest.files['ids.bin']={sha256:digest(f.idsBytes),bytes:f.idsBytes.length};
 f.supplement.prepared_ids_sha256=f.base.manifest.files['ids.bin'].sha256;
 const environment=await createTrainingEnvironment();
 try{
  assert(f.tegulaModel.ids instanceof BigUint64Array);assert.equal(f.tegulaModel.ids.length,805);
  assert.equal(f.tegulaModel.io,f.base.io);assert.equal(f.base.ids,undefined);assert.deepEqual(f.tegulaConfig,f.config.tegulaFeedback);
  const result=await environment.evaluate(job,options);assert.equal(result.metrics.error,null);
  assert(f.worlds[0].bodies.get(1).wingLoadEnabled);
  assert(f.order.indexOf('initial-forward')<f.order.indexOf('enable-wing-load'));
  assert(f.order.indexOf('enable-wing-load')<f.order.indexOf('initial-pose'));
 }finally{environment.dispose();}
});

test('tegula identity corruption rejects before allocating neural or body state',async()=>{
 const f=fixture();f.config.tegulaFeedback={schema:1};f.idsBytes=new Uint8Array(805*8);
 f.base.manifest.files['ids.bin'].bytes=f.idsBytes.length;
 await assert.rejects(createTrainingEnvironment(),/identity checksum mismatch/);
 assert.equal(f.brains.length,0);assert.equal(f.worlds.length,0);assert.equal(f.tegulaModel,undefined);
});

function decoderFixture(){
 const f=fixture();f.config.parameterContract='fixture-motor-decoder';f.config.wingEventExcitation={test:'event-mode'};
 f.config.assets['/motor-decoder.js']=f.config.assets['/fixture.bin'];return f;
}

test('decoder validates actual io and enables after event setup, before baseline/physics',async()=>{
 const f=decoderFixture(),environment=await createTrainingEnvironment();
 try{
  assert.equal(f.checkedDecoderIo,f.base.io);assert.equal(f.worlds.length,0);assert.equal(f.brains.length,0);
  const ready=await environment.ready();assert.equal(ready.parameterScope,'individual-mn-motor-decoder-only');
  assert.deepEqual(f.worlds[0].bodies.get(1).decoderVector,[0]);assert(f.worlds[0].disposed);
  assert(!ready.limitations.some(line=>line.includes('27 wing')));
  const candidate={...job,parameters:[-.25]},result=await environment.evaluate(candidate,options);
  assert.equal(result.metrics.error,null);assert.deepEqual(result.parameters,[-.25]);
  const body=f.worlds[1].bodies.get(1);assert.deepEqual(body.decoderVector,[-.25]);
  assert(f.order.indexOf('enable-events')<f.order.indexOf('enable-decoder'));
  assert(f.order.indexOf('enable-decoder')<f.order.indexOf('initial-forward'));
  assert.deepEqual(body.eventPackets.map(packet=>packet.timeMs),[0,2,4]);
  assert.deepEqual(f.reads.map(row=>[row.time,row.stride]),[[0,9],[2,8],[2,9],[4,8],[4,9]]);
 }finally{environment.dispose();}
});

test('decoder identity mismatch or missing executable pin rejects before body or brain state',async()=>{
 for(const mode of ['identity','pin']){
  const f=decoderFixture();if(mode==='identity')f.decoderContractError='decoder io mismatch';else delete f.config.assets['/motor-decoder.js'];
  await assert.rejects(createTrainingEnvironment(),mode==='identity'?/decoder io mismatch/:/pinned executable/);
  assert.equal(f.worlds.length,0);assert.equal(f.brains.length,0);
 }
});

test('decoder initialization failure disposes the body and never starts brain/physics',async()=>{
 const f=decoderFixture();f.decoderError='decoder initialization failed';const environment=await createTrainingEnvironment();
 try{
  const result=await environment.evaluate(job,options);assert.equal(result.metrics.error,f.decoderError);
  assert.equal(result.steps,0);assert.equal(f.brains.length,0);assert.equal(f.ticks.length,0);
  assert.equal(f.worlds.length,1);assert(f.worlds[0].disposed);
 }finally{environment.dispose();}
});

function wasmFixture(){
 const f=fixture();f.wasmBytes=Uint8Array.of(0,97,115,109,1,0,0,0);
 const moduleSha256=digest(f.wasmBytes);f.config.assets['/banc-engine/dist/core.wasm']=moduleSha256;
 f.config.optimizer.acceptance={nativeExecution:{backend:'wasm',moduleSha256}};return f;
}

test('explicit WASM instantiates verified bytes and reports matching ready/result evidence',async()=>{
 const f=wasmFixture(),environment=await createTrainingEnvironment();
 try{
  assert.equal(f.requestedBackend,'wasm');assert.equal(f.wasmFetches,1);
  assert(f.wasmOptions.wasmBinary instanceof Uint8Array);assert.deepEqual(f.wasmOptions.wasmBinary,f.wasmBytes);
  assert.notEqual(f.wasmOptions.wasmBinary.buffer,f.wasmBytes.buffer);
  const ready=await environment.ready();assert.equal(ready.backend,'wasm');assert.equal(ready.neuralEngine,'wasm');
  assert.deepEqual(ready.wasmExecution,f.config.optimizer.acceptance.nativeExecution);assert.equal(Object.hasOwn(ready,'nativeWebGPU'),false);
  ready.wasmExecution.moduleSha256='f'.repeat(64);
  const result=await environment.evaluate(job,options);assert.equal(result.metrics.error,null);
  assert.equal(result.backend,'wasm');assert.equal(result.metrics.actualNeuralBackend,'wasm');assert.equal(result.provenance.backend,'wasm');
  assert.equal(result.provenance.neuralEngine,'wasm');assert.deepEqual(result.provenance.wasmExecution,f.config.optimizer.acceptance.nativeExecution);
  assert.equal(Object.hasOwn(result.provenance,'nativeWebGPU'),false);assert.equal(f.wasmFetches,1);
 }finally{environment.dispose();}
});

test('WASM pin corruption or absent bytes fails before allocating brain or body',async()=>{
 for(const failure of ['checksum','missing']){
  const f=wasmFixture();if(failure==='checksum')f.wasmBytes[0]=255;else delete f.config.assets['/banc-engine/dist/core.wasm'];
  await assert.rejects(createTrainingEnvironment(),failure==='checksum'?/checksum mismatch/:/verified binary/);
  assert.equal(f.brains.length,0);assert.equal(f.worlds.length,0);assert.equal(f.wasmOptions,undefined);
 }
});

test('WASM cohort refuses inconsistent actual backend before reporting ready',async()=>{
 for(const field of ['reportedBackend','actualBackend']){
  const f=wasmFixture();f[field]='webgpu';const environment=await createTrainingEnvironment();
  try{await assert.rejects(environment.ready(),/backend mismatch/);assert(f.brains.every(b=>b.disposed));assert(f.worlds.every(w=>w.disposed));assert.equal(f.ticks.length,0);}
  finally{environment.dispose();}
 }
});

test('legacy initialization retains automatic backend selection and adds no WASM claim',async()=>{
 const f=fixture(),environment=await createTrainingEnvironment();
 try{
  assert.equal(f.requestedBackend,'auto');assert.equal(f.wasmOptions,undefined);
  const ready=await environment.ready(),result=await environment.evaluate(job,options);
  for(const value of [ready,result.provenance]){assert.equal(Object.hasOwn(value,'wasmExecution'),false);assert.equal(Object.hasOwn(value,'neuralEngine'),false);}
 }finally{environment.dispose();}
});

test('maintained WASM warmup and scored blocks keep one actual backend and matching result provenance',async()=>{
 const f=wasmFixture();f.config.stage='maintained_flight';f.config.stages=[{id:'maintained_flight',durationSeconds:5}];
 f.config.initialCondition={warmupSeconds:.5};f.config.wingEventExcitation={test:'event-mode'};
 f.extraAssetUrls=['/training/maintained-flight-objective.js','/training/airborne-reset-contract.js','/training/airborne-reset.js','/body-model/flybody-mujoco.json'];
 for(const url of f.extraAssetUrls)f.config.assets[url]=f.config.assets['/fixture.bin'];
 const environment=await createTrainingEnvironment();
 try{
  const result=await environment.evaluate({...job,stage:'maintained_flight',durationSeconds:5},options);
  assert.equal(result.metrics.error,null);assert.equal(result.backend,'wasm');assert.equal(result.provenance.neuralEngine,'wasm');
  assert.deepEqual(result.provenance.wasmExecution,f.config.optimizer.acceptance.nativeExecution);
  assert.equal(result.warmupSteps,250);assert.equal(result.steps,2);assert.equal(f.brains.length,1);
  assert.equal(f.warmupBrain,f.brains[0]);assert.equal(result.releaseNeuralTimeMs,500);assert.equal(result.neuralMs,504);
  assert.equal(f.releaseSnapshot.bodyEventElapsedMs,500);assert.equal(f.releaseSnapshot.bodyEventObservedMs,500);
  assert.equal(f.wasmFetches,1);assert(f.warmupAborted);assert(f.brains[0].disposed);
 }finally{environment.dispose();}
});

test.after(()=>{hooks.deregister();globalThis.fetch=previousFetch;delete globalThis[key];});
