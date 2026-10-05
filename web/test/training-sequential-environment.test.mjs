// Real environment orchestration, deterministic small body/brain fixtures.
// Parameter arithmetic and full BANC sensory mapping have their own real-data
// tests; these checks isolate per-candidate reconstruction and hook timing.
import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
import {createHash} from 'node:crypto';
import {DEFAULT_RECOVERY_DISTURBANCE,createRecoveryDisturbance} from '../training/recovery-objective.js';
import {SPACIOUS_MAINTAINED_SCENE} from '../flight-scene-profile.js';
const KEY='__sequentialEnvironmentFixture',get=`const f=()=>globalThis.${KEY};`;
const score=`${get} const make=()=>({state:{return:0,success:false},step(){return {terminated:++f().scoreCalls===2,success:false,reason:'fixture_complete'};}});`;
const recoveryURL=new URL('../training/recovery-objective.js',import.meta.url).href;
const mocks=new Map([
 ['./episode.js',`${get} export const STAGES=['takeoff','landing','flight','maintained_flight','recovery'];export const CRITERIA={minimumDurationSeconds:{takeoff:3,landing:8,flight:5}};
  export const clip=(x,a,b)=>Math.max(a,Math.min(b,x));export const seededRandom=()=>()=>.5;
  export function parameterValues(config,values,io){f().parameterIo=io??f().parameterIo;const vector=[...values],sensoryConfig=structuredClone(config);sensoryConfig.legProprioception={marker:Math.exp(vector[0])};sensoryConfig.haltereFeedback.maxCurrentPa=10*Math.exp(vector[0]);return {vector,sensoryConfig,sensoryCount:1,motorDecoder:vector.slice(1),interpreter:{}};}
  export function createEpisodeScore(){return {state:{return:0,success:false},step(){return {terminated:++f().scoreCalls===2,success:false,reason:'fixture_complete'};}};}`],
 ['./flight-observation.js',`export const measureFlightObservation=()=>({environmentContacts:0,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0});export const measureFlightKinematics=()=>({height:3,verticalSpeed:0,speedCmPerSecond:0,position:[0,0,3]});`],
 ['./flight-telemetry.js','export const makeFlightTelemetry=()=>({});'],
 ['./brain-resources.js',`${get} export function createTrainingBrainResources(){f().brainResourceBuilds++;return {backend:'fixture',create:async()=>f().brain(),dispose(){}};}`],
 ['./sensory-feedback.js',`${get} export async function createTrainingSensoryResources({config}){const resource={config:structuredClone(config),instances:[]};f().sensoryResources.push(resource);return {indices:Uint32Array.from(f().badLayout&&f().sensoryResources.length>1?[3]:[1]),visionEnabled:false,create(){const instance={updates:0,disposed:false};resource.instances.push(instance);const encoder={sample:{food:null}};return {encoder,update(){instance.updates++;return {indices:Uint32Array.of(1),ratesHz:Float32Array.of(config.legProprioception.marker),sample:encoder.sample};},summary(){return {marker:config.legProprioception.marker,updates:instance.updates};},dispose(){instance.disposed=true;}};}};}`],
 ['../banc-haltere.js',`${get} export function createHaltereCurrentMapper(settings){const record={settings:structuredClone(settings),writes:0,resets:0};f().haltereMappers.push(record);return {reset(){record.resets++;},writeInto(input){record.writes++;input[2]=settings.maxCurrentPa;return {marker:settings.maxCurrentPa};}};}`],
 ['/banc-engine/src/index.js',`${get} export const loadBancModel=async()=>{f().graphLoads++;return f().base;};export const createWasmCore=async()=>({});export const intrinsicLayout=()=>({});export class WebGPUBrain{}export class WasmBrain{}`],
 ['../flybody-world.js',`${get} export const createBancBodyFactory=async()=>{f().bodyFactoryBuilds++;return f().world;};`],
 ['../sensory-encoder.js','export class SensoryEncoder{}'],
 ['../banc-taste.js','export const createBancTasteMapper=()=>({coverage:{unmapped:[]}});'],
 ['../banc-sensory-current.js',`${get} export const createBancSensoryCurrentMapper=async()=>{f().currentMapperBuilds++;return {current:(_index,hz)=>hz};};`],
 ['./maintained-flight-objective.js',`${score}export const createMaintainedFlightScore=make;export const MAINTAINED_FLIGHT_CRITERIA={};export const maintainedFlightCriteria=()=>({fixture:true});`],
 ['./airborne-reset-contract.js',`export const selectFlightInitialCondition=(_stage,value)=>value;export function createMaintainedFlightClock(){let steps=0;return {warmupBlocks:250,release:s=>({releaseNativeTimeSeconds:s.nativeTimeSeconds,releaseNeuralTimeMs:s.neuralTimeMs}),advance:()=>({scoredSteps:++steps})};}`],
 ['./airborne-reset.js',`${get}export function beginAirborneWarmup({body}){let held=true;return {audit:()=>({rootRestraintActive:held,rootWriteCount:held?10000:10001}),finish({releaseVelocity}){held=false;f().releaseVelocity=releaseVelocity;if(releaseVelocity)body.data.qvel.set(releaseVelocity);},abort(){}};}`],
 ['./recovery-objective.js',`import {createRecoveryDisturbance} from '${recoveryURL}';${score}export {createRecoveryDisturbance};export const createRecoveryScore=make;export const recoveryFlightCriteria=()=>({stage:'recovery',fixture:true});`],
]);
const environmentURL=new URL('../training/environment.js',import.meta.url).href;
const eventURL=new URL('../../packages/banc-runtime/src/motor-events.js',import.meta.url).href;
const hooks=registerHooks({resolve(specifier,context,next){
 if(context.parentURL===environmentURL&&mocks.has(specifier))return {url:'data:text/javascript,'+encodeURIComponent(mocks.get(specifier)),shortCircuit:true};
 if(specifier==='/banc-engine/src/motor-events.js')return {url:eventURL,shortCircuit:true};return next(specifier,context);
}});
const {createTrainingEnvironment}=await import(environmentURL),savedFetch=globalThis.fetch;
const digest=x=>createHash('sha256').update(x).digest('hex'),asset='sequential-fixture',hash=digest(asset);

function fixture(){
 const f={sensoryResources:[],haltereMappers:[],brains:[],worlds:[],order:[],scoreCalls:0,graphLoads:0,bodyFactoryBuilds:0,brainResourceBuilds:0,currentMapperBuilds:0};globalThis[KEY]=f;
 const idBytes=new Uint8Array(new BigUint64Array(64).buffer),params=new Float32Array(64*16);for(let k=0;k<64;k++)params[k*16+5]=2;
 f.base={manifest:{dt_ms:.5,neuron_count:64,files:{'ids.bin':{sha256:digest(idBytes),bytes:idBytes.length},'io.json':{sha256:'b'.repeat(64)}}},params,io:{sensory:[],motor_neurons:Array.from({length:48},(_,index)=>({index})),muscles:[{kind:'asynchronous_wing',indices:Array.from({length:48},(_,i)=>i)}]}};
 const extra=['/motor-decoder.js','/training/maintained-flight-objective.js','/training/airborne-reset-contract.js','/training/airborne-reset.js','/training/recovery-objective.js','/body-model/flybody-mujoco.json'];
 const catalog={fixture:true};f.config={schemaVersion:1,parameterContract:'banc-sensorimotor-sequence-v1',parameters:[{initial:0},{initial:.5}],modelFingerprint:'c'.repeat(64),
  assets:Object.fromEntries([['/fixture.bin',hash],...extra.map(url=>[url,hash]),['/body-model/banc-leg-proprioception-v1.json',digest(JSON.stringify(catalog))]]),
  stage:'maintained_flight',stages:[{id:'maintained_flight',durationSeconds:5},{id:'recovery',durationSeconds:5},{id:'takeoff',durationSeconds:3},{id:'landing',durationSeconds:8}],
  dtMs:.5,bodyBlockMs:2,vision:false,optimizer:{seed:1},environmentVersion:'sequential-fixture',wingEventExcitation:{fixture:true},
  legProprioception:{marker:1},haltereFeedback:{profile:{fixture:true},maxCurrentPa:10},maintainedScene:SPACIOUS_MAINTAINED_SCENE,
  initialCondition:{warmupSeconds:.5},recoveryDisturbance:DEFAULT_RECOVERY_DISTURBANCE};
 const values={'/training/config.json':f.config,'/habitat.json':{flies:[{x:0,z:0,heading:0}],fruit:[]},'/banc-data/console/groups.json':{sweet:[]},'/banc-data/console/sensory-inputs.json':{},'/body-model/banc-leg-proprioception-v1.json':catalog,
  '/body-model/banc-taste-peg-annotations.json':{annotations:[],prepared_ids_sha256:digest(idBytes),prepared_io_sha256:'b'.repeat(64)}};
 globalThis.fetch=async url=>{if(url==='/banc-data/ids.bin')return new Response(idBytes);if(Object.hasOwn(values,url))return new Response(JSON.stringify(values[url]));if(url==='/fixture.bin'||extra.includes(url))return new Response(asset);throw new Error('Unexpected fetch '+url);};
 f.brain=()=>{const b={timeMs:0,inputs:[],disposed:false,backend:'fixture',async stepSequence(inputs){this.inputs.push(inputs.map(row=>Array.from(row)));this.timeMs+=2;f.order.push('neural:'+this.timeMs);},async readState(indices,options){const stride=options?.includeSpikeTime?9:8,r=new Float32Array(indices.length*stride);if(stride===9)for(let k=0;k<indices.length;k++)r[k*stride+8]=-1e30;f.order.push('read:'+this.timeMs);return r;},dispose(){this.disposed=true;}};f.brains.push(b);return b;};
 f.world=(_fruit,[fly],options)=>{
  const data={qpos:Float64Array.from([0,0,3,1,0,0,0]),qvel:new Float64Array(6),qfrc_applied:new Float64Array(6),xfrc_applied:new Float64Array(6),geom_xpos:[],geom_xmat:[],xpos:[],xanchor:[]};
  const body={data,time:0,x:0,y:0,z:3,quaternion:[1,0,0,0],feet:[],internal:{},remainder:0,environmentContactCount:0,foodContactCount:0,legFoodContact:[],mouthFoodContact:[],wingFoodContact:[],
   wings:{phase:0,frequencyHz:235.8,setInterpreterParameters(){}},monitor:{resetContinuity(){}},refresh(){},
   enableWingMotorEvents(){this._wingMotorEvents={initialized:true,observedMs:0,elapsedMs:0,adapter:{snapshot:()=>({pending:null})}};},
   acceptWingMotorEvents(p){this._wingMotorEvents.observedMs=p.timeMs;f.order.push('events:'+p.timeMs);},enableMotorDecoder(v){this.decoder=[...v];}};
  const world={options:structuredClone(options),bodies:new Map([[1,body]]),mj:{mj_forward(){}},model:{},habitat:{fruit:[],ceiling:500},metadata:{leg_bodies:[]},wingLandmarks:[],mouthLandmarks:{joints:[],geoms:[]},rates:()=>new Map(),copyPose(){fly.bodyTime=body.time;fly.feedback={};},disposed:false,
   tick(dt){assert.equal(body._wingMotorEvents.observedMs,Math.round((body.time+dt)*1000));f.order.push('tick:'+body.time);body.time+=dt;body._wingMotorEvents.elapsedMs=body._wingMotorEvents.observedMs;fly.bodyTime=body.time;},dispose(){this.disposed=true;}};
  f.worlds.push(world);return world;
 };return f;
}
const options={checkpoint:async()=>{},previewHz:0,dutyCycle:1};
const job=(stage='takeoff',parameters=[0,.5])=>({stage,seed:888,durationSeconds:stage==='takeoff'?3:5,parameters});

test('candidate sensory observers and haltere state are reconstructed independently while expensive resources are reused',async()=>{
 const f=fixture(),env=await createTrainingEnvironment();try{
  const first=await env.evaluate(job(),options);f.scoreCalls=0;const second=await env.evaluate(job('takeoff',[Math.log(1.25),.7]),options);
  assert.equal(first.metrics.error,null);assert.equal(second.metrics.error,null);assert.deepEqual(second.parameters,[Math.log(1.25),.7]);
  assert.equal(f.graphLoads,1);assert.equal(f.bodyFactoryBuilds,1);assert.equal(f.brainResourceBuilds,1);assert.equal(f.currentMapperBuilds,1);
  assert.equal(f.sensoryResources.length,3);assert.equal(f.sensoryResources[0].instances.length,0);
  assert.equal(f.sensoryResources[1].config.legProprioception.marker,1);assert.equal(f.sensoryResources[2].config.legProprioception.marker,1.25);
  assert(f.sensoryResources.slice(1).every(r=>r.instances[0].updates===2&&r.instances[0].disposed));
  assert.equal(f.haltereMappers.length,3);assert(f.haltereMappers.slice(1).every(m=>m.writes===8&&m.resets===1));
  assert.equal(f.brains[0].inputs[0][0][1],1);assert.equal(f.brains[1].inputs[0][0][1],1.25);
  assert.equal(f.brains[0].inputs[0][0][2],10);assert.equal(f.brains[1].inputs[0][0][2],12.5);
  assert.deepEqual(f.worlds.map(w=>w.bodies.get(1).decoder),[[.5],[.7]]);assert(f.worlds.every(w=>w.disposed));
  assert.equal(f.config.legProprioception.marker,1);assert.equal(f.config.haltereFeedback.maxCurrentPa,10);
  assert.equal(first.configHash,second.configHash);assert.deepEqual(f.worlds[0].options.maintainedScene,f.worlds[1].options.maintainedScene);
 }finally{env.dispose();}
});

test('candidate identity-layout changes fail before body allocation',async()=>{
 const f=fixture(),env=await createTrainingEnvironment();try{f.badLayout=true;const result=await env.evaluate(job(),options);assert.match(result.metrics.error,/identity layout/);assert.equal(f.worlds.length,0);assert.equal(f.brains.length,0);}finally{env.dispose();}
});

test('failed candidate construction never reports the previous candidate sensor summary',async()=>{
 const f=fixture(),env=await createTrainingEnvironment();try{
  const good=await env.evaluate(job(),options);assert.equal(good.metrics.sensoryFeedback.marker,1);
  f.badLayout=true;const bad=await env.evaluate(job('takeoff',[.1,.7]),options);
  assert.match(bad.metrics.error,/identity layout/);assert.equal(bad.metrics.sensoryFeedback,null);assert.equal(f.worlds.length,1);
 }finally{env.dispose();}
});

test('before-physics instrumentation sees the upcoming consumed motor interval and does not run by default',async()=>{
 const f=fixture(),env=await createTrainingEnvironment();try{
  const contexts=[],result=await env.evaluate(job(),{...options,onBeforePhysicsStep(c){
   contexts.push([c.index,c.episodeIndex,c.phase,c.episodeTimeSeconds,c.timeBefore,c.neuralMs]);
   assert.equal(c.body.time,c.timeBefore);assert.equal(c.body._wingMotorEvents.observedMs,c.neuralMs);assert.equal(c.brain.timeMs,c.neuralMs);
   assert.equal(f.order.at(-1),'events:'+c.neuralMs);f.order.push('before:'+c.timeBefore);
  },onPhysicsStep(c){assert.equal(f.order.at(-1),'tick:'+c.timeBefore);assert(Math.abs(c.body.time-c.timeBefore-.002)<1e-12);}});
  assert.equal(result.metrics.error,null);assert.deepEqual(contexts,[[0,0,'scored',0,0,2],[1,1,'scored',.002,.002,4]]);
  f.scoreCalls=0;await env.evaluate(job(),options);assert.equal(contexts.length,2);
  const count=f.worlds.length;await assert.rejects(env.evaluate(job(),{...options,onBeforePhysicsStep:42}),/must be a function/);assert.equal(f.worlds.length,count);
 }finally{env.dispose();}
});

test('fixed-hash airborne and recovery jobs preserve warmup clocks and apply a seeded release exactly once',async()=>{
 const f=fixture(),env=await createTrainingEnvironment();try{
  const observed=[],result=await env.evaluate(job('recovery'),{...options,onBeforePhysicsStep:c=>observed.push([c.index,c.episodeIndex,c.phase,c.episodeTimeSeconds,c.timeBefore,c.neuralMs]),onInitialState(c){assert.equal(c.phase,'scored-release');assert.deepEqual(Array.from(c.body.data.qvel),createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,888).releaseVelocity);}});
  assert.equal(result.metrics.error,null);assert.equal(result.stage,'recovery');assert.equal(result.warmupSteps,250);assert.equal(result.steps,2);
  assert.deepEqual(observed[0],[0,null,'warmup',0,0,2]);assert.equal(observed[249][2],'warmup');assert.equal(observed[250][0],250);assert.equal(observed[250][1],0);assert.equal(observed[250][2],'scored');assert.equal(observed[250][3],0);assert.equal(observed[250][5],502);
  assert(Math.abs(observed[250][4]-.5)<1e-10);assert.deepEqual(f.releaseVelocity,createRecoveryDisturbance(DEFAULT_RECOVERY_DISTURBANCE,888).releaseVelocity);
  assert.deepEqual(result.initialCondition,undefined);assert.equal(result.metrics.initialCondition.kind,'recovery-scored-release');assert.equal(result.metrics.criteria.stage,'recovery');
  assert.equal(result.provenance.sensoryParameterCount,1);assert.equal(result.provenance.motorParameterCount,1);assert.deepEqual(result.parameters,[0,.5]);
  f.scoreCalls=0;const sameHash=await env.evaluate(job('maintained_flight'),options);assert.equal(sameHash.metrics.error,null);assert.equal(sameHash.configHash,result.configHash);assert.equal(f.releaseVelocity,undefined);
  assert.deepEqual(f.worlds[0].options.maintainedScene,f.worlds[1].options.maintainedScene);
 }finally{env.dispose();}
});

test.after(()=>{globalThis.fetch=savedFetch;delete globalThis[KEY];hooks.deregister();});
