// Orchestration regressions only. These fixtures are not neural/physics evidence.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient} from '../training/client.js';
import {checkpoint,makeGeneration} from '../training/optimizer.js';
const original=JSON.parse(await fs.readFile(new URL('../training/config.json',import.meta.url),'utf8'));
const deferred=()=>{let resolve;return {promise:new Promise(r=>{resolve=r;}),resolve:value=>resolve(value)};};
const url='http://127.0.0.1:9999';
async function setup(){
 const config=structuredClone(original);config.optimizer.populationPairs=1;
 const encoded=JSON.stringify(config),configHash=createHash('sha256').update(encoded).digest('hex'),data=new Map(),workers=[];
 class Worker extends EventTarget{
  constructor(){super();this.dead=false;workers.push(this);}
  send(message){this.dispatchEvent(new MessageEvent('message',{data:message}));}
  postMessage(message){if(message.type==='initialize')queueMicrotask(()=>this.send({type:'ready',id:message.id,configHash,modelFingerprint:config.modelFingerprint,backend:'wasm'}));
   if(message.type==='evaluate')queueMicrotask(()=>{const job=message.job,provenance={environmentVersion:config.environmentVersion,backend:'wasm',bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,configHash,modelFingerprint:config.modelFingerprint,seed:job.seed,stage:job.stage,generation:job.generation,pairId:job.pairId,sign:job.sign};this.send({type:'evaluation',id:message.id,result:{...provenance,provenance,parameters:[...job.parameters],return:job.parameters[0],success:job.parameters[0]>0,simSeconds:.002,steps:1,reason:'orchestration fixture',metrics:{}}});});
  }
  terminate(){this.dead=true;}
 }
 const c=new TrainingClient({workerFactory:()=>new Worker(),storage:{getItem:k=>data.get(k),setItem:(k,v)=>data.set(k,v)},fetcher:async()=>new Response(encoded)});await c.initialize();
 const shared=checkpoint(config,configHash,config.parameters.map(()=>.1),9,'feeding',{status:'unverified'});
 const status={modelFingerprint:config.modelFingerprint,configHash,generation:9,checkpoint:shared};
 const job={jobId:'shared-fixture',leaseToken:'fixture-token',modelFingerprint:config.modelFingerprint,configHash,parameters:[...shared.parameters],stage:'feeding',seed:1,durationSeconds:3,generation:9,pairId:'shared-pair',sign:1};
 return {c,config,data,workers,shared,status,job};
}
test('shared work preserves the exact local checkpoint and pending optimizer round on return',async()=>{
 const {c,config,data,status,job}=await setup();
 c.state.generation=2;c.round=makeGeneration(c.parameters,2,config,'posture');const round=c.round;
 c.savedValidation={stage:'posture',meanReturn:-1,passed:false};const validation=c.savedValidation;
 const waiting=deferred(),lease=deferred();let leases=0;c.api=async path=>path==='/status'?status:path==='/result'?{accepted:true}:path==='/lease'?(++leases===1?{job}:(waiting.resolve(),lease.promise)):{released:true};
 await c.start({mode:'shared',coordinatorUrl:url});await waiting.promise;
 assert.equal(c.state.stage,'feeding');assert.equal(c.state.generation,9);assert.equal(c.state.parameters[0],.1);
 const saved=JSON.parse(data.get(c.key)).checkpoint;assert.equal(saved.stage,'posture');assert.equal(saved.generation,2);assert.deepEqual(saved.parameters,config.parameters.map(p=>p.initial));
 await c.stop();lease.resolve({job:null});await c.loop;
 let restored;c.runLocal=async()=>{restored={stage:c.state.stage,generation:c.state.generation,parameters:[...c.state.parameters]};};
 await c.start({mode:'local'});await c.loop;assert.deepEqual(restored,{stage:'posture',generation:2,parameters:config.parameters.map(p=>p.initial)});assert.equal(c.round,round);assert.equal(c.savedValidation,validation);await c.dispose();
});
test('a delayed shared status cannot mutate a stopped client or a newer run',async()=>{
 const {c,status,job}=await setup(),waiting=deferred(),response=deferred();let statuses=0;
 c.api=async path=>path==='/status'?(++statuses===1?status:(waiting.resolve(),response.promise)):path==='/lease'?{job}:path==='/result'?{accepted:true}:{released:true};
 await c.start({mode:'shared',coordinatorUrl:url});const oldLoop=c.loop;await waiting.promise;await c.stop();
 c.runLocal=async()=>{};await c.start({mode:'local'});await c.loop;const state={generation:c.state.generation,message:c.state.message,parameters:[...c.state.parameters],stage:c.state.stage};response.resolve({...status,generation:99});await oldLoop;
 assert.deepEqual({generation:c.state.generation,message:c.state.message,parameters:c.state.parameters,stage:c.state.stage},state);assert.equal(c.state.contributedEpisodes,1);await c.dispose();
});
test('late lease release from Stop does not replace a new run status',async()=>{
 const {c}=await setup(),release=deferred();c.activeLease={coordinatorUrl:url};c.releaseLease=()=>release.promise;
 const stopping=c.stop();assert.equal(c.state.phase,'stopped');c.runLocal=async()=>{};await c.start({mode:'local'});await c.loop;const message=c.state.message;release.resolve();await stopping;assert.equal(c.state.phase,'training');assert.equal(c.state.message,message);await c.dispose();
});
test('independent shared checkpoint evaluation exports the tested vector and test evidence as local',async()=>{
 const {c,shared}=await setup();c.options.mode='shared';c.localView=c.localViewSnapshot();c.sharedCheckpoint=shared;c.round={stale:true};c.completedStages.add('posture');
 const validation=await c.evaluateCheckpoint(),exported=c.exportCheckpoint();assert.equal(validation.independent,true);assert.equal(validation.passed,true);assert.equal(exported.status,'locally-tested');assert.deepEqual(exported.parameters,shared.parameters);assert.deepEqual(exported.testValidation,validation);assert.equal(exported.stage,'feeding');assert.equal(exported.generation,9);assert.equal(c.options.mode,'local');assert.equal(c.sharedCheckpoint,null);assert.equal(c.round,null);assert.equal(c.localView,null);assert.equal(c.completedStages.size,0);assert(c.state.curriculum.every(s=>s.status==='available'));await c.dispose();
});
test('import and stage changes clear validation evidence for the previous vector or task',async()=>{
 const {c,shared}=await setup();c.completedStages.add('posture');c.state.curriculum=c.config.stages.map(s=>({...s,status:'validated'}));c.testValidation={stage:'posture',passed:true};c.importCheckpoint(shared);
 assert.equal(c.state.checkpointStatus,'unverified');assert.equal(c.completedStages.size,0);assert(c.state.curriculum.every(s=>s.status==='available'));assert.equal(c.testValidation,null);
 c.savedValidation={stage:'feeding',passed:true};c.testValidation={stage:'feeding',passed:true};c.state.checkpointStatus='locally-tested';c.runLocal=async()=>{};await c.start({mode:'local',stage:'landing'});await c.loop;assert.equal(c.savedValidation,null);assert.equal(c.testValidation,null);assert.equal(c.state.checkpointStatus,'unverified');await c.dispose();
});
test('a newly accepted vector invalidates earlier-stage badges, and final stage alone is not called all stages',async()=>{
 const {c}=await setup();c.state.stage='sequence';c.completedStages.add('posture');c.state.curriculum=c.config.stages.map(s=>({...s,status:s.id==='posture'?'validated':'available'}));c.savedValidation={stage:'sequence',meanReturn:-10,passed:false};
 await c.start({mode:'local',stage:'sequence'});await c.loop;assert(c.parameters[0]>0);assert.equal(c.completedStages.has('posture'),false);assert.equal(c.completedStages.has('sequence'),true);assert.equal(c.state.curriculum.find(s=>s.id==='posture').status,'available');assert.match(c.state.message,/complete-cycle stage passed/);assert.doesNotMatch(c.state.message,/All curriculum stages/);await c.dispose();
});
test('Stop invalidates a pending coordinator connection',async()=>{
 const {c,status}=await setup(),response=deferred(),waiting=deferred();c.api=async()=>{waiting.resolve();return response.promise;};const connecting=c.connectCoordinator(url);await waiting.promise;await c.stop();response.resolve(status);await assert.rejects(connecting,{name:'AbortError'});assert.equal(c.state.coordinator.connected,false);await c.dispose();
});
test('finishing an old shared request does not clear the new lease heartbeat',async()=>{
 const {c,status,job}=await setup(),oldStatus=deferred(),oldWaiting=deferred(),newResult=deferred(),newWaiting=deferred();let statuses=0,results=0,leases=0;
 c.api=async path=>{
  if(path==='/status')return ++statuses===2?(oldWaiting.resolve(),oldStatus.promise):status;
  if(path==='/lease')return {job:{...job,jobId:'lease-'+(++leases)}};
  if(path==='/result')return ++results===2?(newWaiting.resolve(),newResult.promise):{accepted:true};
  return {released:true};
 };
 await c.start({mode:'shared',coordinatorUrl:url});const oldLoop=c.loop;await oldWaiting.promise;await c.stop();
 await c.start({mode:'shared',coordinatorUrl:url});const newLoop=c.loop;await newWaiting.promise;const heartbeat=c.heartbeat;assert.ok(heartbeat);
 oldStatus.resolve(status);await oldLoop;assert.equal(c.heartbeat,heartbeat);
 await c.stop();newResult.resolve({accepted:true});await newLoop;await c.dispose();
});
test('an error event from a retired worker cannot reject the new worker request',async()=>{
 const {c,workers,job}=await setup();await c.ensureWorker();const old=workers[0];await c.stop();await c.ensureWorker();
 const pending=c.rpc('evaluate',{job});const error=new Event('error');Object.defineProperty(error,'message',{value:'retired worker error'});old.dispatchEvent(error);
 const reply=await pending;assert.equal(reply.type,'evaluation');assert.notEqual(c.state.phase,'error');assert.equal(c.worker,workers[1]);await c.dispose();
});
