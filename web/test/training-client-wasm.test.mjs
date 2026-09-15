// Worker and HTTP orchestration fixtures, never actual training evidence.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient,browserTrainingAvailable} from '../training/client.js';
import {checkpoint} from '../training/optimizer.js';
import {nativeConfigFixture,wasmConfigFixture} from './fixtures/training-native-config.mjs';
const base=JSON.parse(await readFile(new URL('../training/config.json',import.meta.url)));
const origin='https://wasm.example';
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:value=>resolve(value)};};
function setup({mutateReady=value=>value,mutateResult=value=>value,automatic=true}={}){
  const config=wasmConfigFixture(base),bytes=JSON.stringify(config),configHash=createHash('sha256').update(bytes).digest('hex');
  const calls=[],workers=[],evaluating=deferred(),submitted=deferred(),data=new Map();let offline=false,accepted=0;
  const execution={backend:'wasm',neuralEngine:'wasm',wasmExecution:{...config.optimizer.acceptance.nativeExecution}};
  const current=()=>checkpoint(config,configHash,config.parameters.map(p=>p.initial),1,config.stage);
  const job={jobId:'wasm-fixture-job',leaseToken:'wasm-fixture-token',pairId:'g1-p0',sign:1,generation:1,seed:888,
    modelFingerprint:config.modelFingerprint,configHash,parameters:current().parameters,parametersHash:'a'.repeat(64),stage:config.stage,durationSeconds:config.durationSeconds};
  const status=()=>({modelFingerprint:config.modelFingerprint,configHash,generation:1,stage:config.stage,acceptedResults:accepted,checkpoint:current()});
  class FixtureWorker extends EventTarget{
    constructor(){super();this.messages=[];workers.push(this);}
    send(message){queueMicrotask(()=>{if(!this.terminated)this.dispatchEvent(new MessageEvent('message',{data:message}));});}
    complete(){
      const {id,job:j}=this.pendingJob,provenance={...execution,environmentVersion:config.environmentVersion,configHash,modelFingerprint:config.modelFingerprint,
        bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,stage:j.stage,seed:j.seed,generation:j.generation,pairId:j.pairId,sign:j.sign,parametersHash:j.parametersHash};
      this.send({type:'evaluation',id,result:mutateResult({...provenance,provenance,parameters:[...j.parameters],return:-1,
        success:false,terminated:true,cancelled:false,steps:1,simSeconds:config.bodyBlockMs/1000,reason:'excessive_rotation',metrics:{wallSeconds:1}})});
    }
    postMessage(message){
      this.messages.push(message);
      if(message.type==='initialize')this.send({type:'ready',id:message.id,...mutateReady({...execution,configHash,modelFingerprint:config.modelFingerprint})});
      if(message.type==='evaluate'){this.pendingJob=message;evaluating.resolve();if(automatic)this.complete();}
    }
    terminate(){this.terminated=true;}
  }
  const client=new TrainingClient({sharedOnly:true,coordinatorUrl:origin,workerFactory:()=>new FixtureWorker(),
    storage:{getItem:key=>data.get(key),setItem:(key,value)=>data.set(key,value)},fetcher:async(url,options={})=>{
      const path=new URL(url).pathname;calls.push({path,options});
      if(path.endsWith('/config.json'))return new Response(bytes);
      if(offline)throw new Error('Coordinator offline');
      if(path.endsWith('/status'))return Response.json(status());
      if(path.endsWith('/lease'))return Response.json({job});
      if(path.endsWith('/heartbeat'))return Response.json({renewed:true});
      if(path.endsWith('/result')){accepted++;submitted.resolve(JSON.parse(options.body));return Response.json({accepted:true});}
      if(path.endsWith('/release'))return Response.json({released:true});
      throw new Error('Unexpected fixture request '+path);
    }});
  return {client,workers,calls,config,job,evaluating,submitted,setOffline:value=>offline=value};
}

function holdApiRequest(t,client,path){
  const requested=deferred(),fetcher=client.fetcher,setTimeout=globalThis.setTimeout;
  let expire;
  t.mock.method(globalThis,'setTimeout',(callback,milliseconds,...args)=>{
    if(milliseconds===15000){expire=callback;return {};}
    return setTimeout(callback,milliseconds,...args);
  });
  client.fetcher=(url,options)=>{
    if(new URL(url).pathname!==path)return fetcher(url,options);
    return new Promise((resolve,reject)=>{
      options.signal.addEventListener('abort',()=>reject(new DOMException('Timed out fixture request','AbortError')),{once:true});
      requested.resolve();
    });
  };
  return {requested:requested.promise,expire:()=>expire()};
}

test('browser eligibility follows the explicit execution class without opening Dawn runs',()=>{
  assert.equal(browserTrainingAvailable(wasmConfigFixture(base)),true);
  assert.equal(browserTrainingAvailable(nativeConfigFixture(base)),false);
  assert.equal(browserTrainingAvailable({...base,schemaVersion:1}),true);
  assert.equal(browserTrainingAvailable({schemaVersion:2,optimizer:{acceptance:{nativeExecution:{backend:'unknown'}}}}),false);
});

test('a guarded WASM page is idle until Start and uploads the assigned job with exact execution and outcome fields',async()=>{
  const f=setup();await f.client.initialize();await f.client.connectCoordinator();assert.equal(f.workers.length,0);
  assert.equal(f.client.state.browserTrainingAvailable,true);assert.equal(f.calls.some(c=>c.path.endsWith('/lease')),false);
  const counted=deferred();f.client.addEventListener('state',event=>{if(event.detail.contributedEpisodes===1){f.client.pause();counted.resolve();}});
  await f.client.start();const payload=await f.submitted.promise;await counted.promise;await f.client.stop();await f.client.loop;
  assert.equal(f.workers.length,1);assert.equal(f.client.state.completedEpisodes,1);assert.equal(f.client.state.contributedEpisodes,1);
  assert.equal(payload.provenance.backend,'wasm');assert.equal(payload.provenance.neuralEngine,'wasm');
  assert.deepEqual(payload.provenance.wasmExecution,f.config.optimizer.acceptance.nativeExecution);
  assert.equal(Object.hasOwn(payload.provenance,'nativeWebGPU'),false);assert.deepEqual(payload.provenance.parameters,f.job.parameters);
  assert.equal(payload.provenance.parametersHash,f.job.parametersHash);
  assert.equal(payload.metrics.terminated,true);assert.equal(payload.metrics.cancelled,false);assert.equal(payload.metrics.reason,'excessive_rotation');
  assert.equal(f.workers[0].terminated,true);
});

test('wrong ready backend or binary pin is refused before any lease is acquired',async()=>{
  for(const mutateReady of [x=>({...x,backend:'webgpu'}),x=>({...x,wasmExecution:{...x.wasmExecution,moduleSha256:'d'.repeat(64)}}),x=>({...x,nativeWebGPU:{}})]){
    const f=setup({mutateReady});await f.client.initialize();await assert.rejects(f.client.start(),/execution does not match/);
    assert.equal(f.calls.some(c=>c.path.endsWith('/lease')),false);assert.equal(f.workers[0].terminated,true);await f.client.dispose();
  }
});

test('wrong result pins and partial/cancelled trials are never counted or uploaded',async()=>{
  for(const mutateResult of [r=>({...r,provenance:{...r.provenance,wasmExecution:{...r.provenance.wasmExecution,moduleSha256:'d'.repeat(64)}}}),
    r=>({...r,terminated:false,reason:'time_limit'}),r=>({...r,cancelled:true}),r=>({...r,provenance:{...r.provenance,parametersHash:'wrong'}})]){
    const f=setup({mutateResult});await f.client.initialize();await f.client.start();await f.client.loop;
    assert.equal(f.client.state.completedEpisodes,0);assert.equal(f.client.state.contributedEpisodes,0);
    assert.equal(f.calls.some(c=>c.path.endsWith('/result')),false);await f.client.dispose();
  }
});

test('Pause, Resume and intensity reach the running worker; Stop cancels and releases unfinished work',async()=>{
  const f=setup({automatic:false});await f.client.initialize();await f.client.start();await f.evaluating.promise;
  const worker=f.workers[0];f.client.pause();assert.equal(f.client.state.phase,'paused');
  f.client.setBudget({dutyCycle:.25,previewHz:3});f.client.resume();assert.equal(f.client.state.phase,'training');
  assert(worker.messages.some(m=>m.type==='pause'));assert(worker.messages.some(m=>m.type==='resume'));
  assert(worker.messages.some(m=>m.type==='budget'&&m.dutyCycle===.25&&m.previewHz===3));
  await f.client.stop();await f.client.loop;assert(worker.messages.some(m=>m.type==='cancel'));assert.equal(worker.terminated,true);
  assert.equal(f.calls.filter(c=>c.path.endsWith('/release')).length,1);assert.equal(f.calls.some(c=>c.path.endsWith('/result')),false);
});

test('brain observation is opt-in, isolated from training errors and tied to the current assigned vector',async()=>{
  const f=setup({automatic:false}),received=[],errors=[];
  f.client.addEventListener('brain',event=>received.push(event.detail));f.client.addEventListener('brain-error',event=>errors.push(event.detail));
  f.client.setBrainObservation({enabled:true,indices:[3,7]});assert.equal(f.workers.length,0);
  assert.throws(()=>f.client.setBrainObservation({enabled:true,indices:[0x100000000]}),/Invalid/);
  await f.client.initialize();await f.client.start();await f.evaluating.promise;
  const worker=f.workers[0],job=f.client.state.activeJob;assert.equal(Object.hasOwn(job,'leaseToken'),false);assert.deepEqual(job.parameters,f.job.parameters);
  assert(worker.messages.some(message=>message.type==='observe-brain'&&message.enabled&&message.indices.join(',')==='3,7'));
  const snapshot={jobId:job.jobId,neuralTimeMs:10};worker.send({type:'brain',snapshot});worker.send({type:'brain',snapshot:{jobId:'old-job'}});
  worker.send({type:'brain-error',message:'Readout unavailable'});await Promise.resolve();
  assert.deepEqual(received,[snapshot]);assert.equal(errors.length,1);assert.equal(f.client.running,true);
  f.client.setBrainObservation({enabled:false});worker.send({type:'brain',snapshot});await Promise.resolve();assert.equal(received.length,1);
  f.client.emit({parameters:f.client.state.parameters.map(()=>0)});assert.deepEqual(f.client.state.activeJob.parameters,f.job.parameters);
  await f.client.stop();await f.client.loop;assert.equal(f.client.state.activeJob,job);
  const restarted=deferred();f.client.addEventListener('state',event=>{if(event.detail.activeJob?.instance>job.instance)restarted.resolve();});
  await f.client.start();await restarted.promise;assert(f.client.state.activeJob.instance>job.instance,'A re-leased job has a fresh evaluation identity');
  await f.client.stop();await f.client.loop;
});

test('hiding the tab keeps the trial, uploads and lease renewals running while manual Pause remains explicit',async t=>{
  const previousDocument=Object.getOwnPropertyDescriptor(globalThis,'document'),document=new EventTarget();document.hidden=false;
  Object.defineProperty(globalThis,'document',{value:document,configurable:true});
  let heartbeatTick;
  t.mock.method(globalThis,'setInterval',(callback,milliseconds)=>{assert([1000,60000].includes(milliseconds));if(milliseconds===60000)heartbeatTick=callback;return 1;});
  t.mock.method(globalThis,'clearInterval',()=>{});
  const f=setup({automatic:false});
  t.after(async()=>{await f.client.dispose();if(previousDocument)Object.defineProperty(globalThis,'document',previousDocument);else delete globalThis.document;});
  await f.client.initialize();await f.client.start();await f.evaluating.promise;
  const worker=f.workers[0],renewals=()=>f.calls.filter(call=>call.path.endsWith('/heartbeat')).length;
  document.hidden=true;document.dispatchEvent(new Event('visibilitychange'));
  assert.equal(f.client.state.phase,'training');assert.equal(f.client.running,true);assert.equal(f.client.paused,false);
  assert.equal(worker.messages.some(message=>message.type==='pause'),false);
  heartbeatTick();assert.equal(renewals(),1);await new Promise(resolve=>setTimeout(resolve,0));
  f.client.pause();assert.equal(f.client.state.phase,'paused');heartbeatTick();assert.equal(renewals(),1);
  document.hidden=false;document.dispatchEvent(new Event('visibilitychange'));
  document.hidden=true;document.dispatchEvent(new Event('visibilitychange'));
  assert.equal(f.client.paused,true,'Visibility changes must not undo a manual pause');
  assert.equal(worker.messages.filter(message=>message.type==='pause').length,1);
  f.client.resume();heartbeatTick();assert.equal(renewals(),2);
  const counted=deferred();f.client.addEventListener('state',event=>{if(event.detail.contributedEpisodes===1){f.client.pause();counted.resolve();}});
  worker.complete();await f.submitted.promise;await counted.promise;
  assert.equal(document.hidden,true);assert.equal(f.client.state.completedEpisodes,1);assert.equal(f.client.state.contributedEpisodes,1);
  await f.client.stop();await f.client.loop;assert.equal(worker.terminated,true);
});

test('guarded WASM still fails closed when the coordinator is unavailable and cannot run a local optimizer',async()=>{
  const f=setup();await f.client.initialize();f.setOffline(true);await assert.rejects(f.client.start(),/offline/);
  assert.equal(f.workers.length,0);await assert.rejects(f.client.start({mode:'local'}),/always contributes/);
  await assert.rejects(f.client.runLocal(f.client.runToken),/always contributes/);await f.client.dispose();
});

test('a lease API timeout becomes an error instead of silently leaving a stopped loop labeled training',async t=>{
  const f=setup({automatic:false});await f.client.initialize();
  const request=holdApiRequest(t,f.client,'/api/training/lease');await f.client.start();await request.requested;
  assert.equal(f.client.state.activity,'waiting');request.expire();await f.client.loop;
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'error');assert.equal(f.client.state.activity,null);
  assert.match(f.client.state.error,/Connection timed out/);assert.equal(f.client.pending.size,0);
  await f.client.dispose();
});

test('an API timeout after intentional Stop cannot replace the stopped state',async t=>{
  const f=setup({automatic:false});await f.client.initialize();
  const request=holdApiRequest(t,f.client,'/api/training/lease');await f.client.start();await request.requested;
  await f.client.stop();request.expire();await f.client.loop;
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'stopped');assert.equal(f.client.state.error,null);
  assert.equal(f.workers[0].terminated,true);
});

test('a timed-out restored upload stays in the outbox and reports the failure',async t=>{
  const f=setup({automatic:false});await f.client.initialize();
  f.client.outbox={...f.client.leaseIdentity(f.job),objective:-1};f.client.outboxUrl=origin;
  const saved=structuredClone(f.client.outbox),request=holdApiRequest(t,f.client,'/api/training/result');
  await f.client.start();await request.requested;assert.equal(f.client.state.activity,'uploading');
  request.expire();await f.client.loop;
  assert.equal(f.client.state.phase,'error');assert.match(f.client.state.error,/Connection timed out/);
  assert.deepEqual(f.client.outbox,saved);assert.equal(f.client.state.contributedEpisodes,0);
  await f.client.dispose();
});

test('unexpected worker cancellation reports an interruption without counting an incomplete trial',async()=>{
  const f=setup({mutateResult:r=>({...r,cancelled:true})});await f.client.initialize();await f.client.start();await f.client.loop;
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'error');assert.match(f.client.state.error,/Training was interrupted/);
  assert.equal(f.client.state.completedEpisodes,0);assert.equal(f.client.state.contributedEpisodes,0);await f.client.dispose();
});
