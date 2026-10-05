// Orchestration fixtures only: no brain simulation, real leases, or training.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient,EVALUATION_STALL_MS} from '../training/client.js';
import {checkpoint} from '../training/optimizer.js';
import {wasmConfigFixture} from './fixtures/training-native-config.mjs';

const base=JSON.parse(await readFile(new URL('../training/config.json',import.meta.url)));
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:value=>resolve(value)};};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function until(predicate){
  for(let i=0;i<100&&!predicate();i++)await new Promise(resolve=>setTimeout(resolve,5));
  assert.ok(predicate(),'fixture did not reach the expected orchestration state');
}
const expiredResponse=(code='assignment_timeout',status=410)=>Response.json({error:code,message:'This assignment has expired'}, {status});
async function setup(t,{start=true,savedOutbox=false}={}){
  let now=0,heartbeatResponse=()=>Response.json({renewed:true}),resultResponse=()=>Response.json({accepted:true});
  t.mock.method(performance,'now',()=>now);
  const timers=new Map();let nextTimer=0;
  t.mock.method(globalThis,'setInterval',(callback,ms)=>{const id=++nextTimer;timers.set(id,{callback,ms});return id;});
  t.mock.method(globalThis,'clearInterval',id=>timers.delete(id));
  const config=wasmConfigFixture(base),text=JSON.stringify(config),configHash=createHash('sha256').update(text).digest('hex');
  const execution={backend:'wasm',neuralEngine:'wasm',wasmExecution:config.optimizer.acceptance.nativeExecution};
  const modelFingerprint=config.modelFingerprint,current=checkpoint(config,configHash,config.parameters.map(p=>p.initial),1,config.stage);
  const job={jobId:'fixture-job',leaseToken:'fixture-lease',generation:1,pairId:'g1-p0',sign:1,seed:888,
    configHash,modelFingerprint,parameters:current.parameters,parametersHash:'a'.repeat(64),stage:config.stage,durationSeconds:config.durationSeconds};
  const workers=[],calls=[],requests=[],started=deferred();let leaseCount=0;
  const saved={checkpoint:current,outbox:savedOutbox?{jobId:job.jobId,leaseToken:job.leaseToken,configHash,modelFingerprint,objective:-1}:null,outboxUrl:'https://fixture.example'};
  let stored=savedOutbox?JSON.stringify(saved):null;
  class FixtureWorker extends EventTarget{
    send(data){this.dispatchEvent(new MessageEvent('message',{data}));}
    postMessage(message){
      if(message.type==='initialize')queueMicrotask(()=>this.send({type:'ready',id:message.id,...execution,configHash,modelFingerprint}));
      if(message.type==='evaluate'){this.evaluation=message;started.resolve();}
    }
    progress(values={}){this.send({type:'progress',id:this.evaluation.id,...values});}
    complete(){
      const j=this.evaluation.job,provenance={...execution,configHash,modelFingerprint,environmentVersion:config.environmentVersion,
        seed:j.seed,stage:j.stage,dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,bodyBackend:'mujoco-wasm',
        generation:j.generation,pairId:j.pairId,sign:j.sign,parametersHash:j.parametersHash};
      this.send({type:'evaluation',id:this.evaluation.id,result:{...provenance,provenance,parameters:j.parameters,return:-1,
        success:false,terminated:true,cancelled:false,steps:1,simSeconds:config.bodyBlockMs/1000,reason:'excessive_rotation',metrics:{wallSeconds:1}}});
    }
    terminate(){this.terminated=true;}
  }
  const client=new TrainingClient({sharedOnly:true,coordinatorUrl:'https://fixture.example',storage:{getItem:()=>stored,setItem:(_key,value)=>{stored=value;}},
    workerFactory:()=>{const worker=new FixtureWorker();workers.push(worker);return worker;},fetcher:async(url,options)=>{
      const pathname=new URL(url).pathname;calls.push(pathname);requests.push({path:pathname,body:options?.body?JSON.parse(options.body):null});
      if(pathname.endsWith('/config.json'))return new Response(text);
      if(pathname.endsWith('/status'))return Response.json({configHash,modelFingerprint,generation:1,checkpoint:current});
      if(pathname.endsWith('/lease'))return Response.json({job:{...job,leaseToken:'fixture-lease-'+(++leaseCount)}});
      if(pathname.endsWith('/heartbeat'))return heartbeatResponse();
      if(pathname.endsWith('/result'))return resultResponse();
      if(pathname.endsWith('/release'))return Response.json({released:true});
      throw new Error('Unexpected fixture request '+pathname);
    }});
  t.after(async()=>{await client.dispose();await flush();});
  await client.initialize();if(start){await client.start();await started.promise;}
  return {client,workers,calls,requests,stored:()=>stored&&JSON.parse(stored),advance:milliseconds=>{now+=milliseconds;},
    tick:ms=>{for(const timer of [...timers.values()])if(timer.ms===ms)timer.callback();},
    heartbeat:value=>heartbeatResponse=value,result:value=>resultResponse=value,
    count:name=>calls.filter(path=>path.endsWith('/'+name)).length};
}

test('a silent worker is terminated after 180 active seconds, releases its job and never uploads',async t=>{
  const f=await setup(t);f.advance(EVALUATION_STALL_MS-1);f.tick(1000);assert.equal(f.client.running,true);
  f.advance(1);f.tick(1000);await f.client.loop;await flush();
  assert.equal(f.workers[0].terminated,true);assert.equal(f.client.state.phase,'error');assert.match(f.client.state.error,/no progress for 3 minutes/);
  assert.equal(f.client.evaluationWatch,null);assert.equal(f.count('release'),1);assert.equal(f.count('result'),0);
  assert.equal(f.client.state.completedEpisodes,0);assert.equal(f.client.outbox,undefined);
});

test('repeated clocks, preview frames, brain activity and wrong RPC IDs cannot keep frozen work leased',async t=>{
  const f=await setup(t),worker=f.workers[0];worker.progress({neuralMs:2,nativeTimeSeconds:.002,steps:0,warmupSteps:1});
  f.advance(EVALUATION_STALL_MS);
  worker.progress({neuralMs:2,nativeTimeSeconds:.002,steps:0,warmupSteps:1,message:'Still running'});
  worker.send({type:'progress',id:worker.evaluation.id-1,neuralMs:999});
  worker.send({type:'frame',id:worker.evaluation.id,frame:{neuralMs:999}});
  worker.send({type:'brain',snapshot:{jobId:'fixture-job',neuralTimeMs:999}});
  f.tick(60000);await f.client.loop;
  assert.equal(f.count('heartbeat'),0);assert.equal(f.count('release'),1);assert.equal(worker.terminated,true);
});

test('warm-up clock advancement keeps work live even with zero scored steps and while hidden',async t=>{
  const f=await setup(t),worker=f.workers[0];
  const previous=Object.getOwnPropertyDescriptor(globalThis,'document');Object.defineProperty(globalThis,'document',{value:{hidden:true},configurable:true});
  t.after(()=>{if(previous)Object.defineProperty(globalThis,'document',previous);else delete globalThis.document;});
  for(let i=1;i<=4;i++){
    f.advance(EVALUATION_STALL_MS-1);worker.progress({phase:'warmup',simSeconds:0,steps:0,warmupSteps:i,neuralMs:i*2,nativeTimeSeconds:i*.002});
    f.tick(1000);f.tick(60000);await flush();assert.equal(f.client.running,true);
  }
  assert.equal(f.count('heartbeat'),4);assert.equal(f.client.paused,false);assert.equal(worker.terminated,undefined);
});

test('manual Pause excludes time and Resume automatically replaces an expired assignment',async t=>{
  const f=await setup(t);f.client.pause();f.advance(EVALUATION_STALL_MS*3);f.tick(1000);f.tick(60000);
  assert.equal(f.client.running,true);assert.equal(f.count('heartbeat'),0);
  f.heartbeat(()=>expiredResponse('lease_expired'));f.client.resume();assert.equal(f.count('heartbeat'),1);
  await until(()=>f.workers[1]?.evaluation);
  assert.equal(f.client.state.phase,'training');assert.equal(f.client.state.error,null);assert.equal(f.client.running,true);
  assert.equal(f.workers[0].terminated,true);assert.equal(f.count('result'),0);assert.equal(f.count('release'),0);
  assert.equal(f.client.activeLease.leaseToken,'fixture-lease-2');
});

test('Resume grants a fresh progress window without treating repeated clocks as new progress',async t=>{
  const f=await setup(t),worker=f.workers[0];worker.progress({neuralMs:4});f.advance(EVALUATION_STALL_MS-1);f.client.pause();
  f.advance(EVALUATION_STALL_MS*2);f.client.resume();await flush();f.tick(1000);assert.equal(f.client.running,true);
  f.advance(EVALUATION_STALL_MS-1);worker.progress({neuralMs:4});f.tick(1000);assert.equal(f.client.running,true);
  f.advance(1);f.tick(1000);await f.client.loop;assert.equal(worker.terminated,true);
});

test('late obsolete heartbeat and worker messages cannot cancel or update a replacement evaluation',async t=>{
  const f=await setup(t),oldWorker=f.workers[0],held=deferred();f.heartbeat(()=>held.promise);
  const renewal=f.client.renewLease(f.client.activeLease,f.client.runToken);
  await f.client.stop();await f.client.loop;await f.client.start();await flush();
  assert.equal(f.workers.length,2);const current=f.client.evaluationWatch,episodeTime=f.client.state.episodeSimSeconds;
  held.resolve(expiredResponse('stale_lease',409));await renewal;
  oldWorker.send({type:'progress',id:oldWorker.evaluation.id,simSeconds:99});oldWorker.complete();
  assert.equal(f.client.evaluationWatch,current);assert.equal(f.client.running,true);assert.equal(f.workers[1].terminated,undefined);
  assert.equal(f.client.state.episodeSimSeconds,episodeTime);assert.equal(f.client.state.completedEpisodes,0);
});

test('completed result upload remains retryable beyond the watchdog and a late expired heartbeat',async t=>{
  const f=await setup(t),renewalResponse=deferred(),upload=deferred();f.heartbeat(()=>renewalResponse.promise);f.result(()=>upload.promise);
  const renewal=f.client.renewLease(f.client.activeLease,f.client.runToken);f.workers[0].complete();await flush();
  assert.equal(f.client.state.activity,'uploading');assert.equal(f.client.evaluationWatch,null);assert.ok(f.client.outbox);
  f.advance(EVALUATION_STALL_MS*2);f.tick(1000);
  renewalResponse.resolve(expiredResponse('assignment_timeout'));await renewal;
  assert.equal(f.client.running,true);assert.equal(f.workers[0].terminated,undefined);assert.ok(f.client.outbox);
  await f.client.stop();upload.resolve(Response.json({accepted:true}));await f.client.loop;
  assert.ok(f.client.outbox);assert.equal(f.count('release'),0);assert.equal(f.client.state.contributedEpisodes,0);
});

for(const [code,status] of [['assignment_timeout',410],['lease_expired',410],['stale_lease',409]]){
  test(`${code} heartbeat terminates obsolete compute and starts a fresh assignment without Start`,async t=>{
    const f=await setup(t),oldWorker=f.workers[0],oldId=oldWorker.evaluation.id,token=f.client.runToken;
    f.heartbeat(()=>expiredResponse(code,status));await f.client.renewLease(f.client.activeLease,token);
    await until(()=>f.workers[1]?.evaluation);
    const current=f.client.evaluationWatch;
    oldWorker.progress({neuralMs:9999});oldWorker.complete();await flush();
    assert.equal(oldWorker.terminated,true);assert.equal(f.client.running,true);assert.equal(f.client.runToken,token);
    assert.equal(f.client.evaluationWatch,current);assert.notEqual(current.id,oldId);
    assert.equal(current.job.leaseToken,'fixture-lease-2');assert.equal(f.count('lease'),2);
    assert.equal(f.count('result'),0);assert.equal(f.count('release'),0);
    assert.equal(f.client.state.completedEpisodes,0);assert.equal(f.client.state.contributedEpisodes,0);
    assert.equal(f.client.state.phase,'training');assert.equal(f.client.state.error,null);
  });

  test(`${code} result rejection discards only the obsolete outbox and continues on the same worker`,async t=>{
    const f=await setup(t),worker=f.workers[0],oldId=worker.evaluation.id;
    f.result(()=>expiredResponse(code,status));worker.complete();
    await until(()=>worker.evaluation.id!==oldId);
    assert.equal(f.client.running,true);assert.equal(f.client.state.error,null);
    assert.equal(f.workers.length,1);assert.equal(worker.terminated,undefined);
    assert.equal(f.client.outbox,null);assert.equal(f.stored().outbox,null);
    assert.equal(f.count('result'),1);assert.equal(f.count('lease'),2);assert.equal(f.count('release'),0);
    assert.equal(f.client.state.completedEpisodes,1);assert.equal(f.client.state.contributedEpisodes,0);
    assert.equal(f.requests.find(r=>r.path.endsWith('/result')).body.leaseToken,'fixture-lease-1');
    assert.equal(f.client.activeLease.leaseToken,'fixture-lease-2');
  });
}

test('an expiry reply during manual Pause stops obsolete compute but leases nothing until Resume',async t=>{
  const f=await setup(t),held=deferred();f.heartbeat(()=>held.promise);
  const renewal=f.client.renewLease(f.client.activeLease,f.client.runToken);f.client.pause();
  held.resolve(expiredResponse());await renewal;await flush();
  assert.equal(f.workers[0].terminated,true);assert.equal(f.client.worker,null);
  assert.equal(f.client.running,true);assert.equal(f.client.paused,true);assert.equal(f.client.state.phase,'paused');
  assert.equal(f.count('lease'),1);assert.equal(f.count('result'),0);
  f.client.resume();await until(()=>f.workers[1]?.evaluation);
  assert.equal(f.count('lease'),2);assert.equal(f.client.running,true);assert.equal(f.client.state.phase,'training');
});

test('manual Stop while heartbeat is pending never automatically acquires another assignment',async t=>{
  const f=await setup(t),held=deferred();f.heartbeat(()=>held.promise);
  const renewal=f.client.renewLease(f.client.activeLease,f.client.runToken);
  await f.client.stop();held.resolve(expiredResponse());await renewal;await f.client.loop;
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'stopped');assert.equal(f.client.worker,null);
  assert.equal(f.workers.length,1);assert.equal(f.count('lease'),1);assert.equal(f.count('result'),0);
});

test('expiry of a completed evaluation waiting on Pause prevents its eventual upload',async t=>{
  const f=await setup(t),held=deferred();f.heartbeat(()=>held.promise);
  const renewal=f.client.renewLease(f.client.activeLease,f.client.runToken);
  f.client.pause();f.workers[0].complete();await flush();assert.equal(f.client.evaluationWatch,null);
  held.resolve(expiredResponse());await renewal;
  assert.equal(f.count('result'),0);assert.equal(f.count('lease'),1);assert.equal(f.client.state.phase,'paused');
  f.client.resume();await until(()=>f.count('lease')===2&&f.client.evaluationWatch);
  assert.equal(f.count('result'),0);assert.equal(f.client.state.completedEpisodes,0);
  assert.equal(f.workers.length,1);assert.equal(f.client.activeLease.leaseToken,'fixture-lease-2');
});

test('result expiry received during Pause clears its outbox but waits for Resume to acquire work',async t=>{
  const f=await setup(t),held=deferred();f.result(()=>held.promise);f.workers[0].complete();
  await until(()=>f.client.outbox);f.client.pause();held.resolve(expiredResponse('stale_lease',409));
  await until(()=>f.client.outbox===null);
  assert.equal(f.stored().outbox,null);assert.equal(f.count('lease'),1);assert.equal(f.client.state.phase,'paused');
  f.client.resume();await until(()=>f.count('lease')===2&&f.client.evaluationWatch);
  assert.equal(f.count('result'),1);assert.equal(f.client.state.contributedEpisodes,0);
});

test('late expiry of an upload after Stop clears the rejected outbox without restarting work',async t=>{
  const f=await setup(t),held=deferred();f.result(()=>held.promise);f.workers[0].complete();
  await until(()=>f.client.outbox);await f.client.stop();held.resolve(expiredResponse());await f.client.loop;
  assert.equal(f.client.outbox,null);assert.equal(f.stored().outbox,null);
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'stopped');
  assert.equal(f.count('lease'),1);assert.equal(f.count('result'),1);assert.equal(f.count('release'),0);
});

test('a restored obsolete outbox is attempted once, removed from storage, then replaced by assigned work',async t=>{
  const f=await setup(t,{start:false,savedOutbox:true});assert.ok(f.client.outbox);
  f.result(()=>Response.json({error:'This assignment has expired',code:'assignment_timeout'},{status:409}));
  await f.client.start();await until(()=>f.client.evaluationWatch);
  assert.equal(f.count('result'),1);assert.equal(f.count('lease'),1);
  assert.ok(f.calls.findIndex(p=>p.endsWith('/result'))<f.calls.findIndex(p=>p.endsWith('/lease')));
  assert.equal(f.client.outbox,null);assert.equal(f.stored().outbox,null);assert.equal(f.client.state.contributedEpisodes,0);
  assert.equal(f.client.running,true);assert.equal(f.client.state.error,null);
});

test('heartbeat expiry before an evaluation RPC skips that assignment without starting compute',async t=>{
  const f=await setup(t,{start:false}),evaluate=f.client.evaluate.bind(f.client);let first=true;
  f.heartbeat(()=>expiredResponse('lease_expired'));
  f.client.evaluate=async(job,role,token)=>{
    if(first){first=false;await f.client.renewLease(job,token);}
    return evaluate(job,role,token);
  };
  await f.client.start();await until(()=>f.client.evaluationWatch);
  assert.equal(f.count('lease'),2);assert.equal(f.count('heartbeat'),1);assert.equal(f.count('result'),0);
  assert.equal(f.workers.length,1);assert.equal(f.workers[0].evaluation.job.leaseToken,'fixture-lease-2');
  assert.equal(f.client.evaluationInstance,1);assert.equal(f.client.state.completedEpisodes,0);
});

test('a successful in-flight result remains authoritative after a late expired heartbeat',async t=>{
  const f=await setup(t),heldHeartbeat=deferred(),heldResult=deferred(),worker=f.workers[0],oldId=worker.evaluation.id;
  f.heartbeat(()=>heldHeartbeat.promise);f.result(()=>heldResult.promise);
  const renewal=f.client.renewLease(f.client.activeLease,f.client.runToken);
  worker.complete();await until(()=>f.client.outbox);
  heldHeartbeat.resolve(expiredResponse());await renewal;assert.ok(f.client.outbox);
  heldResult.resolve(Response.json({accepted:true}));await until(()=>worker.evaluation.id!==oldId);
  assert.equal(f.client.running,true);assert.equal(f.client.state.contributedEpisodes,1);assert.equal(f.client.outbox,null);
  assert.equal(f.workers.length,1);assert.equal(worker.terminated,undefined);assert.equal(f.count('result'),1);assert.equal(f.count('lease'),2);
});

test('an unrelated heartbeat conflict remains an error instead of an automatic replacement',async t=>{
  const f=await setup(t);f.heartbeat(()=>Response.json({error:'build_mismatch',message:'Coordinator uses a different build'},{status:409}));
  await f.client.renewLease(f.client.activeLease,f.client.runToken);await f.client.loop;
  assert.equal(f.client.running,false);assert.equal(f.client.state.phase,'error');assert.match(f.client.state.error,/different build/);
  assert.equal(f.count('lease'),1);assert.equal(f.count('result'),0);assert.equal(f.count('release'),1);
});

for(const [code,status] of [['build_mismatch',409],['invalid_result',422],['assignment_timeout',404]]){
  test(`unrelated ${status} ${code} result errors preserve the outbox and do not seek more work`,async t=>{
    const f=await setup(t);f.result(()=>Response.json({error:code},{status}));f.workers[0].complete();await f.client.loop;
    assert.equal(f.client.state.phase,'error');assert.equal(f.client.state.error,code);assert.equal(f.client.running,false);
    assert.ok(f.client.outbox);assert.equal(f.stored().outbox.leaseToken,'fixture-lease-1');
    assert.equal(f.count('result'),1);assert.equal(f.count('lease'),1);assert.equal(f.count('release'),0);
    assert.equal(f.client.state.contributedEpisodes,0);
  });
}

test('an unrelated rejection of a restored outbox is visible and retains that upload',async t=>{
  const f=await setup(t,{start:false,savedOutbox:true});f.result(()=>Response.json({error:'invalid_result'},{status:422}));
  await f.client.start();await f.client.loop;
  assert.equal(f.client.state.phase,'error');assert.equal(f.client.state.error,'invalid_result');
  assert.ok(f.client.outbox);assert.ok(f.stored().outbox);assert.equal(f.count('lease'),0);assert.equal(f.count('result'),1);
});
