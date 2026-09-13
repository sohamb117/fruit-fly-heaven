import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient} from '../training/client.js';
const base=JSON.parse(await fs.readFile(new URL('../training/config.json',import.meta.url),'utf8'));
const storage=()=>{const data=new Map();return {getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)};};
function setup({delayed=false,mutateResult=x=>x}={}){
 const config=structuredClone(base);config.stages=config.stages.slice(0,1);config.optimizer.populationPairs=1;
 const encoded=JSON.stringify(config),configHash=createHash('sha256').update(encoded).digest('hex'),workers=[];
 class Worker extends EventTarget{
  constructor(){super();this.messages=[];this.dead=false;workers.push(this);}
  send(data){if(!this.dead)this.dispatchEvent(new MessageEvent('message',{data}));}
  postMessage(message){this.messages.push(message);
   if(message.type==='initialize'&&!delayed)queueMicrotask(()=>this.send({type:'ready',id:message.id,configHash,modelFingerprint:config.modelFingerprint,backend:'wasm'}));
   if(message.type==='evaluate')queueMicrotask(()=>{
    // This is an orchestration fixture, never a native simulation result.
    const job=message.job,provenance={environmentVersion:config.environmentVersion,backend:'wasm',bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,configHash,modelFingerprint:config.modelFingerprint,seed:job.seed,stage:job.stage,generation:job.generation,pairId:job.pairId,sign:job.sign};
    this.send({type:'evaluation',id:message.id,result:mutateResult({...provenance,provenance,parameters:job.parameters.slice(),return:job.parameters[0],success:job.parameters[0]>0,simSeconds:1,steps:500,reason:'test fixture'})});
   });
  }
  terminate(){this.dead=true;}
 }
 return {client:new TrainingClient({workerFactory:()=>new Worker(),storage:storage(),fetcher:async()=>new Response(encoded)}),workers,config,configHash};
}
test('opening training settings starts neither compute nor a worker',async()=>{
 const {client,workers}=setup();await client.initialize();assert.equal(client.state.phase,'ready');assert.equal(workers.length,0);assert.equal(client.running,false);await client.dispose();
});
test('wrong worker parameters, seed, or clock are not counted or scored',async()=>{
 for(const mutateResult of [r=>({...r,seed:r.seed+1}),r=>({...r,parameters:r.parameters.map((v,i)=>i===0?v+.01:v)}),r=>({...r,steps:r.steps+1}),r=>({...r,provenance:undefined})]){
  const {client}=setup({mutateResult});await client.initialize();await client.start();await client.loop;
  assert.equal(client.state.phase,'error');assert.equal(client.state.completedEpisodes,0);assert.equal(client.state.contributedEpisodes,0);await client.dispose();
 }
});
test('a lease issued after Stop is released at its original coordinator',async()=>{
 const {client,config,configHash}=setup();await client.initialize();const url='http://127.0.0.1:7850',calls=[];let resolveLease;
 const lease=new Promise(resolve=>{resolveLease=resolve;});
 client.fetcher=async(endpoint,options)=>{calls.push({endpoint,body:options?.body});if(endpoint.endsWith('/lease'))return lease;return Response.json({released:true});};
 client.state.coordinator={connected:true,url};client.running=true;
 const loop=client.runShared(client.runToken);await new Promise(r=>setTimeout(r,0));await client.stop();
 client.state.coordinator.url='http://127.0.0.1:9999';
 resolveLease(Response.json({job:{jobId:'late-job',leaseToken:'token',parameters:client.parameters.slice(),stage:'posture',seed:888,durationSeconds:1,modelFingerprint:config.modelFingerprint,configHash}}));
 await assert.rejects(loop,{name:'AbortError'});assert.equal(calls.length,2);assert.equal(calls[1].endpoint,url+'/api/training/release');
 assert.equal(client.state.contributedEpisodes,0);await client.dispose();
});
test('coordinator cannot be changed during running compute',async()=>{
 const {client}=setup();await client.initialize();client.running=true;
 await assert.rejects(client.connectCoordinator('http://127.0.0.1:9999'),/Stop training/);await client.dispose();
});
test('an HTTP 200 with accepted=false does not count a recovered contribution',async()=>{
 const {client,config,configHash}=setup();await client.initialize();const url='http://127.0.0.1:7850';
 client.fetcher=async()=>Response.json({accepted:false});client.state.coordinator={connected:true,url};client.running=true;
 client.outbox={modelFingerprint:config.modelFingerprint,configHash,objective:0};client.outboxUrl=url;
 await assert.rejects(client.runShared(client.runToken),/did not accept/);assert.equal(client.state.contributedEpisodes,0);assert.ok(client.outbox);await client.dispose();
});
test('local reward search updates a checkpoint after complete pairs and separate validation',async()=>{
 const {client,workers}=setup();await client.initialize();await client.start();await client.loop;
 assert.equal(client.state.generation,1);assert.ok(client.parameters[0]>0);assert.equal(client.state.completedEpisodes,6);
 assert.equal(client.state.checkpointStatus,'locally-validated');assert.equal(client.state.contributedEpisodes,0);
 assert.equal(workers[0].messages.filter(x=>x.type==='evaluate').length,6);await client.dispose();
});
test('stop during initialization releases the worker and cannot restart a stale loop',async()=>{
 const {client,workers}=setup({delayed:true});await client.initialize();const start=client.start();await new Promise(r=>setTimeout(r,0));
 assert.equal(workers.length,1);await client.stop();await start;assert.equal(client.state.phase,'stopped');assert.equal(client.running,false);
 assert.equal(workers[0].dead,true);assert.equal(workers[0].messages.filter(x=>x.type==='evaluate').length,0);
});
test('imported checkpoints do not preserve a claimed validation result',async()=>{
 const {client}=setup();await client.initialize();const c=client.exportCheckpoint();c.status='locally-tested';c.validation={passed:true};
 client.importCheckpoint(c);assert.equal(client.state.checkpointStatus,'unverified');assert.equal(client.savedValidation,null);
 assert.throws(()=>client.importCheckpoint({...c,configHash:'other'}));await client.dispose();
});
test('coordinator connection is checked before any lease and refuses mismatched builds',async()=>{
 const {client}=setup();await client.initialize();const calls=[];client.fetcher=async url=>{calls.push(url);return Response.json({modelFingerprint:'other',configHash:'other'});};
 await assert.rejects(client.connectCoordinator('http://127.0.0.1:7850'),/different build/);assert.equal(calls.length,1);assert.ok(calls[0].endsWith('/status'));assert.equal(client.state.coordinator.connected,false);
 assert.throws(()=>client.coordinatorUrl('http://public.example'),/HTTPS/);await client.dispose();
});
