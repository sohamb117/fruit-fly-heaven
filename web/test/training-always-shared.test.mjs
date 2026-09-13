// Orchestration fixtures only; these are not native simulation results.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient} from '../training/client.js';
import {checkpoint} from '../training/optimizer.js';

const configBytes=await fs.readFile(new URL('../training/config.json',import.meta.url));
const config=JSON.parse(configBytes),configHash=createHash('sha256').update(configBytes).digest('hex');
const origin='https://shared.example';
function setup(){
  const calls=[],workers=[],stored=new Map();let generation=3,unavailable=false,resultFailure=false;
  const current=()=>({...checkpoint(config,configHash,config.parameters.map(()=>.1),generation,'posture',{status:'unverified'}),createdAt:1234,provenance:'Fixture server metadata',completedGenerations:generation});
  const status=()=>({configHash,modelFingerprint:config.modelFingerprint,generation,checkpoint:current()});
  const job={jobId:'fixture-job',leaseToken:'fixture-token',generation:3,pairId:'fixture-pair',sign:1,seed:888,
    stage:'posture',durationSeconds:1,configHash,modelFingerprint:config.modelFingerprint,parameters:current().parameters};
  class FixtureWorker extends EventTarget{
    constructor(){super();workers.push(this);}
    postMessage(message){
      const send=data=>queueMicrotask(()=>this.dispatchEvent(new MessageEvent('message',{data})));
      if(message.type==='initialize')send({type:'ready',id:message.id,configHash,modelFingerprint:config.modelFingerprint,backend:'wasm'});
      if(message.type==='evaluate'){
        const j=message.job,provenance={environmentVersion:config.environmentVersion,configHash,modelFingerprint:config.modelFingerprint,
          backend:'wasm',bodyBackend:'mujoco-wasm',dtMs:config.dtMs,bodyBlockMs:config.bodyBlockMs,
          seed:j.seed,stage:j.stage,generation:j.generation,pairId:j.pairId,sign:j.sign};
        send({type:'evaluation',id:message.id,result:{...provenance,provenance,parameters:[...j.parameters],return:0,success:false,steps:1,simSeconds:.002,reason:'orchestration fixture'}});
      }
    }
    terminate(){this.terminated=true;}
  }
  const client=new TrainingClient({sharedOnly:true,coordinatorUrl:origin,workerFactory:()=>new FixtureWorker(),
    storage:{getItem:key=>stored.get(key),setItem:(key,value)=>stored.set(key,value)},
    fetcher:async(url,options={})=>{
      const path=new URL(url).pathname;calls.push({path,options});
      if(path.endsWith('/config.json'))return new Response(configBytes);
      if(unavailable)throw new Error('Coordinator offline');
      if(path.endsWith('/status'))return Response.json(status());
      if(path.endsWith('/checkpoint'))return Response.json(current());
      if(path.endsWith('/lease'))return Response.json({job});
      if(path.endsWith('/result'))return resultFailure?Response.json({error:'rejected fixture'},{status:422}):Response.json({accepted:true});
      if(path.endsWith('/release'))return Response.json({released:true});
      throw new Error('Unexpected fixture request '+path);
    }});
  return {client,calls,workers,current,setGeneration:value=>generation=value,setUnavailable:value=>unavailable=value,setResultFailure:value=>resultFailure=value};
}

test('shared-only client connects metadata without compute and displays the server checkpoint',async()=>{
  const {client,workers,current}=setup();await client.initialize();assert.equal(client.options.mode,'shared');
  assert.equal(workers.length,0);await client.connectCoordinator();assert.equal(workers.length,0);
  assert.equal(client.state.generation,3);assert.deepEqual(client.state.parameters,current().parameters);
  await client.dispose();
});

test('shared-only client rejects local work, opt-out, and a different coordinator',async()=>{
  const {client,workers}=setup();await client.initialize();
  await assert.rejects(client.start({mode:'local'}),/always contributes/);
  await assert.rejects(client.runLocal(client.runToken),/always contributes/);
  await assert.rejects(client.evaluateCheckpoint(),/only runs shared/);
  assert.throws(()=>client.importCheckpoint({}),/local checkpoint imports are disabled/);
  await assert.rejects(client.disconnectCoordinator(),/Sharing is always enabled/);
  await assert.rejects(client.start({coordinatorUrl:'https://another.example'}),/fixed shared coordinator/);
  await assert.rejects(client.connectCoordinator('https://another.example'),/fixed shared coordinator/);
  assert.equal(workers.length,0);await client.dispose();
});

test('an unavailable coordinator prevents all worker and local training initialization',async()=>{
  const {client,workers,setUnavailable}=setup();await client.initialize();await client.connectCoordinator();
  setUnavailable(true);let localRuns=0;client.runLocal=async()=>localRuns++;
  await assert.rejects(client.start(),/Coordinator offline/);
  assert.equal(client.state.coordinator.connected,false);assert.equal(client.state.phase,'error');
  assert.equal(workers.length,0);assert.equal(localRuns,0);assert.equal(client.options.mode,'shared');
  await client.dispose();
});

test('Start without a mode runs and submits an assigned shared job',async()=>{
  const {client,calls,workers}=setup();await client.initialize();
  client.fail(new Error('Previous connection failure'));
  const ensureWorker=client.ensureWorker.bind(client);
  client.ensureWorker=async()=>{assert.equal(client.state.error,null,'Retry must clear the old error before worker initialization');return ensureWorker();};
  const accepted=new Promise(resolve=>client.addEventListener('state',event=>{
    if(event.detail.contributedEpisodes===1){client.pause();resolve();}
  }));
  await client.start();await accepted;await client.stop();await client.loop;
  assert.equal(client.state.completedEpisodes,1);assert.equal(client.state.contributedEpisodes,1);
  assert.equal(calls.filter(call=>call.path.endsWith('/result')).length,1);
  assert.equal(client.state.history[0].role,'contribution');assert.equal(workers[0].terminated,true);
});

test('rejected shared results stay in the retry outbox and never become local training',async()=>{
  const {client,setResultFailure}=setup();await client.initialize();setResultFailure(true);
  await client.start();await client.loop;
  assert.equal(client.state.phase,'error');assert.equal(client.options.mode,'shared');
  assert.equal(client.state.contributedEpisodes,0);assert.ok(client.outbox);assert.equal(client.outboxUrl,origin);
  await client.dispose();
});

test('checkpoint downloads fetch each current generation without compute or cached local fallback',async()=>{
  const {client,workers,calls,setGeneration,setUnavailable}=setup();await client.initialize();
  const first=await client.downloadSharedCheckpoint();assert.equal(first.generation,3);
  assert.equal(first.createdAt,1234);assert.equal(first.provenance,'Fixture server metadata');assert.equal(first.completedGenerations,3);
  setGeneration(4);const next=await client.downloadSharedCheckpoint();assert.equal(next.generation,4);
  assert.equal(workers.length,0);
  const requests=calls.filter(call=>call.path.endsWith('/checkpoint'));assert.equal(requests.length,2);
  assert(requests.every(call=>call.options.cache==='no-store'));
  setUnavailable(true);await assert.rejects(client.downloadSharedCheckpoint(),/Coordinator offline/);
  await client.dispose();
});

test('checkpoint download rejects another model identity',async()=>{
  const {client,current}=setup();await client.initialize();
  client.fetcher=async()=>Response.json({...current(),configHash:'incorrect'});
  await assert.rejects(client.downloadSharedCheckpoint(),/configHash|configuration|different/i);
  await client.dispose();
});

test('Stop and upload failure preserve a completed result lease for a later Start retry',async()=>{
  for(const action of ['stop','fail']){
    const {client,current}=setup();await client.initialize();
    let leaseState='leased',releases=0,accepted=0;
    client.activeLease={jobId:'completed-pending-upload',leaseToken:'original-token',coordinatorUrl:origin};
    client.outbox={...client.leaseIdentity(client.activeLease),objective:0};client.outboxUrl=origin;
    client.api=async path=>{
      if(path==='/release'){releases++;leaseState='pending';return {released:true};}
      if(path==='/status')return {configHash,modelFingerprint:config.modelFingerprint,generation:3,checkpoint:current()};
      if(path==='/result'){
        if(leaseState!=='leased')throw Object.assign(new Error('Lease was released'),{status:410});
        accepted++;leaseState='completed';return {accepted:true};
      }
      throw new Error('Unexpected recovery request '+path);
    };
    if(action==='stop')await client.stop();else client.fail(new Error('Temporary upload failure'));
    assert.equal(releases,0);assert.equal(leaseState,'leased');assert.ok(client.outbox);
    const recovered=new Promise(resolve=>client.addEventListener('state',event=>{
      if(event.detail.contributedEpisodes===1){client.pause();resolve();}
    }));
    await client.start();await recovered;await client.stop();await client.loop;
    assert.equal(accepted,1);assert.equal(client.state.contributedEpisodes,1);assert.equal(client.outbox,null);
    assert.equal(leaseState,'completed');assert.equal(releases,0);
  }
});

test('Stop releases incomplete work and an outbox belonging to another lease does not retain it',async()=>{
  for(const outbox of [null,{jobId:'another-job',leaseToken:'another-token'}]){
    const {client}=setup();await client.initialize();let releases=0;
    client.activeLease={jobId:'incomplete-job',leaseToken:'current-token',coordinatorUrl:origin};
    client.outbox=outbox;client.outboxUrl=origin;client.releaseLease=async()=>releases++;
    await client.stop();assert.equal(releases,1);
  }
});
