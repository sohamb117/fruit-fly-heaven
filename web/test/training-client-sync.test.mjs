// Metadata orchestration only. These fixtures cannot run or score a fly.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {TrainingClient} from '../training/client.js';
import {checkpoint} from '../training/optimizer.js';
import {nativeConfigFixture} from './fixtures/training-native-config.mjs';

const base=JSON.parse(await readFile(new URL('../training/config.json',import.meta.url)));
const config=nativeConfigFixture(base),bytes=JSON.stringify(config),configHash=createHash('sha256').update(bytes).digest('hex');
const origin='https://training.example',initial=config.parameters.map(p=>p.initial);
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:value=>resolve(value)};};
function status(generation=1){
  const parameters=initial.map((v,i)=>i===0?v+generation/100:v);
  return {modelFingerprint:config.modelFingerprint,configHash,generation,acceptedResults:generation*8,jobs:{leased:1},
    recentWallSeconds:25,recentTrials:[{episode:generation*8,generation,stage:config.stage,return:generation,success:false,simSeconds:1,wallSeconds:25,completedAt:1700000000}],
    lastParameterUpdate:{generation,changedCount:1,parameterCount:672,completedAt:1700000000},
    checkpoint:checkpoint(config,configHash,parameters,generation,config.stage)};
}
function setup(stored=new Map()){
  const calls=[],reads=[];let workers=0,response=status();
  const client=new TrainingClient({sharedOnly:true,coordinatorUrl:origin,workerFactory:()=>{workers++;throw new Error('No browser worker allowed');},
    storage:{getItem:key=>(reads.push(key),stored.get(key)),setItem:(key,value)=>stored.set(key,value)},
    fetcher:async(url,options={})=>{calls.push({url:String(url),options});return new URL(url).pathname.endsWith('/config.json')?new Response(bytes):Response.json(response);}});
  return {client,calls,reads,get workers(){return workers;},setResponse:value=>response=value};
}

test('a previous 27-value cache and outbox cannot cause a new 672-value run mismatch',async()=>{
  const stored=new Map([['heaven-training-v1:old-config',{checkpoint:{parameters:Array(27).fill(0)},outbox:{objective:3}}]]),f=setup(stored);
  await f.client.initialize();await f.client.connectCoordinator();
  assert.deepEqual(f.reads,[`heaven-training-v1:${configHash}`]);assert.equal(f.client.parameters.length,672);
  assert.equal(f.client.outbox,undefined);assert.equal(f.client.state.syncError,null);assert.equal(f.workers,0);
  assert.equal(f.calls[0].options.cache,'no-store');await f.client.dispose();
});

test('a corrupt cache at the current key is ignored before fresh server metadata is applied',async()=>{
  const saved={checkpoint:{...status().checkpoint,parameters:Array(27).fill(0)},outbox:{objective:3}};
  const f=setup(new Map([[`heaven-training-v1:${configHash}`,JSON.stringify(saved)]]));
  await f.client.initialize();assert.deepEqual(f.client.parameters,initial);await f.client.connectCoordinator();
  assert.equal(f.client.state.coordinator.connected,true);assert.equal(f.client.outbox,undefined);await f.client.dispose();
});

test('native-only runs reject every browser execution entry before a worker or lease exists',async()=>{
  const f=setup();await f.client.initialize();await f.client.connectCoordinator();
  assert.equal(f.client.state.browserTrainingAvailable,false);
  for(const action of [()=>f.client.start(),()=>f.client.ensureWorker(),()=>f.client.runShared(f.client.runToken)])
    await assert.rejects(action(),{code:'native_trainer_required'});
  assert.equal(f.workers,0);assert.equal(f.calls.some(call=>/\/(lease|result|heartbeat)$/.test(call.url)),false);
  await f.client.dispose();
});

test('refresh advances native checkpoint and recorded progress without changing local counters or starting compute',async()=>{
  const f=setup();await f.client.initialize();await f.client.connectCoordinator();f.setResponse(status(2));
  await f.client.refreshCoordinatorStatus();
  assert.equal(f.client.state.generation,2);assert.deepEqual(f.client.parameters,status(2).checkpoint.parameters);
  assert.equal(f.client.state.coordinator.recentTrials[0].episode,16);assert.equal(f.client.state.coordinator.lastParameterUpdate.changedCount,1);
  assert.equal(f.client.state.completedEpisodes,0);assert.equal(f.workers,0);assert.equal(f.client.state.phase,'ready');
  await f.client.dispose();
});

test('true build or checkpoint mismatch preserves the trusted vector and a matching retry reconnects',async()=>{
  for(const mutate of [s=>({...s,configHash:'other'}),s=>({...s,checkpoint:{...s.checkpoint,parameters:Array(27).fill(0)}})]){
    const f=setup();await f.client.initialize();await f.client.connectCoordinator();const trusted=f.client.parameters.slice();
    f.setResponse(mutate(status(2)));await assert.rejects(f.client.refreshCoordinatorStatus(),{code:'build_mismatch'});
    assert.deepEqual(f.client.parameters,trusted);assert.equal(f.client.state.coordinator.connected,false);assert.equal(f.client.state.syncError.code,'build_mismatch');
    f.setResponse(status(2));await f.client.refreshCoordinatorStatus();assert.equal(f.client.state.syncError,null);assert.equal(f.client.state.coordinator.connected,true);
    await f.client.dispose();
  }
});

test('overlapping or regressed status responses cannot roll back newer progress',async()=>{
  const f=setup();await f.client.initialize();await f.client.connectCoordinator();const older=deferred(),newer=deferred();let calls=0;
  f.client.api=()=>++calls===1?older.promise:newer.promise;
  const first=f.client.refreshCoordinatorStatus(),firstRejected=assert.rejects(first,{name:'AbortError'}),second=f.client.refreshCoordinatorStatus();
  newer.resolve(status(3));await second;older.resolve(status(2));await firstRejected;
  assert.equal(f.client.state.generation,3);f.client.api=async()=>status(1);await f.client.refreshCoordinatorStatus();
  assert.equal(f.client.state.generation,3);assert.equal(f.client.state.coordinator.acceptedResults,24);await f.client.dispose();
});

test('Stop invalidates an in-flight monitor response',async()=>{
  const f=setup();await f.client.initialize();await f.client.connectCoordinator();const waiting=deferred();f.client.api=()=>waiting.promise;
  const pending=f.client.refreshCoordinatorStatus(),rejected=assert.rejects(pending,{name:'AbortError'});await f.client.stop();
  waiting.resolve(status(2));await rejected;assert.equal(f.client.state.generation,1);assert.equal(f.client.state.phase,'stopped');
});
