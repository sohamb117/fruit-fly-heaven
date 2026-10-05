import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {assertGuardedContributorPlan,guardedResultMetrics,validateContributorBackendPlan,runContributor} from '../../scripts/contribute-training-native.mjs';

const nativeExecution={backend:'dawn-metal',moduleSha256:'a'.repeat(64),packageLockSha256:'b'.repeat(64)};
const config={schemaVersion:2,optimizer:{acceptance:{profile:1,proposal:'best-search-job',seedCount:3,nativeExecution}}};
const plan={backend:'dawn-metal',nativeWebGPU:{moduleSha256:nativeExecution.moduleSha256,packageLockSha256:nativeExecution.packageLockSha256}};

test('guarded workers reject a missing, fallback or mismatched backend plan before simulation setup',()=>{
 assert.doesNotThrow(()=>assertGuardedContributorPlan(config,plan));
 for(const invalid of [null,{backend:'wasm'},{...plan,nativeWebGPU:{...plan.nativeWebGPU,moduleSha256:'c'.repeat(64)}}])
  assert.throws(()=>assertGuardedContributorPlan(config,invalid),/pinned native Dawn/);
 assert.doesNotThrow(()=>assertGuardedContributorPlan({schemaVersion:1},null));
});

test('guarded payloads retain explicit termination and cancellation evidence',()=>{
 assert.deepEqual(guardedResultMetrics({terminated:true,reason:'excessive_rotation'},config),{terminated:true,cancelled:false});
 assert.deepEqual(guardedResultMetrics({terminated:false,reason:'time_limit'},config),{terminated:false,cancelled:false});
 assert.deepEqual(guardedResultMetrics({}, {schemaVersion:1}),{});
});

test('technical failures are never converted into guarded scientific scores',()=>{
 for(const reason of ['cancelled','simulation_error','invalid_observation','unexpected_external_force'])
  assert.throws(()=>guardedResultMetrics({terminated:true,reason},config),/cannot enter/);
 assert.throws(()=>guardedResultMetrics({terminated:false,cancelled:true,reason:'time_limit'},config),/cannot enter/);
 assert.throws(()=>guardedResultMetrics({reason:'excessive_rotation'},config),/cannot enter/);
});

test('generation limits are explicit positive integers in a pinned backend plan',()=>{
 const complete={...plan,schemaVersion:1,configHash:'c'.repeat(64),modelFingerprint:'d'.repeat(64),stopAfterGeneration:2};
 assert.equal(validateContributorBackendPlan(complete).stopAfterGeneration,2);
 for(const stopAfterGeneration of [0,-1,1.5,true,'2',Infinity,Number.MAX_SAFE_INTEGER+1])
  assert.throws(()=>validateContributorBackendPlan({...complete,stopAfterGeneration}),/generation limit/);
});

test('an already completed run stops before leasing or loading any simulation',async()=>{
 const reports=new URL('../../reports/',import.meta.url);await fs.mkdir(reports,{recursive:true});
 const directory=await fs.mkdtemp(new URL('guard-stop-fixture-',reports));
 const modelFingerprint='d'.repeat(64),raw=JSON.stringify({schemaVersion:1,assets:{},modelFingerprint});
 const configHash=createHash('sha256').update(raw).digest('hex');
 const backendPlan={schemaVersion:1,backend:'wasm',configHash,modelFingerprint,stopAfterGeneration:2};
 const planFile=directory+'/backend.json';await fs.writeFile(planFile,JSON.stringify(backendPlan));
 const original=globalThis.fetch,requests=[];
 globalThis.fetch=async input=>{
  const path=new URL(input).pathname;requests.push(path);
  if(path==='/training/config.json')return new Response(raw);
  if(path==='/api/training/status')return Response.json({configHash,modelFingerprint,checkpoint:{generation:2}});
  throw new Error('Unexpected compute or lease request: '+path);
 };
 try{
  await runContributor(['http://127.0.0.1:7899/',directory+'/output','28',planFile]);
  assert.deepEqual(requests,['/training/config.json','/api/training/status']);
 }finally{globalThis.fetch=original;await fs.rm(directory,{recursive:true,force:true});}
});
