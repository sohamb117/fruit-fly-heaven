import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {validateContributorOrigin,createContributorFetch,runContributor} from '../../scripts/contribute-training-native.mjs';

const sha=value=>createHash('sha256').update(value).digest('hex');
const wasmPlan=()=>({schemaVersion:1,backend:'wasm',configHash:'a'.repeat(64),modelFingerprint:'b'.repeat(64)});

test('HTTP loopback remains available without a plan; public HTTP stays rejected',()=>{
 for(const origin of ['http://localhost:7862/','http://127.0.0.1:7862/'])
  assert.equal(validateContributorOrigin(origin).href,origin);
 for(const origin of ['http://flytrain.morisoba.moe/','http://127.0.0.1.example.com/','ftp://127.0.0.1/'])
  assert.throws(()=>validateContributorOrigin(origin,wasmPlan()),/HTTPS or an HTTP loopback/);
});

test('HTTPS requires an explicit fully pinned backend plan',()=>{
 assert.throws(()=>validateContributorOrigin('https://flytrain.morisoba.moe/'),/explicit pinned/);
 assert.throws(()=>validateContributorOrigin('https://localhost/'),/explicit pinned/);
 assert.throws(()=>validateContributorOrigin('https://flytrain.morisoba.moe/',{backend:'wasm'}),/backend plan schema/);
 assert.equal(validateContributorOrigin('https://flytrain.morisoba.moe',wasmPlan()).href,'https://flytrain.morisoba.moe/');
 const dawn={...wasmPlan(),backend:'dawn-metal',nativeWebGPU:{moduleSha256:'c'.repeat(64),packageLockSha256:'d'.repeat(64)}};
 assert.equal(validateContributorOrigin('https://flytrain.morisoba.moe/',dawn).protocol,'https:');
 delete dawn.nativeWebGPU.packageLockSha256;
 assert.throws(()=>validateContributorOrigin('https://flytrain.morisoba.moe/',dawn),/exact loader and package-lock/);
});

test('coordinator origins cannot carry credentials, query, fragment or an endpoint path',()=>{
 for(const origin of ['https://user:secret@flytrain.morisoba.moe/','https://flytrain.morisoba.moe/?token=secret',
  'https://flytrain.morisoba.moe/#secret','https://flytrain.morisoba.moe/api/training','http://localhost:7862/config'])
  assert.throws(()=>validateContributorOrigin(origin,wasmPlan()),/without credentials, path, query or fragment/);
});

test('fetch stays on the exact selected origin and enforces redirect error',async()=>{
 const calls=[],controller=new AbortController(),origin=new URL('https://flytrain.morisoba.moe:8443/');
 const guarded=createContributorFetch(origin,async(input,options)=>{calls.push({url:input.href,options});return new Response('ok');});
 origin.hostname='changed.example'; // The installed guard owns the selection.
 await guarded('/api/training/status',{redirect:'follow',signal:controller.signal});
 assert.equal(calls[0].url,'https://flytrain.morisoba.moe:8443/api/training/status');
 assert.equal(calls[0].options.redirect,'error');assert.equal(calls[0].options.signal,controller.signal);
 for(const request of ['https://other.example/','https://flytrain.morisoba.moe/','http://flytrain.morisoba.moe:8443/',
  'https://user:secret@flytrain.morisoba.moe:8443/','//other.example/x','file:///tmp/unknown.wgsl'])
  assert.throws(()=>guarded(request),/denied/);
 assert.equal(calls.length,1);
});

test('only the two exact native shader file URLs map to the selected HTTPS origin',async()=>{
 const calls=[],guarded=createContributorFetch('https://flytrain.morisoba.moe/',async input=>{calls.push(input.href);return new Response('');});
 for(const shader of ['neural.wgsl','neural-dlm.wgsl'])
  await guarded(new URL('../../packages/banc-runtime/src/'+shader,import.meta.url));
 assert.deepEqual(calls,['https://flytrain.morisoba.moe/banc-engine/src/neural.wgsl','https://flytrain.morisoba.moe/banc-engine/src/neural-dlm.wgsl']);
 assert.throws(()=>guarded(new URL('../../packages/banc-runtime/src/model.js',import.meta.url)),/denied/);
});

async function fixture(run){
 const directory=await fs.mkdtemp(new URL('../../reports/native-cloud-fixture-',import.meta.url));
 const originalFetch=globalThis.fetch,originalLocation=Object.getOwnPropertyDescriptor(globalThis,'location');
 const modelFingerprint='d'.repeat(64),raw=JSON.stringify({schemaVersion:1,assets:{},modelFingerprint}),configHash=sha(raw);
 const backendPlan={schemaVersion:1,backend:'wasm',configHash,modelFingerprint,stopAfterGeneration:2};
 const planFile=directory+'/backend.json';await fs.writeFile(planFile,JSON.stringify(backendPlan));
 const requests=[];let serveRaw=raw;
 const fake=async(input,options)=>{
  const url=new URL(input);requests.push(url.pathname);assert.equal(url.origin,'https://flytrain.morisoba.moe');assert.equal(options.redirect,'error');
  if(url.pathname==='/training/config.json')return new Response(serveRaw);
  if(url.pathname==='/api/training/status')return Response.json({configHash,modelFingerprint,checkpoint:{generation:2}});
  throw new Error('Unexpected lease/result/simulation request: '+url.pathname);
 };
 globalThis.fetch=fake;
 try{await run({directory,planFile,requests,raw,changeRaw:value=>{serveRaw=value;}});assert.equal(globalThis.fetch,fake);
  assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis,'location'),originalLocation);}
 finally{globalThis.fetch=originalFetch;if(originalLocation)Object.defineProperty(globalThis,'location',originalLocation);else delete globalThis.location;await fs.rm(directory,{recursive:true,force:true});}
}

test('pinned HTTPS completed generation reads only config/status and restores global transport',async()=>{
 await fixture(async({directory,planFile,requests})=>{
  await runContributor(['https://flytrain.morisoba.moe/',directory+'/output','28',planFile]);
  assert.deepEqual(requests,['/training/config.json','/api/training/status']);
 });
});

test('HTTPS without a plan fails before requests, output creation or provider setup',async()=>{
 await fixture(async({directory,requests})=>{
  await assert.rejects(runContributor(['https://flytrain.morisoba.moe/',directory+'/output','1']),/explicit pinned/);
  assert.deepEqual(requests,[]);await assert.rejects(fs.access(directory+'/output'),{code:'ENOENT'});
 });
});

test('HTTPS config pin mismatch fails before status, lease or simulation and restores globals',async()=>{
 await fixture(async({directory,planFile,requests,raw,changeRaw})=>{
  changeRaw(raw+' ');
  await assert.rejects(runContributor(['https://flytrain.morisoba.moe/',directory+'/output','1',planFile]),/does not match coordinator/);
  assert.deepEqual(requests,['/training/config.json']);
 });
});

test('a redirected transport error is propagated without a second origin request',async()=>{
 let calls=0;
 const guarded=createContributorFetch('https://flytrain.morisoba.moe/',async(_input,options)=>{
  calls++;assert.equal(options.redirect,'error');throw new TypeError('fetch failed: unexpected redirect');
 });
 await assert.rejects(guarded('/training/config.json',{redirect:'follow'}),/unexpected redirect/);assert.equal(calls,1);
});
