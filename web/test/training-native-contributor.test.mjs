import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {validateContributorBackendPlan,withContributorBackend,assertContributorBackendEvidence} from '../../scripts/contribute-training-native.mjs';
const hash=value=>createHash('sha256').update(value).digest('hex');
const pins={moduleSha256:hash('module'),packageLockSha256:hash('lock')};
const plan=()=>({schemaVersion:1,backend:'dawn-metal',configHash:hash('config'),modelFingerprint:hash('model'),nativeWebGPU:{...pins}});
function provider(change=()=>{}){
 const log=[],provenance={...pins,backend:'dawn-metal',adapter:{isFallbackAdapter:false}};change(provenance);
 const native={provenance,uninstall(){log.push('uninstall');}};
 return{log,native,operations:{read:async file=>Buffer.from(file.endsWith('backend.mjs')?'module':'lock'),load:async()=>({installNativeWebGPU:async()=>{log.push('install');return native;}})}};
}
test('a supplied plan has an explicit backend, exact pins and no ignored evaluator fields',()=>{
 const accepted=validateContributorBackendPlan(plan());assert.equal(accepted.backend,'dawn-metal');assert(Object.isFrozen(accepted));assert(Object.isFrozen(accepted.nativeWebGPU));
 for(const mutate of [p=>delete p.backend,p=>p.schemaVersion=2,p=>p.configHash='wrong',p=>p.modelFingerprint=null,p=>p.jobs=[],p=>p.nativeWebGPU.extra=1,p=>delete p.nativeWebGPU.moduleSha256,p=>p.scope=1]){
  const p=plan();mutate(p);assert.throws(()=>validateContributorBackendPlan(p));
 }
 const p=plan();p.backend='wasm';assert.throws(()=>validateContributorBackendPlan(p),/WASM/);delete p.nativeWebGPU;assert.equal(validateContributorBackendPlan(p).backend,'wasm');
 const source=plan(),copy=validateContributorBackendPlan(source);source.nativeWebGPU.moduleSha256='changed';assert.equal(copy.nativeWebGPU.moduleSha256,pins.moduleSha256);
});
test('WASM default never imports or installs native WebGPU',async()=>{
 let calls=0;const result=await withContributorBackend(null,async native=>{assert.equal(native,null);return 7;},{read:()=>{calls++;throw Error('read');},load:()=>{calls++;throw Error('load');}});
 assert.equal(result,7);assert.equal(calls,0);
});
test('wrong on-disk pins fail before provider installation or callback',async()=>{
 const p=provider();let entered=false;
 await assert.rejects(withContributorBackend(plan(),async()=>{entered=true;},{...p.operations,read:async()=>Buffer.from('changed')}),/differs from backend plan/);
 assert.deepEqual(p.log,[]);assert.equal(entered,false);
});
test('installed provider is cleaned after setup/evaluation failure and successful completion',async()=>{
 for(const outcome of ['setup failed','evaluation failed',null]){
  const p=provider();const run=withContributorBackend(plan(),async native=>{assert.equal(native,p.native);p.log.push('callback');if(outcome)throw Error(outcome);return 'done';},p.operations);
  if(outcome)await assert.rejects(run,new RegExp(outcome));else assert.equal(await run,'done');
  assert.deepEqual(p.log,['install','callback','uninstall']);
 }
});
test('post-install wrong provenance and fallback adapters clean up before callback',async()=>{
 for(const change of [p=>p.packageLockSha256='wrong',p=>p.moduleSha256='wrong',p=>p.backend='wasm',p=>p.adapter.isFallbackAdapter=true]){
  const p=provider(change);let entered=false;
  await assert.rejects(withContributorBackend(plan(),async()=>{entered=true;},p.operations),/installation differs/);
  assert.equal(entered,false);assert.deepEqual(p.log,['install','uninstall']);
 }
});
test('accepted engine evidence cannot mix WASM, generic WebGPU, Dawn claim or body engine',()=>{
 for(const backend of ['wasm','webgpu']){
  const e={backend,bodyBackend:'mujoco-wasm',provenance:{backend,bodyBackend:'mujoco-wasm'},metrics:{actualNeuralBackend:backend}};
  assert.doesNotThrow(()=>assertContributorBackendEvidence(e,backend));
  for(const mutate of [x=>x.backend=backend==='wasm'?'webgpu':'wasm',x=>x.provenance.backend='dawn-metal',x=>delete x.metrics.actualNeuralBackend,x=>x.bodyBackend='native',x=>x.provenance.bodyBackend='other']){
   const x=structuredClone(e);mutate(x);assert.throws(()=>assertContributorBackendEvidence(x,backend),/backend evidence/);
  }
 }
});
