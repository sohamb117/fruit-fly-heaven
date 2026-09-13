import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {validateConfig,makeGeneration,updateGeneration,checkpoint,readCheckpoint,validateResult} from '../training/optimizer.js';
const config=validateConfig(JSON.parse(await fs.readFile(new URL('../training/config.json',import.meta.url),'utf8')));
const initial=config.parameters.map(p=>p.initial);
const result=(reward)=>({return:reward,success:false,simSeconds:1,steps:500});
test('paired exploration is reproducible with common episode seeds and bounded parameters',()=>{
  const a=makeGeneration(initial,0,config),b=makeGeneration(initial,0,config);
  assert.deepEqual(a,b);
  for(const p of a.pairs){assert.equal(p.jobs[0].seed,p.jobs[1].seed);for(const j of p.jobs)j.parameters.forEach((x,k)=>assert.ok(x>=config.parameters[k].min&&x<=config.parameters[k].max));}
  assert.notDeepEqual(a,makeGeneration(initial,1,config));
});
test('reward search improves a known objective; identical returns cannot move parameters',()=>{
  const round=makeGeneration(initial,0,config);
  for(const p of round.pairs)for(const job of p.jobs)p.results[job.sign]=result(job.parameters[0]);
  assert.ok(updateGeneration(round,config)[0]>initial[0]);
  for(const p of round.pairs)p.results={[-1]:result(1),[1]:result(1)};
  assert.deepEqual(updateGeneration(round,config),initial);
});
test('incomplete/invalid returns cannot update a policy',()=>{
  const round=makeGeneration(initial,0,config);assert.throws(()=>updateGeneration(round,config));
  for(const bad of [NaN,Infinity,config.objective.max+1])assert.throws(()=>validateResult(result(bad),config));
});
test('checkpoint import rejects incompatibility and discards purported validation',()=>{
  const c=checkpoint(config,'abc',initial,2,'posture',{status:'validated',score:10});
  assert.equal(readCheckpoint(c,config,'abc').status,'unverified');
  assert.throws(()=>readCheckpoint(c,config,'def'));
  assert.throws(()=>readCheckpoint({...c,parameters:[NaN,...initial.slice(1)]},config,'abc'));
  assert.throws(()=>readCheckpoint({...c,modelFingerprint:'wrong'},config,'abc'));
  assert.throws(()=>readCheckpoint({...c,parameterNames:[]},config,'abc'));
});
