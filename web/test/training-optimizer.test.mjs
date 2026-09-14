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
test('bounded ES uses realized perturbations after clipping',()=>{
  const bounded=structuredClone(config);bounded.optimizer.populationPairs=1;bounded.optimizer.sigma=.25;bounded.optimizer.learningRate=.1;
  bounded.parameters=bounded.parameters.map((p,i)=>i===0?{...p,min:-1,max:1,initial:.99}:p);
  const baseline=bounded.parameters.map(p=>p.initial),round=makeGeneration(baseline,0,bounded),pair=round.pairs[0];
  pair.results[-1]=result(0);pair.results[1]=result(1);
  const k=0,realized=(pair.jobs.find(j=>j.sign===1).parameters[k]-pair.jobs.find(j=>j.sign===-1).parameters[k])/(2*bounded.optimizer.sigma);
  const expected=Math.min(1,baseline[k]+Math.max(-bounded.optimizer.maximumUpdate,Math.min(bounded.optimizer.maximumUpdate,bounded.optimizer.learningRate/(2*bounded.optimizer.sigma)*realized)));
  assert.equal(updateGeneration(round,bounded)[k],expected);
  assert.notEqual(realized,pair.noise[k]);
});
test('incomplete/invalid returns cannot update a policy',()=>{
  const round=makeGeneration(initial,0,config);assert.throws(()=>updateGeneration(round,config));
  for(const bad of [NaN,Infinity,config.objective.max+1])assert.throws(()=>validateResult(result(bad),config));
});
test('checkpoint import rejects incompatibility and discards purported validation',()=>{
  const c=checkpoint(config,'abc',initial,2,config.stage,{status:'validated',score:10});
  assert.equal(readCheckpoint(c,config,'abc').status,'unverified');
  assert.throws(()=>readCheckpoint(c,config,'def'));
  assert.throws(()=>readCheckpoint({...c,parameters:[NaN,...initial.slice(1)]},config,'abc'));
  assert.throws(()=>readCheckpoint({...c,modelFingerprint:'wrong'},config,'abc'));
  assert.throws(()=>readCheckpoint({...c,parameterNames:[]},config,'abc'));
  assert.throws(()=>readCheckpoint({...c,stage:'posture'},config,'abc'),/Invalid checkpoint progress/);
});
