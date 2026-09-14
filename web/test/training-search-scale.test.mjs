import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import * as staged from '../training/optimizer.js';

const canonical = {"schemaVersion":1,"environmentVersion":"banc-flybody-flight-objective-v3","modelFingerprint":"4d53af18ac499bf90ecc2d98342cd596bc65e12ddbbf9f92c66d87a3e6757cc6","assets":{"fixture.wasm":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"algorithm":"antithetic-evolution-strategies","dtMs":0.5,"bodyBlockMs":2,"parameters":[{"name":"gain_log","min":-2,"max":2,"initial":0}],"optimizer":{"populationPairs":2,"sigma":0.25,"learningRate":0.035,"maximumUpdate":0.15,"seed":888},"objective":{"min":-10,"max":10,"direction":"maximize"},"stage":"landing","durationSeconds":8,"stages":[{"id":"takeoff","durationSeconds":3},{"id":"flight","durationSeconds":5},{"id":"landing","durationSeconds":8}],"contribution":{"leaseSeconds":10,"maxRequestBytes":1024}};
const result = value => ({return:value, success:false, simSeconds:1, steps:500});
const clone = x => structuredClone(x);
function fixture(scales=[.5,.25], bounds=[[-10,10],[-10,10]], initial=[0,0]) {
  const config=clone(canonical);
  config.parameters=scales.map((scale,k)=>({name:`p${k}`,min:bounds[k][0],max:bounds[k][1],initial:initial[k],...(scale===undefined?{}:{searchScale:scale})}));
  config.optimizer={...config.optimizer,populationPairs:2,sigma:.25,learningRate:.035,maximumUpdate:.15};
  return staged.validateConfig(config);
}
function scored(round) {
  for (const pair of round.pairs) for (const job of pair.jobs) pair.results[job.sign]=result(Math.tanh(job.parameters.reduce((s,x,k)=>s+(k+1)*x,0)));
  return round;
}
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-14, `${a} != ${b}`);

test('omitted search scales preserve the pinned legacy generation/update signature',()=>{
  const config=fixture([undefined,undefined]), before=JSON.stringify(config), records=[];
  for(let generation=0;generation<32;generation++) {
    const round=scored(staged.makeGeneration([0,0],generation,config));
    records.push([round,staged.updateGeneration(round,config)]);
  }
  // Generated once from the legacy implementation before searchScale existed.
  assert.equal(createHash('sha256').update(JSON.stringify(records)).digest('hex'),'9052ccfdd54f7dac6721d9fc55f05aa3a07695a6eb771a175484ee6397854549');
  assert.equal(JSON.stringify(config),before);
});

test('explicit one equals omission; per-coordinate scales change only assigned displacements',()=>{
  const missing=fixture([undefined,undefined]), ones=fixture([1,1]), scaled=fixture();
  for(let generation=0;generation<16;generation++) {
    const base=staged.makeGeneration([0,0],generation,missing);
    assert.deepEqual(staged.makeGeneration([0,0],generation,ones),base);
    const current=staged.makeGeneration([0,0],generation,scaled);
    current.pairs.forEach((pair,i)=>{
      assert.deepEqual(pair.noise,base.pairs[i].noise);
      pair.jobs.forEach((job,j)=>{
        assert.equal(job.seed,base.pairs[i].jobs[j].seed);
        job.parameters.forEach((x,k)=>assert.equal(x,base.pairs[i].jobs[j].parameters[k]*scaled.parameters[k].searchScale));
      });
    });
  }
});

test('scaled candidates clip at original bounds and aggregate from realized, not raw, displacement',()=>{
  const c=fixture([.5,.125],[[-1,1],[-.001,.001]],[.999,0]);
  const round=staged.makeGeneration(c.parameters.map(p=>p.initial),0,c);
  let clipped=0;
  for(const pair of round.pairs) for(const job of pair.jobs) {
    pair.results[job.sign]=result(job.sign);
    job.parameters.forEach((x,k)=>{
      const p=c.parameters[k],raw=round.baseline[k]+job.sign*c.optimizer.sigma*p.searchScale*pair.noise[k];
      assert.equal(x,Math.max(p.min,Math.min(p.max,raw)));
      clipped+=Number(x!==raw);
    });
  }
  assert.ok(clipped>0);
  const actual=staged.updateGeneration(round,c);
  round.baseline.forEach((x,k)=>{
    const direction=round.pairs.reduce((s,p)=>s+2*(p.jobs.find(j=>j.sign===1).parameters[k]-p.jobs.find(j=>j.sign===-1).parameters[k])/(2*c.optimizer.sigma),0);
    const update=c.optimizer.learningRate/(2*c.optimizer.populationPairs*c.optimizer.sigma)*direction;
    const p=c.parameters[k];
    assert.equal(actual[k],Math.max(p.min,Math.min(p.max,x+Math.max(-c.optimizer.maximumUpdate,Math.min(c.optimizer.maximumUpdate,update)))));
  });
});

test('realized aggregate equals isotropic ES in scaled coordinates, including transformed candidate bounds',()=>{
  for(const bounds of [[[-10,10],[-10,10]],[[-.001,.001],[-.002,.002]]]) {
    const c=fixture([.5,.25],bounds), round=scored(staged.makeGeneration([0,0],3,c));
    const normalized=clone(c);
    normalized.parameters=normalized.parameters.map(p=>({...p,min:p.min/p.searchScale,max:p.max/p.searchScale,initial:p.initial/p.searchScale,searchScale:1}));
    const normalizedRound=clone(round);
    normalizedRound.baseline=round.baseline.map((x,k)=>x/c.parameters[k].searchScale);
    normalizedRound.pairs.forEach(pair=>pair.jobs.forEach(job=>job.parameters=job.parameters.map((x,k)=>x/c.parameters[k].searchScale)));
    const physical=staged.updateGeneration(round,c), y=staged.updateGeneration(normalizedRound,normalized);
    physical.forEach((x,k)=>near(x,y[k]*c.parameters[k].searchScale));
  }
});

test('orthogonal linear objective has s squared response; physical update cap is retained',()=>{
  const c=fixture([.5,.25]); c.optimizer.learningRate=.1;
  const noises=[[Math.SQRT2,0],[0,Math.SQRT2]], gradient=[2,-3];
  const round={generation:0,stage:c.stage,baseline:[0,0],pairs:noises.map(noise=>({noise,jobs:[-1,1].map(sign=>({sign,parameters:noise.map((x,k)=>sign*c.optimizer.sigma*c.parameters[k].searchScale*x)})),results:{}}))};
  for(const pair of round.pairs) for(const job of pair.jobs) pair.results[job.sign]=result(job.parameters.reduce((s,x,k)=>s+x*gradient[k],0));
  const next=staged.updateGeneration(round,c);
  next.forEach((x,k)=>near(x,c.optimizer.learningRate*c.parameters[k].searchScale**2*gradient[k]));
  c.optimizer.maximumUpdate=.005;
  assert.deepEqual(staged.updateGeneration(round,c),[.005,-.005]);
});

test('equal rewards do not move the scaled center; malformed scales reject',()=>{
  const c=fixture(), round=staged.makeGeneration([0,0],0,c);
  for(const pair of round.pairs) pair.results={[-1]:result(1),[1]:result(1)};
  assert.deepEqual(staged.updateGeneration(round,c),[0,0]);
  for(const value of [0,-1,1.0001,NaN,Infinity,-Infinity,null,true,false,'0.5',[],{}]) {
    const bad=clone(c); bad.parameters[0].searchScale=value;
    assert.throws(()=>staged.validateConfig(bad),/searchScale/,String(value));
  }
  for(const value of [Number.MIN_VALUE,.01,.5,1]) { const valid=clone(c);valid.parameters[0].searchScale=value;assert.equal(staged.validateConfig(valid),valid); }
});
