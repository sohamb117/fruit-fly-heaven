import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createBrainModule} from '../dist/index.js';
const runtime=await createBrainModule();
const ids=(...a)=>Uint32Array.from(a), floats=(...a)=>Float32Array.from(a);
function graph(weight=250){return runtime.createConnectome({neuronCount:2,rowOffsets:ids(0,1,1),targets:ids(1),weights:floats(weight)});}

test('signals propagate through supplied edges with delayed excitation',()=>{
  const g=graph(),b=g.createBrain({seed:16});
  b.setRefractoryPeriod(ids(0),0).setPoissonInputs({indices:ids(0),ratesHz:floats(600)});b.step(100);
  const h=b.readSpikes(),observed=Array.from(h.timesMs,(t,i)=>[Math.round(t*10),h.neuronIndices[i]]);
  assert.equal(h.dropped,0);
  // Independent dense reference using the same sensory spike train.
  const sensory=new Set(observed.filter(x=>x[1]===0).map(x=>x[0]));
  const v=[0,0],syn=[0,0],ref=[0,0],pending=new Map(),expected=[];
  const a=Math.exp(-.1/20),c=Math.exp(-.1/5);
  for(let t=0;t<1000;t++){
    if(t)for(let i=0;i<2;i++)if(t>ref[i]){v[i]=v[i]*a+syn[i]*(a-c)/3;syn[i]*=c;}
    const arriving=pending.get(t)||0;if(t>=ref[1])syn[1]+=arriving;
    if(sensory.has(t))v[0]+=68.75;
    for(let i=0;i<2;i++)if(v[i]>7 && t>=ref[i]){
      expected.push([t,i]);v[i]=syn[i]=0;ref[i]=t+(i===0?0:22);
      if(i===0)pending.set(t+18,(pending.get(t+18)||0)+68.75);
    }
  }
  // The intrusive scheduler orders same-tick cells differently; spike sets must agree.
  const order=(x,y)=>x[0]-y[0]||x[1]-y[1];
  assert.deepEqual(observed.sort(order),expected.sort(order));assert.ok(observed.some(x=>x[1]===1));
  b.dispose();g.dispose();
});

test('inhibition suppresses the postsynaptic cell; zero input is silent',()=>{
  const g=graph(-250),b=g.createBrain();b.step(100);assert.equal(b.totalSpikes,0);
  b.setPoissonInputs({indices:ids(0),ratesHz:floats(600)}).step(100);
  const counts=b.readActivations({field:'spikeCount'});assert.ok(counts[0]>0);assert.equal(counts[1],0);
  assert.ok(b.readActivations()[1]<-52);b.dispose();g.dispose();
});

test('quiet neurons and slow tonic drives remain finite beyond 10,000 ticks',()=>{
  const g=graph(),quiet=g.createBrain(),driven=g.createBrain({tauMembraneMs:2000});
  quiet.step(1500);
  assert.deepEqual([...quiet.readActivations()],[-52,-52]);
  assert.deepEqual([...quiet.readActivations({field:'synapticDrive'})],[0,0]);
  quiet.injectVoltage(ids(0),floats(-3)).step(1500);
  assert.ok(quiet.readActivations().every(Number.isFinite));
  driven.setCurrentInputs(ids(1),floats(8)).step(5000);
  assert.ok(driven.readActivations().every(Number.isFinite));
  assert.ok(driven.readSpikes().timesMs[0]>4000);
  quiet.dispose();driven.dispose();g.dispose();
});

test('seeded trajectories do not depend on step partition or observing matrices',()=>{
  const g=graph();const brains=[g.createBrain({seed:42}),g.createBrain({seed:42}),g.createBrain({seed:43})];
  for(const b of brains)b.setPoissonInputs({indices:ids(0,1),ratesHz:floats(140,70)});
  brains[0].step(100);
  for(let i=0;i<20;i++){brains[1].step(5);brains[1].readActivations();}
  brains[2].step(100);
  assert.deepEqual(brains[0].readSpikes(),brains[1].readSpikes());assert.notDeepEqual(brains[0].readSpikes(),brains[2].readSpikes());
  for(const b of brains)b.dispose();g.dispose();
});

test('100 independent instances expose a row-major activation matrix',()=>{
  const g=graph(),brains=g.createPopulation(100,{seed:2026});
  for(const b of brains)b.setPoissonInputs({indices:ids(0),ratesHz:floats(150)}).step(50);
  const m=runtime.readActivationMatrix(brains,{field:'spikeCount'});
  assert.deepEqual(m.shape,[100,2]);assert.equal(m.values.length,200);assert.equal(m.unit,'spikes');
  assert.ok(new Set(brains.map(b=>JSON.stringify(b.readSpikes()))).size>95);
  for(let i=0;i<100;i++)assert.deepEqual(Array.from(m.values.slice(i*2,i*2+2)),Array.from(brains[i].readActivations({field:'spikeCount'})));
  brains[0].step(10);assert.equal(brains[1].timeMs,50);
  for(const b of brains)b.dispose();g.dispose();
});

test('arbitrary pulses, tonic current and per-neuron refractory overrides',()=>{
  const g=graph(),a=g.createBrain(),b=g.createBrain();
  a.injectVoltage(ids(0),floats(9)).step(.1);assert.equal(a.totalSpikes,1);
  b.setCurrentInputs(ids(1),floats(20)).step(100);
  assert.ok(b.readActivations({field:'spikeCount'})[1]>5);
  const first=b.readSpikes().timesMs[0];assert.ok(Math.abs(first-8.7)<1e-9);
  a.dispose();b.dispose();g.dispose();
});

test('history overflow is explicit and snapshots are owned copies',()=>{
  const g=graph(),b=g.createBrain({spikeHistoryCapacity:2});
  const before=b.readActivations();b.setCurrentInputs(ids(0),floats(30)).step(100);
  const h=b.readSpikes();assert.equal(h.timesMs.length,2);assert.equal(h.dropped,h.total-2);
  assert.deepEqual(Array.from(before),[-52,-52]);b.dispose();g.dispose();
});

test('validates graph, inputs, configuration, duration and disposed handles',()=>{
  assert.throws(()=>runtime.createConnectome({neuronCount:2,rowOffsets:ids(0,2,1),targets:ids(1),weights:floats(1)}));
  const g=graph();assert.throws(()=>g.createBrain({tauSynapseMs:20}));
  const b=g.createBrain();assert.throws(()=>b.injectVoltage(ids(2),floats(1)));assert.throws(()=>b.step(.15));
  assert.throws(()=>b.setPoissonInputs({indices:ids(0,0),ratesHz:floats(10,20)}));
  assert.throws(()=>b.setCurrentInputs(ids(0),floats(NaN)));
  g.dispose();b.step(1);assert.equal(b.timeMs,1);assert.throws(()=>g.createBrain());
  b.dispose();b.dispose();assert.throws(()=>b.step(1));
});
