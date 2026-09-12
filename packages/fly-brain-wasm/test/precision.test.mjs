import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createBrainModule} from '../dist/index.js';
const ids=(...v)=>Uint32Array.from(v),floats=(...v)=>Float32Array.from(v);
const runtime=await createBrainModule({precision:'float32'});
const graph=(weight=250)=>runtime.createConnectome({neuronCount:2,rowOffsets:ids(0,1,1),targets:ids(1),weights:floats(weight)});

test('FP32 rounds stored neuron state and exposes precision in readouts',()=>{
  assert.equal(runtime.precision,'float32');
  const g=graph(),b=g.createBrain();let expected=0;
  for(let i=0;i<100;i++){b.injectVoltage(ids(0),floats(.01));expected=Math.fround(expected+Math.fround(.01));}
  assert.equal(b.precision,'float32');assert.equal(b.readActivations()[0],-52+expected);
  const matrix=runtime.readActivationMatrix([b]);assert.equal(matrix.precision,'float32');assert.ok(matrix.values instanceof Float64Array);
  matrix.values[0]=999;assert.equal(b.readActivations()[0],-52+expected);
  b.dispose();g.dispose();
});

test('precision selection rejects unknown modes and mismatched WASM artifacts',async()=>{
  await assert.rejects(createBrainModule({precision:'int8'}),/precision must be/);
  await assert.rejects(createBrainModule({precision:'float32',wasmBinary:fs.readFileSync(new URL('../dist/core.wasm',import.meta.url))}),/does not match requested precision/);
  const standard=await createBrainModule();assert.equal(standard.precision,'float64');
});

for(const parameters of [{},{dtMs:1,delayMs:2,refractoryMs:2}]){
  test(`FP32 ${parameters.dtMs??.1} ms grid preserves seeded partitioning and signed signaling`,()=>{
    const positive=graph(),negative=graph(-250),a=positive.createBrain({...parameters,seed:44}),b=positive.createBrain({...parameters,seed:44}),c=negative.createBrain({...parameters,seed:44});
    for(const brain of [a,b,c])brain.setRefractoryPeriod(ids(0),0).setPoissonInputs({indices:ids(0),ratesHz:floats(150)});
    a.step(100);c.step(100);for(let i=0;i<50;i++){b.step(2);b.readActivations();}
    assert.deepEqual(a.readActivations(),b.readActivations());assert.deepEqual(a.readSpikes(),b.readSpikes());
    assert.ok(a.readActivations({field:'spikeCount'})[1]>0);assert.equal(c.readActivations({field:'spikeCount'})[1],0);
    const dt=parameters.dtMs??.1;for(const t of a.readSpikes().timesMs)assert.ok(Math.abs(t/dt-Math.round(t/dt))<1e-7);
    for(const brain of [a,b,c])brain.dispose();positive.dispose();negative.dispose();
  });
}

test('FP32 long quiet intervals and slow tonic currents remain finite',()=>{
  const g=graph(),quiet=g.createBrain(),tonic=g.createBrain({tauMembraneMs:2000});
  quiet.step(1500);assert.deepEqual([...quiet.readActivations()],[-52,-52]);
  quiet.injectVoltage(ids(1),floats(-3)).step(1500);assert.ok(quiet.readActivations().every(Number.isFinite));
  tonic.setCurrentInputs(ids(1),floats(8)).step(5000);assert.ok(tonic.totalSpikes>0);assert.ok(tonic.readActivations().every(Number.isFinite));
  quiet.dispose();tonic.dispose();g.dispose();
});
