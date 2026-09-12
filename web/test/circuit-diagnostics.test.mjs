import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {CircuitDiagnostics,validateCircuitProbe} from '../circuit-diagnostics.js';
import {createBrainModule} from '../../packages/fly-brain-wasm/dist/index.js';

const probe={schema_version:1,neuron_count:5,groups:[{key:'retina',label:'Inputs',indices:[0,1]}],
  channels:[{key:'forward',label:'Forward',indices:[3,4]}],incoming:{3:[[0,2],[1,-4]],4:[[0,2],[2,-3]]},
  cells:Object.fromEntries(Array.from({length:5},(_,i)=>[i,{index:i,type:i===1||i===2?'Inhibitor':'Cell '+i,root_id:String(i),side:'left'}]))};

test('diagnostics separate silent neurons, subthreshold voltage and signed input rankings',()=>{
  const d=new CircuitDiagnostics(probe),counts=new Float64Array([4,2,1,0,1]),v=new Float64Array([-52,-52,-52,-70,-48]);
  assert.equal(d.update(50,counts,v,new Float64Array(2)),null);
  const sample=d.update(100,counts,v,new Float64Array([-18,4]));
  assert.equal(sample.groups[0].meanHz,30);assert.equal(sample.channels[0].meanHz,5);
  assert.equal(sample.channels[0].active,1);assert.equal(sample.channels[0].meanMv,-59);
  assert.equal(sample.channels[0].cells[0].rateHz,0);assert.equal(sample.channels[0].meanDriveMv,-7);
  assert.equal(sample.channels[0].incomingPositive[0].weightedHz,80);
  assert.equal(sample.channels[0].incomingNegative[0].weightedHz,-55);
  assert.deepEqual(counts,new Float64Array([4,2,1,0,1]));
  const quiet=d.update(200,counts,v,new Float64Array(2));
  assert.equal(quiet.channels[0].spikes,0);assert.equal(quiet.channels[0].incomingNegative.length,0);
});

test('selecting a running brain establishes a new time and spike baseline',()=>{
  const d=new CircuitDiagnostics(probe),counts=new Float64Array([100,50,20,30,10]);
  d.reset(1000,counts);assert.equal(d.ready(1050),false);
  counts[3]++;const sample=d.update(1100,counts,new Float64Array(5).fill(-52),new Float64Array(2));
  assert.equal(sample.windowMs,100);assert.equal(sample.channels[0].spikes,1);assert.equal(sample.channels[0].cells[0].rateHz,10);
});

test('the diagnostic annotations fail closed for another graph or invalid edges',async()=>{
  const actual=JSON.parse(await readFile(new URL('../circuit-probe.json',import.meta.url)));
  assert.doesNotThrow(()=>validateCircuitProbe(actual,138639));
  assert.equal(actual.channels.flatMap(c=>c.indices).length,29);
  assert.throws(()=>validateCircuitProbe(actual,12),/match/);
  const bad=structuredClone(probe);bad.incoming[3][0][0]=10;
  assert.throws(()=>validateCircuitProbe(bad,5),/edge/);
});

test('inspecting a WASM brain does not change its electrical state or future spikes',async()=>{
  const module=await createBrainModule(),graph=module.createConnectome({neuronCount:5,rowOffsets:Uint32Array.of(0,1,2,2,2,2),targets:Uint32Array.of(3,4),weights:Float32Array.of(100,-100)});
  const [a,b]=graph.createPopulation(2,{seed:91});b.dispose();const twin=graph.createBrain({seed:91});
  try{
    for(const brain of [a,twin])brain.setPoissonInputs({indices:Uint32Array.of(0,1),ratesHz:Float32Array.of(50,50)}).step(100);
    const d=new CircuitDiagnostics(probe);d.update(a.timeMs,a.readActivations({field:'spikeCount'}),a.readActivations(),a.readActivations({field:'synapticDrive',indices:d.motorIds}));
    for(const brain of [a,twin])brain.step(100);
    for(const field of ['voltage','spikeCount','synapticDrive'])assert.deepEqual(a.readActivations({field}),twin.readActivations({field}));
  }finally{a.dispose();twin.dispose();graph.dispose();}
});
