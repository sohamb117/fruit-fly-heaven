// Differential validation against the immutable 0.1.1 release, independently
// of the performance workload. Extracts its own baseline from the release archive.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
const baseline=new URL('../build/performance/release-baseline/',import.meta.url);
fs.mkdirSync(baseline,{recursive:true});
execFileSync('tar',['-xzf',new URL('../releases/fruit-fly-brain-wasm-0.1.1.tgz',import.meta.url).pathname,'-C',baseline.pathname]);
const original=await import(new URL('package/dist/index.js',baseline));
const updated=await import('../packages/fly-brain-wasm/dist/index.js');
const modules=await Promise.all([original.createBrainModule(),updated.createBrainModule()]);
let seed=8;const random=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)/2**32);
const ids=Uint32Array.from([0,1,2,3,4,5,6,7]);
let comparisons=0;
for(const fractional of [false,true])for(const config of [{},{tauMembraneMs:5,tauSynapseMs:4,dtMs:.2},{tauMembraneMs:2000,tauSynapseMs:5},{tauMembraneMs:20,tauSynapseMs:19.99,resetMv:-50}]){
  const n=257,rowOffsets=new Uint32Array(n+1),targets=[],weights=[];
  for(let i=0;i<n;i++){
    for(let k=0;k<12;k++){targets.push(Math.floor(random()*n));weights.push((Math.floor(random()*401)-120)*(fractional?.375:1));}
    rowOffsets[i+1]=targets.length;
  }
  const graphs=modules.map(m=>m.createConnectome({neuronCount:n,rowOffsets,targets:Uint32Array.from(targets),weights:Float32Array.from(weights)}));
  const brains=graphs.map(g=>g.createBrain({...config,seed:1234,spikeHistoryCapacity:65536}));
  for(let block=0;block<200;block++){
    if(block%17===0){
      const ratesHz=Float32Array.from(ids,()=>random()*500),amplitudesMv=Float32Array.from(ids,()=>random()*100-20);
      for(const b of brains)b.setPoissonInputs({indices:ids,ratesHz,amplitudesMv});
    }
    if(block%13===0){const values=Float32Array.from(ids,()=>random()*40-20);for(const b of brains)b.setCurrentInputs(ids,values);}
    if(block%7===0){const values=Float32Array.from(ids,()=>random()*20-10);for(const b of brains)b.injectVoltage(ids,values);}
    if(block%31===0)for(const b of brains)b.setRefractoryPeriod(ids,block%2?0:2);
    const ms=(1+block%23)*(config.dtMs??.1);
    for(const b of brains)b.step(ms);
    for(const field of ['voltage','synapticDrive','spikeCount'])assert.deepEqual(brains[0].readActivations({field}),brains[1].readActivations({field}),`${JSON.stringify(config)} block ${block} ${field}`);
    assert.deepEqual(brains[0].readSpikes(),brains[1].readSpikes());comparisons++;
  }
  for(const b of brains)b.dispose();for(const g of graphs)g.dispose();
}
console.log(`Matched all neuron states and retained spike histories at ${comparisons} checkpoints across 8 signed/fractional graph and parameter combinations.`);
