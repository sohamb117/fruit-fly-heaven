// Full-connectome speed / activity tradeoff. Pause the live app while timing.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {createBrainModule} from '../packages/fly-brain-wasm/dist/index.js';
import {SIMULATION_MODES} from '../web/simulation-modes.js';
const root=new URL('../',import.meta.url),fields=['voltage','synapticDrive','spikeCount'];
const metadata=JSON.parse(fs.readFileSync(new URL('data/prepared/metadata.json',root)));
const groups=JSON.parse(fs.readFileSync(new URL('data/prepared/groups.json',root)));
const modes={reference:SIMULATION_MODES.reference,float32:{precision:'float32',parameters:{},dtMs:.1},fast:SIMULATION_MODES.fast};
const read=(name,Type)=>{const b=fs.readFileSync(new URL('data/prepared/'+name,root));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
if(!isMainThread){
  const mode=modes[workerData.mode],module=await createBrainModule({precision:mode.precision});
  const graph=module.createConnectome({neuronCount:metadata.neurons_per_brain,rowOffsets:read('indptr.bin',Uint32Array),targets:read('targets.bin',Uint32Array),weights:read('weights.bin',Float32Array)});
  const indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]);
  const ratesHz=Float32Array.from([...groups.odor_left.map(()=>35),...groups.odor_right.map(()=>45),...groups.sweet.map(()=>150)]);
  const brains=graph.createPopulation(workerData.count,{...mode.parameters,seed:2026+workerData.first});
  for(const brain of brains)brain.setRefractoryPeriod(indices,0).setPoissonInputs({indices,ratesHz});
  let before;
  parentPort.on('message',({op,ms})=>{
    if(op==='advance')for(let t=0;t<ms;t+=2)for(const brain of brains)brain.step(Math.min(2,ms-t));
    if(op==='startMeasurement')before=brains.map(b=>b.readActivations({field:'spikeCount'}));
    if(op==='state'){
      const state=Object.fromEntries(fields.map(field=>[field,new Float64Array(brains.length*graph.neuronCount)]));
      for(let k=0;k<brains.length;k++)for(const field of fields){
        const values=brains[k].readActivations({field});
        if(field==='spikeCount')for(let j=0;j<values.length;j++)values[j]-=before[k][j];
        if(!values.every(Number.isFinite))throw new Error(`${workerData.mode}: non-finite ${field}`);
        state[field].set(values,k*graph.neuronCount);
      }
      parentPort.postMessage({state,heapBytes:module.allocatedHeapBytes},Object.values(state).map(a=>a.buffer));return;
    }
    parentPort.postMessage({ok:true});
  });
  parentPort.postMessage({ready:true});
}else{
  const [count=100,warmupMs=100,durationMs=100,workerCount=4,trials=3]=process.argv.slice(2).map(Number);
  assert.ok([count,warmupMs,durationMs,workerCount,trials].every(Number.isInteger)&&count>0&&warmupMs>=0&&durationMs>0&&workerCount>0&&workerCount<=count&&trials>0,'Usage: node scripts/benchmark-precision.mjs [brains warmupMs durationMs workers trials]');
  const samples=[],digests={};let reference;
  const response=w=>new Promise((resolve,reject)=>{
    const clean=()=>{w.off('message',ok);w.off('error',bad);w.off('exit',exit);},ok=data=>{clean();resolve(data);},bad=error=>{clean();reject(error);},exit=code=>bad(new Error(`Worker exited (${code})`));
    w.once('message',ok);w.once('error',bad);w.once('exit',exit);
  });
  const command=(w,data)=>{const result=response(w);w.postMessage(data);return result;};
  for(let trial=1;trial<=trials;trial++)for(const mode of trial%2?Object.keys(modes):Object.keys(modes).reverse()){
    const workers=[];
    try{
      const ready=[];
      for(let k=0;k<workerCount;k++){
        const first=Math.floor(k*count/workerCount),end=Math.floor((k+1)*count/workerCount);
        const worker=new Worker(new URL(import.meta.url),{workerData:{mode,count:end-first,first}});workers.push(worker);ready.push(response(worker));
      }
      await Promise.all(ready);
      await Promise.all(workers.map(w=>command(w,{op:'advance',ms:warmupMs})));
      await Promise.all(workers.map(w=>command(w,{op:'startMeasurement'})));
      const start=performance.now();await Promise.all(workers.map(w=>command(w,{op:'advance',ms:durationMs})));const wallMs=performance.now()-start;
      const result=await Promise.all(workers.map(w=>command(w,{op:'state'})));
      if(mode==='reference')reference??=result;
      const hash=createHash('sha256');for(const worker of result)for(const field of fields)hash.update(new Uint8Array(worker.state[field].buffer));
      const stateSha256=hash.digest('hex');if(digests[mode])assert.equal(stateSha256,digests[mode],`${mode} must repeat deterministically`);digests[mode]=stateSha256;
      const differences={};
      for(const field of fields){
        let absolute=0,squared=0,max=0,referenceSum=0,total=0,different=0,n=0;
        for(let w=0;w<result.length;w++){
          const actual=result[w].state[field],baseline=reference[w].state[field];
          for(let i=0;i<actual.length;i++){
            const delta=actual[i]-baseline[i];absolute+=Math.abs(delta);squared+=delta*delta;max=Math.max(max,Math.abs(delta));referenceSum+=baseline[i];total+=actual[i];different+=delta!==0;n++;
          }
        }
        differences[field]={meanAbsolute:absolute/n,rootMeanSquared:Math.sqrt(squared/n),maxAbsolute:max,fractionDifferent:different/n};
        if(field==='spikeCount')Object.assign(differences[field],{total,referenceTotal:referenceSum,relativeTotalDifference:(total-referenceSum)/Math.max(1,referenceSum),normalizedL1:absolute/Math.max(1,referenceSum)});
      }
      const readouts={};
      for(const name of ['walk','steer_left','steer_right','feed']){
        let actual=0,baseline=0;
        for(let w=0;w<result.length;w++)for(let offset=0;offset<result[w].state.spikeCount.length;offset+=metadata.neurons_per_brain)for(const index of groups[name]){
          actual+=result[w].state.spikeCount[offset+index];baseline+=reference[w].state.spikeCount[offset+index];
        }
        const scale=1000/(durationMs*count*groups[name].length);readouts[name]={meanHz:actual*scale,referenceMeanHz:baseline*scale};
      }
      const sample={mode,trial,wallMs,cohortSpeed:durationMs/wallMs,wasmHeapBytes:result.reduce((n,w)=>n+w.heapBytes,0),stateSha256,differences,readouts};samples.push(sample);
      console.log(JSON.stringify({mode,trial,wallMs,speed:sample.cohortSpeed,spikeChangePercent:100*differences.spikeCount.relativeTotalDifference,voltageMaeMv:differences.voltage.meanAbsolute}));
    }finally{await Promise.all(workers.map(w=>w.terminate()));}
  }
  const median=a=>{const b=a.toSorted((x,y)=>x-y),i=Math.floor(b.length/2);return b.length%2?b[i]:(b[i-1]+b[i])/2;};
  const medianWallMs=Object.fromEntries(Object.keys(modes).map(mode=>[mode,median(samples.filter(s=>s.mode===mode).map(s=>s.wallMs))]));
  const sha=name=>createHash('sha256').update(fs.readFileSync(new URL(name,root))).digest('hex');
  const report={date:new Date().toISOString(),runtime:process.version,platform:process.platform,arch:process.arch,brains:count,workers:workerCount,warmupMs,durationMs,trials,stepBlockMs:2,seed:2026,inputsHz:{odorLeft:35,odorRight:45,sweet:150},neuronsPerBrain:metadata.neurons_per_brain,connectionRows:metadata.connection_rows,modes,coreSha256:{float64:sha('packages/fly-brain-wasm/dist/core.wasm'),float32:sha('packages/fly-brain-wasm/dist/core-f32.wasm')},graphSha256:Object.fromEntries(['indptr.bin','targets.bin','weights.bin'].map(name=>[name,sha('data/prepared/'+name)])),scope:'Neural stepping only; live app paused; loading, warm-up, readouts, drift comparisons and rendering excluded. Spike-count differences cover the measured window. Voltage and drive differences are snapshots at its end. Different timestep grids also change discretized Poisson input trains; matching seeds do not imply identical stimuli between reference and fast modes. This is not biological validation.',samples,medianWallMs,speedup:Object.fromEntries(Object.keys(modes).map(mode=>[mode,medianWallMs.reference/medianWallMs[mode]]))};
  const output=new URL(`reports/wasm-precision-${count}.json`,root);fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({medianWallMs,speedup:report.speedup}));
}
