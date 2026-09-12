// Compare the shipped 0.1.1 engine with the current build, on the same workload.
// Pause the live app and other CPU-heavy work before running this benchmark.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {performance} from 'node:perf_hooks';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
const root=new URL('../',import.meta.url);
const read=(name,Type)=>{const b=fs.readFileSync(new URL('data/prepared/'+name,root));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};

if(!isMainThread){
  const {createBrainModule}=await import(workerData.moduleUrl);
  const meta=JSON.parse(fs.readFileSync(new URL('data/prepared/metadata.json',root)));
  const groups=JSON.parse(fs.readFileSync(new URL('data/prepared/groups.json',root)));
  const module=await createBrainModule(),graph=module.createConnectome({neuronCount:meta.neurons_per_brain,rowOffsets:read('indptr.bin',Uint32Array),targets:read('targets.bin',Uint32Array),weights:read('weights.bin',Float32Array)});
  const indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]);
  const ratesHz=Float32Array.from([...groups.odor_left.map(()=>35),...groups.odor_right.map(()=>45),...groups.sweet.map(()=>150)]);
  let brains=[];
  parentPort.on('message',({op,ms})=>{
    if(op==='reset'){
      for(const brain of brains)brain.dispose();
      brains=graph.createPopulation(workerData.count,{seed:2026+workerData.first});
      for(const brain of brains)brain.setRefractoryPeriod(indices,0).setPoissonInputs({indices,ratesHz});
    }else if(op==='advance'){
      // Match the browser's round-robin 2 ms scheduling, including cache churn.
      for(let t=0;t<ms;t+=2)for(const brain of brains)brain.step(Math.min(2,ms-t));
    }else if(op==='state'){
      const hashes=[],spikes=[];
      for(const brain of brains){
        const hash=createHash('sha256');
        for(const field of ['voltage','synapticDrive','spikeCount']){
          const values=brain.readActivations({field});
          if(!values.every(Number.isFinite))throw new Error('Non-finite activation');
          hash.update(new Uint8Array(values.buffer));
        }
        const history=brain.readSpikes();
        hash.update(new Uint8Array(history.timesMs.buffer));hash.update(new Uint8Array(history.neuronIndices.buffer));
        hashes.push(hash.digest('hex'));spikes.push(brain.totalSpikes);
      }
      parentPort.postMessage({hashes,spikes,heapBytes:module.allocatedHeapBytes});return;
    }
    parentPort.postMessage({ok:true});
  });
  parentPort.postMessage({ready:true});
}else{
  const [count=100,warmupMs=100,measuredNeuralMs=100,workerCount=4,trials=3]=process.argv.slice(2).map(Number);
  if(![count,warmupMs,measuredNeuralMs,workerCount,trials].every(Number.isInteger)||count<1||warmupMs<0||measuredNeuralMs<1||workerCount<1||workerCount>count||trials<1)throw new Error('Use: node scripts/benchmark-brain-performance.mjs [brains warmupMs measuredMs workers trials]');
  const baseline=new URL('build/performance/release-baseline/',root);fs.mkdirSync(baseline,{recursive:true});
  execFileSync('tar',['-xzf',new URL('releases/fruit-fly-brain-wasm-0.1.1.tgz',root).pathname,'-C',baseline.pathname]);
  const engines=[{name:'0.1.1',moduleUrl:new URL('package/dist/index.js',baseline).href},{name:'current',moduleUrl:new URL('packages/fly-brain-wasm/dist/index.js',root).href}];
  const samples=[];let referenceHashes;
  async function run(engine,trial){
    const workers=[];
    const response=w=>new Promise((resolve,reject)=>{const clean=()=>{w.off('message',ok);w.off('error',bad);w.off('exit',exit);};const ok=m=>{clean();resolve(m);};const bad=e=>{clean();reject(e);};const exit=code=>bad(new Error(`Worker exited before responding (${code})`));w.once('message',ok);w.once('error',bad);w.once('exit',exit);});
    const command=(w,msg)=>{const result=response(w);w.postMessage(msg);return result;};
    try{
      const ready=[];
      for(let k=0;k<workerCount;k++){
        const first=Math.floor(k*count/workerCount),end=Math.floor((k+1)*count/workerCount);
        const w=new Worker(new URL(import.meta.url),{workerData:{moduleUrl:engine.moduleUrl,first,count:end-first}});workers.push(w);ready.push(response(w));
      }
      await Promise.all(ready);
      await Promise.all(workers.map(w=>command(w,{op:'reset'})));
      await Promise.all(workers.map(w=>command(w,{op:'advance',ms:warmupMs})));
      const start=performance.now();await Promise.all(workers.map(w=>command(w,{op:'advance',ms:measuredNeuralMs})));const wallMs=performance.now()-start;
      const states=await Promise.all(workers.map(w=>command(w,{op:'state'})));
      const hashes=states.flatMap(s=>s.hashes),spikeCounts=states.flatMap(s=>s.spikes);
      if(referenceHashes && JSON.stringify(hashes)!==JSON.stringify(referenceHashes))throw new Error(`Trajectory mismatch: ${engine.name}, trial ${trial}`);
      referenceHashes??=hashes;
      const sample={engine:engine.name,trial,wallMs,cohortSpeed:measuredNeuralMs/wallMs,wasmHeapBytes:states.reduce((sum,s)=>sum+s.heapBytes,0),totalSpikes:spikeCounts.reduce((a,b)=>a+b,0),stateSha256:createHash('sha256').update(hashes.join('\n')).digest('hex')};
      samples.push(sample);console.log(JSON.stringify(sample));
    }finally{await Promise.all(workers.map(w=>w.terminate()));}
  }
  // Alternate order to reduce warm-machine/order bias. Never run A/B together.
  for(let trial=1;trial<=trials;trial++)for(const engine of trial%2?engines:[...engines].reverse())await run(engine,trial);
  const median=values=>{const sorted=values.toSorted((a,b)=>a-b),m=Math.floor(sorted.length/2);return sorted.length%2?sorted[m]:(sorted[m-1]+sorted[m])/2;};
  const medians=Object.fromEntries(engines.map(e=>[e.name,median(samples.filter(s=>s.engine===e.name).map(s=>s.wallMs))]));
  const checksum=path=>createHash('sha256').update(fs.readFileSync(new URL(path,root))).digest('hex');
  const metadata=JSON.parse(fs.readFileSync(new URL('data/prepared/metadata.json',root)));
  const provenance={currentVersion:JSON.parse(fs.readFileSync(new URL('packages/fly-brain-wasm/package.json',root))).version,currentWasmSha256:checksum('packages/fly-brain-wasm/dist/core.wasm'),baselineArchiveSha256:checksum('releases/fruit-fly-brain-wasm-0.1.1.tgz'),neuronsPerBrain:metadata.neurons_per_brain,connectionRows:metadata.connection_rows,graphSha256:Object.fromEntries(['indptr.bin','targets.bin','weights.bin'].map(name=>[name,checksum('data/prepared/'+name)]))};
  const report={date:new Date().toISOString(),runtime:process.version,platform:process.platform,arch:process.arch,provenance,brains:count,workers:workerCount,warmupMs,measuredNeuralMs,stepBlockMs:2,trials,inputsHz:{odorLeft:35,odorRight:45,sweet:150},seed:2026,scope:'Full measured graph; round-robin neural stepping only. Excludes graph loading, rendering, sensory updates and activation transfers.',correctness:'Bit-identical final voltage, synaptic drive, cumulative neuron spike counts and retained spike histories for every brain in every trial.',samples,medianWallMs:medians,speedup:medians['0.1.1']/medians.current};
  fs.writeFileSync(new URL('reports/wasm-performance.json',root),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({medianWallMs:medians,speedup:report.speedup}));
}
