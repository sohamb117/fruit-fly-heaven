import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
import {createBrainModule} from '../packages/fly-brain-wasm/dist/index.js';
const count=Number(process.argv[2]??100),duration=Number(process.argv[3]??5);
const data=new URL('../data/prepared/',import.meta.url);
const read=(name,Type)=>{const b=fs.readFileSync(new URL(name,data));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const meta=JSON.parse(fs.readFileSync(new URL('metadata.json',data))),groups=JSON.parse(fs.readFileSync(new URL('groups.json',data)));
const module=await createBrainModule();
const graph=module.createConnectome({neuronCount:meta.neurons_per_brain,rowOffsets:read('indptr.bin',Uint32Array),targets:read('targets.bin',Uint32Array),weights:read('weights.bin',Float32Array)});
let start=performance.now();const brains=graph.createPopulation(count,{seed:2026});const allocationMs=performance.now()-start;
const indices=Uint32Array.from([...groups.odor_left,...groups.odor_right,...groups.sweet]);
const ratesHz=Float32Array.from([...groups.odor_left.map(()=>35),...groups.odor_right.map(()=>45),...groups.sweet.map(()=>150)]);
for(const b of brains)b.setRefractoryPeriod(indices,0).setPoissonInputs({indices,ratesHz});
start=performance.now();for(const b of brains)b.step(duration);const wallMs=performance.now()-start;
start=performance.now();const matrix=module.readActivationMatrix(brains,{field:'voltage'});const matrixReadMs=performance.now()-start;
if(matrix.shape[0]!==count||matrix.shape[1]!==meta.neurons_per_brain||!matrix.values.every(Number.isFinite))throw new Error('Invalid full activation matrix');
const result={engine:'WASM / sequential Node',brains:count,neuronsPerBrain:meta.neurons_per_brain,connectionRows:meta.connection_rows,synapsesRepresented:meta.synapses_represented,
  simulatedMsPerBrain:duration,allocationMs,wallMs,speedVsRealTime:duration/wallMs,matrixShape:matrix.shape,matrixBytes:matrix.values.byteLength,matrixReadMs,
  wasmHeapBytes:module.allocatedHeapBytes,spikeCounts:brains.map(b=>b.totalSpikes),distinctSpikeCounts:new Set(brains.map(b=>b.totalSpikes)).size};
const filename=new URL(`../reports/wasm-${count}-brains.json`,import.meta.url);fs.writeFileSync(filename,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({...result,spikeCounts:undefined},null,2));
for(const b of brains)b.dispose();graph.dispose();
