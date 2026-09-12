import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
import {createBrainViewModule} from '../packages/brain-view-wasm/dist/index.js';
const base=new URL('../data/anatomy/',import.meta.url);
const read=(name,Type)=>{const b=fs.readFileSync(new URL(name,base));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const meta=JSON.parse(fs.readFileSync(new URL('metadata.json',base)));
const module=await createBrainViewModule();
const geometry=module.createGeometry({positions:read('skeleton-positions.bin',Float32Array),neuronIndices:read('skeleton-neurons.bin',Uint32Array),neuronCount:meta.neuronCount});
const points=module.createGeometry({positions:read('positions.bin',Float32Array),neuronIndices:read('neuron-indices.bin',Uint32Array),neuronCount:meta.neuronCount});
const volume=module.createVolume({values:read('em-volume.bin',Uint8Array),dimensions:meta.volume.dimensions});
const voltage=Float64Array.from({length:meta.neuronCount},(_,i)=>-60+(i%20));
function measure(fn){fn();const times=[];for(let i=0;i<5;i++){const start=performance.now();fn();times.push(performance.now()-start);}return times.sort((a,b)=>a-b)[2];}
const report={engine:'SIMD WASM / Node',geometry:'Measured FlyWire v783; synthetic voltage ramp for kernel timing only',neurons:meta.neuronCount,mappedNeurons:meta.mappedNeurons,skeletonVertices:meta.skeletonVertices,volumeVoxels:meta.volume.dimensions.reduce((a,b)=>a*b),
  activityMapMedianMs:measure(()=>geometry.updateActivity({voltage})),
  slabFilterMedianMs:measure(()=>points.filter({normal:[0,0,1],offset:150,halfThickness:6,mode:'slab'})),
  volumeSliceMedianMs:measure(()=>volume.samplePlane({origin:[0,0,100],u:[.75,0,0],v:[0,.75,0],width:640,height:360})),
  wasmHeapBytes:module.allocatedHeapBytes};
fs.writeFileSync(new URL('../reports/wasm-view-benchmark.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));geometry.dispose();points.dispose();volume.dispose();
