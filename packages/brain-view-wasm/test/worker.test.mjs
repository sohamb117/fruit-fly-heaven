import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';
import {once} from 'node:events';
test('Worker transfers volume slices and geometry colors and rejects repeated init',async()=>{
  const worker=new Worker(new URL('./worker-host.mjs',import.meta.url));
  try{
    await once(worker,'message');let next=1;const pending=new Map();
    worker.on('message',m=>{const p=pending.get(m.requestId);if(p){pending.delete(m.requestId);m.error?p.reject(new Error(m.error)):p.resolve(m.result);}});
    const call=(op,args={})=>new Promise((resolve,reject)=>{const requestId=next++;pending.set(requestId,{resolve,reject});worker.postMessage({requestId,op,args});});
    await call('init');await assert.rejects(call('init'),/already initialized/);
    const volume=await call('createVolume',{values:new Uint8Array([10,20,30,40,50,60,70,80]),dimensions:[2,2,2]});
    assert.deepEqual([...await call('samplePlane',{id:volume.id,options:{origin:[0,0,1],u:[1,0,0],v:[0,1,0],width:2,height:2}})],[50,60,70,80]);
    const geometry=await call('createGeometry',{positions:new Float32Array([0,0,0]),neuronIndices:new Uint32Array([0]),neuronCount:1});
    const colors=await call('updateActivity',{id:geometry.id,options:{voltage:new Float64Array([-45])}});assert(colors instanceof Float32Array);assert.equal(colors[0],1);
    await call('dispose',{id:volume.id});await call('dispose',{id:geometry.id});await assert.rejects(call('filter',{id:geometry.id}),/Unknown object/);
  }finally{await worker.terminate();}
});
