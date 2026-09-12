import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';
import {once} from 'node:events';
test('Worker protocol loads a graph, steps 100 brains and transfers a matrix',async()=>{
  const worker=new Worker(new URL('./worker-host.mjs',import.meta.url));
  try{
    await once(worker,'message');let next=1;const pending=new Map();
    worker.on('message',m=>{const pair=pending.get(m.requestId);if(pair){pending.delete(m.requestId);m.error?pair.reject(new Error(m.error)):pair.resolve(m.result);}});
    const call=(op,args={})=>new Promise((resolve,reject)=>{const requestId=next++;pending.set(requestId,{resolve,reject});worker.postMessage({requestId,op,args});});
    await call('init');
    const g=await call('loadConnectome',{csr:{neuronCount:2,rowOffsets:new Uint32Array([0,1,1]),targets:new Uint32Array([1]),weights:new Float32Array([250])}});
    const ids=await call('createPopulation',{connectomeId:g.id,count:100,options:{seed:8}});assert.equal(ids.length,100);
    await call('injectVoltage',{id:ids[0],indices:new Uint32Array([0]),values:new Float32Array([20])});
    await call('stepMany',{brainIds:ids,durationMs:20});
    const matrix=await call('matrix',{brainIds:ids,options:{field:'spikeCount'}});
    assert.deepEqual(matrix.shape,[100,2]);assert.ok(matrix.values instanceof Float64Array);
    assert.ok(matrix.values[0]>0);assert.equal(matrix.values[2],0);
    await assert.rejects(call('step',{id:-123,durationMs:10}),/Unknown brain/);
    for(const id of ids)await call('disposeBrain',{id});await call('disposeConnectome',{id:g.id});
  }finally{await worker.terminate();}
});
