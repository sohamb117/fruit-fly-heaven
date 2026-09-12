import {parentPort} from 'node:worker_threads';
globalThis.self={postMessage:(data,transfer)=>parentPort.postMessage(data,transfer),onmessage:null};
await import('../dist/worker.js');
parentPort.on('message',data=>self.onmessage({data}));
parentPort.postMessage({ready:true});
