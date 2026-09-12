import {createBrainViewModule} from './index.js';
let runtime=null,next=1,queue=Promise.resolve();
const objects=new Map();
function object(id){if(!objects.has(id))throw new Error(`Unknown object ${id}`);return objects.get(id);}
async function handle({op,args={}}){
  if(op==='init'){if(runtime)throw new Error('Worker is already initialized');runtime=await createBrainViewModule(args);return {ready:true};}
  if(!runtime)throw new Error('Call init first');
  if(op==='createGeometry'||op==='createVolume'){const id=next++;objects.set(id,runtime[op](args));return {id};}
  if(op==='sliceMesh')return runtime.sliceMesh(args);
  if(op==='dispose'){object(args.id).dispose();objects.delete(args.id);return null;}
  if(['updateActivity','filter','pick','samplePlane'].includes(op))return object(args.id)[op](args.options);
  throw new Error(`Unknown operation ${op}`);
}
function transfers(value){return ArrayBuffer.isView(value)?[value.buffer]:[];}
self.onmessage=({data})=>{queue=queue.then(async()=>{try{const result=await handle(data);self.postMessage({requestId:data.requestId,result},transfers(result));}catch(error){self.postMessage({requestId:data?.requestId,error:String(error.message??error)});}});};
