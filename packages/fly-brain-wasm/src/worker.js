// Browser Worker protocol. Every response has the request id; errors are explicit.
// A worker loads one module, then any number of connectomes and brain instances.
import {createBrainModule,loadConnectome} from './index.js';
let runtime=null,nextId=1,queue=Promise.resolve();
const graphs=new Map(),brains=new Map();
function find(map,id,name){if(!map.has(id))throw new Error(`Unknown ${name}: ${id}`);return map.get(id);}
async function handle({op,args={}}){
  if(op==='init'){
    if(runtime)throw new Error('Worker already initialized');
    runtime=await createBrainModule(args);return {ready:true,precision:runtime.precision};
  }
  if(!runtime)throw new Error('Call init before other operations');
  if(op==='loadConnectome'){
    const data=args.csr??await loadConnectome(args.url),graph=runtime.createConnectome(data),id=nextId++;
    graphs.set(id,graph);return {id,neuronCount:graph.neuronCount,edgeCount:graph.edgeCount,metadata:data.metadata};
  }
  if(op==='createPopulation'){
    const graph=find(graphs,args.connectomeId,'connectome');
    return graph.createPopulation(args.count,args.options).map(brain=>{const id=nextId++;brains.set(id,brain);return id;});
  }
  if(op==='matrix')return runtime.readActivationMatrix(args.brainIds.map(id=>find(brains,id,'brain')),args.options);
  if(op==='stepMany')return args.brainIds.map(id=>{const b=find(brains,id,'brain');b.step(args.durationMs);return {id,timeMs:b.timeMs,totalSpikes:b.totalSpikes};});
  if(op==='disposeConnectome'){find(graphs,args.id,'connectome').dispose();graphs.delete(args.id);return null;}
  const brain=find(brains,args.id,'brain');
  if(op==='setPoissonInputs'){brain.setPoissonInputs(args.input);return null;}
  if(op==='setCurrentInputs'){brain.setCurrentInputs(args.indices,args.values);return null;}
  if(op==='injectVoltage'){brain.injectVoltage(args.indices,args.values);return null;}
  if(op==='setRefractoryPeriod'){brain.setRefractoryPeriod(args.indices,args.ms);return null;}
  if(op==='step'){brain.step(args.durationMs);return {timeMs:brain.timeMs,totalSpikes:brain.totalSpikes};}
  if(op==='activations')return brain.readActivations(args.options);
  if(op==='spikes')return brain.readSpikes();
  if(op==='disposeBrain'){brain.dispose();brains.delete(args.id);return null;}
  throw new Error(`Unknown operation: ${op}`);
}
function transfers(value,set=new Set()){
  if(ArrayBuffer.isView(value))set.add(value.buffer);
  else if(value&&typeof value==='object')for(const v of Object.values(value))transfers(v,set);
  return [...set];
}
self.onmessage=event=>{
  const request=event.data;
  queue=queue.then(async()=>{
    try{const result=await handle(request);self.postMessage({requestId:request.requestId,result},transfers(result));}
    catch(error){self.postMessage({requestId:request.requestId,error:error.message});}
  });
};
