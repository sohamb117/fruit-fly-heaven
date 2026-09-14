import {createTrainingWorkerController} from './worker.js';
import {createBrainObserver} from './observer-runtime.js';

// A separate presentation entry keeps the pinned controller/environment bytes
// intact. Only accepted evaluate() calls establish a preview's trial identity.
export function createObservedTrainingWorker({Brain,createEnvironment,postMessage,close=()=>{},now}){
 const observer=createBrainObserver({Brain,postMessage,now});
 const controller=createTrainingWorkerController({postMessage,close:()=>{observer.dispose();close();},
  createEnvironment:async(...args)=>{
   const environment=await createEnvironment(...args);
   return {...environment,
    ready(){observer.endJob();return environment.ready();},
    async evaluate(job,options){
     const token=observer.beginJob(job?.jobId);
     try{return await environment.evaluate(job,options);}finally{observer.endJob(token);}
    },
    dispose(){observer.endJob();return environment.dispose();},
   };
  },
 });
 return {
  async handle(message){
   if(message?.type==='observe-brain'){
    try{observer.configure(message);}catch(error){observer.report(error);}return;
   }
   if(['cancel','stop'].includes(message?.type))observer.endJob();
   if(message?.type==='stop')observer.dispose();
   return controller.handle(message);
  },
  dispose(){observer.dispose();},
  get state(){return {...controller.state,observation:observer.state};},
 };
}

if(typeof WorkerGlobalScope!=='undefined'&&globalThis instanceof WorkerGlobalScope){
 // Importing worker.js installs its default handler but performs no work. The
 // explicit controller below replaces that handler before messages are read.
 const queued=[];self.onmessage=event=>queued.push(event.data);
 const {WasmBrain}=await import('/banc-engine/src/wasm.js');
 const controller=createObservedTrainingWorker({Brain:WasmBrain,postMessage:(value,transfer)=>self.postMessage(value,transfer),close:()=>self.close(),
  createEnvironment:async(config,options)=>(await import('./environment.js')).createTrainingEnvironment(config,options)});
 const dispatch=message=>{controller.handle(message).catch(error=>self.postMessage({type:'error',message:error.stack||error.message,recoverable:false}));};
 self.onmessage=event=>dispatch(event.data);for(const message of queued)dispatch(message);
}
