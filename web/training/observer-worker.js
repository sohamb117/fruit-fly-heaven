import {createTrainingWorkerController} from './worker.js';
import {createBrainObserver} from './observer-runtime.js';
import {createEyeObserver} from './eye-observer-runtime.js';

// A separate presentation entry keeps the pinned controller/environment bytes
// intact. Only accepted evaluate() calls establish a preview's trial identity.
export function createObservedTrainingWorker({Brain,createEnvironment,postMessage,close=()=>{},now}){
 const observer=createBrainObserver({Brain,postMessage,now});
 const eyes=createEyeObserver({postMessage,now});let evaluationRequestId=null;
 const controller=createTrainingWorkerController({postMessage,close:()=>{observer.dispose();eyes.dispose();close();},
  createEnvironment:async(...args)=>{
   const environment=await createEnvironment(...args);
   return {...environment,
    ready(){observer.endJob();eyes.endJob();return environment.ready();},
    async evaluate(job,options){
     const token=observer.beginJob(job?.jobId),eyeToken=eyes.beginJob(job?.jobId,evaluationRequestId);
     try{return await environment.evaluate(job,{...options,onPhysicsStep(context){
      options.onPhysicsStep?.(context);eyes.observe(context);
     }});}finally{observer.endJob(token);eyes.endJob(eyeToken);}
    },
    dispose(){observer.endJob();eyes.endJob();return environment.dispose();},
   };
  },
 });
 return {
  async handle(message){
   if(message?.type==='observe-brain'){
    try{observer.configure(message);}catch(error){observer.report(error);}return;
   }
   if(message?.type==='observe-eyes'){
    try{eyes.configure(message);}catch(error){eyes.report(error);}return;
   }
   if(message?.type==='evaluate'){
    const state=controller.state;
    if(state.ready&&!state.initializing&&!state.busy&&!state.stopping)evaluationRequestId=message.id;
   }
   if(['cancel','stop'].includes(message?.type)){observer.endJob();eyes.endJob();}
   if(message?.type==='stop'){observer.dispose();eyes.dispose();}
   return controller.handle(message);
  },
  dispose(){observer.dispose();eyes.dispose();},
  get state(){return {...controller.state,observation:observer.state,eyeObservation:eyes.state};},
 };
}

if(typeof WorkerGlobalScope!=='undefined'&&globalThis instanceof WorkerGlobalScope){
 // Importing worker.js installs its default handler but performs no work. The
 // explicit controller below replaces that handler before messages are read.
 const queued=[];self.onmessage=event=>queued.push(event.data);
 const {WasmBrain}=await import('/banc-engine/src/wasm.js');
 const controller=createObservedTrainingWorker({Brain:WasmBrain,postMessage:(value,transfer)=>self.postMessage(value,transfer),close:()=>self.close(),
  createEnvironment:async(config,options)=>(await import('./sequential-environment.js')).createSequentialTrainingEnvironment(config,options)});
 const dispatch=message=>{controller.handle(message).catch(error=>self.postMessage({type:'error',message:error.stack||error.message,recoverable:false}));};
 self.onmessage=event=>dispatch(event.data);for(const message of queued)dispatch(message);
}
