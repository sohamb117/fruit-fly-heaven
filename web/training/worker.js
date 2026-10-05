// Dedicated one-fly training worker. The UI controls scheduling, never actuators.
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
export function createTrainingWorkerController({postMessage,createEnvironment,close=()=>{}}){
 let environment=null,initializing=false,busy=false,paused=false,cancelled=false,stopping=false,generation=0;
 let budget={previewHz:6,dutyCycle:.65};
 const setBudget=message=>{
  if(message.previewHz!==undefined){if(!Number.isFinite(message.previewHz)||message.previewHz<0||message.previewHz>10)throw new Error('previewHz must be between 0 and 10');budget.previewHz=message.previewHz;}
  if(message.dutyCycle!==undefined){if(!Number.isFinite(message.dutyCycle)||message.dutyCycle<.05||message.dutyCycle>1)throw new Error('dutyCycle must be between 0.05 and 1');budget.dutyCycle=message.dutyCycle;}
 };
 const emit=(type,extra={})=>postMessage({type,...extra});
 const dispose=()=>{environment?.dispose();environment=null;};
 const checkpoint=async()=>{
  // Yield every physical feedback block, including WASM fallback, so messages
  // are processed without waiting for an episode to finish.
  await delay(0);
  while(paused&&!cancelled&&!stopping)await delay(20);
  if(cancelled||stopping){const error=new Error('Evaluation cancelled');error.name='AbortError';throw error;}
 };
 return {async handle(message){
  const {type,id}=message||{};
  if(type==='budget'){try{setBudget(message);emit('budget',{id,...budget});}catch(error){emit('error',{id,message:error.message,recoverable:true});}return;}
  if(type==='pause'){paused=true;emit('paused',{id,busy});return;}
  if(type==='resume'){paused=false;emit('resumed',{id,busy});return;}
  if(type==='cancel'){cancelled=true;paused=false;emit('cancelling',{id,busy});return;}
  if(type==='stop'){
   stopping=true;cancelled=true;paused=false;generation++;
   if(!busy&&!initializing){dispose();emit('stopped',{id});close();}
   return;
  }
  if(type==='initialize'){
   if(busy||initializing){emit('error',{id,message:'Worker is busy',recoverable:true});return;}
   initializing=true;cancelled=false;stopping=false;paused=false;const token=++generation;
   try{
    dispose();environment=await createEnvironment(message.config,{configUrl:message.configUrl,onProgress:value=>emit('progress',{id,phase:'initializing',...value})});
    if(token!==generation||stopping){dispose();emit('stopped',{id});close();return;}
    const initial=await environment.ready();
    if(token!==generation||stopping){dispose();emit('stopped',{id});close();return;}
    emit('ready',{id,...initial});
   }catch(error){dispose();emit('error',{id,message:error.stack||error.message,recoverable:false});}
   finally{initializing=false;}
   return;
  }
  if(type==='evaluate'){
   if(!environment||initializing||busy||stopping){emit('error',{id,message:'Training environment is not ready for evaluation',recoverable:true});return;}
   busy=true;cancelled=false;
   try{
    setBudget(message);
    const result=await environment.evaluate(message.job,{...budget,getBudget:()=>({...budget}),checkpoint,
     onFrame:frame=>emit('frame',{id,frame}),onProgress:progress=>emit('progress',{id,phase:'evaluating',...progress})});
    emit('evaluation',{id,result});
   }catch(error){
    if(error.name==='AbortError')emit('evaluation',{id,result:{return:0,success:false,terminated:false,truncated:true,reason:'cancelled',cancelled:true,simSeconds:0,steps:0,metrics:{}}});
    else{dispose();emit('error',{id,message:error.stack||error.message,recoverable:false});}
   }finally{
    busy=false;
    if(stopping){dispose();emit('stopped',{id});close();}
   }
   return;
  }
  emit('error',{id,message:'Unknown training worker message: '+String(type),recoverable:true});
 },get state(){return {ready:!!environment,initializing,busy,paused,cancelled,stopping,budget:{...budget}};}};
}

if(typeof WorkerGlobalScope!=='undefined'&&globalThis instanceof WorkerGlobalScope){
 const controller=createTrainingWorkerController({postMessage:value=>self.postMessage(value),close:()=>self.close(),
  createEnvironment:async(config,options)=>(await import('./sequential-environment.js')).createSequentialTrainingEnvironment(config,options)});
 self.onmessage=event=>{controller.handle(event.data).catch(error=>self.postMessage({type:'error',message:error.stack||error.message,recoverable:false}));};
}
