import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';

// Read-only production audit. Run the actual worker scheduling/message loop
// with deterministic fake asynchronous brains.
// Neural equations and rendering are deliberately outside this concurrency test.
const fullSource=await fs.readFile('web/banc-world-worker.js','utf8');
const start=fullSource.indexOf('async function run(){'),end=fullSource.indexOf("self.addEventListener('unhandledrejection'");
assert(start>=0&&end>start);
const production=fullSource.slice(start,end);

const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(predicate,label){
 const deadline=performance.now()+2000;
 while(!predicate()){assert(performance.now()<deadline,`Timed out: ${label}`);await sleep(1);}
}
function harness(source,{count=1,vision=true,pauseDuringFirstStep=false}={}){
 const messages=[],context=vm.createContext({console,performance,setTimeout,Float32Array,Float64Array,Map,Math,
  __messages:messages,__vision:vision,__count:count,__pauseDuringFirstStep:pauseDuringFirstStep});
 const prelude=`
 const self={postMessage:message=>__messages.push(structuredCloneForAudit(message))};
 function structuredCloneForAudit(message){return { ...message,flies:message.flies?.map(f=>({...f,brain:{...f.brain}}))};}
 const sleep=ms=>new Promise(resolve=>setTimeout(resolve,Math.min(ms,1)));
 let paused=false,running=true,requestedInspection=false,bodySynchronized=true,lastFeedbackMs=0,resolveFeedback=null;
 let selected=-1,selectedNeuron=0,diagnosticFly=null,dt=.5,poses=[];
 let switches={vision:__vision,odor:true,taste:true,bodySense:true},eyes=new Map();
 const currents=()=>{},applyPoses=()=>{},init=()=>{};
 let fired=false;
 let brains=Array.from({length:__count},(_,index)=>({
  brain:{timeMs:0,async step(){
   if(__pauseDuringFirstStep&&!fired){fired=true;self.onmessage({data:{type:'control',paused:true}});await sleep(1);}
   this.timeMs+=2;return new Float32Array(32);
  }},
  fly:{id:index+1,brain:{time_ms:0}},encoder:{sample:{food:[0,0,0]},update(){return null;}},
  graded:{update(){}},color:{update(){}},input:new Float32Array(1),lastInspection:-Infinity,trace:[]
 }));
 const byId=new Map(brains.map(item=>[item.fly.id,item]));
 async function inspect(item,full){
  item.fly.brain={time_ms:item.brain.timeMs};
  return full?{selectedId:item.fly.id,activation:new Float64Array(1),lastSpikeMs:new Float64Array(1)}:null;
 }
 `;
 const postlude=`
 globalThis.audit={
  run,send:data=>self.onmessage({data}),times:()=>brains.map(item=>item.brain.timeMs),
  pending:()=>!!resolveFeedback,feedback:()=>lastFeedbackMs,
  stop(){running=false;if(resolveFeedback){const wake=resolveFeedback;resolveFeedback=null;wake();}}
 };
 `;
 vm.runInContext(prelude+source+postlude,context);
 const api=context.audit;
 const eye=id=>api.send({type:'eyes',frames:[{id,sequence:0,pixels:new Uint8Array(1024)}]});
 return {api,messages,eye};
}
async function runCase(source,options,fn){
 const h=harness(source,options);let pending;
 try{pending=h.api.run();await fn(h);}
 finally{h.api.stop();await pending;}
}
const results=[];
await runCase(production,{count:2},async({api,eye})=>{
 eye(1);await sleep(15);assert.deepEqual(Array.from(api.times()),[0,0]);
 eye(2);await until(()=>api.pending(),'complete initial frames');
 assert.deepEqual(Array.from(api.times()),[2,2]);results.push('Production loop waits for all initial local eye frames');
});
await runCase(production,{},async({api,eye,messages})=>{
 eye(1);await until(()=>api.pending(),'production pose wait');
 api.send({type:'control',paused:true,selected:1});
 await until(()=>messages.some(m=>m.inspectionOnly===true),'paused read-only inspection');
 assert.equal(api.feedback(),0);assert.equal(api.pending(),false);assert.deepEqual(Array.from(api.times()),[2]);
 api.send({type:'control',paused:false});await until(()=>api.pending(),'resume waits for real pose');
 api.send({type:'poses',poses:[],neuralTimeMs:1});await sleep(5);assert.deepEqual(Array.from(api.times()),[2]);
 api.send({type:'poses',poses:[],neuralTimeMs:2});await until(()=>api.times()[0]===4,'real pose acknowledgment resumes');
 results.push('Pause releases the wait for inspection; resume and stale poses cannot fake acknowledgment');
});
await runCase(production,{count:2,vision:false},async({api})=>{
 await until(()=>api.pending(),'vision disconnected');assert.deepEqual(Array.from(api.times()),[2,2]);
 results.push('Vision disconnected permits synchronized batches without any eye frames');
});
await runCase(production,{count:2},async({api})=>{
 await sleep(5);assert.deepEqual(Array.from(api.times()),[0,0]);
 api.send({type:'control',vision:false});await until(()=>api.pending(),'vision toggle releases eye gate');
 assert.deepEqual(Array.from(api.times()),[2,2]);results.push('Disabling vision releases an existing initial-eye wait');
});
await runCase(production,{count:2,pauseDuringFirstStep:true},async({api,eye})=>{
 eye(1);eye(2);await until(()=>api.times()[1]===2,'finish started batch after pause');await sleep(5);
 assert.deepEqual(Array.from(api.times()),[2,2]);assert.equal(api.feedback(),0);
 results.push('A started 2 ms batch completes for every local fly before pausing');
});
await runCase(production,{},async({api,eye})=>{
 eye(1);await until(()=>api.pending(),'synchronized pose wait');
 api.send({type:'control',bodySynchronized:false});await until(()=>api.times()[0]>=4,'clock switch releases pose wait');
 results.push('Disabling body synchronization releases a pending wait without changing acknowledgment');
});
// Execute the actual app message handler as well. An inspector response can
// arrive while its brain is ahead of the mechanical clock; it must not turn
// that read-only response into a mechanical step or fabricated pose ACK.
const app=await fs.readFile('web/app.js','utf8');
const advanceStart=app.indexOf('function advanceNeuralBodies(){'),advanceEnd=app.indexOf('function render(',advanceStart);
const handlerStart=app.indexOf('worker.onmessage=')+'worker.onmessage='.length,handlerEnd=app.indexOf('    worker.onerror=',handlerStart);
assert(advanceStart>=0&&advanceEnd>advanceStart&&handlerEnd>handlerStart);
const bodySteps=[],poseAcks=[];
const host=vm.createContext({performance,Math,
 generation:1,workerGeneration:1,snapshot:{flies:[{id:1,brain:{time_ms:0,spikes:0}}],time_ms:0,paused:true},
 readyWorkers:1,totalWorkers:1,banc:true,bodyClock:'neural',lastMotionNeuralMs:0,
 bodyWorld:{advance:seconds=>bodySteps.push(seconds)},sendBodyPoses:()=>poseAcks.push(true),
 applyNeuralOutput:(target,source)=>Object.assign(target,source),control:()=>{},updatePanel:()=>{},selected:-1,
 startedWall:performance.now(),pauseStarted:0,pausedWall:0,
 failSimulation:error=>{throw error;},showError:message=>{throw new Error(message);}
});
vm.runInContext(app.slice(advanceStart,advanceEnd)+'\nglobalThis.receive='+app.slice(handlerStart,handlerEnd),host);
const payload={type:'update',flies:[{id:1,brain:{time_ms:2,spikes:0}}]};
host.receive({data:{...payload,inspectionOnly:true}});
assert.equal(host.snapshot.time_ms,2);assert.equal(bodySteps.length,0);assert.equal(poseAcks.length,0);
host.receive({data:payload});
assert.deepEqual(bodySteps,[.002]);assert.equal(poseAcks.length,1);
results.push('Actual app handler keeps paused inspection read-only; a real completed batch advances and acknowledges the body');
const report={date:new Date().toISOString(),scope:'Actual current worker run/message loop with fake asynchronous brains; no production source rewriting',
 results,productionChanges:false,limitation:'VM scheduling and message-handler checks do not exercise browser GPU execution or rendering',passed:true};
await fs.writeFile('reports/banc-worker-barrier-audit.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
