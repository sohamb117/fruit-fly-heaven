// Actual retinal/compact-motion math plus mocked worker orchestration.
// No neural/body simulation, camera redraw for display, or coordinator I/O.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createEyeObserver,EYE_SAMPLE_INTERVAL_MS} from '../training/eye-observer-runtime.js';
import {createObservedTrainingWorker} from '../training/observer-worker.js';
import {createRetinalSensor} from '../training/retinal-sensor.js';
import {createCompactVision} from '../training/compact-vision.js';

function frame({width=8,height=8,sequence=0,bodyTime=0}={}){
 const rgb=Uint8Array.from({length:width*height*6},(_,i)=>(Math.floor(i/(width*height*3))*97+i*13)%256);
 return {width,height,rgb,pixels:Uint8Array.from({length:width*height*2},(_,i)=>i%256),sequence,bodyTime,
  eyes:[{side:'left'},{side:'right'}]};
}
const context=(source,nativeTimeSeconds=source.lastFrame.bodyTime+.002)=>({sensoryFeedback:source,body:{time:nativeTimeSeconds},neuralMs:nativeTimeSeconds*1000});
function fixture(){
 let wall=0;const messages=[],observer=createEyeObserver({now:()=>wall,
  postMessage:(message,transfer)=>messages.push(structuredClone(message,{transfer}))});
 return {observer,messages,wall:value=>wall=value};
}
function assertPixel(snapshot,source,eye,x,y){
 const sx=Math.min(source.width-1,Math.floor((x+.5)*source.width/snapshot.width)),
  sy=Math.min(source.height-1,Math.floor((y+.5)*source.height/snapshot.height)),
  from=(eye*source.width*source.height+sy*source.width+sx)*3,to=(y*snapshot.width+x)*4;
 assert.deepEqual(Array.from((eye?snapshot.right:snapshot.left).slice(to,to+4)),[...source.rgb.slice(from,from+3),255]);
}

test('disabled observer reads no sensory state; enabled transfer owns exact left/right pixels and capture time',()=>{
 const f=fixture(),source={lastFrame:frame()},original=source.lastFrame.rgb.slice(),token=f.observer.beginJob('assigned-job',7);
 f.observer.observe({get sensoryFeedback(){throw new Error('disabled observer read');}});assert.equal(f.messages.length,0);
 f.observer.configure({enabled:true});f.observer.observe(context(source));
 assert.equal(f.messages.length,1);const {id,snapshot}=f.messages[0];assert.equal(id,7);assert.equal(snapshot.jobId,'assigned-job');
 assert.equal(snapshot.source,'sensory-retina');assert.equal(snapshot.sensoryInput,'luminance-derived-from-rgb');assert.equal(snapshot.format,'rgba8');
 assert.equal(snapshot.frameTimeSeconds,0);assert.equal(snapshot.nativeTimeSeconds,.002);assert.equal(snapshot.neuralTimeMs,2);
 assert.equal(snapshot.width,8);assert.equal(snapshot.height,8);assert.equal(snapshot.sourceWidth,8);assert.equal(snapshot.sourceHeight,8);
 for(let side=0;side<2;side++)for(let y=0;y<8;y++)for(let x=0;x<8;x++)assertPixel(snapshot,source.lastFrame,side,x,y);
 assert.deepEqual(source.lastFrame.rgb,original);assert.equal(source.lastFrame.rgb.byteLength,8*8*6);
 snapshot.left.fill(0);assert.deepEqual(source.lastFrame.rgb,original,'Transferred display copy never aliases sensor pixels');
 f.observer.endJob(token);f.wall(1000);source.lastFrame=frame({sequence:1,bodyTime:.02});f.observer.observe(context(source));assert.equal(f.messages.length,1);
 f.observer.dispose();assert.throws(()=>f.observer.configure({enabled:true}),/disposed/);
});

test('wall throttling and retinal deduplication preserve trial resets and cumulative sample sequence',()=>{
 const f=fixture(),source={lastFrame:frame()};f.observer.configure({enabled:true});f.observer.beginJob('fit-job',8);
 f.observer.observe(context(source));source.lastFrame=frame({sequence:1,bodyTime:.02});
 f.wall(EYE_SAMPLE_INTERVAL_MS-1);f.observer.observe(context(source));assert.equal(f.messages.length,1);
 f.wall(EYE_SAMPLE_INTERVAL_MS);f.observer.observe(context(source));assert.equal(f.messages.length,2);
 f.wall(1000);f.observer.observe(context(source));assert.equal(f.messages.length,2,'Repeated consumed image is not resent');
 const nextTrial={lastFrame:frame()};f.observer.observe(context(nextTrial));assert.equal(f.messages.length,3);
 assert.deepEqual(f.messages.map(m=>m.snapshot.sampleSequence),[1,2,3]);assert.deepEqual(f.messages.map(m=>m.snapshot.trialSequence),[1,1,2]);
 assert.deepEqual(f.messages.map(m=>m.snapshot.sequence),[0,1,0]);assert.deepEqual(f.messages.map(m=>m.id),[8,8,8]);
 f.observer.configure({enabled:false});f.wall(1500);f.observer.observe(context(nextTrial));assert.equal(f.messages.length,3);
 f.observer.configure({enabled:true});nextTrial.lastFrame=frame({sequence:1,bodyTime:.02});f.observer.observe(context(nextTrial));
 assert.equal(f.messages[3].snapshot.trialSequence,2,'Toggling display cannot invent a new demonstration');
 f.observer.dispose();
});

test('larger configured retina is bounded for display by sampling existing pixels, without flips or redraw',()=>{
 const f=fixture(),source={lastFrame:frame({width:1024,height:512}),render(){throw new Error('Display must not render');}};
 f.observer.configure({enabled:true});f.observer.beginJob('large-retina-job',9);f.observer.observe(context(source));
 const {snapshot}=f.messages[0];assert.equal(snapshot.width,256);assert.equal(snapshot.height,128);
 assert.equal(snapshot.sourceWidth,1024);assert.equal(snapshot.sourceHeight,512);
 assert.equal(snapshot.left.byteLength+snapshot.right.byteLength,262144);
 for(let side=0;side<2;side++)for(const [x,y]of [[0,0],[255,0],[0,127],[255,127],[80,50]])assertPixel(snapshot,source.lastFrame,side,x,y);
 assert.equal(source.lastFrame.rgb.byteLength,1024*512*6);f.observer.dispose();
});

test('unavailable, malformed, future and failed presentation sinks never interrupt simulation',()=>{
 const absent=fixture();absent.observer.configure({enabled:true});absent.observer.beginJob('no-vision',10);
 absent.observer.observe({sensoryFeedback:{lastFrame:null}});assert.equal(absent.messages.length,0);assert.equal(absent.observer.state.enabled,true);absent.observer.dispose();
 for(const mutate of [f=>f.rgb=new Uint8Array(1),f=>f.bodyTime=10,f=>f.eyes.reverse(),f=>f.sequence=-1]){
  const fixtureValue=fixture(),source={lastFrame:frame()};mutate(source.lastFrame);
  fixtureValue.observer.configure({enabled:true});fixtureValue.observer.beginJob('invalid-image',11);
  assert.doesNotThrow(()=>fixtureValue.observer.observe(context(source,.002)));
  assert.equal(fixtureValue.messages[0].type,'eyes-error');assert.equal(fixtureValue.messages[0].id,11);assert.equal(fixtureValue.observer.state.enabled,false);fixtureValue.observer.dispose();
 }
 const broken=createEyeObserver({postMessage(){throw new Error('Detached page');},now:()=>0});
 broken.configure({enabled:true});broken.beginJob('sink-failure',12);assert.doesNotThrow(()=>broken.observe(context({lastFrame:frame()})));assert.equal(broken.state.enabled,false);broken.dispose();
 const f=fixture();assert.throws(()=>f.observer.configure({enabled:1}),/boolean/);f.observer.configure({enabled:true});
 f.observer.beginJob('no-rpc',null);f.observer.observe(context({lastFrame:frame()}));assert.equal(f.messages.length,0);f.observer.dispose();
});

test('observing actual rendered eyes leaves the luminance inputs and compact-motion outputs exactly unchanged',()=>{
 const retina=createRetinalSensor({width:32,height:16,textureAntialias:true}),mapping={neuron_count:16,cells:
  ['left','right'].flatMap((side,e)=>['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'].map((type,k)=>({index:e*8+k,side,type,u:.5,v:.5})))},
  motion=createCompactVision({mapping,width:32,height:16}),control=createCompactVision({mapping,width:32,height:16}),f=fixture(),source={lastFrame:null};
 f.observer.configure({enabled:true});f.observer.beginJob('real-optics-fixture',13);
 for(let k=0;k<3;k++){
  const rendered=retina.render({position:[k*.03,0,3],quaternion:[1,0,0,0],bodyTime:k*.02,
   bowl:{radiusCm:6.5,ceilingCm:5.6,floor:{baseCm:.15,radialCoefficientPerCm:.037}},food:[]});
  const rgb=rendered.rgb.slice(),pixels=rendered.pixels.slice();source.lastFrame=rendered;
  motion.update(rendered);control.update(rendered);const before=motion.ratesHz.slice(),summary=structuredClone(motion.summary);
  f.wall(k*500);f.observer.observe(context(source));
  assert.deepEqual(rendered.rgb,rgb);assert.deepEqual(rendered.pixels,pixels);
  assert.deepEqual(motion.ratesHz,before);assert.deepEqual(motion.ratesHz,control.ratesHz);assert.deepEqual(motion.summary,summary);
  assert.equal(source.lastFrame.sequence,k);assert.equal(f.messages.at(-1).snapshot.sequence,k);
 }
 assert.equal(f.messages.length,3);assert.equal(retina.render({position:[0,0,3],quaternion:[1,0,0,0],bodyTime:.06,
  bowl:{radiusCm:6.5,ceilingCm:5.6,floor:{baseCm:.15,radialCoefficientPerCm:.037}},food:[]}).sequence,3,'Display caused no extra render');
 f.observer.dispose();
});

test('worker allows active eye toggles, preserves result/frame payloads and cannot steal RPC identity',async()=>{
 class Brain{readState(){return new Float32Array(0);}}
 let wall=0,finish,observe;const messages=[],worker=createObservedTrainingWorker({Brain,now:()=>wall,
  postMessage:(message,transfer)=>messages.push(structuredClone(message,{transfer})),
  createEnvironment:async()=>({ready:()=>({backend:'fixture'}),dispose(){},async evaluate(job,options){
   observe=options.onPhysicsStep;options.onFrame({time:0,fixture:true});await new Promise(resolve=>finish=resolve);
   return {return:0,parameters:job.parameters,fixture:true};
  }})});
 try{
  await worker.handle({type:'initialize',id:1});const evaluation=worker.handle({type:'evaluate',id:2,job:{jobId:'accepted-fit',mode:'decoder-fit',parameters:[.2]}});
  observe({get sensoryFeedback(){throw new Error('disabled cache read');}});assert.equal(messages.filter(m=>m.type==='eyes').length,0);
  await worker.handle({type:'observe-eyes',enabled:true});const source={lastFrame:frame()};observe(context(source));
  await worker.handle({type:'evaluate',id:3,job:{jobId:'rejected-concurrent'}});
  wall=500;source.lastFrame=frame({sequence:1,bodyTime:.02});observe(context(source));
  const eyes=messages.filter(m=>m.type==='eyes');assert.deepEqual(eyes.map(m=>m.id),[2,2]);assert(eyes.every(m=>m.snapshot.jobId==='accepted-fit'));
  await worker.handle({type:'observe-eyes',enabled:false});wall=1000;source.lastFrame=frame({sequence:2,bodyTime:.04});observe(context(source));assert.equal(messages.filter(m=>m.type==='eyes').length,2);
  await worker.handle({type:'observe-eyes',enabled:true});await worker.handle({type:'cancel'});observe(context(source));assert.equal(messages.filter(m=>m.type==='eyes').length,2);
  finish();await evaluation;
  assert.deepEqual(messages.find(m=>m.type==='frame').frame,{time:0,fixture:true});
  assert.deepEqual(messages.find(m=>m.type==='evaluation').result,{return:0,parameters:[.2],fixture:true});
  assert.equal(worker.state.eyeObservation.jobId,null);assert.equal(worker.state.busy,false);
  await worker.handle({type:'observe-eyes',enabled:'yes'});assert.equal(messages.at(-1).type,'eyes-error');
  await worker.handle({type:'stop'});assert.equal(worker.state.eyeObservation.disposed,true);
 }finally{finish?.();worker.dispose();}
});
