import test from 'node:test';
import assert from 'node:assert/strict';
import {makeContributorPreviewFrame,createContributorHeartbeat} from '../../scripts/contribute-training-native.mjs';

const rotation=[1,0,0,0,1,0,0,0,1];
function frame(){
 const shape={position:[0,0,3.5],rotation:rotation.slice(),size:[.1,.02,.001]};
 return {time:0,simSeconds:0,stage:'maintained_flight',position:[0,0,3.5],quaternion:[1,0,0,0],
  feet:Array.from({length:6},()=>[0,0,3.4]),legs:Array.from({length:6},()=>[[0,0,3.4],[.1,0,3.3],[.2,0,3.2]]),
  wings:['left','right'].map(side=>({...structuredClone(shape),side,anchor:[0,0,3.5]})),
  mouth:{anchors:[[0,0,3.5]],ellipsoids:[structuredClone(shape)]},
  contacts:{environment:0,food:0,legs:[0,0,0,0,0,0],mouth:[0,0],wings:[0,0]},
  bowl:{radiusCm:50,ceilingCm:50,floor:{baseCm:.15,radialCoefficientPerCm:.037,capRadiusCm:6.5}},
  phase:'warmup',motion:'falling',neuralMs:0,neuralSpikes:0,vision:false,warmup:true,warmupSeconds:0,
  episodeTimeSeconds:0,nativeTimeSeconds:0,releaseNativeTime:null,releaseNeuralTimeMs:null,scoredSteps:0,
  metrics:{parameters:[1,2,3],large:'x'.repeat(100000)},food:[{private:'excluded'}],leaseToken:'excluded'};
}

test('preview is a bounded owned geometry snapshot with null warmup clocks omitted',()=>{
 const original=frame(),preview=makeContributorPreviewFrame(original);
 assert(preview);assert(Buffer.byteLength(JSON.stringify(preview))<=32768);
 assert.deepEqual(preview.position,original.position);assert.deepEqual(preview.wings,original.wings);
 for(const key of ['food','metrics','leaseToken','releaseNativeTime','releaseNeuralTimeMs'])assert(!Object.hasOwn(preview,key));
 original.position[2]=999;original.wings[0].position[2]=999;original.bowl.floor.baseCm=999;
 assert.equal(preview.position[2],3.5);assert.equal(preview.wings[0].position[2],3.5);assert.equal(preview.bowl.floor.baseCm,.15);
});

test('released clocks are preserved and no pose is normalized or interpolated',()=>{
 const original=frame();Object.assign(original,{time:.702,simSeconds:.202,episodeTimeSeconds:.202,nativeTimeSeconds:.702,
  releaseNativeTime:.5,releaseNeuralTimeMs:500,warmup:false,quaternion:[.99999,.001,0,0]});
 const preview=makeContributorPreviewFrame(original);
 assert.deepEqual(preview.quaternion,original.quaternion);assert.equal(preview.releaseNativeTime,.5);
 assert.equal(preview.nativeTimeSeconds,.702);assert.equal(preview.warmup,false);
});

test('invalid telemetry is omitted, including excessive lists and unbounded coordinates',()=>{
 for(const mutate of [f=>f.position[0]=NaN,f=>f.position[0]=1000001,f=>f.quaternion=[2,0,0,0],
  f=>f.wings[0].size[0]=-.1,f=>f.wings[0].side='unknown',f=>f.feet.push([0,0,0]),
  f=>f.mouth.anchors=Array.from({length:5},()=>[0,0,0]),f=>f.neuralMs=Infinity,f=>f.simSeconds=-.1,
  f=>f.phase='Invalid label',f=>f.contacts.legs[0]=-1,f=>f.warmup='true']){
  const input=frame();mutate(input);assert.equal(makeContributorPreviewFrame(input),null);
 }
 assert.equal(makeContributorPreviewFrame(null),null);
});

function fakeTimer(){
 let callback=null,cleared=false,delay=null;
 return {setTimer(fn,ms){callback=fn;delay=ms;return 7;},clearTimer(id){assert.equal(id,7);cleared=true;},
  tick(){assert(callback);callback();},get cleared(){return cleared;},get delay(){return delay;}};
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));

test('renewal sends latest recorded frame every five seconds using fixed lease identity',async()=>{
 const timer=fakeTimer(),sent=[],identity={jobId:'g0-p0-pos',leaseToken:'lease-fixture'};
 let latest=makeContributorPreviewFrame(frame());
 const heartbeat=createContributorHeartbeat({identity,send:async payload=>{sent.push(payload);},getPreview:()=>latest,...timer});
 assert.equal(timer.delay,5000);identity.jobId='different-job';timer.tick();await settle();
 assert.equal(sent[0].jobId,'g0-p0-pos');assert.equal(sent[0].previewFrame.time,0);
 latest=makeContributorPreviewFrame({...frame(),time:.2});timer.tick();await settle();
 assert.equal(sent[1].previewFrame.time,.2);latest.position[0]=999;assert.equal(sent[1].previewFrame.position[0],0);
 await heartbeat.stop();assert(timer.cleared);timer.tick();await settle();assert.equal(sent.length,2);
});

test('plain renewal still works when preview is unavailable or invalid',async()=>{
 const timer=fakeTimer(),sent=[];
 const heartbeat=createContributorHeartbeat({identity:{jobId:'job'},send:async payload=>sent.push(payload),getPreview:()=>null,...timer});
 await heartbeat.flush();assert.deepEqual(sent,[{jobId:'job'}]);await heartbeat.stop();
});

test('overlapping timer ticks share one request and final flush sends the newest frame',async()=>{
 const timer=fakeTimer(),sent=[];let resolveFirst,latest=makeContributorPreviewFrame(frame());
 const heartbeat=createContributorHeartbeat({identity:{jobId:'job'},getPreview:()=>latest,...timer,
  send:payload=>{sent.push(payload);return sent.length===1?new Promise(resolve=>{resolveFirst=resolve;}):Promise.resolve();}});
 timer.tick();timer.tick();await settle();assert.equal(sent.length,1);
 latest=makeContributorPreviewFrame({...frame(),time:.8});const flushing=heartbeat.flush();
 resolveFirst();await flushing;assert.equal(sent.length,2);assert.equal(sent[1].previewFrame.time,.8);
 await heartbeat.stop();
});

test('lease failure propagates through checkpoint/flush without unhandled timer rejection',async()=>{
 const timer=fakeTimer(),failure=new Error('lease expired');let requests=0;
 const heartbeat=createContributorHeartbeat({identity:{jobId:'job'},getPreview:()=>null,...timer,
  send:async()=>{requests++;throw failure;}});
 timer.tick();await settle();assert.throws(()=>heartbeat.check(),error=>error===failure);
 await assert.rejects(heartbeat.flush(),error=>error===failure);timer.tick();await settle();assert.equal(requests,1);
 await heartbeat.stop();assert(timer.cleared);
});

test('stop drains an in-flight renewal before a caller releases its lease',async()=>{
 const timer=fakeTimer(),events=[];let complete;
 const heartbeat=createContributorHeartbeat({identity:{jobId:'job'},getPreview:()=>null,...timer,
  send:async()=>{events.push('heartbeat');await new Promise(resolve=>{complete=resolve;});events.push('heartbeat-complete');}});
 timer.tick();await settle();const stopped=heartbeat.stop().then(()=>{events.push('release');});
 await settle();assert.deepEqual(events,['heartbeat']);complete();await stopped;
 assert.deepEqual(events,['heartbeat','heartbeat-complete','release']);timer.tick();await settle();assert.equal(events.length,3);
});
