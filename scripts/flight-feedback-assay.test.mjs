import test from 'node:test';
import assert from 'node:assert/strict';
import {createWasmCore,WasmBrain} from '../packages/banc-runtime/src/wasm.js';
import {fixture} from '../packages/banc-runtime/test/fixture.mjs';
import {createWingMotorEventReader} from '../packages/banc-runtime/src/motor-events.js';
import {createWingEventExcitation} from '../web/flybody-wing-event-excitation.js';
import {createMotorDecoder,buildMotorDecoderContract} from '../web/motor-decoder.js';
import {STEERING_MUSCLE_TYPES} from '../web/training/flight-parameters.js';
import {captureWasmFlightState,runFlightFeedbackAssay} from './flight-feedback-assay.mjs';

// Only a 50-cell native WASM fixture. No actual BANC graph or body is allocated.
const core=await createWasmCore();
function ioFixture(){
 let next=0;const muscles=[];
 const add=(side,target,count,steering=false)=>{
  const indices=Array.from({length:count},()=>next++);
  muscles.push({kind:steering?'wing_steering_assumption':'asynchronous_wing',
   joint:(steering?'wing_steer_':'wing_power_')+side,target,indices,root_ids:indices.map(i=>String(1000+i)),sign:1});
 };
 for(const side of ['left','right']){
  add(side,'dorsal_longitudinal_muscle',5);add(side,'dorsoventral_muscle',7);
  for(const target of STEERING_MUSCLE_TYPES)add(side,target,1,true);
 }
 return {muscles};
}
function contextFixture(){
 const model=fixture({n:50,edges:[{source:48,post:12,weight:8,delay:1},{source:49,post:36,weight:8,delay:1}]}),io=ioFixture();model.io=io;
 const brain=new WasmBrain(core,model),contract=buildMotorDecoderContract(io),decoder=createMotorDecoder(io,contract.parameters.map(p=>p.initial));
 const adapter=createWingEventExcitation({io}),wingIndices=Uint32Array.from(contract.indices);
 const reader=createWingMotorEventReader({indices:wingIndices,params:model.params,dtMs:.5,bodyBlockMs:2});
 const read=()=>brain.readState(wingIndices,{includeSpikeTime:true,includeStatistics:false,includeSpikeHistory:false});
 const input=new Float32Array(50),internal={hunger:.1,insulin:.2,akh:.3};
 adapter.accept(reader.read(read(),0));
 // Real rest evolution establishes 500ms clocks; still only 50 fixture cells.
 for(let i=0;i<250;i++){
  brain.step(4,input,internal,true);adapter.accept(reader.read(read(),brain.timeMs));
  for(const time of [brain.timeMs-1,brain.timeMs])decoder.advance(adapter.finishInterval(time).unitExcitation);
 }
 const body={time:.5,remainder:0,environmentContactCount:0,data:{qfrc_applied:new Float64Array(6),xfrc_applied:new Float64Array(6),qvel:Float64Array.from([0,0,0,.2,.1,-.3])},
  internal,halterePower:[1,.8],wings:{phase:.4,frequencyHz:236},motorDecoder:decoder,
  _wingMotorEvents:{adapter,initialized:true,elapsedMs:500,observedMs:500}};
 return {brain,body,world:{io},input,inputSequence:null,phase:'scored-release',fly:{feedback:{bodyTime:.5}},
  encoder:{ratesHz:Float32Array.from([1,2]),sample:{fixture:true}},sensoryFeedback:{update(){throw new Error('Never call controller');}},provenance:{fixture:true}};
}
function mapperFactory({maxCurrentPa}){
 return {maxCurrentPa,neuronCount:50,indices:[48,49],writeInto(input,sample){
  // Explicit synthetic sign-sensitive mapping, not receptor physiology.
  const gain=(1+Math.sin(sample.wingPhaseRadians+2*Math.PI*sample.wingFrequencyHz*sample.elapsedSeconds))*.5;
  input[48]=Math.max(0,sample.omegaRootRadS[1])*maxCurrentPa*gain;
  input[49]=Math.max(0,-sample.omegaRootRadS[1])*maxCurrentPa*gain;
  return {kind:'synthetic-test-prior',maxCurrentPa};
 }};
}
const options={haltereMapper:mapperFactory,maxCurrentPa:[100],phaseOffsetsRadians:[0],pitchRateRadS:5,pulseMs:10,recoveryMs:6};

test('capture owns full forward state after source disposal and all six branches keep exact baselines',async()=>{
 const context=contextFixture(),before=context.brain.readState(),snapshot=captureWasmFlightState(context,{afferentIndices:[48,49]});
 assert.deepEqual(context.brain.readState(),before);assert.equal(context.brain.shared.references,2);
 context.input.fill(99);context.body.halterePower[0]=0;context.encoder.sample.fixture=false;
 assert.deepEqual(snapshot.nativeSample.halterePower,[1,.8]);assert.deepEqual(snapshot.sensorySample,{fixture:true});
 const shared=context.brain.shared;context.brain.dispose();assert.equal(shared.references,1);
 try{
  const result=await runFlightFeedbackAssay(snapshot,options),group=result.groups[0];
  assert.equal(result.completed,true);assert.equal(group.arms.length,6);assert.equal(group.duplicateShamExact,true);assert.equal(group.unsignedPairExact,true);
  assert.equal(shared.references,1);assert.equal(result.initialDecoderSnapshot.timeMs,500);
  assert.equal(group.arms[0].trace[0].timeMs,500.5);assert.equal(group.arms[0].motorPackets[0].timeMs,502);
  assert.deepEqual(group.arms[0].decoder.slice(0,2).map(r=>r.timeMs),[501,502]);
  assert(group.arms[2].trace.some(r=>r.events.some(e=>e.index===48)),'Synthetic input should evoke real afferent events');
  assert(group.arms[3].trace.some(r=>r.events.some(e=>e.index===49)));
  assert.notEqual(group.response.firstDifferentAfferentEventMs,null);assert.notEqual(group.response.firstDifferentWingEventMs,null);
  assert.notEqual(group.response.firstDifferentDecoderFeatureMs,null);
  for(const arm of group.arms){
   assert.equal(arm.finalHashes.tick,1032);assert.equal(arm.trace.length,32);
   for(const row of arm.trace)assert.deepEqual(row.actualCurrentsPa,row.requestedCurrentsPa);
   assert(arm.trace.flatMap(r=>r.events).every(e=>e.timeMs>500&&e.timeMs<=516));
  }
 }finally{snapshot.dispose();snapshot.dispose();}
 assert.equal(shared.references,0);await assert.rejects(runFlightFeedbackAssay(snapshot,options),/disposed/);
});

test('invalid capture phase, clocks and native external forces fail before retaining memory',()=>{
 const context=contextFixture();
 try{
  for(const change of [c=>{c.phase='warmup';},c=>{c.body.time=.499;},c=>{c.body.data.xfrc_applied[0]=1;},c=>{c.body.environmentContactCount=1;}]){
   const candidate={...context,body:{...context.body,data:{...context.body.data,xfrc_applied:context.body.data.xfrc_applied.slice()}}};change(candidate);
   assert.throws(()=>captureWasmFlightState(candidate));assert.equal(context.brain.shared.references,1);
  }
 }finally{context.brain.dispose();}
});

test('mapper escapes and callback failures clean branch memory without consuming retained state',async()=>{
 const context=contextFixture(),snapshot=captureWasmFlightState(context),shared=context.brain.shared;
 context.brain.dispose();
 try{
  const escaped=({maxCurrentPa})=>({...mapperFactory({maxCurrentPa}),writeInto(input){input[0]=10;}});
  await assert.rejects(runFlightFeedbackAssay(snapshot,{...options,haltereMapper:escaped}),/non-haltere/);assert.equal(shared.references,1);
  let calls=0;await assert.rejects(runFlightFeedbackAssay(snapshot,{...options,checkpoint:async()=>{if(++calls===2)throw new Error('fixture cancellation');}}),/fixture cancellation/);
  assert.equal(shared.references,1);
  const result=await runFlightFeedbackAssay(snapshot,{...options,pulseMs:2,recoveryMs:2});assert.equal(result.completed,true);
 }finally{snapshot.dispose();}
 assert.equal(shared.references,0);
});
