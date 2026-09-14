import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createMotorReplayCapture,captureInitialMotorState,packCaptureArray} from './training-motor-capture.js';
import {FlyBodyPhysics,flybodyScene} from '../flybody-physics.js';
import {createWasmCore,WasmMuscles} from '../../packages/banc-runtime/src/wasm.js';
import loadMujoco from '../../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';

const unpack = (base64, bits) => {
  const bytes = Uint8Array.from(atob(base64), value => value.charCodeAt(0)), view = new DataView(bytes.buffer);
  assert.equal(bytes.length % (bits / 8), 0);
  return Array.from({length: bytes.length / (bits / 8)}, (_, i) => bits === 32 ? view.getFloat32(i * 4, true) : view.getFloat64(i * 8, true));
};

test('packed rates and native state round-trip exactly with explicit little-endian precision', () => {
  const rates = [0,-0,Math.fround(.1),100,Math.fround(1e-35)];
  assert.deepEqual(unpack(packCaptureArray(rates,32),32),rates);
  const native = [0,-0,Math.PI,1e-250,-1e250];
  assert.deepEqual(unpack(packCaptureArray(native),64),native);
  assert.throws(() => packCaptureArray([.1],32),/lose float32 precision/);
  assert.throws(() => packCaptureArray([Infinity]),/Nonfinite/);
  assert.throws(() => packCaptureArray([0],16),/Unsupported/);
});

let assets;
async function fixture() {
  assets ??= Promise.all([loadMujoco(),createWasmCore(),
    fs.readFile(new URL('../../models/flybody-mujoco.xml',import.meta.url),'utf8'),
    fs.readFile(new URL('../../models/flybody-mujoco.json',import.meta.url),'utf8').then(JSON.parse),
    fs.readFile(new URL('../../data/prepared/banc888/io.json',import.meta.url),'utf8').then(JSON.parse)]);
  const [mj,core,sourceXml,metadata,io] = await assets;
  const habitat = {fruit:[],ceiling:50}, scene = flybodyScene(sourceXml,habitat), model = mj.MjModel.from_xml_string(scene.xml);
  model.hfield_data.set(scene.heights);
  const environment = {surface:(x,y)=>.15+.037*(x*x+y*y),odor:()=>0,foodAt:()=>null};
  const makeBody = () => {
    const body = new FlyBodyPhysics(mj,model,metadata,io,count=>new WasmMuscles(core,count),environment);
    body.place(0,0,.4);body.data.qvel[3]=.2;body.data.qvel[7]=-0;body.input[0]=-0;mj.mj_forward(model,body.data);body.refresh();return body;
  };
  const body = makeBody(), control = makeBody();
  const world = {habitat,metadata,io,movementMode:'direct',motorCoupling:true,flightEnabled:true};
  const fly = {id:1,x:0,y:0,z:0,heading:.4,brain:{motorNeuronRates:io.motor_neurons.map((_,i)=>Math.fround(10+(i%7)*.125))}};
  const context = {body,world,fly,motorIndices:Uint32Array.from(io.motor_neurons,m=>m.index),job:{name:'capture-test',parameters:Array(27).fill(0)},
    provenance:{configHash:'test',modelFingerprint:'test'},initialCondition:{stage:'landing'},initialObservation:{finite:true}};
  return {sourceXml,body,control,world,fly,context,dispose(){body.dispose();control.dispose();model.delete();}};
}

test('native observer captures all motor blocks without mutating integration, muscle or wing state', async t => {
  const f = await fixture();
  try {
    const recorder = createMotorReplayCapture({sourceXml:f.sourceXml,seconds:.01});
    const before = captureInitialMotorState(f.body);
    recorder.onInitialState(f.context);
    assert.deepEqual(captureInitialMotorState(f.body),before);
    assert.equal(before.native.history.length,f.body.data.history.length);
    assert.equal(before.native.plugin_state.length,f.body.data.plugin_state.length);
    const serialized=JSON.parse(JSON.stringify(before));
    assert.deepEqual(unpack(serialized.packed.nativeIntegration.data,64),before.nativeIntegration.values);
    assert(Object.is(unpack(serialized.packed.muscleInput.data,32)[0],-0),'binary initial buffers preserve signed zero through JSON');
    const expected=[];
    for(let index=0;index<8;index++) {
      f.fly.brain.motorNeuronRates=f.fly.brain.motorNeuronRates.map(value=>Math.fround(value+.03125));
      const rates=new Map(Array.from(f.context.motorIndices,(id,k)=>[id,f.fly.brain.motorNeuronRates[k]]));
      const timeBefore=f.body.data.time;
      f.body.step(rates,.002,{coupling:true,flight:true});f.control.step(rates,.002,{coupling:true,flight:true});
      const stateBeforeObserver=captureInitialMotorState(f.body);
      recorder.onPhysicsStep({body:f.body,world:f.world,fly:f.fly,index,durationSeconds:.002,timeBefore,neuralMs:(index+1)*2});
      assert.deepEqual(captureInitialMotorState(f.body),stateBeforeObserver,'observer changes no native or JS state');
      assert.deepEqual(Array.from(f.body.data.qpos),Array.from(f.control.data.qpos),'observed and unobserved native trajectories match');
      assert.deepEqual(Array.from(f.body.data.qvel),Array.from(f.control.data.qvel));
      assert.deepEqual(f.body.muscleState,f.control.muscleState);
      if(index<5)expected.push({rates:[...f.fly.brain.motorNeuronRates],qpos:Array.from(f.body.data.qpos),qvel:Array.from(f.body.data.qvel)});
    }
    const result=recorder.finish({reason:'test_end',simSeconds:.016,steps:8,cancelled:false,success:false});
    assert.equal(result.steps.length,5);assert.equal(result.capturedSeconds,.01);assert.equal(result.horizonReached,true);
    assert.equal(result.completeThroughEvaluation,false);assert.equal(result.endedBy,'capture_limit');
    assert.deepEqual(result.initial,before);assert.equal(result.dimensions.motorCount,805);
    assert.deepEqual(result.motorNeuronIndices,Array.from(f.context.motorIndices));
    assert.deepEqual(unpack(result.scene.heights,32),Array.from(f.body.model.hfield_data));
    result.steps.forEach((step,i)=>{assert.deepEqual(unpack(step.ratesHz,32),expected[i].rates);assert.deepEqual(unpack(step.postQpos,64),expected[i].qpos);assert.deepEqual(unpack(step.postQvel,64),expected[i].qvel);});
    assert.equal(result.initial.wing.opening,undefined);assert.equal(result.initial.wing.interpreter.steering.b1_muscle.biasGain,1);
    const size=new TextEncoder().encode(JSON.stringify(result)).byteLength;
    const oneStepBytes=new TextEncoder().encode(JSON.stringify(result.steps[0])).byteLength;
    assert(size+245*oneStepBytes<7*1024*1024,'full 0.5 s capture stays within the allocated report bound');
    t.diagnostic(`Native capture fixture: ${size} bytes for 5 blocks; conservatively projected 250 blocks ${size+245*oneStepBytes} bytes.`);
    assert.throws(()=>recorder.finish({}),/already finalized/);
  } finally {f.dispose();}
});

test('capture retains early termination and rejects wrong scene, motor identity or block clocks',async()=>{
  const f=await fixture();
  try {
    const make=()=>createMotorReplayCapture({sourceXml:f.sourceXml,seconds:.5});
    const badScene=make(),original=f.body.model.hfield_data[0];f.body.model.hfield_data[0]=.7;
    assert.throws(()=>badScene.onInitialState(f.context),/heightfield/);f.body.model.hfield_data[0]=original;
    const wrongIds=make(),ids=f.context.motorIndices.slice();ids[0]++;
    assert.throws(()=>wrongIds.onInitialState({...f.context,motorIndices:ids}),/identity mismatch/);
    const recorder=make();recorder.onInitialState(f.context);
    assert.throws(()=>recorder.onPhysicsStep({body:f.body,world:f.world,fly:f.fly,index:1,durationSeconds:.002,timeBefore:0,neuralMs:2}),/discontinuity/);
    const rates=new Map(Array.from(f.context.motorIndices,(id,k)=>[id,f.fly.brain.motorNeuronRates[k]]));f.body.step(rates,.002);
    recorder.onPhysicsStep({body:f.body,world:f.world,fly:f.fly,index:0,durationSeconds:.002,timeBefore:0,neuralMs:2});
    const result=recorder.finish({reason:'excessive_rotation',simSeconds:.002,steps:1,cancelled:false,success:false});
    assert.equal(result.endedBy,'episode_end');assert.equal(result.completeThroughEvaluation,true);assert.equal(result.horizonReached,false);
    assert.equal(make().finish({cancelled:true,steps:0}),null);
    for(const seconds of [0,-1,.501,.003,NaN])assert.throws(()=>createMotorReplayCapture({sourceXml:f.sourceXml,seconds}));
  } finally {f.dispose();}
});
