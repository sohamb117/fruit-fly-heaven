import test from 'node:test';
import assert from 'node:assert/strict';
import {WebGPUBrain} from '../src/webgpu.js';

// This fixture exercises subscription ownership, not GPU work.
function deviceFixture(){
  let lostCallback;const device={subscriptions:0,destroyed:0,listeners:[],
    lost:{then(callback){device.subscriptions++;lostCallback=callback;}},
    addEventListener(type,callback){this.listeners.push({type,callback});},destroy(){this.destroyed++;},
    lose(message){lostCallback({message});}};return device;
}

test('many episode wrappers subscribe to device health only once and observe the same error',()=>{
  const device=deviceFixture(),model={manifest:{neuron_count:1}},brains=Array.from({length:64},()=>new WebGPUBrain(device,model));
  const shared={references:brains.length,buffers:[]};for(const brain of brains)brain.shared=shared;
  assert.equal(device.subscriptions,1);assert.equal(device.listeners.length,1);
  for(const brain of brains.slice(1))brain.dispose();assert.equal(device.destroyed,0);
  const error=new Error('GPU validation failure');device.listeners[0].callback({error});
  assert.throws(()=>brains[0].live(),error);assert.equal(brains[0].health,brains[1].health);
  brains[0].dispose();assert.equal(device.destroyed,1);
});

test('a device-loss record propagates to fresh wrappers without adding callbacks',()=>{
  const device=deviceFixture(),model={manifest:{neuron_count:1}},first=new WebGPUBrain(device,model);
  device.lose('fixture loss');const later=new WebGPUBrain(device,model);
  assert.equal(device.subscriptions,1);assert.equal(device.listeners.length,1);
  assert.throws(()=>first.live(),/fixture loss/);assert.throws(()=>later.live(),/fixture loss/);
  const separate=new WebGPUBrain(deviceFixture(),model);assert.doesNotThrow(()=>separate.live());
  first.dispose();later.dispose();separate.dispose();
});
