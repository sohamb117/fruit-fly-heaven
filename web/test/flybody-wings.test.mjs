import test from 'node:test';
import assert from 'node:assert/strict';
import {FlyBodyWings} from '../flybody-wings.js';

function fixture(){
 const joints=['left','right'].flatMap((side,s)=>['yaw','roll','pitch'].map((axis,a)=>({name:`wing_${axis}_${side}`,qpos:s*3+a,neutral:0,range:[-2,2]})));
 const actuators=joints.map((j,id)=>({name:j.name,id,range:[-1,1]}));
 const zero=[0,0,0,0,0,0],full=[1,1,1,1,1,1];
 const metadata={joints,actuators,wing_actuation:{steering:{iii1_muscle:[.1,0,0,0,0,0],b1_muscle:[0,.1,0,0,0,0]},targets:[[zero,zero],[full,full]],powers:[0,1],frequency_hz:200,deployment_tau_s:.01,deployment_before_beating:0}};
 return {wings:new FlyBodyWings(metadata),q:new Float64Array(6),ctrl:new Float64Array(6)};
}

test('flight interpreter preserves independent left/right BANC power',()=>{
 const {wings,q,ctrl}=fixture();wings.step(q,ctrl,1,0,{left:{},right:{}},.05);
 assert.ok(wings.power[0]>.95);assert.equal(wings.power[1],0);assert.ok(ctrl[0]>.9);assert.equal(ctrl[3],0);
});

test('steering muscles perturb the fixed wingbeat basis but cannot gate deployment',()=>{
 const a=fixture(),b=fixture();
 a.wings.step(a.q,a.ctrl,1,1,{left:{iii1_muscle:1},right:{}},.05);
 b.wings.step(b.q,b.ctrl,1,1,{left:{},right:{}},.05);
 assert.ok(a.wings.deployment[0]>.95);assert.ok(a.wings.power[0]>.95);
 assert.notEqual(a.ctrl[0],b.ctrl[0]);
});

test('learned scales change only the thin flight interface',()=>{
 const {wings,q,ctrl}=fixture();
 wings.setInterpreterParameters({powerGain:.5,deploymentTauScale:2,frequencyScale:.9,steeringBiasGain:.5,steeringAmplitudeGain:2});
 wings.step(q,ctrl,1,0,{left:{},right:{}},.05);
 assert.ok(wings.power[0]<=.5);assert.equal(wings.frequencyHz,180);assert.equal(wings.config.frequency_hz,180);
 assert.deepEqual(wings.controlState().parameters,{powerGain:.5,deploymentTauScale:2,frequencyScale:.9,steeringBiasGain:.5,steeringAmplitudeGain:2});
 assert.throws(()=>wings.setInterpreterParameters({frequencyScale:0}));
 assert.throws(()=>wings.setInterpreterParameters({unknown:1}));
});
