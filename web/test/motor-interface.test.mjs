import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_MOTOR_INTERFACE,validateMotorInterface,motorInterfaceBuffers,configureMotorInterface} from '../motor-interface.js';
const mappings=[
 {kind:'asynchronous_wing',target:'dorsal_longitudinal_muscle'},
 {kind:'asynchronous_wing',target:'dorsoventral_muscle'},
 {kind:'wing_steering_assumption'},
 {kind:'asynchronous_haltere'},
 {kind:'claw_grip_assumption',joint:'adhere_claw_T3_left'},
 {kind:'claw_grip_assumption',joint:'adhere_claw_T1_right'},
 {kind:'leg'},
];
test('motor interface defaults retain original rates, forces and muscle kinetics',()=>{
 const b=motorInterfaceBuffers(mappings);assert.deepEqual([...b.rates],Array(7).fill(80));assert.deepEqual([...b.forces],Array(7).fill(1));
 assert.deepEqual([...b.kinetics],Array.from({length:7},()=>[.015,.04,.08,.03]).flat().map(Math.fround));
 assert.deepEqual(validateMotorInterface(),DEFAULT_MOTOR_INTERFACE);
});
test('power rate, steering rate, kinetics and grip force have separate anatomical scope',()=>{
 const b=motorInterfaceBuffers(mappings,{dlmRateHz:8,dvmRateHz:12,steeringRateHz:200,powerRiseSeconds:.2,powerFallSeconds:1,hindGripScale:.25});
 assert.deepEqual([...b.rates],[8,12,200,80,80,80,80]);assert.deepEqual([...b.forces],[1,1,1,1,.25,1,1]);
 assert.equal(b.kinetics[0],Math.fround(.2));assert.equal(b.kinetics[4],Math.fround(.2));assert.equal(b.kinetics[8],Math.fround(.015));
});
test('unknown anatomy, invented fields and unbounded parameters fail closed',()=>{
 for(const value of [{foo:1},{dlmRateHz:0},{hindGripScale:NaN},{activeServoScale:Infinity},{deploymentThreshold:1},{deploymentPowerSpan:-.1},{deploymentPowerSpan:.51},null,[]])assert.throws(()=>validateMotorInterface(value));
 assert.throws(()=>motorInterfaceBuffers([{kind:'asynchronous_wing',target:'unknown'}]));
 assert.throws(()=>motorInterfaceBuffers([{kind:'claw_grip_assumption',joint:'wrong'}]));
});
test('configuration is per episode and does not mutate shared wing metadata',()=>{
 const config={deployment_tau_s:.012,deployment_before_beating:.85},body={time:0,data:{time:0},mappings,wings:{config},muscles:{dispose(){this.disposed=true;}}};
 const old=body.muscles;configureMotorInterface(body,{hindGripScale:.25,deploymentSeconds:.06,deploymentPowerSpan:.1},()=>({dispose(){}}));
 assert.equal(old.disposed,true);assert.equal(body.wings.config.deployment_tau_s,.06);assert.equal(config.deployment_tau_s,.012);
 assert.equal(body.wings.config.deployment_power_span,.1);assert.equal(config.deployment_power_span,undefined);
 assert.throws(()=>configureMotorInterface(body,{},()=>({})));
 assert.throws(()=>configureMotorInterface({...body,motorInterface:null,time:1}, {},()=>({})));
});
