import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {FlyBodyFlightReference} from '../web/flybody-flight-reference.js';

const [mj,xml,metadata,policy,fixtures]=await Promise.all([loadMujoco(),fs.readFile('models/flybody-flight-reference.xml','utf8'),
 ...['models/flybody-flight-reference.json','models/flybody-flight-policy.json','models/flybody-flight-reference-fixtures.json'].map(p=>fs.readFile(p,'utf8').then(JSON.parse))]);
const report={date:new Date().toISOString(),scope:'Released FlyBody learned flight controller and original dynamics in MuJoCo WASM; starts airborne, floor contacts disabled, not BANC.',xmlSha256:metadata.source.xml_sha256,checks:[]};
const b=new FlyBodyFlightReference(mj,xml,metadata,policy);
const error={qpos:0,qvel:0,ctrl:0,observations:0};
for(const frame of fixtures.frames){
 b.step(frame.canonical_action);
 for(const key of ['qpos','qvel','ctrl'])error[key]=Math.max(error[key],...frame.state_after[key].map((v,i)=>Math.abs(v-b.data[key][i])));
 for(const [key,value] of Object.entries(frame.observation_after))error.observations=Math.max(error.observations,...value.flat().map((v,i)=>Math.abs(v-b.observation[key][i])));
 assert.equal(b.wingbeat.frequency_index,frame.wingbeat_state_after.frequency_index);assert.equal(b.wingbeat.step,frame.wingbeat_state_after.step);
}
assert(error.qpos<1e-8&&error.qvel<1e-5&&error.ctrl<1e-6&&error.observations<1e-3,JSON.stringify(error));
report.nativeInterfaceReplay=error;report.checks.push('Native controls, wingbeat phase, sensor observations and physical state match');
b.reset();
let maximumError=0,maximumAttitudeError=0,minimumHeight=Infinity,maximumHeight=-Infinity,maximumAppliedForce=0,maximumRootActuatorForce=0;
const trace=[],start=performance.now(),steps=5994;
for(let i=0;i<steps;i++){
 assert(b.step());
 const s=b.sample();assert(s.finite);
 maximumError=Math.max(maximumError,s.referenceError);maximumAttitudeError=Math.max(maximumAttitudeError,s.attitudeError);
 minimumHeight=Math.min(minimumHeight,s.position[2]);maximumHeight=Math.max(maximumHeight,s.position[2]);
 maximumAppliedForce=Math.max(maximumAppliedForce,s.maximumAppliedForce);maximumRootActuatorForce=Math.max(maximumRootActuatorForce,s.maximumRootActuatorForce);
 if(i%25===0||i===steps-1)trace.push(s);
}
report.closedLoop={seconds:b.data.time,steps,wallSeconds:(performance.now()-start)/1000,maximumReferenceErrorCm:maximumError,maximumAttitudeErrorDegrees:maximumAttitudeError,minimumHeightCm:minimumHeight,maximumHeightCm:maximumHeight,maximumAppliedForce,maximumRootActuatorForce};
assert.equal(b.terminationReason,'trajectory complete');
assert(maximumError<.08&&maximumAttitudeError<15&&minimumHeight>.9&&maximumHeight<1.1,JSON.stringify(report.closedLoop));
assert.equal(maximumAppliedForce,0);assert.equal(maximumRootActuatorForce,0);
report.checks.push('One fly maintains 1.1988 seconds of reference flight with zero external root forces');
b.reset();for(let i=0;i<500&&!b.done;i++)b.step(undefined,{actuated:false});
report.disconnected=b.sample();assert(report.disconnected.position[2]<metadata.termination.minimum_height_cm);assert.equal(Math.max(...Array.from(b.data.ctrl,Math.abs)),0);
assert.equal(b.terminationReason,'below minimum height');
report.checks.push('Disconnecting actuation removes support: the same model falls under gravity');
b.dispose();report.passed=true;
await fs.writeFile('reports/flybody-reference-validation.json',JSON.stringify({...report,trace},null,2)+'\n');
console.log(JSON.stringify(report,null,2));
