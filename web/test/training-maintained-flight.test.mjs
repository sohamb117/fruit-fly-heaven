import test from 'node:test';
import assert from 'node:assert/strict';
import {createMaintainedFlightScore,MAINTAINED_FLIGHT_CRITERIA} from '../training/maintained-flight-objective.js';
const dt=.002;
const obs=(height=10,verticalSpeed=0,extra={})=>({finite:true,externalForce:false,up:1,angularSpeed:0,speedCmPerSecond:Math.abs(verticalSpeed),verticalSpeed,height,radius:0,ceiling:50,wingPower:.9,environmentContacts:0,nonFootEnvironmentContacts:0,footSupportCount:0,footSupportFraction:0,...extra});
function run(make){const score=createMaintainedFlightScore(5,make(0));for(let i=1;i<=2500&&!score.state.reason;i++)score.step(make(i*dt),dt);return score.state;}
test('wingbeat velocity oscillations do not fragment stationary maintained flight',()=>{
 const amplitude=.0009,omega=2*Math.PI*235.813;
 const state=run(t=>obs(10+amplitude*Math.sin(omega*t),amplitude*omega*Math.cos(omega*t)));
 assert(state.success);assert(Math.abs(state.flightSeconds-4.952)<1e-8);assert(Math.abs(state.diagnostics.meanVerticalSpeed)<.04);
});
test('the same -1 cm/s threshold rejects sustained descent',()=>{
 const state=run(t=>obs(10-1.5*t,-1.5));assert(!state.success);assert.equal(state.reason,'time_limit');assert.equal(state.totalQualifiedFlightSeconds,0);
});
test('exact -1 cm/s tolerates subtraction roundoff, while a real violation fails',()=>{
 const boundary=run(t=>obs(10-t,-1));assert(boundary.success);assert(Math.abs(boundary.flightSeconds-4.952)<1e-8);
 const speed=-1-1e-7,below=run(t=>obs(10+speed*t,speed));assert(!below.success);assert.equal(below.totalQualifiedFlightSeconds,0);
});
test('slow descent already permitted by the threshold remains eligible',()=>{
 const state=run(t=>obs(10-.5*t,-.5));assert(state.success);assert(Math.abs(state.diagnostics.meanVerticalSpeed+.5)<1e-8);
});
test('ballistic freefall still fails support and cannot earn flight credit',()=>{
 const state=run(t=>obs(10-.5*981*t*t,-981*t));assert(!state.success);assert.equal(state.totalQualifiedFlightSeconds,0);assert.equal(state.reason,'outside_habitat');
});
test('release starts with empty windows and no setup or takeoff credit',()=>{
 const score=createMaintainedFlightScore(5,obs());for(let i=1;i<25;i++)score.step(obs(),dt);
 assert.equal(score.state.flightSeconds,0);score.step(obs(),dt);assert.equal(score.state.flightSeconds,dt);assert.equal(score.state.hasTakenOff,false);
});
test('contact and its height jump are excluded; a fresh window is required',()=>{
 const score=createMaintainedFlightScore(5,obs());for(let i=0;i<100;i++)score.step(obs(),dt);
 score.step(obs(20,0,{environmentContacts:1,nonFootEnvironmentContacts:1}),dt);assert.equal(score.state.flightSeconds,0);assert.equal(score.state.diagnostics.meanVerticalSpeed,null);
 score.step(obs(10),dt);assert.equal(score.state.diagnostics.supportWindowSeconds,0);
 for(let i=1;i<25;i++){score.step(obs(10),dt);assert.equal(score.state.flightSeconds,0);}
 score.step(obs(10),dt);assert.equal(score.state.flightSeconds,dt);assert.equal(score.state.diagnostics.meanVerticalSpeed,0);
});
test('sustained late descent invalidates an earlier long flight bout',()=>{
 const state=run(t=>t<=4?obs():obs(10-2*(t-4),-2));assert(!state.success);assert.equal(state.reason,'time_limit');assert.equal(state.flightSeconds,0);assert(state.bestFlightSeconds>3.9);
});
test('attitude, angular-rate and external-force rejection stay active',()=>{
 for(const patch of [{up:.5},{angularSpeed:30}]){const state=run(()=>obs(10,0,patch));assert(!state.success);assert.equal(state.totalQualifiedFlightSeconds,0);}
 const state=run(()=>obs(10,0,{externalForce:true}));assert.equal(state.reason,'unexpected_external_force');assert.equal(state.return,-10);
});
test('criterion explicitly identifies the new measurement without changing limits',()=>{
 assert.equal(MAINTAINED_FLIGHT_CRITERIA.version,2);assert.equal(MAINTAINED_FLIGHT_CRITERIA.verticalSpeedWindowSeconds,.05);assert.equal(MAINTAINED_FLIGHT_CRITERIA.flight.minimumVerticalSpeed,-1);assert.equal(MAINTAINED_FLIGHT_CRITERIA.flight.minimumContinuousSeconds,1);assert.equal(MAINTAINED_FLIGHT_CRITERIA.durationSeconds,5);
});
