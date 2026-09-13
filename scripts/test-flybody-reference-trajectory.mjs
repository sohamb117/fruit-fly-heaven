import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createBoundedFlightTrajectory} from '../web/flybody-reference-trajectory.js';

const original=JSON.parse(await fs.readFile('models/flybody-flight-reference.json','utf8'));
const originalInitial=JSON.stringify(original.initial_state),originalReference=JSON.stringify(original.reference.root_qpos.slice(0,6));
const hover=createBoundedFlightTrajectory(original,{durationSeconds:3600});
assert.equal(hover.initial_state,original.initial_state,'Physical initial state must remain shared and unchanged');
assert.equal(hover.episode.maxControlSteps,18_000_000);
assert(hover.reference.root_qpos[0].every((v,i)=>Math.abs(v-original.reference.root_qpos[0][i])<1e-12));
assert.equal(hover.reference.root_qpos.length,18_000_007);
const stop=Math.round(.3/original.control_timestep);
const final=hover.reference.center_of_mass_qpos[stop];
assert(Math.abs(final[0]-3)<1e-12);
assert.deepEqual(hover.reference.center_of_mass_qpos[stop+500],final);
assert.deepEqual(hover.reference.center_of_mass_qpos[18_000_000],final,'Long-duration hover must be stationary without trajectory resets');
assert.deepEqual(hover.reference.qvel[stop],[0,0,0,0,0,0]);
assert(Math.abs(hover.reference.qvel[stop-1][0])<.00003,'Braking must approach zero velocity smoothly');
const orbit=createBoundedFlightTrajectory(original,{kind:'orbit',radiusCm:2.5,durationSeconds:10});
let maximumRadius=0;
for(let i=0;i<50_000;i+=127){
 const row=orbit.reference.center_of_mass_qpos[i],radius=Math.hypot(row[0],row[1]-2.5);
 maximumRadius=Math.max(maximumRadius,radius);
 assert(Math.abs(radius-2.5)<1e-10);
 assert(Math.abs(Math.hypot(...row.slice(3))-1)<1e-12,'Desired orientation must remain normalized');
}
assert.equal(JSON.stringify(original.initial_state),originalInitial);
assert.equal(JSON.stringify(original.reference.root_qpos.slice(0,6)),originalReference);
assert.throws(()=>createBoundedFlightTrajectory(original,{kind:'teleport'}));
console.log(JSON.stringify({passed:true,longDurationSteps:hover.episode.maxControlSteps,hoverDistanceCm:final[0],maximumOrbitRadiusCm:maximumRadius,physicalStateUnchanged:true},null,2));
