import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {FlyBodyWings} from '../flybody-wings.js';
const metadata=JSON.parse(fs.readFileSync(new URL('../../models/flybody-mujoco.json',import.meta.url)));
const neutral=Float64Array.from({length:57},()=>0);
for(const joint of metadata.joints)neutral[joint.qpos]=joint.neutral;
function response(power,span,steps=1000){
 const config=structuredClone(metadata);
 if(span!==undefined)config.wing_actuation.deployment_power_span=span;
 const wings=new FlyBodyWings(config),control=new Float64Array(56),steering={left:{},right:{}};
 for(let i=0;i<steps;i++)wings.step(neutral,control,power,power,steering,.0002);
 return {deployment:[...wings.deployment],power:[...wings.power],target:[...wings.target],control:[...control],phase:wings.phase};
}
test('zero deployment span preserves the absent-setting response across onset and high drive',()=>{
 for(const power of [0,.009,.01,.0100001,.05,.15,.5,1])assert.deepEqual(response(power,0),response(power,undefined));
});
test('continuous deployment has vanishing movement immediately above recruitment onset',()=>{
 const baseline=response(.01,.1),above=response(.010000001,.1);
 assert(above.deployment.every(value=>value<1e-12));
 assert(Math.max(...above.target.map((value,i)=>Math.abs(value-baseline.target[i])))<1e-10);
 assert(response(.010000001,0).deployment.every(value=>value>.99));
});
test('span keeps deployment bounded and progressively recruits opening',()=>{
 const powers=[.01,.02,.04,.06,.08,.1,.11,.5],responses=powers.map(power=>response(power,.1));
 assert(responses.every(r=>r.deployment.every(value=>value>=0&&value<=1)));
 assert(responses.every((r,i)=>i===0||r.deployment[0]>=responses[i-1].deployment[0]));
 assert(Math.abs(responses[3].deployment[0]-.5)<1e-6);
 assert(responses.at(-1).deployment.every(value=>value>.999));
});
