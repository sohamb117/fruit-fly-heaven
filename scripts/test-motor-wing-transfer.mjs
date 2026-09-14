import test from 'node:test';
import assert from 'node:assert/strict';
import {worldComWrench, equalityRootForce, Moments, rateAt, groupRate, makeCases, makeFollowupCases, makeContinuousCases, readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

test('COM wrench rotates root-local torque and translates the reference point', () => {
  const rotation = [0, -1, 0, 1, 0, 0, 0, 0, 1];
  assert.deepEqual(worldComWrench([2, 0, 0, 1, 0, 3], rotation, [0, 0, 0], [0, 2, 0]), [2, 0, 0, 0, 1, 7]);
});
test('restraint wrench excludes contacts and other equality rows', () => {
  const data = {nefc: 3, efc_type: [0, 6, 0], efc_id: [0, 0, 1], efc_force: [2, 99, 99], efc_J: [...[1, 2, 3, 4, 5, 6], ...Array(12).fill(1)]};
  assert.deepEqual(equalityRootForce(data, 6, 0), [2, 4, 6, 8, 10, 12]);
});
test('motor channels and ramp distinguish requested rate from excitation saturation', () => {
  assert.equal(rateAt(.075, 100, 'ramp'), 50);
  assert.equal(rateAt(.2, 100, 'ramp'), 100);
  assert.equal(groupRate({kind: 'asynchronous_wing', target: 'dorsoventral_muscle', joint: 'wing_right'}, 80, 'dlm'), 0);
  assert.equal(groupRate({kind: 'asynchronous_wing', target: 'dorsoventral_muscle', joint: 'wing_right'}, 80, 'dvm'), 80);
  assert.equal(groupRate({kind: 'wing_steering_assumption'}, 80, 'all'), 0);
});
test('aggregate native samples retain signed mean and RMS', () => {
  const m = new Moments(2); m.add([-2, 3]); m.add([2, 3]);
  assert.deepEqual(m.json(), {samples: 2, mean: [0, 3], rms: [2, 3], minimum: [-2, 3], maximum: [2, 3]});
  assert.throws(() => m.add([NaN, 0]));
});
test('all free-body cases exceed one second and include unilateral controls', () => {
  const cases = makeCases();
  assert(cases.filter(c => c.context !== 'restrained').every(c => c.seconds >= 1));
  assert(cases.some(c => c.context === 'air' && c.selection === 'left'));
  assert(cases.some(c => c.steering === 'iii1_both'));
  assert.equal(new Set(cases.map(c => JSON.stringify(c))).size, cases.length);
});
test('followup changes one deployment time constant and samples both sides of onset', () => {
  const cases = makeFollowupCases();
  assert.equal(cases.length, 17);
  assert(cases.some(c => c.rateHz === .8)); assert(cases.some(c => c.rateHz === .81));
  assert.deepEqual([...new Set(cases.filter(c => c.deploymentTau).map(c => c.deploymentTau))], [.012, .06, .12]);
});
test('continuous candidates include zero-span controls and never change the time constant',()=>{
  const cases=makeContinuousCases();assert.equal(cases.length,27);
  assert(cases.every(c=>c.deploymentTau===undefined));
  assert.deepEqual([...new Set(cases.map(c=>c.deploymentPowerSpan))],[0,.1,.25]);
});
test('warning reader disposes owned entries and retains the borrowed vector',()=>{
  let disposed=0;
  const names=['mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL'];
  const enums=Object.fromEntries(names.map((name,i)=>[name,{value:i+3}]));
  const vector={get(index){return {number:index===6?1:0,lastinfo:7,delete(){disposed++;}};},delete(){throw new Error('Borrowed vector must not be deleted');}};
  assert.deepEqual(readInstabilityWarnings(vector,enums),[{name:'mjWARN_BADCTRL',number:1,lastinfo:7}]);
  assert.equal(disposed,4);
});
test('native invalid-control warning remains readable without deleting the borrowed vector',async()=>{
  const {default:load}=await import('../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js');
  const mj=await load(),model=mj.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="hinge"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'),data=new mj.MjData(model);
  try{
    const vector=data.warning;data.ctrl[0]=NaN;mj.mj_step(model,data);
    for(let i=0;i<2;i++)assert(readInstabilityWarnings(vector,mj.mjtWarning).some(w=>w.name==='mjWARN_BADCTRL'&&w.number>0));
  }finally{data.delete();model.delete();}
});
