import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
import {FlyBodyWings} from '../flybody-wings.js';
import {makeFlightTelemetry} from '../training/flight-telemetry.js';

const metadata = JSON.parse(fs.readFileSync(new URL('../../models/flybody-mujoco.json', import.meta.url)));
const io = JSON.parse(fs.readFileSync(new URL('../../data/prepared/banc888/io.json', import.meta.url)));
const records = table => table.rows.map(row => Object.fromEntries(table.columns.map((name, index) => [name, row[index]])));

function fixture() {
  const nu = Math.max(...metadata.actuators.map(a => a.id)) + 1;
  const nq = Math.max(...metadata.joints.map(j => j.qpos)) + 1;
  const nv = Math.max(...metadata.joints.map(j => j.dof)) + 1;
  const model = {jnt_type: new Int32Array([0]), jnt_qposadr: new Int32Array([0]), jnt_dofadr: new Int32Array([0]),
    actuator_gear: new Float64Array(nu * 6), actuator_gainprm: new Float64Array(nu * 10)};
  for (const a of metadata.actuators) {model.actuator_gear[a.id * 6] = 1; model.actuator_gainprm[a.id * 10] = a.gain;}
  const data = {time: .072, qpos: Float64Array.from({length: nq}, (_, i) => i / 100),
    qvel: Float64Array.from({length: nv}, (_, i) => 10 + i), qacc: Float64Array.from({length: nv}, (_, i) => -20 - i),
    ctrl: Float64Array.from({length: nu}, (_, i) => i / 1000), actuator_force: Float64Array.from({length: nu}, (_, i) => 30 + i)};
  const body = {time: .072, model, data, mappings: io.muscles, actuators: new Map(metadata.actuators.map(a => [a.name, a])),
    input: Float64Array.from({length: io.muscles.length * 5}, (_, i) => i / 1000),
    muscleState: Float64Array.from({length: io.muscles.length * 3}, (_, i) => i / 2000), wings: new FlyBodyWings(metadata),
    mj: new Proxy({}, {get() {throw new Error('Observation must not call native simulation methods');}})};
  const rates = new Map(io.motor_neurons.map((motor, i) => [motor.index, (i % 100) + .125]));
  return {body, rates};
}

test('prepared sparse BANC identities, state strides, all wing muscles and six claws map exactly', () => {
  const {body, rates} = fixture(), result = makeFlightTelemetry(body, rates);
  const wing = records(result.wingMuscles), claws = records(result.claws);
  assert.equal(wing.length, 28); assert.equal(claws.length, 6); assert.equal(result.finite, true);
  for (const row of [...wing, ...claws]) {
    const mapping = io.muscles[row.mappingIndex];
    assert.equal(row.joint, mapping.joint); assert.equal(row.target, mapping.target); assert.equal(row.motorCount, mapping.indices.length);
    assert.equal(row.requestedRateHz, mapping.indices.reduce((sum, id) => sum + rates.get(id), 0) / mapping.indices.length);
    assert.equal(row.excitationInput, body.input[row.mappingIndex * 5]);
    assert.equal(row.activationState, body.muscleState[row.mappingIndex * 3]);
    assert.equal(row.fatigueState, body.muscleState[row.mappingIndex * 3 + 1]);
    assert.equal(row.normalizedMuscleForce, body.muscleState[row.mappingIndex * 3 + 2]);
  }
  assert.deepEqual(claws.map(row => row.joint), ['left', 'right'].flatMap(side => [1, 2, 3].map(leg => `adhere_claw_T${leg}_${side}`)));
  for (const row of claws) {
    const actuator = body.actuators.get(row.joint);
    assert.equal(row.nativeAdhesionControl, body.data.ctrl[actuator.id]);
    assert.equal(row.nativeAdhesionActuatorForce, body.data.actuator_force[actuator.id]);
    assert.equal(row.nativeAdhesionGain, body.model.actuator_gainprm[actuator.id * 10]);
  }
  // An array readback's ordinal position must not substitute for a BANC index.
  const dlm = wing.find(row => row.target === 'dorsal_longitudinal_muscle' && row.side === 'left');
  const actualId = io.muscles[dlm.mappingIndex].indices[0];
  const ordinal = io.motor_neurons.findIndex(motor => motor.index === actualId);
  assert.notEqual(actualId, ordinal);
  rates.set(ordinal, 9000);
  assert.equal(records(makeFlightTelemetry(body, rates).wingMuscles).find(row => row.mappingIndex === dlm.mappingIndex).requestedRateHz, dlm.requestedRateHz);
});

test('wing native addresses, raw controls and retained actuator forces are observed without recomputation', () => {
  const {body, rates} = fixture();
  body.wings.target.set([.12, .23, .34, .45, .56, .67]);
  const result = makeFlightTelemetry(body, rates), rows = records(result.wingJoints);
  assert.equal(rows.length, 6);
  rows.forEach((row, i) => {
    const joint = body.wings.joints[i], actuator = body.wings.actuators[i];
    assert.equal(row.name, joint.name); assert.equal(row.actuatorId, actuator.id);
    assert.equal(row.targetAngleRad, body.wings.target[i]); assert.equal(row.actualAngleRad, body.data.qpos[joint.qpos]);
    assert.equal(row.velocityRadPerSecond, body.data.qvel[joint.dof]);
    assert.equal(row.control, body.data.ctrl[actuator.id]); assert.equal(row.nativeActuatorForce, body.data.actuator_force[actuator.id]);
  });
  assert.deepEqual(result.root.qpos, Array.from(body.data.qpos.slice(0, 7)));
  assert.deepEqual(result.root.qvel, Array.from(body.data.qvel.slice(0, 6)));
  assert.deepEqual(result.root.qacc, Array.from(body.data.qacc.slice(0, 6)));
  assert.match(result.sampling.nativeForces, /No mj_forward/);
  assert.match(result.sampling.adhesion, /not measured vertical foot support/);
});

test('ready frame retains absent commands, and stepped frame distinguishes request, deployment ramp and effective power', () => {
  const {body, rates} = fixture();
  let result = makeFlightTelemetry(body, rates);
  assert.equal(result.wing.controlHasStepped, false); assert.equal(result.finite, true);
  for (const row of records(result.wing.sides)) {assert.equal(row.muscleDrive, null); assert.equal(row.requestedPower, null);}
  body.wingDriveLeft = .3; body.wingDriveRight = .9;
  body.wings.setInterpreterParameters({powerGain: 1.5});
  body.wings.deployment.set([.925, 1]); body.wings.power.set([.45, .8]); body.wings.phase = 1.234;
  result = makeFlightTelemetry(body, rates);
  assert.equal(result.wing.controlHasStepped, true); assert.equal(result.wing.phaseRad, 1.234);
  const [left, right] = records(result.wing.sides);
  assert.equal(left.muscleDrive, .3); assert.equal(left.requestedPower, .3 * 1.5);
  assert.equal(right.requestedPower, 1); assert.equal(left.deployment, .925);
  const gate = body.wings.config.deployment_before_beating;
  assert.equal(left.deploymentRamp, (.925 - gate) / (1 - gate));
  assert.equal(right.deploymentRamp, 1); assert.equal(right.effectivePower, .8);
});

test('all observations are detached and topology caching never caches dynamic body state', () => {
  const {body, rates} = fixture();
  const before = {data: structuredClone(body.data), input: body.input.slice(), state: body.muscleState.slice(),
    parameters: body.wings.controlState(), phase: body.wings.phase, target: body.wings.target.slice(), rates: new Map(rates)};
  const first = makeFlightTelemetry(body, rates);
  for (let i = 0; i < 4; i++) makeFlightTelemetry(body, rates);
  assert.deepEqual(body.data, before.data); assert.deepEqual(body.input, before.input); assert.deepEqual(body.muscleState, before.state);
  assert.deepEqual(body.wings.controlState(), before.parameters); assert.equal(body.wings.phase, before.phase);
  assert.deepEqual(body.wings.target, before.target); assert.deepEqual(rates, before.rates);
  first.root.qacc[0] = 900; first.wingMuscles.rows[0][10] = 999;
  assert.deepEqual(body.data, before.data); assert.deepEqual(body.muscleState, before.state);
  body.data.qacc[0] = -55; body.muscleState[first.wingMuscles.rows[0][0] * 3 + 2] = .77;
  const next = makeFlightTelemetry(body, rates);
  assert.equal(next.root.qacc[0], -55); assert.equal(records(next.wingMuscles)[0].normalizedMuscleForce, .77);
});

test('integration errors cannot silently fabricate missing rate identities or incompatible native layout', () => {
  const {body, rates} = fixture();
  assert.throws(() => makeFlightTelemetry(body, new Float32Array(100)), /Map keyed by BANC/);
  const wingMapping = io.muscles.find(mapping => mapping.kind === 'asynchronous_wing'); rates.delete(wingMapping.indices[0]);
  assert.throws(() => makeFlightTelemetry(body, rates), /Missing BANC motor rate/);
  const a = fixture(); a.body.model.jnt_dofadr[0] = 2;
  assert.throws(() => makeFlightTelemetry(a.body, a.rates), /free-root layout/);
  const b = fixture(); b.body.model.actuator_gear[b.body.wings.actuators[0].id * 6] = 2;
  assert.throws(() => makeFlightTelemetry(b.body, b.rates), /unit-gear/);
});

test('nonfinite native and muscle measurements survive JSON as explicit invalid observations', () => {
  const {body, rates} = fixture(), mappingIndex = io.muscles.findIndex(mapping => mapping.kind === 'asynchronous_wing');
  body.data.qacc[3] = Infinity; body.data.qpos[0] = NaN; body.muscleState[mappingIndex * 3 + 1] = NaN;
  rates.set(io.muscles[mappingIndex].indices[0], -2);
  const result = makeFlightTelemetry(body, rates);
  assert.equal(result.finite, false); assert.equal(result.root.qacc[3], null); assert.equal(result.root.qpos[0], null);
  const row = records(result.wingMuscles).find(row => row.mappingIndex === mappingIndex);
  assert.equal(row.fatigueState, null); assert.equal(row.requestedRateHz, null);
  assert(result.invalidFields.includes('root.qacc[3]')); assert(result.invalidFields.includes(`muscle[${mappingIndex}].fatigue`));
  assert.deepEqual(JSON.parse(JSON.stringify(result)), result);
});

test('sampling cost is measured separately from native simulation', t => {
  const {body, rates} = fixture();
  for (let i = 0; i < 100; i++) makeFlightTelemetry(body, rates);
  const start = performance.now();
  for (let i = 0; i < 1000; i++) makeFlightTelemetry(body, rates);
  const milliseconds = (performance.now() - start) / 1000;
  t.diagnostic(`Synthetic-array snapshot: ${milliseconds.toFixed(3)} ms/call, ${JSON.stringify(makeFlightTelemetry(body, rates)).length} JSON bytes. Native binding and browser costs are not measured.`);
});
