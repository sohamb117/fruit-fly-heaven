import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createMotorExcitation, STEERING_RECRUITMENT_BOUNDS} from '../flybody-motor-excitation.js';
import {FlyBodyPhysics} from '../flybody-physics.js';
import {createWasmCore, WasmMuscles} from '../../packages/banc-runtime/src/wasm.js';
import loadMujoco from '../../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';

const kind = 'wing_steering_assumption';
const config = (halfActivationHz = 80, exponent = 1) => ({steering: {kind: 'hill', halfActivationHz, exponent}});
const legacy = rate => Math.max(0, Math.min(1, rate / 80));

test('absent recruitment metadata exactly preserves the original clamp for all muscle kinds', () => {
  const decoder = createMotorExcitation();
  const rates = [-Infinity, -100, -Number.MIN_VALUE, -0, 0, Number.MIN_VALUE,
    .1, 79.99999, 80, 80.00001, 100, 250, Number.MAX_VALUE, Infinity, NaN];
  for (let rate = -10; rate <= 500; rate += .125) rates.push(rate);
  for (const type of [kind, 'asynchronous_wing', 'leg', 'claw_grip_assumption', 'haltere_steering_assumption', 'unknown'])
    for (const rate of rates) assert(Object.is(decoder.fromRate(type, rate), legacy(rate)), `${type}: ${rate}`);
  assert.equal(decoder.steering, null);
});

test('Hill recruitment retains distinction above 80 Hz and matches half activation', () => {
  const decoder = createMotorExcitation(config());
  assert.equal(decoder.fromRate(kind, 80), .5);
  assert.equal(decoder.fromRate(kind, 100), 1 / 1.8);
  assert.equal(decoder.fromRate(kind, 250), 1 / 1.32);
  const rates = [.00001, .1, 1, 10, 40, 79, 80, 81, 100, 160, 250, 500, 1000];
  for (const half of STEERING_RECRUITMENT_BOUNDS.halfActivationHz)
    for (const exponent of [...STEERING_RECRUITMENT_BOUNDS.exponent, 1]) {
      const curve = createMotorExcitation(config(half, exponent));
      const values = rates.map(rate => curve.fromRate(kind, rate));
      assert(values.every(value => value > 0 && value < 1));
      assert(values.every((value, index) => !index || value > values[index - 1]));
      assert.equal(curve.fromRate(kind, half), .5);
    }
});

test('Hill recruitment has bounded limits without numerical overflow', () => {
  for (const half of [1, 80, 1000]) for (const exponent of [.25, 1, 4]) {
    const curve = createMotorExcitation(config(half, exponent));
    for (const rate of [-Number.MAX_VALUE, -0, 0]) assert.equal(curve.fromRate(kind, rate), 0);
    for (const rate of [Number.MIN_VALUE, 1e-250, 1e250, Number.MAX_VALUE]) {
      const value = curve.fromRate(kind, rate);
      assert(Number.isFinite(value) && value >= 0 && value <= 1);
    }
    for (const rate of [NaN, Infinity, -Infinity, '80', undefined])
      assert.throws(() => curve.fromRate(kind, rate), /finite rate/);
  }
});

test('opt-in changes only wing steering and does not couple left/right calls', () => {
  const decoder = createMotorExcitation(config());
  const left = decoder.fromRate(kind, 250), right = decoder.fromRate(kind, 100);
  assert(left > right);
  decoder.fromRate(kind, 0);
  assert.equal(decoder.fromRate(kind, 250), left);
  assert.equal(decoder.fromRate(kind, 100), right);
  for (const type of ['asynchronous_wing', 'leg', 'claw_grip_assumption', 'asynchronous_haltere',
    'haltere_steering_assumption', 'proboscis', 'unknown'])
    for (const rate of [-2, 0, 40, 80, 100, 250, Infinity, NaN])
      assert(Object.is(decoder.fromRate(type, rate), legacy(rate)));
});

test('compiled priors own frozen configuration and ignore subsequent metadata mutation', () => {
  const source = config(), decoder = createMotorExcitation(source), independent = createMotorExcitation(config(160));
  source.steering.halfActivationHz = 1000;
  source.steering.exponent = 4;
  assert.equal(decoder.fromRate(kind, 80), .5);
  assert.equal(independent.fromRate(kind, 160), .5);
  assert(Object.isFrozen(decoder) && Object.isFrozen(decoder.steering));
  assert.notEqual(decoder.steering, source.steering);
  assert.throws(() => {decoder.steering.halfActivationHz = 1;}, TypeError);
});

test('malformed or unbounded priors are rejected instead of silently defaulted', () => {
  for (const source of [null, false, [], 'hill', {}, {steering: null}, {steering: []},
    {...config(), power: {}}, {steering: {...config().steering, gain: 1}},
    {steering: {kind: 'hill', halfActivationHz: 80}},
    {steering: {...config().steering, kind: 'linear'}},
    Object.assign(config(), {[Symbol('unknown')]: 1})])
      assert.throws(() => createMotorExcitation(source));
  for (const half of [NaN, Infinity, -Infinity, '80', 0, .999, 1000.001])
    assert.throws(() => createMotorExcitation(config(half)), /halfActivationHz/);
  for (const exponent of [NaN, Infinity, -Infinity, '1', 0, .2499, 4.0001])
    assert.throws(() => createMotorExcitation(config(80, exponent)), /exponent/);
});

let nativeAssets;
async function nativeFixture() {
  nativeAssets ??= Promise.all([loadMujoco(), createWasmCore(),
    fs.readFile(new URL('../../models/flybody-mujoco.xml', import.meta.url), 'utf8'),
    fs.readFile(new URL('../../models/flybody-mujoco.json', import.meta.url), 'utf8').then(JSON.parse),
    fs.readFile(new URL('../../data/prepared/banc888/io.json', import.meta.url), 'utf8').then(JSON.parse)]);
  const [mj, core, xml, metadata, io] = await nativeAssets;
  const model = mj.MjModel.from_xml_string(xml), bodies = [];
  const make = calibration => {
    const meta = {...metadata};
    if (calibration !== undefined) meta.motor_excitation = calibration;
    const body = new FlyBodyPhysics(mj, model, meta, io, count => new WasmMuscles(core, count),
      {surface: () => 0, odor: () => 0, foodAt: () => null});
    bodies.push(body);
    return body;
  };
  return {mj, model, io, make, dispose() {for (const body of bodies) body.dispose(); model.delete();}};
}

const bytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);

test('native default trajectory is byte-identical to the original clamp reference', async () => {
  const fixture = await nativeFixture();
  try {
    const current = fixture.make(), reference = fixture.make();
    reference.motorExcitation = {fromRate: (_kind, rateHz) => Math.max(0, Math.min(1, rateHz / 80))};
    for (let block = 0; block < 8; block++) {
      const rates = new Map(fixture.io.motor_neurons.map((motor, i) =>
        [motor.index, Math.fround(((i * 37 + block * 23) % 351) - 1)]));
      current.step(rates, .002); reference.step(rates, .002);
      for (const field of ['qpos', 'qvel', 'act', 'ctrl'])
        assert.deepEqual(bytes(current.data[field]), bytes(reference.data[field]), `native ${field}: block ${block}`);
      for (const field of ['activation', 'input', 'muscleState'])
        assert.deepEqual(bytes(current[field]), bytes(reference[field]), `muscle ${field}: block ${block}`);
      assert.deepEqual(current.internal, reference.internal);
      assert.deepEqual(current.wings.controlState(), reference.wings.controlState());
    }
  } finally {fixture.dispose();}
});

test('native adapter changes steering excitation only and preserves bilateral input identity', async () => {
  const fixture = await nativeFixture();
  try {
    const original = fixture.make(), calibrated = fixture.make(config());
    const rates = new Map(fixture.io.motor_neurons.map((motor, i) => [motor.index, Math.fround(20 + i % 260)]));
    const left = fixture.io.muscles.findIndex(m => m.target === 'b2_muscle' && m.joint === 'wing_steer_left');
    const right = fixture.io.muscles.findIndex(m => m.target === 'b2_muscle' && m.joint === 'wing_steer_right');
    assert(left >= 0 && right >= 0);
    for (const id of fixture.io.muscles[left].indices) rates.set(id, 80);
    for (const id of fixture.io.muscles[right].indices) rates.set(id, 250);
    // One native muscle interval starts from identical geometry/internal state.
    // Later geometry feedback is intentionally not asserted identical.
    original.step(rates, .001); calibrated.step(rates, .001);
    let changed = 0;
    for (let i = 0; i < fixture.io.muscles.length; i++) {
      const mapping = fixture.io.muscles[i];
      assert.deepEqual(calibrated.input.slice(i * 5 + 1, i * 5 + 5), original.input.slice(i * 5 + 1, i * 5 + 5));
      if (mapping.kind !== kind) {
        assert.equal(calibrated.activation[i], original.activation[i], mapping.target);
        assert.deepEqual(calibrated.muscleState.slice(i * 3, i * 3 + 3), original.muscleState.slice(i * 3, i * 3 + 3));
      } else {
        const rateHz = mapping.indices.reduce((sum, id) => sum + rates.get(id), 0) / mapping.indices.length;
        assert.equal(calibrated.input[i * 5], Math.fround(calibrated.motorExcitation.fromRate(kind, rateHz)));
        changed += Number(calibrated.activation[i] !== original.activation[i]);
      }
    }
    assert.equal(changed, 24);
    assert.equal(calibrated.activation[left], .5);
    assert.equal(calibrated.activation[right], Math.fround(1 / 1.32));
    assert.equal(calibrated.wingDriveLeft, original.wingDriveLeft);
    assert.equal(calibrated.wingDriveRight, original.wingDriveRight);
    const before = calibrated.activation.slice();
    for (const id of fixture.io.muscles[right].indices) rates.set(id, 0);
    calibrated.step(rates, .0001); // Below a native muscle interval: input dispatch only.
    assert.equal(calibrated.activation[right], 0);
    assert.equal(calibrated.activation[left], before[left]);
    for (let i = 0; i < before.length; i++) if (i !== right) assert.equal(calibrated.activation[i], before[i]);
    calibrated.step(rates, .0001, {coupling: false});
    assert(calibrated.activation.every(value => value === 0));
  } finally {fixture.dispose();}
});

test('invalid metadata is rejected before allocating native body or muscles', () => {
  let allocated = false;
  const mj = {MjData: class {constructor() {allocated = true;}}};
  assert.throws(() => new FlyBodyPhysics(mj, null, {motor_excitation: config(0)}), /halfActivationHz/);
  assert.equal(allocated, false);
});
