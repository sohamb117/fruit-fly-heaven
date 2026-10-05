import test from 'node:test';
import assert from 'node:assert/strict';
import {FlyBodyWings} from '../flybody-wings.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';

const profile = {schemaVersion: 1, profile: 'activation-amplitude-v1', activationGain: 2};
function fixture(...optionalProfile) {
  const joints = ['left', 'right'].flatMap((side, s) => ['yaw', 'roll', 'pitch'].map((axis, a) =>
    ({name: `wing_${axis}_${side}`, qpos: s * 3 + a, neutral: 0, range: [-2, 2]})));
  const metadata = {joints, actuators: joints.map((j, id) => ({name: j.name, id, range: [-2, 2]})),
    wing_actuation: {frequency_hz: 200, deployment_tau_s: .01, deployment_before_beating: .9,
      powers: [0, 1], targets: [[[0, 0, 0, 0, 0, 0]], [[1, 1, 1, 1, 1, 1]]],
      steering: Object.fromEntries(STEERING_MUSCLE_TYPES.map(name => [name, [.1, 0, 0, 0, 0, 0]]))}};
  if (optionalProfile.length) metadata.wing_actuation.power_transfer = optionalProfile[0];
  const wings = new FlyBodyWings(metadata), q = new Float64Array(6), ctrl = new Float64Array(6);
  wings.deployment.fill(1);
  return {wings, metadata, q, ctrl, step: (left, right, steering = {}) => wings.step(q, ctrl, left, right, steering, .0002)};
}
const near = (actual, expected) => assert(Math.abs(actual - expected) < 1e-14, `${actual} != ${expected}`);

test('absent power profile retains the legacy gain-before-saturation contract', () => {
  const f = fixture();
  assert.equal(f.wings.powerTransfer, null);
  f.wings.setInterpreterParameters({powerGain: 2});
  f.step(.25, .8);
  assert.deepEqual([...f.wings.power], [.5, 1]);
  assert.deepEqual([...f.ctrl], [.5, .5, .5, 1, 1, 1]);
  f.step(0, 0);
  assert.deepEqual([...f.wings.power], [0, 0]);
});

test('amplitude stays responsive after activation saturates and preserves unequal side input', () => {
  const f = fixture(profile);
  for (const amplitude of [.8, .81, .9, 1]) {
    f.wings.setInterpreterParameters({powerGain: amplitude});
    f.step(.8, .95);
    assert.deepEqual([...f.wings.power], [amplitude, amplitude]);
    assert.deepEqual([...f.ctrl], Array(6).fill(amplitude));
  }
  f.wings.setInterpreterParameters({powerGain: .8});
  f.step(.1, .2);
  near(f.wings.power[0], .16); near(f.wings.power[1], .32);
  near(f.ctrl[0], .16); near(f.ctrl[3], .32);
});

test('zero power remains zero after deployment without borrowing opposite-side or steering drive', () => {
  const f = fixture(profile);
  f.wings.setInterpreterParameters({powerGain: .9});
  f.step(.8, .8);
  f.step(0, .8, {left: {b1_muscle: 1}});
  assert.deepEqual([...f.wings.power], [0, .9]);
  assert.deepEqual([...f.ctrl.slice(0, 3)], [0, 0, 0]);
  f.step(0, 0, {left: {b1_muscle: 1}, right: {b1_muscle: 1}});
  assert.deepEqual([...f.wings.power], [0, 0]);
  assert.deepEqual([...f.ctrl], Array(6).fill(0));
});

test('old gains above one are rejected transactionally only for the new profile', () => {
  const f = fixture(profile);
  f.wings.setInterpreterParameters({powerGain: .81, frequencyScale: .95});
  const previous = f.wings.interpreter, state = f.wings.controlState(), hz = f.wings.config.frequency_hz;
  for (const powerGain of [1 + Number.EPSILON, Math.exp(.693147), 2]) {
    assert.throws(() => f.wings.setInterpreterParameters({powerGain, frequencyScale: .7}), /powerGain <= 1/);
    assert.equal(f.wings.interpreter, previous);
    assert.deepEqual(f.wings.controlState(), state);
    assert.equal(f.wings.config.frequency_hz, hz);
  }
  f.wings.setInterpreterParameters({powerGain: 1});
  assert.equal(f.wings.interpreter.powerGain, 1);
  const legacy = fixture(); legacy.wings.setInterpreterParameters({powerGain: 2});
  assert.equal(legacy.wings.interpreter.powerGain, 2);
});

test('power profiles require an exact owned schema and retain a frozen snapshot', () => {
  for (const value of [undefined, null, [], {}, Object.create(profile),
    {...profile, schemaVersion: 2}, {...profile, profile: 'unknown'}, {...profile, extra: true},
    {...profile, [Symbol('extra')]: true}, ...[0, -1, NaN, Infinity, '2'].map(activationGain => ({...profile, activationGain}))])
    assert.throws(() => fixture(value), TypeError);
  const supplied = {...profile}, f = fixture(supplied);
  supplied.activationGain = 10;
  assert.deepEqual(f.wings.powerTransfer, profile);
  assert(Object.isFrozen(f.wings.powerTransfer));
  assert.notEqual(f.wings.powerTransfer, supplied);
  assert.equal(f.wings.config.power_transfer, f.wings.powerTransfer);
});

test('deployment threshold uses effective request while retaining the mechanical ramp', () => {
  const f = fixture(profile); f.wings.deployment.fill(0);
  f.wings.setInterpreterParameters({powerGain: .9});
  f.step(.00999 / 1.8, .01001 / 1.8);
  assert.equal(f.wings.deployment[0], 0);
  assert(f.wings.deployment[1] > 0);
  assert.deepEqual([...f.wings.power], [0, 0], 'Deployment must reach its existing gate before beating');
});
