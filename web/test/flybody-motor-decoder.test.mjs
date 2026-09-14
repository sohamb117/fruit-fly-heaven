import test from 'node:test';
import assert from 'node:assert/strict';
import {FlyBodyWings} from '../flybody-wings.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';

// Pure command-generation fixtures: no native engine or physics stepping.
function fixture(withReference = false) {
  const joints = ['left', 'right'].flatMap((side, s) => ['yaw', 'roll', 'pitch'].map((axis, a) =>
    ({name: `wing_${axis}_${side}`, qpos: s * 3 + a, neutral: 0, range: [-2, 2]})));
  const steering = Object.fromEntries(STEERING_MUSCLE_TYPES.map(name => [name, [0, 0, 0, 0, 0, 0]]));
  steering.b1_muscle = [.1, .2, -.1, .05, .02, .03];
  const wing_actuation = {frequency_hz: 200, deployment_tau_s: .01, deployment_before_beating: 0,
    powers: [0, 1], steering,
    targets: [[Array(6).fill(.2), Array(6).fill(.2)], [Array(6).fill(.4), Array(6).fill(.6)]]};
  if (withReference) wing_actuation.steering_force_reference =
    Object.fromEntries(STEERING_MUSCLE_TYPES.map(name => [name, .4]));
  const wings = new FlyBodyWings({joints, actuators: joints.map((j, id) => ({name: j.name, id, range: [-2, 2]})),
    wing_actuation});
  wings.deployment.fill(1);
  return {wings, q: new Float64Array(6), ctrl: new Float64Array(6)};
}
const decoded = (power = [.6, .7]) => ({power, steering: [[0, 0, 0], [0, 0, 0]]});
const legacyResiduals = () => [new Float64Array(6), new Float64Array(6)];

test('zero learned steering ignores optional legacy force references at nonzero decoded power', () => {
  const plain = fixture(), referenced = fixture(true);
  for (const f of [plain, referenced]) f.wings.stepDecoded(f.q, f.ctrl, decoded(), .0002);
  assert.deepEqual(referenced.wings.residuals, legacyResiduals(), 'No legacy steering bias may enter decoded mode');
  assert.deepEqual(referenced.wings.target, plain.wings.target);
  assert.deepEqual(referenced.ctrl, plain.ctrl);
  assert.deepEqual(referenced.wings.power, plain.wings.power);
  assert.deepEqual(referenced.wings.deployment, plain.wings.deployment);
  assert.equal(referenced.wings.phase, plain.wings.phase);
});

test('ordinary muscle-force step still applies the declared legacy force reference', () => {
  const plain = fixture(), referenced = fixture(true);
  for (const f of [plain, referenced]) f.wings.step(f.q, f.ctrl, .6, .7, {}, .0002);
  assert.notDeepEqual(referenced.wings.residuals, legacyResiduals(), 'Fixture must actually exercise the reference path');
  assert.notDeepEqual(referenced.ctrl, plain.ctrl);
  assert.deepEqual(referenced.wings.power, plain.wings.power);
});

test('decoded mode clears previous legacy steering and preserves only its ipsilateral requested residual', () => {
  const plain = fixture(), referenced = fixture(true);
  for (const f of [plain, referenced]) f.wings.step(f.q, f.ctrl, .6, .7, {}, .0002);
  assert.notDeepEqual(referenced.wings.residuals, legacyResiduals());
  const command = decoded(); command.steering[0][1] = -.1;
  for (const f of [plain, referenced]) f.wings.stepDecoded(f.q, f.ctrl, command, .0002);
  assert.deepEqual(referenced.wings.residuals, legacyResiduals(), 'Old per-muscle residuals must be cleared each decoded call');
  assert.deepEqual(referenced.wings.target, plain.wings.target);
  assert.deepEqual(referenced.ctrl, plain.ctrl);
  assert.deepEqual(referenced.wings.decodedControls.appliedResidual, [[0, -.06, 0], [0, 0, 0]]);
});

test('silent decoder power and residuals retain the existing folding and position-restoring servo', () => {
  const f = fixture(true); f.q.fill(.4);
  f.wings.stepDecoded(f.q, f.ctrl, decoded([0, 0]), .0002);
  assert.deepEqual([...f.wings.power], [0, 0]);
  assert.deepEqual(f.wings.decodedControls.appliedResidual, [[0, 0, 0], [0, 0, 0]]);
  assert.ok(f.wings.deployment.every(value => value > 0 && value < 1));
  assert.ok(f.wings.target.every(value => value > 0 && value < .2), 'The deployed zero-power pose folds toward neutral');
  for (let k = 0; k < 6; k++) {
    assert.equal(f.ctrl[k], f.wings.target[k] - f.q[k]);
    assert.notEqual(f.ctrl[k], 0, 'Zero decoded drive does not imply zero total native actuator control');
  }
  // This is a command-level distinction, not a native torque/energy measurement.
});
