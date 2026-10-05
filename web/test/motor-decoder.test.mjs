import test from 'node:test';
import assert from 'node:assert/strict';
import {MOTOR_DECODER_VERSION, buildMotorDecoderContract, validateMotorDecoderContract,
  validateMotorDecoderVector, createMotorDecoder} from '../motor-decoder.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';

// Compact synthetic anatomical identities; no prepared-data or report fixture.
function fixture() {
  let serial = 0;
  const muscles = [{kind: 'nonwing_fixture', target: 'leg_fixture', joint: 'leg_fixture'}], motor_neurons = [];
  function add(side, target, family, count) {
    const indices = Array.from({length: count}, () => 1000 + ((serial++ * 17) % 48) * 7);
    const root_ids = indices.map(index => (720575940000000000n + BigInt(index)).toString());
    muscles.push({kind: family === 'steering' ? 'wing_steering_assumption' : 'asynchronous_wing',
      joint: (family === 'steering' ? 'wing_steer_' : 'wing_power_') + side,
      target, sign: 1, indices, root_ids});
    indices.forEach((index, k) => motor_neurons.push({index, root_id: root_ids[k], side,
      peripheral_target_type: target, super_class: 'motor', region: 'ventral_nerve_cord'}));
  }
  for (const side of ['right', 'left']) {
    add(side, 'dorsal_longitudinal_muscle', 'dlm', 5);
    for (const target of STEERING_MUSCLE_TYPES) add(side, target, 'steering', 1);
    add(side, 'dorsoventral_muscle', 'dvm', 7);
  }
  return {muscles, motor_neurons};
}
const zeroOutput = {power: [0, 0], steering: [[0, 0, 0], [0, 0, 0]]};
const initial = contract => contract.parameters.map(p => p.initial);
const close = (actual, expected, tolerance = 1e-14) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `expected ${expected}, got ${actual}`);
const jsonCopy = value => JSON.parse(JSON.stringify(value));
const coefficient = (contract, unit, axis, lagMs, basis) => contract.parameters.findIndex(p =>
  p.kind === 'steering' && p.unitIndex === unit.index && p.axis === axis && p.lagMs === lagMs && p.basis === basis);

test('snapshot restores exact causal features and rejects mismatched parameters transactionally', () => {
  const io=fixture(),contract=buildMotorDecoderContract(io),weights=initial(contract);
  weights[24]=.1;
  const live=createMotorDecoder(io,weights),restored=createMotorDecoder(io,weights);
  for(let t=0;t<19;t++)live.advance(Array.from({length:48},(_,i)=>(i+t)%11/11));
  const saved=live.snapshot();restored.restore(saved);
  for(let t=0;t<7;t++){
    assert.deepEqual(restored.features(t*.7),live.features(t*.7));assert.deepEqual(restored.sample(t*.7),live.sample(t*.7));
    const excitation=Array.from({length:48},(_,i)=>(i+t)%7/7);live.advance(excitation);restored.advance(excitation);
  }
  const before=restored.snapshot(),bad=structuredClone(before);bad.weights[0]+=.01;
  assert.throws(()=>restored.restore(bad),/weights differ/);assert.deepEqual(restored.snapshot(),before);
  const badHistory=structuredClone(before);badHistory.history[2][5]=NaN;
  assert.throws(()=>restored.restore(badHistory));assert.deepEqual(restored.snapshot(),before);
  saved.history[0][0]=99;assert.notEqual(restored.snapshot().history[0][0],99);
});

test('contract preserves exact sorted unit identities and the bilateral 672-coordinate mask', () => {
  const io = fixture(), c = buildMotorDecoderContract(io);
  assert.equal(c.version, MOTOR_DECODER_VERSION);
  assert.equal(c.intervalMs, 1);
  assert.equal(c.indices.length, 48);
  assert.equal(c.muscles.length, 28);
  assert.equal(c.parameters.length, 672);
  assert.equal(c.powerParameterCount, 24);
  assert.equal(c.steeringParameterCount, 648);
  assert.deepEqual(c.indices, io.muscles.flatMap(m => m.indices || []).sort((a, b) => a - b));
  assert.deepEqual(c.units.map(u => u.position), Array.from({length: 48}, (_, k) => k));
  assert.equal(new Set(c.parameters.map(p => p.name)).size, 672);
  assert.deepEqual(c.parameters.slice(0, 24).map(p => p.unitIndex), c.units.filter(u => u.family !== 'steering').map(u => u.index));
  for (const u of c.units) {
    const ps = c.parameters.filter(p => p.unitIndex === u.index);
    assert.ok(ps.every(p => p.side === u.side && p.sideIndex === u.sideIndex && p.rootId === u.rootId &&
      p.unitPosition === u.position && p.name.includes(String(u.index))));
    if (u.family === 'steering') {
      assert.equal(ps.length, 27);
      assert.deepEqual(ps.map(p => [p.axis, p.lagMs, p.basis]),
        ['yaw', 'roll', 'pitch'].flatMap(axis => [0, 1, 4].flatMap(lag =>
          ['constant', 'sin', 'cos'].map(basis => [axis, lag, basis]))));
      assert.ok(ps.every(p => p.min === -.25 && p.max === .25 && p.initial === 0));
    } else {
      assert.equal(ps.length, 1);
      assert.equal(ps[0].normalization, u.family === 'dlm' ? 1 / 10 : 1 / 14);
      assert.deepEqual([ps[0].min, ps[0].max, ps[0].initial], [0, 4, 1]);
    }
  }
  const restored = validateMotorDecoderContract(jsonCopy(c));
  assert.deepEqual(restored, c);
  assert.ok(Object.isFrozen(restored.parameters[24]) && Object.isFrozen(restored.muscles[0].rootIds));
  io.muscles[1].indices[0] = 42;
  assert.notEqual(c.muscles[0].indices[0], 42);
});

test('unknown, duplicated, incomplete and inconsistent wing identities fail closed', () => {
  const mutations = [
    io => { io.muscles.pop(); },
    io => { io.muscles[1].indices.pop(); },
    io => { io.muscles[1].root_ids.pop(); },
    io => { io.muscles[1].indices[1] = io.muscles[1].indices[0]; },
    io => { io.muscles[1].root_ids[1] = io.muscles[1].root_ids[0]; },
    io => { io.muscles[1].root_ids[0] = 720575940000000000; },
    io => { io.muscles[1].indices[0] = 1.5; },
    io => { io.muscles[1].sign = -1; },
    io => { io.muscles[1].kind = 'unknown_wing_controller'; },
    io => { io.muscles[1].target = 'unknown_muscle'; },
    io => { io.muscles[1].joint = 'wing_power_middle'; },
    io => { io.motor_neurons[0].root_id = '1234'; },
    io => { io.motor_neurons[0].side = 'left'; },
    io => { io.motor_neurons[0].peripheral_target_type = 'other'; },
    io => { io.motor_neurons.pop(); },
    io => { io.motor_neurons.push({...io.motor_neurons[0]}); },
  ];
  for (const mutate of mutations) { const io = fixture(); mutate(io); assert.throws(() => buildMotorDecoderContract(io), /Motor decoder/); }
  assert.throws(() => buildMotorDecoderContract({}), /io.muscles/);
  const io = fixture(); assert.doesNotThrow(() => buildMotorDecoderContract({muscles: io.muscles}));
  const c = buildMotorDecoderContract(io);
  for (const mutate of [
    c => { c.bodyController = true; }, c => { c.parameters[24].axis = 'root_yaw'; },
    c => { c.parameters[24].side = c.parameters[24].side === 'left' ? 'right' : 'left'; },
    c => { c.parameters[0].min = -1; }, c => { c.units[0].position = 47; },
    c => { c.indices.reverse(); }, c => { c.muscles[0].unknown = 1; },
  ]) { const altered = jsonCopy(c); mutate(altered); assert.throws(() => validateMotorDecoderContract(altered), /Motor decoder/); }
});

test('fresh silence gives zero active output even with nonzero learned coefficients and arbitrary phase', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), vector = c.parameters.map(p => p.max);
  const d = createMotorDecoder(io, vector);
  for (const phase of [0, Math.PI / 2, -10, 1000]) assert.deepEqual(d.sample(phase), zeroOutput);
  assert.equal(d.advance(new Float64Array(48)), 1);
  assert.deepEqual(d.sample(.71), zeroOutput);
  assert.ok(d.features(.71).every(x => x === 0));
});

test('power uses individual nonnegative weights with equal DLM/DVM family means and bilateral clipping', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), v = initial(c);
  const x = Float64Array.from(c.units, u => u.side === 'left' ? (u.family === 'dlm' ? .2 : u.family === 'dvm' ? .8 : 0) :
    u.family === 'dlm' ? 1 : 0);
  const d = createMotorDecoder(io, v); d.advance(x);
  close(d.sample(0).power[0], .5); close(d.sample(0).power[1], .5);
  assert.deepEqual(d.sample(0).steering, zeroOutput.steering);
  const k = c.parameters.findIndex(p => p.kind === 'power' && p.side === 'left' && p.family === 'dlm');
  v[k] = 2;
  const weighted = createMotorDecoder(io, v); weighted.advance(x);
  close(weighted.sample(1).power[0], .52); close(weighted.sample(1).power[1], .5);
  v.fill(4, 0, 24);
  const saturated = createMotorDecoder(io, v); saturated.advance(x);
  assert.deepEqual(saturated.sample(0).power, [1, 1]);
  x.fill(0); saturated.advance(x);
  assert.deepEqual(saturated.sample(0).power, [0, 0]);
});

test('causal lag 0/1/4 ms and radians phase features are exact; sampling does not advance history', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), u = c.units.find(u => u.family === 'steering');
  const v = initial(c);
  v[coefficient(c, u, 'yaw', 0, 'constant')] = .1;
  v[coefficient(c, u, 'roll', 1, 'sin')] = .2;
  v[coefficient(c, u, 'pitch', 4, 'cos')] = -.2;
  const d = createMotorDecoder(io, v), x = new Float64Array(48); x[u.position] = .5;
  d.advance(x);
  assert.equal(d.timeMs, 1);
  assert.deepEqual(d.sample(Math.PI / 2).steering[u.sideIndex], [.05, 0, 0]);
  const first = d.features(0);
  assert.equal(first[coefficient(c, u, 'yaw', 0, 'constant')], .5);
  assert.equal(first[coefficient(c, u, 'yaw', 1, 'constant')], 0);
  for (let k = 0; k < 20; k++) d.sample(k);
  assert.equal(d.timeMs, 1);
  assert.deepEqual(d.features(0), first);
  x.fill(0); d.advance(x);
  assert.deepEqual(d.sample(Math.PI / 2).steering[u.sideIndex], [0, .1, 0]);
  d.advance(x); d.advance(x); d.advance(x);
  assert.equal(d.timeMs, 5);
  assert.deepEqual(d.sample(0).steering[u.sideIndex], [0, 0, -.1]);
  assert.equal(d.features(0)[coefficient(c, u, 'yaw', 4, 'constant')], .5);
  d.advance(x);
  assert.deepEqual(d.sample(.31), zeroOutput);
});

test('each steering unit affects only its own three axes; commands clip after summing', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), v = initial(c);
  for (let k = 24; k < v.length; k++) v[k] = c.parameters[k].axis === 'pitch' ? -.25 : .25;
  for (const side of ['left', 'right']) {
    const d = createMotorDecoder(io, v), x = Float64Array.from(c.units,
      u => u.side === side && u.family === 'steering' ? 1 : 0);
    for (let k = 0; k < 5; k++) d.advance(x);
    const result = d.sample(0), sideIndex = side === 'left' ? 0 : 1;
    assert.deepEqual(result.power, [0, 0]);
    assert.deepEqual(result.steering[sideIndex], [.25, .25, -.25]);
    assert.deepEqual(result.steering[1 - sideIndex], [0, 0, 0]);
  }
});

test('regression features reproduce every unclipped model output and own their storage', () => {
  const io = fixture(), c = buildMotorDecoderContract(io);
  const v = c.parameters.map((p, k) => p.kind === 'power' ? .3 + k / 100 : Math.sin(k) / 1000);
  const d = createMotorDecoder(io, v);
  for (let tick = 0; tick < 7; tick++) d.advance(Float64Array.from(c.units, (_, j) => ((j + tick * 3) % 17) / 20));
  for (const phase of [0, .7, -2, Math.PI]) {
    const f = d.features(phase), expected = {power: [0, 0], steering: [[0, 0, 0], [0, 0, 0]]};
    assert.ok(f instanceof Float64Array); assert.equal(f.length, 672);
    c.parameters.forEach((p, k) => {
      if (p.kind === 'power') expected.power[p.sideIndex] += v[k] * f[k];
      else expected.steering[p.sideIndex][p.axisIndex] += v[k] * f[k];
    });
    const actual = d.sample(phase);
    expected.power.forEach((value, side) => close(actual.power[side], value));
    expected.steering.forEach((values, side) => values.forEach((value, axis) => close(actual.steering[side][axis], value)));
    f.fill(NaN); assert.deepEqual(d.sample(phase), actual);
    actual.power.fill(123); actual.steering[0].fill(123);
    close(d.sample(phase).power[0], expected.power[0]);
  }
});

test('constructor and vector validator reject missing, nonfinite and out-of-bounds values; vectors are owned', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), base = initial(c);
  for (const bad of [undefined, [], new Float64Array(671), {...base, length: 672}])
    assert.throws(() => createMotorDecoder(io, bad), /parameter vector/);
  for (const [k, value] of [[0, -1e-6], [0, 4.00001], [24, -.250001], [24, .250001],
    [1, NaN], [1, Infinity], [1, '1'], [1, null]]) {
    const v = base.slice(); v[k] = value;
    assert.throws(() => validateMotorDecoderVector(c, v), /Motor decoder/);
    assert.throws(() => createMotorDecoder(io, v), /Motor decoder/);
  }
  const checked = validateMotorDecoderVector(jsonCopy(c), base);
  assert.ok(checked instanceof Float64Array);
  const d = createMotorDecoder(io, checked);
  checked.fill(0); base.fill(0);
  d.advance(new Float64Array(48).fill(.4));
  close(d.sample(0).power[0], .4); close(d.sample(0).power[1], .4);
});

test('invalid inputs do not advance or mutate history, and invalid phase cannot change state', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), d = createMotorDecoder(io, initial(c));
  const x = new Float64Array(48).fill(.3); d.advance(x);
  const before = d.features(.7);
  for (const bad of [[], new Float64Array(47), new DataView(new ArrayBuffer(48)), null,
    Array(48).fill(NaN), Array(48).fill(Infinity), Array(48).fill(-.001), Array(48).fill(1.001), Array(48).fill('0')]) {
    assert.throws(() => d.advance(bad), /Motor decoder/);
    assert.equal(d.timeMs, 1); assert.deepEqual(d.features(.7), before);
  }
  for (const phase of [NaN, Infinity, undefined, null, '0', {root: [0, 0, 0]}]) {
    assert.throws(() => d.sample(phase), /phase/); assert.throws(() => d.features(phase), /phase/);
  }
  x.fill(0); // Caller mutation must not rewrite consumed history.
  assert.deepEqual(d.features(.7), before);
});

test('instances own their history and reset clears all lagged activation and clock', () => {
  const io = fixture(), c = buildMotorDecoderContract(io), v = c.parameters.map(p => p.max);
  const a = createMotorDecoder(io, v), b = createMotorDecoder(io, v);
  a.advance(new Float64Array(48).fill(.5));
  assert.notDeepEqual(a.sample(.3), zeroOutput); assert.deepEqual(b.sample(.3), zeroOutput);
  for (let k = 0; k < 4; k++) a.advance(new Float64Array(48));
  assert.notDeepEqual(a.sample(.3), zeroOutput); // The 4 ms feature remains causal.
  a.reset(); assert.equal(a.timeMs, 0); assert.deepEqual(a.sample(.3), zeroOutput);
  assert.ok(a.features(.3).every(x => x === 0));
  a.advance(new Float64Array(48)); assert.deepEqual(a.sample(.3), zeroOutput);
  assert.equal(b.timeMs, 0);
  assert.throws(() => { a.timeMs = 100; }, TypeError);
});
