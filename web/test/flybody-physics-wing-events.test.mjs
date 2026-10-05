// Small real native muscle objects with a recorded mechanical CLOCK STUB.
// These tests do not run MuJoCo, a neural model, or claim physical parity.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {FlyBodyPhysics} from '../flybody-physics.js';
import {createMotorExcitation} from '../flybody-motor-excitation.js';
import {createBancProboscisDecoder} from '../banc-proboscis.js';
import {InternalState} from '../banc/embodiment.js';
import {createWasmCore, WasmMuscles} from '../../packages/banc-runtime/src/wasm.js';

const prepared = JSON.parse(fs.readFileSync(new URL('../../data/prepared/banc888/io.json', import.meta.url)));
const wingMappings = prepared.muscles.filter(m => ['asynchronous_wing', 'wing_steering_assumption'].includes(m.kind));
const mappings = [{joint: 'leg', kind: 'leg', indices: [1], sign: 1}, ...wingMappings.slice(0, 13),
  {joint: 'pump', kind: 'pump', indices: [2], sign: 1}, ...wingMappings.slice(13)];
const wingRows = mappings.flatMap((m, i) => wingMappings.includes(m) ? [i] : []);
const indices = wingMappings.flatMap(m => m.indices).sort((a, b) => a - b);
const selected = mappings.findIndex(m => m.joint === 'wing_steer_left' && m.target === 'b1_muscle');
const selectedIndex = mappings[selected].indices[0], selectedUnit = indices.indexOf(selectedIndex);
const dlm = mappings.findIndex(m => m.joint === 'wing_power_left' && m.target === 'dorsal_longitudinal_muscle');
const corePromise = createWasmCore();
const bytes = array => Buffer.from(array.buffer, array.byteOffset, array.byteLength);
const close = (a, b, tolerance = 1e-14) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);

async function fixture(energy = .05) {
  const core = await corePromise, body = Object.create(FlyBodyPhysics.prototype), allocations = [], commands = [], mechanical = [], effort = [];
  const data = {time: 0, qpos: new Float64Array(10), qvel: new Float64Array(9), ctrl: new Float64Array(9), deleted: 0,
    delete() { this.deleted++; }};
  data.qpos[3] = 1; data.qpos[7] = .2; data.qvel[6] = .3;
  const createMuscles = count => {
    const native = new WasmMuscles(core, count), entry = {native, count, calls: [], disposed: 0}; allocations.push(entry);
    return {step(input, dt) {
      const snapshot = input.slice(), output = native.step(input, dt);
      entry.calls.push({time: data.time, input: snapshot, dt, output: output.slice()}); return output;
    }, dispose() { entry.disposed++; native.dispose(); }};
  };
  const byJoint = new Map([['leg', {qpos: 7, dof: 6, neutral: 0, range: [-1, 1]}],
    ['rostrum', {qpos: 8, dof: 7, neutral: 0, range: [-1, 1]}],
    ['haustellum', {qpos: 9, dof: 8, neutral: 0, range: [-1, 1]}]]);
  const internal = new InternalState(energy), internalStep = internal.step.bind(internal);
  internal.step = (dt, intake, value) => { effort.push(value); internalStep(dt, intake, value); };
  Object.assign(body, {data, model: {}, metadata: {timestep: .00005, maxJointExcursion: .35},
    mappings, byJoint, actuators: new Map([['leg', {id: 0, range: [-1, 1]}], ['rostrum', {id: 1, range: [-1, 1]}], ['haustellum', {id: 2, range: [-1, 1]}]]),
    muscleDirections: Float64Array.from(mappings, () => 1), _createMuscles: createMuscles, _wingMotorEvents: null,
    muscles: createMuscles(mappings.length), input: new Float32Array(mappings.length * 5), activation: new Float32Array(mappings.length),
    muscleState: new Float32Array(mappings.length * 3), motorExcitation: createMotorExcitation(),
    proboscisDecoder: createBancProboscisDecoder(mappings), internal, restPose: data.qpos.slice(), remainder: 0, time: 0,
    heading: 0, x: 0, y: 0, z: 0, contactFood: null, proboscis: 0, smell: () => 0,
    contactForce: {deleted: 0, delete() { this.deleted++; }}, monitor: {step() {}},
    refresh() { this.time = data.time; },
    mj: {mj_step(_model, d) { mechanical.push({step: mechanical.length, ...structuredClone(commands.at(-1))}); d.time += .00005; }},
    wings: {phase: 0, power: [0, 0],
      step(_q, _ctrl, left, right, steering, dt) {
        commands.push({time: data.time, left, right, steering: structuredClone(steering)});
        this.power = [left, right]; this.phase += dt * 2 * Math.PI * 235.813447;
      }},
  });
  return {body, allocations, commands, mechanical, effort, dispose: () => body.dispose()};
}
function packetList(events = [], durationMs = 4) {
  const counts = Array(48).fill(0), ratesHz = Array(48).fill(0);
  const packets = [{initialized: true, fromTimeMs: null, timeMs: 0, indices: indices.slice(), counts: counts.slice(), ratesHz, events: []}];
  for (let timeMs = 2; timeMs <= durationMs; timeMs += 2) {
    const selected = events.filter(e => e.timeMs > timeMs - 2 && e.timeMs <= timeMs).sort((a, b) => a.timeMs - b.timeMs || a.index - b.index);
    for (const event of selected) counts[indices.indexOf(event.index)]++;
    packets.push({initialized: false, fromTimeMs: timeMs - 2, timeMs, indices: indices.slice(), counts: counts.slice(), ratesHz,
      events: selected.map(e => ({...e}))});
  }
  return packets;
}
const rates = wingHz => new Map([[1, 40], [2, 16], ...indices.map(index => [index, wingHz])]);
const observations = f => ({time: f.body.time, remainder: f.body.remainder, event: f.body.readWingMotorEvents(),
  muscle: Array.from(f.body.muscleState), input: Array.from(f.body.input), activation: Array.from(f.body.activation),
  native: f.allocations.map(a => Array.from(a.native.core.HEAPF32.slice(a.native.state / 4, a.native.state / 4 + a.count * 3))),
  calls: f.allocations.map(a => a.calls.length), commands: structuredClone(f.commands), mechanical: structuredClone(f.mechanical)});

test('silent event path matches disabled default bytes and call sequence for nonwing/native inputs and body clock', async () => {
  const a = await fixture(), b = await fixture();
  try {
    assert.equal(a.body.readWingMotorEvents(), null); assert.equal(a.allocations.length, 1);
    b.body.enableWingMotorEvents();
    const packets = packetList([], 8); b.body.acceptWingMotorEvents(packets[0]);
    for (const packet of packets.slice(1)) {
      b.body.acceptWingMotorEvents(packet); a.body.step(rates(0), .002); b.body.step(rates(10000), .002);
      for (const key of ['activation', 'input', 'muscleState']) assert.deepEqual(bytes(a.body[key]), bytes(b.body[key]), key);
      assert.deepEqual(a.commands, b.commands); assert.deepEqual(a.mechanical, b.mechanical); assert.deepEqual(a.effort, b.effort);
      for (const key of ['energy', 'hunger', 'crop', 'akh', 'insulin']) assert.equal(a.body.internal[key], b.body.internal[key]);
      assert.equal(a.body.time, b.body.time); assert.equal(b.body.readWingMotorEvents().bodyConsumedThroughMs, packet.timeMs);
    }
    assert.deepEqual(a.allocations[0].calls, b.allocations[0].calls);
    assert.equal(a.allocations.length, 1); assert.equal(b.allocations.length, 2);
  } finally { a.dispose(); b.dispose(); }
});

test('event force is held causally; exact right-edge spikes wait until the following completed interval', async () => {
  for (const eventTime of [.5, 1, 1.5, 2]) {
    const f = await fixture(.35);
    try {
      f.body.enableWingMotorEvents(); const packets = packetList([{index: selectedIndex, timeMs: eventTime}]);
      f.body.acceptWingMotorEvents(packets[0]);
      for (const packet of packets.slice(1)) { f.body.acceptWingMotorEvents(packet); f.body.step(rates(10000), .002); }
      const firstEffectMs = Math.floor(eventTime) + 1;
      const first = f.mechanical.find(row => row.steering.left.b1_muscle > 0);
      assert.equal(first.step, firstEffectMs * 20, 'first mechanical step using event force');
      assert(f.mechanical.slice(0, first.step).every(row => row.steering.left.b1_muscle === 0));
      assert.deepEqual(f.allocations[0].calls.map(row => Math.round(row.time * 1000)), [0, 1, 2, 3]);
      assert.deepEqual(f.allocations[1].calls.map(row => Math.round(row.time * 1000)), [1, 2, 3, 4]);
      for (const call of f.allocations[0].calls) for (const i of wingRows) {
        assert.equal(call.input[i * 5], 0); assert(call.output.slice(i * 3, i * 3 + 3).every(x => x === 0));
      }
      const hidden = f.allocations[0].native;
      for (const i of wingRows) assert(hidden.core.HEAPF32.slice(hidden.state / 4 + i * 3, hidden.state / 4 + i * 3 + 3).every(x => x === 0));
      const exposed = f.body.readWingMotorEvents(), unit = exposed.mappings.findIndex(m => m.mappingIndex === selected);
      assert.deepEqual(bytes(f.body.muscleState.slice(selected * 3, selected * 3 + 3)), bytes(exposed.nativeState.slice(unit * 3, unit * 3 + 3)));
      assert.equal(f.body.input[selected * 5], exposed.nativeInput[unit * 5]);
    } finally { f.dispose(); }
  }
});

test('actual fuel and nonwing inputs persist; wing groups feed power once and effort uses held wing activation once', async () => {
  const f = await fixture(.05);
  try {
    f.body.enableWingMotorEvents();
    const packets = packetList([{index: selectedIndex, timeMs: .5}, {index: mappings[dlm].indices[0], timeMs: .5}]);
    f.body.acceptWingMotorEvents(packets[0]);
    for (const packet of packets.slice(1)) { f.body.acceptWingMotorEvents(packet); f.body.step(rates(500), .002); }
    const full = f.allocations[0].calls, event = f.allocations[1].calls;
    for (let tick = 0; tick < full.length; tick++) {
      assert.equal(full[tick].input[0], .5); close(full[tick].input[1], Math.fround(.96)); close(full[tick].input[2], Math.fround(.06));
      for (let i = 0; i < 28; i++) {
        assert.equal(event[tick].input[i * 5 + 1], 1); assert.equal(event[tick].input[i * 5 + 2], 0);
        assert.equal(event[tick].input[i * 5 + 3], 1); assert.equal(event[tick].input[i * 5 + 4], full[tick].input[4]);
      }
      let sum = 0;
      for (let i = 0; i < mappings.length; i++) {
        const wi = wingRows.indexOf(i);
        sum += wi < 0 ? full[tick].output[i * 3] : tick ? event[tick - 1].output[wi * 3] : 0;
      }
      assert.equal(f.effort[tick], sum / mappings.length);
      const command = f.commands[tick * 5], wingIndex = wingRows.indexOf(dlm);
      assert.equal(command.left, tick ? event[tick - 1].output[wingIndex * 3 + 2] / 2 : 0);
      assert.equal(command.right, 0);
    }
    assert(event[0].input[4] < 1 && event[0].input[4] > 0);
    assert(event.at(-1).input[4] < event[0].input[4], 'native fuel follows the existing reserve metabolism');
  } finally { f.dispose(); }
});

test('missing/malformed packets, wrong duration and clock mismatch reject before adapter/native/body mutation', async () => {
  const f = await fixture();
  try {
    f.body.enableWingMotorEvents(); let before = observations(f);
    assert.throws(() => f.body.step(rates(0), .002), /observed2ms/); assert.deepEqual(observations(f), before);
    const [baseline, valid] = packetList([{index: selectedIndex, timeMs: .5}], 2);
    f.body.acceptWingMotorEvents(baseline); before = observations(f);
    const bad = structuredClone(valid); bad.events = [];
    assert.throws(() => f.body.acceptWingMotorEvents(bad), /mismatch/); assert.deepEqual(observations(f), before);
    f.body.acceptWingMotorEvents(valid); before = observations(f);
    assert.throws(() => f.body.acceptWingMotorEvents(valid), /previous packet/); assert.deepEqual(observations(f), before);
    for (const dt of [.001, .004, .00200000001]) {
      assert.throws(() => f.body.step(rates(0), dt), /observed2ms/); assert.deepEqual(observations(f), before);
    }
    f.body.data.time = .00005;
    assert.throws(() => f.body.step(rates(0), .002), /clock mismatch/); assert.deepEqual(observations(f), before);
    f.body.data.time = 1e-12; f.body.time = 1e-12; // harmless native accumulation epsilon
    assert.doesNotThrow(() => f.body.step(rates(0), .002));
    before = observations(f); assert.throws(() => f.body.step(rates(0), .002), /observed2ms/);
    assert.deepEqual(observations(f), before);
  } finally { f.dispose(); }
});

test('wing rate decoding is never called; coupling and flight flags retain separate meanings', async () => {
  for (const options of [{coupling: false}, {flight: false}]) {
    const f = await fixture();
    try {
      f.body.enableWingMotorEvents();
      const original = f.body.motorExcitation;
      f.body.motorExcitation = {fromRate(kind, hz) { assert(!['asynchronous_wing', 'wing_steering_assumption'].includes(kind)); return original.fromRate(kind, hz); }};
      const packets = packetList([{index: mappings[dlm].indices[0], timeMs: .5}]); f.body.acceptWingMotorEvents(packets[0]);
      for (const packet of packets.slice(1)) { f.body.acceptWingMotorEvents(packet); f.body.step(rates(1e8), .002, options); }
      assert(f.commands.every(c => c.left === 0 && c.right === 0));
      const s = f.body.readWingMotorEvents(); assert.equal(s.counts.reduce((a, b) => a + b, 0), 1);
      assert(s.lastInterval.unitExcitation.some(x => x > 0));
      if (options.coupling === false) assert(s.nativeState.every(x => x === 0));
      else assert(s.nativeState.some(x => x > 0));
    } finally { f.dispose(); }
  }
});

test('enable is fresh-only and validated before allocation; diagnostics own copies and both kernels dispose', async () => {
  const f = await fixture();
  try {
    assert.throws(() => f.body.acceptWingMotorEvents(packetList()[0]), /not enabled/);
    assert.throws(() => f.body.enableWingMotorEvents({}), /schema/); assert.equal(f.allocations.length, 1);
    f.body.time = .001; assert.throws(() => f.body.enableWingMotorEvents(), /fresh/); f.body.time = 0;
    f.body.byJoint.set('wing_steer_left', {}); assert.throws(() => f.body.enableWingMotorEvents(), /virtual/); f.body.byJoint.delete('wing_steer_left');
    assert.equal(f.allocations.length, 1); f.body.enableWingMotorEvents(); assert.equal(f.allocations.length, 2);
    assert.throws(() => f.body.enableWingMotorEvents(), /already enabled/);
    const before = f.body.readWingMotorEvents(), exposed = f.body.readWingMotorEvents();
    exposed.nativeState.fill(1); exposed.nativeInput.fill(1); exposed.indices.fill(0); exposed.config.steering.decayMs = 900;
    assert.deepEqual(f.body.readWingMotorEvents(), before);
  } finally { f.dispose(); }
  f.dispose(); // repeated cleanup must not free native ownership twice
  assert.deepEqual(f.allocations.map(a => a.disposed), [1, 1]); assert.equal(f.body.data.deleted, 1);
  assert.equal(f.body.contactForce.deleted, 1); assert.throws(() => f.body.enableWingMotorEvents(), /disposed/);
});
