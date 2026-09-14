import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createWingEventExcitation, DEFAULT_WING_EVENT_PRIORS} from '../flybody-wing-event-excitation.js';
import {STEERING_MUSCLE_TYPES} from '../training/flight-parameters.js';

const clone = value => structuredClone(value);
function fixture() {
  let next = 1000;
  const muscles = [{kind: 'nonwing', indices: [4]}];
  for (const side of ['right', 'left']) {
    for (const target of STEERING_MUSCLE_TYPES) {
      muscles.push({joint: 'wing_steer_' + side, kind: 'wing_steering_assumption', target, sign: 1, indices: [next--]});
      muscles.push({kind: 'nonwing', indices: [3]});
    }
    for (const [target, count] of [['dorsal_longitudinal_muscle', 5], ['dorsoventral_muscle', 7]])
      muscles.push({joint: 'wing_power_' + side, kind: 'asynchronous_wing', target, sign: 1,
        indices: Array.from({length: count}, () => next--)});
  }
  return {muscles};
}
const io = fixture();
const ids = io.muscles.filter(m => m.joint).flatMap(m => m.indices).sort((a, b) => a - b);
const slot = id => ids.indexOf(id);
const target = (name, side = 'left') => io.muscles.find(m => m.target === name && m.joint.endsWith(side));
const steering = target('b1_muscle').indices[0];
const dlm = target('dorsal_longitudinal_muscle').indices[0];
const dvm = target('dorsoventral_muscle').indices[0];
function* packets(events, durationMs) {
  const counts = Array(48).fill(0);
  yield {initialized: true, fromTimeMs: null, timeMs: 0, indices: ids.slice(), counts: counts.slice(), ratesHz: Array(48).fill(0), events: []};
  for (let timeMs = 2; timeMs <= durationMs; timeMs += 2) {
    const current = events.filter(e => e.timeMs > timeMs - 2 && e.timeMs <= timeMs)
      .map(e => ({...e})).sort((a, b) => a.timeMs - b.timeMs || a.index - b.index);
    for (const e of current) counts[slot(e.index)]++;
    yield {initialized: false, fromTimeMs: timeMs - 2, timeMs, indices: ids.slice(), counts: counts.slice(),
      ratesHz: ids.map((_, k) => Math.fround(k * .3)), events: current};
  }
}
const model = (config = DEFAULT_WING_EVENT_PRIORS) => createWingEventExcitation({io, config});
function run(events, durationMs, config) {
  const m = model(config), outputs = [];
  for (const packet of packets(events, durationMs)) {
    m.accept(packet);
    if (!packet.initialized) outputs.push(m.finishInterval(packet.timeMs - 1), m.finishInterval(packet.timeMs));
  }
  return {m, outputs};
}
function normalizedKernel(age, p) {
  if (age < 0) return 0;
  const peak = Math.log(p.decayMs / p.riseMs) / (1 / p.riseMs - 1 / p.decayMs);
  const norm = Math.exp(-peak / p.decayMs) - Math.exp(-peak / p.riseMs);
  return (Math.exp(-age / p.decayMs) - Math.exp(-age / p.riseMs)) / norm;
}
// Independent dense composite Simpson rule: full event-history evaluation,
// with no adapter state, GL nodes, or recurrence reused. Split derivative cusps.
function denseAverage(events, a, b, p, nonlinear = true) {
  const cuts = [...new Set([a, ...events.filter(t => t > a && t < b), b])].sort((x, y) => x - y);
  const f = t => {
    const c = events.reduce((sum, s) => sum + normalizedKernel(t - s, p), 0);
    return nonlinear ? 1 - Math.exp(-p.recruitmentGain * c) : c;
  };
  let area = 0;
  for (let j = 1; j < cuts.length; j++) {
    const from = cuts[j - 1], to = cuts[j], n = 2048, h = (to - from) / n;
    let sum = f(from) + f(to);
    for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * f(from + i * h);
    area += h * sum / 3;
  }
  return area / (b - a);
}
function near(actual, expected, tolerance = 2e-11) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected} (tol ${tolerance})`);
}

test('default priors are explicit; original 28 mapping order and sorted 48 unit identities are retained', () => {
  const m = model(), s = m.readState();
  assert.deepEqual(Array.from(s.indices), ids);
  const original = io.muscles.flatMap((m, i) => m.joint ? [i] : []);
  assert.deepEqual(Array.from(s.mappingIndices), original);
  assert.deepEqual(s.mappings.map(m => m.indices), original.map(i => io.muscles[i].indices));
  assert.deepEqual(s.unitFamilies.reduce((a, f) => ({...a, [f]: (a[f] || 0) + 1}), {}), {dvm: 14, dlm: 10, steering: 24});
  assert.deepEqual(s.config, {steering: {riseMs: 1, decayMs: 5, recruitmentGain: Math.LN2},
    dlm: {riseMs: 6.2, decayMs: 82, recruitmentGain: Math.LN2}, dvm: {riseMs: 6.2, decayMs: 82, recruitmentGain: Math.LN2}});
  assert.ok(Object.isFrozen(DEFAULT_WING_EVENT_PRIORS.dlm));
});

test('current prepared IO also satisfies the explicit 48/28 anatomical join, without neural execution', () => {
  const prepared = JSON.parse(fs.readFileSync(new URL('../../data/prepared/banc888/io.json', import.meta.url)));
  const s = createWingEventExcitation({io: prepared}).readState();
  assert.equal(s.indices.length, 48); assert.equal(s.mappingIndices.length, 28);
  for (const m of s.mappings) assert.deepEqual(m.indices, prepared.muscles[m.mappingIndex].indices);
});

test('analytic kernel averages and nonlinear quadrature agree with independent dense history integration', () => {
  const times = [.5, 3, 6.5, 10], active = [[steering, 'steering'], [dlm, 'dlm'], [dvm, 'dvm']];
  const config = clone(DEFAULT_WING_EVENT_PRIORS);
  config.dvm = {riseMs: 2, decayMs: 12, recruitmentGain: .9}; // prove separate family configuration
  const {outputs} = run(active.flatMap(([index]) => times.map(timeMs => ({index, timeMs}))), 16, config);
  for (const out of outputs) for (const [id, family] of active) {
    const p = config[family];
    near(out.unitExcitation[slot(id)], denseAverage(times, out.fromTimeMs, out.timeMs, p));
    near(out.unitMeanKernel[slot(id)], denseAverage(times, out.fromTimeMs, out.timeMs, p, false));
    assert.ok(out.quadratureDiscrepancy <= 1e-7);
  }
  // The nonlinearity of mean kernel is observably NOT the correct interval mean.
  const first = outputs[0];
  assert.ok(Math.abs(first.unitExcitation[slot(steering)] -
    -Math.expm1(-config.steering.recruitmentGain * first.unitMeanKernel[slot(steering)])) > 1e-4);
});

test('later events cannot alter an earlier interval; exact right-edge events seed only the next interval', () => {
  const silent = run([], 4).outputs;
  for (const eventTime of [.5, 1, 1.5, 2]) {
    const {outputs} = run([{index: steering, timeMs: eventTime}], 4);
    for (const out of outputs) {
      if (out.timeMs <= eventTime) assert.deepEqual(out.unitExcitation, silent[out.timeMs - 1].unitExcitation);
      else assert.ok(out.unitExcitation[slot(steering)] > 0);
    }
  }
  const early = run([{index: steering, timeMs: .5}], 2).outputs;
  const late = run([{index: steering, timeMs: 1.5}], 2).outputs;
  assert.ok(early[0].unitExcitation[slot(steering)] > late[0].unitExcitation[slot(steering)]);
  assert.ok(early[1].unitExcitation[slot(steering)] > late[1].unitExcitation[slot(steering)]);
});

test('pooling happens after each unit nonlinearity; DLM and DVM membership is never conflated', () => {
  const {m, outputs} = run([{index: dlm, timeMs: .5}], 8);
  const mappings = m.readState().mappings;
  const group = mappings.findIndex(row => row.indices.includes(dlm));
  for (const out of outputs) {
    near(out.excitation[group], out.unitExcitation[slot(dlm)] / 5, 1e-15);
    assert.equal(out.excitation.filter(x => x > 0).length, 1);
    assert.equal(out.unitExcitation.filter(x => x > 0).length, 1);
  }
  const p = DEFAULT_WING_EVENT_PRIORS.dlm;
  const wrong = denseAverage([.5], 0, 1, {...p, recruitmentGain: p.recruitmentGain / 5});
  assert.ok(wrong > outputs[0].excitation[group] + 1e-6);
  const all = target('dorsal_longitudinal_muscle').indices;
  const synchronous = run(all.map(index => ({index, timeMs: .5})), 8).outputs;
  near(synchronous[0].excitation[group], outputs[0].unitExcitation[slot(dlm)]);
  const staggered = run(all.map((index, i) => ({index, timeMs: .5 + i * .5})), 8).outputs;
  assert.notEqual(staggered[0].excitation[group], synchronous[0].excitation[group]);
});

test('empty packets advance analytical decay and rates remain diagnostic, never excitation input', () => {
  const events = [{index: steering, timeMs: .5}], packetsA = [...packets(events, 100)];
  const a = model(), b = model();
  for (const p of packetsA) {
    a.accept(p); b.accept({...p, ratesHz: Array(48).fill(Math.fround(10000))});
    if (!p.initialized) for (const t of [p.timeMs - 1, p.timeMs]) assert.deepEqual(a.finishInterval(t), b.finishInterval(t));
  }
  const state = a.readState(), p = DEFAULT_WING_EVENT_PRIORS.steering;
  near(state.decayState[slot(steering)], Math.exp(-99.5 / p.decayMs), 1e-18);
  near(state.riseState[slot(steering)], Math.exp(-99.5 / p.riseMs), 1e-48);
  assert.equal(state.counts[slot(steering)], 1);
  assert.equal(b.readState().ratesHz[slot(steering)], 10000);
});

test('packet index order is explicitly joined; inputs and every returned state are owned copies', () => {
  const mutableIO = clone(io), config = clone(DEFAULT_WING_EVENT_PRIORS);
  const m = createWingEventExcitation({io: mutableIO, config});
  mutableIO.muscles.find(x => x.joint).indices[0] = 0; config.steering.decayMs = 200;
  const [baseline, next] = [...packets([{index: steering, timeMs: 1.5}], 2)];
  for (const p of [baseline, next]) for (const key of ['indices', 'counts', 'ratesHz']) p[key].reverse();
  m.accept(baseline); m.accept(next);
  next.events[0].timeMs = .5; next.counts.fill(0); next.ratesHz.fill(-1);
  const s = m.readState(); s.config.steering.decayMs = 40; s.pending.events[0].timeMs = .5;
  s.indices.fill(0); s.mappingIndices.fill(0); s.mappings[0].indices[0] = 0; s.decayState.fill(50);
  const first = m.finishInterval(1);
  assert.ok(first.excitation.every(x => x === 0)); first.unitExcitation.fill(100);
  const second = m.finishInterval(2);
  const reference = run([{index: steering, timeMs: 1.5}], 2).outputs[1];
  assert.deepEqual(second, reference);
});

test('zero-count fresh baseline is mandatory; no events are fabricated from history', () => {
  const baseline = [...packets([], 0)][0];
  for (const mutate of [p => { p.timeMs = 2; }, p => { p.initialized = false; },
    p => { p.counts[0] = 1; }, p => { p.fromTimeMs = 0; },
    p => { p.events = [{index: ids[0], timeMs: 0}]; }]) {
    const m = model(), before = m.snapshot(), p = clone(baseline); mutate(p);
    assert.throws(() => m.accept(p)); assert.deepEqual(m.snapshot(), before);
  }
});

test('malformed packets reject transactionally: loss, duplication, overrun, invalid identity/times/rates', () => {
  const [baseline, valid] = [...packets([{index: steering, timeMs: .5}], 2)];
  const mutations = [
    p => { p.fromTimeMs = 2; p.timeMs = 4; }, p => { p.timeMs = 0; },
    p => { p.indices[0] = p.indices[1]; }, p => { p.indices[0] = 123456; },
    p => { p.counts[slot(steering)] = 0; }, p => { p.counts[slot(steering)] = 2; },
    p => { p.counts[0] = -1; }, p => { p.counts[0] = 2 ** 24; }, p => { p.counts.pop(); },
    p => { p.events = []; }, p => { p.events.push({...p.events[0]}); },
    p => { p.events[0].index = 12345; }, p => { p.events[0].timeMs = 0; },
    p => { p.events[0].timeMs = 2.5; }, p => { p.events[0].timeMs = .75; },
    p => { p.events[0].timeMs = NaN; }, p => { p.ratesHz[0] = Infinity; },
    p => { p.ratesHz[0] = -.1; }, p => { p.ratesHz[0] = .1; },
  ];
  for (const mutate of mutations) {
    const m = model(); m.accept(baseline); const before = m.snapshot(), p = clone(valid); mutate(p);
    assert.throws(() => m.accept(p)); assert.deepEqual(m.snapshot(), before);
    m.accept(valid); assert.ok(m.finishInterval(1).unitExcitation[slot(steering)] > 0);
  }
  const m = model(); m.accept(baseline); m.accept(valid); const before = m.snapshot();
  assert.throws(() => m.accept(valid)); assert.deepEqual(m.snapshot(), before);
  for (const end of [0, .5, 2, 3, NaN, Infinity]) {
    assert.throws(() => m.finishInterval(end)); assert.deepEqual(m.snapshot(), before);
  }
});

test('strict sorting, cross-packet refractory spacing and count regressions are checked', () => {
  const m = model(), p = [...packets([{index: steering, timeMs: 2}, {index: steering, timeMs: 4}], 4)];
  m.accept(p[0]); m.accept(p[1]); m.finishInterval(1); m.finishInterval(2);
  const before = m.snapshot(); assert.throws(() => m.accept(p[2]), /refractory/); assert.deepEqual(m.snapshot(), before);
  p[2].events = []; p[2].counts[slot(steering)] = 0;
  assert.throws(() => m.accept(p[2]), /regression/);
  const sorted = [...packets([{index: steering, timeMs: .5}, {index: dlm, timeMs: .5}], 2)];
  const n = model(); n.accept(sorted[0]); sorted[1].events.reverse();
  assert.throws(() => n.accept(sorted[1]), /sorted/);
  assert.doesNotThrow(() => run([{index: steering, timeMs: .5}, {index: steering, timeMs: 3}], 4));
});

test('snapshot round-trips before baseline, queued packets, midpoint and completed packets reproduce exact outputs', () => {
  const events = [{index: steering, timeMs: .5}, {index: dlm, timeMs: 1}, {index: dvm, timeMs: 1.5},
    {index: steering, timeMs: 3}, {index: dlm, timeMs: 4}, {index: dvm, timeMs: 5}];
  const a = model(), b = model();
  b.restore(JSON.parse(JSON.stringify(a.snapshot())));
  for (const packet of packets(events, 10)) {
    a.accept(packet); b.restore(JSON.parse(JSON.stringify(a.snapshot())));
    if (packet.initialized) continue;
    for (const time of [packet.timeMs - 1, packet.timeMs]) {
      assert.deepEqual(b.finishInterval(time), a.finishInterval(time));
      assert.deepEqual(b.snapshot(), a.snapshot());
      const snap = a.snapshot(); b.restore(snap); snap.decayState.fill(0);
    }
  }
});

test('malformed snapshots fail transactionally without trusting clocks, cursor, history, shape or contract', () => {
  const m = model(), input = [...packets([{index: steering, timeMs: .5}, {index: dlm, timeMs: 1.5}], 2)];
  m.accept(input[0]); m.accept(input[1]); m.finishInterval(1);
  const valid = m.snapshot();
  const mutations = [s => { s.contract.config.dlm.decayMs = 90; }, s => { s.kind = 'other'; },
    s => { s.extra = 1; }, s => { s.integratedThroughMs = .5; }, s => { s.observedThroughMs = 4; },
    s => { s.pending = null; }, s => { s.pending.cursor = 0; }, s => { s.pending.fromTimeMs = -2; },
    s => { s.pending.events[1].timeMs = 1; }, s => { s.counts[slot(dlm)] = 0; },
    s => { s.decayState[slot(dlm)] = 1; }, s => { s.decayState[0] = NaN; },
    s => { s.decayState[0] = Infinity; }, s => { s.riseState[slot(steering)] = 100; },
    s => { s.lastEventTimesMs[slot(steering)] = .75; }, s => { s.lastEventTimesMs[slot(dlm)] = 2; },
    s => { s.counts[slot(steering)] = 3; }, s => { s.counts.pop(); }, s => { s.initialized = false; }];
  for (const mutate of mutations) {
    const bad = clone(valid); mutate(bad);
    assert.throws(() => m.restore(bad)); assert.deepEqual(m.snapshot(), valid);
  }
  const b = model(); b.restore(valid); assert.deepEqual(m.finishInterval(2), b.finishInterval(2));
});

test('unsupported mapping/configuration and unresolvable narrow kernels fail closed', () => {
  for (const mutate of [x => { x.muscles.find(m => m.joint).sign = -1; },
    x => { x.muscles.find(m => m.joint).target = 'invented'; },
    x => { x.muscles.find(m => m.joint).indices.push(50); },
    x => { x.muscles.push(clone(x.muscles.find(m => m.joint))); },
    x => { const rows = x.muscles.filter(m => m.joint); rows[1].indices = rows[0].indices.slice(); }]) {
    const bad = clone(io); mutate(bad); assert.throws(() => createWingEventExcitation({io: bad}));
  }
  for (const mutate of [p => { delete p.dlm; }, p => { p.extra = {}; },
    p => { p.steering.riseMs = 0; }, p => { p.steering.decayMs = 1; },
    p => { p.dlm.decayMs = Infinity; }, p => { p.dvm.recruitmentGain = NaN; },
    p => { p.dvm.recruitmentGain = 0; }, p => { p.dlm.riseMs = 1e-9; }]) {
    const bad = clone(DEFAULT_WING_EVENT_PRIORS); mutate(bad); assert.throws(() => model(bad));
  }
  const config = clone(DEFAULT_WING_EVENT_PRIORS); config.steering = {riseMs: .05, decayMs: .06, recruitmentGain: 16};
  const m = model(config), p = [...packets([{index: steering, timeMs: .5}], 2)];
  m.accept(p[0]); m.accept(p[1]); const before = m.snapshot();
  assert.throws(() => m.finishInterval(1), /quadrature/); assert.deepEqual(m.snapshot(), before);
});
