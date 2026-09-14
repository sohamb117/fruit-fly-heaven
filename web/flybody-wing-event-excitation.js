import {STEERING_MUSCLE_TYPES} from './training/flight-parameters.js';

// Opt-in phenomenological excitation only: no native muscle/brain/body writes.
// Steering is an unmeasured fast prior; DLM is calcium-like, not measured force;
// DVM borrows DLM constants. Equal unit weights and peak recruitment are priors.
export const DEFAULT_WING_EVENT_PRIORS = Object.freeze({
  steering: Object.freeze({riseMs: 1, decayMs: 5, recruitmentGain: Math.LN2}),
  dlm: Object.freeze({riseMs: 6.2, decayMs: 82, recruitmentGain: Math.LN2}),
  dvm: Object.freeze({riseMs: 6.2, decayMs: 82, recruitmentGain: Math.LN2}),
});

const FAMILIES = ['steering', 'dlm', 'dvm'], UNIT_COUNT = 48, MAX_EXACT = 2 ** 24;
const GL4 = [[.3399810435848563, .6521451548625461], [.8611363115940526, .3478548451374538]];
const GL8 = [[.1834346424956498, .362683783378362], [.525532409916329, .3137066458778873],
  [.7966664774136267, .2223810344533745], [.9602898564975363, .1012285362903763]];
const QUADRATURE_TOLERANCE = 1e-7;
const record = x => x !== null && typeof x === 'object' && !Array.isArray(x) && !ArrayBuffer.isView(x);
const copy = x => JSON.parse(JSON.stringify(x));
const fail = message => { throw new Error('Wing event excitation: ' + message); };
const requireThat = (condition, message) => { if (!condition) fail(message); };
const finite = x => typeof x === 'number' && Number.isFinite(x);
function keys(value, expected, label) {
  requireThat(record(value) && Reflect.ownKeys(value).length === expected.length &&
    expected.every(k => Object.hasOwn(value, k)), label + ' has an invalid schema');
}
function exactTime(t, label) {
  requireThat(finite(t) && t >= 0 && Number.isInteger(t * 2) && t * 2 < MAX_EXACT &&
    Math.fround(t) === t, label + ' must be an exact 0.5 ms tick below 2^24 ticks');
}
function values(value, length, predicate, label) {
  requireThat((Array.isArray(value) || ArrayBuffer.isView(value)) && value.length === length,
    label + ' has an invalid length');
  const result = Array.from(value);
  requireThat(result.every(predicate), label + ' contains an invalid value');
  return result;
}
function parameters(config) {
  keys(config, FAMILIES, 'config');
  return Object.fromEntries(FAMILIES.map(family => {
    const p = config[family];
    keys(p, ['riseMs', 'decayMs', 'recruitmentGain'], family);
    const {riseMs, decayMs, recruitmentGain} = p;
    // Numerical support bounds, not biological bounds. Close time constants and
    // arbitrarily narrow kernels are outside this deliberately bounded adapter.
    requireThat([riseMs, decayMs, recruitmentGain].every(finite) && riseMs >= .05 &&
      decayMs <= 10000 && decayMs / riseMs >= 1.001 && recruitmentGain > 0 && recruitmentGain <= 16,
    family + ' requires rise >= 0.05 ms, decay <= 10000 ms, decay/rise >= 1.001, and gain in (0,16]');
    return [family, {riseMs, decayMs, recruitmentGain}];
  }));
}
function wingMappings(io) {
  requireThat(record(io) && Array.isArray(io.muscles), 'io.muscles must be an array');
  const seen = new Set(), seenUnits = new Set();
  const mappings = io.muscles.flatMap((m, mappingIndex) => {
    if (!['wing_steering_assumption', 'asynchronous_wing'].includes(m?.kind)) return [];
    const steering = m.kind === 'wing_steering_assumption';
    const family = steering ? 'steering' : m.target === 'dorsal_longitudinal_muscle' ? 'dlm' :
      m.target === 'dorsoventral_muscle' ? 'dvm' : null;
    const prefix = steering ? 'wing_steer_' : 'wing_power_';
    const side = m.joint === prefix + 'left' ? 'left' : m.joint === prefix + 'right' ? 'right' : null;
    requireThat(family && side && m.sign === 1 && (!steering || STEERING_MUSCLE_TYPES.includes(m.target)),
      'unsupported wing mapping target, joint or sign');
    const id = m.target + ':' + side;
    requireThat(!seen.has(id), 'duplicate wing muscle mapping'); seen.add(id);
    const indices = values(m.indices, steering ? 1 : family === 'dlm' ? 5 : 7,
      n => Number.isInteger(n) && n >= 0 && n <= 0xffffffff, 'mapping indices');
    for (const index of indices) {
      requireThat(!seenUnits.has(index), 'a wing motor unit appears in multiple mappings'); seenUnits.add(index);
    }
    return [{mappingIndex, joint: m.joint, target: m.target, kind: m.kind, sign: m.sign, family, side, indices}];
  });
  requireThat(mappings.length === 28 && seenUnits.size === UNIT_COUNT, 'expected exactly 28 wing mappings and 48 unique units');
  return mappings;
}

/**
 * Own 48 double-exponential event states and return 28 muscle excitations.
 * accept() consumes the motor-events.js packet schema (extra observer fields
 * are ignored). Start at a fresh zero-time/count baseline. Packets cover exactly
 * (fromTimeMs,timeMs] = 2 ms; consume each through two finishInterval() calls at
 * successive 1 ms endpoints before accepting another packet. No event/rate
 * interpolation or inferred spikes. Returned arrays/configs are owned copies.
 *
 * The output is the COMPLETED interval's average excitation. A later integrator
 * must apply it to its native 1 ms muscle update at that interval's right edge,
 * using the resulting force thereafter. This module owns no native state and
 * cannot enforce a caller's body/force schedule. ratesHz remain diagnostics.
 */
export function createWingEventExcitation({io, config = DEFAULT_WING_EVENT_PRIORS, eventContract}) {
  const priors = parameters(config), mappings = wingMappings(io);
  const indices = mappings.flatMap(m => m.indices).sort((a, b) => a - b);
  const positions = new Map(indices.map((id, k) => [id, k]));
  const groups = mappings.map(m => m.indices.map(id => positions.get(id)));
  const unitFamilies = Array(UNIT_COUNT);
  mappings.forEach((m, k) => groups[k].forEach(i => { unitFamilies[i] = m.family; }));
  // Only timestamps change; kernels, recruitment and body scheduling stay fixed.
  // Same model-derived declaration consumed by the runtime motor-event reader.
  const dlm = new Set();
  if (eventContract !== undefined) {
    keys(eventContract, ['schema', 'profile', 'indices'], 'motor event contract');
    requireThat(eventContract.schema === 1 && eventContract.profile === 'dlm-snl-2023-v1' &&
      Array.isArray(eventContract.indices) && eventContract.indices.length > 0 && eventContract.indices.length <= 10,
      'invalid DLM event contract');
    eventContract.indices.forEach((id, k) => {
      requireThat(Number.isInteger(id) && positions.has(id) && unitFamilies[positions.get(id)] === 'dlm' &&
        (k === 0 || id > eventContract.indices[k - 1]), 'timing overrides require sorted unique DLM units');
      dlm.add(id);
    });
  }
  function eventTime(id, time, label) {
    if (!dlm.has(id)) return exactTime(time, label);
    const tick = Math.round(time * 10);
    requireThat(finite(time) && time >= 0 && tick < MAX_EXACT && tick / 10 === time &&
      6 * 2 ** (Math.floor(Math.log2(Math.max(time + .1, .1))) - 23) < .1,
      label + ' must be a unique canonical 0.1 ms substep');
  }
  const kernels = unitFamilies.map(family => {
    const p = priors[family], peakMs = p.riseMs * p.decayMs / (p.decayMs - p.riseMs) * Math.log(p.decayMs / p.riseMs);
    const normalization = Math.exp(-peakMs / p.decayMs) - Math.exp(-peakMs / p.riseMs);
    requireThat(finite(normalization) && normalization > 0, 'invalid kernel normalization');
    return {...p, normalization};
  });
  const contract = {config: priors, mappings, indices, neuralDtMs: .5, packetMs: 2, intervalMs: 1};
  if (eventContract !== undefined) contract.eventContract = copy(eventContract);
  const schemaVersion = eventContract === undefined ? 1 : 2;
  const contractText = JSON.stringify(contract);
  let state = {initialized: false, integratedThroughMs: 0, observedThroughMs: 0,
    decayState: Array(UNIT_COUNT).fill(0), riseState: Array(UNIT_COUNT).fill(0),
    counts: Array(UNIT_COUNT).fill(0), ratesHz: Array(UNIT_COUNT).fill(0),
    lastEventTimesMs: Array(UNIT_COUNT).fill(null), pending: null};

  function eventList(input, from, to) {
    requireThat(Array.isArray(input) && input.length <= UNIT_COUNT, 'invalid event list');
    const used = new Set(); let previous = null;
    return input.map(event => {
      requireThat(record(event) && positions.has(event.index), 'unknown motor event index');
      eventTime(event.index, event.timeMs, 'event time');
      requireThat(event.timeMs > from && event.timeMs <= to, 'event is outside (fromTimeMs,timeMs]');
      requireThat(!used.has(event.index), 'duplicate event for a motor unit in one packet'); used.add(event.index);
      requireThat(!previous || event.timeMs > previous.timeMs ||
        (event.timeMs === previous.timeMs && event.index > previous.index), 'events must be sorted by time then index');
      const result = {index: event.index, timeMs: event.timeMs}; previous = result; return result;
    });
  }
  function accept(packet) {
    requireThat(record(packet), 'packet must be a record');
    requireThat(state.pending === null, 'consume the previous packet before accepting another');
    exactTime(packet.timeMs, 'packet time');
    const baseline = !state.initialized;
    requireThat(packet.initialized === baseline && packet.fromTimeMs === (baseline ? null : state.observedThroughMs) &&
      packet.timeMs === (baseline ? 0 : state.observedThroughMs + 2), 'missing, repeated or invalid 2 ms packet/baseline');
    const supplied = values(packet.indices, UNIT_COUNT, n => positions.has(n), 'packet indices');
    requireThat(new Set(supplied).size === UNIT_COUNT, 'packet indices must cover the 48 units exactly');
    const inputCounts = values(packet.counts, UNIT_COUNT, n => Number.isInteger(n) && n >= 0 && n < MAX_EXACT, 'counts');
    const inputRates = values(packet.ratesHz, UNIT_COUNT, n => finite(n) && n >= 0 && Math.fround(n) === n, 'ratesHz');
    const counts = Array(UNIT_COUNT), ratesHz = Array(UNIT_COUNT);
    supplied.forEach((id, k) => { counts[positions.get(id)] = inputCounts[k]; ratesHz[positions.get(id)] = inputRates[k]; });
    const events = eventList(packet.events, baseline ? 0 : packet.fromTimeMs, packet.timeMs);
    const byUnit = new Map(events.map(e => [positions.get(e.index), e]));
    const lastEventTimesMs = state.lastEventTimesMs.slice();
    for (let k = 0; k < UNIT_COUNT; k++) {
      const delta = counts[k] - state.counts[k], event = byUnit.get(k);
      requireThat(delta >= 0 && delta <= 1, 'spike count regression or overrun');
      requireThat(delta === (event ? 1 : 0), 'event/count mismatch');
      requireThat(!baseline || counts[k] === 0, 'fresh baseline must have zero counts; no historical events are fabricated');
      if (event) {
        requireThat(lastEventTimesMs[k] === null || (dlm.has(indices[k]) ? Math.round(event.timeMs * 10) - Math.round(lastEventTimesMs[k] * 10) >= 100 : event.timeMs - lastEventTimesMs[k] >= 2.5),
          'events violate the declared refractory/detector interval');
        lastEventTimesMs[k] = event.timeMs;
      }
    }
    state = {...state, initialized: true, observedThroughMs: packet.timeMs, counts, ratesHz, lastEventTimesMs,
      pending: baseline ? null : {fromTimeMs: packet.fromTimeMs, timeMs: packet.timeMs, events, cursor: 0}};
  }

  function finishInterval(endTimeMs) {
    exactTime(endTimeMs, 'interval endpoint');
    requireThat(state.initialized && state.pending && endTimeMs === state.integratedThroughMs + 1 &&
      endTimeMs <= state.observedThroughMs, 'finish exactly the next observed 1 ms interval');
    const decayState = state.decayState.slice(), riseState = state.riseState.slice();
    const kernelArea = new Float64Array(UNIT_COUNT), excitationArea = new Float64Array(UNIT_COUNT);
    const discrepancies = new Float64Array(UNIT_COUNT);
    let now = state.integratedThroughMs, cursor = state.pending.cursor;
    function integrate(until) {
      // Fixed bounded refinement on the already exact neural grid. One whole
      // millisecond can make the 4-point comparison too coarse near a new spike.
      while (now < until) {
      const dt = Math.min(.5, until - now);
      for (let k = 0; k < UNIT_COUNT; k++) {
        const p = kernels[k], d = decayState[k], r = riseState[k];
        const kernel = t => Math.max(0, (d * Math.exp(-t / p.decayMs) - r * Math.exp(-t / p.riseMs)) / p.normalization);
        const quadrature = rule => {
          let sum = 0;
          for (const [node, weight] of rule) for (const sign of [-1, 1])
            sum += weight * -Math.expm1(-p.recruitmentGain * kernel(dt * (1 + sign * node) / 2));
          return sum * dt / 2;
        };
        const area8 = quadrature(GL8), area4 = quadrature(GL4);
        excitationArea[k] += area8; discrepancies[k] += Math.abs(area8 - area4);
        kernelArea[k] += Math.max(0, (d * p.decayMs * -Math.expm1(-dt / p.decayMs) -
          r * p.riseMs * -Math.expm1(-dt / p.riseMs)) / p.normalization);
        decayState[k] *= Math.exp(-dt / p.decayMs); riseState[k] *= Math.exp(-dt / p.riseMs);
      }
      now += dt;
      }
    }
    while (cursor < state.pending.events.length && state.pending.events[cursor].timeMs <= endTimeMs) {
      const event = state.pending.events[cursor++]; integrate(event.timeMs);
      const k = positions.get(event.index); decayState[k] += 1; riseState[k] += 1;
    }
    integrate(endTimeMs);
    const quadratureDiscrepancy = Math.max(...discrepancies);
    requireThat(finite(quadratureDiscrepancy) && quadratureDiscrepancy <= QUADRATURE_TOLERANCE,
      '4/8-point quadrature discrepancy exceeds 1e-7; configuration cannot be advanced');
    requireThat([...decayState, ...riseState, ...kernelArea].every(x => finite(x) && x >= 0) &&
      Array.from(excitationArea).every(x => finite(x) && x >= 0 && x <= 1 + 1e-14), 'nonfinite or out-of-range interval result');
    const excitation = Float64Array.from(groups, group => group.reduce((sum, k) => sum + excitationArea[k], 0) / group.length);
    const fromTimeMs = state.integratedThroughMs;
    state = {...state, integratedThroughMs: endTimeMs, decayState, riseState,
      pending: endTimeMs === state.observedThroughMs ? null : {...state.pending, cursor}};
    return {fromTimeMs, timeMs: endTimeMs, excitation, unitExcitation: excitationArea,
      unitMeanKernel: kernelArea, quadratureDiscrepancy};
  }

  function snapshot() { return {schemaVersion, kind: 'wing-event-excitation', contract: copy(contract), ...copy(state)}; }
  function restore(input) {
    keys(input, ['schemaVersion', 'kind', 'contract', ...Object.keys(state)], 'snapshot');
    requireThat(input.schemaVersion === schemaVersion && input.kind === 'wing-event-excitation' &&
      JSON.stringify(input.contract) === contractText, 'snapshot contract mismatch');
    const s = {...input}; delete s.schemaVersion; delete s.kind; delete s.contract;
    requireThat(typeof s.initialized === 'boolean', 'invalid snapshot initialization flag');
    exactTime(s.integratedThroughMs, 'snapshot integrated time'); exactTime(s.observedThroughMs, 'snapshot observed time');
    requireThat(Number.isInteger(s.integratedThroughMs) && s.observedThroughMs % 2 === 0 &&
      s.observedThroughMs >= s.integratedThroughMs && s.observedThroughMs - s.integratedThroughMs <= 2,
    'invalid snapshot clock relationship');
    for (const key of ['decayState', 'riseState']) s[key] = values(s[key], UNIT_COUNT, n => finite(n) && n >= 0, key);
    s.counts = values(s.counts, UNIT_COUNT, n => Number.isInteger(n) && n >= 0 && n < MAX_EXACT, 'snapshot counts');
    s.ratesHz = values(s.ratesHz, UNIT_COUNT, n => finite(n) && n >= 0 && Math.fround(n) === n, 'snapshot rates');
    s.lastEventTimesMs = values(s.lastEventTimesMs, UNIT_COUNT, n => n === null || finite(n), 'snapshot last event times');
    const future = new Set(); const pendingByUnit = new Map();
    if (s.pending !== null) {
      keys(s.pending, ['fromTimeMs', 'timeMs', 'events', 'cursor'], 'snapshot pending');
      requireThat(s.initialized && s.pending.timeMs === s.observedThroughMs &&
        s.pending.fromTimeMs === s.observedThroughMs - 2 && s.pending.fromTimeMs >= 0 &&
        s.integratedThroughMs >= s.pending.fromTimeMs && s.integratedThroughMs < s.observedThroughMs,
      'invalid snapshot pending interval');
      const events = eventList(s.pending.events, s.pending.fromTimeMs, s.pending.timeMs);
      requireThat(s.pending.cursor === events.filter(e => e.timeMs <= s.integratedThroughMs).length,
        'snapshot pending cursor does not match integrated time');
      s.pending = {...s.pending, events};
      for (const event of events) {
        const k = positions.get(event.index); pendingByUnit.set(k, event);
        if (event.timeMs > s.integratedThroughMs) future.add(k);
      }
    } else requireThat(s.integratedThroughMs === s.observedThroughMs, 'snapshot has missing pending coverage');
    for (let k = 0; k < UNIT_COUNT; k++) {
      const count = s.counts[k], last = s.lastEventTimesMs[k], processed = count - (future.has(k) ? 1 : 0);
      requireThat((count === 0) === (last === null), 'snapshot count/last-event mismatch');
      if (last !== null) {
        eventTime(indices[k], last, 'snapshot last event');
        const maximumCount = dlm.has(indices[k]) ? Math.floor((Math.round(last * 10) - 1) / 100) + 1 : Math.floor((last - .5) / 2.5) + 1;
        requireThat(last > 0 && last <= s.observedThroughMs && count <= maximumCount,
          'snapshot has impossible spike history');
        if (pendingByUnit.has(k)) requireThat(last === pendingByUnit.get(k).timeMs, 'snapshot pending/last-event mismatch');
        else requireThat(last <= (s.pending?.fromTimeMs ?? s.integratedThroughMs), 'snapshot is missing a recent event');
      }
      requireThat(processed >= 0 && s.riseState[k] <= s.decayState[k] && s.decayState[k] <= processed + 1e-12 &&
        (processed !== 0 || (s.decayState[k] === 0 && s.riseState[k] === 0)), 'snapshot kernel state is inconsistent with counts');
    }
    if (!s.initialized) requireThat(s.integratedThroughMs === 0 && s.observedThroughMs === 0 && s.pending === null &&
      s.counts.every(x => x === 0) && s.ratesHz.every(x => x === 0), 'uninitialized snapshot contains history');
    state = s;
  }
  function readState() {
    return {...copy(state), indices: Uint32Array.from(indices), mappingIndices: Uint32Array.from(mappings, m => m.mappingIndex),
      mappings: copy(mappings), unitFamilies: unitFamilies.slice(), config: copy(priors)};
  }
  return Object.freeze({accept, finishInterval, readState, snapshot, restore});
}
