import {STEERING_MUSCLE_TYPES} from './training/flight-parameters.js';

// Experimental engineering interface, not measured neuromuscular physiology.
// This module has no body, task, reward, native-state or clock inputs.
export const MOTOR_DECODER_VERSION = 'banc-masked-motor-decoder-v1';
const SIDES = ['left', 'right'], AXES = ['yaw', 'roll', 'pitch'];
const LAGS = [0, 1, 4], BASES = ['constant', 'sin', 'cos'];
const POWER_TARGETS = {dorsal_longitudinal_muscle: 'dlm', dorsoventral_muscle: 'dvm'};
const UNIT_COUNT = 48, POWER_COUNT = 24, STEERING_COUNT = 648;
const validatedContracts = new WeakSet();
const finite = x => typeof x === 'number' && Number.isFinite(x);
const record = x => x !== null && typeof x === 'object' &&
  (Object.getPrototypeOf(x) === Object.prototype || Object.getPrototypeOf(x) === null);
const uint32 = x => Number.isInteger(x) && x >= 0 && x <= 0xffffffff;
const rootId = x => typeof x === 'string' && /^[1-9][0-9]*$/.test(x);
const fail = message => { throw new TypeError('Motor decoder: ' + message); };
const requireThat = (condition, message) => { if (!condition) fail(message); };
const clip = (x, min, max) => Math.max(min, Math.min(max, x));

function freeze(value) {
  if (value !== null && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

function sameSchema(actual, expected) {
  if (actual === expected) return true;
  if (Array.isArray(expected)) return Array.isArray(actual) &&
    Reflect.ownKeys(actual).length === expected.length + 1 && actual.length === expected.length &&
    expected.every((value, k) => sameSchema(actual[k], value));
  if (record(expected)) return record(actual) &&
    Reflect.ownKeys(actual).length === Object.keys(expected).length &&
    Object.keys(expected).every(key => Object.hasOwn(actual, key) && sameSchema(actual[key], expected[key]));
  return false;
}

function numericVector(value, length, label) {
  requireThat((Array.isArray(value) || ArrayBuffer.isView(value)) && value.length === length,
    label + ' requires exactly ' + length + ' entries');
  for (let k = 0; k < length; k++) requireThat(finite(value[k]), label + ' contains a nonfinite/non-numeric entry at ' + k);
}

function normalizeMuscle(m, mappingIndex) {
  requireThat(record(m), 'invalid wing muscle record');
  const steering = m.kind === 'wing_steering_assumption';
  const family = steering ? 'steering' : m.kind === 'asynchronous_wing' &&
    Object.hasOwn(POWER_TARGETS, m.target) ? POWER_TARGETS[m.target] : null;
  const prefix = steering ? 'wing_steer_' : 'wing_power_';
  const side = SIDES.find(s => m.joint === prefix + s);
  requireThat(family && side && m.sign === 1 && (!steering || STEERING_MUSCLE_TYPES.includes(m.target)),
    'unknown wing muscle target, kind, side or sign');
  const count = steering ? 1 : family === 'dlm' ? 5 : 7;
  requireThat(uint32(mappingIndex) && Array.isArray(m.indices) && m.indices.length === count &&
    m.indices.every(uint32) && Array.isArray(m.root_ids) && m.root_ids.length === count && m.root_ids.every(rootId),
  'wing muscle indices/root identities do not match its required unit count');
  return {mappingIndex, joint: m.joint, target: m.target, kind: m.kind, sign: 1,
    family, side, indices: m.indices.slice(), rootIds: m.root_ids.slice()};
}

function assemble(muscles) {
  requireThat(muscles.length === 28, 'expected exactly 28 wing muscle groups');
  const groupIds = new Set(), mappingIds = new Set(), unitIds = new Set(), roots = new Set();
  const units = [];
  muscles.forEach((m, k) => {
    const group = m.target + ':' + m.side;
    requireThat(!groupIds.has(group) && !mappingIds.has(m.mappingIndex) &&
      (k === 0 || m.mappingIndex > muscles[k - 1].mappingIndex), 'duplicate or unordered wing muscle mapping');
    groupIds.add(group); mappingIds.add(m.mappingIndex);
    m.indices.forEach((index, j) => {
      requireThat(!unitIds.has(index) && !roots.has(m.rootIds[j]), 'duplicate wing motor index or root identity');
      unitIds.add(index); roots.add(m.rootIds[j]);
      units.push({index, rootId: m.rootIds[j], mappingIndex: m.mappingIndex,
        target: m.target, family: m.family, side: m.side, sideIndex: SIDES.indexOf(m.side)});
    });
  });
  requireThat(units.length === UNIT_COUNT && units.filter(u => u.family === 'steering').length === 24,
    'expected exactly 48 wing motor neurons, including 24 steering and 24 power units');
  for (const side of SIDES) for (const target of [...STEERING_MUSCLE_TYPES, ...Object.keys(POWER_TARGETS)])
    requireThat(groupIds.has(target + ':' + side), 'missing bilateral wing muscle identity');
  units.sort((a, b) => a.index - b.index).forEach((u, position) => { u.position = position; });
  const unitFields = u => ({unitIndex: u.index, unitPosition: u.position, rootId: u.rootId,
    side: u.side, sideIndex: u.sideIndex, family: u.family, target: u.target});
  const parameters = [];
  for (const u of units.filter(u => u.family !== 'steering')) {
    parameters.push({name: `motor_power_mn_${u.index}_weight`, min: 0, max: 4, initial: 1,
      kind: 'power', ...unitFields(u), normalization: u.family === 'dlm' ? 1 / 10 : 1 / 14});
  }
  for (const u of units.filter(u => u.family === 'steering')) for (let axisIndex = 0; axisIndex < AXES.length; axisIndex++)
    for (const lagMs of LAGS) for (const basis of BASES) {
      const axis = AXES[axisIndex];
      parameters.push({name: `motor_steering_mn_${u.index}_${axis}_lag_${lagMs}ms_${basis}`,
        min: -.25, max: .25, initial: 0, kind: 'steering', ...unitFields(u), axis, axisIndex, lagMs, basis});
    }
  const contract = freeze({version: MOTOR_DECODER_VERSION, intervalMs: 1,
    indices: units.map(u => u.index), muscles, units, axes: AXES.slice(), lagsMs: LAGS.slice(), bases: BASES.slice(),
    powerParameterCount: POWER_COUNT, steeringParameterCount: STEERING_COUNT, parameters});
  validatedContracts.add(contract);
  return contract;
}

/** Derive the event adapter's sorted 48-unit order and an immutable 672-coordinate
 * anatomy mask from IO identities. Other (non-wing) muscle groups are ignored.
 * If the IO includes its motor-neuron table, selected identities must also agree
 * with that table; serialized contracts are checked against actual IO by callers.
 */
export function buildMotorDecoderContract(io) {
  requireThat(record(io) && Array.isArray(io.muscles), 'io.muscles must be an array');
  const muscles = [];
  io.muscles.forEach((m, mappingIndex) => {
    requireThat(record(m), 'invalid IO muscle record');
    const wing = ['wing_steering_assumption', 'asynchronous_wing'].includes(m.kind);
    const looksWing = typeof m.joint === 'string' && m.joint.startsWith('wing_') ||
      STEERING_MUSCLE_TYPES.includes(m.target) || Object.hasOwn(POWER_TARGETS, m.target);
    requireThat(wing || !looksWing, 'unknown mapping kind for a wing muscle');
    if (wing) muscles.push(normalizeMuscle(m, mappingIndex));
  });
  const contract = assemble(muscles);
  if (Object.hasOwn(io, 'motor_neurons')) {
    requireThat(Array.isArray(io.motor_neurons), 'io.motor_neurons must be an array');
    const selected = new Set(contract.indices), rows = new Map();
    for (const n of io.motor_neurons) if (selected.has(n?.index)) {
      requireThat(!rows.has(n.index), 'duplicate selected motor-neuron table index'); rows.set(n.index, n);
    }
    for (const u of contract.units) {
      const n = rows.get(u.index);
      requireThat(record(n) && n.root_id === u.rootId && n.side === u.side &&
        n.peripheral_target_type === u.target && n.super_class === 'motor',
      'wing identity disagrees with the motor-neuron table at index ' + u.index);
    }
  }
  return contract;
}

/** Validate a JSON-round-trippable contract without requiring the full IO asset.
 * Every field is regenerated from its 28 anatomical records. This verifies the
 * schema and mask, not independent anatomical provenance: compare with build(IO)
 * before using a saved contract on an actual model.
 */
export function validateMotorDecoderContract(value) {
  if (validatedContracts.has(value)) return value;
  requireThat(record(value) && Array.isArray(value.muscles), 'missing serialized contract');
  const muscles = value.muscles.map(m => {
    requireThat(record(m), 'invalid serialized muscle identity');
    return normalizeMuscle({...m, root_ids: m.rootIds}, m.mappingIndex);
  });
  const expected = assemble(muscles);
  requireThat(sameSchema(value, expected), 'serialized contract has unknown, missing or inconsistent fields');
  return expected;
}

/** Validate and own an explicit full vector. Bounds are engineering constraints. */
export function validateMotorDecoderVector(contract, vector) {
  const checked = validateMotorDecoderContract(contract);
  numericVector(vector, checked.parameters.length, 'parameter vector');
  const result = new Float64Array(vector.length);
  checked.parameters.forEach((p, k) => {
    requireThat(vector[k] >= p.min && vector[k] <= p.max, 'parameter outside bounds: ' + p.name);
    result[k] = vector[k];
  });
  return result;
}

/** Own a causal five-sample history. advance() consumes one completed 1 ms
 * unitExcitation interval, in contract.indices order. The caller must enforce its
 * external 1 ms schedule; no wall/body time or hidden controller is consulted.
 * Lag 0 is the most recently consumed interval, and unavailable history is zero.
 * sample/features accept wing phase in radians and never advance time.
 * Power is 0.5*(weighted mean 5 DLM + weighted mean 7 DVM), per side. Steering is
 * a signed, ipsilateral linear residual, clipped after summing to +/-0.25.
 */
export function createMotorDecoder(io, vector) {
  const contract = buildMotorDecoderContract(io), weights = validateMotorDecoderVector(contract, vector);
  const history = Array.from({length: 5}, () => new Float64Array(UNIT_COUNT));
  let cursor = 4, timeMs = 0;
  const parameters = contract.parameters;
  const phaseValues = phase => {
    requireThat(finite(phase), 'phase must be a finite number in radians');
    return [1, Math.sin(phase), Math.cos(phase)];
  };
  const historical = (lag, position) => history[(cursor - lag + 5) % 5][position];
  function advance(unitExcitation) {
    numericVector(unitExcitation, UNIT_COUNT, 'unit excitation');
    for (let k = 0; k < UNIT_COUNT; k++) requireThat(unitExcitation[k] >= 0 && unitExcitation[k] <= 1,
      'unit excitation must be in [0,1] at position ' + k);
    requireThat(Number.isSafeInteger(timeMs + 1), '1 ms history clock exhausted');
    cursor = (cursor + 1) % 5;
    history[cursor].set(unitExcitation); timeMs++;
    return timeMs;
  }
  function features(phase) {
    const basis = phaseValues(phase), values = new Float64Array(parameters.length);
    for (let k = 0; k < POWER_COUNT; k++) {
      const p = parameters[k]; values[k] = historical(0, p.unitPosition) * p.normalization;
    }
    for (let k = POWER_COUNT; k < parameters.length; k++) {
      const p = parameters[k]; values[k] = historical(p.lagMs, p.unitPosition) * basis[(k - POWER_COUNT) % 3];
    }
    return values;
  }
  function sample(phase) {
    const basis = phaseValues(phase), power = [0, 0], steering = [[0, 0, 0], [0, 0, 0]];
    for (let k = 0; k < POWER_COUNT; k++) {
      const p = parameters[k]; power[p.sideIndex] += weights[k] * historical(0, p.unitPosition) * p.normalization;
    }
    for (let k = POWER_COUNT; k < parameters.length; k++) {
      const p = parameters[k];
      steering[p.sideIndex][p.axisIndex] += weights[k] * historical(p.lagMs, p.unitPosition) * basis[(k - POWER_COUNT) % 3];
    }
    for (let side = 0; side < 2; side++) {
      power[side] = clip(power[side], 0, 1);
      for (let axis = 0; axis < 3; axis++) steering[side][axis] = clip(steering[side][axis], -.25, .25);
    }
    return {power, steering};
  }
  function reset() { history.forEach(h => h.fill(0)); cursor = 4; timeMs = 0; }
  function snapshot() {
    return {version: MOTOR_DECODER_VERSION, indices: contract.indices.slice(), weights: Array.from(weights),
      cursor, timeMs, history: history.map(h => Array.from(h))};
  }
  function restore(value) {
    requireThat(record(value) && Object.keys(value).sort().join(',') === 'cursor,history,indices,timeMs,version,weights', 'invalid snapshot fields');
    requireThat(value.version === MOTOR_DECODER_VERSION && sameSchema(value.indices, contract.indices) &&
      sameSchema(value.weights, Array.from(weights)), 'snapshot identity or weights differ');
    requireThat(Number.isSafeInteger(value.timeMs) && value.timeMs >= 0 && value.cursor === (4 + value.timeMs) % 5,
      'invalid snapshot clock');
    requireThat(Array.isArray(value.history) && value.history.length === 5, 'invalid snapshot history');
    for (const h of value.history) {
      numericVector(h, UNIT_COUNT, 'snapshot excitation');
      requireThat(Array.from(h).every(x => x >= 0 && x <= 1), 'snapshot excitation outside [0,1]');
    }
    // Validate everything before touching the live decoder. No body state is
    // accepted: this restores only its existing causal motor-input history.
    value.history.forEach((h, i) => history[i].set(h)); cursor = value.cursor; timeMs = value.timeMs;
  }
  return Object.freeze({advance, sample, features, reset, snapshot, restore, contract, get timeMs() { return timeMs; }});
}
