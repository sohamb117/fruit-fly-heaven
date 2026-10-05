// Pure numerical helpers for the diagnostic assay. No production controller.
export const RATE_LEVELS = [0, 1, 3, 5, 8, 12, 20, 40, 60, 80, 100];
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function rotate(matrix, vector) {
  return [0, 1, 2].map(i => matrix[i * 3] * vector[0] + matrix[i * 3 + 1] * vector[1] + matrix[i * 3 + 2] * vector[2]);
}
// MuJoCo free-joint generalized forces: translation is world-frame, rotation
// is root-local. The moment is about the free-joint origin, not whole-body COM.
export function worldComWrench(generalizedRoot, rootRotation, rootPosition, wholeCom) {
  const force = Array.from(generalizedRoot.slice(0, 3));
  const originTorque = rotate(rootRotation, generalizedRoot.slice(3, 6));
  const lever = cross(wholeCom.map((v, i) => v - rootPosition[i]), force);
  return [...force, ...originTorque.map((v, i) => v - lever[i])];
}
// Select equality rows by type/id before projecting J^T f. Contact and joint
// limit contributions must not be mislabeled as restraint reaction.
export function equalityRootForce(data, nv, equalityId) {
  const result = [0, 0, 0, 0, 0, 0];
  for (let row = 0; row < data.nefc; row++) {
    if (data.efc_type[row] !== 0 || data.efc_id[row] !== equalityId) continue;
    for (let column = 0; column < 6; column++) result[column] += data.efc_J[row * nv + column] * data.efc_force[row];
  }
  return result;
}
export function rateAt(time, rateHz, waveform) {
  if (waveform === 'step') return rateHz;
  if (waveform === 'ramp') return rateHz * Math.min(1, Math.max(0, time / .15));
  throw new Error(`Unknown waveform ${waveform}`);
}
export function groupRate(group, rateHz, selection) {
  if (group.kind !== 'asynchronous_wing') return 0;
  if (selection === 'all') return rateHz;
  if (selection === 'dlm') return group.target === 'dorsal_longitudinal_muscle' ? rateHz : 0;
  if (selection === 'dvm') return group.target === 'dorsoventral_muscle' ? rateHz : 0;
  if (selection === 'left') return group.joint.endsWith('left') ? rateHz : 0;
  throw new Error(`Unknown motor selection ${selection}`);
}
export class Moments {
  constructor(length = 1) {
    this.n = 0; this.sum = new Float64Array(length); this.sumSquares = new Float64Array(length);
    this.minimum = new Float64Array(length).fill(Infinity); this.maximum = new Float64Array(length).fill(-Infinity);
  }
  add(values) {
    if (values.length !== this.sum.length || !values.every(Number.isFinite)) throw new Error('Nonfinite or incorrectly sized observation');
    this.n++;
    for (let i = 0; i < values.length; i++) {
      this.sum[i] += values[i]; this.sumSquares[i] += values[i] ** 2;
      this.minimum[i] = Math.min(this.minimum[i], values[i]); this.maximum[i] = Math.max(this.maximum[i], values[i]);
    }
  }
  json() {
    const divide = values => Array.from(values, x => this.n ? x / this.n : null);
    return {samples: this.n, mean: divide(this.sum), rms: divide(this.sumSquares).map(x => x === null ? null : Math.sqrt(x)),
      minimum: Array.from(this.minimum, x => Number.isFinite(x) ? x : null), maximum: Array.from(this.maximum, x => Number.isFinite(x) ? x : null)};
  }
}
export function makeCases(pilot = false) {
  if (pilot) return [
    ...[0, 12, 80].map(rateHz => ({context: 'restrained', rateHz, waveform: 'step', selection: 'all', seconds: .2})),
    {context: 'flat_floor', rateHz: 80, waveform: 'step', selection: 'all', seconds: .2}
  ];
  return [
    ...['step', 'ramp'].flatMap(waveform => RATE_LEVELS.map(rateHz => ({context: 'restrained', rateHz, waveform, selection: 'all', seconds: .6}))),
    ...['dlm', 'dvm', 'left'].flatMap(selection => [3, 12, 40, 80].map(rateHz => ({context: 'restrained', rateHz, waveform: 'step', selection, seconds: .6}))),
    ...['air', 'flat_floor'].flatMap(context => ['step', 'ramp'].flatMap(waveform => [0, 3, 12, 40, 80, 100].map(rateHz => ({context, rateHz, waveform, selection: 'all', seconds: 1.2})))),
    ...['air', 'flat_floor'].flatMap(context => ['dlm', 'dvm', 'left'].flatMap(selection => [12, 80].map(rateHz => ({context, rateHz, waveform: 'step', selection, seconds: 1.2})))),
    ...['b2_left', 'b2_right', 'iii1_both'].map(steering => ({context: 'restrained', rateHz: 80, waveform: 'step', selection: 'all', steering, seconds: .6}))
  ];
}
export function makeFollowupCases() {
  return [
    ...[.75, .8, .81, .85, .9].map(rateHz => ({context: 'restrained', rateHz, waveform: 'step', selection: 'all', seconds: .6})),
    ...['restrained', 'flat_floor'].flatMap(context => [12, 80].flatMap(rateHz => [.012, .06, .12].map(deploymentTau =>
      ({context, rateHz, waveform: 'step', selection: 'all', seconds: context === 'restrained' ? .6 : 1.2, deploymentTau}))))
  ];
}
export function makeContinuousCases() {
  return [
    ...[.8,.81,.9,1,3,12,80].flatMap(rateHz => [0,.1,.25].map(deploymentPowerSpan => ({context:'restrained',rateHz,waveform:'step',selection:'all',seconds:.6,deploymentPowerSpan}))),
    ...[12,80].flatMap(rateHz => [0,.1,.25].map(deploymentPowerSpan => ({context:'flat_floor',rateHz,waveform:'step',selection:'all',seconds:1.2,deploymentPowerSpan})))
  ];
}
// MjData owns the borrowed vector. Its get() entries are individually owned
// copies in the pinned binding; dispose each copy, never the vector itself.
export function readInstabilityWarnings(vector, warningEnums) {
  const warnings = [];
  for (const name of ['mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL']) {
    const entry = vector.get(warningEnums[name].value);
    try { if (entry.number > 0) warnings.push({name,number:entry.number,lastinfo:entry.lastinfo}); }
    finally { entry.delete(); }
  }
  return warnings;
}
