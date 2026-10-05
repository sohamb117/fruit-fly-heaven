import test from 'node:test';
import assert from 'node:assert/strict';
import {singularValues3, responseRank, recoverMoment, analyzeLoads} from './audit-tegula-load-observability.mjs';

const close = (a, b, tolerance = 1e-11) => assert(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
const origins = [[1, 2, 3], [1, 2, 3], [1, 2, 3]];
test('singular values and rank retain a known weak independent direction', () => {
  // An orthogonal change of rows preserves the three known column norms.
  const a = [[3 / Math.sqrt(2), 2 / Math.sqrt(2), 0], [3 / Math.sqrt(2), -2 / Math.sqrt(2), 0], [0, 0, 1e-8]];
  singularValues3(a).forEach((x, i) => close(x, [3, 2, 1e-8][i]));
  assert.equal(responseRank(a).rank, 2);
  assert.equal(responseRank(a, 1e-10).rank, 3);
  assert.deepEqual(singularValues3([[0, 0, 0], [0, 0, 0]]), [0, 0, 0]);
  assert.equal(responseRank([[1, 2, 3], [2, 4, 6], [-1, -2, -3]]).rank, 1);
});

test('moment reconstruction respects nonorthogonal hinge axes and a translated anchor', () => {
  const axes = [[1, 0, 0], [Math.SQRT1_2, Math.SQRT1_2, 0], [0, 0, 1]];
  const actual = [2, -3, .25], tau = [2, -Math.SQRT1_2, .25];
  const r = recoverMoment(axes, origins, tau);
  r.momentWorld.forEach((x, i) => close(x, actual[i]));
  close(r.magnitude, Math.hypot(...actual));
  close(r.condition, 1 + Math.SQRT2);
  assert(r.residualMaximum < 1e-14);
});

test('unresolved geometry and nonfinite loads fail closed', () => {
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  assert.throws(() => recoverMoment(identity, [[0, 0, 0], [1e-7, 0, 0], [0, 0, 0]], [0, 0, 0]), /coincident/);
  assert.throws(() => recoverMoment([[1, 0, 0], [1, 0, 0], [0, 0, 1]], origins, [1, 1, 1]), /conditioned/);
  assert.throws(() => recoverMoment(identity, origins, [0, NaN, 0]), /Malformed/);
});

test('actual two-ms subsets expose information loss instead of borrowing other phases', () => {
  const states = Array.from({length: 16}, (_, sampleIndex) => {
    // Per-phase magnitude sensitivity is one of three independent axes.
    // Each exact2ms offset sees only one phase class; the full series sees3.
    const axisAtThisPhase = Math.min(sampleIndex % 4, 2);
    const evaluations = [{axis: null, sign: 0}, ...[0, 1, 2].flatMap(axis => [1, -1].map(sign => ({axis, sign})))].map(e => ({...e,
      wings: [0, 1].map(() => ({magnitude: 10 + (e.axis === axisAtThisPhase ? e.sign : 0),
        momentWorld: [10 + (e.axis === axisAtThisPhase ? e.sign : 0), 0, 0]}))}));
    return {sampleIndex, relativeTimeMs: sampleIndex * .5, tableClockPhaseRad: sampleIndex, evaluations};
  });
  const result = analyzeLoads(states, 1);
  assert.equal(result.fullHalfMs.magnitude.rank, 3);
  assert.deepEqual(result.everyTwoMs.map(x => x.magnitude.rank), [1, 1, 1, 1]);
  assert.deepEqual(result.everyTwoMs[2].sampleIndices, [2, 6, 10, 14]);
  assert.deepEqual(result.rows[0].sides[0].plusMinusLoadContrast, [2, 0, 0]);
  assert.deepEqual(result.rows[0].sides[0].evenLoadChange, [0, 0, 0]);
});
