import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {WasmBrain} from '../src/wasm.js';
import {historySlots, SPIKE_CAPACITY} from '../src/model.js';
import {fixture} from './fixture.mjs';

// Explicit binaries make this a differential check against the unmodified
// release, rather than comparing two calls through the same optimized kernel.
const baselinePath = process.env.BANC_NATIVE_BASELINE;
const candidatePath = process.env.BANC_NATIVE_CANDIDATE;
const configured = !!baselinePath && !!candidatePath;
const internal = {hunger: .7, insulin: .2, akh: .4};
const reports = [];

function model({ionic, uniformDelay, empty = false}) {
  const n = 12;
  const edges = empty ? [] : Array.from({length: n * 9}, (_, k) => ({
    source: (k * 7 + 3) % n,
    post: Math.floor(k / 9),
    receptor: k % 9,
    delay: uniformDelay ? 4 : [1, 2, 3, 4, 7, 11, 19, 27, 31][k % 9],
    weight: k % 13 === 0 ? 0 : .02 + (k % 11) * .03125,
  }));
  const value = fixture({n, edges, graded: [2, 7],
    gaps: [{a: 0, b: 4, weight: .08}, {a: 3, b: 9, weight: .04}],
    overrides: {0: {11: 2, 12: -1, 13: 1}, 4: {11: -1, 12: 2, 13: .5}, 8: {6: 150, 15: 3}}});
  if (ionic) value.manifest.intrinsic_models = {
    schema: 1, profile: 'dlm-snl-2023-v1',
    cells: [{index: 1, root_id: 'fixture1'}], ionic_step_ms: .1,
    initial_gates: {h: .146, b: .146}, event_policy: 'threshold-10ms-guard',
  };
  return value;
}

function inputAt(tick, n) {
  return Float32Array.from({length: n}, (_, i) =>
    i === 1 ? 108.75 + 7 * Math.sin(tick * .037) :
    i === 2 || i === 7 ? 18 + 12 * Math.sin(tick * .13 + i) :
    (tick + i * 3) % 23 < 11 ? 35 + i : -9 + i);
}

function exactSnapshot(a, b, label) {
  assert.equal(a.tick, b.tick, label + ': tick');
  assert.equal(a.timeMs, b.timeMs, label + ': time');
  const fields = [
    ['parameters', a.params, b.params, a.intrinsic.packedLength * 4],
    ['state 0', a.states[0], b.states[0], a.n * 8 * 4],
    ['state 1', a.states[1], b.states[1], a.n * 8 * 4],
    ['history', a.history, b.history, a.n * historySlots(a.model) * 4],
    ['kinetics and ionic state', a.kinetics, b.kinetics, a.intrinsic.kineticsLength * 4],
    ['events', a.events, b.events, (2 + SPIKE_CAPACITY * 2) * 4],
  ];
  const digest = createHash('sha256');
  for (const [name, pa, pb, length] of fields) {
    const left = a.core.HEAPU8.subarray(pa, pa + length);
    const right = b.core.HEAPU8.subarray(pb, pb + length);
    assert.equal(left.length, length);
    assert.equal(right.length, length);
    if (Buffer.compare(left, right) !== 0) {
      let index = 0;
      while (index < length && left[index] === right[index]) index++;
      assert.equal(right[index], left[index], `${label}: ${name}, byte ${index}`);
    }
    digest.update(left);
  }
  assert.deepEqual(b.readState(undefined, {includeSpikeTime: true}),
    a.readState(undefined, {includeSpikeTime: true}), label + ': public readout');
  return digest.digest('hex');
}

test('native hot-loop optimization is bit-identical to the supplied baseline', {skip: !configured}, async t => {
  const [{default: baselineCreate}, {default: candidateCreate}] = await Promise.all([
    import(pathToFileURL(path.resolve(baselinePath))),
    import(pathToFileURL(path.resolve(candidatePath))),
  ]);
  const [baselineCore, candidateCore] = await Promise.all([baselineCreate(), candidateCreate()]);
  for (const ionic of [false, true]) for (const uniformDelay of [false, true]) for (const gaps of [false, true]) {
    const name = `ionic=${ionic}, uniformDelay=${uniformDelay}, gaps=${gaps}`;
    await t.test(name, () => {
      const value = model({ionic, uniformDelay});
      const a = new WasmBrain(baselineCore, value), b = new WasmBrain(candidateCore, value);
      let checkpoints = 0, digest;
      try {
        digest = exactSnapshot(a, b, name + ': initial'); checkpoints++;
        for (let round = 0; round < 5; round++) for (const steps of [1, 4, 7, 128, 31, 65]) {
          const inputs = Array.from({length: steps}, (_, k) => inputAt(a.tick + k, a.n));
          if (round % 2) {
            a.step(steps, inputs[0], internal, gaps);
            b.step(steps, inputs[0], internal, gaps);
          } else {
            a.stepSequence(inputs, internal, gaps);
            b.stepSequence(inputs, internal, gaps);
          }
          digest = exactSnapshot(a, b, name + ': tick ' + a.tick); checkpoints++;
        }
        // Recompute factors on every call: no stale cached coefficients when
        // a direct native caller updates receptor parameters between ticks.
        const offset = a.n * 17;
        for (const brain of [a, b]) {
          brain.core.HEAPF32[brain.params / 4 + offset] = .875;
          brain.core.HEAPF32[brain.params / 4 + offset + 1] = 4.625;
        }
        a.step(31, inputAt(a.tick, a.n), internal, gaps);
        b.step(31, inputAt(b.tick, b.n), internal, gaps);
        digest = exactSnapshot(a, b, name + ': changed receptor'); checkpoints++;
        reports.push({name, ticks: a.tick, checkpoints, finalStateSha256: digest});
      } finally {a.dispose(); b.dispose();}
    });
  }

  await t.test('empty chemical graph and wrapped event ring retain exact state', () => {
    const value = fixture({n: 40});
    const a = new WasmBrain(baselineCore, value), b = new WasmBrain(candidateCore, value);
    const input = new Float32Array(value.manifest.neuron_count).fill(2000);
    let digest, checkpoints = 0;
    try {
      for (let k = 0; k < 24; k++) {
        a.step(128, input, internal); b.step(128, input, internal);
        digest = exactSnapshot(a, b, 'event ring tick ' + a.tick); checkpoints++;
      }
      assert(a.readState().totalSpikes > SPIKE_CAPACITY, 'exercise event ring wrap');
      reports.push({name: 'empty chemical graph and event ring wrap', ticks: a.tick, checkpoints, finalStateSha256: digest});
    } finally {a.dispose(); b.dispose();}
  });

  await t.test('DLM failure checks and partial state remain identical', () => {
    const value = model({ionic: true, uniformDelay: false});
    const a = new WasmBrain(baselineCore, value), b = new WasmBrain(candidateCore, value);
    try {
      for (const brain of [a, b]) {
        brain.step(4, inputAt(0, brain.n), internal);
        brain.core.HEAPF32[brain.kinetics / 4 + brain.n * 19] = NaN;
      }
      const failures = [a, b].map(brain => {
        try {brain.step(1, inputAt(4, brain.n), internal); assert.fail('expected ionic failure');}
        catch (error) {assert.match(error.message, /DLM ionic integration failed at cell 1/); return error.message;}
      });
      assert.equal(failures[0], failures[1]);
      assert.equal(a.tick, 4); assert.equal(b.tick, 4);
      // Temporarily clear only the wrapper's poison flag to inspect native
      // state. The kernels' failure marker and all partial writes stay intact.
      a.failed = b.failed = null;
      const digest = exactSnapshot(a, b, 'DLM failure');
      reports.push({name: 'DLM failure', ticks: a.tick, checkpoints: 1, finalStateSha256: digest});
    } finally {a.dispose(); b.dispose();}
  });

  if (process.env.BANC_NATIVE_EQUIVALENCE_REPORT && reports.length === 10) {
    const binary = file => fs.readFile(path.resolve(path.dirname(file), 'core.wasm'));
    const hashes = await Promise.all([baselinePath, candidatePath].map(async file => {
      const bytes = await binary(file);
      return {bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')};
    }));
    await fs.writeFile(process.env.BANC_NATIVE_EQUIVALENCE_REPORT, JSON.stringify({
      completedAt: new Date().toISOString(), baseline: hashes[0], candidate: hashes[1],
      scope: 'Small deterministic neural fixtures only; no full-graph or fly simulation.',
      comparison: 'Exact bytes of both state buffers, parameters, history, kinetics/ionic state and event ring at every checkpoint, plus public readouts.',
      cases: reports,
    }, null, 2) + '\n');
  }
});
