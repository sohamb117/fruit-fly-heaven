// DIAGNOSTIC ONLY. Prepare without loading MuJoCo; execution requires --run.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sha = x => createHash('sha256').update(x).digest('hex');
const TAU = 2 * Math.PI, wrap = x => ((x % TAU) + TAU) % TAU;
const finite = values => values.every(Number.isFinite);

// One-sided Jacobi SVD on three columns avoids squaring the condition number.
// Fixed sweep budget; used for both the three-axis inverse and response rank.
export function singularValues3(rows) {
  assert(rows.length > 0 && rows.every(row => row.length === 3 && finite(row)), 'Finite three-column matrix required');
  const scale = Math.max(...rows.flat().map(Math.abs));
  if (scale === 0) return [0, 0, 0];
  const a = rows.map(row => row.map(x => x / scale));
  for (let sweep = 0; sweep < 40; sweep++) {
    let rotated = false;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      let pp = 0, qq = 0, pq = 0;
      for (const row of a) { pp += row[p] ** 2; qq += row[q] ** 2; pq += row[p] * row[q]; }
      if (Math.abs(pq) <= 1e-15 * Math.sqrt(pp * qq)) continue;
      const theta = .5 * Math.atan2(2 * pq, qq - pp), c = Math.cos(theta), s = Math.sin(theta);
      for (const row of a) { const x = row[p], y = row[q]; row[p] = c * x - s * y; row[q] = s * x + c * y; }
      rotated = true;
    }
    if (!rotated) break;
    assert(sweep < 39, 'Three-column SVD did not converge');
  }
  return [0, 1, 2].map(j => Math.hypot(...a.map(row => row[j])) * scale).sort((a, b) => b - a);
}

export function responseRank(rows, relativeTolerance = 1e-6, absoluteTolerance = 1e-12) {
  const singularValues = singularValues3(rows), threshold = Math.max(absoluteTolerance, relativeTolerance * singularValues[0]);
  return {rows: rows.length, singularValues, threshold, relativeTolerance, absoluteTolerance,
    rank: singularValues.filter(value => value > threshold).length,
    smallestToLargestRatio: singularValues[0] ? singularValues[2] / singularValues[0] : 0};
}

export function recoverMoment(axes, anchors, tau, {anchorToleranceCm = 1e-10, conditionMaximum = 1e4} = {}) {
  assert(axes.length === 3 && anchors.length === 3 && axes.every(a => a.length === 3 && finite(a)) &&
    anchors.every(a => a.length === 3 && finite(a)) && tau.length === 3 && finite(tau), 'Malformed hinge observation');
  const anchorSpreadCm = Math.max(...anchors.map(a => Math.hypot(...a.map((x, i) => x - anchors[0][i]))));
  assert(anchorSpreadCm <= anchorToleranceCm, 'Wing hinge anchors are not coincident');
  assert(axes.every(a => Math.abs(Math.hypot(...a) - 1) < 1e-10), 'Hinge axis is not a unit vector');
  const singularValues = singularValues3(axes), condition = singularValues[0] / singularValues[2];
  assert(Number.isFinite(condition) && condition <= conditionMaximum, 'Ill-conditioned hinge axes');
  const a = axes.map((row, i) => [...row, tau[i]]);
  for (let column = 0; column < 3; column++) {
    let pivot = column;
    for (let row = column + 1; row < 3; row++) if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    [a[column], a[pivot]] = [a[pivot], a[column]];
    const divisor = a[column][column]; assert(divisor !== 0);
    for (let j = column; j < 4; j++) a[column][j] /= divisor;
    for (let row = 0; row < 3; row++) if (row !== column) {
      const factor = a[row][column]; for (let j = column; j < 4; j++) a[row][j] -= factor * a[column][j];
    }
  }
  const moment = a.map(row => row[3]), residual = axes.map((row, i) => row.reduce((s, x, j) => s + x * moment[j], 0) - tau[i]);
  const residualMaximum = Math.max(...residual.map(Math.abs));
  assert(finite(moment) && residualMaximum <= 1e-12 + 1e-10 * Math.max(...tau.map(Math.abs)), 'Hinge moment reconstruction residual');
  return {momentWorld: moment, magnitude: Math.hypot(...moment), condition, singularValues, anchorSpreadCm, residualMaximum};
}

export function analyzeLoads(states, deltaOmega) {
  assert(states.length === 16 && Number.isFinite(deltaOmega) && deltaOmega > 0);
  const rows = states.map(state => {
    const zero = state.evaluations.find(e => e.axis === null);
    assert(state.evaluations.length === 7 && zero);
    const sides = [0, 1].map(side => {
      const odd = [], even = [], momentDerivative = [];
      for (let axis = 0; axis < 3; axis++) {
        const plus = state.evaluations.find(e => e.axis === axis && e.sign === 1).wings[side];
        const minus = state.evaluations.find(e => e.axis === axis && e.sign === -1).wings[side];
        odd.push((plus.magnitude - minus.magnitude) / 2);
        even.push((plus.magnitude + minus.magnitude) / 2 - zero.wings[side].magnitude);
        momentDerivative.push(plus.momentWorld.map((x, j) => (x - minus.momentWorld[j]) / (2 * deltaOmega)));
      }
      return {side: side ? 'right' : 'left', baselineLoad: zero.wings[side].magnitude,
        plusMinusLoadContrast: odd.map(x => 2 * x), oddLoad: odd, evenLoadChange: even,
        magnitudeDerivative: odd.map(x => x / deltaOmega), momentDerivativeByInputAxis: momentDerivative};
    });
    return {sampleIndex: state.sampleIndex, relativeTimeMs: state.relativeTimeMs, tableClockPhaseRad: state.tableClockPhaseRad, sides};
  });
  const summarize = selected => ({sampleIndices: selected.map(row => row.sampleIndex),
    magnitude: responseRank(selected.flatMap(row => row.sides.map(side => side.magnitudeDerivative))),
    // An upper-bound diagnostic only: these Cartesian components are NOT
    // assigned to individual BANC cells or proposed as measured tunings.
    momentComponents: responseRank(selected.flatMap(row => row.sides.flatMap(side => [0, 1, 2].map(j =>
      side.momentDerivativeByInputAxis.map(axis => axis[j])))))});
  return {rows, fullHalfMs: summarize(rows), everyTwoMs: [0, 1, 2, 3].map(offset => ({offsetMs: offset * .5,
    ...summarize(rows.filter(row => row.sampleIndex % 4 === offset))})),
    interpretation: 'Ranks describe local raw-load observability under prescribed wing states at ±1 rad/s. They do not establish neural decoding, physiological tuning, closed-loop stability, or an actual runtime sensory implementation. Repeating each 2 ms value under zero-order hold adds no independent rows.'};
}

async function main() {
  const [outputArg, flag] = process.argv.slice(2);
  assert(outputArg && ['--prepare-only', '--run'].includes(flag) && process.argv.length === 4,
    'Usage: node scripts/audit-tegula-load-observability.mjs reports/new-output --prepare-only|--run');
  const output = path.resolve(root, outputArg), reports = path.join(root, 'reports'), execute = flag === '--run';
  assert(output.startsWith(reports + path.sep), 'Output must be inside reports');
  const sourceFiles = ['scripts/audit-tegula-load-observability.mjs', 'scripts/motor-wing-calibration-helpers.mjs',
    'web/flybody-wings.js', 'web/training/flight-parameters.js', 'models/flybody-mujoco.xml', 'models/flybody-mujoco.json',
    'models/flybody-wing-actuation.json', 'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
    'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
  const artifactFiles = ['reports/flight-classical-control/result.json', 'reports/flight-classical-control/fixed-nonwing.xml',
    'reports/flight-tegula-inputs/anatomy.json'];
  const bytes = Object.fromEntries(await Promise.all([...sourceFiles, ...artifactFiles].map(async file => [file, await fs.readFile(path.join(root, file))])));
  const sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, sha(bytes[file])])), artifactHashes = Object.fromEntries(artifactFiles.map(file => [file, sha(bytes[file])]));
  const metadata = JSON.parse(bytes['models/flybody-mujoco.json']), reference = JSON.parse(bytes[artifactFiles[0]]);
  assert(reference.completed && reference.sourceUnchanged && reference.pairedInitialStateIdentical);
  assert.equal(sha(bytes[artifactFiles[1]]), reference.modelHash);
  for (const file of ['models/flybody-mujoco.xml', 'models/flybody-mujoco.json', 'models/flybody-wing-actuation.json'])
    assert.equal(sourceHashes[file], reference.sourceHashes[file], 'Historical plant/trim context changed: ' + file);
  assert.equal(sourceHashes['models/flybody-mujoco.xml'], metadata.xml_sha256);
  assert.deepEqual(metadata.wing_actuation, JSON.parse(bytes['models/flybody-wing-actuation.json']));
  assert(!Object.hasOwn(metadata.wing_actuation, 'steering_force_reference'), 'Historical trim uses the uncentered absolute-force basis');
  const muscleNames = Object.keys(metadata.wing_actuation.steering), controls = reference.trimControls.slice();
  const controlNames = ['left', 'right'].flatMap(side => muscleNames.map(name => `${side}:${name}`)).concat('common_power');
  assert.deepEqual(controlNames, reference.controlNames); assert.equal(controls.length, 25); assert(finite(controls));
  const frequencyHz = metadata.wing_actuation.frequency_hz, warmupMs = 100, sampleMs = .5, samples = 16, deltaOmega = 1;
  const rootPose = [0, 0, 10, 1, 0, 0, 0], initialPhase = wrap(-TAU * frequencyHz * warmupMs / 1000);
  const plan = {schemaVersion: 1, kind: 'tegula-native-load-observability-plan', sourceHashes, artifactHashes,
    scope: 'Raw aerodynamic wing-hinge loads only. No BANC, afferent current, neural gain, optimizer, controller, new physics, or free flight.',
    modelHash: reference.modelHash, expectedDimensions: {nq: 13, nv: 12, njnt: 7, nu: 6, neq: 0}, rootPose,
    operatingPoint: {controlNames, syntheticNormalizedForces: controls, steeringForceReference: null,
      interpreter: 'Current pinned FlyBodyWings legacy branch; all27 default multipliers1. Historical trim includes both b1 channels unchanged.',
      bypass: 'All wing force commands bypass BANC and muscle dynamics. They only generate a fixed reference trajectory; no force fitting or adaptive compensation.'},
    stateGeneration: {warmupMs, nativeStepMs: .05, targetStepMs: .2, initialTableClockPhaseRad: initialPhase, frequencyHz,
      sampleSpacingMs: sampleMs, samples, sampledTimesAfterWarmupMs: Array.from({length: samples}, (_, i) => i * sampleMs),
      nativeSteps: 2150, nativeDurationMs: 107.5, rootRestraint: 'Write root pose and zero root velocity before each state-generation step. Wing coordinates remain dynamic.',
      phase: '16 consecutive0.5ms states, not16 uniformly spaced phases. Record actual extrapolated table clock and last generated target phase separately; wing motion is native.'},
    perturbations: {frame: 'Native free-root local angular-velocity coordinates; root quaternion identity and zero root translation velocity.',
      deltaOmegaRadPerSecond: deltaOmega, evaluationsPerState: 7, explicitDiagnosticForwardEvaluations: 112,
      note: '112 excludes the single initialization forward and forward work internal to the bounded state-generation mj_step calls. No integration occurs in the112 evaluations.',
      restore: ['time', 'qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart'], onlyChangedInput: 'qvel[3+axis] is0,+1,or−1rad/s; all other restored inputs are identical.'},
    moment: {formula: 'A rows=data.xaxis[resolved wing joint IDs]. tau=data.qfrc_fluid[resolved wing DOFs]. Solve A*Mworld=tau at coincident data.xanchor.',
      units: 'Moment g cm^2/s^2; derivative moment/(rad/s).', interpretation: 'Aerodynamic moment about each native wing hinge. No compliant tegula, receptor strain, or total joint reaction is modeled.',
      gates: {coincidentAnchorToleranceCm: 1e-10, maximumAxisCondition2: 1e4, reconstructionAbsoluteTolerance: 1e-12,
        reconstructionRelativeTolerance: 1e-10, responseRankRelativeTolerance: 1e-6, responseRankAbsoluteTolerance: 1e-12}},
    analysis: {loads: 'Report complete world moments, their magnitudes, per-axis±contrasts, odd/even finite-difference parts, and raw3-column response singular values.',
      sampling: 'Full0.5ms series and each of four exact2ms offset subsets; no interpolation, cycle averaging, phase tuning, or chosen winner. Zero-order hold duplicates samples without adding rank.',
      rankIsOutcomeNotGate: true, recruitment: 'None applied. An optional future common monotone recruitment model is a hypothesis, not needed for this assay.'},
    numericalGates: ['Leaf wing bodies with exactly3hinge coordinates and coincident anchors.', 'No contacts, instability warnings, nonfinite values, or applied external forces.',
      'All112 forward evaluations retain the prescribed input state and do not advance time.', 'All source and artifact pins unchanged.']};
  await fs.mkdir(output, {recursive: true});
  assert((await fs.realpath(output)).startsWith((await fs.realpath(reports)) + path.sep), 'Output symlink escapes reports');
  const existing = await fs.readdir(output);
  assert(!existing.some(name => ['result.json', 'run.lock', 'failure.json'].includes(name)), 'Output already has an attempted run; use a new directory');
  const writeSameOrNew = async (name, content) => {
    try { assert((await fs.readFile(path.join(output, name))).equals(content), 'Prepared artifact changed: ' + name); }
    catch (error) { if (error.code !== 'ENOENT') throw error; await fs.writeFile(path.join(output, name), content, {flag: 'wx'}); }
  };
  await writeSameOrNew('plan.json', Buffer.from(JSON.stringify(plan, null, 2) + '\n'));
  await writeSameOrNew('source.used.mjs', bytes[sourceFiles[0]]);
  if (!execute) { console.log(JSON.stringify({prepared: true, nativeExecution: false, output, explicitDiagnosticForwardEvaluations: 112})); return; }
  // Acquire ownership before creating result/failure artifacts. A rejected
  // duplicate invocation must not overwrite another run's evidence.
  await fs.writeFile(path.join(output, 'run.lock'), JSON.stringify({pid: process.pid, startedAt: new Date().toISOString()}) + '\n', {flag: 'wx'});
  const verifyPins = async () => { for (const [file, expected] of Object.entries({...sourceHashes, ...artifactHashes}))
    assert.equal(sha(await fs.readFile(path.join(root, file))), expected, 'Pinned dependency changed: ' + file); };
  let mj, model, data;
  const result = {schemaVersion: 1, kind: 'tegula-native-load-observability-result', sourceHashes, artifactHashes,
    planSha256: sha(await fs.readFile(path.join(output, 'plan.json'))), nodeVersion: process.version, startedAt: new Date().toISOString(),
    completed: false, explicitDiagnosticForwardEvaluations: 0, states: []};
  try {
    await verifyPins();
    const {default: loadMujoco} = await import('../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js');
    mj = await loadMujoco(); model = mj.MjModel.from_xml_string(String(bytes[artifactFiles[1]])); data = new mj.MjData(model);
    result.nativeVersion = mj.mj_versionString();
    assert.deepEqual(Object.fromEntries(Object.keys(plan.expectedDimensions).map(key => [key, model[key]])), plan.expectedDimensions);
    assert.equal(model.opt.timestep, .00005); assert.equal(model.jnt_type[0], 0);
    assert.equal(model.nmocap, 0); assert.equal(model.na, 0);
    const resolve = (kind, name) => { const id = mj.mj_name2id(model, mj.mjtObj[kind].value, name); assert(id >= 0, name); return id; };
    const names = ['left', 'right'].flatMap(side => ['yaw', 'roll', 'pitch'].map(axis => `wing_${axis}_${side}`));
    const joints = names.map(name => { const id = resolve('mjOBJ_JOINT', name); assert.equal(model.jnt_type[id], 3); return {
      ...metadata.joints.find(j => j.name === name), id, body: model.jnt_bodyid[id], qpos: model.jnt_qposadr[id], dof: model.jnt_dofadr[id],
      range: Array.from(model.jnt_range.slice(id * 2, id * 2 + 2))}; });
    const actuators = names.map(name => { const id = resolve('mjOBJ_ACTUATOR', name); return {name, id, range: Array.from(model.actuator_ctrlrange.slice(id * 2, id * 2 + 2))}; });
    for (let side = 0; side < 2; side++) {
      const body = joints[side * 3].body;
      assert(joints.slice(side * 3, side * 3 + 3).every(j => j.body === body));
      assert.equal(model.body_jntnum[body], 3);
      assert(!Array.from(model.body_parentid).some((parent, child) => child !== body && parent === body), 'Wing body must be a leaf');
    }
    result.resolvedJoints = joints; result.resolvedActuators = actuators;
    const wings = new FlyBodyWings({...metadata, joints, actuators}); wings.phase = initialPhase;
    const steering = Object.fromEntries(['left', 'right'].map((side, k) => [side,
      Object.fromEntries(muscleNames.map((name, i) => [name, controls[k * 12 + i]]))]));
    const retainRoot = () => { data.qpos.set(rootPose); data.qvel.fill(0, 0, 6); };
    const assertSafe = () => {
      assert(finite(Array.from(data.qpos)) && finite(Array.from(data.qvel)) && finite(Array.from(data.qfrc_fluid)));
      assert.equal(data.ncon, 0, 'Contact invalidates the declared airborne reference');
      for (const array of [data.qfrc_applied, data.xfrc_applied]) assert(Array.from(array).every(x => x === 0));
      assert.deepEqual(readInstabilityWarnings(data.warning, mj.mjtWarning), []);
    };
    const snapshot = () => Object.fromEntries(['qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart'].map(key => [key, Array.from(data[key])]));
    mj.mj_resetData(model, data); retainRoot(); for (const joint of joints) data.qpos[joint.qpos] = joint.neutral;
    mj.mj_forward(model, data); assertSafe();
    let lastTargetPhase = null, lastTargetNativeTimeMs = null;
    for (let step = 0; step <= 2150; step++) {
      retainRoot();
      if (step % 4 === 0) {
        lastTargetPhase = wings.phase; lastTargetNativeTimeMs = step * .05;
        wings.step(data.qpos, data.ctrl, controls[24], controls[24], steering, .0002);
      }
      if (step >= 2000 && (step - 2000) % 10 === 0) {
        const sampleIndex = (step - 2000) / 10;
        result.states.push({sampleIndex, relativeTimeMs: sampleIndex * .5, nativeTimeSeconds: data.time,
          tableClockPhaseRad: wrap(initialPhase + TAU * frequencyHz * data.time),
          lastGeneratedTargetPhaseRad: lastTargetPhase, lastGeneratedTargetNativeTimeMs: lastTargetNativeTimeMs,
          targetGeneratorNextPhaseRad: wings.phase, wingState: wings.controlState(),
          input: snapshot(), evaluations: []});
      }
      if (step === 2150) break;
      mj.mj_step(model, data); assertSafe();
    }
    assert.equal(result.states.length, 16);
    result.stateGeneration = {nativeSteps: 2150, finalNativeTimeSeconds: data.time, initializationForwardEvaluations: 1};
    const perturbations = [{axis: null, sign: 0}, ...[0, 1, 2].flatMap(axis => [1, -1].map(sign => ({axis, sign})))];
    for (const state of result.states) {
      state.inputSha256 = sha(JSON.stringify({time: state.nativeTimeSeconds, ...state.input}));
      for (const perturbation of perturbations) {
        mj.mj_resetData(model, data); data.time = state.nativeTimeSeconds;
        for (const key of Object.keys(state.input)) data[key].set(state.input[key]);
        if (perturbation.axis !== null) data.qvel[3 + perturbation.axis] = perturbation.sign * deltaOmega;
        const expected = snapshot();
        mj.mj_forward(model, data); result.explicitDiagnosticForwardEvaluations++;
        assertSafe(); assert.equal(data.time, state.nativeTimeSeconds);
        // qacc_warmstart is a solver output as well as an input; all actual
        // kinematic/control inputs must remain bit-identical after forward.
        for (const key of ['qpos', 'qvel', 'ctrl', 'act']) assert.deepEqual(Array.from(data[key]), expected[key], 'Forward changed ' + key);
        const observed = [0, 1].map(side => {
          const js = joints.slice(side * 3, side * 3 + 3);
          const axes = js.map(j => Array.from(data.xaxis.slice(j.id * 3, j.id * 3 + 3)));
          const anchors = js.map(j => Array.from(data.xanchor.slice(j.id * 3, j.id * 3 + 3)));
          const tau = js.map(j => data.qfrc_fluid[j.dof]);
          return {side: side ? 'right' : 'left', axesWorld: axes, anchorsWorldCm: anchors, fluidGeneralizedTorques: tau,
            ...recoverMoment(axes, anchors, tau)};
        });
        state.evaluations.push({...perturbation, rootOmega: Array.from(data.qvel.slice(3, 6)), timeAdvanced: 0, inputStatePreserved: true, wings: observed});
      }
      console.log(JSON.stringify({sample: state.sampleIndex + 1, samples: 16, explicitDiagnosticForwardEvaluations: result.explicitDiagnosticForwardEvaluations}));
    }
    assert.equal(result.explicitDiagnosticForwardEvaluations, 112);
    result.analysis = analyzeLoads(result.states, deltaOmega);
    await verifyPins(); result.sourceUnchanged = true; result.completed = true; result.finishedAt = new Date().toISOString();
    await fs.writeFile(path.join(output, 'result.json'), JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
    console.log(JSON.stringify({completed: true, fullRank: result.analysis.fullHalfMs.magnitude.rank,
      twoMsRanks: result.analysis.everyTwoMs.map(row => row.magnitude.rank), output}));
  } catch (error) {
    result.error = String(error?.stack || error); result.finishedAt = new Date().toISOString();
    await fs.writeFile(path.join(output, 'failure.json'), JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
    throw error;
  } finally { data?.delete(); model?.delete(); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
