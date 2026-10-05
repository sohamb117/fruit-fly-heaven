// DIAGNOSTIC ONLY: four matched restrained phase pairs, no brain/controller/fit.
// node scripts/probe-wing-event-phase.mjs reports/wing-event-phase --prepare-only
// Omit --prepare-only only when the eight serial native runs are authorized.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore, WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyWings} from '../web/flybody-wings.js';
import {createWingEventExcitation, DEFAULT_WING_EVENT_PRIORS} from '../web/flybody-wing-event-excitation.js';
import {worldComWrench, readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [outputArg, flag] = process.argv.slice(2);
assert(outputArg && (!flag || flag === '--prepare-only') && process.argv.length <= 4,
  'Usage: node scripts/probe-wing-event-phase.mjs reports/new-output [--prepare-only]');
const output = path.resolve(root, outputArg), reports = path.join(root, 'reports'), prepareOnly = flag === '--prepare-only';
assert(output.startsWith(reports + path.sep), 'Output must be inside reports');
const sha = bytes => createHash('sha256').update(bytes).digest('hex'), TAU = 2 * Math.PI;
const wrap = x => ((x % TAU) + TAU) % TAU, degrees = x => x * 180 / Math.PI;
const sourceFiles = ['scripts/probe-wing-event-phase.mjs', 'scripts/motor-wing-calibration-helpers.mjs',
  'web/flybody-wing-event-excitation.js', 'web/flybody-wings.js', 'web/training/flight-parameters.js',
  'models/flybody-mujoco.xml', 'models/flybody-mujoco.json', 'models/flybody-wing-actuation.json',
  'data/prepared/banc888/io.json', 'packages/banc-runtime/src/wasm.js', 'packages/banc-runtime/src/model.js',
  'packages/banc-runtime/native/core.cpp', 'packages/banc-runtime/dist/core.js', 'packages/banc-runtime/dist/core.wasm',
  'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js',
  'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm'];
const artifactFiles = ['reports/flight-classical-control/result.json', 'reports/flight-classical-control/fixed-nonwing.xml'];
const bytes = Object.fromEntries(await Promise.all([...sourceFiles, ...artifactFiles].map(async file =>
  [file, await fs.readFile(path.join(root, file))])));
const sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, sha(bytes[file])]));
const artifactHashes = Object.fromEntries(artifactFiles.map(file => [file, sha(bytes[file])]));
const reference = JSON.parse(bytes[artifactFiles[0]]), metadata = JSON.parse(bytes['models/flybody-mujoco.json']);
const io = JSON.parse(bytes['data/prepared/banc888/io.json']), reducedXml = String(bytes[artifactFiles[1]]);
assert(reference.completed && reference.sourceUnchanged && reference.pairedInitialStateIdentical);
assert(reference.cases.some(row => row.mode === 'closed_loop' && row.meetsDeclaredPositiveControl));
assert.equal(sha(bytes[artifactFiles[1]]), reference.modelHash, 'Historical reduced model changed');
assert.equal(sha(bytes['models/flybody-mujoco.xml']), metadata.xml_sha256);
assert.deepEqual(metadata.wing_actuation, JSON.parse(bytes['models/flybody-wing-actuation.json']));
for (const file of ['models/flybody-mujoco.xml', 'models/flybody-mujoco.json', 'models/flybody-wing-actuation.json'])
  assert.equal(sourceHashes[file], reference.sourceHashes[file], 'Historical model context changed: ' + file);
assert(!Object.hasOwn(metadata.wing_actuation, 'steering_force_reference'), 'Use the declared legacy absolute-force basis');
const muscleNames = Object.keys(metadata.wing_actuation.steering);
const controlNames = ['left', 'right'].flatMap(side => muscleNames.map(name => `${side}:${name}`)).concat('common_power');
assert.deepEqual(controlNames, reference.controlNames);
const mappings = io.muscles.map((m, mappingIndex) => ({...m, mappingIndex}))
  .filter(m => ['asynchronous_wing', 'wing_steering_assumption'].includes(m.kind));
const selectedMapping = mappings.find(m => m.joint === 'wing_steer_left' && m.target === 'b1_muscle');
assert(selectedMapping && selectedMapping.indices.length === 1);
const selectedIndex = selectedMapping.indices[0], selectedMuscle = mappings.indexOf(selectedMapping);
const selectedControl = controlNames.indexOf('left:b1_muscle'), background = reference.trimControls.slice();
background[selectedControl] = 0;
const indices = [...new Set(mappings.flatMap(m => m.indices))].sort((a, b) => a - b), selectedUnit = indices.indexOf(selectedIndex);
assert.equal(indices.length, 48); assert.equal(mappings.length, 28);
const eventMs = 20, durationMs = 140, warmupMs = 100, frequencyHz = metadata.wing_actuation.frequency_hz;
const rootPose = [0, 0, 10, 1, 0, 0, 0], weight = metadata.mass_g * 981, lengthCm = .27;
const phases = [0, 90, 180, 270].map((requestedDegrees, k) => {
  const offsetMs = [1, 2.5, 3.5, 0][k], nominalEventPhase = wrap(TAU * frequencyHz * (offsetMs + eventMs) / 1000);
  // The offsets are selected on the existing 0.5 ms grid solely for proximity
  // to four declared clock phases, never from force/torque measurements.
  const candidates = Array.from({length: 9}, (_, tick) => tick * .5);
  const error = ms => Math.abs(degrees(wrap(TAU * frequencyHz * (eventMs + ms) / 1000 - requestedDegrees * Math.PI / 180 + Math.PI)) - 180);
  assert(error(offsetMs) <= Math.min(...candidates.map(error)) + 1e-10);
  return {requestedDegrees, offsetMs, initialWarmPhaseRad: wrap(TAU * frequencyHz * (offsetMs - warmupMs) / 1000),
    nominalEventPhaseRad: nominalEventPhase, nominalEventPhaseDegrees: degrees(nominalEventPhase),
    nominalQuantizationErrorDegrees: degrees(wrap(nominalEventPhase - requestedDegrees * Math.PI / 180 + Math.PI)) - 180};
});
const config = structuredClone(DEFAULT_WING_EVENT_PRIORS);
const contract = createWingEventExcitation({io, config}).readState(); // pure mapping/config validation only
assert.deepEqual(Array.from(contract.indices), indices);
assert.deepEqual(Array.from(contract.mappingIndices), mappings.map(m => m.mappingIndex));
function packets(pulse) {
  const counts = Array(48).fill(0), ratesHz = Array(48).fill(0);
  const result = [{initialized: true, fromTimeMs: null, timeMs: 0, indices: indices.slice(), counts: counts.slice(), ratesHz: ratesHz.slice(), events: []}];
  for (let timeMs = 2; timeMs <= durationMs; timeMs += 2) {
    const events = pulse && timeMs === eventMs ? [{index: selectedIndex, timeMs: eventMs}] : [];
    if (events.length) counts[selectedUnit]++;
    result.push({initialized: false, fromTimeMs: timeMs - 2, timeMs, indices: indices.slice(), counts: counts.slice(), ratesHz: ratesHz.slice(), events});
  }
  return result;
}
const plan = {schemaVersion: 1, kind: 'restrained-wing-event-phase-plan', sourceHashes, artifactHashes,
  scope: 'Four matched pairs; eight serial restrained native runs. No brain, controller, gain search, force fit, or biological preferred-phase inference.',
  reducedModelHash: reference.modelHash, expectedDimensions: {nq: 13, nv: 12, njnt: 7, nu: 6, neq: 0},
  timing: {nativeMs: .05, wingMs: .2, muscleMs: 1, packetMs: 2, warmupMs, durationMs, eventMs, postEventMs: durationMs - eventMs, frequencyHz},
  phases, selected: {index: selectedIndex, rootId: selectedMapping.root_ids?.[0] ?? null, mappingIndex: selectedMapping.mappingIndex,
    musclePosition: selectedMuscle, controlIndex: selectedControl, target: selectedMapping.target, joint: selectedMapping.joint},
  rootPose, controlNames, historicalTrim: reference.trimControls, effectiveBackgroundControls: background,
  operatingPoint: {steeringForceReference: null, interpreter: 'All 27 default multipliers equal 1; canonical absolute-force basis.',
    background: 'Historical synthetic normalized trim forces are held for 23 steering channels and common power. Left b1 alone is zero during warm-up/baseline and replaced by its event/native muscle force in pulse arms. This changed background is not the measured hover trim.',
    mixedRoute: 'Only selected b1 force uses the event adapter/native muscle kernel. The other held background forces bypass neural and muscle dynamics deliberately. No additive b1 force offset, automatic centering, or ongoing compensation.',
    commonPower: background[24], selectedBaselineForce: 0},
  eventModel: {config, initial: 'Zero event kernels/counts and native activation/fatigue/force in all 28 rows.',
    muscleInput: {normalizedLength: 1, positiveShorteningVelocity: 0, Fmax: 1, energy: 1},
    integration: 'Advance body with previously available force; at each 1 ms right boundary apply the completed interval mean excitation to native muscles. Event at20ms seeds[20,21], first nonzero native force can be available at21ms.',
    packetDigests: {baseline: sha(JSON.stringify(packets(false))), pulse: sha(JSON.stringify(packets(true)))},
    eventCount: {baseline: 0, pulse: 1}, inactiveNativeGroups: 'Remain exactly zero; their held synthetic background is merged only at FlyBodyWings input.'},
  phaseConvention: 'Quantized initial offsets are set once before100ms warm-up, then the oscillator evolves continuously. Event/force boundary phases are the existing table clock. Last generated target phase is logged separately because FlyBodyWings advances phase after generating targets. Neither phase is measured wing angle.',
  restraint: 'Reset only free-root qpos/qvel before every50us native step; six wing joints remain dynamic. No equality constraint or measured restraint reaction is claimed; xfrc_applied/qfrc_applied stay zero. No free flight.',
  wrench: {extraction: 'Immediately after mj_step, cache qfrc_fluid[0:6], xmat/xpos of root and subtree_com before any refresh. Fworld=q[0:3]; MworldCOM=R*q[3:6]−(COM−rootPos)×Fworld; MbodyCOM=R^T*MworldCOM.',
    forceUnits: 'g cm/s^2', torqueUnits: 'g cm^2/s^2', torqueImpulseUnits: 'g cm^2/s', weight, momentScale: weight * lengthCm,
    limitation: 'Fluid wrench only; not actuator/restraint/total torque, angular acceleration, or generalized armature momentum.'},
  gates: ['Identical native+wing warm hashes within each phase pair.', 'Identical body prefix through21ms before the first newly available b1 force can affect a subsequent native step.',
    'Byte-identical selected excitation/kernel/activation/fatigue/force traces across all four pulse phases.',
    'Exactly1 selected event per pulse and0 per baseline, all other event/native rows silent.', 'No contacts, instability warnings, nonfinite states, or applied external forces.'],
  analysis: {paired: 'Subtract each phase-matched no-event wrench at each native sample. Report all four contrasts without selecting a preferred phase.',
    windowsMsAfterEvent: [[0, 20], [20, 80], [80, 120], [0, 120]],
    horizon: 'All integrals are finite-horizon contrasts; report final force and torque contrast because the native40ms deactivation tail may persist.'}};

await fs.mkdir(output, {recursive: true});
assert((await fs.realpath(output)).startsWith((await fs.realpath(reports)) + path.sep), 'Output symlink escapes reports');
async function writeSameOrNew(name, content) {
  const file = path.join(output, name);
  try { assert((await fs.readFile(file)).equals(content), 'Prepared artifact changed: ' + name); }
  catch (error) { if (error.code !== 'ENOENT') throw error; await fs.writeFile(file, content, {flag: 'wx'}); }
}
// A rejected duplicate run must not touch any existing result artifact.
const existing = await fs.readdir(output);
assert(!existing.some(name => name === 'result.json' || /^phase-.*\.json$/.test(name)), 'Use a directory without prior phase results');
await writeSameOrNew('plan.json', Buffer.from(JSON.stringify(plan, null, 2) + '\n'));
await writeSameOrNew('source.used.mjs', bytes['scripts/probe-wing-event-phase.mjs']);
await writeSameOrNew('excitation-source.used.js', bytes['web/flybody-wing-event-excitation.js']);
if (prepareOnly) {
  console.log(JSON.stringify({prepared: true, nativeExecution: false, plan: path.join(output, 'plan.json'), arms: 8}));
  process.exit(0);
}

async function verifyPins() {
  for (const [file, expected] of Object.entries({...sourceHashes, ...artifactHashes}))
    assert.equal(sha(await fs.readFile(path.join(root, file))), expected, 'Source/artifact changed: ' + file);
}
let stop = false;
process.on('SIGINT', () => { stop = true; }); process.on('SIGTERM', () => { stop = true; });
async function checkpoint() {
  await new Promise(resolve => setTimeout(resolve, 0));
  let marker = false; try { await fs.access(path.join(output, 'STOP')); marker = true; } catch (e) { if (e.code !== 'ENOENT') throw e; }
  assert(!stop && !marker, 'Operator stopped phase probe');
}
const result = {schemaVersion: 1, kind: 'restrained-wing-event-phase-result', sourceHashes, artifactHashes,
  planSha256: sha(await fs.readFile(path.join(output, 'plan.json'))), startedAt: new Date().toISOString(), nodeVersion: process.version,
  cases: [], pairs: [], completed: false, error: null};
async function save() { await fs.writeFile(path.join(output, 'result.tmp'), JSON.stringify(result, null, 2) + '\n'); await fs.rename(path.join(output, 'result.tmp'), path.join(output, 'result.json')); }
let mj, model, data;
try {
  await verifyPins(); await checkpoint();
  mj = await loadMujoco(); model = mj.MjModel.from_xml_string(reducedXml); data = new mj.MjData(model);
  const core = await createWasmCore({wasmBinary: bytes['packages/banc-runtime/dist/core.wasm']});
  assert.equal(typeof core._muscle_step, 'function');
  const dimensions = Object.fromEntries(['nq', 'nv', 'njnt', 'nu', 'neq'].map(key => [key, model[key]]));
  assert.deepEqual(dimensions, plan.expectedDimensions); result.dimensions = dimensions; result.nativeVersion = mj.mj_versionString();
  const h = model.opt.timestep, wingStride = 4, muscleStride = 20, packetStride = 40;
  assert.equal(h, .00005); assert.equal(h, metadata.timestep); assert.equal(model.jnt_type[0], 0);
  const rootBody = model.jnt_bodyid[0], names = ['left', 'right'].flatMap(side => ['yaw', 'roll', 'pitch'].map(axis => `wing_${axis}_${side}`));
  const id = (kind, name) => { const i = mj.mj_name2id(model, mj.mjtObj[kind].value, name); assert(i >= 0, name); return i; };
  const joints = names.map(name => { const i = id('mjOBJ_JOINT', name); return {...metadata.joints.find(j => j.name === name), id: i,
    qpos: model.jnt_qposadr[i], dof: model.jnt_dofadr[i], range: Array.from(model.jnt_range.slice(i * 2, i * 2 + 2))}; });
  const actuators = names.map(name => { const i = id('mjOBJ_ACTUATOR', name); return {name, id: i, range: Array.from(model.actuator_ctrlrange.slice(i * 2, i * 2 + 2))}; });
  const meta = {...metadata, joints, actuators};
  const steer = controls => Object.fromEntries(['left', 'right'].map((side, k) => [side,
    Object.fromEntries(muscleNames.map((name, i) => [name, controls[k * 12 + i]]))]));
  const forceZeros = () => {
    for (const array of [data.qfrc_applied, data.xfrc_applied]) for (const value of array) assert.equal(value, 0, 'Applied external force');
  };
  const retainRoot = () => { data.qpos.set(rootPose); data.qvel.fill(0, 0, 6); };
  const drive = (wings, controls) => wings.step(data.qpos, data.ctrl, controls[24], controls[24], steer(controls), wingStride * h);
  const bodyState = wings => ({time: data.time, qpos: Array.from(data.qpos), qvel: Array.from(data.qvel), act: Array.from(data.act),
    ctrl: Array.from(data.ctrl), qaccWarmstart: Array.from(data.qacc_warmstart), wingPhase: wings.phase, wingTarget: Array.from(wings.target),
    wingResiduals: wings.residuals.map(values => Array.from(values)), wingState: wings.controlState()});
  const sampleWrench = () => {
    // These all describe the same native forward evaluation. Do not refresh
    // kinematics using the now-integrated qpos before taking this copy.
    const generalized = Array.from(data.qfrc_fluid.slice(0, 6)), rotation = Array.from(data.xmat.slice(rootBody * 9, rootBody * 9 + 9));
    const origin = Array.from(data.xpos.slice(rootBody * 3, rootBody * 3 + 3)), com = Array.from(data.subtree_com.slice(rootBody * 3, rootBody * 3 + 3));
    const world = worldComWrench(generalized, rotation, origin, com);
    const bodyTorque = [0, 1, 2].map(i => rotation[i] * world[3] + rotation[i + 3] * world[4] + rotation[i + 6] * world[5]);
    assert([...world, ...bodyTorque].every(Number.isFinite)); return {world, bodyTorque};
  };
  function warm(phase) {
    mj.mj_resetData(model, data); retainRoot(); for (const joint of joints) data.qpos[joint.qpos] = joint.neutral;
    mj.mj_forward(model, data); const wings = new FlyBodyWings(meta); wings.phase = phase.initialWarmPhaseRad;
    for (let s = 0; s < warmupMs * 20; s++) { retainRoot(); if (s % wingStride === 0) drive(wings, background); mj.mj_step(model, data); forceZeros(); }
    retainRoot(); mj.mj_forward(model, data); return wings;
  }
  async function runCase(phase, pulse) {
    await checkpoint(); await verifyPins();
    const wings = warm(phase), initial = bodyState(wings), initialHash = sha(JSON.stringify(initial));
    const adapter = createWingEventExcitation({io, config}), muscles = new WasmMuscles(core, 28);
    const input = new Float32Array(28 * 5); for (let i = 0; i < 28; i++) input.set([0, 1, 0, 1, 1], i * 5);
    const controls = background.slice(), packetList = packets(pulse), bodyPrefix = createHash('sha256'), waveform = createHash('sha256');
    const row = {phase, pulse, initial, initialHash, events: pulse ? 1 : 0, samples: [], muscleSamples: [],
      nominalEventMs: eventMs, counterfactualPhaseInBaseline: !pulse, quadratureDiscrepancyMax: 0,
      firstForce: null, firstGeneratedTargetWithForce: null, peakForce: null, clippedWingStepFraction: 0, contacts: 0,
      completed: false, sourcePinsVerified: false};
    let lastTargetPhase = wrap(wings.phase - TAU * frequencyHz * h * wingStride), clipped = 0, packetContext = null;
    const phaseAtZero = wings.phase;
    adapter.accept(packetList[0]);
    try {
      for (let s = 0; s < durationMs * 20; s++) {
        if (s % 400 === 0) await checkpoint();
        const startMs = s / 20;
        if (s % packetStride === 0) {
          const packet = packetList[s / packetStride + 1]; adapter.accept(packet);
          if (packet.timeMs === eventMs) packetContext = {fromTimeMs: packet.fromTimeMs, phaseRad: wings.phase,
            eventClockPhaseRad: wrap(wings.phase + TAU * frequencyHz * (eventMs - packet.fromTimeMs) / 1000)};
        }
        retainRoot();
        if (s % wingStride === 0) {
          lastTargetPhase = wings.phase; drive(wings, controls);
          if (controls[selectedControl] > 0 && row.firstGeneratedTargetWithForce === null)
            row.firstGeneratedTargetWithForce = {timeMs: startMs, tablePhaseRad: lastTargetPhase, force: controls[selectedControl]};
        }
        const heldForce = controls[selectedControl];
        mj.mj_step(model, data); const wrench = sampleWrench(); // cache before any refresh
        forceZeros(); assert(data.qpos.every(Number.isFinite) && data.qvel.every(Number.isFinite));
        const endMs = (s + 1) / 20;
        row.contacts += Number(data.ncon > 0);
        clipped += Number(actuators.some(a => Math.abs(data.ctrl[a.id] - a.range[0]) < 1e-10 || Math.abs(data.ctrl[a.id] - a.range[1]) < 1e-10));
        row.samples.push({fromTimeMs: startMs, timeMs: endMs, ...wrench, heldSelectedForce: heldForce, lastGeneratedTargetPhaseRad: lastTargetPhase});
        if (endMs <= eventMs + 1) bodyPrefix.update(JSON.stringify(bodyState(wings)) + '\n');
        if (endMs === eventMs) {
          row.eventContext = {packetContext, clockPhaseRad: wings.phase, clockPhaseDegrees: degrees(wings.phase),
            lastGeneratedTargetPhaseRad: lastTargetPhase, wingQpos: joints.map(j => data.qpos[j.qpos]), wingQvel: joints.map(j => data.qvel[j.dof])};
          assert(Math.abs(Math.atan2(Math.sin(wings.phase - packetContext.eventClockPhaseRad), Math.cos(wings.phase - packetContext.eventClockPhaseRad))) < 1e-10);
        }
        if ((s + 1) % muscleStride === 0) {
          const interval = adapter.finishInterval(endMs);
          row.quadratureDiscrepancyMax = Math.max(row.quadratureDiscrepancyMax, interval.quadratureDiscrepancy);
          for (let i = 0; i < 28; i++) { input[i * 5] = interval.excitation[i]; if (i !== selectedMuscle) assert.equal(input[i * 5], 0); }
          const state = muscles.step(input, .001), i = selectedMuscle * 3;
          for (let m = 0; m < 28; m++) if (m !== selectedMuscle) for (let k = 0; k < 3; k++) assert.equal(state[m * 3 + k], 0);
          const selectedForce = state[i + 2], clockPhaseRad = wings.phase;
          controls[selectedControl] = selectedForce;
          const ms = {timeMs: endMs, meanKernel: interval.unitMeanKernel[selectedUnit], excitation: interval.excitation[selectedMuscle],
            nativeExcitation: input[selectedMuscle * 5], activation: state[i], fatigue: state[i + 1], force: selectedForce,
            clockPhaseRad, lastGeneratedTargetPhaseRad: lastTargetPhase};
          assert([ms.meanKernel, ms.excitation, ms.nativeExcitation, ms.activation, ms.fatigue, ms.force].every(Number.isFinite));
          row.muscleSamples.push(ms);
          const binary = Float64Array.from([endMs, ms.meanKernel, ms.excitation, ms.nativeExcitation, ms.activation, ms.fatigue, ms.force]);
          waveform.update(Buffer.from(binary.buffer));
          if (selectedForce > 0 && row.firstForce === null) row.firstForce = {timeMs: endMs, clockPhaseRad, force: selectedForce};
          if (!row.peakForce || selectedForce > row.peakForce.force) row.peakForce = {timeMs: endMs, clockPhaseRad, force: selectedForce};
        }
      }
      assert.equal(adapter.readState().counts[selectedUnit], pulse ? 1 : 0);
      assert.equal(adapter.readState().counts.reduce((a, b) => a + b, 0), pulse ? 1 : 0);
      row.nativeSteps = durationMs * 20; row.restraintWrites = warmupMs * 20 + durationMs * 20 + 2;
      row.phaseAtZeroRad = phaseAtZero; row.finalState = bodyState(wings); row.finalEventState = adapter.snapshot();
      row.finalSelectedForce = row.muscleSamples.at(-1).force; row.finalSelectedFatigue = row.muscleSamples.at(-1).fatigue;
      row.bodyPrefixThrough21msSha256 = bodyPrefix.digest('hex'); row.selectedWaveformSha256 = waveform.digest('hex');
      row.clippedWingStepFraction = clipped / row.nativeSteps; row.warnings = readInstabilityWarnings(data.warning, mj.mjtWarning);
      assert.equal(row.contacts, 0, 'Contact in restrained phase assay'); assert.equal(row.warnings.length, 0, 'Native instability');
      if (pulse) { assert.equal(row.firstForce?.timeMs, 21); assert.equal(row.firstGeneratedTargetWithForce?.timeMs, 21); }
      else assert(row.muscleSamples.every(s => s.force === 0));
      await verifyPins(); row.sourcePinsVerified = true; row.completed = true; return row;
    } finally { muscles.dispose(); }
  }
  function comparePair(baseline, pulse) {
    assert.equal(baseline.initialHash, pulse.initialHash, 'Matched warm state differs');
    assert.equal(baseline.bodyPrefixThrough21msSha256, pulse.bodyPrefixThrough21msSha256, 'Pre-effect body prefix differs');
    assert.equal(baseline.eventContext.clockPhaseRad, pulse.eventContext.clockPhaseRad);
    const contrasts = pulse.samples.map((s, k) => {
      const b = baseline.samples[k]; assert.equal(s.timeMs, b.timeMs);
      return {fromTimeMs: s.fromTimeMs, timeMs: s.timeMs,
        world: s.world.map((v, i) => v - b.world[i]), bodyTorque: s.bodyTorque.map((v, i) => v - b.bodyTorque[i])};
    });
    const windows = plan.analysis.windowsMsAfterEvent.map(([from, to]) => {
      const rows = contrasts.filter(s => s.fromTimeMs >= eventMs + from && s.timeMs <= eventMs + to);
      assert.equal(rows.length, (to - from) * 20);
      const worldImpulse = [0, 0, 0, 0, 0, 0], bodyTorqueImpulse = [0, 0, 0], maximumAbsBodyTorque = [0, 0, 0];
      for (const s of rows) {
        s.world.forEach((v, i) => { worldImpulse[i] += v * h; });
        s.bodyTorque.forEach((v, i) => { bodyTorqueImpulse[i] += v * h; maximumAbsBodyTorque[i] = Math.max(maximumAbsBodyTorque[i], Math.abs(v)); });
      }
      return {fromMsAfterEvent: from, toMsAfterEvent: to, samples: rows.length, worldImpulse, bodyTorqueImpulse,
        meanBodyTorque: bodyTorqueImpulse.map(v => v / ((to - from) / 1000)), maximumAbsBodyTorque,
        normalizedMeanBodyTorque: bodyTorqueImpulse.map(v => v / ((to - from) / 1000) / (weight * lengthCm))};
    });
    return {phase: pulse.phase, actualEventClockPhaseDegrees: pulse.eventContext.clockPhaseDegrees,
      warmStateIdentical: true, bodyPrefixThrough21msIdentical: true, firstForce: pulse.firstForce,
      firstGeneratedTargetWithForce: pulse.firstGeneratedTargetWithForce, peakForce: pulse.peakForce,
      finalSelectedForce: pulse.finalSelectedForce, finalBodyTorqueContrast: contrasts.at(-1).bodyTorque,
      finiteHorizonOnly: true, windows, contrasts};
  }
  await save(); let expectedWaveform = null;
  for (const phase of phases) {
    const pair = [];
    for (const pulse of [false, true]) {
      const row = await runCase(phase, pulse), file = `phase-${phase.requestedDegrees}-${pulse ? 'pulse' : 'baseline'}.json`;
      const content = Buffer.from(JSON.stringify(row) + '\n'); await fs.writeFile(path.join(output, file), content, {flag: 'wx'});
      result.cases.push({file, sha256: sha(content), pulse, phase, completed: row.completed, initialHash: row.initialHash,
        bodyPrefixThrough21msSha256: row.bodyPrefixThrough21msSha256, selectedWaveformSha256: row.selectedWaveformSha256,
        finalSelectedForce: row.finalSelectedForce, contacts: row.contacts, warnings: row.warnings});
      if (pulse) { expectedWaveform ??= row.selectedWaveformSha256; assert.equal(row.selectedWaveformSha256, expectedWaveform, 'Pulse waveform differs across phases'); }
      pair.push(row); await save(); console.log(JSON.stringify({completedArms: result.cases.length, total: 8, file}));
    }
    const contrast = comparePair(...pair), file = `phase-${phase.requestedDegrees}-contrast.json`;
    const content = Buffer.from(JSON.stringify(contrast) + '\n'); await fs.writeFile(path.join(output, file), content, {flag: 'wx'});
    const {contrasts, ...summary} = contrast; result.pairs.push({...summary, file, sha256: sha(content)}); await save();
  }
  await verifyPins(); result.sourceUnchanged = true; result.pulseWaveformsIdentical = true; result.completed = true; await save();
  console.log(JSON.stringify({completed: true, output, arms: result.cases.length, pairs: result.pairs.length}));
} catch (error) {
  result.error = String(error.stack || error); await save(); throw error;
} finally { data?.delete(); model?.delete(); }
