// Offline report verification/summary. No brain, muscle, or body modules load.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sha = value => createHash('sha256').update(value).digest('hex');
const read = async file => { const bytes = await fs.readFile(path.join(root, file)); return {value: JSON.parse(bytes), sha256: sha(bytes)}; };
const controlDir = 'reports/flight-event-control/run-001';
const phaseDir = 'reports/wing-event-phase';
const control = await read(`${controlDir}/result.json`);
const phase = await read(`${phaseDir}/result.json`);
const plan = await read(`${phaseDir}/plan.json`);
const c = control.value, p = phase.value;
assert(c.completed && c.sourceUnchanged && c.pairedInitialStateIdentical && c.historicalNativeGate.passed);
assert(p.completed && p.sourceUnchanged && p.pulseWaveformsIdentical && !p.error);
assert.equal(plan.sha256, p.planSha256);
assert.equal(sha(await fs.readFile(path.join(root, controlDir, 'diagnostic-source.used.mjs'))), c.sourceHashes['scripts/diagnose-flight-event-control.mjs']);
assert.equal(sha(await fs.readFile(path.join(root, phaseDir, 'source.used.mjs'))), p.sourceHashes['scripts/probe-wing-event-phase.mjs']);
assert.equal(sha(await fs.readFile(path.join(root, phaseDir, 'excitation-source.used.js'))), p.sourceHashes['web/flybody-wing-event-excitation.js']);
assert.equal(sha(await fs.readFile(path.join(root, controlDir, 'fixed-nonwing.xml'))), c.reference.modelSha256);
const nativeReference = await read(c.protocol.nativeReference.file);
assert.equal(nativeReference.sha256, c.protocol.nativeReference.sha256);
const oldNative = nativeReference.value.cases.find(row => row.mode === 'muscle_only');
const historical = c.cases.find(row => row.mode === 'historical_native');
for (const key of ['samples', 'final', 'metrics', 'outputMetrics']) assert.deepEqual(historical[key], oldNative[key]);
assert.equal(historical.samples.length, 750);
const liveSourceCheck = {};
for (const [file, expected] of Object.entries({...c.sourceHashes, ...p.sourceHashes})) {
  liveSourceCheck[file] = {recordedSha256: expected, currentSha256: sha(await fs.readFile(path.join(root, file)))};
  liveSourceCheck[file].matches = liveSourceCheck[file].currentSha256 === expected;
}
const tables = c.protocol.eventInverseTables;
for (const rows of Object.values(tables)) {
  assert.equal(rows.length, 12);
  for (let i = 1; i < rows.length; i++) { assert(rows[i].activation >= rows[i - 1].activation); assert.equal(rows[i].settleError, 0); }
}
const controlRows = c.cases.map(row => {
  assert.equal(row.nativeSteps, 30000); assert.equal(row.poseWritesAfterRelease, 0);
  assert.equal(row.maximumAppliedExternalForce, 0); assert.deepEqual(row.warnings, []);
  return {mode: row.mode, pass: row.meetsDeclaredPositiveControl, ...row.metrics, ...row.outputMetrics};
});
const phaseRows = [];
const phaseCaseMetrics = [];
for (const entry of p.cases) {
  const saved = await read(`${phaseDir}/${entry.file}`);
  assert.equal(saved.sha256, entry.sha256); assert(saved.value.completed);
  assert.equal(saved.value.contacts, 0); assert.deepEqual(saved.value.warnings, []);
  phaseCaseMetrics.push({file: entry.file, pulse: entry.pulse, events: saved.value.events,
    wingControlClippedStepFraction: saved.value.clippedWingStepFraction,
    quadratureDiscrepancyMaximum: saved.value.quadratureDiscrepancyMax});
}
for (const pair of p.pairs) {
  const saved = await read(`${phaseDir}/${pair.file}`);
  assert.equal(saved.sha256, pair.sha256);
  assert(pair.warmStateIdentical && pair.bodyPrefixThrough21msIdentical && pair.finiteHorizonOnly);
  assert.equal(pair.firstForce.timeMs, 21); assert.equal(pair.peakForce.timeMs, 30);
  const changed = saved.value.contrasts.find(row => row.world.some(v => v !== 0));
  assert(changed && changed.fromTimeMs >= 21);
  for (const row of saved.value.contrasts.filter(row => row.timeMs <= 21)) assert(row.world.every(v => v === 0));
  phaseRows.push({requestedDegrees: pair.phase.requestedDegrees, actualEventClockPhaseDegrees: pair.actualEventClockPhaseDegrees,
    quantizationErrorDegrees: pair.phase.nominalQuantizationErrorDegrees,
    firstForce: pair.firstForce, firstChangedFluidWrenchInterval: [changed.fromTimeMs, changed.timeMs], peakForce: pair.peakForce,
    finalSelectedForce: pair.finalSelectedForce, finalBodyTorqueContrast: pair.finalBodyTorqueContrast,
    windows: pair.windows});
}
const variation = [0, 1, 2, 3].map(windowIndex => ({windowMs: [p.pairs[0].windows[windowIndex].fromMsAfterEvent, p.pairs[0].windows[windowIndex].toMsAfterEvent],
  axes: ['roll', 'pitch', 'yaw'].map((axis, index) => {
    const values = p.pairs.map(pair => pair.windows[windowIndex].bodyTorqueImpulse[index]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return {axis, min: Math.min(...values), max: Math.max(...values), mean,
      rangeOverAbsoluteMean: (Math.max(...values) - Math.min(...values)) / Math.abs(mean),
      sameSignAtEveryPhase: values.every(v => Math.sign(v) === Math.sign(values[0]))};
  })}));
const provenance = {analysisSourceSha256: sha(await fs.readFile(fileURLToPath(import.meta.url))),
  controlResultSha256: control.sha256, phaseResultSha256: phase.sha256, phasePlanSha256: plan.sha256,
  nativeVersion: c.nativeVersion, reducedModelSha256: c.reference.modelSha256,
  sourceHashes: {...c.sourceHashes, ...p.sourceHashes}, liveSourceCheck};
const causalReview = {
  method: 'Read-only inspection of the archived diagnostic source; no additional simulation.',
  findings: [
    'At each 2 ms control boundary, body feedback is read from current qpos/qvel and COM caches. The controller computes one held-rate command.',
    'nextEvents uses only those current commands and the existing per-unit phase/count/refractory state. Future event timestamps contain no future body measurements.',
    'Causal arms integrate a physical 1 ms interval with the prior muscle force before consuming its event integral and updating the native muscle state.',
    'The next wing update uses that completed-interval force. Events at the right endpoint cannot contribute to the preceding excitation integral.',
    'The activation lookup is constructed without body integration or feedback, using the same 0.5 ms generator and staggered 5/7-unit power populations.',
    'The classical controller and inverse force/rate mapping are explicit diagnostic controllers. Neither is a BANC response or a proposed runtime event decoder.'
  ],
  futureBodyFeedbackLeakFound: false,
  limitation: 'This source review is not a bytewise all-native-step causality proof. Historical parity covers all 750 saved 2 ms samples, final state and metrics.'
};
await fs.writeFile(path.join(root, controlDir, 'analysis.json'), JSON.stringify({schemaVersion: 1, kind: 'offline-event-control-analysis', provenance, causalReview,
  rows: controlRows, lookup: {nonzeroRates: 11, measuredSecondsPerRate: 1, warmupSecondsPerRate: 3, maximumPeriodicActivationError: 0,
    activationMonotonicity: 'nondecreasing', tables}}, null, 2) + '\n');
await fs.writeFile(path.join(root, phaseDir, 'analysis.json'), JSON.stringify({schemaVersion: 1, kind: 'offline-wing-event-phase-analysis', provenance,
  rows: phaseRows, caseMetrics: phaseCaseMetrics, variation, gates: {allEightCasesComplete: true, allFourPairsMatched: true, allPulseWaveformsIdentical: true,
    noFluidWrenchDifferenceThrough21ms: true, noContactsOrWarnings: true}, limitations: [
    'Root is restrained; reported wrench is fluid torque at COM, not total torque, restraint reaction or angular acceleration.',
    'Phase is the table oscillator clock, not measured wing angle; quantized event phases are not exact quadrants.',
    'Only left b1 is event/native driven. Other channels hold synthetic forces; left b1 background is zero, changing the historical hover trim.',
    'Native force is still nonzero at 120 ms after the event. Impulses are finite-horizon values.',
    'One muscle, four phases and one operating point establish neither biological preferred phases nor general flight controllability.'
  ]}, null, 2) + '\n');

const fmt = (value, digits = 3) => value.toFixed(digits);
const controlTable = controlRows.map(row => `| ${row.mode} | ${fmt(row.maximumTiltDegrees)} | ${fmt(row.finalFilteredAngularSpeed)} | ${fmt(row.COMriseCm)} | ${fmt(100 * row.wingControlClippedStepFraction, 2)}% | ${row.pass ? 'pass' : 'fail'} |`).join('\n');
await fs.writeFile(path.join(root, controlDir, 'README.md'), `The frozen classical feedback controller kept the event-driven native body within 7.20 degrees of upright for the full 1.5 second free-flight assay. The matching fixed-trim event arm reached 43.52 degrees. This is a mechanical positive control using synthetic events and prepared flight conditions; it does not demonstrate BANC flight, takeoff, landing, or biological calibration.\n\n` +
`| Arm | Maximum tilt (deg) | Final filtered angular speed (rad/s) | COM rise (cm) | Wing control clipped steps | Declared result |\n|---|---:|---:|---:|---:|---|\n${controlTable}\n\n` +
`All arms ran 30,000 native steps (50 microseconds each), with no contacts, instability warnings, applied root forces, or root pose writes after release. The historical native arm exactly matched the earlier latency assay's 750 saved 2 ms samples, final state and metrics. This is not a claim of hashing every native step.\n\n` +
`The causal native arm changes the muscle update to the completed 1 ms interval boundary and performs similarly to the historical update. The event arm passes through 48 quantized synthetic unit event streams, per-unit kernels, 28 mapped excitations and the unchanged native activation/fatigue/force kernel. Its maximum force tracking error is ${fmt(controlRows[2].muscleForceMaximumTrackingError)} and RMS error ${fmt(controlRows[2].muscleForceRmsTrackingError)}; ${fmt(controlRows[2].eventRateClippedCommandFraction * 100, 2)}% of event-rate commands clip to the declared maximum. The approximately 18.8% wing-control clipping in the successful arms remains a material feature of this operating point.\n\n` +
`The rate inverse uses actual native mean activation, not mean excitation. Eleven nonzero rates from 2 to 400 Hz each receive 3 seconds of muscle warm-up and a 1 second measurement, with the same quantized event generator and within-muscle unit staggering used during release. Every activation waveform matches its 1-second-earlier waveform exactly at Float32 precision. Means are nondecreasing; high-rate DLM/DVM values plateau. Fatigue evolves in this measurement, but only activation is inverted; the current-fatigue force inverse is separate. This lookup is a diagnostic feedforward map, not reward fitting or a neural component.\n\n` +
`Timing review found no future body-feedback use: current body measurements produce held commands at each 2 ms boundary, packet events depend only on that command and previous generator state, and the event/native state advances after each completed 1 ms physical interval. Prior force drives the interval. A packet may contain its full upcoming 2 ms event schedule, but the adapter consumes only events at or before the completed boundary.\n\n` +
`All arms share a 100 ms restrained physical warm-up and prepared native trim activation with zero fatigue. Event arms additionally prepare 2 seconds of kernel history without evolving native muscles. Thus this result does not test cold onset. The existing classical controller has access to body orientation, angular velocity, height and vertical velocity. Its gains and physical allocation Jacobian are frozen from the earlier diagnostic; BANC is not executed. Horizontal position is uncontrolled (event feedback drift ${fmt(controlRows[2].maximumHorizontalDriftCm)} cm). The permissive COM criterion is absolute rise below 10 cm; this is not precise hover.\n\n` +
`Results: [result.json](result.json), [offline analysis](analysis.json), [executed source](diagnostic-source.used.mjs). MuJoCo ${c.nativeVersion}; result SHA256 ${control.sha256}; executed source SHA256 ${c.sourceHashes['scripts/diagnose-flight-event-control.mjs']}; reduced XML SHA256 ${c.reference.modelSha256}; native muscle WASM SHA256 ${c.sourceHashes['packages/banc-runtime/dist/core.wasm']}. Full dependency hashes are in both JSON files. Recreate only this offline summary with \`node scripts/analyze-flight-event-controls.mjs\`.\n`);
const phaseTable = phaseRows.map(row => {
  const short = row.windows[0].bodyTorqueImpulse.map(v => fmt(v * 1e6, 4));
  const total = row.windows[3].bodyTorqueImpulse.map(v => fmt(v * 1e6, 4));
  return `| ${fmt(row.actualEventClockPhaseDegrees, 2)} | ${short.join(' / ')} | ${total.join(' / ')} |`;
}).join('\n');
const spread = index => variation[index].axes.map(row => `${row.axis} ${fmt(100 * row.rangeOverAbsoluteMean, 2)}%`).join(', ');
await fs.writeFile(path.join(root, phaseDir, 'README.md'), `A single left-b1 event produces different early fluid-torque impulses at four oscillator phases, while its full native muscle waveform is identical. All four 120 ms net impulses retain the same roll-positive, pitch-negative, yaw-positive directions at this operating point. Phase changes the transient; this assay does not identify a biological preferred phase or show phase alone reversing a steering command.\n\n` +
`| Actual event clock phase (deg) | 0–20 ms impulse: roll / pitch / yaw | 0–120 ms impulse: roll / pitch / yaw |\n|---:|---:|---:|\n${phaseTable}\n\n` +
`Impulse units in the table are 10^-6 g cm^2/s. Across phases, the range divided by absolute mean is ${spread(0)} in the first 20 ms, and ${spread(3)} over 120 ms. These are four deterministic probes, not statistical confidence intervals.\n\n` +
`All eight arms completed. Each pulse has its own phase-matched no-event baseline; initial selected native/wing state hashes and body prefixes through 21 ms agree within every pair. The four selected kernel/excitation/activation/fatigue/force waveforms have the same binary Float64 hash. These finite body-state hashes use JSON serialization of the explicitly listed fields, not the entire native integration state. All recorded fluid-wrench differences are zero through 21 ms, with no contacts or instability warnings.\n\n` +
`The event occurs at 20 ms. First nonzero native force is ${fmt(phaseRows[0].firstForce.force, 6)} at 21 ms, and peak force ${fmt(phaseRows[0].peakForce.force, 6)} is at 30 ms. First fluid-wrench contrast appears in ${phaseRows.map(row => row.firstChangedFluidWrenchInterval.join('–') + ' ms').join(', ')} for the table rows. Force remains ${fmt(phaseRows[0].finalSelectedForce, 6)} at the 140 ms endpoint (${fmt(100 * phaseRows[0].finalSelectedForce / phaseRows[0].peakForce.force, 2)}% of peak), so these are finite-window impulses rather than complete impulse responses.\n\n` +
`The root is explicitly restrained at level pose; six wing joints remain dynamic. Only left b1 MN75865 uses the event/native muscle route, from zero initial kernel and muscle states. Its baseline force is zero; the other 23 steering channels and common power hold historical synthetic trim forces. This differs from the historical hover operating point. Wing servo controls clip during 18.86–19.29% of native steps, with the same fraction within each matched pair; the measured transfer includes that nonlinear actuation. No brain, feedback controller, gain fitting, free flight, or root angular response is measured.\n\n` +
`The 235.813447 Hz table clock runs continuously after its pre-warm initialization. Quantized 0.5 ms starting offsets yield actual event phases shown above, with target-quadrant errors from -17.25 to +20.09 degrees. One millisecond of causal force delay spans about 84.89 clock degrees; event phase, first force phase, last generated target phase, and peak-force phase are recorded separately. None is asserted to equal measured wing angle.\n\n` +
`Wrench extraction uses the native fluid-force cache immediately after each 50 microsecond step, rotating local root torque to world axes and shifting to the matching COM, then rotating torque into body axes. It excludes actuator torque and the unmeasured restraint reaction. The result is not total torque or angular acceleration.\n\n` +
`Results: [result.json](result.json), [offline analysis](analysis.json), [predeclared plan](plan.json), [executed source](source.used.mjs). MuJoCo ${p.nativeVersion}; result SHA256 ${phase.sha256}; plan SHA256 ${plan.sha256}; executed source SHA256 ${p.sourceHashes['scripts/probe-wing-event-phase.mjs']}; reduced XML SHA256 ${plan.value.reducedModelHash}. Full source and per-arm hashes are in the JSON files. Recreate only this offline summary with \`node scripts/analyze-flight-event-controls.mjs\`.\n`);
console.log(JSON.stringify({offlineOnly: true, controls: controlRows.map(({mode, pass}) => ({mode, pass})), phaseCases: p.cases.length,
  phaseWindows: variation, currentPinnedSourcesMatch: Object.values(liveSourceCheck).every(row => row.matches)}));
