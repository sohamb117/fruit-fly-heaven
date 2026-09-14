// Offline comparison only. Usage: node scripts/compare-live-wing-events.mjs baseline.json events.json output-directory
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {isDeepStrictEqual as equal} from 'node:util';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
assert.equal(argv.length, 3, 'Supply baseline file, event file and output directory');
const resolve = value => {const file = path.resolve(root, value); assert(file.startsWith(path.join(root, 'reports') + path.sep)); return file;};
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function read(file) {const bytes = await fs.readFile(file); return {file: path.relative(root, file), sha256: sha(bytes), value: JSON.parse(bytes)};}
const previous = await read(resolve('reports/flight-proximal-sensory-repair/evaluation/000-proximal-exclusions-4e7373d0-2236-4fc4-8fea-cc4331a79c6f.json'));
const baseline = await read(resolve(argv[0])), events = await read(resolve(argv[1])), output = resolve(argv[2]);
const records = [previous, baseline, events];
const ioBytes = await fs.readFile(path.join(root, 'data/prepared/banc888/io.json'));
const ioSha = sha(ioBytes), io = JSON.parse(ioBytes);
for (const record of records) {
  const r = record.value;
  assert(r.complete && !r.error && r.evaluation && !r.evaluation.cancelled, `Incomplete ${record.file}`);
  assert.equal(r.sourceHashes['/banc-data/io.json'], ioSha, 'IO identity changed');
  assert.equal(r.evaluation.backend, 'webgpu'); assert.equal(r.backendLabel, 'dawn-metal');
  assert(r.motorEvents?.length > 1 && r.physicsDigest?.sha256, 'Missing event capture or physical digest');
  assert.equal(r.motorEvents[0].timeMs, 0); assert(r.motorEvents[0].initialized);
  assert.equal(r.physicsDigest.rows, r.evaluation.steps + 1);
  assert.equal(r.motorEvents.length, r.evaluation.steps + 1);
  assert.equal(r.motorEvents.at(-1).timeMs, r.evaluation.steps * 2);
  assert(r.frames.length && r.frames[0].time === 0);
}
const p = previous.value, b = baseline.value, e = events.value;
const bundleRecords = await Promise.all(['baseline', 'event'].map(name => read(resolve(`reports/flight-event-live/${name}-bundle.json`))));
const [baseConfig, eventConfig] = bundleRecords.map(record => JSON.parse(record.value.configText));
for (let i = 0; i < 2; i++) {
  const bundle = bundleRecords[i].value, record = [b, e][i], config = [baseConfig, eventConfig][i];
  assert.equal(sha(bundle.configText), record.configHash); assert.equal(config.modelFingerprint, record.modelFingerprint);
  for (const [asset, contents] of Object.entries(bundle.assets)) assert.equal(sha(contents), config.assets[asset]);
}
const baseMeta = JSON.parse(bundleRecords[0].value.assets['/body-model/flybody-mujoco.json']);
const eventMeta = JSON.parse(bundleRecords[1].value.assets['/body-model/flybody-mujoco.json']);
const metadataWithoutEvent = structuredClone(eventMeta); delete metadataWithoutEvent.diagnosticVariant.wingEventExcitation;
const filteredConfig = config => Object.fromEntries(Object.entries(config).filter(([key]) => !['environmentVersion', 'modelFingerprint', 'assets', 'notes', 'wingEventExcitation'].includes(key)));
const changedAssetPaths = Object.keys(baseConfig.assets).filter(asset => baseConfig.assets[asset] !== eventConfig.assets[asset]);
const selectivity = {
  nativeXmlIdentical: bundleRecords[0].value.assets['/body-model/flybody-mujoco.xml'] === bundleRecords[1].value.assets['/body-model/flybody-mujoco.xml'],
  metadataDiffersOnlyByDeclaredEventPriors: equal(baseMeta, metadataWithoutEvent),
  eventPriorsMatchBetweenConfigAndMetadata: equal(eventConfig.wingEventExcitation, eventMeta.diagnosticVariant.wingEventExcitation),
  baselineHasNoEventOption: !Object.hasOwn(baseConfig, 'wingEventExcitation'),
  otherConfigurationIdentical: equal(filteredConfig(baseConfig), filteredConfig(eventConfig)),
  onlyChangedAssetIsMetadata: equal(changedAssetPaths, ['/body-model/flybody-mujoco.json']),
  changedAssetPaths,
  eventPriors: eventConfig.wingEventExcitation,
};
const behavioralMetrics = metrics => Object.fromEntries(Object.entries(metrics).filter(([key]) => !['wallSeconds', 'setupWallSeconds', 'executionWallSeconds'].includes(key)));
const behavior = record => Object.fromEntries(['return', 'success', 'terminated', 'truncated', 'reason', 'cancelled', 'simSeconds', 'steps', 'parameters', 'seed', 'stage', 'backend', 'bodyBackend', 'dtMs', 'bodyBlockMs'].map(key => [key, record.evaluation[key]]));
const finalFrame = record => record.frames.at(-1);
const oldBaselineGates = {
  parameters: equal(p.evaluation.parameters, b.evaluation.parameters),
  initialCondition: equal(p.evaluation.metrics.initialCondition, b.evaluation.metrics.initialCondition),
  initialObservation: equal(p.evaluation.metrics.initialObservation, b.evaluation.metrics.initialObservation),
  initialFrame: equal(p.frames[0], b.frames[0]),
  physicsDigest: equal(p.physicsDigest, b.physicsDigest),
  everyMotorEventPacket: equal(p.motorEvents, b.motorEvents),
  evaluationBehavior: equal(behavior(p), behavior(b)),
  behavioralMetrics: equal(behavioralMetrics(p.evaluation.metrics), behavioralMetrics(b.evaluation.metrics)),
  finalFrame: equal(finalFrame(p), finalFrame(b)),
};
const matchedArms = {
  assignment: ['seed', 'stage', 'durationSeconds', 'parameters'].every(key => equal(b.assignment[key], e.assignment[key])),
  initialCondition: equal(b.evaluation.metrics.initialCondition, e.evaluation.metrics.initialCondition),
  initialObservation: equal(b.evaluation.metrics.initialObservation, e.evaluation.metrics.initialObservation),
  nativeBackend: equal(b.nativeWebGPU, e.nativeWebGPU),
  sourceHashes: equal(b.sourceHashes, e.sourceHashes),
  bundleSelectivity: ['nativeXmlIdentical', 'metadataDiffersOnlyByDeclaredEventPriors', 'eventPriorsMatchBetweenConfigAndMetadata',
    'baselineHasNoEventOption', 'otherConfigurationIdentical', 'onlyChangedAssetIsMetadata'].every(key => selectivity[key]),
};
const sourceChanges = (a, z) => [...new Set([...Object.keys(a.sourceHashes), ...Object.keys(z.sourceHashes)])].sort()
  .filter(file => a.sourceHashes[file] !== z.sourceHashes[file]).map(file => ({file, before: a.sourceHashes[file] ?? null, after: z.sourceHashes[file] ?? null}));
const neuralPacket = packet => Object.fromEntries(['timeMs', 'indices', 'ratesHz', 'counts', 'events'].map(key => [key, packet[key]]));
let firstNeuralDifferenceMs = null, firstCountDifferenceMs = null, firstPhaseDifferenceMs = null;
for (let k = 0; k < Math.min(b.motorEvents.length, e.motorEvents.length); k++) {
  const a = b.motorEvents[k], z = e.motorEvents[k];
  if (firstNeuralDifferenceMs === null && !equal(neuralPacket(a), neuralPacket(z))) firstNeuralDifferenceMs = a.timeMs;
  if (firstCountDifferenceMs === null && !equal(a.counts, z.counts)) firstCountDifferenceMs = a.timeMs;
  if (firstPhaseDifferenceMs === null && a.wingPhaseRadians !== z.wingPhaseRadians) firstPhaseDifferenceMs = a.timeMs;
}
const wingMappings = io.muscles.flatMap((m, mappingIndex) => ['asynchronous_wing', 'wing_steering_assumption'].includes(m.kind) ? [{...m, mappingIndex}] : []);
assert.equal(wingMappings.length, 28);
const average = values => values.length ? values.reduce((a, z) => a + z, 0) / values.length : null;
function motorWindow(record, fromMs, toMs) {
  if (!(toMs > fromMs) || toMs > record.motorEvents.at(-1).timeMs) return null;
  const slots = new Map(record.motorEvents[0].indices.map((index, slot) => [index, slot]));
  const packets = record.motorEvents.filter(packet => packet.timeMs > fromMs && packet.timeMs <= toMs);
  const count = new Map();
  for (const packet of packets) for (const event of packet.events) if (event.timeMs > fromMs && event.timeMs <= toMs) count.set(event.index, (count.get(event.index) || 0) + 1);
  const rows = wingMappings.map(m => {
    const unitRates = m.indices.map(index => (count.get(index) || 0) * 1000 / (toMs - fromMs));
    const ema = packets.map(packet => average(m.indices.map(index => packet.ratesHz[slots.get(index)])));
    return {mappingIndex: m.mappingIndex, target: m.target, kind: m.kind, side: m.joint.endsWith('_left') ? 'left' : 'right',
      indices: m.indices, events: m.indices.reduce((sum, index) => sum + (count.get(index) || 0), 0),
      rawMeanRateHz: average(unitRates), rawUnitRateRangeHz: [Math.min(...unitRates), Math.max(...unitRates)],
      meanFilteredRateHz: average(ema), maximumFilteredRateHz: Math.max(...ema),
      rate80ClampEquivalentMean: average(ema.map(rate => Math.max(0, Math.min(1, rate / 80)))),
      rate80ClampEquivalentSaturatedFraction: average(ema.map(rate => Number(rate >= 80)))};
  });
  return {fromMs, toMs, durationMs: toMs - fromMs, packetSamples: packets.length,
    convention: 'Raw events in (from,to]; filtered rates sampled at 2ms block endpoints. Rate80 equivalents describe the old command clamp, not event-mode excitation or force.', rows};
}
function frameMechanics(record) {
  return record.frames.map(frame => {
    const m = frame.metrics.mechanics, table = m.wingMuscles;
    const rows = table.rows.map(row => Object.fromEntries(table.columns.map((key, i) => [key, row[i]])));
    return {timeSeconds: frame.time, observation: frame.metrics.observation, phase: frame.metrics.phase,
      diagnostics: frame.metrics.diagnostics, wing: m.wing, wingMuscles: rows, root: m.root,
      sampleLimit: 'Wall-clock preview sample; no time average or dense per-millisecond force inference.'};
  });
}
function summary(record) {
  const v = record.evaluation, m = v.metrics, terminalMs = record.motorEvents.at(-1).timeMs;
  return {name: record.assignment.name, seed: v.seed, configHash: record.configHash, modelFingerprint: record.modelFingerprint,
    timeSeconds: v.simSeconds, reason: v.reason, success: v.success, return: v.return,
    hasTakenOff: m.hasTakenOff, takeoffTime: m.takeoffTime, bestFlightSeconds: m.bestFlightSeconds,
    landingTime: m.landingTime, landingSeconds: m.landingSeconds, phase: m.phase, diagnostics: m.diagnostics,
    initialObservation: m.initialObservation, finalObservation: m.finalObservation,
    totalMotorEvents: record.motorEvents.reduce((sum, packet) => sum + packet.events.length, 0),
    timing: {wallSeconds: m.wallSeconds, setupWallSeconds: m.setupWallSeconds, executionWallSeconds: m.executionWallSeconds,
      executionSecondsPerSimulatedSecond: m.executionWallSeconds / v.simSeconds,
      simulatedSecondsPerExecutionSecond: v.simSeconds / m.executionWallSeconds,
      interpretation: 'One recorded episode; active execution excludes setup/checkpoint pauses. Different trajectory lengths are normalized, but this is not a repeated performance benchmark.'},
    frameTimes: record.frames.map(frame => frame.time), physicsDigest: record.physicsDigest,
    windows: {full: motorWindow(record, 0, terminalMs), startup: motorWindow(record, 0, Math.min(100, terminalMs)),
      fixed100to280: motorWindow(record, 100, 280),
      common: motorWindow(record, 100, Math.min(280, b.motorEvents.at(-1).timeMs, e.motorEvents.at(-1).timeMs))},
    sparseMechanics: frameMechanics(record)};
}
const rows = [b, e].map(summary);
const analysis = {schemaVersion: 1, kind: 'offline-live-wing-event-comparison',
  provenance: {inputs: records.map(({value, ...rest}) => rest), bundles: bundleRecords.map(({value, ...rest}) => rest), ioSha256: ioSha,
    analysisSourceSha256: sha(await fs.readFile(fileURLToPath(import.meta.url))), nativeWebGPU: b.nativeWebGPU},
  oldBaselineGates, oldBaselineParityPassed: Object.values(oldBaselineGates).every(Boolean), matchedArms, selectivity,
  baselineToEventComparable: Object.values(matchedArms).every(Boolean),
  previousToBaselineSourceChanges: sourceChanges(p, b), baselineToEventSourceChanges: sourceChanges(b, e),
  firstNeuralDifferenceMs, firstCountDifferenceMs, firstPhaseDifferenceMs, rows,
  limitations: ['One fixed seed; no optimizer or training update.',
    'The event arm changes the motor-to-muscle interface and then receives its own evolving body feedback; later neural differences are not fixed-input causal effects.',
    'The digest covers initialization and every completed 2ms body block, not every50us native step.',
    'Exact events/rates are dense; actual muscle excitation/activation/force are only available in sparse preview frames.',
    'Different frame schedules depend on wall time and cannot supply matched-window native force averages.',
    'No physics, neural or muscle simulation was executed by this comparison script.']};
await fs.mkdir(output, {recursive: true});
await fs.writeFile(path.join(output, 'comparison.json'), JSON.stringify(analysis, null, 2) + '\n', {flag: 'wx'});
const fmt = value => typeof value === 'number' ? value.toFixed(3) : String(value);
const table = rows.map(row => `| ${row.name} | ${fmt(row.timeSeconds)} | ${row.hasTakenOff ? 'yes' : 'no'} | ${fmt(row.bestFlightSeconds)} | ${fmt(row.return)} | ${row.reason} |`).join('\n');
const dlm = rows.map(row => {const window = row.windows.fixed100to280; const cells = window?.rows.filter(m => m.target === 'dorsal_longitudinal_muscle');
  return cells ? `${row.name}: left ${fmt(cells.find(m => m.side === 'left').rawMeanRateHz)} Hz, right ${fmt(cells.find(m => m.side === 'right').rawMeanRateHz)} Hz` : `${row.name}: (100,280] ms window unavailable`;}).join('; ');
const silentB1 = rows.map(row => `${row.name}: ${row.windows.full.rows.filter(m => m.target === 'b1_muscle').map(m => `${m.side} ${m.events} events`).join(', ')}`).join('; ');
const speedRatio = rows[1].timing.executionSecondsPerSimulatedSecond / rows[0].timing.executionSecondsPerSimulatedSecond;
await fs.writeFile(path.join(output, 'README.md'), `The saved corrected baseline ${analysis.oldBaselineParityPassed ? 'was reproduced exactly on all declared regression gates' : 'did not pass every declared regression gate'}. The opt-in event comparison ${analysis.baselineToEventComparable ? 'has matching assignments, initial conditions and recorded runtime/backend sources' : 'has a comparability mismatch that must be inspected before interpreting behavior'}. This is one actual BANC/native-body seed, with no training or optimizer update.\n\n` +
`| Arm | Terminal time (s) | Takeoff | Longest qualifying flight (s) | Return | Termination |\n|---|---:|---|---:|---:|---|\n${table}\n\n` +
`Regression gates: ${Object.entries(oldBaselineGates).map(([key, value]) => `${key}=${value}`).join(', ')}. The physics digest covers initialization plus completed 2 ms blocks (${b.physicsDigest.rows} rows of ${b.physicsDigest.valuesPerRow} Float64 values), including qpos/qvel, controls, muscle state and wing state; it does not cover every native 50 microsecond step. Preview schedules may differ with wall time, so only initial/final preview equality is a parity gate.\n\n` +
`Raw DLM recruitment over the same (100,280] ms window: ${dlm}. B1 full-run event counts: ${silentB1}. These are actual cumulative event counts, not filtered-rate estimates. Complete per-type bilateral counts, raw/filtered rates and old-clamp equivalents are in [comparison.json](comparison.json).\n\n` +
`First baseline/event neural packet difference: ${firstNeuralDifferenceMs === null ? 'none over the common recorded horizon' : firstNeuralDifferenceMs + ' ms'}; first cumulative-count difference: ${firstCountDifferenceMs === null ? 'none over the common recorded horizon' : firstCountDifferenceMs + ' ms'}. Once body feedback diverges, these are two live closed-loop trajectories, not a matched fixed-input motor replay.\n\n` +
`Source selectivity passed: both arms used the same runtime/backend sources and byte-identical native XML. Metadata differs only by the explicit event-prior diagnostic tag; the corresponding config option, version/fingerprint and explanatory note differ. Other configuration, neural graph/profile, sensory mapping and all 27 parameters remain identical. The baseline is the existing no-adhesion development benchmark with power gain 1.5 and tied steering gains 0.05, not a freshly learned or physiological trim.\n\n` +
`Active execution cost was ${fmt(rows[0].timing.executionSecondsPerSimulatedSecond)} seconds per simulated second for the baseline and ${fmt(rows[1].timing.executionSecondsPerSimulatedSecond)} for events (${fmt((speedRatio - 1) * 100)}% more in this pair). Raw execution times were ${fmt(rows[0].timing.executionWallSeconds)} and ${fmt(rows[1].timing.executionWallSeconds)} seconds. The shorter event run must not be called faster from raw duration alone. This single normalized comparison excludes setup and does not isolate implementation overhead from different body trajectories.\n\n` +
`Actual wing power, excitation, activation, fatigue and force are retained for every available preview in the JSON. Their sampling times are ${rows.map(row => `${row.name} [${row.frameTimes.map(fmt).join(', ')}] s`).join('; ')}. These sparse samples do not justify dense force curves or matched-window force means. The rate80-clamp equivalents are labeled command comparisons and are not the event arm's actual excitation.\n\n` +
`The initial state is native grounded stance with the same seeded disturbance. See the JSON for foot support, contacts, final COM/rotation state, takeoff/landing diagnostics and exact source/config hashes. Sustained flight and landing criteria remain unchanged. A short increase in flight time is not stable flight, biological validation or evidence of learning.\n`, {flag: 'wx'});
console.log(JSON.stringify({output: path.relative(root, output), oldBaselineParityPassed: analysis.oldBaselineParityPassed,
  oldBaselineGates, matchedArms, firstNeuralDifferenceMs, firstCountDifferenceMs,
  rows: rows.map(({name, timeSeconds, bestFlightSeconds, reason, return: value}) => ({name, timeSeconds, bestFlightSeconds, reason, return: value}))}));
