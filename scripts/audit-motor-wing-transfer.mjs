// Deterministic system-identification assay; never starts the brain/coordinator,
// changes production assets, fits a controller, or submits training results.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {gzipSync} from 'node:zlib';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';
import {createWasmCore, WasmMuscles} from '../packages/banc-runtime/src/wasm.js';
import {FlyBodyPhysics} from '../web/flybody-physics.js';
import {Moments, worldComWrench, equalityRootForce, rateAt, groupRate, makeCases, makeFollowupCases, makeContinuousCases, readInstabilityWarnings} from './motor-wing-calibration-helpers.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(x => x.replace(/^--/, '').split('=')));
const pilot = args.pilot === 'true', suite = args.suite || 'baseline';
assert(['baseline', 'followup', 'continuous'].includes(suite));
const requestedCases = suite === 'continuous' ? makeContinuousCases() : suite === 'followup' ? makeFollowupCases() : makeCases(pilot);
const out = args.output || `reports/motor-wing-calibration/${pilot ? 'pilot' : suite}`;
const maxWallSeconds = Number(args['max-wall-seconds'] || 600);
assert(maxWallSeconds > 0 && maxWallSeconds <= 3600);
const sha = value => createHash('sha256').update(value).digest('hex');
const sourcePaths = ['models/flybody-mujoco.xml', 'models/flybody-mujoco.json', 'data/prepared/banc888/io.json',
  'web/flybody-physics.js', 'web/flybody-wings.js', 'web/flybody-stance.js', 'web/flybody-leg-actuation.js',
  'web/banc-proboscis.js', 'web/banc/embodiment.js', 'packages/banc-runtime/native/core.cpp',
  'web/motor-interface.js',
  'packages/banc-runtime/src/wasm.js', 'packages/banc-runtime/dist/core.js', 'packages/banc-runtime/dist/core.wasm',
  'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js', 'packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.wasm',
  'scripts/audit-motor-wing-transfer.mjs', 'scripts/motor-wing-calibration-helpers.mjs'];
const buffers = Object.fromEntries(await Promise.all(sourcePaths.map(async file => [file, await fs.readFile(file)])));
const sourceHashes = Object.fromEntries(sourcePaths.map(file => [file, sha(buffers[file])]));
const xml = String(buffers['models/flybody-mujoco.xml']);
const metadata = JSON.parse(buffers['models/flybody-mujoco.json']);
const io = JSON.parse(buffers['data/prepared/banc888/io.json']);
const [mj, core] = await Promise.all([loadMujoco(), createWasmCore()]);
const h = metadata.timestep, controlDt = 20 * h, weight = metadata.mass_g * 981;
const powerGroups = io.muscles.map((group, index) => ({...group, index})).filter(g => g.kind === 'asynchronous_wing');
const steeringGroups = io.muscles.map((group, index) => ({...group, index})).filter(g => g.kind === 'wing_steering_assumption');
const wingJoints = ['left', 'right'].flatMap(side => ['yaw', 'roll', 'pitch'].map(axis => metadata.joints.find(j => j.name === `wing_${axis}_${side}`)));
const wingActuators = wingJoints.map(j => metadata.actuators.find(a => a.name === j.name));
assert.equal(powerGroups.length, 4); assert.equal(wingJoints.length, 6);
const ground = '<geom name="transfer_floor" type="plane" size="100 100 .1" contype="1" conaffinity="1" condim="3" friction=".6 .005 .0001" solref=".002 1" solimp=".95 .99 .01"/>';
const weld = '<equality><weld name="transfer_weld" body1="thorax" body2="transfer_rig" relpose="0 0 0 1 0 0 0" solref=".0001 1" solimp=".999 .999 .001" /></equality>';
const sceneXml = context => context === 'flat_floor' ? xml.replace('<worldbody>', `<worldbody>${ground}`) : context === 'restrained' ?
  xml.replace('</worldbody>', '<body name="transfer_rig" mocap="true" pos="0 0 10" /></worldbody>').replace('<actuator>', `${weld}<actuator>`) : xml;
const report = {
  createdAt: new Date().toISOString(), command: process.argv.join(' '), pilot, suite, instrumentationVersion: 2, nativeVersion: mj.mj_versionString(), sourceHashes,
  scope: 'Current COM-corrected production motor adapter, actual compiled WASM muscles, native MuJoCo body. Synthetic motor signals isolate the body interface; no live BANC, optimizer or policy. Baseline preserves all parameters; candidates vary only their explicitly recorded cloned deployment field.',
  sourceRevision: metadata.source.revision,
  definitions: {
    time: 'seconds; native sampling at 50 microseconds; muscles at 1 ms; wing command at 200 microseconds',
    muscleChannels: 'Input excitation = clamp(mean requested motor rate / 80); actual WASM activation, fatigue and normalized force are recorded separately. Other motor channels are zero except explicitly named steering controls.',
    force: 'g cm / s^2; divide by metadata.mass_g * 981 for bodyweights; multiply by 1e-5 for newtons',
    torqueAndWork: 'Torque g cm^2 / s^2; work torque * pre-step joint velocity * dt, summed at every native step. Multiply either native torque or native work by 1e-7 for N m or joules respectively. Power g cm^2 / s^3.',
    fluidWrench: 'Entire articulated fly qfrc_fluid root entries. World force; root-local moment rotated using data.xmat, then translated to instantaneous whole-body subtree_com: M_com = R M_root - (COM - root) cross F. This is aerodynamic wrench, not total load or direct wing-joint torque.',
    samplingAlignment: 'Forces, qacc, xmat, xpos and subtree_com returned by mj_step refer to its evaluated pre-integration state. Mechanical work uses saved pre-step qvel. End-step qpos/qvel and time are sampled separately; target holds four native steps.',
    restraint: 'Only restrained cases add a native thorax-to-static-mocap weld with solref .0001 1, solimp .999 .999 .001. No qpos/velocity clamping or applied root force. Weld is finite-compliance; maximum translation/angular error is reported. All other joints remain dynamic.',
    restraintReaction: 'J^T f using only efc_type equality / efc_id transfer_weld rows, transformed to world COM reference. It includes the rig supporting gravity and wing reaction; it is not aerodynamic force. Other contacts/limits are excluded.',
    airborne: 'Only flat-floor cases, observed at production 1 ms contact-refresh cadence: first observed loss of all native environment proximity contacts after support; separately require 20 ms of contact-free observations and >= 0.02 cm root rise for qualified takeoff. Intervening sub-ms contacts can be missed. Initial contact-free air/rig is not takeoff.',
    inversion: 'First native step with root upZ < 0. This is a clear inversion event, not a full behavioral judgment.',
    ramp: 'Requested rate rises linearly from 0 to target over .15 s, then holds; step is immediate. This differs from intrinsic WASM activation filtering.',
    averaging: 'All, early (first .1 s), and late (last .1 s) statistics use every native sample. Late is a finite late window, not a claim of physiological or mathematical steady state; fatigue still evolves.',
    clipping: 'Fraction of native steps at each wing actuator control limit; target limiting and error are captured separately. Native torque is data.qfrc_actuator at each wing DoF.',
    trace: 'Low-rate rows every 10 ms. Selected restrained 12/80 Hz step cases additionally contain the first .1s every four native steps and the final 10 ms at native resolution. Low-rate traces alone cannot resolve the ~236 Hz wingbeat.',
    candidates: 'Followup varies cloned wing_actuation.deployment_tau_s only; continuous suite varies cloned deployment_power_span only. Zero-span defaults are repeated. Native gain/damping/muscle kinetics and target tables remain unchanged. These engineering limits are not calibrated biology.',
    acceleration: 'jointAcceleration/rootAcceleration are native forward-dynamics qacc evaluated before implicit damping integration. realizedJointAcceleration/realizedRootAcceleration are (post-step qvel - pre-step qvel)/dt, with root translation in world and root angular components in generalized root coordinates. They can differ substantially under implicit damping.',
    numericalCompletion: 'Expected native step count and elapsed simulation time are checked. BADQPOS/BADQVEL/BADQACC/BADCTRL counters are read after every native step; any warning terminates the case. This rejects native instability resets even if MuJoCo later produces finite state.'
  },
  units: {massG: metadata.mass_g, bodyweight: weight, frequencyHz: metadata.wing_actuation.frequency_hz},
  motorGroups: powerGroups.map(({index, joint, target, indices}) => ({index, joint, target, indices})),
  wingJointNames: wingJoints.map(j => j.name),
  scenes: Object.fromEntries(['restrained', 'air', 'flat_floor'].map(context => [context, {xmlSha256: sha(sceneXml(context)), additions: context === 'restrained' ? 'one massless mocap body and weld' : context === 'flat_floor' ? 'one native plane' : 'none'}])),
  cases: [], checks: {}, limitations: [
    'No waveform/controller parameters are fitted. These observations identify current-model behavior, not biological calibration or stable flight.',
    'Zero non-wing motor rates retain native passive mechanics and leg posture servos but remove active claw recruitment and takeoff coordination. Actual-fruit contact and recorded BANC replay remain separate required tests.',
    'Shared thoracic averaging is preserved: unilateral power muscle drive does not imply unilateral wing power.',
    'Mean aerodynamic moment can coexist with oscillating actuator work and body dynamics. Large instantaneous torque alone does not prove excessive biological gain.',
    'No conclusion about free-flight stability follows from the restrained rig; each free-body case is tested separately for 1.2 s.'
  ]
};
await fs.mkdir(out, {recursive: true});
// Prepare all scenes from exactly the same native flat-floor stance. The air
// and rig retain this articulated state after moving the root to z=10 cm.
const preparationModel = mj.MjModel.from_xml_string(sceneXml('flat_floor'));
const preparationBody = new FlyBodyPhysics(mj, preparationModel, metadata, io, n => new WasmMuscles(core, n), {surface: () => 0, odor: () => 0, foodAt: () => null});
preparationBody.place(0, 0, 0);
const initialState = Object.fromEntries(['qpos', 'qvel', 'ctrl', 'act'].map(key => [key, Array.from(preparationBody.data[key])]));
initialState.restPose = Array.from(preparationBody.restPose);
report.initialState = {sha256: sha(JSON.stringify(initialState)), preparation: 'Identical production initializeStance on the native flat floor; air/rig then translated to 10 cm with no remaining floor geom.'};
preparationBody.dispose(); preparationModel.delete();
// Independently verify the world/COM transformation against native force
// application at several orientations, including an oblique free-root frame.
const checkModel = mj.MjModel.from_xml_string(xml), checkData = new mj.MjData(checkModel), generalized = new mj.DoubleBuffer(checkModel.nv);
report.coordinateChecks = [];
for (const rawQuaternion of [[1, 0, 0, 0], [Math.SQRT1_2, 0, 0, Math.SQRT1_2], [.8, .1, .3, -.4]]) {
  const length = Math.hypot(...rawQuaternion), quaternion = rawQuaternion.map(x => x / length);
  checkData.qpos.set(initialState.qpos); checkData.qpos.set(quaternion, 3); mj.mj_forward(checkModel, checkData);
  const rootBody = checkModel.jnt_bodyid[0], com = Array.from(checkData.subtree_com.slice(rootBody * 3, rootBody * 3 + 3));
  const force = [1, -2, 3], moment = [.02, -.03, .01]; generalized.GetView().fill(0);
  mj.mj_applyFT(checkModel, checkData, force, moment, com, rootBody, generalized.GetView());
  const recovered = worldComWrench(generalized.GetView().slice(0, 6), Array.from(checkData.xmat.slice(rootBody * 9, rootBody * 9 + 9)),
    Array.from(checkData.xpos.slice(rootBody * 3, rootBody * 3 + 3)), com);
  const error = Math.max(...recovered.map((x, i) => Math.abs(x - [...force, ...moment][i])));
  assert(error < 1e-12, 'Native COM wrench coordinate check failed');
  report.coordinateChecks.push({quaternion, maximumAbsoluteError: error});
}
generalized.delete(); checkData.delete(); checkModel.delete();
const started = performance.now();
const norm = values => Math.hypot(...values);
const maxAbs = values => Math.max(...values.map(Math.abs));
const createStats = () => ({torque: new Moments(6), target: new Moments(6), angle: new Moments(6), trackingError: new Moments(6),
  jointVelocity: new Moments(6), jointAcceleration: new Moments(6), aerodynamicWrench: new Moments(6), restraintWrench: new Moments(6), rootAcceleration: new Moments(6),
  realizedJointAcceleration: new Moments(6), realizedRootAcceleration: new Moments(6),
  power: new Moments(2), deployment: new Moments(2), muscle: new Moments(12)});
const flatten = stats => Object.fromEntries(Object.entries(stats).map(([key, value]) => [key, value.json()]));

function runCase(spec) {
  const model = mj.MjModel.from_xml_string(sceneXml(spec.context));
  const caseMetadata = structuredClone(metadata);
  if (spec.deploymentTau !== undefined) caseMetadata.wing_actuation.deployment_tau_s = spec.deploymentTau;
  if (spec.deploymentPowerSpan !== undefined) caseMetadata.wing_actuation.deployment_power_span = spec.deploymentPowerSpan;
  assert.equal(mj.mj_isSparse(model), 0, 'Dense Jacobian required for exact equality-row projection');
  assert.equal(model.nq, 57); assert.equal(model.nv, 56);
  const rootBody = model.jnt_bodyid[0];
  for (let i = 0; i < 6; i++) {
    assert.equal(model.jnt_qposadr[wingJoints[i].id], wingJoints[i].qpos);
    assert.equal(model.jnt_dofadr[wingJoints[i].id], wingJoints[i].dof);
    assert.equal(model.actuator_gainprm[wingActuators[i].id * 10], 18);
  }
  const stats = createStats(), early = createStats(), late = createStats(), work = new Float64Array(6), positiveWork = new Float64Array(6), clipped = new Uint32Array(6);
  const earlyWork = new Float64Array(6), earlyPositiveWork = new Float64Array(6), lateWork = new Float64Array(6), latePositiveWork = new Float64Array(6);
  const earlyClipped = new Uint32Array(6), lateClipped = new Uint32Array(6);
  const traces = [], highResolution = [], rates = new Map(io.motor_neurons.map(n => [n.index, 0]));
  let body, observing = false, initialRoot, nativeSamples = 0, maximumRootTranslation = 0, maximumRootAngle = 0;
  let minUpZ = 1, maxOmega = 0, maxRootRise = 0, maxApplied = 0, firstInversion = null, firstNoContact = null, qualifiedTakeoff = null;
  let hadSupport = false, unsupportedSince = null, failure = null, targetRate = 0, firstDeployment = null, firstBeat = null;
  let warningVector;
  const observedWarnings = [], totalSteps = Math.round(spec.seconds / h), windowSteps = Math.round(.1 / h);
  const nativeStep = mj.mj_step;
  const adapter = Object.create(mj);
  let observation;
  adapter.mj_step = (m, d) => {
    if (!observing) return nativeStep(m, d);
    const oldWingVelocity = wingJoints.map(j => d.qvel[j.dof]);
    const oldRootVelocity = Array.from(d.qvel.slice(0, 6));
    nativeStep(m, d);
    nativeSamples++;
    const warnings = readInstabilityWarnings(warningVector, mj.mjtWarning);
    if (warnings.length) { observedWarnings.push({time:d.time,warnings}); throw new Error('Native MuJoCo instability warning'); }
    const torque = wingJoints.map(j => d.qfrc_actuator[j.dof]);
    const angle = wingJoints.map(j => d.qpos[j.qpos]), velocity = wingJoints.map(j => d.qvel[j.dof]);
    const target = Array.from(body.wings.target), rotation = Array.from(d.xmat.slice(rootBody * 9, rootBody * 9 + 9));
    const rootPosition = Array.from(d.xpos.slice(rootBody * 3, rootBody * 3 + 3)), com = Array.from(d.subtree_com.slice(rootBody * 3, rootBody * 3 + 3));
    const aero = worldComWrench(d.qfrc_fluid.slice(0, 6), rotation, rootPosition, com);
    const restraint = spec.context === 'restrained' ? worldComWrench(equalityRootForce(d, m.nv, 0), rotation, rootPosition, com) : [0, 0, 0, 0, 0, 0];
    const q = Array.from(d.qpos.slice(0, 7)), v = Array.from(d.qvel.slice(0, 6));
    const upZ = 1 - 2 * (q[4] ** 2 + q[5] ** 2), omega = norm(v.slice(3));
    const rootAngle = 2 * Math.acos(Math.min(1, Math.abs(q[3])));
    maximumRootTranslation = Math.max(maximumRootTranslation, norm(q.slice(0, 3).map((x, i) => x - initialRoot[i])));
    maximumRootAngle = Math.max(maximumRootAngle, rootAngle);
    minUpZ = Math.min(minUpZ, upZ); maxOmega = Math.max(maxOmega, omega); maxRootRise = Math.max(maxRootRise, q[2] - initialRoot[2]);
    if (upZ < 0 && firstInversion === null) firstInversion = d.time;
    if (body.wings.deployment[0] > .01 && firstDeployment === null) firstDeployment = d.time;
    if (body.wings.power[0] > .001 && firstBeat === null) firstBeat = d.time;
    maxApplied = Math.max(maxApplied, maxAbs(Array.from(d.qfrc_applied)), maxAbs(Array.from(d.xfrc_applied)));
    const muscle = powerGroups.flatMap(g => Array.from(body.muscleState.slice(g.index * 3, g.index * 3 + 3)));
    const values = {torque, angle, target, trackingError: target.map((x, i) => x - angle[i]), jointVelocity: velocity,
      jointAcceleration: wingJoints.map(j => d.qacc[j.dof]), aerodynamicWrench: aero, restraintWrench: restraint,
      rootAcceleration: Array.from(d.qacc.slice(0, 6)), power: Array.from(body.wings.power), deployment: Array.from(body.wings.deployment), muscle};
    values.realizedJointAcceleration = velocity.map((value,i)=>(value-oldWingVelocity[i])/h);
    values.realizedRootAcceleration = v.map((value,i)=>(value-oldRootVelocity[i])/h);
    const isEarly = nativeSamples <= windowSteps, isLate = nativeSamples > totalSteps - windowSteps;
    for (const [key, value] of Object.entries(values)) {
      stats[key].add(value); if (isEarly) early[key].add(value); if (isLate) late[key].add(value);
    }
    const controls = wingActuators.map(a => d.ctrl[a.id]);
    for (let i = 0; i < 6; i++) {
      const increment = torque[i] * oldWingVelocity[i] * h;
      work[i] += increment; positiveWork[i] += Math.max(0, increment);
      const clip = Math.abs(controls[i] - wingActuators[i].range[0]) < 1e-10 || Math.abs(controls[i] - wingActuators[i].range[1]) < 1e-10 ? 1 : 0;
      clipped[i] += clip;
      if (isEarly) { earlyWork[i] += increment; earlyPositiveWork[i] += Math.max(0, increment); earlyClipped[i] += clip; }
      if (isLate) { lateWork[i] += increment; latePositiveWork[i] += Math.max(0, increment); lateClipped[i] += clip; }
    }
    observation = {time: d.time, requestedRateHz: targetRate,
      groupRatesHz: powerGroups.map(g => groupRate(g, targetRate, spec.selection)), excitation: powerGroups.map(g => body.activation[g.index]),
      muscle, deployment: values.deployment, opening: Array.from(body.wings.opening), power: values.power,
      target, angle, velocity, acceleration: values.jointAcceleration, controls,
      realizedJointAcceleration: values.realizedJointAcceleration,
      actuatorForce: wingActuators.map(a => d.actuator_force[a.id]), torque, cumulativeWork: Array.from(work),
      aerodynamicWrench: aero, restraintWrench: restraint, root: q, rootVelocity: v, rootAcceleration: values.rootAcceleration,
      realizedRootAcceleration: values.realizedRootAcceleration,
      upZ, omega, energy: body.internal.energy};
    if (spec.context === 'restrained' && spec.waveform === 'step' && spec.selection === 'all' && [12, 80].includes(spec.rateHz) &&
      (d.time >= spec.seconds - .01 - 1e-10 || isEarly && nativeSamples % 4 === 0)) highResolution.push(observation);
  };
  const caseStart = performance.now();
  try {
    body = new FlyBodyPhysics(adapter, model, caseMetadata, io, n => new WasmMuscles(core, n), {surface: () => 0, odor: () => 0, foodAt: () => null});
    for (const key of ['qpos', 'qvel', 'ctrl', 'act']) body.data[key].set(initialState[key]);
    body.restPose = Float64Array.from(initialState.restPose);
    mj.mj_forward(model, body.data); body.refresh();
    if (spec.context !== 'flat_floor') { body.data.qpos[2] = 10; mj.mj_forward(model, body.data); body.refresh(); }
    if (spec.context === 'restrained') {
      body.data.mocap_pos.set(body.data.qpos.slice(0, 3)); body.data.mocap_quat.set(body.data.qpos.slice(3, 7));
      mj.mj_forward(model, body.data); body.refresh();
    }
    initialRoot = Array.from(body.data.qpos.slice(0, 7)); hadSupport = body.contactCount > 0;
    warningVector = body.data.warning;
    observing = true;
    for (let tick = 0; tick < Math.round(spec.seconds / controlDt); tick++) {
      if ((performance.now() - started) / 1000 > maxWallSeconds) throw new Error('Assay wall-time limit reached');
      targetRate = rateAt(tick * controlDt, spec.rateHz, spec.waveform);
      for (const group of powerGroups) for (const id of group.indices) rates.set(id, groupRate(group, targetRate, spec.selection));
      for (const group of steeringGroups) {
        const selected = spec.steering === 'b2_left' && group.target === 'b2_muscle' && group.joint.endsWith('left') ||
          spec.steering === 'b2_right' && group.target === 'b2_muscle' && group.joint.endsWith('right') || spec.steering === 'iii1_both' && group.target === 'iii1_muscle';
        if (selected) for (const id of group.indices) rates.set(id, 80);
      }
      body.step(rates, controlDt, {coupling: true, flight: true});
      if (spec.context === 'flat_floor') {
        if (body.contactCount > 0) { hadSupport = true; unsupportedSince = null; }
        else if (hadSupport) {
          if (firstNoContact === null) firstNoContact = body.time;
          unsupportedSince ??= body.time;
          if (qualifiedTakeoff === null && body.time - unsupportedSince >= .02 - 1e-10 && body.z - initialRoot[2] >= .02) qualifiedTakeoff = body.time;
        }
      }
      if (tick % 10 === 9 || tick === 0) traces.push({...observation, contacts: body.contactCount, environmentContacts: body.environmentContactCount});
    }
    assert.equal(nativeSamples,totalSteps,'Incorrect native sample count');
    assert(Math.abs(body.data.time-spec.seconds)<1e-8,'Native simulation time reset or incomplete integration');
  } catch (error) { failure = String(error.message || error); }
  const result = {...spec, id: [spec.context, spec.selection, spec.rateHz, spec.waveform, spec.steering, spec.deploymentTau === undefined ? undefined : `tau${spec.deploymentTau}`,spec.deploymentPowerSpan===undefined?undefined:`span${spec.deploymentPowerSpan}`].filter(x => x !== undefined).join('-'),
    metadataOverride: {...(spec.deploymentTau === undefined ? {} : {'wing_actuation.deployment_tau_s': spec.deploymentTau}),...(spec.deploymentPowerSpan===undefined?{}:{'wing_actuation.deployment_power_span':spec.deploymentPowerSpan})}, metadataSha256: sha(JSON.stringify(caseMetadata)),
    observedWarnings,
    complete: failure === null, failure, nativeSamples, measuredSeconds: body?.data.time ?? 0, wallSeconds: (performance.now() - caseStart) / 1000,
    initialRoot, final: observation, metrics: {firstDeployment, firstBeat, firstInversion, firstNoContact, qualifiedTakeoff,
      minimumUpZ: minUpZ, maximumAngularSpeed: maxOmega, maximumRootRiseCm: maxRootRise, maximumRootTranslationCm: maximumRootTranslation,
      maximumRootAngleRad: maximumRootAngle, maximumAppliedForce: maxApplied},
    work: {signed: Array.from(work), positive: Array.from(positiveWork), negative: Array.from(work, (x, i) => x - positiveWork[i])},
    earlyWork: {signed: Array.from(earlyWork), positive: Array.from(earlyPositiveWork)}, lateWork: {signed: Array.from(lateWork), positive: Array.from(latePositiveWork)},
    actuatorClippedFraction: Array.from(clipped, x => x / nativeSamples),
    earlyClippedFraction: Array.from(earlyClipped, x => x / early.torque.n), lateClippedFraction: Array.from(lateClipped, x => x / late.torque.n),
    all: flatten(stats), early: flatten(early), late: flatten(late)};
  body?.dispose(); model.delete();
  return {result, trace: {case: spec, samples: traces, highResolution}};
}

for (const spec of requestedCases) {
  const {result, trace} = runCase(spec);
  const compressed = gzipSync(JSON.stringify(trace));
  const filename = `${result.id}.json.gz`;
  await fs.writeFile(path.join(out, filename), compressed);
  result.trace = {file: filename, bytes: compressed.length, sha256: sha(compressed), samples: trace.samples.length, nativeBurstSamples: trace.highResolution.length};
  report.cases.push(result);
  console.log(JSON.stringify({id: result.id, complete: result.complete, wallSeconds: result.wallSeconds,
    latePower: result.late.power.mean, lateLiftBodyweights: result.late.aerodynamicWrench.mean[2] / weight,
    maxTorque: Math.max(maxAbs(result.all.torque.maximum),maxAbs(result.all.torque.minimum)), maxOmega: result.metrics.maximumAngularSpeed,
    inversion: result.metrics.firstInversion, rigTranslation: result.metrics.maximumRootTranslationCm, failure: result.failure}));
  await fs.writeFile(path.join(out, 'summary.json'), JSON.stringify(report) + '\n');
  if ((performance.now() - started) / 1000 > maxWallSeconds) break;
}
report.wallSeconds = (performance.now() - started) / 1000;
report.checks = {
  nativeCoordinateChecks: report.coordinateChecks.every(check => check.maximumAbsoluteError < 1e-12),
  allRequestedCasesRecorded: report.cases.length === requestedCases.length,
  sourceUnchanged: (await Promise.all(sourcePaths.map(async file => sha(await fs.readFile(file)) === sourceHashes[file]))).every(Boolean),
  noUserAppliedForce: report.cases.every(c => c.metrics.maximumAppliedForce === 0),
  completeNumericalRuns: report.cases.every(c => c.complete),
  completeNativeDurations: report.cases.every(c => c.nativeSamples===Math.round(c.seconds/h)&&Math.abs(c.measuredSeconds-c.seconds)<1e-8),
  noNativeInstabilityWarnings: report.cases.every(c => c.observedWarnings.length===0),
  restrainedRootWithinOneMicronAndOneMilliradian: report.cases.filter(c => c.context === 'restrained').every(c => c.metrics.maximumRootTranslationCm < .0001 && c.metrics.maximumRootAngleRad < .001)
};
report.passed = Object.values(report.checks).every(Boolean);
await fs.writeFile(path.join(out, 'summary.json'), JSON.stringify(report) + '\n');
const number = x => x === null ? '—' : Number(x).toPrecision(5);
const table = report.cases.filter(c => c.context === 'restrained' && c.selection === 'all' && c.waveform === 'step' && !c.steering).map(c =>
  `| ${c.rateHz} | ${c.deploymentTau ?? metadata.wing_actuation.deployment_tau_s} | ${c.deploymentPowerSpan??0} | ${number(c.final?.excitation[0])} | ${number(c.late.power.mean[0])} | ${number(c.metrics.firstBeat)} | ${number(c.late.aerodynamicWrench.mean[2] / weight)} | ${number(Math.max(...c.all.torque.minimum.map(Math.abs), ...c.all.torque.maximum.map(Math.abs)))} | ${number(Math.max(...c.actuatorClippedFraction))} |`).join('\n');
const freeTable = report.cases.filter(c => c.context !== 'restrained' && c.selection === 'all').map(c =>
  `| ${c.context} | ${c.waveform} | ${c.rateHz} | ${c.deploymentTau ?? metadata.wing_actuation.deployment_tau_s} | ${c.deploymentPowerSpan??0} | ${number(c.metrics.qualifiedTakeoff)} | ${number(c.metrics.firstInversion)} | ${number(c.metrics.minimumUpZ)} | ${number(c.metrics.maximumAngularSpeed)} |`).join('\n');
await fs.writeFile(path.join(out, 'README.md'), `# Current motor-to-wing transfer assay\n\nExecuted ${report.cases.length} deterministic cases in ${number(report.wallSeconds)} wall seconds. This is current-model identification, not biological validation or trained flight. Source hashes, every-case outcomes, trace hashes, units and definitions are in [summary.json](summary.json). Compressed traces preserve low-rate samples and selected native-rate bursts without full brain/body dumps.\n\n## Native restrained rig\n\nThe root is held by a finite-compliance MuJoCo weld; all other joints stay dynamic. Only equality rows belonging to this weld contribute to the reported restraint reaction. There is no pose reset or applied-force clamp. Aerodynamic torque is transferred from root-local coordinates to the instantaneous whole-body COM before averaging. See [MuJoCo force/constraint equations](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework) and [weld constraints](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld).\n\n| DLM/DVM Hz | Deployment tau s | Opening span | Excitation | Late effective power | First beat s | Late lift / weight | Peak abs torque | Max clipped fraction |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n${table}\n\n“Late” averages the final 0.1 s, while fatigue continues to change; it is not a biological steady state. Step and 150 ms input-ramp cases, DLM-only, DVM-only, left-only, b2 and III1 controls are included in the JSON.\n\n## Free-body cases\n\nEach full-run air/floor case lasts 1.2 s. Zero non-wing input retains passive mechanics and posture servos but excludes active foot adhesion and coordinated takeoff. Air cases start unsupported and cannot count as takeoff. Floor takeoff requires sustained contact loss plus rise; inversion is root upZ < 0.\n\n| Scene | Input | Hz | Deployment tau s | Opening span | Qualified takeoff s | Inversion s | Minimum upZ | Peak angular speed rad/s |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n${freeTable}\n\n## Instrumentation gates\n\n\`\`\`json\n${JSON.stringify(report.checks, null, 2)}\n\`\`\`\n\nNo outcome above establishes realistic flight, identifies a biological servo gain, or justifies changing neural excitability. Qualified takeoff can precede instability. Reproduce with \`node scripts/audit-motor-wing-transfer.mjs${pilot ? ' --pilot=true' : suite !== 'baseline' ? ' --suite='+suite : ''}\`; pure helper checks: \`node --test scripts/test-motor-wing-transfer.mjs\`.\n`);
console.log(JSON.stringify({passed: report.passed, checks: report.checks, cases: report.cases.length, wallSeconds: report.wallSeconds, output: out}));
if (!report.passed) process.exitCode = 1;
