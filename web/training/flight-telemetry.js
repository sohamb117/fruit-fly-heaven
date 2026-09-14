// Observation only. Sample at preview cadence, never in the physics-step loop.
// Topology is cached weakly; native arrays and neural rates are read afresh.
const topology = new WeakMap();
const clamp = x => Math.max(0, Math.min(1, x));
const wingKinds = new Set(['asynchronous_wing', 'wing_steering_assumption']);
const wingColumns = Object.freeze(['mappingIndex', 'target', 'joint', 'kind', 'side', 'motorCount',
  'requestedRateHz', 'excitationInput', 'activationState', 'fatigueState', 'normalizedMuscleForce']);
const clawColumns = Object.freeze(['mappingIndex', 'target', 'joint', 'side', 'legSegment', 'motorCount',
  'requestedRateHz', 'excitationInput', 'activationState', 'fatigueState', 'normalizedMuscleForce',
  'nativeAdhesionControl', 'nativeAdhesionActuatorForce', 'nativeAdhesionGain']);
const jointColumns = Object.freeze(['name', 'actuatorId', 'targetAngleRad', 'actualAngleRad',
  'velocityRadPerSecond', 'control', 'nativeActuatorForce']);
const sideColumns = Object.freeze(['side', 'muscleDrive', 'requestedPower', 'deployment', 'deploymentRamp', 'effectivePower']);
export const FLIGHT_TELEMETRY_UNITS = Object.freeze({
  time: 's', rate: 'Hz', wingAngleAndPhase: 'rad', wingVelocity: 'rad/s', frequency: 'Hz',
  muscleExcitationActivationFatigueForceAndWingPower: 'normalized model quantities',
  wingControl: 'native position error control, rad for this model',
  wingActuatorForce: 'g cm^2/s^2 for the current unit-gear hinge actuators',
  clawControl: 'normalized native adhesion control', clawActuatorForceAndGain: 'g cm/s^2',
  rootQpos: '[world x,y,z in cm; root quaternion w,x,y,z]',
  rootQvel: '[world linear cm/s; root-local angular rad/s]',
  rootQacc: '[world generalized linear cm/s^2; root-local generalized angular rad/s^2]',
});
export const FLIGHT_TELEMETRY_SAMPLING = Object.freeze({
  rates: 'Means use explicit BANC indices in each prepared muscle mapping, never positions in a motor readback vector.',
  wingPhase: 'Stored phase is after the last wing-control advance; targets were generated before that advance.',
  nativeForces: 'Native actuator_force and qacc are the retained last forward-dynamics evaluation. No mj_forward is called. They need not coincide with the subsequently integrated qpos/qvel.',
  rootAcceleration: 'qacc is not a finite difference of velocity, especially with implicit damping.',
  adhesion: 'Claw actuator force is the scalar adhesion-actuator output, not measured vertical foot support or a contact normal force.',
  missingDrive: 'muscleDrive/requestedPower are null before the body has issued its first wing command.',
});

function requireIndex(value, name) {
  if (!Number.isInteger(value) || value < 0) throw new Error('Invalid flight telemetry index: ' + name);
  return value;
}
function mappingSide(mapping) {
  if (mapping.joint.endsWith('_left')) return 'left';
  if (mapping.joint.endsWith('_right')) return 'right';
  throw new Error('Wing/claw mapping has no explicit side: ' + mapping.joint);
}
function describeMapping(mapping, index) {
  if (!Array.isArray(mapping.indices) || !mapping.indices.length || typeof mapping.target !== 'string' || typeof mapping.joint !== 'string')
    throw new Error('Invalid prepared muscle mapping for telemetry');
  return {index, target: mapping.target, joint: mapping.joint, kind: mapping.kind, side: mappingSide(mapping),
    indices: mapping.indices.map((id, k) => requireIndex(id, `mapping ${index} neuron ${k}`))};
}
// Absence is allowed only for the explicit structurally reduced native model.
function reducedClawActuation(body) {
  const flag = body.metadata?.dynamics_variant;
  if (flag === undefined) return false;
  const m = body.model, metadata = body.metadata;
  if (!flag || Reflect.ownKeys(flag).length !== 2 || flag.schemaVersion !== 1 || flag.kind !== 'fixed-nonwing-flight-v1' ||
    metadata.initializeStance !== false || m.nq !== 13 || m.nv !== 12 || m.njnt !== 7 || m.nu !== 6 || m.neq !== 0 ||
    !Array.isArray(metadata.fixed_joint_poses) || metadata.fixed_joint_poses.length !== 44 ||
    !Array.isArray(metadata.fixed_muscle_inputs) || metadata.fixed_muscle_inputs.length !== 135 ||
    !(body.actuators instanceof Map) || body.actuators.size !== 6 ||
    [...body.actuators].some(([name, actuator]) => !/^wing_(yaw|roll|pitch)_(left|right)$/.test(name) ||
      actuator.name !== name || !Number.isInteger(actuator.id) || actuator.id < 0 || actuator.id >= 6) ||
    new Set([...body.actuators.values()].map(a => a.id)).size !== 6)
    throw new Error('Flight telemetry requires the strict fixed-nonwing-flight-v1 layout');
  return true;
}
function prepare(body) {
  const cached = topology.get(body), variant = body.metadata?.dynamics_variant;
  if (cached && cached.mappings === body.mappings && cached.wings === body.wings && cached.model === body.model && cached.variant === variant) return cached;
  const {model, mappings, wings, actuators} = body;
  if (!Array.isArray(mappings) || !wings || !(actuators instanceof Map) ||
    model.jnt_type[0] !== 0 || model.jnt_qposadr[0] !== 0 || model.jnt_dofadr[0] !== 0)
    throw new Error('Flight telemetry requires the prepared FlyBody free-root layout');
  const reduced = reducedClawActuation(body), wingMuscles = [], claws = [];
  mappings.forEach((mapping, index) => {
    if (wingKinds.has(mapping.kind)) wingMuscles.push(describeMapping(mapping, index));
    if (mapping.kind === 'claw_grip_assumption') {
      const row = describeMapping(mapping, index), match = /^adhere_claw_T([123])_(left|right)$/.exec(mapping.joint);
      if (!match || (!reduced && !actuators.has(mapping.joint)) || (reduced && actuators.has(mapping.joint))) throw new Error('Unresolved native claw actuator: ' + mapping.joint);
      row.segment = Number(match[1]); row.actuatorId = reduced ? null : requireIndex(actuators.get(mapping.joint).id, mapping.joint); claws.push(row);
    }
  });
  claws.sort((a, b) => (a.side === 'right') - (b.side === 'right') || a.segment - b.segment);
  if (claws.length !== 6 || new Set(claws.map(row => row.joint)).size !== 6)
    throw new Error('Flight telemetry requires six distinct mapped claw actuators');
  if (wings.joints.length !== 6 || wings.actuators.length !== 6)
    throw new Error('Flight telemetry requires six native wing joints and actuators');
  const wingJoints = wings.joints.map((joint, index) => {
    const actuator = wings.actuators[index], id = requireIndex(actuator.id, actuator.name);
    if (actuator.name !== joint.name || model.actuator_gear[id * 6] !== 1)
      throw new Error('Wing telemetry units require matching unit-gear hinge actuators');
    return {name: joint.name, qpos: requireIndex(joint.qpos, joint.name), dof: requireIndex(joint.dof, joint.name), actuatorId: id};
  });
  const value = {model, mappings, wings, wingMuscles, claws, wingJoints, reduced, variant}; topology.set(body, value); return value;
}

/** Compact JSON-safe native snapshot. Missing BANC-rate identities are an
 * integration error. Nonfinite runtime measurements are retained as null with
 * explicit invalidFields, allowing the diagnostic itself to remain writable.
 */
export function makeFlightTelemetry(body, ratesMap) {
  if (!(ratesMap instanceof Map)) throw new TypeError('Flight telemetry requires a Map keyed by BANC neuron index');
  const cached = prepare(body), data = body.data, wings = body.wings, invalidFields = [];
  const read = (value, field, absentAllowed = false) => {
    if (absentAllowed && value === undefined) return null;
    if (typeof value !== 'number' || !Number.isFinite(value)) { invalidFields.push(field); return null; }
    return value;
  };
  const vector = (source, start, count, field) => Array.from({length: count}, (_, k) => read(source[start + k], `${field}[${k}]`));
  const meanRate = row => {
    let sum = 0, valid = true;
    for (const index of row.indices) {
      if (!ratesMap.has(index)) throw new Error('Missing BANC motor rate for telemetry: ' + index);
      const value = read(ratesMap.get(index), `motorRate[${index}]`);
      if (value === null || value < 0) { if (value !== null) invalidFields.push(`motorRate[${index}]`); valid = false; }
      else sum += value;
    }
    return valid ? sum / row.indices.length : null;
  };
  const muscleValues = row => [meanRate(row), read(body.input[row.index * 5], `muscle[${row.index}].excitation`),
    read(body.muscleState[row.index * 3], `muscle[${row.index}].activation`),
    read(body.muscleState[row.index * 3 + 1], `muscle[${row.index}].fatigue`),
    read(body.muscleState[row.index * 3 + 2], `muscle[${row.index}].force`)];
  const wingRows = cached.wingMuscles.map(row => [row.index, row.target, row.joint, row.kind, row.side, row.indices.length, ...muscleValues(row)]);
  const clawRows = cached.claws.map(row => [row.index, row.target, row.joint, row.side, row.segment, row.indices.length,
    ...muscleValues(row), cached.reduced ? null : read(data.ctrl[row.actuatorId], `${row.joint}.ctrl`),
    cached.reduced ? null : read(data.actuator_force[row.actuatorId], `${row.joint}.actuatorForce`),
    cached.reduced ? null : read(body.model.actuator_gainprm[row.actuatorId * 10], `${row.joint}.gain`)]);
  const jointRows = cached.wingJoints.map((row, index) => [row.name, row.actuatorId,
    read(wings.target[index], `${row.name}.target`), read(data.qpos[row.qpos], `${row.name}.qpos`),
    read(data.qvel[row.dof], `${row.name}.qvel`), read(data.ctrl[row.actuatorId], `${row.name}.ctrl`),
    read(data.actuator_force[row.actuatorId], `${row.name}.actuatorForce`)]);
  const transfer = wings.powerTransfer;
  if (transfer && (transfer.schemaVersion !== 1 || transfer.profile !== 'activation-amplitude-v1' ||
    !Number.isFinite(transfer.activationGain) || transfer.activationGain <= 0))
    throw new Error('Unsupported wing power transfer in flight telemetry');
  const sides = ['left', 'right'].map((side, index) => {
    const drive = read(side === 'left' ? body.wingDriveLeft : body.wingDriveRight, `wing.${side}.drive`, true),
      deployment = read(wings.deployment[index], `wing.${side}.deployment`),
      threshold = read(wings.config.deployment_before_beating, 'wing.deploymentThreshold'),
      gain = read(wings.interpreter.powerGain, 'wing.powerGain');
    const ramp = deployment === null || threshold === null ? null : read(clamp((deployment - threshold) / (1 - threshold)), `wing.${side}.ramp`);
    return [side, drive, drive === null || gain === null ? null : read(transfer ? gain * clamp(drive * transfer.activationGain) : clamp(drive * gain), `wing.${side}.requestedPower`),
      deployment, ramp, read(wings.power[index], `wing.${side}.effectivePower`)];
  });
  const result = {schemaVersion: 1, timeSeconds: read(body.time, 'body.time'), nativeTimeSeconds: read(data.time, 'native.time'),
    units: FLIGHT_TELEMETRY_UNITS, sampling: FLIGHT_TELEMETRY_SAMPLING,
    wing: {phaseRad: read(wings.phase, 'wing.phase'), frequencyHz: read(wings.frequencyHz, 'wing.frequency'),
      controlHasStepped: body.wingDriveLeft !== undefined && body.wingDriveRight !== undefined,
      sides: {columns: sideColumns, rows: sides}},
    wingJoints: {columns: jointColumns, rows: jointRows}, wingMuscles: {columns: wingColumns, rows: wingRows},
    claws: {columns: clawColumns, rows: clawRows},
    root: {qpos: vector(data.qpos, 0, 7, 'root.qpos'), qvel: vector(data.qvel, 0, 6, 'root.qvel'),
      qacc: vector(data.qacc, 0, 6, 'root.qacc')}, finite: invalidFields.length === 0, invalidFields};
  if (cached.reduced) result.claws.nativeActuation = {available: false, reason: 'structurally-absent', bodyVariant: 'fixed-nonwing-flight-v1'};
  return result;
}
