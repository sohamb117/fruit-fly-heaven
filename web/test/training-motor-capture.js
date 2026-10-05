// Development-only, read-only capture of held BANC motor output at the actual
// 2 ms native feedback boundary. This module is never imported by production.
import {flybodyScene} from '../flybody-physics.js';

const clone = value => structuredClone(value);
const integrationFields = Object.freeze(['qpos','qvel','act','history','qacc_warmstart','ctrl','qfrc_applied',
  'xfrc_applied','mocap_pos','mocap_quat','userdata','plugin_state']);
const array = (value, name) => {
  if (!value || typeof value.length !== 'number') throw new Error('Missing native capture array: ' + name);
  const result = Array.from(value);
  if (!result.every(Number.isFinite)) throw new Error('Nonfinite native capture array: ' + name);
  return result;
};
const requireFinite = (value, name) => {
  if (!Number.isFinite(value)) throw new Error('Nonfinite native capture value: ' + name);
  return value;
};

// DataView fixes byte order rather than relying on the browser's native order.
export function packCaptureArray(values, bits = 64) {
  if (bits !== 32 && bits !== 64) throw new Error('Unsupported capture precision');
  const bytes = new Uint8Array(values.length * (bits / 8)), view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) {
    const value = requireFinite(values[i], 'packed[' + i + ']');
    if (bits === 32) {
      if (!Object.is(Math.fround(value), value)) throw new Error('Capture would lose float32 precision');
      view.setFloat32(i * 4, value, true);
    } else view.setFloat64(i * 8, value, true);
  }
  let text = '';
  for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
  return btoa(text);
}

export function captureInitialMotorState(body) {
  const {data, muscles, wings} = body;
  // MuJoCo's current boolean eq_active getter is not bound in Embind. Its
  // complete integration-state API includes those flags without touching the
  // simulator. This is a read-only copy into recorder-owned scratch storage.
  const spec = body.mj.mjtState.mjSTATE_INTEGRATION.value;
  const buffer = new body.mj.DoubleBuffer(body.mj.mj_stateSize(body.model, spec));
  let nativeIntegration;
  try {
    body.mj.mj_getState(body.model, data, buffer, spec);
    nativeIntegration = {spec, name:'mjSTATE_INTEGRATION', values:array(buffer.GetView(), 'nativeIntegration')};
  } finally {buffer.delete();}
  const native = {time: requireFinite(data.time, 'native.time'),
    ...Object.fromEntries(integrationFields.map(key => [key, array(data[key], 'native.' + key)]))};
  // The muscle solver owns its state in WASM; body.muscleState is a copied
  // readback. Preserve both instead of assuming those buffers are identical.
  const wasmState = muscles.core.HEAPF32.subarray(muscles.state / 4, muscles.state / 4 + muscles.count * 3);
  const packed = (values, bits) => ({encoding:`float${bits}-little-endian-base64`,data:packCaptureArray(values,bits)});
  // Binary copies are authoritative for replay: JSON numbers erase -0 even
  // though ordinary finite numeric magnitudes otherwise round-trip exactly.
  const initialPacked = {nativeIntegration:packed(nativeIntegration.values,64), muscleWasmState:packed(wasmState,32),
    muscleState:packed(body.muscleState,32), muscleInput:packed(body.input,32), activation:packed(body.activation,32), restPose:packed(body.restPose,64),
    wing:{deployment:packed(wings.deployment,64),power:packed(wings.power,64),target:packed(wings.target,64),
      residuals:wings.residuals.map(values=>packed(values,64))}};
  return {native, nativeIntegration, packed:initialPacked, muscleWasmState: array(wasmState, 'muscleWasmState'), muscleState: array(body.muscleState, 'muscleState'),
    muscleInput: array(body.input, 'muscleInput'), activation: array(body.activation, 'activation'),
    restPose: array(body.restPose, 'restPose'), restHeight: requireFinite(body.restHeight, 'restHeight'),
    remainder: requireFinite(body.remainder, 'remainder'), internal: clone({...body.internal}), food: clone(body.food),
    monitor: clone({...body.monitor}),
    wing: {phase: wings.phase, deployment: array(wings.deployment, 'wing.deployment'), power: array(wings.power, 'wing.power'),
      target: array(wings.target, 'wing.target'), residuals: wings.residuals.map((values, i) => array(values, 'wing.residuals.' + i)),
      frequencyHz: wings.frequencyHz, interpreter: clone(wings.interpreter)},
    wrapper: {wingDriveLeft: body.wingDriveLeft ?? null, wingDriveRight: body.wingDriveRight ?? null,
      wingPhase: body.wingPhase, wingPower: body.wingPower, wingPowerLeft: body.wingPowerLeft, wingPowerRight: body.wingPowerRight,
      halterePower: clone(body.halterePower), haltereSteering: clone(body.haltereSteering),
      proboscis: body.proboscis, proboscisChannels: clone(body.proboscisChannels ?? null), pump: body.pump, odor: clone(body.odor),
      lastIntake: body.lastIntake ?? null, time: body.time}};
}

export function createMotorReplayCapture({sourceXml, seconds = .5, bodyBlockMs = 2, sourceAssets = {}}) {
  if (typeof sourceXml !== 'string' || !sourceXml.includes('<mujoco') || !Number.isFinite(seconds) || seconds <= 0 || seconds > .5 || bodyBlockMs !== 2)
    throw new Error('Motor capture requires pinned model XML, a 2 ms block and at most 0.5 seconds');
  const maximumSteps = Math.floor(seconds * 1000 / bodyBlockMs + 1e-9);
  if (maximumSteps < 1 || Math.abs(maximumSteps * bodyBlockMs / 1000 - seconds) > 1e-9)
    throw new Error('Motor capture horizon must contain an exact number of 2 ms blocks');
  let output = null, sourceBody = null, finished = false;
  return {
    onInitialState({body,world,fly,motorIndices,job,provenance,initialCondition,initialObservation}) {
      if (output || finished) throw new Error('Motor recorder supports exactly one episode');
      if (Math.abs(body.data.time) > 1e-12) throw new Error('Motor recorder missed the native initial state');
      const indices = Array.from(motorIndices);
      if (indices.length !== world.io.motor_neurons.length || new Set(indices).size !== indices.length ||
        indices.some((value, i) => value !== world.io.motor_neurons[i].index)) throw new Error('BANC motor-index identity mismatch');
      const scene = flybodyScene(sourceXml, world.habitat), nativeHeights = body.model.hfield_data;
      if (nativeHeights.length !== scene.heights.length || scene.heights.some((value, i) => value !== nativeHeights[i]))
        throw new Error('Reconstructed capture scene differs from native heightfield');
      output = {schemaVersion: 1, kind: 'training-native-motor-replay', horizonSeconds: seconds, bodyBlockSeconds: bodyBlockMs / 1000,
        provenance: {...clone(provenance), job: clone(job), observerAssets: clone(sourceAssets)},
        encoding: {ratesHz: 'IEEE754 float32 little-endian base64', postQpos: 'IEEE754 float64 little-endian base64',
          postQvel: 'IEEE754 float64 little-endian base64', sceneHeights: 'IEEE754 float32 little-endian base64'},
        dimensions: {motorCount: indices.length, nq: body.data.qpos.length, nv: body.data.qvel.length,
          muscleCount: body.mappings.length, heightfieldSamples: scene.heights.length}, motorNeuronIndices: indices,
        scene: {xml: scene.xml, heights: packCaptureArray(scene.heights, 32), fruitGeomNames: clone(scene.fruitGeomNames),
          metadata: clone(world.metadata), io: clone(world.io), fruit: clone(world.habitat.fruit), ceiling: world.habitat.ceiling,
          options: {movementMode: world.movementMode, motorCoupling: world.motorCoupling, flightEnabled: world.flightEnabled},
          sourceXmlSha256: world.metadata.xml_sha256, nativeHeightfieldExact: true},
        initial: captureInitialMotorState(body), initialCondition: clone(initialCondition), initialObservation: clone(initialObservation),
        initialFly: {id: fly.id, x: fly.x, y: fly.y, z: fly.z, heading: fly.heading}, steps: [],
        physicalStateWritesByRecorder: 0, sceneReconstruction: 'Pinned source XML plus the initial habitat; every heightfield sample checked against the actual model.',
        rateTiming: 'Each exact held motor vector is read immediately after world.tick, before the next neural step. It drove the entire recorded 2 ms body block.',
        postStateTiming: 'Immediately after world.tick, before reward observations. No native forward or integrator call is issued by this recorder.'};
      sourceBody = body;
    },
    onPhysicsStep({body,world,fly,index,durationSeconds,timeBefore,neuralMs}) {
      if (!output) throw new Error('Motor step observed before initial state');
      if (finished || output.steps.length >= maximumSteps) return;
      if (body !== sourceBody || index !== output.steps.length || Math.abs(durationSeconds - output.bodyBlockSeconds) > 1e-12 ||
        Math.abs(timeBefore - index * durationSeconds) > 1e-9 || Math.abs(body.data.time - timeBefore - durationSeconds) > 1e-9 ||
        Math.abs(neuralMs / 1000 - body.data.time) > 1e-9) throw new Error('Motor capture body/index/clock discontinuity');
      const rates = fly.brain.motorNeuronRates;
      if (!Array.isArray(rates) || rates.length !== output.dimensions.motorCount) throw new Error('Motor capture rate-vector size mismatch');
      output.steps.push({index, durationSeconds, timeBefore, timeAfter: body.data.time, neuralMs,
        ratesHz: packCaptureArray(rates, 32), postQpos: packCaptureArray(body.data.qpos), postQvel: packCaptureArray(body.data.qvel),
        bodyOptions: {coupling: world.motorCoupling, flight: world.flightEnabled},
        food: clone(body.food)});
    },
    finish(evaluation) {
      if (finished) throw new Error('Motor capture already finalized');
      finished = true;
      if (!output) return null; // A cancelled, paused lifecycle check may never create a body.
      const {reason,simSeconds,steps,cancelled,success} = evaluation;
      output.evaluation = {reason,simSeconds,steps,cancelled,success};
      output.capturedSeconds = output.steps.length * output.bodyBlockSeconds;
      output.horizonReached = output.steps.length === maximumSteps;
      output.endedBy = output.horizonReached ? 'capture_limit' : 'episode_end';
      output.completeThroughEvaluation = output.steps.length === steps;
      if (new TextEncoder().encode(JSON.stringify(output)).byteLength > 7 * 1024 * 1024)
        throw new Error('Motor capture exceeds its 7 MiB report allocation');
      sourceBody = null;
      return output;
    },
  };
}
