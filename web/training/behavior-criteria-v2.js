// Research observation criteria, not measured biological limits or motor commands.
// This version is intentionally separate from the deployed v1 objective.
const freeze = Object.freeze;
const cos = degrees => Math.cos(degrees * Math.PI / 180);
const EPSILON = 1e-9;
const clamp = (x, low = 0, high = 1) => Math.max(low, Math.min(high, x));

export const BEHAVIOR_CRITERIA_V2 = freeze({
  version: 2,
  interpretation: 'Research criteria for observable behavior; thresholds are not measured biological facts.',
  localizationEvidence: 'Odor/heading localization proxy; this does not establish sensory identification of food.',
  minimumDurationSeconds: freeze({posture: 2, localization: 2, approach: 3, probing: 3, feeding: 3, takeoff: 2, flight: 2, landing: 3, sequence: 8}),
  sequence: freeze(['localization', 'approach', 'landing', 'probing', 'feeding', 'takeoff', 'flight']),
  maximumStepSeconds: .05,
  maximumRadiusCm: 6.5,
  minimumHeightCm: -.5,
  ceilingAllowanceCm: .2,
  maximumAngularSpeed: 300,
  overturnedSeconds: .1,
  gravityCmPerSecondSquared: 981,
  support: freeze({minimumFeet: 3, minimumFraction: .8}),
  posture: freeze({minimumInitialTiltDegrees: 25, minimumInitialAngularSpeed: 8, minimumInitialSpeed: .5,
    minimumUp: cos(15), maximumAngularSpeed: 3, maximumSpeed: .2, recoveryDeadlineSeconds: 1,
    finalStableSeconds: .75}),
  approach: freeze({minimumProgressCm: .23, minimumContinuousSeconds: .2, minimumUp: cos(60), minimumFoodFeet: 2}),
  probing: freeze({minimumContinuousSeconds: .1}),
  feeding: freeze({minimumIntake: .001}),
  takeoff: freeze({minimumSupportedSeconds: .05, maximumSupportToDepartureSeconds: .1, minimumVerticalSpeed: .5}),
  flight: freeze({minimumContinuousSeconds: 1, minimumWingPower: .1, minimumUp: cos(45),
    maximumAngularRms: 20, minimumSupportFraction: .8, minimumMeanVerticalSpeed: -1}),
  landing: freeze({minimumContinuousSeconds: .5, maximumSettlementSeconds: .1, minimumUp: cos(20), maximumAngularSpeed: 5,
    maximumApproachSpeed: 1, maximumApproachVerticalSpeed: 1, maximumContactSpeed: .3, minimumFoodFeet: 2}),
});

const finite = (value, name, low = -Infinity, high = Infinity) => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high)
    throw new TypeError(`Invalid behavior observation: ${name}`);
  return value;
};
const count = (value, name, maximum = Number.MAX_SAFE_INTEGER) => {
  finite(value, name, 0, maximum);
  if (!Number.isInteger(value)) throw new TypeError(`Invalid behavior observation: ${name}`);
  return value;
};
const flag = (value, name) => {
  if (typeof value !== 'boolean') throw new TypeError(`Invalid behavior observation: ${name}`);
  return value;
};

/**
 * Normalize measurements only. In particular, environment contact is never a
 * substitute for foot support. footSupportFraction is summed upward force from
 * supporting feet / body weight; footSupportCount counts distinct loaded feet.
 * angularVelocity is a three-component instantaneous native angular velocity;
 * angularSpeed may instead supply its norm, never a signed/vector time average.
 * probing means an extended proboscis actually contacting food. localized is
 * the explicitly limited existing odor/heading proxy, not a desired heading.
 */
export function normalizeBehaviorObservationV2(value) {
  if (!value || typeof value !== 'object' || value.finite === false)
    throw new TypeError('Invalid behavior observation: nonfinite state');
  let angularSpeed;
  if (value.angularVelocity !== undefined) {
    if ((!Array.isArray(value.angularVelocity) && !ArrayBuffer.isView(value.angularVelocity)) || value.angularVelocity.length !== 3)
      throw new TypeError('Invalid behavior observation: angularVelocity');
    angularSpeed = finite(Math.hypot(...Array.from(value.angularVelocity, (v, i) => finite(v, `angularVelocity[${i}]`))), 'angularSpeed', 0);
  } else angularSpeed = finite(value.angularSpeed, 'angularSpeed', 0);
  const observation = {
    up: finite(value.up, 'up', -1, 1), angularSpeed,
    speedCmPerSecond: finite(value.speedCmPerSecond, 'speedCmPerSecond', 0),
    verticalSpeed: finite(value.verticalSpeed, 'verticalSpeed'),
    height: finite(value.height, 'height'), radius: finite(value.radius, 'radius', 0),
    ceiling: finite(value.ceiling, 'ceiling'), distance: finite(value.distance, 'distance', 0),
    intake: finite(value.intake, 'intake', 0), wingPower: finite(value.wingPower, 'wingPower', 0, 1),
    environmentContacts: count(value.environmentContacts, 'environmentContacts'),
    footSupportCount: count(value.footSupportCount, 'footSupportCount', 6),
    footSupportFraction: finite(value.footSupportFraction, 'footSupportFraction', 0),
    footFoodContacts: count(value.footFoodContacts, 'footFoodContacts', 6),
    airborne: flag(value.airborne, 'airborne'), externalForce: flag(value.externalForce, 'externalForce'),
    probing: flag(value.probing, 'probing'), onFood: flag(value.onFood, 'onFood'),
    localized: flag(value.localized, 'localized'),
  };
  if (observation.environmentContacts === 0 && (observation.footSupportCount > 0 || observation.footSupportFraction > 0 || observation.footFoodContacts > 0))
    throw new TypeError('Invalid behavior observation: loaded feet require environment contacts');
  if (observation.footSupportCount === 0 && observation.footSupportFraction > 0)
    throw new TypeError('Invalid behavior observation: foot load without a supporting foot');
  if (observation.airborne && observation.environmentContacts > 0)
    throw new TypeError('Invalid behavior observation: airborne state has environment contact');
  if (observation.speedCmPerSecond + EPSILON < Math.abs(observation.verticalSpeed))
    throw new TypeError('Invalid behavior observation: total speed is less than vertical speed');
  return observation;
}

function supported(o) {
  const c = BEHAVIOR_CRITERIA_V2.support;
  return !o.airborne && o.environmentContacts > 0 && o.footSupportCount >= c.minimumFeet && o.footSupportFraction >= c.minimumFraction;
}
function stablePosture(o) {
  const c = BEHAVIOR_CRITERIA_V2.posture;
  return supported(o) && o.up >= c.minimumUp && o.angularSpeed <= c.maximumAngularSpeed && o.speedCmPerSecond <= c.maximumSpeed;
}

// Time-weighted observations prevent cancellation of alternating angular
// velocities and make the result independent of render/preview sample cadence.
function flightWindow() {
  const samples = [];
  let seconds = 0;
  return {
    clear() { samples.length = 0; seconds = 0; },
    sample(o, previous, dt) {
      const c = BEHAVIOR_CRITERIA_V2.flight;
      samples.push({dt, omegaSquared: o.angularSpeed ** 2, startVelocity: previous.verticalSpeed, endVelocity: o.verticalSpeed});
      seconds += dt;
      while (samples.length && seconds - samples[0].dt >= c.minimumContinuousSeconds - EPSILON) seconds -= samples.shift().dt;
      if (seconds > c.minimumContinuousSeconds && samples.length) {
        const first = samples[0], removed = seconds - c.minimumContinuousSeconds;
        first.startVelocity += (first.endVelocity - first.startVelocity) * removed / first.dt;
        first.dt -= removed; seconds -= removed;
      }
      const angularRms = Math.sqrt(samples.reduce((sum, row) => sum + row.omegaSquared * row.dt, 0) / seconds);
      const meanVerticalSpeed = samples.reduce((sum, row) => sum + (row.startVelocity + row.endVelocity) * .5 * row.dt, 0) / seconds;
      const supportFraction = 1 + (samples.at(-1).endVelocity - samples[0].startVelocity) / seconds / BEHAVIOR_CRITERIA_V2.gravityCmPerSecondSquared;
      return {seconds, angularRms, meanVerticalSpeed, supportFraction,
        qualified: seconds + EPSILON >= c.minimumContinuousSeconds && angularRms <= c.maximumAngularRms &&
          meanVerticalSpeed >= c.minimumMeanVerticalSpeed && supportFraction >= c.minimumSupportFraction};
    },
  };
}

/** Pure observer: step() returns {terminated, success, reason}; state.return is [-10, 10]. */
export function createBehaviorScoreV2(stage, duration, initial) {
  const c = BEHAVIOR_CRITERIA_V2;
  if (typeof stage !== 'string' || !Object.hasOwn(c.minimumDurationSeconds, stage)) throw new RangeError('Unknown v2 behavior stage');
  const minimum = c.minimumDurationSeconds[stage];
  if (!Number.isFinite(duration) || duration + EPSILON < minimum || duration > 3600)
    throw new RangeError(`The ${stage} stage requires at least ${minimum} simulated seconds`);
  let previous = normalizeBehaviorObservationV2(initial), finished = null;
  const window = flightWindow();
  const disturbed = previous.up <= cos(c.posture.minimumInitialTiltDegrees) || previous.angularSpeed >= c.posture.minimumInitialAngularSpeed ||
    previous.speedCmPerSecond >= c.posture.minimumInitialSpeed;
  const state = {version: 2, stage, duration, elapsed: 0, success: false, reason: null, return: 0,
    initialDisturbance: disturbed, recoveryTime: null, stableSeconds: 0, supportedSeconds: 0,
    maximumAngularSpeed: 0, overturnedSeconds: 0, approachSeconds: 0, approachProgress: 0, approachBoutProgress: 0, approachObserved: false,
    probingSeconds: 0, intakeWithProbing: 0, hasTakenOff: false, takeoffTime: null,
    flight: {seconds: 0, angularRms: 0, meanVerticalSpeed: 0, supportFraction: 0, qualified: false},
    landingSeconds: 0, landingTime: null, localizationTime: null,
    localizationEvidence: c.localizationEvidence, sequenceIndex: 0, events: []};
  const initialDistance = previous.distance;
  let footSupportSeconds = 0, lastQualifiedSupportTime = -Infinity, landingCandidate = false, landingRecorded = false, touchdownTime = 0;
  let sequenceStart = 0, sequenceIntake = 0, sequenceApproachSeconds = 0, sequenceApproachProgress = 0, sequenceProbeSeconds = 0;
  const record = (name, evidence) => state.events.push({stage: name, time: state.elapsed, evidence});
  const terminate = (reason, success = false) => {
    state.success = success; state.reason = reason;
    finished = {terminated: true, success, reason};
    return finished;
  };

  return {state, step(value, dt) {
    if (finished) return {...finished};
    if (!Number.isFinite(dt) || dt <= 0 || dt > c.maximumStepSeconds) throw new RangeError('Invalid v2 behavior step duration');
    let o;
    try { o = normalizeBehaviorObservationV2(value); }
    catch (error) { state.invalidObservation = error.message; state.return = -10; return terminate('invalid_observation'); }
    const h = Math.min(dt, duration - state.elapsed);
    state.elapsed += h;
    state.maximumAngularSpeed = Math.max(state.maximumAngularSpeed, o.angularSpeed);
    state.overturnedSeconds = o.up < 0 ? state.overturnedSeconds + h : 0;
    const failure = o.externalForce ? 'unexpected_external_force' : o.radius > c.maximumRadiusCm || o.height < c.minimumHeightCm || o.height > o.ceiling + c.ceilingAllowanceCm ?
      'outside_habitat' : o.angularSpeed > c.maximumAngularSpeed ? 'excessive_rotation' : state.overturnedSeconds + EPSILON >= c.overturnedSeconds ? 'overturned' :
        o.intake + EPSILON < previous.intake ? 'nonmonotonic_intake' : null;
    if (failure) { state.return = -10; return terminate(failure); }

    const footSupported = supported(o), postureStable = stablePosture(o);
    if (footSupported) state.supportedSeconds += h;
    footSupportSeconds = footSupported ? footSupportSeconds + h : 0;
    if (footSupportSeconds + EPSILON >= c.takeoff.minimumSupportedSeconds) lastQualifiedSupportTime = state.elapsed;
    state.stableSeconds = postureStable ? state.stableSeconds + h : 0;
    // A transient stable frame does not certify a later, unrelated recovery.
    state.recoveryTime = disturbed && state.stableSeconds + EPSILON >= c.posture.finalStableSeconds ? state.elapsed - state.stableSeconds : null;
    const progressNow = previous.distance - o.distance;
    const approaching = progressNow > 0 && o.up >= c.approach.minimumUp;
    state.approachSeconds = approaching ? state.approachSeconds + h : 0;
    state.approachBoutProgress = approaching ? state.approachBoutProgress + progressNow : 0;
    state.approachProgress = Math.max(0, initialDistance - o.distance);
    state.probingSeconds = o.probing && o.onFood ? state.probingSeconds + h : 0;
    if (o.probing && o.onFood) state.intakeWithProbing += Math.max(0, o.intake - previous.intake);
    if (o.localized && !previous.localized && state.localizationTime === null) {
      state.localizationTime = state.elapsed; record('localization', c.localizationEvidence);
    }

    // Feet normally unload one at a time. Preserve recent full support, while
    // requiring the final departure to follow actual foot loading, not a wing hit.
    const tookOff = !previous.airborne && previous.footSupportCount > 0 && previous.footSupportFraction > 0 && o.airborne && o.environmentContacts === 0 &&
      state.elapsed - lastQualifiedSupportTime <= c.takeoff.maximumSupportToDepartureSeconds + EPSILON &&
      o.wingPower > c.flight.minimumWingPower && o.verticalSpeed > c.takeoff.minimumVerticalSpeed;
    if (tookOff) { state.hasTakenOff = true; state.takeoffTime = state.elapsed; record('takeoff', 'Upward powered departure after measured foot support.'); }
    const touchedDown = previous.airborne && !o.airborne;
    if (touchedDown) {
      landingCandidate = state.flight.qualified && o.footSupportCount > 0 && o.footSupportFraction > 0 && o.onFood && o.footFoodContacts > 0 &&
        previous.speedCmPerSecond <= c.landing.maximumApproachSpeed && Math.abs(previous.verticalSpeed) <= c.landing.maximumApproachVerticalSpeed;
      state.landingSeconds = 0; landingRecorded = false; touchdownTime = state.elapsed;
    }
    const flying = o.airborne && o.environmentContacts === 0 && o.wingPower > c.flight.minimumWingPower && o.up >= c.flight.minimumUp;
    if (flying) {
      // The ground's takeoff impulse is not aerodynamic support during flight.
      state.flight = window.sample(o, previous.airborne && previous.environmentContacts === 0 ? previous : o, h);
    } else {
      window.clear(); state.flight = {seconds: 0, angularRms: 0, meanVerticalSpeed: 0, supportFraction: 0, qualified: false};
    }
    const stableLanding = landingCandidate && footSupported && o.onFood && o.footFoodContacts >= c.landing.minimumFoodFeet &&
      o.up >= c.landing.minimumUp && o.angularSpeed <= c.landing.maximumAngularSpeed && o.speedCmPerSecond <= c.landing.maximumContactSpeed;
    state.landingSeconds = stableLanding ? state.landingSeconds + h : 0;
    if (!stableLanding && (o.airborne || !o.onFood || o.footSupportCount === 0 || o.footSupportFraction <= 0 ||
      state.elapsed - touchdownTime > c.landing.maximumSettlementSeconds + EPSILON)) landingCandidate = false;
    if (state.landingSeconds + EPSILON >= c.landing.minimumContinuousSeconds && !landingRecorded) {
      state.landingTime = state.elapsed; landingRecorded = true;
      record('landing', 'Stable loaded-foot contact on food after sustained powered flight and a bounded-speed approach.');
    }

    if (stage === 'sequence') {
      const next = c.sequence[state.sequenceIndex];
      if (next === 'approach') {
        sequenceApproachSeconds = approaching ? sequenceApproachSeconds + h : 0;
        sequenceApproachProgress = approaching ? sequenceApproachProgress + progressNow : 0;
      }
      if (next === 'probing') sequenceProbeSeconds = o.probing && o.onFood ? sequenceProbeSeconds + h : 0;
      const transition = {
        localization: state.localizationTime !== null && state.localizationTime >= sequenceStart,
        approach: sequenceApproachProgress >= c.approach.minimumProgressCm && sequenceApproachSeconds + EPSILON >= c.approach.minimumContinuousSeconds,
        landing: state.landingTime !== null && state.landingTime >= sequenceStart && stableLanding,
        probing: sequenceProbeSeconds + EPSILON >= c.probing.minimumContinuousSeconds,
        feeding: state.intakeWithProbing - sequenceIntake >= c.feeding.minimumIntake && o.probing,
        takeoff: tookOff,
        flight: state.flight.qualified && state.flight.seconds <= state.elapsed - sequenceStart + EPSILON,
      }[next];
      if (transition) {
        record(`sequence:${next}`, next === 'localization' ? c.localizationEvidence : 'Observed after the preceding sequence stage.');
        state.sequenceIndex++; sequenceStart = state.elapsed; sequenceIntake = state.intakeWithProbing;
      }
    }

    const recovered = disturbed && state.recoveryTime !== null && state.recoveryTime <= c.posture.recoveryDeadlineSeconds &&
      state.stableSeconds + EPSILON >= c.posture.finalStableSeconds;
    const approachGoal = state.approachProgress >= c.approach.minimumProgressCm && o.footFoodContacts >= c.approach.minimumFoodFeet && footSupported && o.up >= c.approach.minimumUp;
    // Preserve evidence of a real approach before the fly stops at food.
    state.approachObserved ||= state.approachSeconds + EPSILON >= c.approach.minimumContinuousSeconds && state.approachBoutProgress >= c.approach.minimumProgressCm;
    const goals = {posture: recovered, localization: state.localizationTime !== null,
      approach: approachGoal && state.approachObserved, probing: state.probingSeconds + EPSILON >= c.probing.minimumContinuousSeconds,
      feeding: state.intakeWithProbing >= c.feeding.minimumIntake && o.probing,
      takeoff: state.hasTakenOff && state.flight.qualified, flight: state.hasTakenOff && state.flight.qualified,
      landing: state.landingTime !== null && stableLanding && state.landingSeconds + EPSILON >= c.landing.minimumContinuousSeconds,
      sequence: state.sequenceIndex === c.sequence.length && state.flight.qualified};
    const progress = {posture: disturbed ? clamp(state.stableSeconds / c.posture.finalStableSeconds) : 0,
      localization: Number(state.localizationTime !== null), approach: clamp(state.approachProgress / c.approach.minimumProgressCm),
      probing: clamp(state.probingSeconds / c.probing.minimumContinuousSeconds), feeding: clamp(state.intakeWithProbing / c.feeding.minimumIntake),
      takeoff: state.hasTakenOff ? clamp(state.flight.seconds / c.flight.minimumContinuousSeconds) : 0,
      flight: state.hasTakenOff ? clamp(state.flight.seconds / c.flight.minimumContinuousSeconds) : 0,
      landing: clamp(state.landingSeconds / c.landing.minimumContinuousSeconds), sequence: state.sequenceIndex / c.sequence.length}[stage];
    const atEnd = state.elapsed + EPSILON >= duration;
    state.success = atEnd && !!goals[stage];
    state.return = stage === 'posture' && !disturbed ? -5 : clamp(5 * progress + 2 * Number(goals[stage]) + (state.success ? 3 : 0) -
      .5 * clamp(state.maximumAngularSpeed / c.maximumAngularSpeed), -10, 10);
    previous = o;
    if (atEnd) return terminate(state.success ? 'stage_success' : stage === 'posture' && !disturbed ? 'insufficient_disturbance' : 'time_limit', state.success);
    return {terminated: false, success: false, reason: null};
  }};
}
