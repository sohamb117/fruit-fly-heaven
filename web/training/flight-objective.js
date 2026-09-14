// Pure measurements and modeled task criteria. This observer never supplies a
// motor command, external force, target pose, or corrective controller.
const EPSILON = 1e-9;
const clamp = (x, low = 0, high = 1) => Math.max(low, Math.min(high, x));
const cos = degrees => Math.cos(degrees * Math.PI / 180);
const horizons = Object.freeze({takeoff: 3, flight: 5, landing: 8});

export const FLIGHT_CRITERIA = Object.freeze({
  version: 3,
  interpretation: 'Modeled behavioral thresholds, not measured biological limits.',
  minimumDurationSeconds: horizons,
  maximumDurationSeconds: horizons,
  maximumStepSeconds: .02,
  maximumRadiusCm: 6.5,
  minimumHeightCm: -.5,
  ceilingAllowanceCm: .2,
  maximumAngularSpeed: 300,
  overturnedSeconds: .1,
  gravityCmPerSecondSquared: 981,
  takeoff: Object.freeze({minimumFootLoadFraction: .05, recentSupportSeconds: .12,
    minimumVerticalSpeed: .5, minimumClearanceCm: .15, minimumFlightSeconds: .15}),
  flight: Object.freeze({minimumWingPower: .1, minimumUp: cos(45), maximumAngularSpeed: 20,
    angularSpeedWindowSeconds: .02,
    minimumVerticalSpeed: -1, minimumSupportFraction: .8,
    supportWindowSeconds: .05, minimumContinuousSeconds: 1}),
  landing: Object.freeze({minimumFlightSeconds: 1, approachWindowSeconds: .05,
    maximumApproachSpeed: 1, maximumApproachVerticalSpeed: 1,
    settlementSeconds: .1, minimumFeet: 3, minimumFootLoadFraction: .8,
    minimumUp: cos(20), maximumAngularSpeed: 5, maximumSpeed: .3,
    minimumStableSeconds: .5}),
});

function number(value, name, low = -Infinity, high = Infinity) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high)
    throw new TypeError(`Invalid flight observation: ${name}`);
  return value;
}
function count(value, name, maximum = Number.MAX_SAFE_INTEGER) {
  number(value, name, 0, maximum);
  if (!Number.isInteger(value)) throw new TypeError(`Invalid flight observation: ${name}`);
  return value;
}
function observation(value) {
  if (!value || value.finite !== true) throw new TypeError('Invalid flight observation: finite state');
  if (typeof value.externalForce !== 'boolean') throw new TypeError('Invalid flight observation: externalForce');
  const o = {
    up: clamp(number(value.up, 'up', -1 - 1e-6, 1 + 1e-6), -1, 1),
    angularSpeed: number(value.angularSpeed, 'angularSpeed', 0),
    speedCmPerSecond: number(value.speedCmPerSecond, 'speedCmPerSecond', 0),
    verticalSpeed: number(value.verticalSpeed, 'verticalSpeed'),
    height: number(value.height, 'height'), radius: number(value.radius, 'radius', 0),
    ceiling: number(value.ceiling, 'ceiling'), wingPower: number(value.wingPower, 'wingPower', 0, 1),
    environmentContacts: count(value.environmentContacts, 'environmentContacts'),
    nonFootEnvironmentContacts: count(value.nonFootEnvironmentContacts, 'nonFootEnvironmentContacts'),
    footSupportCount: count(value.footSupportCount, 'footSupportCount', 6),
    footSupportFraction: number(value.footSupportFraction, 'footSupportFraction', 0),
    externalForce: value.externalForce,
  };
  if (o.nonFootEnvironmentContacts > o.environmentContacts || o.footSupportCount > o.environmentContacts)
    throw new TypeError('Invalid flight observation: contradictory contact counts');
  if ((o.footSupportCount === 0) !== (o.footSupportFraction === 0))
    throw new TypeError('Invalid flight observation: foot count/load disagreement');
  if (o.speedCmPerSecond + 1e-6 < Math.abs(o.verticalSpeed))
    throw new TypeError('Invalid flight observation: total speed below vertical speed');
  return o;
}
const airborne = o => o.environmentContacts === 0;
const loadedFeet = o => o.footSupportCount > 0 && o.footSupportFraction > 0;
const supported = o => loadedFeet(o) && o.nonFootEnvironmentContacts === 0 &&
  o.footSupportFraction >= FLIGHT_CRITERIA.takeoff.minimumFootLoadFraction;
const flying = (o, angularSpeed = o.angularSpeed) => airborne(o) && o.wingPower > FLIGHT_CRITERIA.flight.minimumWingPower &&
  o.up >= FLIGHT_CRITERIA.flight.minimumUp && angularSpeed <= FLIGHT_CRITERIA.flight.maximumAngularSpeed &&
  o.verticalSpeed >= FLIGHT_CRITERIA.flight.minimumVerticalSpeed;
const stableLanding = o => o.nonFootEnvironmentContacts === 0 &&
  o.footSupportCount >= FLIGHT_CRITERIA.landing.minimumFeet &&
  o.footSupportFraction >= FLIGHT_CRITERIA.landing.minimumFootLoadFraction &&
  o.up >= FLIGHT_CRITERIA.landing.minimumUp && o.angularSpeed <= FLIGHT_CRITERIA.landing.maximumAngularSpeed &&
  o.speedCmPerSecond <= FLIGHT_CRITERIA.landing.maximumSpeed;

// Retain only airborne velocity intervals. In particular, the ground impulse
// between a supported sample and the first airborne sample is never included.
function velocityWindow() {
  const rows = [];
  let seconds = 0;
  return {
    clear() { rows.length = 0; seconds = 0; },
    append(previous, current, dt) {
      const limit = FLIGHT_CRITERIA.flight.supportWindowSeconds;
      rows.push({dt, from: previous.verticalSpeed, to: current.verticalSpeed}); seconds += dt;
      while (rows.length && seconds - rows[0].dt >= limit - EPSILON) seconds -= rows.shift().dt;
      if (rows.length && seconds > limit) {
        const row = rows[0], removed = seconds - limit;
        row.from += (row.to - row.from) * removed / row.dt; row.dt -= removed; seconds -= removed;
      }
      return {seconds, supportFraction: rows.length ?
        1 + (rows.at(-1).to - rows[0].from) / seconds / FLIGHT_CRITERIA.gravityCmPerSecondSquared : null};
    },
  };
}

// Root motion includes wingbeat recoil. Measure its magnitude over several
// beats (20 ms at the nominal 236 Hz), without cancelling opposite rotations.
// The instantaneous attitude and catastrophic angular-speed limits still apply.
function angularSpeedWindow() {
  const rows = [];
  let seconds = 0, squaredIntegral = 0;
  return {
    clear() { rows.length = 0; seconds = 0; squaredIntegral = 0; },
    append(speed, dt) {
      const limit = FLIGHT_CRITERIA.flight.angularSpeedWindowSeconds, squared = speed * speed;
      rows.push({dt, squared}); seconds += dt; squaredIntegral += dt * squared;
      while (rows.length && seconds - rows[0].dt >= limit - EPSILON) {
        const row = rows.shift(); seconds -= row.dt; squaredIntegral -= row.dt * row.squared;
      }
      if (rows.length && seconds > limit) {
        const removed = seconds - limit; rows[0].dt -= removed; seconds -= removed;
        squaredIntegral -= removed * rows[0].squared;
      }
      return {seconds, rms: Math.sqrt(Math.max(0, squaredIntegral / seconds))};
    },
  };
}

/** Full-horizon observer. Failed horizons return terminated:false/time_limit;
 * only physical failures and completed successful horizons terminate. */
export function createFlightScore(stage, duration, initial) {
  const c = FLIGHT_CRITERIA;
  if (!Object.hasOwn(horizons, stage)) throw new RangeError('Unknown flight objective stage');
  if (!Number.isFinite(duration) || Math.abs(duration - horizons[stage]) > EPSILON)
    throw new RangeError(`The ${stage} objective requires a full ${horizons[stage]} second horizon`);
  let previous = observation(initial), finished = null;
  let lastSupportTime = supported(previous) ? 0 : -Infinity;
  let supportHeight = previous.height, launchHeight = previous.height;
  let eligibleDeparture = false, departureTime = null, boutTakeoffConfirmed = false, upwardDeparture = false;
  let tentativeAirborneSeconds = 0;
  let landingCandidate = false, touchdownTime = null, landingQualificationTime = null;
  let flightQualifiedBeforeContact = 0;
  const velocity = velocityWindow(), angular = angularSpeedWindow(), approach = [];
  const state = {stage, duration, elapsed: 0, success: false, reason: null, return: 0,
    hasTakenOff: false, takeoffTime: null, flightSeconds: 0, bestFlightSeconds: 0,
    landingSeconds: 0, landingTime: null, phase: airborne(previous) ? 'unqualified_airborne' : 'grounded',
    diagnostics: {maximumAngularSpeed: previous.angularSpeed, overturnedSeconds: 0,
      launchHeight: null, clearanceCm: 0, inferredSupportFraction: null, supportWindowSeconds: 0,
      angularSpeedRms: null, angularSpeedWindowSeconds: 0,
      departureCount: 0, confirmedTakeoffs: 0, rejectedTouchdowns: 0,
      poweredAirborneSeconds: 0, tentativeProgress: 0, bestEarnedProgress: 0, failurePenalty: 0,
      initialSupported: supported(previous), invalidObservation: null}};
  const finish = (reason, success, terminated) => {
    state.reason = reason; state.success = success;
    if (terminated && !success) state.phase = 'failed';
    Object.freeze(state.diagnostics); Object.freeze(state);
    finished = Object.freeze({terminated, success, reason}); return finished;
  };
  const fail = reason => {
    if (reason === 'invalid_observation' || reason === 'unexpected_external_force') state.return = -10;
    else {
      // Physical termination already rejects the failed episode. Rank failed
      // attempts by bounded, measured task progress; the last impact's pose or
      // angular speed must not outweigh an improvement in powered airtime.
      // Standing still earns no progress. Crash measurements remain diagnostic.
      state.diagnostics.failurePenalty = 3;
      state.return = clamp(state.diagnostics.bestEarnedProgress - state.diagnostics.failurePenalty, -8, 3);
    }
    return finish(reason, false, true);
  };
  const physicalFailure = o => o.externalForce ? 'unexpected_external_force' :
    o.radius > c.maximumRadiusCm || o.height < c.minimumHeightCm || o.height > o.ceiling + c.ceilingAllowanceCm ? 'outside_habitat' :
    o.angularSpeed > c.maximumAngularSpeed ? 'excessive_rotation' :
    state.diagnostics.overturnedSeconds + EPSILON >= c.overturnedSeconds ? 'overturned' : null;
  const initialFailure = physicalFailure(previous);
  if (initialFailure) fail(initialFailure);

  return {state, step(value, dt) {
    if (finished) return finished;
    if (!Number.isFinite(dt) || dt <= 0 || dt > c.maximumStepSeconds || state.elapsed + dt > duration + 1e-7)
      throw new RangeError('Invalid flight objective timestep');
    let o;
    try { o = observation(value); }
    catch (error) { state.diagnostics.invalidObservation = error.message; return fail('invalid_observation'); }
    const h = Math.min(dt, duration - state.elapsed); state.elapsed += h;
    if (duration - state.elapsed < EPSILON) state.elapsed = duration;
    const d = state.diagnostics;
    d.maximumAngularSpeed = Math.max(d.maximumAngularSpeed, o.angularSpeed);
    d.overturnedSeconds = o.up < 0 ? d.overturnedSeconds + h : 0;
    const failure = physicalFailure(o); if (failure) return fail(failure);

    const inAir = airborne(o), wasAir = airborne(previous);
    if (inAir) {
      const window = angular.append(o.angularSpeed, h);
      d.angularSpeedRms = window.rms; d.angularSpeedWindowSeconds = window.seconds;
    } else { angular.clear(); d.angularSpeedRms = null; d.angularSpeedWindowSeconds = 0; }
    const controlled = inAir && d.angularSpeedWindowSeconds + EPSILON >= c.flight.angularSpeedWindowSeconds &&
      flying(o, d.angularSpeedRms);
    if (supported(o)) { lastSupportTime = state.elapsed; supportHeight = o.height; }
    // A body/wing impact cannot borrow an earlier foot-supported launch.
    // Foot-only unloading or a brief bounce may still start a fresh attempt.
    else if (o.nonFootEnvironmentContacts > 0) lastSupportTime = -Infinity;
    if (inAir && !wasAir) {
      // Contact loss identifies the foot-origin attempt, not successful
      // takeoff. Its first sample may have low velocity or undeployed wings.
      // Require powered airborne support, ascent and clearance below instead.
      eligibleDeparture = previous.nonFootEnvironmentContacts === 0 &&
        state.elapsed - lastSupportTime <= c.takeoff.recentSupportSeconds + EPSILON;
      departureTime = state.elapsed; launchHeight = supportHeight; boutTakeoffConfirmed = false; upwardDeparture = false;
      landingCandidate = false; state.landingSeconds = 0; state.landingTime = null; landingQualificationTime = null;
      velocity.clear(); approach.length = 0; state.flightSeconds = 0;
      tentativeAirborneSeconds = 0;
      if (eligibleDeparture) { d.departureCount++; d.launchHeight = launchHeight; }
    }
    if (!inAir && wasAir) {
      const recent = approach.filter(row => row.time >= state.elapsed - h - c.landing.approachWindowSeconds - EPSILON);
      const slowApproach = recent.length > 0 && recent.at(-1).time - recent[0].time + EPSILON >= c.landing.approachWindowSeconds &&
        recent.every(row => row.speed <= c.landing.maximumApproachSpeed && Math.abs(row.verticalSpeed) <= c.landing.maximumApproachVerticalSpeed);
      flightQualifiedBeforeContact = state.flightSeconds;
      landingCandidate = eligibleDeparture && boutTakeoffConfirmed &&
        state.flightSeconds + EPSILON >= c.landing.minimumFlightSeconds && slowApproach &&
        loadedFeet(o) && o.nonFootEnvironmentContacts === 0;
      if (!landingCandidate) d.rejectedTouchdowns++;
      touchdownTime = state.elapsed; state.landingSeconds = 0; state.landingTime = null; landingQualificationTime = null;
      eligibleDeparture = false; boutTakeoffConfirmed = false; upwardDeparture = false; velocity.clear(); approach.length = 0;
      tentativeAirborneSeconds = 0;
    }

    d.clearanceCm = eligibleDeparture ? Math.max(0, o.height - launchHeight) : 0;
    if (eligibleDeparture && controlled && o.verticalSpeed > c.takeoff.minimumVerticalSpeed) upwardDeparture = true;
    // Before a full support window exists, offer only a small bounded hint.
    // Actual airborne acceleration supplies support; the first sample gets a
    // weak tentative factor rather than counting the ground's launch impulse.
    // Pure ballistic acceleration therefore cannot accumulate this credit.
    if (eligibleDeparture && inAir && o.wingPower > c.flight.minimumWingPower) {
      const supportHint = wasAir ? clamp(1 + (o.verticalSpeed - previous.verticalSpeed) / h / c.gravityCmPerSecondSquared) : .1;
      const controlHint = clamp((o.up - cos(75)) / (1 - cos(75))) *
        clamp(1 - o.angularSpeed / 100) * clamp(1 + Math.min(0, o.verticalSpeed) / 5);
      tentativeAirborneSeconds += h * supportHint * controlHint;
      d.tentativeProgress = Math.min(.2, 4 * tentativeAirborneSeconds);
    } else { tentativeAirborneSeconds = 0; d.tentativeProgress = 0; }
    let qualifiedNow = false;
    // Measure airborne support independently of attitude qualification. A
    // recoil excursion can interrupt controlled airtime without discarding
    // the evidence needed to measure aerodynamic support afterwards.
    if (inAir && wasAir) {
      const window = velocity.append(previous, o, h);
      d.inferredSupportFraction = window.supportFraction; d.supportWindowSeconds = window.seconds;
    } else { velocity.clear(); d.inferredSupportFraction = null; d.supportWindowSeconds = 0; }
    if (eligibleDeparture && controlled) {
      d.poweredAirborneSeconds += h;
      qualifiedNow = d.supportWindowSeconds + EPSILON >= c.flight.supportWindowSeconds &&
        d.inferredSupportFraction >= c.flight.minimumSupportFraction;
      state.flightSeconds = qualifiedNow ? state.flightSeconds + h : 0;
      if (qualifiedNow && upwardDeparture && !boutTakeoffConfirmed && d.clearanceCm + EPSILON >= c.takeoff.minimumClearanceCm &&
        state.flightSeconds + EPSILON >= c.takeoff.minimumFlightSeconds) {
        boutTakeoffConfirmed = true; state.hasTakenOff = true;
        if (state.takeoffTime === null) state.takeoffTime = departureTime;
        d.confirmedTakeoffs++;
      }
    } else {
      state.flightSeconds = 0;
    }
    state.bestFlightSeconds = Math.max(state.bestFlightSeconds, state.flightSeconds);
    if (inAir) {
      approach.push({time: state.elapsed, speed: o.speedCmPerSecond, verticalSpeed: o.verticalSpeed});
      // Keep a bracketing sample so a full interval can be verified at either
      // 2 ms or 10 ms observations; do not choose just the slow final endpoint.
      while (approach.length > 1 && approach[1].time < state.elapsed - c.landing.approachWindowSeconds - EPSILON) approach.shift();
    }

    if (landingCandidate) {
      const stable = stableLanding(o);
      if (inAir || !loadedFeet(o) || o.nonFootEnvironmentContacts > 0 ||
        (!stable && state.elapsed - touchdownTime > c.landing.settlementSeconds + EPSILON)) {
        landingCandidate = false; state.landingSeconds = 0; state.landingTime = null; landingQualificationTime = null;
      } else {
        state.landingSeconds = stable ? state.landingSeconds + h : 0;
        if (state.landingSeconds + EPSILON >= c.landing.minimumStableSeconds && state.landingTime === null) {
          state.landingTime = touchdownTime; landingQualificationTime = state.elapsed;
        }
      }
    }
    const takeoffCurrent = eligibleDeparture && boutTakeoffConfirmed && qualifiedNow &&
      d.clearanceCm + EPSILON >= c.takeoff.minimumClearanceCm;
    const flightCurrent = takeoffCurrent && state.flightSeconds + EPSILON >= c.flight.minimumContinuousSeconds;
    const landedCurrent = landingCandidate && stableLanding(o) && state.landingSeconds + EPSILON >= c.landing.minimumStableSeconds;
    state.phase = landedCurrent ? 'landed' : landingCandidate ? 'settling' : eligibleDeparture ?
      (takeoffCurrent ? 'flight' : 'takeoff') : inAir ? 'unqualified_airborne' : 'grounded';

    // No upright/contact reward. All positive progress needs an observed
    // powered departure. Historical flight contributes to landing only while
    // the same validated touchdown remains a viable, loaded-foot settlement.
    const retainedLanding = stage === 'landing' && landingCandidate;
    const launchProgress = retainedLanding ? 1 : eligibleDeparture && qualifiedNow ?
      clamp(d.clearanceCm / c.takeoff.minimumClearanceCm) * clamp(state.flightSeconds / c.takeoff.minimumFlightSeconds) : 0;
    const currentAir = retainedLanding ? flightQualifiedBeforeContact : state.flightSeconds;
    const airProgress = stage === 'takeoff' ? clamp(currentAir / c.takeoff.minimumFlightSeconds) :
      stage === 'flight' ? 4 * clamp(currentAir / duration) : 3 * clamp(currentAir / c.flight.minimumContinuousSeconds);
    const landingProgress = stage === 'landing' && landingCandidate ? 3 * clamp(state.landingSeconds / c.landing.minimumStableSeconds) : 0;
    const successNow = stage === 'takeoff' ? takeoffCurrent && state.flightSeconds + EPSILON >= c.takeoff.minimumFlightSeconds :
      stage === 'flight' ? flightCurrent : landedCurrent;
    const completionTime = successNow ? stage === 'landing' ? landingQualificationTime : departureTime : state.elapsed;
    const earnedProgress = Math.max(2 * launchProgress + airProgress + landingProgress,
      eligibleDeparture && inAir ? d.tentativeProgress : 0);
    d.bestEarnedProgress = Math.max(d.bestEarnedProgress, earnedProgress);
    state.return = clamp(earnedProgress - .05 * state.elapsed - .5 * completionTime / duration, -10, 10);
    previous = o;
    if (state.elapsed === duration) {
      if (successNow) { state.return = clamp(state.return + (stage === 'landing' ? 2 : 3), -10, 10); return finish('stage_success', true, true); }
      return finish('time_limit', false, false);
    }
    return {terminated: false, success: false, reason: null};
  }};
}
