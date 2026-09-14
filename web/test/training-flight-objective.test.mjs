import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {FLIGHT_CRITERIA as criteria, createFlightScore} from '../training/flight-objective.js';

const ground = changes => ({finite: true, up: 1, angularSpeed: 0, speedCmPerSecond: 0,
  verticalSpeed: 0, height: 1, radius: 1, ceiling: 6, wingPower: 0,
  environmentContacts: 6, nonFootEnvironmentContacts: 0, footSupportCount: 6,
  footSupportFraction: 1, externalForce: false, ...changes});
const air = changes => ground({environmentContacts: 0, footSupportCount: 0,
  footSupportFraction: 0, wingPower: .8, height: 1.32, ...changes});
const hover = t => t < .1 - 1e-9 ? ground() : t < .5 - 1e-9 ?
  air({height: 1 + .8 * (t - .1), verticalSpeed: .8, speedCmPerSecond: .8}) : air();
const land = t => t < 2 - 1e-9 ? hover(t) : t < 2.4 - 1e-9 ?
  air({height: 1.32 - .8 * (t - 2), verticalSpeed: -.8, speedCmPerSecond: .8}) : ground();
function run(stage, factory, {dt = .01, initial = ground()} = {}) {
  const duration = criteria.minimumDurationSeconds[stage], score = createFlightScore(stage, duration, initial);
  const frames = [];
  let result;
  for (let k = 1; k <= Math.round(duration / dt); k++) {
    const t = k * dt; result = score.step(factory(t, k), dt);
    frames.push({t, result, reward: score.state.return, flightSeconds: score.state.flightSeconds});
    if (result.terminated) break;
  }
  return {score, result, frames};
}

test('modeled criteria enforce full horizons and known stages', () => {
  assert.match(criteria.interpretation, /not measured biological/);
  assert.deepEqual(criteria.minimumDurationSeconds, {takeoff: 3, flight: 5, landing: 8});
  for (const [stage, duration] of Object.entries(criteria.minimumDurationSeconds)) {
    assert.throws(() => createFlightScore(stage, duration - .1, ground()), /full/);
    assert.throws(() => createFlightScore(stage, duration + .1, ground()), /full/);
  }
  assert.throws(() => createFlightScore('toString', 3, ground()), /Unknown/);
  assert.throws(() => createFlightScore('posture', 3, ground()), /Unknown/);
});

test('standing still, with or without active wings, earns no positive posture reward', () => {
  for (const stage of ['takeoff', 'flight', 'landing']) for (const wingPower of [0, .8]) {
    const {score, result, frames} = run(stage, () => ground({wingPower}));
    assert.deepEqual(result, {terminated: false, success: false, reason: 'time_limit'});
    assert.equal(score.state.hasTakenOff, false);
    assert(frames.every(frame => frame.reward <= 0));
  }
});

test('powered departure and maintained hover succeed only at their complete horizons', () => {
  for (const stage of ['takeoff', 'flight']) {
    const {score, result, frames} = run(stage, hover);
    assert.equal(result.success, true); assert.equal(result.reason, 'stage_success');
    assert.equal(score.state.elapsed, criteria.minimumDurationSeconds[stage]);
    assert(frames.slice(0, -1).every(frame => !frame.result.success && !frame.result.terminated));
    assert(score.state.hasTakenOff); assert(score.state.takeoffTime >= .09 && score.state.takeoffTime <= .11);
    assert(score.state.flightSeconds > criteria.minimumDurationSeconds[stage] - .2);
    assert(score.state.return > 0 && score.state.return <= 10);
  }
});

test('takeoff requires measured clearance, not just airborne flags and active wings', () => {
  const {score, result} = run('takeoff', t => t < .1 ? ground() :
    air({height: 1.05, verticalSpeed: t < .11 ? .8 : 0, speedCmPerSecond: t < .11 ? .8 : 0}));
  assert.equal(result.success, false); assert.equal(score.state.hasTakenOff, false);
  assert(score.state.flightSeconds > 2);
});

test('recent loaded feet allow sequential unloading but body-only support cannot authorize takeoff', () => {
  const sequential = run('flight', t => t < .06 - 1e-9 ? ground() : t < .1 - 1e-9 ?
    ground({environmentContacts: 1, footSupportCount: 1, footSupportFraction: .02}) : hover(t));
  assert.equal(sequential.result.success, true);
  const bodyContact = ground({environmentContacts: 2, nonFootEnvironmentContacts: 2, footSupportCount: 0, footSupportFraction: 0});
  const collision = run('flight', t => t < .1 ? bodyContact : hover(t), {initial: bodyContact});
  assert.equal(collision.result.success, false); assert.equal(collision.score.state.hasTakenOff, false);
  const stale = run('flight', t => t < .3 ? ground({environmentContacts: 1, footSupportCount: 1, footSupportFraction: .001}) :
    air({height: 1.4, verticalSpeed: .8, speedCmPerSecond: .8}));
  assert.equal(stale.result.success, false); assert.equal(stale.score.state.hasTakenOff, false);
});

test('an initially airborne fly cannot qualify by falling, hovering or landing', () => {
  for (const stage of ['takeoff', 'flight', 'landing']) {
    const {score, result} = run(stage, t => t < 2 ? air({verticalSpeed: -.1, speedCmPerSecond: .1}) : ground(), {initial: air()});
    assert.equal(result.success, false); assert.equal(score.state.hasTakenOff, false);
    assert.equal(score.state.bestFlightSeconds, 0); assert.equal(score.state.landingTime, null);
  }
});

test('no-wing launches and prolonged ballistic wing-active motion cannot earn powered flight', () => {
  const noWings = run('flight', t => ({...hover(t), wingPower: 0}));
  assert.equal(noWings.result.success, false); assert.equal(noWings.score.state.bestFlightSeconds, 0);
  let minimumSupport = Infinity;
  const score = createFlightScore('takeoff', 3, ground({ceiling: 100}));
  for (let k = 1; k <= 300; k++) {
    const t = k * .01, tau = t - .1;
    const vz = 300 - criteria.gravityCmPerSecondSquared * tau;
    const o = t < .1 - 1e-9 || tau > .6 ? ground({ceiling: 100}) :
      air({ceiling: 100, height: 1 + 300 * tau - .5 * criteria.gravityCmPerSecondSquared * tau ** 2,
        verticalSpeed: vz, speedCmPerSecond: Math.abs(vz)});
    score.step(o, .01);
    if (score.state.diagnostics.inferredSupportFraction !== null)
      minimumSupport = Math.min(minimumSupport, score.state.diagnostics.inferredSupportFraction);
  }
  assert(Math.abs(minimumSupport) < 1e-10, 'Ground impulse must not inflate inferred airborne support');
  assert.equal(score.state.hasTakenOff, false); assert.equal(score.state.bestFlightSeconds, 0);
  assert.equal(score.state.success, false);
});

test('wing-active rapid descent never qualifies as maintained flight', () => {
  const {score, result} = run('flight', t => t < .2 ? hover(t) : air({verticalSpeed: -3, speedCmPerSecond: 3}));
  assert.equal(result.success, false); assert.equal(score.state.flightSeconds, 0);
});

test('repeated short hops cannot accumulate into takeoff or sustained airtime', () => {
  for (const stage of ['takeoff', 'flight', 'landing']) {
    const {score, result} = run(stage, t => {
      const phase = t % .2;
      return phase < .06 ? ground() : air({height: 1 + 4 * (phase - .06), verticalSpeed: 4, speedCmPerSecond: 4});
    });
    assert.equal(result.success, false); assert.equal(score.state.hasTakenOff, false);
    assert(score.state.bestFlightSeconds < criteria.takeoff.minimumFlightSeconds);
  }
});

test('sustained rotation rejects oscillations without signed-vector cancellation', () => {
  const {score, result} = run('flight', (t, k) => ({...hover(t), angularSpeed: t < .1 ? 0 : Math.abs(k % 2 ? -80 : 80)}));
  assert.equal(result.success, false); assert.equal(score.state.bestFlightSeconds, 0);
  assert.equal(score.state.diagnostics.maximumAngularSpeed, 80);
});

const climbAfter = (t, start) => {
  const climbing = t < start + .4 - 1e-9;
  return air({height: 1 + .8 * Math.max(0, Math.min(.4, t - start)),
    verticalSpeed: climbing ? .8 : 0, speedCmPerSecond: climbing ? .8 : 0});
};

test('a foot-origin departure can establish powered upward motion after its first airborne sample', () => {
  for (const dt of [.002, .01]) for (const initialWingPower of [0, .8]) {
    const {score, result, frames} = run('flight', t => t < .1 - 1e-9 ? ground() : t < .14 - 1e-9 ?
      air({height: 1, verticalSpeed: .1, speedCmPerSecond: .1, wingPower: initialWingPower}) : climbAfter(t, .14), {dt});
    assert.equal(result.success, true, `Delayed powered ascent must qualify at dt=${dt}, release power=${initialWingPower}`);
    assert.equal(score.state.hasTakenOff, true);
    assert(frames.filter(frame => frame.t < .14 - 1e-9).every(frame => frame.flightSeconds === 0), 'Waiting for powered ascent earns no qualified flight');
  }
});

test('foot-only recontact starts a new flight bout while retaining a fresh stance origin', () => {
  for (const dt of [.002, .01]) {
    const {score, result, frames} = run('flight', t => t < .1 - 1e-9 ? ground() : t < .2 - 1e-9 ? climbAfter(t, .1) :
      t < .22 - 1e-9 ? ground({environmentContacts: 1, footSupportCount: 1, footSupportFraction: .8, wingPower: .8}) :
      t < .26 - 1e-9 ? air({height: 1, verticalSpeed: .1, speedCmPerSecond: .1}) : climbAfter(t, .26), {dt});
    assert.equal(result.success, true); assert.equal(score.state.hasTakenOff, true);
    assert(frames.some(frame => frame.t < .2 && frame.flightSeconds > 0), 'First bout had measurable airborne progress');
    assert(frames.filter(frame => frame.t >= .2 - 1e-9 && frame.t < .26 - 1e-9).every(frame => frame.flightSeconds === 0),
      'Recontact and a new release must not inherit the previous bout airtime');
  }
});

test('brief unloaded foot contact may release again from recent measured stance', () => {
  const {score, result} = run('flight', t => t < .1 - 1e-9 ? ground() : t < .13 - 1e-9 ? climbAfter(t, .1) :
    t < .15 - 1e-9 ? ground({environmentContacts: 1, footSupportCount: 0, footSupportFraction: 0, wingPower: .8}) :
    t < .18 - 1e-9 ? air({height: 1, verticalSpeed: .1, speedCmPerSecond: .1}) : climbAfter(t, .18), {dt: .002});
  assert.equal(result.success, true); assert.equal(score.state.hasTakenOff, true);
});

test('a body impact invalidates earlier foot support until a fresh loaded stance is observed', () => {
  const bodyContact = ground({environmentContacts: 1, nonFootEnvironmentContacts: 1, footSupportCount: 0, footSupportFraction: 0, wingPower: .8});
  for (const freshStance of [false, true]) {
    const {score, result} = run('flight', t => t < .1 - 1e-9 ? ground() : t < .14 - 1e-9 ? climbAfter(t, .1) :
      t < .16 - 1e-9 ? bodyContact : freshStance && t < .2 - 1e-9 ? ground() : climbAfter(t, freshStance ? .2 : .16), {dt: .002});
    assert.equal(result.success, freshStance); assert.equal(score.state.hasTakenOff, freshStance);
    if (!freshStance) assert.equal(score.state.bestFlightSeconds, 0, 'A body collision cannot reuse the pre-impact stance token');
  }
});

test('upward motion without wings followed only by level wing activity does not establish powered ascent', () => {
  const {score, result} = run('flight', t => t < .1 - 1e-9 ? ground() : t < .14 - 1e-9 ?
    air({wingPower: 0, height: 1 + 8 * (t - .1), verticalSpeed: 8, speedCmPerSecond: 8}) : air(), {dt: .002});
  assert.equal(result.success, false); assert.equal(score.state.hasTakenOff, false);
});

test('bounded intermittent wingbeat recoil uses scalar RMS instead of resetting flight at each modest peak', () => {
  for (const dt of [.002, .01]) {
    // Each complete 20 ms window has RMS sqrt((24² + 8²) / 2), below 20.
    // Neither a signed vector average nor an increased speed limit is needed.
    const {score, result} = run('flight', (t, k) => ({...hover(t), angularSpeed: t < .1 - 1e-9 ? 0 : k % 2 ? 24 : 8}), {dt});
    assert.equal(result.success, true); assert.equal(score.state.hasTakenOff, true);
    assert.equal(score.state.diagnostics.maximumAngularSpeed, 24);
  }
});

test('RMS rotation still rejects persistent excessive motion and large alternating directions', () => {
  for (const magnitude of [21, 80]) {
    const {score, result} = run('flight', (t, k) => ({...hover(t), angularSpeed: t < .1 ? 0 : Math.abs(k % 2 ? -magnitude : magnitude)}), {dt: .002});
    assert.equal(result.success, false); assert.equal(score.state.bestFlightSeconds, 0);
  }
});

test('frozen native foot-release trajectory receives takeoff credit while its later crash still fails', () => {
  const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/flight-native-foot-release-crash.json', import.meta.url)));
  assert.equal(fixture.kind, 'frozen-native-training-observations');
  assert.equal(fixture.captureProvenance.backend, 'wasm');
  assert.equal(fixture.captureProvenance.bodyBackend, 'mujoco-wasm');
  assert.equal(fixture.diagnosticVariant.clawScale, 0, 'Fixture is explicitly the diagnostic adhesion variant');
  assert.equal(fixture.baselineGate.passed, true); assert.equal(fixture.baselineGate.details.exact, true);
  assert.equal(fixture.baselineGate.details.checkedSteps, 195);
  assert.equal(fixture.observationGate.passed, true); assert.equal(fixture.observationGate.checkedInitial, true);
  assert.equal(fixture.observationGate.checkedPreviewFrames, 36);
  assert.equal(createHash('sha256').update(JSON.stringify({initial: fixture.initial, steps: fixture.steps})).digest('hex'),
    fixture.observationPayloadSha256, 'Raw native observation payload must stay tied to its provenance');
  assert.equal(fixture.steps.length, 195);
  const score = createFlightScore('landing', 8, fixture.initial);
  let result, firstConfirmation = null;
  for (const [index, step] of fixture.steps.entries()) {
    assert.equal(step.index, index); assert.equal(step.dt, .002);
    assert(Math.abs(step.time - (index + 1) * step.dt) < 1e-9);
    result = score.step(step.observation, step.dt);
    if (score.state.hasTakenOff && firstConfirmation === null) firstConfirmation = step.time;
    if (index < fixture.steps.length - 1) assert.equal(result.terminated, false, 'The measured late crash must remain the terminal event');
    assert.equal(result.success, false, 'Partial airborne progress is never full-horizon flight or landing success');
  }
  assert(firstConfirmation > .2 && firstConfirmation < .32, 'Measured powered ascent is eventually confirmed');
  assert.equal(score.state.hasTakenOff, true);
  assert(Math.abs(score.state.takeoffTime - .08) < 1e-9);
  assert(Math.abs(score.state.bestFlightSeconds - .184) < 1e-9);
  assert.equal(score.state.flightSeconds, 0); assert.equal(score.state.landingTime, null);
  assert(Math.abs(score.state.elapsed - .39) < 1e-9);
  assert.deepEqual(result, {terminated: true, success: false, reason: 'excessive_rotation'});
  assert(score.state.diagnostics.maximumAngularSpeed > 500, 'The catastrophic instantaneous-speed guard remains active');
  assert(score.state.return < 0 && score.state.return > -2, 'The failed flight earns partial behavioral credit without becoming a successful episode');
});

test('landing joins takeoff, sustained powered flight, gentle approach and final loaded-foot settlement', () => {
  const {score, result, frames} = run('landing', land);
  assert.equal(result.success, true); assert.equal(score.state.elapsed, 8);
  assert(score.state.hasTakenOff); assert(score.state.bestFlightSeconds > 2);
  assert.equal(score.state.phase, 'landed'); assert(score.state.landingSeconds > 5);
  assert(Math.abs(score.state.landingTime - 2.4) < 1e-8);
  assert(frames.slice(0, -1).every(frame => !frame.result.success && !frame.result.terminated));
  assert(frames.find(frame => frame.t === 1).reward > 0, 'Landing includes dense real flight progress');
  assert(frames.find(frame => frame.t === 2.5).reward > frames.find(frame => frame.t === 2).reward, 'Settlement increases progress');
});

test('landing permits a short progressive foot settlement', () => {
  const {score, result} = run('landing', t => t < 2.4 - 1e-9 ? land(t) : t < 2.46 - 1e-9 ?
    ground({environmentContacts: 1, footSupportCount: 1, footSupportFraction: .2, speedCmPerSecond: .4}) : ground());
  assert.equal(result.success, true); assert(score.state.landingSeconds > 5);
});

test('wing/body and mixed wing-plus-foot collisions are not landings', () => {
  for (const contact of [
    ground({environmentContacts: 2, nonFootEnvironmentContacts: 2, footSupportCount: 0, footSupportFraction: 0}),
    ground({environmentContacts: 7, nonFootEnvironmentContacts: 1}),
  ]) {
    const {score, result} = run('landing', t => t < 2.4 - 1e-9 ? land(t) : contact);
    assert.equal(result.success, false); assert.equal(score.state.landingTime, null);
    assert.equal(score.state.landingSeconds, 0); assert(score.state.return <= 0);
  }
});

test('touchdown uses preimpact history, rejecting high speed erased by impact or a single slow endpoint', () => {
  for (const slowFinalFrame of [false, true]) {
    const {score, result} = run('landing', t => {
      if (t < 2.3 - 1e-9) return hover(t);
      if (t < 2.4 - 1e-9) return air({verticalSpeed: slowFinalFrame && t >= 2.39 - 1e-9 ? -.5 : -4,
        speedCmPerSecond: slowFinalFrame && t >= 2.39 - 1e-9 ? .5 : 4});
      return ground();
    });
    assert.equal(result.success, false); assert.equal(score.state.landingTime, null);
    assert.equal(score.state.diagnostics.rejectedTouchdowns, 1);
  }
  // A fast horizontal approach does not break the vertical flight support
  // test, so this specifically exercises the preimpact-speed gate.
  const horizontal = run('landing', t => t < 2.3 ? hover(t) : t < 2.4 ?
    air({speedCmPerSecond: t >= 2.39 ? .5 : 4}) : ground());
  assert.equal(horizontal.result.success, false); assert.equal(horizontal.score.state.landingTime, null);
});

test('hovering without touching down, or touching down after insufficient flight, cannot complete landing', () => {
  assert.equal(run('landing', hover).result.success, false);
  const short = run('landing', t => t < .7 ? hover(t) : ground());
  assert.equal(short.result.success, false); assert.equal(short.score.state.landingTime, null);
});

test('late contact or degraded flight clears current flight reward instead of retaining a peak', () => {
  const touched = run('flight', t => t < 4.5 ? hover(t) : ground());
  assert.equal(touched.result.success, false); assert.equal(touched.score.state.flightSeconds, 0);
  assert(touched.score.state.bestFlightSeconds > 4); assert(touched.score.state.return <= 0);
  const endpoint = run('flight', t => t >= 4.99 - 1e-9 ? air({angularSpeed: 80}) : hover(t));
  assert.equal(endpoint.result.success, false); assert.equal(endpoint.score.state.flightSeconds, 0);
  assert(endpoint.score.state.return <= .2, 'Only capped tentative airborne credit may remain');
});

test('landing must remain stable through the horizon, not only at an earlier chosen endpoint', () => {
  const {score, result} = run('landing', t => t < 7.8 ? land(t) : ground({angularSpeed: 10}));
  assert.equal(result.success, false); assert.equal(score.state.landingTime, null);
  assert.equal(score.state.landingSeconds, 0); assert(score.state.return <= 0);
});

test('physical failures override prior flight or landing accomplishments', () => {
  for (const stage of ['flight', 'landing']) {
    const duration = criteria.minimumDurationSeconds[stage], trajectory = stage === 'flight' ? hover : land;
    for (const [changes, reason] of [
      [{externalForce: true}, 'unexpected_external_force'],
      [{radius: 7}, 'outside_habitat'], [{height: -.6}, 'outside_habitat'],
      [{height: 7}, 'outside_habitat'], [{angularSpeed: 301}, 'excessive_rotation'],
      [{up: -1}, 'overturned'],
    ]) {
      const {score, result} = run(stage, t => t < duration - .2 ? trajectory(t) : {...trajectory(t), ...changes});
      assert.equal(result.success, false); assert.equal(result.reason, reason);
      assert.equal(result.terminated, true);
      if (reason === 'unexpected_external_force') assert.equal(score.state.return, -10);
      else assert(score.state.return >= -8 && score.state.return <= 3, 'Physical failure retains only bounded task progress');
      assert(score.state.elapsed < duration);
    }
  }
});

test('failed attempts retain informative progress with a common physical-failure penalty', () => {
  const immediate = run('flight', () => ground({angularSpeed: 600}));
  const controlled = run('flight', t => t < 2 ? hover(t) : air({angularSpeed: 600}));
  const gentleFailure = run('flight', t => t < 2 ? hover(t) : air({radius: 7}));
  const completed = run('flight', hover);
  for (const trial of [immediate, controlled, gentleFailure]) assert.equal(trial.result.success, false);
  assert(controlled.score.state.return > immediate.score.state.return);
  assert.equal(gentleFailure.score.state.return, controlled.score.state.return);
  assert(completed.score.state.return > gentleFailure.score.state.return);
  assert(controlled.score.state.diagnostics.bestEarnedProgress > 0);
  assert.equal(immediate.score.state.diagnostics.bestEarnedProgress, 0);
  for (const trial of [immediate, controlled, gentleFailure]) assert.equal(trial.score.state.diagnostics.failurePenalty, 3);
});

test('identical flight progress earns identical failure returns despite different final crash severity or cause', () => {
  for (const stage of ['takeoff', 'flight', 'landing']) {
    const trials = [
      {angularSpeed: 301},
      {angularSpeed: 10000, up: -1, environmentContacts: 2, nonFootEnvironmentContacts: 2},
      {radius: 7},
      {height: -.6},
    ].map(crash => run(stage, t => t < .7 - 1e-9 ? hover(t) : air(crash), {dt: .002}));
    const first = trials[0].score.state;
    assert(first.hasTakenOff); assert(first.bestFlightSeconds > .15);
    for (const trial of trials) {
      assert.equal(trial.result.terminated, true); assert.equal(trial.result.success, false);
      assert.equal(trial.score.state.elapsed, first.elapsed);
      assert.equal(trial.score.state.bestFlightSeconds, first.bestFlightSeconds);
      assert.equal(trial.score.state.diagnostics.bestEarnedProgress, first.diagnostics.bestEarnedProgress);
      assert.equal(trial.score.state.diagnostics.failurePenalty, 3);
      assert.equal(trial.score.state.return, first.return);
    }
    assert(trials[1].score.state.diagnostics.maximumAngularSpeed > first.diagnostics.maximumAngularSpeed,
      'Crash severity must still be observable without influencing the task reward');
  }
});

test('20 ms more qualified flight beats a softer final crash', () => {
  const shorter = run('flight', t => t < .6 - 1e-9 ? hover(t) : air({angularSpeed: 301}), {dt: .002});
  const longer = run('flight', t => t < .62 - 1e-9 ? hover(t) : air({angularSpeed: 10000, up: -1}), {dt: .002});
  for (const trial of [shorter, longer]) {
    assert.equal(trial.result.reason, 'excessive_rotation'); assert.equal(trial.result.success, false);
    assert.equal(trial.score.state.hasTakenOff, true);
  }
  assert(Math.abs(longer.score.state.bestFlightSeconds - shorter.score.state.bestFlightSeconds - .02) < 1e-9);
  assert(longer.score.state.return > shorter.score.state.return);
  assert(Math.abs((longer.score.state.return - shorter.score.state.return) -
    (longer.score.state.diagnostics.bestEarnedProgress - shorter.score.state.diagnostics.bestEarnedProgress)) < 1e-12);
});

test('early powered motion has bounded dense credit before the strict support window', () => {
  const controlled = createFlightScore('takeoff', 3, ground());
  const ballistic = createFlightScore('takeoff', 3, ground());
  for (let k = 1; k <= 4; k++) {
    const t = k * .01;
    controlled.step(air({height: 1 + .8 * t, verticalSpeed: .8, speedCmPerSecond: .8}), .01);
    const vz = 100 - criteria.gravityCmPerSecondSquared * (t - .01);
    ballistic.step(air({height: 1 + 100 * (t - .01) - .5 * criteria.gravityCmPerSecondSquared * (t - .01) ** 2,
      verticalSpeed: vz, speedCmPerSecond: Math.abs(vz)}), .01);
  }
  assert.equal(controlled.state.flightSeconds, 0); assert.equal(ballistic.state.flightSeconds, 0);
  assert(controlled.state.return > ballistic.state.return);
  assert(controlled.state.diagnostics.tentativeProgress > 0 && controlled.state.diagnostics.tentativeProgress <= .2);
  assert(ballistic.state.diagnostics.tentativeProgress <= .004 + 1e-9, 'No credit for the launch impulse or following ballistic acceleration');
  assert.equal(controlled.state.hasTakenOff, false); assert.equal(ballistic.state.hasTakenOff, false);
});

test('malformed measurements cannot qualify or silently count as zero motion', () => {
  for (const changes of [
    {finite: false}, {angularSpeed: NaN}, {speedCmPerSecond: Infinity},
    {nonFootEnvironmentContacts: undefined}, {environmentContacts: -1},
    {footSupportCount: 2.5}, {footSupportCount: 0}, {up: 2},
    {wingPower: 2}, {externalForce: 0}, {speedCmPerSecond: 0, verticalSpeed: 1},
  ]) {
    const score = createFlightScore('takeoff', 3, ground());
    assert.deepEqual(score.step(ground(changes), .01), {terminated: true, success: false, reason: 'invalid_observation'});
    assert.equal(score.state.return, -10);
  }
  assert.throws(() => createFlightScore('takeoff', 3, ground({finite: false})), /Invalid/);
  const score = createFlightScore('takeoff', 3, ground());
  for (const dt of [0, -.01, NaN, .1]) assert.throws(() => score.step(ground(), dt), /timestep/);
});

test('2 ms and 10 ms observation cadences agree on behavior and near-identical progress', () => {
  for (const stage of ['takeoff', 'flight', 'landing']) {
    const trajectory = stage === 'landing' ? land : hover;
    const fine = run(stage, trajectory, {dt: .002}), coarse = run(stage, trajectory, {dt: .01});
    assert.deepEqual(fine.result, coarse.result); assert.equal(fine.result.success, true);
    assert(Math.abs(fine.score.state.return - coarse.score.state.return) < .011);
    assert(Math.abs(fine.score.state.bestFlightSeconds - coarse.score.state.bestFlightSeconds) < .011);
    assert(Math.abs(fine.score.state.takeoffTime - coarse.score.state.takeoffTime) < .011);
  }
});

test('finished successful and unsuccessful scores are immutable and cannot be revived', () => {
  for (const trajectory of [hover, () => ground()]) {
    const {score, result} = run('flight', trajectory), saved = JSON.stringify(score.state);
    assert(Object.isFrozen(score.state)); assert(Object.isFrozen(score.state.diagnostics)); assert(Object.isFrozen(result));
    assert.strictEqual(score.step(ground({finite: false}), -10), result);
    assert.equal(JSON.stringify(score.state), saved);
    assert.throws(() => { score.state.success = !score.state.success; }, TypeError);
  }
});
