import test from 'node:test';
import assert from 'node:assert/strict';
import {BEHAVIOR_CRITERIA_V2 as criteria, createBehaviorScoreV2, normalizeBehaviorObservationV2} from '../training/behavior-criteria-v2.js';

const observation = changes => ({up: 1, angularVelocity: [0, 0, 0], speedCmPerSecond: 0, verticalSpeed: 0,
  height: 1, radius: 1, ceiling: 5, distance: 1, intake: 0, wingPower: 0,
  environmentContacts: 4, footSupportCount: 4, footSupportFraction: 1, footFoodContacts: 0,
  airborne: false, externalForce: false, probing: false, onFood: false, localized: false, ...changes});
const airborne = changes => observation({environmentContacts: 0, footSupportCount: 0, footSupportFraction: 0,
  footFoodContacts: 0, airborne: true, wingPower: .8, verticalSpeed: .2, speedCmPerSecond: .2, ...changes});
const food = changes => observation({onFood: true, footFoodContacts: 4, distance: .1, ...changes});
const drive = (score, seconds, factory) => {
  let result;
  for (let n = 0; n < Math.round(seconds / .02); n++) result = score.step(typeof factory === 'function' ? factory(n) : factory, .02);
  return result;
};
const takeoff = score => {
  drive(score, .1, observation());
  score.step(airborne({verticalSpeed: .6, speedCmPerSecond: .6}), .02);
};

test('v2 declares research thresholds and rejects shortened stages', () => {
  assert.match(criteria.interpretation, /not measured biological/);
  assert.match(criteria.localizationEvidence, /proxy/);
  for (const [stage, duration] of Object.entries(criteria.minimumDurationSeconds))
    assert.throws(() => createBehaviorScoreV2(stage, duration - .1, observation()), /at least/);
  assert.throws(() => createBehaviorScoreV2('toString', 2, observation()), /Unknown/);
});

test('normalization requires measured foot support and finite instantaneous rotation', () => {
  assert.throws(() => normalizeBehaviorObservationV2(observation({footSupportFraction: undefined})), /footSupportFraction/);
  assert.throws(() => normalizeBehaviorObservationV2(observation({angularVelocity: [NaN, 0, 0]})), /angularVelocity/);
  assert.throws(() => normalizeBehaviorObservationV2(airborne({footSupportCount: 2})), /loaded feet/);
  assert.equal(normalizeBehaviorObservationV2(observation({angularVelocity: [-3, 4, 0]})).angularSpeed, 5);
});

test('an undisturbed initial stance cannot earn posture recovery', () => {
  const score = createBehaviorScoreV2('posture', 2, observation());
  const result = drive(score, 2, observation());
  assert.equal(result.success, false);
  assert.equal(result.reason, 'insufficient_disturbance');
  assert(score.state.return <= 0);
});

test('posture requires recovery from a real initial disturbance and final foot stability', () => {
  const disturbed = observation({up: Math.cos(40 * Math.PI / 180), angularVelocity: [9, 0, 0]});
  const score = createBehaviorScoreV2('posture', 2, disturbed);
  drive(score, .3, disturbed);
  assert.equal(drive(score, 1.7, observation()).success, true);
  assert(score.state.recoveryTime <= criteria.posture.recoveryDeadlineSeconds);
  assert(score.state.stableSeconds >= criteria.posture.finalStableSeconds);
  const delayed = createBehaviorScoreV2('posture', 2, disturbed);
  drive(delayed, .1, observation());
  drive(delayed, 1.02, disturbed);
  assert.equal(drive(delayed, .88, observation()).success, false, 'A brief early stable frame cannot hide late recovery');
});

test('body or wing contact cannot substitute for loaded feet in posture', () => {
  const score = createBehaviorScoreV2('posture', 2, observation({angularVelocity: [9, 0, 0]}));
  const collision = observation({environmentContacts: 2, footSupportCount: 0, footSupportFraction: 0});
  assert.equal(drive(score, 2, collision).success, false);
  assert.equal(score.state.supportedSeconds, 0);
});

test('approach requires substantial continuous progress, not a distance jump followed by tiny movements', () => {
  const incidental = createBehaviorScoreV2('approach', 3, observation());
  incidental.step(food(), .02);
  drive(incidental, .02, food());
  drive(incidental, .4, i => food({distance: .1 - (i + 1) * .0001}));
  assert.equal(drive(incidental, 2.56, food({distance: .098})).success, false);
  const approached = createBehaviorScoreV2('approach', 3, observation());
  drive(approached, .6, i => observation({distance: 1 - (i + 1) * .01, speedCmPerSecond: .5}));
  assert.equal(drive(approached, 2.4, food({distance: .7})).success, true);
});

test('alternating signed rotations do not cancel into qualified flight', () => {
  const score = createBehaviorScoreV2('flight', 2, observation()); takeoff(score);
  const result = drive(score, 1.88, i => airborne({angularVelocity: [i % 2 ? -80 : 80, 0, 0]}));
  assert.equal(result.success, false);
  assert.equal(score.state.hasTakenOff, true);
  assert(Math.abs(score.state.flight.angularRms - 80) < 1e-8);
});

test('a transient airborne bout or a body-collision launch is not flight', () => {
  const transient = createBehaviorScoreV2('flight', 2, observation()); takeoff(transient);
  drive(transient, .6, airborne());
  assert.equal(drive(transient, 1.28, observation()).success, false);
  const collision = observation({environmentContacts: 2, footSupportCount: 0, footSupportFraction: 0});
  const launched = createBehaviorScoreV2('flight', 2, collision);
  drive(launched, .1, collision);
  assert.equal(drive(launched, 1.9, airborne({verticalSpeed: .6, speedCmPerSecond: .6})).success, false);
  assert.equal(launched.state.hasTakenOff, false);
});

test('powered ballistic descent is rejected by kinematic support and descent criteria', () => {
  const score = createBehaviorScoreV2('flight', 2, observation()); takeoff(score);
  drive(score, 1.88, i => {
    const velocity = .6 - criteria.gravityCmPerSecondSquared * (i + 1) * .02;
    return airborne({verticalSpeed: velocity, speedCmPerSecond: Math.abs(velocity)});
  });
  assert.equal(score.state.success, false);
  assert(Math.abs(score.state.flight.supportFraction) < 1e-8);
});

test('a valid takeoff and sustained powered upright flight must survive the full episode', () => {
  const score = createBehaviorScoreV2('flight', 2, observation()); takeoff(score);
  assert.equal(drive(score, 1.5, airborne()).terminated, false);
  assert.equal(drive(score, .38, airborne()).success, true);
  assert.equal(score.state.elapsed, 2);
  const lateTilt = createBehaviorScoreV2('flight', 2, observation()); takeoff(lateTilt);
  drive(lateTilt, 1.5, airborne());
  assert.equal(drive(lateTilt, .38, airborne({up: .4})).success, false);
});

test('takeoff can unload feet progressively without losing recent support evidence', () => {
  const score = createBehaviorScoreV2('flight', 2, observation());
  drive(score, .1, observation());
  drive(score, .04, observation({footSupportCount: 1, footSupportFraction: .15, environmentContacts: 1}));
  score.step(airborne({verticalSpeed: .6, speedCmPerSecond: .6}), .02);
  assert.equal(drive(score, 1.84, airborne()).success, true);
});

test('landing requires prior flight, a slow approach, and sustained support by feet on food', () => {
  const score = createBehaviorScoreV2('landing', 3, airborne());
  drive(score, 1.2, airborne());
  assert.equal(drive(score, 1.8, food()).success, true);
  const preplaced = createBehaviorScoreV2('landing', 3, food());
  assert.equal(drive(preplaced, 3, food()).success, false);
  const impact = createBehaviorScoreV2('landing', 3, airborne());
  drive(impact, 1.2, airborne());
  assert.equal(drive(impact, 1.8, food({footSupportCount: 0, footSupportFraction: 0, footFoodContacts: 0})).success, false);
  const staleFlight = createBehaviorScoreV2('landing', 3, airborne());
  drive(staleFlight, 1.2, airborne());
  drive(staleFlight, .4, airborne({angularVelocity: [80, 0, 0]}));
  assert.equal(drive(staleFlight, 1.4, food()).success, false, 'Landing needs a currently qualified approach, not a flight window from earlier in the episode');
  const progressive = createBehaviorScoreV2('landing', 3, airborne());
  drive(progressive, 1.2, airborne());
  drive(progressive, .06, food({footSupportCount: 1, footSupportFraction: .2, footFoodContacts: 1, environmentContacts: 1}));
  assert.equal(drive(progressive, 1.74, food()).success, true, 'A physical landing may touch one foot before establishing full support');
});

test('flight RMS is time weighted across uneven physics observations', () => {
  const score = createBehaviorScoreV2('flight', 2, observation()); takeoff(score);
  // Over the final second: 3/4 of the time at 4 rad/s and 1/4 at 12 rad/s.
  // Expected RMS sqrt(.75*16 + .25*144), not the unweighted sample RMS.
  drive(score, .88, airborne());
  for (let n = 0; n < 25; n++) {
    score.step(airborne({angularVelocity: [4, 0, 0]}), .03);
    score.step(airborne({angularVelocity: [-12, 0, 0]}), .01);
  }
  assert(Math.abs(score.state.flight.angularRms - Math.sqrt(48)) < 1e-8);
  assert.equal(score.state.success, true);
});

test('feeding requires new intake during actual probing, not a prior full crop or an open proboscis', () => {
  const noIntake = createBehaviorScoreV2('feeding', 3, food({intake: .5}));
  assert.equal(drive(noIntake, 3, food({intake: .5, probing: true})).success, false);
  const noProbe = createBehaviorScoreV2('feeding', 3, food());
  assert.equal(drive(noProbe, 3, i => food({intake: (i + 1) * .00001})).success, false);
  const fed = createBehaviorScoreV2('feeding', 3, food());
  assert.equal(drive(fed, 3, i => food({probing: true, intake: (i + 1) * .00001})).success, true);
});

test('the whole sequence requires each observation in order, including probing', () => {
  const score = createBehaviorScoreV2('sequence', 8, airborne());
  drive(score, .2, airborne({localized: true}));
  drive(score, .4, i => airborne({localized: true, distance: 1 - (i + 1) * .025, speedCmPerSecond: 1.25}));
  drive(score, .7, airborne({localized: true, distance: .5}));
  drive(score, .6, food({localized: true}));
  drive(score, .2, food({localized: true, probing: true}));
  drive(score, .3, i => food({localized: true, probing: true, intake: (i + 1) * .0001}));
  drive(score, .1, food({localized: true, probing: true, intake: .0015}));
  score.step(airborne({localized: true, distance: .1, intake: .0015, verticalSpeed: .6, speedCmPerSecond: .6}), .02);
  const result = drive(score, 5.48, airborne({localized: true, distance: .1, intake: .0015}));
  assert.equal(result.success, true);
  assert.deepEqual(score.state.events.filter(e => e.stage.startsWith('sequence:')).map(e => e.stage.slice(9)), criteria.sequence);
  assert.match(score.state.events.find(e => e.stage === 'sequence:localization').evidence, /proxy/);
  const outOfOrder = createBehaviorScoreV2('sequence', 8, food());
  drive(outOfOrder, 1, i => food({probing: true, intake: (i + 1) * .0001}));
  assert.equal(drive(outOfOrder, 7, airborne({localized: true, intake: .005})).success, false);
  assert(outOfOrder.state.sequenceIndex < criteria.sequence.length);
});

test('invalid observations, external forces and decreasing intake fail closed with bounded returns', () => {
  for (const [changes, reason] of [[{up: NaN}, 'invalid_observation'], [{externalForce: true}, 'unexpected_external_force'], [{intake: .1}, 'nonmonotonic_intake']]) {
    const score = createBehaviorScoreV2('feeding', 3, food({intake: .2}));
    const result = score.step(food(changes), .02);
    assert.equal(result.reason, reason);
    assert.equal(result.terminated, true);
    assert.equal(result.success, false);
    assert(score.state.return >= -10 && score.state.return <= 10);
    assert.deepEqual(score.step(food({intake: 1, probing: true}), .02), result);
  }
});
