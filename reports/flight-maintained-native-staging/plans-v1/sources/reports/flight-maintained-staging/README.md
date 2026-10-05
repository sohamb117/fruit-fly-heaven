# Maintained flight from an explicit airborne reset — staged only

This separates **maintaining powered flight from performing a cold takeoff**. It does not repair the failed combined affine candidate, establish a successful live BANC rollout, or train anything yet. No live source, native simulation, server or optimizer was changed/run by this task.

Current `takeoff` (3 s), `flight` (5 s), and `landing` (8 s) all use the same grounded stance initializer. Even `flight` requires a foot-origin departure. Existing tests explicitly reject initially airborne hovering/falling/landing. The coordinator can hold a configured stage but does not automatically promote a curriculum. Relabeling an airborne reset as the old `flight` stage would be incorrect.

## Pure staged objective

`maintained-flight-objective.mjs` exports:

- `MAINTAINED_FLIGHT_STAGE = 'maintained_flight'`
- `MAINTAINED_FLIGHT_CRITERIA`
- `createMaintainedFlightScore(duration, initialObservation)`

The horizon is exactly **five scored seconds**. The initial observation must have no environment contacts. It uses the current modeled flight criteria: power >0.1; up ≥cos(45°); scalar angular-speed RMS ≤20 rad/s over 20 ms; vertical COM speed ≥−1 cm/s; and inferred non-gravitational vertical support ≥0.8 body weights over 50 ms. Existing catastrophic motion, habitat-boundary and external-force failures remain. These are modeling thresholds, not measured physiological limits.

The support and angular windows begin empty at release. Continuous qualified airtime starts only after the required windows fill. Contact or any qualification failure resets the current bout; contact also clears the airborne support window. A later departure can qualify a new bout, but no takeoff or landing credit is assigned. `hasTakenOff` remains false, `takeoffTime`/`landingTime` remain null, and confirmed takeoffs remain zero. Standing, unpowered hovering observations, ballistic motion and falling cannot complete the objective.

Success is assessed only at the full five-second horizon and requires the **current** qualified continuous bout to meet the existing one-second minimum. This is a five-second evaluation; it does not claim that every one of those seconds was qualified flight. Report current, best and total qualified airtime separately. The reward is `7 * currentQualifiedSeconds / 5`, bounded to [0,7], with a +3 full-horizon success bonus. Physical failures use the existing bounded-progress-minus-3 convention; invalid observations/external-force injection return −10. There is no reset, posture, takeoff or landing reward.

## Exact initialization contract

`airborne-reset-contract.mjs` exports `selectFlightInitialCondition(stage, value, context)` and `createMaintainedFlightClock(initialization, context)`. The proposed config field is:

```js
initialCondition: {
  schemaVersion: 1,
  profile: 'airborne-live-warmup-v1',
  warmupSeconds: 0.5,
  bodyVariant: 'full-native',
  bodyVariantHash: '<verified /body-model/flybody-mujoco.json SHA-256>',
  rootQpos: [0, 0, 3.5, 1, 0, 0, 0] // cm, then quaternion w/x/y/z
}
```

Pass `context = {bodyMetadataSha256: config.assets['/body-model/flybody-mujoco.json']}` only after normal asset verification. The metadata pins its XML through `xml_sha256`. The selector owns/freezes the finite root pose and requires an already normalized quaternion. Warm-up must be explicitly chosen, within **0.1–2 s inclusive** and a whole 2 ms block, matching the native reset helper. The profile implies zero root translational/angular velocity while held. Pose clearance, actual model identity and reset geometry must be checked by the native helper; the pure selector cannot certify them.

Only the current **full-native** body is supported. For any old stage, an absent selector returns null so the existing branch can remain unchanged; an airborne selector is rejected. A fixed-nonwing variant is not accepted merely because a string/hash was supplied. It would require a real prepared native model, correctly regenerated mappings/observations and its own validation. Repeatedly overwriting nonwing joints during the scored interval is not equivalent to compiling a reduced body.

## Warm-up and clock handoff

The parent is separately implementing a native root hold during **unscored setup only**, at the declared root pose before/after each 50 µs native step. Nonwing joints remain free. This is an explicit setup restraint, not evidence of flight. The same live BANC instance must advance through every warm-up tick and deliver newly generated 48-MN events through the existing causal 1 ms muscle handoff. No recorded event stream is replayed or looped.

Keep the complete live state continuous: neural membrane/adaptation/refractory state, delayed spike history, receptor kinetics, DLM ionic gates/counts/timestamps; event-reader history and adapter kernels; native activation/fatigue/force of all muscles; wing phase/deployment/targets/power; native qpos/qvel/actuator state; internal energy/hunger/hormones; and actual body/sensory feedback. Retain the same running objects. A motor-rate snapshot alone is not a restorable brain state, and the runtime currently provides no complete supported brain snapshot/restore interface.

At the declared warm-up endpoint, finish the current 2 ms block, remove the root hold, establish one release record and create the fresh scorer. Do not reset brain/native/event clocks, reset muscles, force an output rate, set `hasTakenOff`, grant accumulated warm-up airtime, or reposition the fly during scoring. At 0.5 s warm-up, the scored interval is absolute native/neural time 0.5–5.5 s and relative score time 0–5 s. `FlyBodyPhysics.place()` already rejects repositioning after event/load timing has begun; bypassing that guard is not a normal episode step.

The clock receives:

```js
{
  nativeTimeSeconds, neuralTimeMs,
  bodyEventElapsedMs, bodyEventObservedMs, lastPacketTimeMs,
  remainderSeconds, pendingEvents,
  rootRestraintActive, externalForceApplied, rootWriteCount
}
```

`clock.release(snapshot)` is called once; it requires the exact declared warm-up endpoint, synchronized absolute clocks, consumed event interval, zero remainder, no pending events, no active root restraint and no applied force. It returns the absolute release times and scored time/steps zero. `clock.advance(snapshot)` validates each subsequent 2 ms step, rejects missing/repeated blocks and any increased direct-root-write count, and returns `scoredElapsedSeconds`, `scoredSteps`, `nativeTimeSeconds`, `neuralTimeMs`. It stops accepting after 2,500 scored blocks.

The write count/constraint flags must come from the native helper's independent audit. This pure helper cannot detect a falsely reported counter or a hidden unreported restraint. Validate the adapter's actual pending state once at release. The pinned adapter guarantees pending=null when integrated and observed time match, allowing the existing cheap event/body guards thereafter instead of cloning all kernel histories per block.

## Integration boundary and validation

Environment integration is staged separately. It must dispatch the new scorer/selector only for `maintained_flight`, run real sensory → neural → event → muscle → body blocks during warm-up and scoring, and report both absolute and release-relative times. `onInitialState`/motor-replay observers currently assume a zero-time baseline; do not silently label a warmed state as time zero or reuse those capture schemas unchanged. Scored observations may use the existing COM/contact measurement functions. Setup failures must be reported, not silently resampled until a favorable state appears.

The original three scorer stages and live source bytes are untouched. Their grounded task requirements remain essential for the eventual takeoff → flight → landing sequence. A future stage should use a separate config/model fingerprint/checkpoint identity; an airborne-stage success is conditional performance from its declared warm state.

Reproduce the pure checks with:

```sh
node --test reports/flight-maintained-staging/maintained-flight.test.mjs web/test/training-flight-objective.test.mjs
```

**46/46 pass:** 13 new maintained-flight/reset tests plus all 33 existing objective tests. Tests cover powered hover, freefall, ballistic motion, powerless motion, all flight qualification gates, contact/reset/requalification, short-bout nonaccumulation, full-horizon and crash semantics, malformed measurements, old-stage behavior, metadata/pose ownership, the inclusive 0.1–2 s warm-up bounds and 2 ms grid, continuous clocks, pending events and forbidden post-release root writes. Synthetic observation tests validate scoring and scheduling rules; they do not demonstrate native or live BANC flight.
