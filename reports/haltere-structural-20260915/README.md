# Opt-in coupled haltere observer, version 2

Implemented in `web/virtual-haltere.js` and `web/banc-haltere.js`. The original version-1 stateless observer and its arithmetic remain available. This report covers the haltere module and its focused checks; the [combined structural implementation](../structural-proprioception-20260915/README.md) documents the full BANC/native-body evaluation. The observer remains uncalibrated. No optimization, training or deployment was performed.

The self-contained versioned inputs are [`models/banc-haltere-directional-v2.json`](../../models/banc-haltere-directional-v2.json) and [`models/banc-haltere-marginalized-v2.json`](../../models/banc-haltere-marginalized-v2.json). Each contains the exact 328 prepared sensory identities, 34 side/type populations, geometry, mechanical coefficients and provenance. Loading them requires no report-only asset. The preparation script here regenerates report and model copies from pinned identity/geometry inputs; it validates contracts without advancing time.

## What is corrected

Version 1 imposed amplitude = `(pi/4) * ownHalterePower`, so zero left hDVM force suppressed all 171 left sensor inputs even with moving left wings. The saved 500/600 ms assays indeed had left power zero; they are not rewritten. The existing [neural and mapping audit](../flight-haltere-live/haltere-power-audit/README.md) and [phase-resolved assay](../sensory-feedback-20260915/README.md) still describe silent left model motor drive.

Version 2 is a separate, stateful, one-way observer. Actual same-side wing joint motion can excite a damped virtual haltere even when its own drive is zero. It preserves `actualMusclePower` as the original normalized native force-derived proxy, separately from wing power, virtual angle, velocity, amplitude and phase. It does not borrow the opposite haltere's activity, create hDVM spikes, apply native force, reset a body joint or claim that the virtual movement was observed biologically.

## Evidence and limits

Postmortem wing actuation and selective thoracic lesions show passive wing/haltere coupling in **soldier flies, Hermetia illucens**, including independent same-side links. This supports the presence of a mechanical route; it supplies no Drosophila coefficient for this implementation. [Deora, Singh & Sane, PNAS 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4321282/).

Wing clipping and haltere loading further support weak frequency coupling through the subepimeral ridge, with loss of synchronization at sufficient mismatch. The observed weak influence of haltere loading on wing frequency motivates a one-way approximation, not an exact mechanical law. [Deora et al., eLife 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8629423/).

Genetic experiments in **Drosophila melanogaster** support active haltere muscle control affecting sensory feedback and wing motor timing. They do not calibrate our native hDVM force-to-drive conversion or identify individual BANC receptor directions. [Dickerson et al., Current Biology 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7307274/).

The numerical coefficients below, the inherited virtual axis, constructed beam frame, selected wing-roll coordinate and local harmonic continuation are declared priors. No damping, spring strength, transmission ratio, preferred direction or current cap was fitted to a neural/flight reward. This is a forced torsional approximation, not an identified asynchronous muscle/thorax model. It omits reciprocal native mechanics, root linear/Euler/centrifugal loading and Coriolis-induced changes in the virtual oscillator trajectory; body angular velocity affects the computed load.

## Mechanics and sampling

For each side independently, with natural angular frequency `w`:

```
theta'' = 2*zeta*w^2*A*ownPower*cos(ownPhase)
        + k*w^2*(transmittedWingAngle - theta)
        + 2*zetaCoupling*w*(transmittedWingVelocity - theta')
        - 2*zeta*w*theta' - w^2*theta
ownPhase' = w
```

The retained state is angle, angular velocity and own-drive phase. Reported amplitude is `hypot(theta, theta'/w)` and phase is `atan2(theta, theta'/w)`; these are state coordinates, not measured stroke amplitude. Own drive therefore changes a force term instead of instantaneously assigning displacement.

| Explicit prior | Value |
|---|---:|
| Natural frequency | 236 Hz |
| Damping ratio | 0.1 |
| Own-drive reference amplitude / phase | pi/4 / pi radians |
| Coupling stiffness ratio / damping ratio | 0.2 / 0.02 |
| Selected native coordinate | wing_roll_left / wing_roll_right |
| Native coordinate sign / transmission ratio | -1 / 0.5, independently per side |
| Rest center | 0.7 rad, actual metadata neutral used by body initialization |
| Current cap | 800 pA, unchanged sensitivity prior |

The selected native coordinate is not renamed anatomical stroke: FlyBody uses a mirrored Euler joint chain. The root frame and per-side sign are explicit. Wing power is the deployed native amplitude indicator; actual native q/qdot supplies motion. Within each held 2 ms body block, powered motion is continued harmonically from the current q/qdot about the declared rest center. This is a local causal approximation and can be inaccurate for non-sinusoidal motion or changing cycle centers. At zero wing power, continuation is linear, so a stationary angle is not invented into a periodic input. Static off-center poses can cause elastic settling; residual native/virtual motion is not erased when drive becomes zero.

Neural samples use a 0.5 ms grid. RK4 uses 50 us internal substeps. Advancing to the next sample integrates with the **previous** observed context, so the new endpoint observation cannot act backwards in time. The load at the endpoint uses its newly available context. At time zero the virtual state is at rest. Identical repeated time/input is idempotent; conflicting duplicate, backward, skipped or off-grid samples fail. Native clock roundoff has a 1e-9 s tolerance. The caller must reset per episode and preserve state across warm-up/release.

## Sensory profiles

Both profiles join exact prepared index/root ID, organ, side and cell type. They do not split indistinguishable cells by root ID. Current prepared annotations provide 19 cell types and 34 side/type populations but no receptor-field orientation; fields remain null.

The **directional candidate** assigns the same declared beam-normal direction to every population on each side: left normal 0, right minus normal 0 (angles 0 and pi). The constructed right normal 0 is approximately the negative mirror of the left, so this states a mirrored polar-frame registration. It is neither measured receptor tuning nor a fitted corrective sign. Signed bending is half-wave rectified before the existing saturating current function. Full signed bending components remain in diagnostics.

This common-axis choice is not a complete set of diverse receptor fields. Normal 0 lies near the virtual oscillation axis, so the choice emphasizes out-of-plane load and can reduce the phasic baseline relative to the old four-angle cycling. A v1/v2 behavioral difference therefore combines mechanical coupling and tuning changes; it does not isolate either cause or establish a faithful clock/gyro population.

The **marginalized control** retains unknown directions. It uses the uniform-orientation average `E[max(0, B*cos(alpha))] = |B|/pi`. It intentionally provides no assigned preferred axis; it must not be described as evidence of directional field tuning. Its purpose is an explicit control, not replacing the directional candidate after a direction-loss diagnosis.

Future population directions can be configured only as `declared-prior` with an evidence record. Claiming `measured` is rejected by this version. A named anatomical field requires exact agreement with a prepared `receptor_field` annotation; an arbitrary within-type directional split is rejected.

## Integration API

```js
const mapper = createHaltereCurrentMapper({
  ...versionedConfig, sensoryManifest, preparedIds
});
mapper.reset(); // once per new episode, not at warm-up release
mapper.writeInto(fullFloat32CurrentVector, {
  bodyTimeSeconds, elapsedSeconds, // 2 ms boundary + 0/0.5/1/1.5 ms
  omegaRootRadS, wingPhaseRadians, wingFrequencyHz,
  halterePower: [leftNativeHalterePower, rightNativeHalterePower],
  wingPower: [leftDeployedWingPower, rightDeployedWingPower],
  wingMotion: {
    left: { joint: 'wing_roll_left', source: 'native-joint',
      frame: 'native-joint-coordinate', angleRadians, angularVelocityRadS },
    right: { joint: 'wing_roll_right', source: 'native-joint',
      frame: 'native-joint-coordinate', angleRadians, angularVelocityRadS }
  }
});
const ownedState = mapper.snapshot();
identicalMapper.restore(ownedState);
```

All input validation/transduction completes before the 328 target entries change; failures roll back observer state. Snapshot identity includes mechanics, geometry, sensory profile and current cap. A fresh mapper cannot start at a nonzero time without restore. The wing phase label is retained as provenance and cannot create motion without the physical driver or own force.

Validation: `node --test web/test/haltere-structural.test.mjs` — **12/12 pass**. The tests cover exact v1 currents/diagnostics, all actual identities, silent-own/active-wing behavior, zero drive, side isolation, signed Coriolis effects, phase/motion separation, snapshot/reset, causal endpoints, malformed samples, tuning status and root-frame covariance. Tiny mathematical oscillator fixtures are not native flight or BANC evidence. `node --check` passes for both modified modules.

The final test is portable: its identity subset and frozen v1 oracle are under `web/test/fixtures/`, and its v2 configs are under `models/`. It reads no ignored report or prepared-data file. The original preparation provenance remains frozen with the experiment; [portable-validation.json](portable-validation.json) records the subsequent test-only portability change. Root's `environment.js` handoff was reviewed read-only: native qpos/dof and source/frame labels match, sampled body time/offsets are correct, and reset occurs per episode rather than at warm-up release.
