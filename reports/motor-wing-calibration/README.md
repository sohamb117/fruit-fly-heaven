# Motor-to-wing identification before behavioral training

This assay holds neural motor input fixed and measures the current adapter rather than asking an optimizer to change neural activity. It uses the actual compiled WASM muscles, `FlyBodyPhysics`, COM-corrected wing tables, and MuJoCo runtime. It neither submits training jobs nor edits production physics.

The [baseline measurements](baseline/README.md) cover 73 deterministic conditions: symmetric DLM/DVM rates 0, 1, 3, 5, 8, 12, 20, 40, 60, 80, and 100 Hz; steps and 150 ms ramps; DLM-only, DVM-only, and left-only controls; and separate b2/III1 steering controls. Native root-restrained cases run for 0.6 seconds; free-air and flat-floor cases run for 1.2 seconds. A four-case pilot precedes the full assay. The [17-case followup](followup/README.md) tests finer onset rates and isolated deployment time constants.

All scenes use the same articulated initial state prepared on a native flat floor. Air and restrained scenes then translate that state above an empty world. The rig uses a native thorax-to-mocap weld with finite compliance, not pose resets or applied root forces. Restraint errors are explicitly bounded and measured. Weld reaction is reconstructed only from its equality rows; contact and joint-limit forces are excluded.

Every native 50-microsecond step contributes muscle state, deployment, effective power, wing target and actual joint position/velocity, native forward-dynamics acceleration, actuator force, generalized joint torque, control clipping, actuator work, root kinematics, and aerodynamic and restraint wrenches. Native `qacc` is explicitly a forward-dynamics acceleration; with implicit damping it differs from realized velocity change. Instrumentation version 2 records both. Force/moment measurements use instantaneous whole-body COM and rotate root-local moments into world coordinates. Native `mj_applyFT` checks independently validate that transform at three root orientations.

`baseline/summary.json` records definitions, source SHA256 values, all successful and failed behavior outcomes, case-level aggregates, trace hashes, and instrumentation checks. Compressed traces contain 10 ms samples plus selected native-rate windows. The late averaging window is approximately 100 ms, not an assertion that fatigue or the whole dynamical system has reached steady state. Phase-dependent means should not be mistaken for perfectly cycle-aligned force calibration. Contact/takeoff observations follow production's 1 ms refresh cadence and can miss contacts between observations; a reported 20 ms contact-free interval means 20 ms of contact-free samples.

Reproduce from the repository root:

```sh
node --test scripts/test-motor-wing-transfer.mjs
node scripts/audit-motor-wing-transfer.mjs --pilot=true
node scripts/audit-motor-wing-transfer.mjs
node scripts/audit-motor-wing-transfer.mjs --suite=followup
node scripts/audit-motor-wing-transfer.mjs --suite=continuous
```

The default wall-time cap is ten minutes. The assay keeps failed conditions and does not use inversion as a stop condition. A numerical failure or changed source makes the instrumentation gate fail instead of silently omitting a condition.

## Baseline observations

All 73 cases completed in 219.1 seconds and passed the instrumentation gates. Maximum rig movement was 0.282 micrometers and 0.349 milliradians. The three native force-to-COM coordinate checks agreed to at most `4.51e-17` in native wrench units. No user-applied force was present in any case.

The restrained late lift curve rises from about 0.097 bodyweights at 12 Hz, through 0.259 at 20 Hz, 0.595 at 40 Hz, 0.905 at 60 Hz, to 1.195 at 80 Hz. This sweep does not show the proposed large gain jump in the 50–80 Hz region. Held 80 and 100 Hz produce identical mechanics because the excitation input saturates at 80 Hz; 100 Hz ramps differ because they reach that saturation earlier.

Pitch actuator control clipping is already present at 8 Hz. At 12 Hz it occurs for approximately 20.4% of native samples, and at 80 Hz approximately 22%; clipping persists in the late window, so it is not exclusively a deployment impulse. Native summed signed wing actuator work over 0.6 seconds is approximately 1.80 mJ at 12 Hz and 3.17 mJ at 80 Hz. These values include work against native damping and are not a calibrated biological metabolic estimate.

DLM-only, DVM-only, and left-side-only recruitment at the same rate produce identical body mechanics in these controls. That follows from the current shared arithmetic pooling; it does not establish that those physiological motor patterns are equivalent. Separate b2 steering controls do produce opposite signed roll moments. Sustained III1 recruitment suppresses beating despite high power-muscle input.

On a flat floor, zero drive stays upright. Constant 3, 12, and 40 Hz also remain uninverted for 1.2 seconds, but unintended motion persists: the 12 Hz case translates approximately 0.634 cm with no leg/claw motor recruitment. Constant 80/100 Hz qualifies as takeoff at 73 ms but inverts at 710.9 ms and reaches 1335 rad/s. A 150 ms ramp to 80 Hz avoids inversion through 1.2 seconds, yet reaches only 0.292 minimum upZ and drifts over 7 cm. Ramping to 100 Hz inverts at 997.7 ms. These are transient, waveform-sensitive outcomes, not stable flight.

The assay therefore supports calibrating the deployment/actuation interface and takeoff coordination before interpreting an improved task score. It does not support globally suppressing wing neurons, an unmeasured fivefold servo reduction, or calling every non-inverted case a successful flight.

## Deployment followup

The hard onset is directly observable below the original coarse sweep: 0.8 Hz produces no deployment, whereas 0.81 Hz eventually opens the wings fully and reaches 2.62 native units of peak actuator torque. This is the `power > 0.01` switch selecting a full opening target, distinct from the 85% deployment-before-beating gate. It is not a measured physiological threshold.

Increasing only the deployment time constant from 12 to 60 or 120 ms is not a demonstrated repair. In the 80 Hz floor control, inversion moves from 710.9 ms to 487.6 and 516.1 ms respectively. All three still reach the native torque limit. Slower deployment changes startup work and timing, but not the need for stable takeoff coordination. The four repeated 12 ms control cases match the earlier baseline's body state and actuator work exactly after adding optional interface branches and rebuilding the WASM runtime.

## Opt-in continuous opening

The [27-case continuous-opening suite](continuous/README.md) tests an explicit alternative, with the default unchanged. `deploymentPowerSpan=0` preserves the original switch. Positive values set the deployment target to `opening * smoothstep(clamp((power - 0.01) / span))`. The interface bounds `0..0.5` are engineering search limits, not physiological confidence intervals. This decoder reads muscle power and time; it does not add root feedback.

All 27 cases passed the final numerical and instrumentation checks in 69.0 seconds. Near onset, the new branch behaves continuously: at 0.81 Hz, span 0.1 gives deployment `4.68e-6` instead of essentially 1, and peak actuator torque falls from 2.6223 to 0.001235 native units. The latter is the same background restoring level present in zero-drive controls. This addresses the abrupt onset of the reduced decoder.

It does **not** solve flight. At 80 Hz on the floor, span 0.1 inverts at 517.8 ms and span 0.25 at 507.4 ms, versus 710.9 ms for the default. At 12 Hz, span 0.25 holds deployment near 0.590, so the existing 85% gate never engages beating; floor translation falls from 0.634 to 0.0080 cm. That outcome is reduced activity/preparatory opening, not successful flight. Span 0.1 retains the 12 Hz beat and gives slightly more translation than the default.

No new span or time constant was selected as a calibrated default. The interface now makes this uncertainty expressible and testable. A task optimizer must still demonstrate stable high-drive behavior rather than gaining a better posture score merely by suppressing beating.

`default-parity.json` records 13 repeated native comparisons spanning the optional interface branches, rebuilt WASM and continuous-opening branch. Root positions/velocities, wing joint positions, total signed actuator work and inversion times match their original controls exactly. The three suites contain 117 executed conditions, plus the initial and final four-case pilots; the complete artifact directory is under 20 MiB.

## Provenance and instrumentation revisions

Baseline and followup reports preserve their original source hashes and exact executed assay/helper source snapshots under their respective `source/` directories. They predate the final instrumentation revision and should not be represented as runs of the final source. Their recorded native sample counts and elapsed times match the requested durations; they did not yet record native warning counters or realized acceleration. Their late window includes 2,001 samples (100.05 ms) instead of version 2's exact 2,000 samples.

Version 2 adds strict elapsed-time/sample-count gates, immediate rejection of native bad-position/velocity/acceleration/control warnings, realized acceleration alongside `qacc`, startup work windows, and the exact 100 ms averaging interval. The final pilot and continuous-deployment suite use this revision. Native warning ownership is independently tested: `MjData` owns the borrowed warning vector, while each copied entry is disposed after reading. This prevents both silent native resets and an incorrect contact-style vector deletion.

Validation commands additionally include `node --test web/test/motor-interface.test.mjs web/test/wing-deployment-continuity.test.mjs`. These and the assay helper tests pass 16 focused checks, including a native bad-control warning probe, default parity across recruitment, bounded continuous opening, and isolated configuration ownership.

## What the COM moment curves establish

All quoted moments are world-axis moments about whole-body COM, in `g cm²/s²`. At the identity-root rig pose, +X is forward, +Y is the left side, and +Z is up. Accordingly +Mx is positive roll about the forward axis, while +My tips the forward axis down by the right-hand rule. These conventions must be distinguished from aircraft nose-up-positive pitch notation.

The earlier [20-wingbeat fixed-pose validation](../flybody-wing-correction/final-steering-validation.json) has small positive mean pitch residuals across its nonzero power knots: approximately `7.87e-7` to `6.53e-5`. Mean roll stays between approximately `-1.26e-6` and `+1.85e-6`, changing sign rather than showing a large persistent unilateral bias. Its full-power lift is about 1.25 bodyweights. This supports approximate force/moment balance at that calibration pose; it does not establish a passively stable flight state.

The present dynamically articulated rig has positive mean My over its approximately 100 ms late windows, from `3.69e-6` at 1 Hz to `1.32e-3` at 5 Hz, then `1.47e-4` at 80 Hz. Roll/yaw means remain much smaller. But periodic moments are far larger than their means: at 80 Hz, My RMS is about `0.0979`. Reintegrating the native-rate final burst over exactly one commanded wingbeat gives pitch means ranging from `-4.48e-4` to `+6.52e-4` as the window slides; 12 Hz also spans both signs. `com-moment-review.json` records the integration and all axes. This sensitivity, changing fatigue, dynamic non-wing joints and a compliant rig prevent treating a single late-window mean as a demonstrated steady pitch bias.

Most importantly, none of these curves measure whether a perturbation produces a restoring moment. There is no measured derivative of moment with respect to pitch/roll angle or angular velocity, nor an identified coupled force/moment stability matrix. A balanced, powered wingbeat can therefore coexist with unstable free motion. The existing data do not establish that passive open-loop lift is impossible in principle, but they provide no evidence that the current decoder is passively stable. Selecting another constant bias from these curves alone would not resolve that uncertainty.

Important interpretation limits:

- Aerodynamic vertical force in a falling free-air control includes drag and can approach bodyweight with zero wing drive. Use the restrained rig for a clean rate-to-lift curve.
- A non-inverted free-body trajectory can still slide, drift, lose height, or rotate rapidly. Qualified takeoff is recorded separately and does not imply stable flight.
- All non-wing motor inputs are zero except named steering controls. Native posture servos and passive mechanics remain, but active claw recruitment and a coordinated takeoff sequence are absent. Actual-fruit replay is a separate required test.
- Native actuator work and force are mechanical quantities. Their magnitudes have not been matched to biological muscle energy or torque measurements. Large values alone do not identify a correct replacement gain.
- DLM/DVM pooling, rate saturation, deployment and waveform interpolation remain modeling assumptions. The assay identifies their consequences; it does not establish their physiological validity.

The coordinate and restraint calculations follow [MuJoCo's constraint force equations](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework) and [weld definition](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld).
