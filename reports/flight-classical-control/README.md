# Synthetic controller / reduced native body — no BANC

The classical controller passed the declared 1.5-second airborne positive control. The paired open-loop run exceeded the 20° tilt criterion. Both began from identical native states after 100 ms of restrained wing warmup, with a 3° rotation-vector component about each body axis. This demonstrates control authority in this reduced plant, not a biological flight result.

| Recorded metric | Open loop | Classical feedback |
|---|---:|---:|
| Maximum tilt | 29.83° | 8.03° |
| Final tilt | 29.06° | 4.81° |
| COM height change | −1.878 cm | −0.118 cm |
| Maximum horizontal drift | 6.902 cm | 5.528 cm |
| Final filtered angular speed | 1.765 rad/s | 2.841 rad/s |
| Wing-control clipping, fraction of native steps | 19.13% | 18.93% |
| Contact steps / instability warnings | 0 / 0 | 0 / 0 |

The pass criterion was: complete the requested duration, remain below 20° tilt, finish below 3 rad/s filtered angular speed, change COM height by less than 10 cm, and encounter no contacts or native instability warnings. It does not require horizontal position holding. No horizontal controller was used; the drift above is material.

## Plant, inputs and controller

- The model has seven joints: a free root plus six wing hinges (`nq=13`, `nv=12`, `nu=6`). Nonwing hinges are removed at the calibration-neutral pose. Rigid-body geometry, masses/inertias, wing armatures, native wing actuators, fluid model, and existing wingbeat/steering tables remain unchanged. This is an airborne start without habitat contacts.
- Inputs are 24 independently bounded synthetic steering-muscle force values in `[0,1]`, plus common power applied equally to both wings. They enter the existing `FlyBodyWings` basis directly. BANC activity and muscle activation dynamics are bypassed; this is not a calibrated neural-to-muscle response.
- Fifty-one restrained finite-difference probes measured the local force/torque response, followed by three trim refinements. Each probe used 100 ms warmup and approximately 16 measured wingbeats. The measured hover trim used common power `0.762204`, lift `1.000144` bodyweights, and near-zero mean COM torque.
- Feedback uses quaternion attitude error, body-local angular velocity filtered with a 6 ms time constant, and constrained control allocation. Default attitude gains are natural frequency 24 rad/s and damping ratio 0.9; common-power height gains are `Kp=16`, `Kd=8`. Only the six existing wing actuator controls change after release. No root pose/velocity resets, root actuators, or `qfrc_applied`/`xfrc_applied` forces are used.
- Fluid moments are rotated into world axes and shifted from the root origin to the matching whole-body COM before the kinematic cache is refreshed. The controller approximates inertia using all rigid-body masses about COM at the level neutral pose. This inertia estimate excludes generalized joint armature, which remains present in the native plant.
- The current wing implementation applies the derivative steering basis to absolute nonnegative force; the cited MPC source instead uses deviations from a muscle-mean operating point. The measured tonic trim compensates the current plant's offset. This diagnostic does not resolve or validate that transfer assumption.

This experiment includes no BANC training, takeoff, landing, or horizontal station keeping. It is one release phase, one initial perturbation, and one default controller setting, not a robustness study.

## Artifacts and execution status

[result.json](result.json) is the authoritative completed assay: `completed=true`, `sourceUnchanged=true`, `pairedInitialStateIdentical=true`, and the closed-loop `meetsDeclaredPositiveControl=true`. It contains all probe inputs, the measured Jacobian, trim iterations, source/model hashes, and 2 ms trajectory samples. [fixed-nonwing.xml](fixed-nonwing.xml) is the reduced diagnostic model.

[failure.json](failure.json) records a separate duplicate invocation rejected with `EEXIST` before any simulation. An error-reporting bug allowed that rejected invocation to add this file to the already-created directory; it did not alter the completed `result.json`. The diagnostic script now writes failure artifacts only when the current invocation created the output directory. No plant simulation was repeated for this file-handling fix.

The historical result's source hash is preserved: `7ea287c4d45b6cb3124ef6f625e9331f567eb76f33ca7bfbe931089667745012`. [diagnostic-source.used.mjs](diagnostic-source.used.mjs) preserves the exact script bytes used by the assay; it is a source snapshot, with imports relative to its original `scripts/` location. The current [diagnostic script](../../scripts/diagnose-flight-classical-control.mjs) differs only in the output-directory ownership guard added afterward.

Matplotlib was unavailable in the project environment, so no plot package was installed and no trajectory plot was generated.
