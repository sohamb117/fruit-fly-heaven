# Native leg direction audit

The previous direct adapter reversed femur and tibia flexion/extension. Independent native geometry measurements establish the required conversion on all six legs, and **all 24 actual production-path pulses pass after the correction**.

The prepared BANC convention remains `flex = +1`, `extend = −1`. Native FlyBody femur/tibia coordinates increase the physical internal angle when their coordinate increases. The adapter now converts those anatomical signs at the native boundary; the source annotations and prepared data remain intact.

## Kinematic evidence

`node scripts/audit-flybody-leg-signs.mjs` exports exact production initialization poses and invokes the independent native Python assay. The test uses neutral pose, the original fruit placement, opposite heading on the fruit and flat-ground placement. It measures angles between actual neighboring joint anchors, rather than inferring motion from XML class names.

For every femur/tibia hinge, +0.01 rad increases the internal angle; −0.01 rad decreases it. A further 31 samples span each native joint range in all four poses: **all 1,488 flexion/extension derivatives are positive**, with the smallest `d(internal angle)/dq = 0.9372`. This is evidence across the sampled working ranges, not a continuous-range proof or an assertion about every possible combination of other joint coordinates.

The [kinematic results](flybody-leg-sign-audit.json) include per-joint angles, sampled range extrema, exact BANC functions and source hashes. The [initialization fixture](flybody-leg-sign-placements.json) records qpos and confirms zero simulated time and zero external forces.

## Corrected production verification

`node scripts/verify-flybody-leg-direction-pulses.mjs` stimulates each flexor family and each extensor family separately at 80 Hz through the actual `FlyBodyPhysics` decoder, compiled WASM muscles and MuJoCo body. The 24 trials each run for 20 ms, one body at a time. No root pose/velocity is changed after initialization; gravity is active and external applied forces remain zero.

Every pulse changes its measured internal angle in the intended direction relative to an independently simulated zero-input baseline. Changes range from **1.787° to 4.388°**. The baseline root falls **0.19324 cm**, demonstrating that the body was not held in place. [Full pulse results](flybody-leg-direction-pulses.json).

This verifies the direction correction, not coordinated stance, gait, landing or takeoff.

## Length and shortening convention

For positive tensile muscle force, virtual work gives `torque = −force × dl/dq`. If `s` is the intended native torque direction and `r > 0` is an explicit reduced moment-arm prior, a consistent input is:

```text
normalizedLength = 1 − r × s × (q − reference)
positiveShorteningSpeed = r × s × qvel = −d(normalizedLength)/dt
```

The existing kernel uses `1 − 0.25 × velocityInput`, so its input must be positive shortening speed. An actual compiled-kernel probe gives force ratios **1.25, 1, 0.75** for velocity inputs **−1, 0, +1**, respectively. The corrected bridge uses the negative sign for length and the positive sign for shortening speed. Its Gaussian length factor is symmetric around 1, so changing only the length sign does not change instantaneous force; converting the anatomical direction also corrects the corresponding velocity modulation.

The position-target adapter still does not represent measured muscle insertions or tendon transmission. The moment-arm magnitude, rate scale and force capacity remain reduced-model priors.

## Coxa and tarsus remain separate questions

At neutral, positive coxa q moves the distal coxa anchor anteriorly on T1 but posteriorly on T2/T3. That anterior-position derivative changes sign within the T2/T3 coordinate ranges. This establishes that one coordinate is not a global anterior/posterior movement command; it does not establish a universal sign flip or a replacement actuator axis.

Positive tarsal q usually decreases the tibia–tarsus internal angle. The nearly straight right T1 posture has the opposite unsigned-angle derivative, and the derivative crosses the straight-line branch within the allowed ranges. A blanket reversal based on `extend_tarsus` class names would therefore be unjustified. Neither coxa nor tarsus mappings were changed by this audit.
