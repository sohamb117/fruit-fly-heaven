# Wing actuator moment correction and remaining contact failure

The deployed correction fixes a measured force–moment frame error. It does **not** establish successful BANC flight: the subsequent live fly still overturns. A source-matched causal replay now shows that wing contact with the banana causes the full flip and explosive launch in that trajectory, after substantial tilt has already developed.

## Deployed correction

The old fitter zeroed aerodynamic moment about the thorax/free-joint origin. That is not the fly's whole-body center of mass (COM): at the calibration pose the COM is displaced by `[-0.03683351, -0.00013658, -0.01581493] cm`. Its old full-power pitch moment about COM was `-0.04127606 g·cm²/s²` despite the near-zero thorax moment.

The fitter now uses the instantaneous native `subtree_com` and the correct wrench transfer:

`torque_COM_world = R_root · torque_root_local − cross(COM_world − root_world, force_world)`.

Native `mj_applyFT` checks at three root orientations agree with this transformation to `3.47e-18`. Each nonzero power knot is recalibrated for zero mean COM moment while preserving its original mean force. Tiny differential trims account for the actual lateral COM offset; they are not fitted to a flight trajectory. Steering calibration uses the same COM reference.

Preserved: 235.813447 Hz frequency, all seven power knots, original force curve (maximum fitting change `7.70e-5` bodyweights), deployment timing/threshold, zero-power targets, native gain/damping/stiffness/masses, motor-rate mapping, and all contact geometry. No root-state controller or external root force was introduced. The wing runtime is unchanged.

Production data changed:

- `models/flybody-wing-actuation.json`: SHA256 `5428033a5967dea38b5840d89efe7eec267ceea530d4b2efcb84dabcb2fcbfb0`.
- `models/flybody-mujoco.json`: only `wing_actuation` changed; SHA256 `c1348adf19decee75abbbb1d2be8a5ef6ac8cfe4fe4850083a502b62877c4a48`.

Preparation changes: `scripts/fit-flybody-wing-interface.py`, `scripts/distill-flybody-wings.py`, and `scripts/calibrate-flybody-steering.py`. The native XML remains SHA256 `8246f5ff573dcb60d05e8d4500b102a4dbac577ff1c73d716f5d8ecf3d09abd7`. See [implementation.json](implementation.json) for the exact mutation record. The pinned upstream FlyBody revision is `d015e9bfe441bd90ae431bac24c55cb74bdbce26`; this is a reduced muscle-to-wing interface, not the released learned flight policy or a measurement of BANC physiology.

## Mechanical validation and limits

[fit.json](fit.json) records all power fits and coordinate checks. [final-steering-validation.json](final-steering-validation.json) passes continuous targets, all power knots, force-curve checks, distinct III1/I1 behavior, and opposite left/right b2 roll signs at powers 0.25, 0.5, 0.75, and 1. The independently measured full-power residual COM pitch moment is `6.53e-5 g·cm²/s²`.

[final/candidate-validation.json](final/candidate-validation.json) preserves all failed conditions:

| Matched condition | Original mapping | COM correction |
| --- | --- | --- |
| New-solid-scene recorded MN rates, 0.4 s | Flip at 0.108 s; peak 1632 rad/s | Flip at 0.366 s; peak 517 rad/s |
| Warmed free-air balanced power 0.5/0.75, 1 s | Flip at 0.03565/0.02830 s | No flip within 1 s; appreciable drift remains |
| Warmed free-air balanced power 1, 1 s | Flip at 0.02435 s | Flip at 0.74395 s |
| Actual WASM muscle 80 Hz DLM/DVM cold ramp, free air, 1 s | Flip at 0.066 s | No flip within 1 s; minimum upZ 0.205 |
| Same cold ramp, native flat floor, 1 s | Flip at 0.078 s | Flip at 0.712 s |

The warmed diagnostic holds the body only during its explicitly recorded preparation period; release has no root correction. Cold ramps start from rest. These results support the frame correction, not stable flight or successful takeoff. Closed-loop neural inputs change when the body changes, so the recorded-input improvement does not predict the live neural run.

## Actual live after-COM onset and causal contact test

The actual BANC/UI capture is [after-com/onset/capture.json](../flybody-solid-wing-repair/after-com/onset/capture.json). [wing-contact-causal.json](../flybody-solid-wing-repair/after-com/wing-contact-causal.json) replays its 200 two-millisecond motor updates with native 50-microsecond stepping. Baseline qpos, qvel, controls, and muscle state match exactly. Fifteen executed mechanics sources are hash checked. All runs have zero applied root forces.

| Condition, 0.4 s | First wing impact | First inversion | Maximum angular speed | Maximum rise |
| --- | --- | --- | --- | --- |
| Captured production | 0.13805 s | 0.14375 s | 2473.28 rad/s | 3.9116 cm |
| Diagnostic wing–environment contacts suppressed | None | None | 116.27 rad/s | No rise |

The diagnostic changes exactly 1020 wing–environment collision pair eligibilities; an exhaustive pair-matrix audit confirms all other pairs are preserved. Root pose and velocity are bitexact across 1761 logged native steps before impact. Its minimum upZ is 0.3609, so it still tilts substantially.

The first impact is `wing_right_membrane_collision` against `habitat_fruit_0_banana_14`, at native position `[-3.339440714, -1.657539629, 1.509714675] cm`, normal `[0.570317816, 0.418062307, -0.707079554]`, normal force `3.33985 g·cm/s²` (33.40 µN), and penetration 0.01914 mm. At that instant upZ is 0.4069 (66° tilt) and angular speed 23.20 rad/s. The body therefore loses balance before impact; the impact causes its subsequent full inversion and large launch in this matched trajectory.

[wing-contact.json](../flybody-solid-wing-repair/after-com/wing-contact.json) also preserves rejected 5/10 ms wing-only contact-time diagnostics. Both cause earlier inversion; 10 ms increases the peak angular speed. Neither is a material stiffness calibration or production fix. Contact suppression is likewise diagnostic only. A compliant physical wing approximation must be calibrated independently and must retain this distinction between pre-impact tilt and collision-driven failure.

## Reproduction

Run from the repository root. Calibration commands write isolated candidates; they do not deploy them:

```sh
uv run --no-project --with scipy --with mujoco==3.13.0 scripts/fit-flybody-wing-interface.py --output=reports/flybody-wing-correction/reproduced.json --report=reports/flybody-wing-correction/reproduced-fit.json
uv run --no-project --with mujoco==3.13.0 scripts/calibrate-flybody-steering.py --config=reports/flybody-wing-correction/reproduced.json --report=reports/flybody-wing-correction/reproduced-steering.json
node scripts/verify-flybody-flight.mjs --output=reports/flybody-wing-correction/rechecked-steering.json
node scripts/verify-flybody-wing-interface.mjs --capture=reports/flybody-solid-wing-repair/before-wing/onset/capture.json --output=reports/flybody-wing-correction/rechecked --free-seconds=1 --powers=.5,.75,1 --cold=true
node scripts/diagnose-flybody-wing-contact.mjs --modes=baseline,no_wing_environment --output=reports/flybody-solid-wing-repair/after-com/rechecked-contact-causal.json
```

The source guards intentionally reject replay against changed mechanics files. Reports named `rejected-*`, `com-218hz-prototype*`, or under `solids/` are earlier experiments, not the deployed configuration. The final evidence is linked above.
