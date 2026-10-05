# Falling-off repair

**Historical result; retraction interpretation corrected.** The run below used an I1-based hinge gate. Re-reading [Heide and Götz (1996)](https://doi.org/10.1242/jeb.199.8.1711) shows that Drosophila III1, rather than I1, is associated with wing retraction and flight termination. I1 participates in reducing stroke amplitude during steering. Current code corrects that identity. The old perching result remains a measurement of the old code; it is not validation of biological retraction or the current controller. Work is now reproducing the released FlyBody flight controller as a reference before further interface calibration.

The original failure was a passive slide followed by inappropriate wingbeats, wing/surface impacts and tumbling. The repaired one-fly BANC run stayed on the fruit for six simulated seconds with its brain connected, followed by one second with the motor connection disabled. There were no airborne samples or wing/surface strikes in either interval. This is a repair of the observed falling-off sequence, **not evidence of successful autonomous takeoff or sustained flight**.

![Measured before and after behavior](flybody-flight-repair.png)

## Mechanical changes

- **Foot placement uses native contacts.** The initializer first fits each leg to the terrain, then refines the claw position using the actual MuJoCo contact geometry. A capsule centre is not its contact point. The refinement also accounts for the contact margin on flat heightfields. This runs only when placing the fly; it does not hold the body up during simulation.
- **Contacts resolve over 2 ms, instead of 0.2 ms.** The stiff source setting repeatedly unloaded the small claws and caused downhill creep. The selected compliance keeps the original friction coefficient, all six root DoFs, masses and inertias. No root position clamp, stabilizing torque or external lift is applied.
- **Unfolding and beating are separate.** A small lookup table transforms the measured wing cycle into a body-relative stroke plane, with continuous joint angles across phase and power. At full activation the fitted frequency is 235.8 Hz. The native solver still computes wing motion, aerodynamic forces and contacts at 50 µs steps.
- **Steering motor units remain distinct.** Each of the 12 annotated steering muscle types has its own response. The local mechanical fit uses force/torque directions from the [published wing-hinge study's RoboFly model](https://github.com/FlyRanch/mpc-simulations/blob/d29a8d3467122addbfcb9929bd57baf4b9c05576/mpc_simulations/MPC_simulations.py), passed through a measured MuJoCo response matrix. This replaces the single positive yaw offset obtained by averaging every muscle on each side.
- **Retraction can disengage the hinge.** Strong i1 activity competes with b1 activity in a separate opening variable. The old averaging discarded this antagonism. Deployment takes 12 ms and the beat engages above 85% deployment; opening also scales transmission. These numerical rules are an explicit, unfitted hinge prior, **not measured BANC physiology**. The qualitative roles of basalar and first-axillary muscles are supported by [Drosophila muscle recordings](https://pubmed.ncbi.nlm.nih.gov/8708578/) and [wing-hinge mechanics](https://doi.org/10.1038/s41586-024-07293-4). The particular competition equation is our modeling choice.

The shared power oscillator and steering maps read muscle forces and time only. Food position, task stage, body attitude, altitude and desired speed do not enter the wing adapter. The BANC topology and neuron physiology were not changed. The UI reports actual transmitted wing activity, separately from the upstream power-muscle drive recorded in the diagnostic.

## Observations and tests

The [long live trace](flybody-flight-repair-final-live.json) contains 7,010 native physical samples. During the connected interval, the maximum displacement from the initial physical position was 0.495 mm, maximum tilt was 34.7°, and maximum angular speed was 6.92 rad/s. The previous run reached 1,904 rad/s. Power-muscle drive reached 0.764 while retractor activity kept the hinge disengaged. The fly made small leg and body adjustments on the fruit; it did not take off, approach food or complete feeding.

The final small placement correction for flat heightfield contact margins was added after that long trace. It is separately covered by the [support regression](flybody-support-validation.json) and the original-console UI check. The model and source hashes in each report identify the tested version.

The [support regression](flybody-support-validation.json) checks one body at a time in the original fruit pose, facing the opposite direction on the fruit, and on a flat surface. Every case starts with six contacting legs and stays supported for one second with zero motor drive. Displacement is 0.085–0.110 mm, including initial settling.

The [wing mechanics regression](flybody-flight-mechanics-validation.json) holds the root above the ground **only in the test**, to measure forces independently of stance and collisions:

| Effective power, open hinge | Mean vertical force / body weight |
|---|---:|
| 0.25 | 0.259 |
| 0.50 | 0.606 |
| 0.75 | 0.934 |
| 1.00 | 1.250 |

At full power, forward force is 0.0036 body weights and mean pitch torque is 0.000187 g·cm²/s². Strong bilateral i1 activation disengages the hinge despite the same power input. Removing retraction restores lift. Left versus right b2 activation produces opposite roll moments (about ±0.00432 g·cm²/s²). All target cycles are checked for angle-wrap discontinuities, and externally applied root forces remain zero.

The [native/WASM check](flybody-runtime-validation.json) passes with maximum cross-engine error 9.59e−10. The original one-fly 3D UI check passes, including rendering registration, vision, controls and instrument windows. Existing web and physiology suites pass (41 and 22 tests).

## Performance

One-fly benchmarks used the original 3D UI, eyes and open anatomy view in Chromium on the same machine. Higher simulated-time throughput is better:

| Mode | Old FlyWire throughput | BANC throughput | BANC slowdown |
|---|---:|---:|---:|
| Reference | 0.0531× real time | 0.0344× real time | 1.54× |
| Fast | 0.0596× real time | 0.0493× real time | 1.21× |

Both pairs are within the requested order of magnitude. See the [reference measurements](flybody-flight-repair-performance.json) and [repeated Fast measurements](flybody-flight-repair-fast-performance.json). The repeated Fast pair completed with valid runtime checks and no page errors; the supplementary FlyWire screenshot timed out after timing had completed. These are shared-machine Chromium measurements, not Safari benchmarks or a guarantee of steady frame rates. The Fast BANC run used the final placement correction and remained on the fruit through 1.122 simulated seconds, with zero takeoffs or landings.

## Limits

A level-body lift measurement does not establish stable free flight. [Power-only and steering-disconnected free-body replays](flybody-flight-free-body-limit.json) still tumble. They are ablations, not the repaired live neural behavior. Sustained flight requires coordinated steering and sensory feedback; the current haltere transducer still encodes rotation magnitude rather than calibrated directional/phase preferences. No hidden flight controller was added to make those ablations succeed.

The steering transfer is a local, regularized fit, with assumed gain and morphology scaling. BANC `iv1`–`iv4` correspond to the source's hg/iv labels; the source's combined `iii24` response is provisionally assigned to BANC `iii4`. The nonlinear retraction rule is separate from that local fit. These assumptions need further physiological calibration before claims about biological flight or the full food-to-flight sequence are justified.

## Reproduction

From the checkout with the pinned model artifacts present:

```sh
uv run --no-project --with scipy --with mujoco==3.13.0 scripts/distill-flybody-wings.py
.venv/bin/python scripts/calibrate-flybody-steering.py
.venv/bin/python scripts/prepare-flybody-runtime.py
node scripts/verify-flybody-support.mjs
node scripts/verify-flybody-flight.mjs
node scripts/verify-flybody-runtime.mjs
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/verify-transplant-ui.mjs --one-fly
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --population=1 --ms=6000 --disconnect-ms=1000 --flight-trace=true --follow=true --output=reports/flybody-flight-repair-final-live.json
```

`diagnose-flybody-flight.py` now explicitly reproduces the **legacy** folded-pose adapter; use `verify-flybody-flight.mjs` for current wing mechanics. The third-party [wing-hinge license](../models/WingHinge-LICENSE) is retained with the existing FlyBody license.
