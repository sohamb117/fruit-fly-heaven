# Native FlyBody contact and transition reuse

The released FlyBody walking and flight policies both work on one common articulated physical body. A contact-triggered **flight → landing → walking** handoff also completed: first ground contact at 0.9044 s, then 0.6000 s of walking, ending upright at 4.87° tilt. The attempted walking → flight handoff lifted off but tumbled and failed its trajectory-error threshold. These are controller/body compatibility experiments with prescribed trajectories, not BANC-controlled behavior or autonomous food seeking.

## Measured results

All runs use the actual released deterministic policy means and native MuJoCo contacts. No externally applied force, root velocity injection, or root pose correction was used.

| Native experiment | Simulated duration | Result |
|---|---:|---|
| Original walking, then walking with wings enabled | 2.0000 s each | Both pass; 4.0337 and 4.0265 cm forward travel |
| Flight with ground collisions enabled | 1.1988 s | Pass; no floor contact at the airborne starting altitude |
| Flight with articulated passive legs | 1.1988 s | Pass; maximum root-reference error 0.04051 cm |
| Common body: walking | 2.0000 s | Pass; 4.01694 cm forward travel, maximum tilt 7.38°, ground contacts on all six legs |
| Common body: flight, retracted leg targets | 1.1988 s | Pass; maximum root-reference error 0.03851 cm |
| Common body: descent → contact → walking | 1.5044 s total | Pass; exact state continuity, 1.2215 cm subsequent walking, ground contact on 90.67% of walking control samples |
| Common body: walking → flight | 0.4000 s walking + 0.2664 s flight attempt | Fails; reaches 1.2517 cm altitude but tumbles and exceeds the 2 cm trajectory-error limit |

The original flight task terminates below a 0.2 cm thorax height. A separate descent with retracted legs reached floor contact, then this task cutoff; that cutoff alone does not demonstrate a biomechanical landing failure. The successful landing handoff switches to the walking task at real contact, at a root height of 0.2320 cm and tilt of 48.09°.

Native claw adhesion uses body transmissions. Its contribution to root generalized actuator forces is legitimate contact actuation; it is reported separately from `qfrc_applied` and `xfrc_applied`, which remain zero. The original failed wings-enabled walking result was an observation projection bug: actuator activations follow compiled XML actuator order, not action-spec order. The consolidated JSON contains the corrected passing run.

## Exact common configuration

`configure_common` in `scripts/experiment-flybody-contact-reuse.py` retains the complete articulated legs and wings, native body geometry and inertias, 50 μs physics, 10 ms exact actuator filters for nonwing actuators, unfiltered wing actuators, and the original flight wing gains, springs, damping, and fluid coefficients. It uses walking's ground and claw contact parameters. Walking observations average 40 physics samples per 2 ms control; flight averages four samples per 0.2 ms control.

Each policy sees its original observation fields, joints, and activation ordering. Outputs map by exact actuator names, using the original action ranges and float32 `canonical2real` transform. Walking adds zero wing commands. Flight retains native wingbeat-pattern processing and adds specified leg targets. The two task wrappers' physical parameter arrays matched exactly, including mass, inertia, frames, joints, collision/fluid properties, actuator gains, dynamics, and transmissions.

The production comparison is in `flybody-common-model-comparison.json`. Production has 57 qpos, 56 velocities, 56 actuators and 44 activation states. The common task has 116 qpos, 114 velocities, 65 actuators and 59 activation states; those counts include a noncolliding reference ghost. All 51 production joints match after resolving the free-root name alias, with identical local axes, anchors, and ranges. The common task retains 52 additional biological joints. Production has added rostrum/haustellum actuators and unfiltered claw adhesion. The two wing body frames differ by 0.08855° after accounting for quaternion sign; every other real body frame matches. Actual fly mass matches to floating-point precision. Production has no native sensors, while this common task retains 30; matching joint names alone does not establish interchangeable policy inputs.

## Reproduction and handoff

Source: [TuragaLab/flybody](https://github.com/TuragaLab/flybody/tree/d015e9bfe441bd90ae431bac24c55cb74bdbce26), revision `d015e9bfe441bd90ae431bac24c55cb74bdbce26`. The policy archive and waveform hashes are recorded with the baseline and teacher artifacts. Use `scripts/flybody-baseline-lock.txt` to recreate the isolated Python 3.11 environment.

```sh
/tmp/flybody-baseline-venv/bin/python scripts/experiment-flybody-contact-reuse.py --cases walking_common flight_common_retracted --output reports/flybody-contact-reuse-common.json
/tmp/flybody-baseline-venv/bin/python scripts/experiment-flybody-transition.py
/tmp/flybody-baseline-venv/bin/python scripts/export-flybody-landing-teacher.py
```

`flybody-landing-teacher/teacher.json` records the complete landing target sequence, initial state, common model parameters, exact observation/action projections, policy and source hashes, and transition continuity. The sibling XMLs remove only noncolliding visual mesh geometry while retaining compiled inertias. `native-teacher-sequences.npz` records every policy input, canonical action, native actuator control, and physical state. `native-replay-fixtures.json` contains the first and last ten controls per phase for independent native/WASM replay.

Continuous native replay of every recorded control on these mesh-free models has **zero real-fly qpos, qvel, and activation error**, without correcting the physical state inside either phase. Small residuals in the full-state comparison are confined to the noncolliding reference ghost. `scripts/verify-flybody-landing-teacher.py` reproduces this check; its results are saved alongside the teacher.

The successful transition uses an explicit 1→0.12 cm descent and 20→2 cm/s deceleration, deploys legs between 0.5 and 0.35 cm root height, and switches policies at the first nonzero native ground-contact force. These choices are declared teacher priors. They do not establish a learned transition strategy or measured muscle physiology. The useful next training target is the failed grounded takeoff transition on this same body, rather than another new mechanical body or a root stabilizer.

## Existing muscle and controller reuse

The [FlyGym FlyBody integration](https://neuromechfly.org/tutorials/5b_using_flybody_model/) already adapts a sensory-feedback walking controller to FlyBody joints, but its two descending commands and position actuators are a higher-level controller interface. It does not supply a connectome motor-neuron-to-muscle mapping.

The [FlyMimic muscle model](https://github.com/gizemozd/FlyMimic), documented by the [NeuroMechFly muscle-imitation tutorial](https://neuromechfly.org/tutorials/6_muscle_imitation/), supplies a concrete Hill-muscle/tendon starting point. I loaded the published model and tested all 15 left-front-leg muscles with 0.25 activation pulses for 20 ms: every response was finite and changed joint motion relative to zero input. The model retains a tethered thorax, one actuated leg, and its own anatomical body frames. Its explicit XML overrides use 0.1/0.4 ms activation/deactivation constants. It is an actual muscle mechanics template, not an already validated whole-body flight/landing solution. Details and asset hashes are in `flymimic-muscle-probe.json`.

For BANC, these artifacts can serve as offline mechanical and action targets while the brain → VNC → motor-neuron path remains explicit. Treating the published policy outputs as BANC muscle activity would not be supported by these tests.
