# Wing actuator contract audit

No index, radian/second conversion, target-error sign or wing-update cadence bug was established. The measured behaviors below are consequential reduced-adapter assumptions. They do not justify changing the native gains or deployment gate without a different, evidence-backed model contract.

Reproduce with `node scripts/audit-flybody-wing-contract.mjs`. [Machine-readable results](flybody-wing-contract-audit.json) record source hashes, native model parameters and individual probes. The active direct model audited here has 56 actuators; the separately exported common policy body has 65. These probes use the actual MuJoCo WASM engine, compiled WASM muscles and production wing module. They do not integrate a body trajectory, restrain a root or apply external forces.

## Zero motor drive still permits strong restoring torque

With both power-muscle inputs zero, a left wing yaw displaced −0.01 rad from its folded neutral receives a +0.01 command. Native `mj_forward` measures **+0.18 g·cm²/s² actuator torque**, versus **+0.0001 passive spring torque**, a **1,800× ratio**. External applied force is zero. The target-minus-position servo remains powered at zero DLM/DVM input; it is not passive-only muscle mechanics.

This follows the documented position-error/force interface and its retained native wing gain 18. The official FlyBody flight task also adds wing target minus current position before sending force-actuator controls. The observation identifies a tonic restoring assumption, not a reason to remove an upstream gain blindly. Cold zero-drive neutral wings receive exactly zero commands for a 0.25 s command probe: the phase clock advances, but no autonomous beat is generated. A previously deployed wing also has a decaying retraction target after power is removed.

Wing gains are 18; proximal leg gains are 0.8 and distal leg gains 0.4. All six legs match by joint type: 30 proximal and 12 distal actuators. Legs additionally use native position bias and a 10 ms activation filter; wings use force actuators without that filter. Both retain passive springs. Raw gain ratios therefore are not calibrated physiological strength ratios.

## The opening gate is steep, and documented

At settled full power, actual module output is:

| Requested opening | Effective power |
|---:|---:|
| 1.00 | 1.000 |
| 0.99 | 0.933 |
| 0.95 | 0.667 |
| 0.90 | 0.333 |
| 0.85 or less | 0 |

The exact steady rule is `min(power × opening², clamp((opening − 0.85) / 0.15))`. Thus a modest retraction signal can eliminate beating despite high DLM drive. The 12 ms exponential deployment time constant reaches 85% after 22.77 ms at full opening.

The [earlier repair report](flybody-flight-repair.md) explicitly describes the absolute 85% deployment requirement and the additional opening attenuation, and labels both as an unfitted hinge prior. Normalizing deployment by requested opening would implement a different meaning—85% of the requested opening—and is not an established bug fix. Current code correctly assigns the retraction channel to III1; the old report flags its historical I1 attribution. The biological paper supports the qualitative III1 association, not this competition equation or numerical threshold. [Heide and Götz, 1996](https://pubmed.ncbi.nlm.nih.gov/8708578/).

## High DLM rates are not the displayed wing power

The decoder averages rates within each annotated group, saturates excitation at 80 Hz, runs the 15/40 ms activation and fatigue model, averages DLM and DVM forces on each side, then shares the mean of both sides through one thoracic oscillator. Each stage is explicit. The wing pseudo-joints supply normalized length 1, velocity 0 and maximum force 1, so this output is normalized effort rather than physical muscle torque.

Using actual compiled WASM muscles with constant inputs for 0.2 s and open hinges gives:

| Held inputs | Left effective power | Right effective power |
|---|---:|---:|
| All DLM and DVM groups at 80 Hz | 0.9855 | 0.9855 |
| Both DLM groups at 80 Hz; DVM silent | 0.4928 | 0.4928 |
| Left DLM alone at 80 Hz | 0.2464 | 0.2464 |

The contralateral output is the documented shared oscillator, not a side-index error. The group averaging, saturation, fatigue and shared-drive equations still need physiological identification.

Recorded observer frame 121 had both DLM groups saturated, DVM excitations 0.799/0.700 and an opening excitation proxy of 1 on both sides. Actual displayed wing power was 0.697. Starting fresh muscles and holding those snapshot inputs produces 0.864 after 0.2 s; this is **not a replay**. The sparse snapshot omits activation, fatigue, muscle force and deployment history, so its actual wing output cannot be reconstructed from rates alone.

## Contract and next discriminating measurement

The active adapter looks up exact actuator and joint names, uses native yaw-Z/roll-X/pitch-Y axes, computes positive target-minus-current radian errors, updates wing commands every 200 µs and physics every 50 µs. Muscle state updates every 1 ms. No verified sign, indexing or milliseconds conversion problem was found in this path.

The fixed 235.8 Hz cycle and transformed target table differ from the released policy's 218 Hz ±5% WPG, learned residuals and frequency output. This is an explicit calibration change. MuJoCo supplies springs, damping, fluids and contacts; it does not contain a hidden flight stabilizer. The direct adapter receives no body attitude, root velocity or desired trajectory.

The useful next measurement is synchronized recording of named DLM/DVM/III1/B1 rates, their actual activation/fatigue/forces, requested opening, deployment, effective power, six wing targets/positions and actual joint torques. Keep those quantities separate in diagnostics. None of the command probes establish stable free flight or explain the complete observed tumble on their own.
