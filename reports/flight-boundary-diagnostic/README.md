# Saved flight-boundary traces

All three instrumented replays exactly match their original v8 g0 physics digests and every behavioral score field. **The first loss of the main continuous flight bout precedes the first native contact after 100 ms in all three seeds.** The later rapid terminal spins follow contacts; measuring their causal contribution still requires an ablation.

| Seed | Main qualified airtime | First loss | Failed gate | First native contact after 100 ms | Termination |
|---:|---:|---:|---|---|---:|
| 1190888 | 328.00 ms | 450.00 ms | angularSpeedRms | 719.15 ms: wing_left_membrane_collision ↔ wall41 | 722.00 ms |
| 1290888 | 504.00 ms | 626.00 ms | verticalSpeed | 729.85 ms: wing_left_membrane_collision ↔ wall43 | 734.00 ms |
| 1390888 | 344.00 ms | 466.00 ms | up | 1087.65 ms: ground ↔ tarsal_claw_T1_left_collision | 1092.00 ms |

The three first-loss measurements are:

- **1190888:** tilt 29.625°, angular RMS 20.003754 rad/s, vertical speed 5.967214 cm/s, inferred support 1.058381 body weights. Only angularSpeedRms fails; contact count is zero.
- **1290888:** tilt 42.638°, angular RMS 7.902618 rad/s, vertical speed -1.258761 cm/s, inferred support 0.853909 body weights. Only verticalSpeed fails; contact count is zero.
- **1390888:** tilt 45.741°, angular RMS 17.391982 rad/s, vertical speed 3.352186 cm/s, inferred support 0.952707 body weights. Only up fails; contact count is zero.

At the last completed 2 ms observation before the first subsequent native contact:

| Seed | Sample time | Age before contact solver | Tilt | COM speed | Vertical speed | Angular speed | Terminal angular speed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1190888 | 718.00 ms | 1.100 ms | 12.61° | 24.83 cm/s | 1.74 cm/s | 35.62 rad/s | 328.18 rad/s |
| 1290888 | 728.00 ms | 1.800 ms | 9.49° | 31.46 cm/s | -0.22 cm/s | 7.48 rad/s | 784.01 rad/s |
| 1390888 | 1086.00 ms | 1.600 ms | 42.38° | 43.73 cm/s | -17.51 cm/s | 26.53 rad/s | 509.18 rad/s |

These are sampled pre-contact states, not reconstructed impact states. Native contact forces belong to `solverTimeSeconds`; integrated position and angular speed belong to `timeSeconds`, 50 µs later. Full contact names, times, force vectors, first wing contacts, later short flight bouts, and independent cloned score states are in [result.json](result.json).

Seeds 1190888 and 1290888 first contact the wall with the left wing. Seed 1390888 first contacts the ground with its left T1 claw, after descending rapidly. All terminate for excessive rotation within about 3–4.4 ms of that first contact. This sequence separates the earlier qualification loss from the later collision-associated spin. It does not establish that removing contacts restores maintained flight.

Reproduce from repository root with `node reports/flight-boundary-diagnostic/analyze.mjs`. The script verifies reference hashes from the declared plan, matching source hashes, exact recorded physics digests, all evaluation fields except three wall-clock timings, and every final `createFlightScore` state/diagnostic field. It uses `structuredClone` for each row, so earlier diagnostics cannot be overwritten by later observer updates. No neural, muscle, or native-body simulation is run.
