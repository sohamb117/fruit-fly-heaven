# Current motor-to-wing transfer assay

Executed 17 deterministic cases in 53.468 wall seconds. This is current-model identification, not biological validation or trained flight. Source hashes, every-case outcomes, trace hashes, units and definitions are in [summary.json](summary.json). Compressed traces preserve low-rate samples and selected native-rate bursts without full brain/body dumps.

## Native restrained rig

The root is held by a finite-compliance MuJoCo weld; all other joints stay dynamic. Only equality rows belonging to this weld contribute to the reported restraint reaction. There is no pose reset or applied-force clamp. Aerodynamic torque is transferred from root-local coordinates to the instantaneous whole-body COM before averaging. See [MuJoCo force/constraint equations](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework) and [weld constraints](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld).

| DLM/DVM Hz | Deployment tau s | Excitation | Late effective power | First beat s | Late lift / weight | Peak abs torque | Max clipped fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 0.012 | 0.0093750 | 0.0000 | — | -4.0877e-14 | 0.0012348 | 0.0000 |
| 0.8 | 0.012 | 0.010000 | 0.0000 | — | -4.0877e-14 | 0.0012348 | 0.0000 |
| 0.81 | 0.012 | 0.010125 | 0.010125 | 0.087650 | -0.0014164 | 2.6223 | 0.0000 |
| 0.85 | 0.012 | 0.010625 | 0.010625 | 0.064650 | -0.0015221 | 3.4764 | 0.0000 |
| 0.9 | 0.012 | 0.011250 | 0.011250 | 0.054650 | -0.0016544 | 2.9106 | 0.0000 |
| 12 | 0.012 | 0.15000 | 0.15000 | 0.023650 | 0.096852 | 18.000 | 0.20367 |
| 12 | 0.06 | 0.15000 | 0.15000 | 0.11485 | 0.096893 | 18.000 | 0.16433 |
| 12 | 0.12 | 0.15000 | 0.15000 | 0.22865 | 0.099984 | 18.000 | 0.11467 |
| 80 | 0.012 | 1.0000 | 0.95750 | 0.022650 | 1.1947 | 18.000 | 0.22000 |
| 80 | 0.06 | 1.0000 | 0.95750 | 0.11385 | 1.1947 | 18.000 | 0.17600 |
| 80 | 0.12 | 1.0000 | 0.92989 | 0.22765 | 1.1528 | 18.000 | 0.12233 |

“Late” averages the final 0.1 s, while fatigue continues to change; it is not a biological steady state. Step and 150 ms input-ramp cases, DLM-only, DVM-only, left-only, b2 and III1 controls are included in the JSON.

## Free-body cases

Each full-run air/floor case lasts 1.2 s. Zero non-wing input retains passive mechanics and posture servos but excludes active foot adhesion and coordinated takeoff. Air cases start unsupported and cannot count as takeoff. Floor takeoff requires sustained contact loss plus rise; inversion is root upZ < 0.

| Scene | Input | Hz | Deployment tau s | Qualified takeoff s | Inversion s | Minimum upZ | Peak angular speed rad/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat_floor | step | 12 | 0.012 | — | — | 0.99856 | 17.654 |
| flat_floor | step | 12 | 0.06 | — | — | 0.99857 | 17.122 |
| flat_floor | step | 12 | 0.12 | — | — | 0.99857 | 17.428 |
| flat_floor | step | 80 | 0.012 | 0.073000 | 0.71090 | -0.99999 | 1335.2 |
| flat_floor | step | 80 | 0.06 | 0.24200 | 0.48760 | -0.99860 | 294.87 |
| flat_floor | step | 80 | 0.12 | 0.45600 | 0.51610 | -0.99992 | 1493.0 |

## Instrumentation gates

```json
{
  "nativeCoordinateChecks": true,
  "allRequestedCasesRecorded": true,
  "sourceUnchanged": true,
  "noUserAppliedForce": true,
  "completeNumericalRuns": true,
  "restrainedRootWithinOneMicronAndOneMilliradian": true
}
```

No outcome above establishes realistic flight, identifies a biological servo gain, or justifies changing neural excitability. Qualified takeoff can precede instability. Reproduce with `node scripts/audit-motor-wing-transfer.mjs --suite=followup`; pure helper checks: `node --test scripts/test-motor-wing-transfer.mjs`.
