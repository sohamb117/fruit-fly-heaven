# Current motor-to-wing transfer assay

Executed 27 deterministic cases in 68.970 wall seconds. This is current-model identification, not biological validation or trained flight. Source hashes, every-case outcomes, trace hashes, units and definitions are in [summary.json](summary.json). Compressed traces preserve low-rate samples and selected native-rate bursts without full brain/body dumps.

## Native restrained rig

The root is held by a finite-compliance MuJoCo weld; all other joints stay dynamic. Only equality rows belonging to this weld contribute to the reported restraint reaction. There is no pose reset or applied-force clamp. Aerodynamic torque is transferred from root-local coordinates to the instantaneous whole-body COM before averaging. See [MuJoCo force/constraint equations](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework) and [weld constraints](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld).

| DLM/DVM Hz | Deployment tau s | Opening span | Excitation | Late effective power | First beat s | Late lift / weight | Peak abs torque | Max clipped fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 0.012 | 0 | 0.010000 | 0.0000 | — | -4.0878e-14 | 0.0012348 | 0.0000 |
| 0.8 | 0.012 | 0.1 | 0.010000 | 0.0000 | — | -4.0878e-14 | 0.0012348 | 0.0000 |
| 0.8 | 0.012 | 0.25 | 0.010000 | 0.0000 | — | -4.0878e-14 | 0.0012348 | 0.0000 |
| 0.81 | 0.012 | 0 | 0.010125 | 0.010125 | 0.087650 | -0.0014151 | 2.6223 | 0.0000 |
| 0.81 | 0.012 | 0.1 | 0.010125 | 0.0000 | — | -4.1060e-14 | 0.0012348 | 0.0000 |
| 0.81 | 0.012 | 0.25 | 0.010125 | 0.0000 | — | -4.1327e-14 | 0.0012348 | 0.0000 |
| 0.9 | 0.012 | 0 | 0.011250 | 0.011250 | 0.054650 | -0.0016529 | 2.9106 | 0.0000 |
| 0.9 | 0.012 | 0.1 | 0.011250 | 0.0000 | — | -4.1705e-14 | 0.0012348 | 0.0000 |
| 0.9 | 0.012 | 0.25 | 0.011250 | 0.0000 | — | -4.1357e-14 | 0.0012348 | 0.0000 |
| 1 | 0.012 | 0 | 0.012500 | 0.012500 | 0.046650 | -0.0019104 | 3.7113 | 0.0000 |
| 1 | 0.012 | 0.1 | 0.012500 | 0.0000 | — | -4.1402e-14 | 0.0013854 | 0.0000 |
| 1 | 0.012 | 0.25 | 0.012500 | 0.0000 | — | -4.1178e-14 | 0.0012348 | 0.0000 |
| 3 | 0.012 | 0 | 0.037500 | 0.037500 | 0.026650 | 0.0029234 | 9.6406 | 0.0000 |
| 3 | 0.012 | 0.1 | 0.037500 | 0.0000 | — | 3.1051e-14 | 0.097473 | 0.0000 |
| 3 | 0.012 | 0.25 | 0.037500 | 0.0000 | — | 7.6251e-15 | 0.017131 | 0.0000 |
| 12 | 0.012 | 0 | 0.15000 | 0.15000 | 0.023650 | 0.096641 | 18.000 | 0.20367 |
| 12 | 0.012 | 0.1 | 0.15000 | 0.15000 | 0.031050 | 0.096641 | 18.000 | 0.20133 |
| 12 | 0.012 | 0.25 | 0.15000 | 0.0000 | — | 3.1029e-14 | 0.32845 | 0.0000 |
| 80 | 0.012 | 0 | 1.0000 | 0.95750 | 0.022650 | 1.1940 | 18.000 | 0.22000 |
| 80 | 0.012 | 0.1 | 1.0000 | 0.95750 | 0.023250 | 1.1940 | 18.000 | 0.22000 |
| 80 | 0.012 | 0.25 | 1.0000 | 0.95750 | 0.024450 | 1.1940 | 18.000 | 0.21933 |

“Late” averages the final 0.1 s, while fatigue continues to change; it is not a biological steady state. Step and 150 ms input-ramp cases, DLM-only, DVM-only, left-only, b2 and III1 controls are included in the JSON.

## Free-body cases

Each full-run air/floor case lasts 1.2 s. Zero non-wing input retains passive mechanics and posture servos but excludes active foot adhesion and coordinated takeoff. Air cases start unsupported and cannot count as takeoff. Floor takeoff requires sustained contact loss plus rise; inversion is root upZ < 0.

| Scene | Input | Hz | Deployment tau s | Opening span | Qualified takeoff s | Inversion s | Minimum upZ | Peak angular speed rad/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_floor | step | 12 | 0.012 | 0 | — | — | 0.99856 | 17.654 |
| flat_floor | step | 12 | 0.012 | 0.1 | — | — | 0.99856 | 17.811 |
| flat_floor | step | 12 | 0.012 | 0.25 | — | — | 0.99900 | 2.5424 |
| flat_floor | step | 80 | 0.012 | 0 | 0.073000 | 0.71090 | -0.99999 | 1335.2 |
| flat_floor | step | 80 | 0.012 | 0.1 | 0.067000 | 0.51780 | -0.99999 | 1748.5 |
| flat_floor | step | 80 | 0.012 | 0.25 | 0.069000 | 0.50740 | -0.99996 | 1123.7 |

## Instrumentation gates

```json
{
  "nativeCoordinateChecks": true,
  "allRequestedCasesRecorded": true,
  "sourceUnchanged": true,
  "noUserAppliedForce": true,
  "completeNumericalRuns": true,
  "completeNativeDurations": true,
  "noNativeInstabilityWarnings": true,
  "restrainedRootWithinOneMicronAndOneMilliradian": true
}
```

No outcome above establishes realistic flight, identifies a biological servo gain, or justifies changing neural excitability. Qualified takeoff can precede instability. Reproduce with `node scripts/audit-motor-wing-transfer.mjs --suite=continuous`; pure helper checks: `node --test scripts/test-motor-wing-transfer.mjs`.
