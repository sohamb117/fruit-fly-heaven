# Current motor-to-wing transfer assay

Executed 73 deterministic cases in 219.09 wall seconds. This is current-model identification, not biological validation or trained flight. Source hashes, every-case outcomes, trace hashes, units and definitions are in [summary.json](summary.json). Compressed traces preserve low-rate samples and selected native-rate bursts without full brain/body dumps.

## Native restrained rig

The root is held by a finite-compliance MuJoCo weld; all other joints stay dynamic. Only equality rows belonging to this weld contribute to the reported restraint reaction. There is no pose reset or applied-force clamp. Aerodynamic torque is transferred from root-local coordinates to the instantaneous whole-body COM before averaging. See [MuJoCo force/constraint equations](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework) and [weld constraints](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld).

| DLM/DVM Hz | Excitation | Late effective power | First beat s | Late lift / weight | Peak abs torque | Max clipped fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 0.0000 | — | -4.0877e-14 | 0.0012348 | 0.0000 |
| 1 | 0.012500 | 0.012500 | 0.046650 | -0.0019122 | 3.7113 | 0.0000 |
| 3 | 0.037500 | 0.037500 | 0.026650 | 0.0029023 | 9.6406 | 0.0000 |
| 5 | 0.062500 | 0.062500 | 0.024650 | 0.0054387 | 15.982 | 0.0000 |
| 8 | 0.10000 | 0.10000 | 0.023650 | 0.030958 | 18.000 | 0.19633 |
| 12 | 0.15000 | 0.15000 | 0.023650 | 0.096852 | 18.000 | 0.20367 |
| 20 | 0.25000 | 0.25000 | 0.022650 | 0.25885 | 18.000 | 0.21233 |
| 40 | 0.50000 | 0.49345 | 0.022650 | 0.59547 | 18.000 | 0.21433 |
| 60 | 0.75000 | 0.72916 | 0.022650 | 0.90502 | 18.000 | 0.21667 |
| 80 | 1.0000 | 0.95750 | 0.022650 | 1.1947 | 18.000 | 0.22000 |
| 100 | 1.0000 | 0.95750 | 0.022650 | 1.1947 | 18.000 | 0.22000 |

“Late” averages the final 0.1 s, while fatigue continues to change; it is not a biological steady state. Step and 150 ms input-ramp cases, DLM-only, DVM-only, left-only, b2 and III1 controls are included in the JSON.

## Free-body cases

Each full-run air/floor case lasts 1.2 s. Zero non-wing input retains passive mechanics and posture servos but excludes active foot adhesion and coordinated takeoff. Air cases start unsupported and cannot count as takeoff. Floor takeoff requires sustained contact loss plus rise; inversion is root upZ < 0.

| Scene | Input | Hz | Qualified takeoff s | Inversion s | Minimum upZ | Peak angular speed rad/s |
| --- | --- | --- | --- | --- | --- | --- |
| air | step | 0 | — | 0.070800 | -0.32493 | 56.563 |
| air | step | 3 | — | 0.10145 | -0.94518 | 72.688 |
| air | step | 12 | — | 0.13435 | -0.77264 | 64.532 |
| air | step | 40 | — | — | 0.18257 | 40.964 |
| air | step | 80 | — | — | 0.20152 | 48.569 |
| air | step | 100 | — | — | 0.20152 | 48.569 |
| air | ramp | 0 | — | 0.070800 | -0.32493 | 56.563 |
| air | ramp | 3 | — | 0.069450 | -0.99998 | 65.284 |
| air | ramp | 12 | — | 0.25845 | -0.81214 | 77.568 |
| air | ramp | 40 | — | — | 0.20769 | 36.914 |
| air | ramp | 80 | — | — | 0.26710 | 47.017 |
| air | ramp | 100 | — | 0.70855 | -0.0055933 | 53.110 |
| flat_floor | step | 0 | — | — | 0.99894 | 2.5399 |
| flat_floor | step | 3 | — | — | 0.99795 | 3.2058 |
| flat_floor | step | 12 | — | — | 0.99856 | 17.654 |
| flat_floor | step | 40 | — | — | 0.99901 | 22.839 |
| flat_floor | step | 80 | 0.073000 | 0.71090 | -0.99999 | 1335.2 |
| flat_floor | step | 100 | 0.073000 | 0.71090 | -0.99999 | 1335.2 |
| flat_floor | ramp | 0 | — | — | 0.99894 | 2.5399 |
| flat_floor | ramp | 3 | — | — | 0.99795 | 3.0881 |
| flat_floor | ramp | 12 | — | — | 0.99835 | 17.536 |
| flat_floor | ramp | 40 | — | — | 0.99876 | 22.904 |
| flat_floor | ramp | 80 | 0.15600 | — | 0.29191 | 56.279 |
| flat_floor | ramp | 100 | 0.12900 | 0.99770 | -0.99997 | 334.60 |

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

No outcome above establishes realistic flight, identifies a biological servo gain, or justifies changing neural excitability. Qualified takeoff can precede instability. Reproduce with `node scripts/audit-motor-wing-transfer.mjs`; pure helper checks: `node --test scripts/test-motor-wing-transfer.mjs`.
