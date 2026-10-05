# Current motor-to-wing transfer assay

Executed 4 deterministic cases in 2.6754 wall seconds. This is current-model identification, not biological validation or trained flight. Source hashes, every-case outcomes, trace hashes, units and definitions are in [summary.json](summary.json). Compressed traces preserve low-rate samples and selected native-rate bursts without full brain/body dumps.

## Native restrained rig

The root is held by a finite-compliance MuJoCo weld; all other joints stay dynamic. Only equality rows belonging to this weld contribute to the reported restraint reaction. There is no pose reset or applied-force clamp. Aerodynamic torque is transferred from root-local coordinates to the instantaneous whole-body COM before averaging. See [MuJoCo force/constraint equations](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework) and [weld constraints](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld).

| DLM/DVM Hz | Deployment tau s | Excitation | Late effective power | First beat s | Late lift / weight | Peak abs torque | Max clipped fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.012 | 0.0000 | 0.0000 | — | -1.5434e-8 | 0.0012348 | 0.0000 |
| 12 | 0.012 | 0.15000 | 0.14997 | 0.023650 | 0.096366 | 18.000 | 0.18300 |
| 80 | 0.012 | 1.0000 | 0.98932 | 0.022650 | 1.2303 | 18.000 | 0.20000 |

“Late” averages the final 0.1 s, while fatigue continues to change; it is not a biological steady state. Step and 150 ms input-ramp cases, DLM-only, DVM-only, left-only, b2 and III1 controls are included in the JSON.

## Free-body cases

Each full-run air/floor case lasts 1.2 s. Zero non-wing input retains passive mechanics and posture servos but excludes active foot adhesion and coordinated takeoff. Air cases start unsupported and cannot count as takeoff. Floor takeoff requires sustained contact loss plus rise; inversion is root upZ < 0.

| Scene | Input | Hz | Deployment tau s | Qualified takeoff s | Inversion s | Minimum upZ | Peak angular speed rad/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat_floor | step | 80 | 0.012 | 0.073000 | — | 0.87290 | 26.546 |

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

No outcome above establishes realistic flight, identifies a biological servo gain, or justifies changing neural excitability. Qualified takeoff can precede instability. Reproduce with `node scripts/audit-motor-wing-transfer.mjs --pilot=true`; pure helper checks: `node --test scripts/test-motor-wing-transfer.mjs`.
