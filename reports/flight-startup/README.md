# Fixed-table flight startup sweep

Generated 2026-09-14T10:55:17.439Z. Run `node scripts/audit-flight-startup.mjs` from the repository root. [Source-pinned JSON](result.json) includes all 32 outcomes, sampled trajectories, time-window statistics, terminal cycles, and release angular momentum.

Each trial drives both wings with constant normalized muscle force 1, without steering, for 1s in free native dynamics. Twenty-four cold trials sweep eight phases and deployment time scales 0.5/1/2. Eight warm trials use scale 1 and sweep release phase after 100ms of root/nonwing restraint. No restraint or applied root forces remain after release. These are airborne mechanics diagnostics with no habitat contacts, not training or takeoff/landing trials.

0/24 cold and 0/8 warm trials pass the descriptive screen: maximum tilt ≤45°, nonnegative COM rise, and last-quarter mean vertical fluid force ≥0.8 bodyweights. This screen does not substitute for the training flight criterion.

| Onset | Deployment time scale | Phase (degrees) | COM rise (cm) | Maximum tilt (degrees) | Maximum angular speed (rad/s) | Clipped control steps | Final 10-cycle mean COM torque magnitude |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 0.5 | 0 | -6.69 | 88.7 | 54.7 | 22.9% | 6.40e-3 |
| Cold | 0.5 | 45 | -6.95 | 91.3 | 55.5 | 22.9% | 7.74e-3 |
| Cold | 0.5 | 90 | -6.81 | 94.4 | 56.4 | 22.9% | 6.77e-3 |
| Cold | 0.5 | 135 | -6.37 | 91.0 | 55.0 | 23.0% | 8.02e-3 |
| Cold | 0.5 | 180 | -6.63 | 91.9 | 55.8 | 23.0% | 8.12e-3 |
| Cold | 0.5 | 225 | -7.04 | 94.5 | 57.0 | 22.9% | 6.95e-3 |
| Cold | 0.5 | 270 | -6.81 | 94.3 | 56.4 | 22.9% | 6.07e-3 |
| Cold | 0.5 | 315 | -6.37 | 89.5 | 54.3 | 22.9% | 7.00e-3 |
| Cold | 1 | 0 | -6.63 | 88.9 | 54.6 | 22.6% | 2.08e-3 |
| Cold | 1 | 45 | -7.11 | 92.9 | 56.5 | 22.6% | 1.92e-3 |
| Cold | 1 | 90 | -7.22 | 94.0 | 56.8 | 22.6% | 1.73e-3 |
| Cold | 1 | 135 | -6.71 | 90.9 | 55.4 | 22.6% | 1.40e-3 |
| Cold | 1 | 180 | -6.50 | 87.4 | 54.3 | 22.6% | 2.07e-3 |
| Cold | 1 | 225 | -6.75 | 89.5 | 54.8 | 22.5% | 2.64e-3 |
| Cold | 1 | 270 | -6.83 | 90.9 | 55.8 | 22.6% | 8.58e-4 |
| Cold | 1 | 315 | -6.53 | 88.9 | 54.4 | 22.6% | 8.98e-4 |
| Cold | 2 | 0 | -21.20 | 155.9 | 89.9 | 22.0% | 1.15e-3 |
| Cold | 2 | 45 | -20.86 | 156.7 | 89.8 | 21.9% | 8.00e-4 |
| Cold | 2 | 90 | -17.08 | 90.2 | 92.3 | 21.9% | 4.31e-4 |
| Cold | 2 | 135 | -15.64 | 90.4 | 66.8 | 22.0% | 3.27e-3 |
| Cold | 2 | 180 | -20.90 | 125.1 | 82.5 | 21.9% | 5.96e-4 |
| Cold | 2 | 225 | -21.05 | 126.4 | 82.8 | 21.9% | 9.67e-4 |
| Cold | 2 | 270 | -15.68 | 92.5 | 66.5 | 21.9% | 3.43e-3 |
| Cold | 2 | 315 | -16.21 | 90.3 | 74.5 | 21.9% | 1.52e-3 |
| Warm | 1 | 0 | 3.51 | 78.2 | 50.1 | 23.3% | 5.20e-3 |
| Warm | 1 | 45 | -4.61 | 81.5 | 91.2 | 23.2% | 1.06e-3 |
| Warm | 1 | 90 | -3.75 | 91.1 | 54.5 | 23.2% | 6.36e-3 |
| Warm | 1 | 135 | -9.00 | 96.9 | 56.2 | 23.3% | 2.90e-3 |
| Warm | 1 | 180 | -1.04 | 89.6 | 54.9 | 23.3% | 5.32e-3 |
| Warm | 1 | 225 | -4.57 | 94.7 | 55.6 | 23.2% | 5.24e-3 |
| Warm | 1 | 270 | -2.12 | 96.4 | 57.1 | 23.3% | 4.59e-3 |
| Warm | 1 | 315 | 1.33 | 90.1 | 54.0 | 23.3% | 3.06e-3 |

COM torque is in g·cm²/s² and refers to fluid forces only. It is transformed using the native root orientation and shifted to whole-body COM; terminal values are means across the final ten wing cycles. Angular momentum is the native rigid-body subtree result, about COM in world axes. Joint armature values are retained in JSON; this quantity should not be described as including unmodeled rotor angular momentum.
