# Native flight interpreter assay

Generated 2026-09-14T10:52:46.837Z. Reproduce from the repository root with `node scripts/audit-flight-parameter-sensitivity.mjs`. The [JSON report](result.json) pins source/WASM hashes and records paired mechanical witnesses, COM trajectories, and cycle summaries.

All **27/27** coefficients affect controls, native actuator forces, and the body's free response at the tested states. Every named muscle is tested separately on each side. Maximum contralateral command change is 0; no tested sensitivity command or target hits a bound. This demonstrates usable mechanical influence, not successful learning or measured biological coefficients.

Both positive controls use equal normalized wing muscle force 1 with no steering for 500ms. The cold control starts from neutral wings. The warm control restrains the root and nonwing joints during 100ms of native wing motion, then releases everything. After release there are no pose resets or applied root forces. The model has no habitat contacts; descent below the initial height is possible.

| Control | COM rise (cm) | Maximum tilt (degrees) | Maximum angular speed (rad/s) | Maximum cycle mean angular vector (rad/s) | Mean fluid lift / weight | Control clipped steps |
|---|---:|---:|---:|---:|---:|---:|
| Cold onset | -3.290 | 64.9 | 45.33 | 31.89 | 0.955 | 21.92% |
| 100ms restrained warmup | 3.506 | 52.0 | 39.75 | 26.40 | 0.987 | 23.20% |

The current equal-wing command produces aerodynamic lift near bodyweight but does not establish stable level flight. The orientation excursions and large cycle mean angular velocity indicate accumulated rotation in addition to wingbeat recoil. These trials do not evaluate takeoff from support or landing.

Sensitivity uses eight phases, common qpos/qvel across paired perturbations, left/right force 0.6/0.4, and a named steering force of 0.35 on one side at a time. Commands are measured after two 200µs interpreter ticks; native forward dynamics and a 1ms free response follow. Deployment starts at 0.94 for its time-constant assay and 1 otherwise. Bounds and full witness arrays are in the JSON.

Each claw's nominal maximum adhesion is 1.020 bodyweights. This is actuator capacity only: actual claw loading and its causal effect on grounded takeoff were not tested.
