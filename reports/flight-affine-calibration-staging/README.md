# Staged full-body affine force calibration

The revised **captured-neutral-mouth** calibration is prepared, not stepped.
Six pure numerical tests pass. Preparation validated and copied the frozen
XML/metadata and captured posture without initializing MuJoCo. Root's original
captured-plan attempt stopped at initial `mj_forward`, before any native step.
The diagnosed active mouth/head contact and the revised fixture are described
below. Only explicitly authorized static forwards were used for this diagnosis;
no neural, native-muscle, free-flight or GPU run was performed here.

The purpose is to measure an absolute force-space operating point **T** for the
existing wing basis on the current full body. This is distinct from the captured
event-muscle reference **F0**. A later, separately reviewed affine interpreter
would use `F_eff = T + G(F - F0)` with explicit bounds. Subtracting T alone would
change the neutral wing waveform and is not this calibration's proposed result.
No runtime file, learned parameter, prepared data, model asset or training state
is changed here.

The current prepared input is
[captured-neutral-mouth/plan.json](captured-neutral-mouth/plan.json). It pins
the current executed dependencies, the exact
`reports/flight-haltere-live/legacy/bundle.json`, both model asset hashes, and
`reports/flight-affine-source-capture/calibration-input.json` (SHA-256
`2165a1f19f6ff22cadd9070b01cd937719b363448d590dfc661557075f72f467`). The historical
classical diagnostic is identified as a method reference; none of its numerical
trim values is used.

## Fixed protocol

- **Full unchanged body:** expected nq=57, nv=56, 51 joints and 56 actuators.
  No joint, geometry, actuator, armature or fluid parameter is deleted or edited.
- **Numerical restraint:** root pose is `[0,0,10,1,0,0,0]` in cm/quaternion.
  The 42 leg hinges retain their recorded 100 ms qpos; rostrum and haustellum
  explicitly use metadata.neutral (both zero) with `--neutral-mouth`.
  All their velocities and the root's six velocities are zero. These assignments
  occur before every 50 µs native step and are explicitly rig restraints. The
  six wing hinges remain dynamic; their starting qpos is metadata.neutral.
  There is no free-body release. Captured root/wing positions and velocities are
  not replayed. Without `--capture`, metadata.neutral defines the nonwing pose.
  Without `--neutral-mouth`, the two captured mouth positions are retained.
- Nonwing actuator controls stay at their held hinge positions, bounded by
  native control ranges; adhesion controls are zero. Native actuator state
  starts at reset and evolves normally. Contact at any step or native
  instability warning rejects the assay.
- **24 independent steering forces plus common power:** all inputs lie in
  `[0,1]`. Bias/amplitude gains, power gain and deployment time scale are one.
  Frequency alone is fixed at **227.204212186366 Hz**. Each force goes directly
  into the unchanged `FlyBodyWings` basis; event kernels, native muscle dynamics,
  fatigue, BANC and their attainable-force limitations are bypassed.
- Each measurement resets from the same initial posture, warms for 100 ms, then
  averages approximately 16 wingbeats, rounded to the nearest 50 µs step.
  Wing targets update every 200 µs. The exact cycle/phase rounding is reported;
  warmups share the same phase convention. Half-window mean differences are
  reported as a transient/finite-window check, not a convergence proof.
- Fresh center: all 24 steering forces **0.5**, common power **0.75**. The 51
  fixed measurements are center plus/minus each of 25 channels; force steps are
  0.15 and the power step is 0.05. These define a 4×25 central-difference Jacobian.
  Each column's symmetric curvature at that step is retained.
- Bounded, row-scaled least squares uses 400 coordinate passes and regularization
  `1e-4` toward the fresh center. At most **three measured trim corrections**
  use that one Jacobian; no controller, broad search or gain fitting occurs.
  Rank must be four at a relative singular-value threshold of `1e-7`, and the
  row-scaled condition number must be at most `1e6`, before attempting the fit.
- A repeated center and final candidate check selected native state and full
  per-step wrench/wing traces byte-for-byte. Maximum **56 serial measurements**;
  one MuJoCo model/data instance. `STOP` in the output directory or SIGINT/SIGTERM
  stops at a bounded checkpoint. An exclusive run lock prevents duplicate runs
  from overwriting an existing result or writing a misleading failure file.

## Measurement and acceptance

Immediately after `mj_step`, before any forward or kinematic refresh, read
`qfrc_fluid`, root `xmat/xpos` and `subtree_com`. With R the root-to-world rotation,

```
F_world = qfrc_fluid[0:3]
M_COM_world = R qfrc_fluid[3:6] - (COM_world - root_world) × F_world
M_COM_body = Rᵀ M_COM_world
y = [F_world.z / W, M_COM_body / (W × 0.27 cm)]
```

The sampled force timestamp is `data.time - 0.00005` for the required Euler
integrator. Force units are g·cm/s² (1 native unit = 1e-5 N); torque units are
g·cm²/s² (1 native unit = 1e-7 N·m). `W = native mass in g × 981 cm/s²`.
The native mass is checked against metadata.

The predeclared target is `[1,0,0,0]`, with absolute tolerances
`[0.01, 0.0001, 0.0001, 0.0001]`. Both final candidate and repeat must meet these
tolerances and all integrity gates. Rank failure or an infeasible measured trim
is a completed negative result, not an invitation to change the protocol.
Raw horizontal force remains unconstrained and is reported in body-weight units.
An accepted four-output aerodynamic trim is **not** a six-dimensional hover,
free-flight stability, feedback controllability, takeoff or landing result.
This measures fluid wrench, not total wing-hinge or restraint reaction.

The report distinguishes shape-residual clipping at ±0.15 rad, target joint-range
clipping, servo control clipping, and allocator input saturation. Clipping is
retained as part of the current plant; the measured Jacobian is local and must
not be treated as a globally linear actuator map.

F0 is the captured arithmetic mean of 201 actual normalized native force
snapshots from 100 through 500 ms inclusive. It is a chosen event-force reference,
not physiological rest or mechanical trim. It affects no measurement input here.
The later feasibility of generating T through current event/native muscles and
of using affine residuals during startup remains to be tested separately.

## Commands

Pure tests and preparation:

```sh
node --test reports/flight-affine-calibration-staging/math.test.mjs
node reports/flight-affine-calibration-staging/calibrate.mjs --prepare-only --neutral-mouth --capture=reports/flight-affine-source-capture/calibration-input.json --output=reports/flight-affine-calibration-staging/captured-neutral-mouth
```

After root review and authorization, root can run the same pinned plan:

```sh
node reports/flight-affine-calibration-staging/calibrate.mjs --run --neutral-mouth --capture=reports/flight-affine-source-capture/calibration-input.json --output=reports/flight-affine-calibration-staging/captured-neutral-mouth
```

No native execution was performed to validate native binding calls during
the original staging; the later static diagnostic verified model loading,
pose/controls, `mj_forward`, contact access and force extraction. Native stepping,
frame-cache, finite-state and mass checks remain gates for the authorized run.
The output directory is not
resumable; a failed or completed run stays intact.

## Initial contact and explicit mouth correction

The original captured-plan is preserved, including its failed result/lock.
[original-contact.json](original-contact.json) records its single active contact:
`haustellum_collision` (geom 6, body haustellum 4) against `head_collision`
(geom 2, body head 2), distance **−0.0004435665957 cm**, inclusion margin zero,
`efc_address=0`, `exclude=0`, dimension one, and normal force
**0.4189937423 native units**. This is penetration with a solver constraint,
not merely an inactive proximity contact. It is unrelated to neutral wing qpos.

The revised protocol changes only the two held mouth coordinates to neutral,
retains every captured leg coordinate, and preserves all collision settings.
This is an explicit numerical fixture deviation, not a claim about biological
mouth posture. The stricter `ncon == 0` assay gate remains unchanged.
[captured-neutral-mouth/static-check.json](captured-neutral-mouth/static-check.json)
records one fresh model and one `mj_forward`, zero `mj_step`: **ncon=0, nefc=0,
and qfrc_constraint=0**. The check does not establish that later dynamic wing
motion remains contact-free; the per-step calibration gate still enforces that.
