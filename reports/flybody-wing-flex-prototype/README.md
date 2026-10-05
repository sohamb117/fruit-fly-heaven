# Passive distal wing-flexion prototype

**Research-only prototype: rejected for production integration.** A single passive bend per wing reproduces one small-load stiffness measurement and preserves neutral mass properties, but the corrected motor replay overturns before its first wing impact and exceeds the measured linear range. No production model or controller was changed. The XML requires the compiled-model fluid overrides described below.

The [corrected replay](mn-replay.json) establishes the following outcomes over 0.4 seconds:

| Model | First wing/environment impact | First inversion |
| --- | --- | --- |
| Original | 0.13805 s | 0.14375 s |
| Rigid split control | 0.13810 s | 0.14430 s |
| Passive flex prototype | 0.15205 s | **0.12420 s, before impact** |

The corrected rigid split matches original root position/quaternion within 3.79 × 10⁻¹³ and root velocity within 1.67 × 10⁻¹⁰ before the original first impact. Later collision responses differ because the mesh collision approximation is not identical to the original ellipsoids. The flex model reaches 2.9993 rad (about 172°) bending and lies outside the source's static range for 5709 of 8000 native steps. Lower peak angular velocity than the original does not constitute recovery. No further tuning or replay was performed after this rejection.

The corrected static probe passed in 2.21 seconds. The `passed` flags in the probe and replay reports refer to their declared reproducibility, source, naming, invariant, and force checks; **they do not mean that the fly behaved successfully.**

The matched native replay in [wing-contact-causal.json](../flybody-solid-wing-repair/after-com/wing-contact-causal.json) justifies investigating impacts: the baseline first strikes the banana at 0.13805 s and inverts at 0.14375 s. Removing only wing/environment collision pairs prevents inversion through 0.4 s. The trajectories are identical before that first impact, when the body is already tilted about 66°. A flexion model cannot be credited with repairing that earlier loss of balance merely by reducing later impact amplification.

## Measured quantity and assumptions

[Wehmann et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6361194/) report 0.024 N/m median stiffness for Drosophila at the end of the third longitudinal vein, load point 2 (Fig. 4C). The range is 0.020–0.031 N/m across three wings, with a linear fit only below about 3.3 µN. This is a local static measurement; stiffness varies substantially across the wing. It supplies neither an anatomical hinge location nor damping or high-frequency dynamics. The paper's typical wing length is about 2.15 mm and example fly mass about 1.6 mg; this FlyBody is about 0.985 mg.

The prototype makes these explicit approximations:

- One chord-axis hinge at the center of the existing wing inertial box. The distal endpoint is a proxy for the measured point; there is no exact vein-to-mesh registration.
- One symmetric linear spring per wing, neutral angle zero. Damping is a declared numerical prior, 0.2 of the critical value calculated from the distal segment's hinge inertia with the proximal segment held. It is not measured damping.
- The original box mass distribution is split into two equal spanwise boxes. Original quaternion and parallel-axis terms preserve aggregate mass, center of mass, and the full inertia tensor at neutral flexion.
- Original collision ellipsoids become clipped convex meshes. Their neutral surface approximation is quantified by a sampled support-function error. A separate rigid split control is generated because primitive-to-mesh collision behavior is not exactly identical.
- The original whole-wing fluid ellipsoid remains attached to the proximal wing. Deformation-dependent aerodynamic loading is absent. **This is not a coupled aeroelastic wing or a drop-in production model.**

New hinges explicitly override inherited wing damping, spring, armature, range enforcement, and friction. No actuator, root constraint, pose correction, or active flex controller is added. The free root and original actuator parameters remain intact. Body-specific contact exclusions are expanded to the distal bodies, including the original parent-weld exclusions.

## Corrected fluid control

The first prototype had a concrete aerodynamic error: new distal bodies acquired MuJoCo's inertia-box fluid fallback, including the welded rigid control. The original full-wing ellipsoid still acted on each proximal wing, adding unwanted extra drag. The failed first models, reports, replay, and scripts are preserved in [before-distal-fluid-correction](before-distal-fluid-correction/manifest.json). Their pre-contact divergence is not evidence against wing flexibility.

[MuJoCo 3.13 source](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_passive.c#L807) selects fluid models separately for each massive body. A positive interaction geom suppresses inertia-box fallback; setting interaction to zero re-enables it. Zero drag coefficients alone retain viscosity and added-mass terms.

The correction marks each existing invisible `wing_left_distal_inertial` / `wing_right_distal_inertial` geom as an ellipsoid-fluid sentinel. **Immediately after compiling either model, set component 0 of that geom's `geom_fluid` row to `1e-300`.** The exact names, stride, component, expected initial value, and required value are in `result.json` → `requiredNativeOverrides`. This is positive-negligible numerical suppression, not mathematical zero. The original proximal fluid geom, geometry, and coefficients remain unchanged. XML cannot encode the fractional interaction because `fluidshape` is boolean. The replay harness explicitly applies and records the override.

The current Euler integrator is required. MuJoCo 3.13's [implicit fluid derivative path](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_derivative.c#L2747) does not scale derivatives by this interaction value, so this override must not be reused with implicit integration without another audit.

The [matched-state audit](fluid-fallback-audit.json) compares all 70 captured states from 0 to 0.138 s by named joint/DoF. The unwanted drag changed root-Y force by up to 0.545681 native units (5.45681 µN). After correction, both rigid and neutral-flex models have **exactly matching `qfrc_fluid`, `qfrc_passive`, spring, and damping forces** on all original DoFs. Shared dense inertia matrix error is at most 1.70 × 10⁻²¹; bias-force error is at most 9.55 × 10⁻¹⁵. This is an instantaneous force-equivalence gate, not a trajectory or contact-equivalence claim.

## Force probe

Run only when other performance measurements have finished:

```sh
uv run --offline --with mujoco --with numpy python scripts/experiment-flybody-wing-flex.py
uv run --offline --with mujoco --with numpy python scripts/experiment-flybody-wing-flex-fluid.py
```

The generated XML preserves the original environment options. The separate force probe disables gravity, fluid, contact, and active actuation to isolate the elastic response, retaining all original passive joints and the free root. Equal and opposite forces act at the same world point on distal and proximal bodies, producing an internal load with zero external wrench. The proximal body is not clamped. After initialization, the native integrator advances every coordinate; no root pose is imposed.

Loads of ±0.25, ±0.5, ±1, and ±2 µN act normal to the proximal wing plane. Deflection is measured relative to that plane. In native g/cm/s units, 0.024 N/m equals 24 g/s²; a 0.114 cm lever gives a rotational spring of 0.311904 g·cm²/s²/rad. Force conversion is 0.1 native force per µN. Checks cover both sides, force sign, convergence, finite state, unchanged neutral mass properties, absence of new actuators/equalities, and zero net applied generalized force on the root.

Results and models: [result.json](result.json), [flex.xml](flex.xml), [rigid-split-control.xml](rigid-split-control.xml). Reproduction script: [experiment-flybody-wing-flex.py](../../scripts/experiment-flybody-wing-flex.py).

| Native check | Result |
| --- | --- |
| Loads | 16 conditions: both wings, ±0.25, ±0.5, ±1, ±2 µN |
| Effective stiffness | 0.0240013–0.0240853 N/m, within 0.356% of the small-angle target |
| Largest tip deflection | 83.038 µm at 2 µN |
| Tail angle peak-to-peak | ≤2.40 × 10⁻¹³ rad |
| Per-wing mass error | Zero |
| Per-wing neutral COM error | ≤1.74 × 10⁻¹⁸ cm |
| Per-wing neutral inertia tensor error | ≤2.65 × 10⁻²³ g·cm² |
| Whole-fly neutral mass and COM errors | Zero |
| New actuators / equality constraints | Zero / zero |
| Net applied generalized force on root | Zero in every native step |
| Free system COM drift during probe | ≤0.260 µm; no position correction |
| Native warnings | Zero |
| Sampled input-mesh support error | ≤0.574 µm; compiled-hull error not measured |

The nonzero effective-stiffness error is expected from finite-angle hinge geometry. The spring remains linear in angle; normal displacement is proportional to sine of angle and force moment arm to cosine of angle.

No dynamic-impact or flight efficacy claim follows from passing the low-load force probe. A later impact assessment requires a rigid split geometry control, free-flapping dynamic validation at the actual wing frequency, and a validated choice for distributed aerodynamic forces and body/render metadata. Recorded millinewton-scale impact peaks lie far outside the source paper's linear load range. The estimated distal mode with the proximal segment held is 675.23 Hz, less than three times a 235.8 Hz wingbeat. Inertial flexion before contact and wingbeat harmonics therefore need explicit measurement; rigid and flexible pre-impact trajectories need not agree.
