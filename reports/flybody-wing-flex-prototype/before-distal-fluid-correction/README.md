# Passive distal wing-flexion prototype

This isolated experiment adds a single passive bend per wing and reproduces one small-load stiffness measurement without changing the fly's neutral mass properties. The native probe passed in 2.42 seconds. **It is not installed in the app, and it does not demonstrate recovered balance, safe impacts, flight, or feeding.**

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

## Force probe

Run only when other performance measurements have finished:

```sh
uv run --offline --with mujoco --with numpy python scripts/experiment-flybody-wing-flex.py
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
