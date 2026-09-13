# Independent review of the passive wing-flex prototype

Reviewed `scripts/experiment-flybody-wing-flex.py` read-only. No production model or controller changes were made during this review. Native execution was deferred during the matched performance benchmark.

## Static design assessment

The neutral mass split is algebraically correct. Each original uniform inertial box is divided into two equal masses and half-span boxes. Their centers move by opposite one-quarter-span offsets. Keeping the original inertial quaternion and summing the two rotated tensors with the parallel-axis terms recovers the original mass, center of mass, and full inertia tensor. The subsequent native checks passed for each wing and the entire body: tensor errors were at most `2.65e-23 g·cm²` and COM error `1.7e-18 cm`. The ±0.25–2 µN force probes returned `0.0240013–0.0240853 N/m`, no solver warnings, and zero applied root generalized force.

The hinge axis is consistent with the intended direction. With original inertial-frame axes `normal=x`, `chord=y`, and outward `span=sign*z`, setting `axis=sign*y` gives `axis × span = normal` on both sides. Positive bending therefore displaces the distal tip toward the original surface normal.

The stiffness and load conversions are correct. In the model's gram/centimeter/second units, `0.024 N/m = 24 g/s²`. With the 0.114 cm hinge-to-probe lever, the small-angle torsional spring is `24 × 0.114² = 0.311904 g·cm²/s²/rad`. One micronewton is `0.1 g·cm/s²`. A 1 µN static tip load predicts approximately 41.7 µm deflection. At finite angle the probe correctly measures `lever × sin(angle)` while the imposed moment includes the changing lever arm.

The load probe does not conceal a body stabilizer: equal/opposite forces act at the same world point on the distal and proximal segments. Their resultant external force and moment cancel. The free root remains dynamic, and the code checks its applied generalized-force components. Gravity, air, contact, and active actuation are disabled only for this explicit stiffness experiment. Production XML options remain intact.

The additional joint explicitly overrides inherited active-wing stiffness, damping, spring reference, armature, friction loss, and limits. The 0.2 damping ratio is a stated numerical prior. Existing body collision exclusions are expanded to the new segments, including formerly implicit parent-weld exclusions. No actuator or equality constraint is added.

## Evidence boundary and integration decision

The cited measured value concerns a point load at the end of the third longitudinal vein, with an approximately 3.3 µN linear range and three tested Drosophila wings. The model's inertial-box tip is only a proxy for that anatomical point; the study does not establish this hinge location, damping, or a uniform wing spring. [Primary measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6361194/).

The design is appropriate for an **offline contact-flexibility experiment**. It is not yet a validated production flight model:

- Original collision ellipsoids become clipped convex meshes. Compare the flexible result with the emitted rigid-split comparator before attributing any improvement to elasticity. The currently calculated support error is a sampled input-point-cloud error, not a bound on compiled contact geometry.
- The whole aerodynamic ellipsoid remains on the proximal rigid body. Aerodynamic deformation, distributed loads, and the altered force–moment relation are unrepresented. This must remain explicit even if an impact test improves.
- An unlimited linear hinge may absorb collision energy through deflections beyond the measured range. Record peak flex angles, tip displacement, and loading beyond calibration rather than interpreting numerical stability as physiological validation.
- Added joints shift later indices. Before integration, map original joint/actuator/body names, preserve transmission targets, regenerate metadata, and support the distal shape in rendering. Existing production metadata and wing rendering are not a drop-in match.

The required next evidence is: compiled conservation and force-probe checks; then a matched native onset comparison of production, rigid-split, and flexible models with identical recorded motor input and named state restoration. Measure pre-contact divergence, contact impulse, deformation, inversion, applied root forces, and solver warnings. The existing causal test supports a contact-driven full flip and launch; it does not attribute the preceding 66° tilt to the wing collision. A compliant-wing result must preserve that distinction.

## First matched replay: not accepted

`scripts/experiment-flybody-wing-flex-replay.mjs` now restores every existing joint, velocity, activation/control slot, body force buffer, rest pose, body/site metadata field, and actuator target by native name. New bends start at zero. It preserves exact captured habitat asset/geometry text because browser/Node trigonometric results can differ in the final bit for distant wall coordinates; the current helper's height samples and food identities are checked. The actual geom-to-food resolver and real WASM muscles/wing control are used. The original baseline is bitexact for qpos, qvel, controls, and muscle state.

The first rigid-split model overturned at 0.10490 s, before its first wing impact at 0.13495 s. The flexible model overturned at 0.10915 s and reached 1.949 rad bend/1.140 mm normal tip deflection, outside its low-load range on 5783 of 8000 native steps. The original overturned at 0.14375 s, following its 0.13805 s wing impact. None received applied root forces.

This result does not justify integration. In particular, the rigid control changes pre-impact dynamics substantially. A named passive/fluid-force audit is in progress to test whether newly massive distal bodies acquire MuJoCo's inertia-based aerodynamic fallback in addition to the retained whole-wing aerodynamic ellipsoid. This is a specific model-construction hypothesis, not evidence that stiffness tuning will fix the behavior. The first artifacts must remain preserved before any corrected comparator is tried.
