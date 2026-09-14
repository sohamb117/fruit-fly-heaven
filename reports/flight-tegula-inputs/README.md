# Tegula anatomy, native load assay, and opt-in input

The prepared BANC graph contains **26 annotated tegula campaniform sensory neurons omitted from the canonical host sensory inputs**. The anatomical audit and 112-case native load assay are complete. An explicit opt-in now adds these cells through a side-specific aerodynamic-load proxy; the canonical prepared annotations stay unchanged. Neither the assay nor the implementation establishes that this stimulus restores b1 firing or stable flight.

[anatomy.json](anatomy.json) records every prepared index, uint64 BANC ID as a string, zero-based raw metadata row, original anatomical fields, membership checks, and SHA-256 hashes of the raw metadata, prepared IDs, and sensory manifest. Its exact join is `str(ids.bin[index]) == meta.feather.banc_888_id`, with `ids.bin` decoded as little-endian uint64.

| Type | Left | Right | Total |
| --- | ---: | ---: | ---: |
| SNpp28 | 8 | 8 | 16 |
| SNpp37 | 4 | 2 | 6 |
| SNpp38 | 2 | 2 | 4 |
| Total | 14 | 12 | 26 |

Every row has matching `cell_type` and `manc_cell_type`, `body_part_sensory=wing_tegula`, `cell_sub_class=wing_tegula_campaniform_sensillum_neuron`, `cell_function_detailed=mechanical_strain`, and `peripheral_target_type=campaniform_sensillum`. The annotated side agrees with the side of the anterior dorsal mesothoracic nerve.

Lesser, Moussa and Tuthill explicitly associate SNpp28/SNpp37/SNpp38 with tegula campaniform sensilla. Their FANC reconstruction identifies direct input to b1 and other steering motor neurons, but says that the sensory encoding remains unresolved. Peripheral identity and central connectivity do not determine individual strain axes, compression polarity, recruitment threshold, adaptation, or preferred spike phase. [Primary article, Appendix 1 table 1 and Figure 3](https://elifesciences.org/articles/107867#app1)

The independent [prepared-CSR audit](../flight-sensory-observability/tegula-csr/README.md) checks the actual BANC routes, rather than transferring FANC connection counts. It finds direct routes to both b1 neurons, including contralateral inputs. This establishes an available graph route; it does not establish sufficient excitation or a physiological transduction model.

## Completed physical observability assay

The current native body has six dynamic wing hinge coordinates, but no deformable tegula or sensillum. Available measurements are wing angles/velocities, world hinge axes and anchors, and native generalized aerodynamic loads. The three hinges of each leaf wing body share an anchor. Let the rows of `A` be their world axes and let `tau` contain the three corresponding `qfrc_fluid` entries. Then `A M = tau` reconstructs the **aerodynamic moment about the wing hinge**, subject to coincident-anchor and conditioning checks. The three generalized torques must not simply be treated as Cartesian components. The native units are g cm²/s².

This excludes inertial, structural, contact, and other contributions to actual tegula strain. Native `cfrc_int` requires a separate load-balance validation before it can be called the complete wing-base reaction under the current aerodynamic model; see the [native wrench note](../flight-sensory-observability/tegula-csr/NATIVE-WRENCH-NOTE.md).

[audit-tegula-load-observability.mjs](../../scripts/audit-tegula-load-observability.mjs) uses the existing reduced plant and the entire historical synthetic trim, including both b1 forces. The forces bypass neural and muscle dynamics. After 100 ms of root-restrained warm-up, it records 16 frozen wing states at exact 0.5 ms intervals over 7.5 ms. The phases are the actual quantized clock phases, not a synthetic uniformly spaced cycle. State generation takes 2,150 native 50 µs steps in total.

Each frozen state is restored and evaluated at zero root angular velocity and ±1 rad/s about each of the three root-local axes. These are **112 explicit diagnostic `mj_forward` evaluations**, with no integration in the perturbation cases. The startup forward call and forward computations internal to state-generation stepping are additional and separately identified. Applied force arrays remain zero. Inputs, times, resolved native indices, raw moments, load magnitudes, signed contrasts, and reconstruction errors are retained.

The primary analysis uses only raw load. It reports the finite-difference response singular values/rank for the full 0.5 ms series and all four exact 2 ms sampling offsets. No offset is selected for success. Zero-order holding a value does not add independent information. Rank is an outcome, not a pass requirement. The declared numerical threshold is `max(1e-12, 1e-6 * largestSingularValue)` in the raw derivative units, not a physiological detection threshold.

The completed [result.json](load-observability/result.json) contains all **112 evaluations over 16 frozen states**, with `completed=true` and `sourceUnchanged=true`. The geometric, finite-state, contact-free, zero-applied-force, restoration, and source-pin gates passed. Every perturbation preserved its kinematic/control inputs and advanced no time. Maximum hinge-axis condition number was **2.1013**, anchor spread was zero, and maximum reconstruction residual was **1.11e-16 g cm²/s²**. The 2,150 state-generation integration steps and one initialization forward call are separate from the 112 frozen-state evaluations.

Raw magnitude response results (two side measurements per sampled state):

| Sampling | Rows | Rank | Singular values | Condition number |
| --- | ---: | ---: | --- | ---: |
| All 0.5 ms samples | 32 | 3 | 0.00185646, 0.000441634, 0.000268542 | 6.91 |
| 2 ms, offset 0 ms | 8 | 3 | 0.00111012, 0.0000942772, 0.0000660906 | 16.80 |
| 2 ms, offset 0.5 ms | 8 | 3 | 0.000479392, 0.000150341, 0.0000721889 | 6.64 |
| 2 ms, offset 1 ms | 8 | 3 | 0.000540542, 0.000208182, 0.000117821 | 4.59 |
| 2 ms, offset 1.5 ms | 8 | 3 | 0.00130377, 0.000360377, 0.000176788 | 7.37 |

Singular values have units `(g cm²/s²)/(rad/s)`. The full signed-moment-component response also has rank 3, with singular values 0.00190788, 0.00119508, and 0.000629676; these Cartesian components are retained as diagnostics and are **not** fed to neurons.

The magnitude signal distinguishes the prescribed signed angular perturbations across this short sequence of wing states. This is a stacked finite-difference result at one synthetic trim. Each instantaneous input still has only two scalars, so cannot independently specify all three angular components. The rank does not establish a noise margin, neural decoding, observability during arbitrary motion, or closed-loop stability. In particular, the weaker directions and sampling-dependent conditioning remain relevant; 26 equal-within-side inputs do not create 26 independent measurements.

The result SHA-256 is `b674e02b317fff35d637027361686bfe04e22acc21b4dd09781f95535a61a9c6`. Its full source/artifact hashes, resolved native indices, restored states, and the predeclared [plan](load-observability/plan.json) are retained. The completed directory is immutable; another execution must use a new directory.

## Fixed recruitment prior and opt-in implementation

[recruitment-prior.json](recruitment-prior.json) fixes `maxRateHz=100` and `halfLoadNative=0.1390686725503348 g cm²/s²`. The half-load is the median of **all 32** left/right zero-root-rotation magnitudes from the independent synthetic trim, with the sorted reference values and result hash saved. The 100 Hz limit inherits the existing body-input cap. Neither constant was fitted to reward, BANC recruitment, b1 firing, or flight behavior.

The input is `r = 100 L / (0.1390686725503348 + L) Hz`. Every registered cell on one side receives that side's rate, with the same constants across both sides and all three types. This is an isotropic aerodynamic-load/recruitment prior, not measured tegula strain or neuronal tuning. No axis, compression polarity, compensatory motor direction, or preferred phase is assigned.

[banc-tegula.js](../../web/banc-tegula.js) exports `createTegulaSensoryManifest(baseModel, sensory, config)`. It requires verified `ids.bin` data and matches all 26 integer indices, root IDs, IO types, organs, and sides before returning an owned manifest clone with two tegula channels and 26 `wing_strain` transducers. Existing sensory declarations and exclusions cannot overlap them. [sensory-encoder.js](../../web/sensory-encoder.js) routes those channels directly from same-side `feedback.wingLoad`; it never substitutes broad body-motion input, including when native leg feedback is absent. Turning body sensing off produces zero tegula current requests.

The optional native [wing-load sampler](../../web/flybody-wing-load.js) reconstructs each moment from matching `xaxis`, `xanchor`, and `qfrc_fluid` caches. It writes no native state and performs no extra forward/step call. Initial feedback follows the final placement forward call at time zero. After each 1 ms body interval, it captures the final Euler force cache at `bodyTime − 0.00005 s`, before refresh. The encoder samples the latest completed value at the existing 2 ms neural/body boundary, holding its rate during that neural block. Missing, nonfinite, future, or stale load feedback rejects explicitly. Geometry, conditioning, and reconstruction failures also reject.

The six new mapping tests and existing routing/ground-sense tests pass (**23/23**), including the actual prepared identities, clone ownership, rate monotonicity, independent sides, stale/missing feedback, body-sense disabling, and unchanged existing channel rate bytes. The native sampler's [seven data/clock tests](../../web/test/flybody-wing-load.test.mjs) also pass: all 112 saved native observations match the independent reconstruction, disabled observation reads no force caches, observed/unobserved commands and clocks match, and world feedback owns its copied sample. These tests performed no new neural or body simulation. The controlled live BANC opt-in comparison remains a separate validation step; no training or stable-flight result is established here.

Physical load magnitude can retain rotation sign through its interaction with baseline wing loading: if `M(±omega) ≈ M0(phase) ± B(phase) omega`, the difference of squared magnitudes is `4 M0 · B omega`. It can also discard directional information outside the tested operating point. This differs from feeding the magnitude of body angular velocity directly. No desired attitude, reward, or b1-firing target enters the load computation.

Actual 2 ms sensory sampling can lose phase information. Any later 0.5 ms transduction implementation needs an explicit causal scheduling and delay contract; future body loads cannot be replayed into an earlier neural interval. Haltere-based physical encoding would additionally require assumed oscillation geometry, amplitude and phase because native halteres are fixed, and a validated per-cell field/tuning map is still missing.
