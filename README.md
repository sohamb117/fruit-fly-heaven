# Fruit Fly Heaven — BANC v888

The isolated [connectome–FlyBody benchmark](experiments/connectome_body/README.md) compares generic learned adapters across frozen BANC/MaleCNS graphs, topology nulls, random RNNs, and brain-free policies. It includes pinned data preparation, native tasks, capacity/experience sweeps, resumable PPO, and analysis; its validation report states the current evidence.

A browser neuromechanics experiment using the **BANC v888 brain-and-ventral-nerve-cord graph**. Chemical connectivity is preserved across brain, VNC, ascending, descending and motor populations. Annotated **motor neurons**, rather than descending-neuron behavior scores, activate modeled muscle groups. A small actuator adapter connects their forces to the actual FlyBody model in MuJoCo WASM.

The default application runs BANC **inside the original 3D console**. The habitat, movable/popout panels, eye cameras, graded vision, color processing, population controls, anatomy, microscopy and circuit instruments are retained. The original FlyWire backend is available at `/?dataset=flywire` or `/flywire.html`, with its [original documentation](README-flywire.md). The standalone research prototype is isolated at `/banc-lab.html`.

**This implements the experimental substrate, not a validated autonomous food-localization → approach → landing → probing/feeding → takeoff/flight cycle.** Physiology and muscle calibration remain necessary. The stage monitor records observed events and never supplies movement commands. Graded, spiking, hormonal and motor activity are not substituted for successful behavior.

The [existing-controller investigation](reports/existing-fly-control.md) reproduces the original 1.1988-second FlyBody flight trial and a [60-simulated-second bounded hover](reports/flybody-bounded-hover-60s.json). The optional [flight reference](http://127.0.0.1:7842/?controller=flybody-reference&follow=1) now brakes into hover inside the visible bowl, using the original 3D console and an explicitly labeled learned controller/BANC observer role. It starts airborne with ground contacts disabled; it does not establish BANC-driven takeoff, landing or feeding. An isolated [common-body landing/walking teacher](reports/flybody-landing-teacher/README.md) also reproduces a prescribed landing and subsequent walking with exact state continuity; the takeoff handoff remains unsuccessful.

The default mechanical path is now **BANC brain → VNC → annotated motor neurons → muscle activation/force → native FlyBody actuators → MuJoCo contacts and wing forces**. The [native/WASM validation](reports/flybody-runtime-validation.json) checks identical trajectories, isolated joint stimulation, claw grip and motor disconnection. [Earlier surrogate-body trials](reports/banc-one-fly-observation.md) are historical failed behavioral trials, not evidence for the new body.

The extended visual audit exposed a native contact-handle leak that silently stopped the body at 2.738 seconds. Contact vectors and their element handles are now explicitly released; a [five-second controlled comparison](reports/mujoco-contact-memory-comparison.json) retains identical physical trajectories while keeping the native heap constant at 17.56 MB. Runtime exceptions visibly pause the console. The completed [200-frame, 60-minute audit](reports/observation-60min-20260913/) records loaded build changes and separates reference-policy flight, frozen/error intervals, and direct BANC behavior.

The [prior one-fly observation](reports/flybody-flight-repair.md) stayed on the fruit, but its I1-based retraction interpretation has been corrected to III1. Those measurements do not validate current controlled flight. The [earlier transplant observation](reports/flybody-transplant.md), which showed brief intake followed by tumbling, is also historical. Neither establishes the complete requested food-seeking and flight cycle.

## Run

For a fresh checkout, follow the [reproducibility guide](docs/reproducibility.md) to install dependencies and prepare the pinned datasets. Once those assets are ready:

```sh
uv run --python .venv/bin/python python scripts/serve.py --port 7842
```

Open [one fly in the original 3D console](http://127.0.0.1:7842/?population=1&movement=direct&clock=neural&follow=1). The console requests WebGPU for neural computation and uses WASM for muscle dynamics and the MuJoCo solver. If GPU initialization fails, it reports the reason and uses the same neural equations in WASM. The status shows the active backend. Device loss stops the run.

Open the [hosted training console](https://flytrain.morisoba.moe/train.html) for a low-resolution native 3D preview and shared parameter search. Every Start contributes to the GCP coordinator; the matching local client joins the same pool. Each participant runs one complete BANC/FlyBody episode on their own computer, and the coordinator combines paired evaluations into a shared candidate. Opening the page connects metadata only; compute waits for Start. **Download checkpoint** fetches the latest saved parameter vector. See [training and checkpoint instructions](docs/training.md). Successful autonomous behavior is not yet established.

The managed deployment uses Cloud Run and Firestore in project `flyheaven`. Cloud Run serves the website and assigns work; the browser runs the neural and body simulation. Firestore keeps generations, accepted results and checkpoints across container restarts and scale-to-zero. See the [deployment guide](deploy/cloudrun/README.md) for builds, migration, custom-domain setup and backups.

The original **1–100 fly** population control and Reference/Fast switch are retained. A fresh BANC session starts with **one fly**, so the first run can advance perceptibly; an explicitly saved population is respected. BANC uses 0.5 ms reference or 1 ms fast neural steps. The body clock can follow neural time or hold the latest output in Live mode. Direct movement uses motor neurons, muscle forces and native FlyBody joint/contact mechanics in the original 3D habitat. Behavior mode retains the original assisted controller for comparison. Pause, sensory switches, motor coupling, flight controls, fly selection, neuron traces and separate instrument windows remain available. Assisted behavior is not evidence that the physiological controller achieves the target sequence.

## Reproduce preparation

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-banc.txt
.venv/bin/python scripts/prepare-banc.py --download

git clone https://github.com/TuragaLab/flybody.git references/flybody
git -C references/flybody checkout d015e9bfe441bd90ae431bac24c55cb74bdbce26
npm ci --prefix packages/flybody-runtime
.venv/bin/python scripts/prepare-flybody-runtime.py
# Optional older reduced-body lab:
.venv/bin/python scripts/distill-flybody.py
.venv/bin/python scripts/prepare-banc-console.py
.venv/bin/python scripts/prepare-banc-anatomy.py
```

The BANC downloader checks SHA-256 against [the source lock](configs/banc-v888.lock.json); changed upstream files are rejected. About 397 MiB of source data and 221 MiB of runtime data are stored in ignored `data/` directories. The public source is the [BANC project](https://github.com/htem/BANC-project), [data deposit](https://doi.org/10.7910/DVN/7WTH1N), and [v888 documentation](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md). BANC data are CC BY 4.0.

The wing lookup and local steering calibration are included in `models/`. To regenerate them, also clone `https://github.com/FlyRanch/mpc-simulations` into `references/mpc-simulations` and check out `d29a8d3467122addbfcb9929bd57baf4b9c05576`, then follow the [wing repair reproduction commands](reports/flybody-flight-repair.md#reproduction).

The current pinned source produces **175,401 modeled cells, 13,542,180 directed chemical edges, and 42,199,458 synaptic contacts in that edge table**. It excludes 13,107 annotations explicitly marked glia, trachea or not-a-neuron, and 78,685 incident edges. All other annotations and edge endpoints are retained, with no connection-count cutoff. These numbers describe the downloaded snapshot and filtering, not the headline counts of the whole imaging volume. See [exact provenance](reports/banc-provenance.json).

The importer preserves root IDs as strings/uint64. Input, output and anatomy mappings use BANC identities; FlyWire indices are never reused. The trained FlyVis model is shared as a visual front end with new BANC projections. Of 805 annotated motor neurons, **454 map to 135 muscle groups**, including newly retained haltere targets. The other 351 motor neurons remain simulated. Unknown transmitter assignments leave 427,715 edges at zero conductance, with topology and counts preserved and coverage reported.

## Physiology and muscle boundary

All numerical physiology priors and cell/receptor/electrical overrides are editable in [banc-physiology.json](configs/banc-physiology.json). Re-run preparation after editing it.

| Component | Implemented behavior | Evidence boundary |
|---|---|---|
| Cell electrophysiology | Per-cell capacitance, leak, threshold, reset, refractory period and adaptation; class/type rules and individual overrides | Numerical priors, not BANC electrophysiological recordings |
| Chemical kinetics | Delayed transmission; separate receptor rise/decay states and reversal potentials | Contact count-to-conductance gain and kinetic constants assumed |
| Graded vs spiking | Graded photoreceptor and L1/L2/L3 release without spike counts; other cells use adaptive integrate-and-fire | Limited type assignment; defaults remain explicit |
| Receptors | nAChR, GABA_A, GluCl, HisCl, excitatory glutamate override | Presynaptic transmitter does not establish receptor expression; CNS glutamate defaults to GluCl |
| Neuromodulators | Slow dopamine, octopamine, serotonin and tyramine state; cell-specific sensitivity | Local edge-mediated approximation; receptor sensitivities largely unknown |
| Electrical synapses | Independent, bidirectional conductances; live switch | Two GF–PSI type-matched pairs from literature, assumed conductance and symmetry; not extracted from BANC EM |
| Internal state | Finite food, crop filling, absorption, expenditure, hunger/satiety, insulin-like and AKH-like modulation | Normalized phenomenological states and timescales |
| Muscles | Motor-target annotations, activation/deactivation, fatigue, energy availability and force–length–velocity response | Reduced Hill-type model; rate scale and actuator calibration assumed |
| Biomechanics | Native FlyBody inertia, joints, contacts, adhesion, friction, gravity and ellipsoid wing aerodynamics in MuJoCo WASM | Frozen coordinates, bowl heightfield with separate fruit solids, locally fitted wing-actuator mapping and unvalidated free-flight control |

The original chemical network includes bidirectional brain–VNC communication. Head motor neurons that reside in the brain remain there; an artificial VNC relay is not inserted into their observed pathways. No descending neuron is treated as a muscle actuator.

The sensory boundary retains actual rendered binocular RGB images, the trained FlyVis network, and spectral color processing. BANC-specific mappings select DM1/VA2 olfactory cells, sugar-sensitive gustatory cells, mechanosensory populations and visual inputs. Native ground forces now drive separate per-leg load afferents; joint motion, shaft contact and rotation use distinct inputs. This snapshot has no annotated R1–R6, so luminance enters existing graded L1/L2/L3 neurons. T2/T3/T4/T5 receive mapped FlyVis drive and R7/R8 receive color drive. Representative-point retinotopy, opsin mosaics and sensory tuning are explicit assumptions; cells without usable laterality remain in the graph without added visual drive. Direct-mode feeding requires native mouth contact at food, probing and pumping. See the [ground feedback repair and observation](reports/flybody-ground-feedback.md).

[Sensory rate-to-current calibration](reports/banc-sensory-current-calibration.json) now uses isolated instances of the actual configured WASM neuron dynamics, including adaptation, refractory time and graded release. It corrects the previous loss of low-rate inputs; this is an interface calibration, not fitted biological sensory tuning. Haltere and wing-base inputs now respect [organ and side](reports/banc-rotation-gating.md), but their directional and phase tuning remains unresolved. [Proboscis mapping](reports/banc-proboscis-mapping.json) separates rostrum/haustellum extensors from retractors; m8 and frozen labellar motions are not rerouted into extension. The rendered mouth follows the actual native joint anchors and contact ellipsoids.

Direct-mode taste inputs require actual native contact between a fruit solid and the annotated organ: the corresponding tarsus, wing or labellum, on the corresponding side. Floor contact beneath fruit does not stimulate taste. The mapper covers 516 of the 539 selected cells; 23 cells with missing laterality receive no guessed contact input. Sixteen source-labelled sugar/taste-peg annotations omitted by the original bristle-only classifier are supplied with checked source hashes in [the annotation supplement](models/banc-taste-peg-annotations.json). Two conflicting coarse labellar labels are resolved by their exact matched FlyWire identities; one unresolved/mechanical cell is excluded. The 150 Hz contact stimulus remains an assumed sensory tuning value. The encoder also excludes added generic posture drive to 360 explicitly auditory-frequency JO cells because native antennal vibration is not modeled; their neural dynamics and graph connections remain intact. See the [bounded encoder assay](reports/banc-sensory-recruitment/encoder-assay.json).

The native habitat uses [separate fruit collision solids](reports/flybody-habitat-collision-validation.md) matching the original rendered tube and sphere meshes, preserving empty space beneath fruit. Only the bowl remains a heightfield. Initial stance and subsequent support recognize all static habitat geoms. [Native contact tests](reports/flybody-contact-environment-validation.json) distinguish bowl, food and unsupported space; quiet one-second support trials retain all six initial feet and no airborne samples in either original fruit heading or on a flat fixture. These contact checks do not establish controlled powered flight.

The original wing meshes now follow native wing-body matrices and fluid-ellipsoid landmarks, correcting a coordinate conversion that displayed a resting wing nearly vertical. [Native/WASM/render registration](reports/flybody-wing-pose-verification.json) and [reference-controller UI validation](reports/flybody-reference-wing-ui.json) cover this correction. It changes the display, not aerodynamic forces or motor output.

The visual audit also corrected [bowl winding](reports/bowl-winding-validation.json): the original inner floor was culled from above, exposing the lower shell even though native feet contacted the higher surface. Follow-camera fruit occlusion is handled without changing the fly or its retinal view. [Eleven unsupported collision mappings](reports/observation-60min-20260913/contact-modality-audit.md) are now explicitly excluded: ten chemosensory cells and one joint-angle receptor without an assigned position response.

## FlyBody and the actuator boundary

[prepare-flybody-runtime.py](scripts/prepare-flybody-runtime.py) exports a [mesh-free FlyBody](models/flybody-mujoco.xml) from pinned revision `d015e9bfe441bd90ae431bac24c55cb74bdbce26`. Removing visual meshes preserves their exact compiled per-body masses and inertias. The model retains **50 articulated coordinates, freezes 52, and keeps all six root coordinates active**. Its 56 native actuators include six claw-grip actuators. Total mass is **0.0009846214691323196 g**, equal to the source model. [Metadata](models/flybody-mujoco.json) records the source, joint indices, limits, hashes and measured wing cycle. FlyBody's Apache-2.0 [license](models/FlyBody-LICENSE) is retained.

BANC muscle-force differences change native leg position-actuator targets. Long-tendon motor units engage claws, and proboscis groups drive native mouth joints. Power muscles drive a shared thoracic oscillator through FlyBody's position-error/force interface; MuJoCo computes wing motion and fluid forces at 50 µs steps. A lookup table separates wing deployment from beating and calibrates the measured cycle to a body-relative stroke plane at 235.8 Hz. Twelve steering muscle types retain separate effects, fitted locally to published RoboFly force/torque directions. III1-associated retraction competes with basalar opening through an explicit hinge prior. The measured cycle, fitted mechanical transfer, and assumed hinge competition are different evidence levels. This is an explicit reduced actuator model, **not a learned locomotion policy**. Rate saturation at 80 Hz, deployment time constants, hinge competition, joint-target scaling and claw-force substitution remain priors. The adapter receives no desired root speed, altitude, heading, food bearing or behavior stage.

The wing calibration now measures torque about the whole-body center of mass, correcting an earlier balance around the thorax/free-joint origin. The 235.8 Hz frequency, native servo, and prior mean-force curve are retained; power-specific waveform fits and steering responses are recalibrated in the correct frame. This improves isolated mechanics, but the actual BANC loop still overturns. Frozen native antennae now supply zero joint-motion feedback instead of feeding the former display oscillator into proprioception.

[Native kinematic measurements](reports/flybody-leg-sign-audit.json) showed that positive femur/tibia coordinates extend all twelve corresponding hinges. Their BANC flexor signals now use negative native actuator direction, correcting the previous reversal while preserving source annotations. Muscle length decreases in the torque direction; the WASM force–velocity input is positive for shortening. The normalized moment arm remains an assumed constant. Coxa and tarsal motion require further anatomical calibration; a direction was not guessed from an unsigned angle at a straight-joint crossing.

The shared chemical conductance prior is now **0.1 nS·ms per contact**. A [five-value calibration sweep](reports/banc-rest-calibration.json) selected the largest tested gain with quiet motor output under zero external input and retained probing/pump response to sugar. The old 0.5 setting produced spontaneous high-rate wing output. This limited calibration does not establish selective sensory responses or coordinated movement.

The earlier [local inertia/foot surrogate](models/flybody-reduced.json) remains in the optional research lab. Its pose-fit errors do not validate the current body or locomotion. Published walking/flight controllers have not been distilled into this adapter.

## Compute and validation

The [BANC runtime](packages/banc-runtime) uses the same Float32 state layout and equations in C++/WASM and WGSL. JavaScript owns the WebGPU device and dispatches shaders; WASM does not directly execute on the GPU. Incoming-connectivity reduction, receptor kinetics, membrane updates, delayed release and neuromodulation run on WebGPU. Immutable GPU graphs are shared across flies in each worker; delay histories allocate only the slots the graph can address. Motor readouts and population counters use small GPU reductions. The selected fly supplies full voltage, spike and circuit inspection. Every selected-neuron timestep is copied within the GPU command batch, then read back once per block. Muscle integration and native MuJoCo body physics run in WASM.

The graph's largest storage buffer is about 207 MiB, exceeding WebGPU's guaranteed default 128 MiB limit. The runtime checks adapter capabilities and explicitly requests the required limits before allocating. It does not assume every device can hold the full graph.

Rebuild with Emscripten 4.0.23:

```sh
source references/emsdk/emsdk_env.sh
bash packages/banc-runtime/build.sh
```

Checked-in `dist/core.js` and `dist/core.wasm` allow running without the compiler. To install the SDK, clone `https://github.com/emscripten-core/emsdk` into `references/emsdk` and run its installer for `4.0.23` using a native Python interpreter.

```sh
node --test packages/banc-runtime/test/*.test.mjs
.venv/bin/python -m unittest discover -s tests -p test_banc_prepare.py
node --test --test-force-exit --test-concurrency=1 web/test/*.test.mjs
node scripts/verify-flybody-runtime.mjs
# With the static server running, Chrome installed, and Playwright available:
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/verify-transplant-ui.mjs --one-fly
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=1 --seconds=20 --output=reports/flybody-transplant-performance.json
# Observe actual movement and test loss of actuation, without a behavior program:
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --population=1 --ms=2000 --disconnect-ms=300
# Independent equation checks and the optional research lab:
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/verify-banc-browser.mjs
# Also reproduce the compute benchmark and one-second closed-loop observation:
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/verify-banc-browser.mjs --benchmark --long
```

[Console validation](reports/transplant-ui-validation.json) exercises the original 3D instruments and controls with BANC. [Pre-audit one-fly end-to-end benchmarks](reports/transplant-performance-reference-reuse.md) compare both backends after the per-neural-block body feedback correction, including rendered eyes, visual processing and open anatomy inspection. [Equation validation](reports/banc-browser-validation.json) checks actual WebGPU against WASM on synthetic circuits and the independent research lab. [Kernel measurements](reports/banc-compute-benchmark.json) are not the end-to-end acceptance test. Full FlyWire regression tests require the separately prepared old dataset and explicitly skip when absent.

The current [contact, sensory and wing-frame repair comparison](reports/transplant-performance-solid-wing-repair.md) completed two 20-second trials per backend/mode with all eight measurements valid and unchanged source hashes. BANC median throughput is **1.65× slower in Reference and 1.68× slower in Fast**; every matched pair passes the requested factor-of-ten threshold. BANC advances at 0.03868× and 0.04563× real time, respectively. This compares the original UI workloads: BANC uses synchronized MuJoCo and FlyWire retains its kinematic body. It does not establish equal rendered-frame counts or successful behavior. The [current short visual observation](reports/flybody-solid-wing-repair/README.md) still fails the autonomous behavioral target; the flexible-wing experiment is excluded from this production benchmark.

The earlier [post-audit comparison](reports/transplant-performance-visual-audit.md) measured BANC **1.25× slower in Reference and 1.38× slower in Fast**, at 0.0140× and 0.0225× real time. These separate shared-machine runs are not a controlled before/after speed experiment.

Before the extended visual audit, the BANC path measured **2.37× slower in Reference and 2.59× slower in Fast** than the old system with one fly, within the requested factor-of-ten threshold. BANC display throughput was 59.9 and 50.8 frames/s respectively; neural throughput was 0.0341× and 0.0365× real time. These are single 20-second shared-machine samples. The [earlier ground-feedback benchmarks](reports/flybody-ground-performance.md) used the previous rendering-dependent body coupling and do not measure the current synchronization.

The earlier [whole-console comparison](reports/transplant-performance.md), before the behavior repair, measured BANC on an Apple M2 Pro at **1.32–1.72× slower with one fly** and **4.18–5.69× slower with 100 flies**, with **59.3–60.0 display frames/s**. Both Reference and Fast met the requested factor-of-ten throughput threshold in those samples. These were single 20-second measurement windows with matching sensory, rendering and inspection settings; display frame rate does not imply real-time neural simulation or validate the autonomous behavioral sequence. Subsequent testing focuses on observed single-fly behavior; those earlier measurements are not a benchmark of every later change.

On this machine's Apple Metal adapter, the two warmed 10 ms replay batches took 114–120 ms in WebGPU versus 916–932 ms in WASM: approximately **8× faster neural compute**, still below real time. Over the 30 ms replay, motor spike counts matched exactly and maximum motor-voltage difference was 0.000027 mV. This does not establish long-horizon numerical equivalence or full behavioral performance.

The earlier [one-second lab observation](reports/banc-closed-loop.json) remained finite and showed small motor-driven motion toward food. It recorded no ingestion or completed feeding/flight sequence; it is not validation of the restored 3D console. Those controlled surrogate-body tests do not validate native FlyBody flight. Native MuJoCo checks are recorded separately in `reports/flybody-runtime-validation.json`.

Implementation details, equations, assumptions and the remaining scientific work are in [the model notes](docs/banc-model.md).
