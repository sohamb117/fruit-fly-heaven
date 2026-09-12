# Fruit Fly Heaven

A generic WebAssembly connectome runtime, a companion anatomical visualization module, and an example habitat with 100 fruit flies, rotting bananas/apples, live spike monitors, and a brain observatory.

The reusable package is [`packages/fly-brain-wasm`](packages/fly-brain-wasm/README.md). Its API is independent of fruit, flies’ bodies, rendering, input-neuron identities, and dataset size. Compiled release artifacts live in `releases/`; nothing has been published to an external registry.

The independent [`packages/brain-view-wasm`](packages/brain-view-wasm/README.md) package processes anatomy, activity colors, neuron picking, triangular surface cuts, and arbitrary microscopy slices using SIMD WASM. The browser draws the results with WebGL. Neither module bundles its dataset.

Current artifacts: simulation **0.2.0**, anatomical viewer **0.1.0**, and graded vision **0.1.0**. The independent [`fly-vision-wasm`](packages/fly-vision-wasm/README.md) package runs generic graded recurrent networks and exposes their activation arrays; its release contains no dataset. Simulation 0.2.0 includes separate Float64 and Float32 kernels. The default Float64 engine retains the lossless 0.1.2 optimizations and the long-interval decay fix from 0.1.1. Both archives include regression tests and full source.

## Open the WASM habitat

With the local data prepared:

```sh
.venv/bin/python scripts/serve.py --port 7842
```

Open `http://127.0.0.1:7842/`. The Python process only serves static files. At the default population of 100, four browser Workers each load one WASM module and instantiate 25 independent brains; graphs are shared within each worker. The app exposes actual neural time and compute speed. The moving bowl opens first; **Explore brain** opens the observatory, and **Back to bowl** returns.

The main view has a **Population** slider and exact number field for 1–100 flies. Reducing the population creates only the selected number of full brain instances and draws only those flies. Worker count scales down for small populations; both fly selectors and all counters follow the chosen count. Changing the population restarts neural state, preserves pause and sensory settings, and keeps the current fly selected when it remains in range (otherwise the last remaining fly). The choice persists locally.

The main view also has a **Fast mode** toggle. Off uses the reference Float64 / 0.1 ms model; on uses approximate Float32 / 1 ms stepping and 2 ms delay/refractory periods. All connections stay included. Switching restarts the selected population and its traces while keeping anatomy loaded, and the preference is saved locally. Actual trace spacing and active precision are shown. See [precision measurements](reports/wasm-precision.md) for the speed/activity tradeoff; this is not INT8 weight quantization.

## Brain outputs and body actuation

The default **Movement → Behavior signals · walking + flight** restores the original habitat controller from commit `1f9fb8b`. Actual WASM firing rates from DNp09 (walking), DNa01/DNa02 (left/right steering), and proboscis cells (feeding) determine ground movement. Steering activity contributes forward drive even when DNp09 and the direct forward outputs are quiet. Feeding slows walking. This is an explicit behavior-level mapping; it does not claim to decode individual muscles.

The restored rule, in habitat distance units per body second, is:

```text
forward speed = 8 × tanh((walkHz + 0.15 × (leftHz + rightHz)) / 30) / (1 + feedHz / 35)
turn rate     = 2.8 × tanh((leftHz − rightHz) / 40) radians/second
```

**Movement → Direct motor neurons** retains the separate actuator controller: DNg97/DNg100 for forward drive and DNa01/DNa02 for steering. Steering alone turns in place in that mode. Changing Movement applies to all flies without restarting or altering brain state. The selected mode persists through reloads and population/precision changes. The sidebar separates actual neural firing rates from modeled walking/turning commands and realized body speed.

Both modes retain MDN reverse drive, bilateral DNg02 wing power, escape-related takeoff DNs, DNp07/DNp10 landing extension, annotated grooming DNs, and direct proboscis/antennal motor outputs. These **97 identified neurons** supply 12 channels, including the direct forward and steering readouts. [The output manifest](web/motor-outputs.json) lists their model indices, FlyWire root IDs, types, sides, input-file checksums, functional sources, and direct decoder assumptions. Its neuron-ID checksum is checked at startup. DNp09 belongs to the original walking readout in the prepared groups, outside this 97-cell manifest. No brain-and-cord dataset or additional nerve-cord connectivity is loaded.

The worker reads each population's actual spike counts and calculates a 100 ms exponentially smoothed mean firing rate. Direct actuator gains have a 2 Hz dead zone and saturate at 42 Hz; behavior walking/turning use the smooth saturating functions above. Neither body controller injects neural activity, selects destinations, or supplies random movement.

**Behavior mode now includes flight preparation, takeoff, cruise, and landing.** Sustained walking/steering activity accumulates preparation; proboscis activity slows it. Once prepared, a body program supplies a jump and sustained wing motion. Height and vertical-velocity feedback regulate lift, and a bounded flight bout or sufficiently strong landing signal initiates descent. A recovery interval prevents immediate relaunch. These gains and timers are explicit host assumptions in `FLIGHT_PROGRAM`, not decoded motivation, metabolism, or a reconstructed VNC. Quiet brains do not prepare flight. The UI shows preparation, flight phase and commanded wing power separately from actual DNg02 spike rates. Direct mode omits this program; its jump still requires a rising neural command.

The lightweight body model supplies the conversion from rate to force, joint patterns, surface support, drag, gravity, and a bowl boundary. These gains are explicit assumptions in `BEHAVIOR_DECODER` and `MOTOR_DECODER` in [body-world.js](web/body-world.js). Ground velocity approaches the commanded speed through the existing body mechanics. In particular, using landing output to brake wing force is an approximation beyond the measured leg-extension association. Turning is not inferred from geometry, and landing happens where the trajectory meets a surface. Inter-fly collisions, detailed muscle mechanics, and detailed aerodynamics are not implemented.

**Follow neural time** is the default for this controller. Bodies advance with the population neural clock; the optional **Live · held outputs** setting holds the latest motor output on a separate wall-clock body simulation. A live body does not imply a real-time brain. Body poses reach brain workers up to 20 times per wall second, updating left/right odor at the current 3D position and sugar input only on fruit contact. Rendering does not advance neural time.

**Motor coupling** disconnects all actuator inputs, including the behavior mapping, without pausing neural computation. On a supporting surface, zero inputs produce no active movement; airborne bodies may continue to fall or coast under gravity and drag. **Flight** controls whether the wing/jump actuators receive their neural inputs. **Pause habitat** freezes both clocks. Population or Fast mode changes reset bodies alongside brains while preserving movement, coupling, and sensory controls.

Open **How the brain drives this body** below the bowl for each channel's current firing rate, source cell types, and evidence. The detailed circuit inspector continues to expose unmodified neural state. Quiet DNg97/DNg100 cells can coexist with behavior-driven walking, and quiet DNg02 cells can coexist with program-assisted wing motion. The body commands and raw neural readouts are labeled separately. The visual repair below uses a distinct published graded network; it does not silently rewrite the original LIF neurons or graph.

Rebuild the output annotations with `.venv/bin/python scripts/prepare-motor-outputs.py` after preparing the original dataset. Run `node --test web/test/*.test.mjs` for behavior and direct movement, 100-body stillness/motion, flight forces, disconnection, timing, sensory feedback, and circuit inspection. With the local connectome prepared, the suite also runs both complete WASM kernels for 600 neural ms, using food/body feedback and a controlled uniform retinal image. The same measured readouts move the behavior body while the direct shadow body stays in place; switching controllers leaves WASM state unchanged. Full-data tests explicitly skip when the dataset is absent.

## Vision and body-sense feedback

The cameras now feed the trained [FlyVis visual model](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021), ported to the new generic graded WASM runtime. Each eye has **45,669 independent graded neurons and 1,513,231 connections**, covering all 65 type entries in the published specification. Models share immutable wiring within a worker; both eyes and every fly have independent state. The trained parameters are from `flow/0000/000`, with the source revision, checkpoint hash and published archive checksum in [visual-model.json](web/visual-model.json). The full 138,639-neuron FlyWire spiking brain remains unchanged.

This front end transmits continuous, rectified activity through the published recurrent network, using its trained signs, strengths, biases and time constants. It runs the authors' 20 ms Euler update, with one second of neutral-gray initialization. It does not wait for visual neurons to cross an LIF spike threshold. Graded output is sampled at 20 ms neural intervals; the body clock and held camera frames remain explicit, including the approximation made by Live mode.

**14,746 mapped T2/T3 and T4/T5 cells** receive additional visual drive in FlyWire. The [projection manifest](web/visual-projections.json) matches cell type and nearest normalized retinotopic position and lists every receiving root ID. Rectified graded activity is converted to Poisson drive at 20 Hz per model unit, capped at 60 Hz. Those gains and the mapping between specimens are host assumptions. No motor-output neuron receives direct visual stimulation. The original photoreceptor surrogate is retained in the spiking graph, but is no longer the sole path for vision. The new activity panel shows actual graded release in model units; the brain inspector continues to show actual FlyWire spikes and voltages. The two are never substituted for one another.

Controlled tests verify reversed-direction responses, ON/OFF edge responses, independent eyes, signal disconnection, and the numerical kernel against an independent reference. The full-connectome validation additionally compares Vision on/off at matched seeds, checks that downstream motor spike counts change, and runs takeoff, sustained flight and landing in both brain precisions. See [validation measurements](reports/embodiment-validation.json). This establishes tested motion/contrast processing and connectivity into the brain; it is not a validation of complete biological vision or obstacle avoidance.

To reproduce model preparation, obtain the pinned public FlyVis source and its `results_pretrained_models.zip` archive using the authors' downloader, then run `.venv/bin/python scripts/prepare-graded-vision.py`. The script verifies the published archive hash, restricts checkpoint deserialization to tensors, reproduces ordered parameter sharing and convex-hull fill, and checks the checkpoint's synapse counts and signs against the graph. Build/package with `bash scripts/build-wasm.sh` and `.venv/bin/python scripts/release.py fly-vision-wasm`. Run `node --test packages/fly-vision-wasm/test/*.test.mjs web/test/*.test.mjs` and `node scripts/validate-embodiment.mjs`.


Every fly has two **32 × 16 grayscale cameras attached to its head**, rendered from the same Three.js habitat, including the fruit, bowl, and other flies. The observer camera, selection ring, and trails are excluded from sensory input. The **What Fly … sees and senses** panel displays the exact frames last supplied to that fly's brain, with body-time stamps and input rates. Selecting another fly changes inspection, not which brains receive vision.

The [sensory manifest](web/sensory-inputs.json) maps **6,244 existing R1–6 photoreceptors** to visual columns. The mapping follows the [visual atlas authors' method](https://github.com/hsseung/OpticLobe.jl/blob/3352e97c37b0f96ab08f27b70b4420f3d4ac2726/src/columnassignment.jl): assign each R1–6 neuron to the same-side postsynaptic column receiving its greatest summed absolute connection weight. Published p/q column coordinates preserve spatial ordering in both eyes; fitting that lattice to two pinhole images is an optical approximation. The remaining 1,694 R1–6 cells have no usable assignment in these inputs and receive no added visual drive. All stay in the full neural graph. R7/R8 color channels are not directly driven.

In [sensory-encoder.js](web/sensory-encoder.js), luminance supplies a 2–20 Hz baseline with a bounded contrast term and 150 ms adaptation in body time. Real photoreceptors use graded signals; this encoder supplies a **Poisson surrogate to the existing LIF model**, not a validated retinal biophysics model. The cameras have 120° vertical fields of view and look 60° to either side; they do not reproduce measured ommatidial optics or spectral sensitivity.

Body feedback uses the same joint poses that render the legs and antennae. The adapter reports joint angle and speed, a kinematic foot-support estimate, actual body velocity/yaw/tilt, nearby head contact, and a decaying landing-impact signal. Eight bilateral input channels target annotated AN_AVLP ascending populations, wind/gravity Johnston's-organ neurons, BM_Ant bristles, and SA_DLV sensory-ascending neurons. **These are boundary proxies for missing peripheral/VNC computation.** The feature mixtures and rate gains do not establish individual neurons' physiological tuning, and this is not a complete leg proprioceptor circuit. The manifest records this distinction for every channel. No input is injected into a motor-output population and no new connectivity is created.

**Vision** and **Body sense** independently remove those inputs without resetting the brain. Odor and sugar remain separately controllable. Changing population or Fast mode preserves all four switches. Turning a sense off clears its applied drive; it does not instantly erase activity already circulating in the network. Pause freezes both bodies and eye sampling.

Eye sampling targets 20 frames per body second, with at most one fly's two renders per animation frame to bound GPU/readback cost. Workers hold the last sample between updates, and wait for the first eye frame before running when vision is enabled. At high populations or in Live body-clock mode, sampling may fall below the target; the panel exposes the sampled body time. Input vectors are rebuilt only when body samples, eye frames, or switches change. Neural computation remains in the generic WASM artifact; the scene renderer and dataset-specific sensory adapter are host code.

To regenerate the mapping, download the [public v783 column annotations](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz) to `data/raw/column_assignment.csv.gz`, then run `.venv/bin/python scripts/prepare-sensory-inputs.py`. The manifest records SHA-256 checksums for annotations and the exact prepared graph. Run `node --test web/test/*.test.mjs` for spatial/eye isolation, adaptation, sensory switches, actual joint feedback, input mapping validation, and 100 independent WASM sensory states in both precisions. These tests verify implementation behavior, not biological fidelity.

## Anatomical observatory

The view contains a translucent measured neuropil surface, 138,625 annotated neuron anchor locations, 876 complete branching skeletons, and the original downsampled electron-microscopy volume. Fourteen model neurons have no matching coordinates and are omitted spatially. Display sampling never changes simulation connectivity. Anchor locations are typically on a neuron's backbone; they are not all somas.

- Orbit and zoom the 3D brain; drag the slice slider or Shift-drag the scene.
- Switch between XY, XZ, and YZ sections, a cutaway, a thin slab, and the full anatomy.
- Scroll over the microscopy section to zoom; click an anatomical point to inspect its neuron.
- Find a cell by root ID or cell type. Live voltage traces sample that cell every simulation step (0.1 ms in reference mode, 1 ms in fast mode), with explicit spike markers.
- Choose any fly in the active population. Geometry and static scan are shared; signals are taken from that fly's own state.

The microscopy overview has 2.048 × 2.048 × 1.280 µm voxels and cannot resolve individual synapses. Its imagery is static; only the overlaid electrical activity is simulated. Colors do not represent measured optical activity. Coordinates retain the source FlyWire imagery orientation.

To prepare the anatomy after preparing the connectome:

```sh
.venv/bin/python -u scripts/prepare-anatomy.py
```

This downloads public geometry and a 31,569,408-voxel microscopy volume. Annotations are pinned to commit `8587524c1748ce5ef2080822a2fc890fc03bf597`. Every displayed skeleton keeps all of its supplied vertices and edges. Exact source URLs, hashes, missing locations, and geometry selection are recorded in `reports/anatomy-provenance.json`; downloaded data stays out of Git and the release archives.

The full 100-brain model does **not** currently run at biological real time on the development Mac. WASM keeps heavy neural computation away from rendering; it does not remove the computational cost of the full connectome. The UI reports measured speed. Live body motion uses a separate, explicitly displayed clock; all displayed neural activity still comes from the WASM simulation.

## Decode movement

The **Decode movement · trace the quiet circuits** disclosure below the eyes is a read-only live circuit inspector. It follows the selected fly through mapped photoreceptors, L1/L2, selected ON/OFF relays, T4/T5, and LC4/LPLC2 populations. It also exposes all 29 forward-walking and wing-power cells individually: interval spike rate, membrane voltage, distance from the model threshold, and exact root ID. A 100 ms neural window is separate from the actuator's exponentially smoothed rate. Sampling is performed only for the selected fly; changing fly starts a new measurement window, and pausing retains the last sample.

The input lists rank signed connection weight × measured presynaptic firing rate, summed by source cell type and averaged per motor cell. They are **ranking proxies**, not delivered currents or causal proof: they omit delay and refractory losses. The displayed net synaptic drive is read directly from WASM. No neuron input, edge, parameter, or motor command is altered by inspection. The probe manifest checks all four graph hashes, and regeneration preserves every graph array.

The [earlier circuit diagnosis](reports/circuit-diagnosis.md) records 11 controlled conditions across three seeds **before the graded visual bridge and behavior flight program**. Its all-spiking visual path failed to recruit motion populations. Those measurements remain a useful baseline, and the original LIF branch retains those limitations; the new graded network supplies a documented alternative visual path. Artificial interventions in that report run only in diagnostic brains. Current validation is recorded separately in [embodiment-validation.json](reports/embodiment-validation.json).

```sh
.venv/bin/python scripts/prepare-circuit-probe.py
node scripts/diagnose-circuits.mjs 1000 3
node --test web/test/*.test.mjs
```

## Source data

The source model is pinned to `eonsystemspbc/fly-brain` commit `a3db62f9436074e485c0278290c2164ed6150808`. It contains 138,639 neurons and 15,091,983 directed connection rows representing 54,492,922 synapses. Every connection row is retained. FlyWire’s full classification includes 139,255 neurons; the 616-neuron difference comes from the upstream computational model.

The included `scripts/prepare.py` converts the source model to outgoing CSR arrays without filtering its connectivity. It records source and prepared-file SHA-256 hashes, exact dimensions, input/output populations, and the upstream commit in `reports/data-provenance.json`. Raw data, virtual environments, and downloaded toolchains are ignored by Git.

To reconstruct data in a fresh checkout:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p references data/raw
git clone https://github.com/eonsystemspbc/fly-brain.git references/fly-brain
git -C references/fly-brain checkout a3db62f9436074e485c0278290c2164ed6150808
curl -fL https://storage.googleapis.com/flywire-data/codex/data/fafb/783/classification.csv.gz -o data/raw/classification.csv.gz
curl -fL https://storage.googleapis.com/flywire-data/codex/data/fafb/783/consolidated_cell_types.csv.gz -o data/raw/consolidated_cell_types.csv.gz
.venv/bin/python scripts/prepare.py
```

## Build and test the module

Emscripten 4.0.23 is pinned for the release:

```sh
git clone https://github.com/emscripten-core/emsdk.git references/emsdk
.venv/bin/python references/emsdk/emsdk.py install 4.0.23
.venv/bin/python references/emsdk/emsdk.py activate 4.0.23
bash scripts/build-wasm.sh
node --test packages/fly-brain-wasm/test/*.test.mjs
node --test packages/brain-view-wasm/test/*.test.mjs
node scripts/benchmark-wasm.mjs 100 5
node scripts/benchmark-view-wasm.mjs
# Pause the live browser simulation before timing this comparison.
node scripts/benchmark-brain-performance.mjs 100 100 100 4 3
node scripts/check-brain-equivalence.mjs
node scripts/benchmark-precision.mjs 100 100 100 4 3
.venv/bin/python scripts/release.py
.venv/bin/python scripts/release.py brain-view-wasm
```

`reports/wasm-100-brains.json` records the actual full-connectome instantiation/matrix benchmark. The small numerical tests also verify model equations against an independent dense reference, inhibition, stable seeded input streams, step-size partitioning, isolation, and memory ownership. A short benchmark is not evidence of long-run biological fidelity or real-time performance.

`reports/wasm-view-benchmark.json` measures the custom viewer kernels against the actual display geometry. It uses a synthetic voltage ramp solely for repeatable performance measurement, not for the live app. Archive checksums cover both `.tgz` and `.zip` packages and their manifests in `releases/SHA256SUMS`.

The repeatable A/B benchmark in `reports/wasm-performance.json` records the 0.1.2 comparison against the shipped 0.1.1 artifact using 100 full brains, four workers, identical seeded sensory inputs, 100 ms warm-up and 100 ms measured neural time. Trials alternate engine order and exclude loading/rendering. It requires bit-identical voltage, synaptic drive, cumulative spike counts, and retained spike histories for every brain. See [performance notes](reports/wasm-performance.md) for the measured gain and its limits.

## What is modeled

The recorded wiring and annotated neuron identities come from FlyWire. Electrical dynamics, neurotransmitter signs, sensory stimulation rates, and the body decoder involve modeling assumptions. The habitat stimulates annotated food-odor, sugar/water, and mapped visual inputs, including the trained graded visual bridge, plus a documented body-sense boundary approximation. Identified brain activity drives the selected behavior or direct actuator decoder; there is no ventral nerve cord, detailed aerodynamics, learning, or metabolism.

Feeding activity is a neural signal, not a validated measure of happiness or subjective experience. The module exposes electrical values and leaves behavioral interpretation to the host application.

The original CPU prototype remains under `src/brain.cpp`, `src/brain.py`, and `src/server.py` for reference. The active browser example uses `packages/fly-brain-wasm/native/core.cpp` and `web/wasm-world-worker.js`.

## Credits and licenses

- Runtime and application code: GPL-2.0-only; see `LICENSE`.
- FlyWire connectome: Dorkenwald et al., [Nature (2024)](https://www.nature.com/articles/s41586-024-07558-y); dataset attribution and licensing remain applicable.
- Electrical model defaults: Shiu et al. and the [Eon implementation](https://github.com/eonsystemspbc/fly-brain).
- Food-odor input rationale: [Semmelhack & Wang (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2702439/).
- Three.js renderer: MIT, Three.js authors. Vendored files preserve their license headers; see `web/vendor/LICENSE.three`.

Connectome data is excluded from the generic module release and must be obtained under its own license.
