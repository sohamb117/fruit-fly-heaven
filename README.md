# Fruit Fly Heaven

A generic WebAssembly connectome runtime, a companion anatomical visualization module, and an example habitat with 100 fruit flies, rotting bananas/apples, live spike monitors, and a brain observatory.

The reusable package is [`packages/fly-brain-wasm`](packages/fly-brain-wasm/README.md). Its API is independent of fruit, flies’ bodies, rendering, input-neuron identities, and dataset size. Compiled release artifacts live in `releases/`; nothing has been published to an external registry.

The independent [`packages/brain-view-wasm`](packages/brain-view-wasm/README.md) package processes anatomy, activity colors, neuron picking, triangular surface cuts, and arbitrary microscopy slices using SIMD WASM. The browser draws the results with WebGL. Neither module bundles its dataset.

Current artifacts: simulation **0.1.1** and anatomical viewer **0.1.0**. Simulation 0.1.1 fixes an unsigned-negation error in decay outside the exponential lookup table; long quiet intervals now retain finite voltage values. Prefer it over 0.1.0. Both archives include the regression tests and full source.

## Open the WASM habitat

With the local data prepared:

```sh
.venv/bin/python scripts/serve.py --port 7842
```

Open `http://127.0.0.1:7842/`. The Python process only serves static files. Four browser Workers each load one WASM module and instantiate 25 independent brains; graphs are shared within each worker. The app exposes actual neural time and compute speed. The observatory opens first; **Back to bowl** switches to the habitat, and **Explore brain** returns.

## Anatomical observatory

The view contains a translucent measured neuropil surface, 138,625 annotated neuron anchor locations, 876 complete branching skeletons, and the original downsampled electron-microscopy volume. Fourteen model neurons have no matching coordinates and are omitted spatially. Display sampling never changes simulation connectivity. Anchor locations are typically on a neuron's backbone; they are not all somas.

- Orbit and zoom the 3D brain; drag the slice slider or Shift-drag the scene.
- Switch between XY, XZ, and YZ sections, a cutaway, a thin slab, and the full anatomy.
- Scroll over the microscopy section to zoom; click an anatomical point to inspect its neuron.
- Find a cell by root ID or cell type. Live voltage traces sample that cell every 0.1 ms of neural time, with explicit spike markers.
- Choose any of the 100 independent flies. Geometry and static scan are shared; signals are taken from that fly's own state.

The microscopy overview has 2.048 × 2.048 × 1.280 µm voxels and cannot resolve individual synapses. Its imagery is static; only the overlaid electrical activity is simulated. Colors do not represent measured optical activity. Coordinates retain the source FlyWire imagery orientation.

To prepare the anatomy after preparing the connectome:

```sh
.venv/bin/python -u scripts/prepare-anatomy.py
```

This downloads public geometry and a 31,569,408-voxel microscopy volume. Annotations are pinned to commit `8587524c1748ce5ef2080822a2fc890fc03bf597`. Every displayed skeleton keeps all of its supplied vertices and edges. Exact source URLs, hashes, missing locations, and geometry selection are recorded in `reports/anatomy-provenance.json`; downloaded data stays out of Git and the release archives.

The full 100-brain model does **not** currently run at biological real time on the development Mac. WASM keeps heavy neural computation away from rendering; it does not remove the computational cost of the full connectome. The UI reports measured speed. No movement or neural activity is fabricated to conceal a slow simulation.

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
.venv/bin/python scripts/release.py
.venv/bin/python scripts/release.py brain-view-wasm
```

`reports/wasm-100-brains.json` records the actual full-connectome instantiation/matrix benchmark. The small numerical tests also verify model equations against an independent dense reference, inhibition, stable seeded input streams, step-size partitioning, isolation, and memory ownership. A short benchmark is not evidence of long-run biological fidelity or real-time performance.

`reports/wasm-view-benchmark.json` measures the custom viewer kernels against the actual display geometry. It uses a synthetic voltage ramp solely for repeatable performance measurement, not for the live app. Archive checksums cover both `.tgz` and `.zip` packages and their manifests in `releases/SHA256SUMS`.

## What is modeled

The recorded wiring and annotated neuron identities come from FlyWire. Electrical dynamics, neurotransmitter signs, sensory stimulation rates, and the body decoder involve modeling assumptions. The habitat stimulates annotated DM1/VA2 food-odor channels and sugar/water cells. Motor-population activity controls a supplied kinematic body; there is no ventral nerve cord, flight physics, learning, or metabolism.

Feeding activity is a neural signal, not a validated measure of happiness or subjective experience. The module exposes electrical values and leaves behavioral interpretation to the host application.

The original CPU prototype remains under `src/brain.cpp`, `src/brain.py`, and `src/server.py` for reference. The active browser example uses `packages/fly-brain-wasm/native/core.cpp` and `web/wasm-world-worker.js`.

## Credits and licenses

- Runtime and application code: GPL-2.0-only; see `LICENSE`.
- FlyWire connectome: Dorkenwald et al., [Nature (2024)](https://www.nature.com/articles/s41586-024-07558-y); dataset attribution and licensing remain applicable.
- Electrical model defaults: Shiu et al. and the [Eon implementation](https://github.com/eonsystemspbc/fly-brain).
- Food-odor input rationale: [Semmelhack & Wang (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2702439/).
- Three.js renderer: MIT, Three.js authors. Vendored files preserve their license headers; see `web/vendor/LICENSE.three`.

Connectome data is excluded from the generic module release and must be obtained under its own license.
