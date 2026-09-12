# Fruit Fly Heaven

A generic WebAssembly connectome runtime and a separate example habitat with 100 fruit flies, rotting bananas/apples, live spike monitors, and a full voltage matrix for the selected brain.

The reusable package is [`packages/fly-brain-wasm`](packages/fly-brain-wasm/README.md). Its API is independent of fruit, flies’ bodies, rendering, input-neuron identities, and dataset size. Compiled release artifacts live in `releases/`; nothing has been published to an external registry.

## Open the WASM habitat

With the local data prepared:

```sh
.venv/bin/python scripts/serve.py --port 7842
```

Open `http://127.0.0.1:7842/`. The Python process only serves static files. Four browser Workers each load one WASM module and instantiate 25 independent brains; graphs are shared within each worker. The app exposes actual neural time and compute speed. Choose a fly, then **Show brain matrix** to see all 138,639 membrane voltages.

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
node scripts/benchmark-wasm.mjs 100 5
.venv/bin/python scripts/release.py
```

`reports/wasm-100-brains.json` records the actual full-connectome instantiation/matrix benchmark. The small numerical tests also verify model equations against an independent dense reference, inhibition, stable seeded input streams, step-size partitioning, isolation, and memory ownership. A short benchmark is not evidence of long-run biological fidelity or real-time performance.

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
