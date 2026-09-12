# fruit-fly-color-wasm

A portable RGB-to-photoreceptor approximation, plus a generic WASM lookup engine. Runs in browsers, workers and Node 20+. Includes the default model and LUT; no brain dataset or runtime Python dependency is required.

```js
import {createColorModule, loadFlyColorModel} from 'fruit-fly-color-wasm';
const model = await loadFlyColorModel();
const runtime = await createColorModule();
const mapper = runtime.createMapper({...model, maxPixels: 1024});
const rgb = new Uint8Array(32 * 16 * 2 * 3); // packed sRGB bytes, left then right
const captures = mapper.map(rgb); // interleaved Rh3, Rh4, Rh5, Rh6 per pixel
// Supply an existing Float32Array as a second argument to reuse output storage.
mapper.dispose();
```

Serve all `dist/` files together. `loadFlyColorModel(baseURL)` accepts an alternate model directory, verifies the binary SHA-256, and returns its metadata. `createColorModule({wasmUrl, wasmBinary})` accepts an alternate WASM location or bytes. A mapper copies its LUT at creation and returns independent output arrays. Disposal is idempotent; mapping after disposal throws. The stateless mapper can process frames from any number of independent flies sequentially. No camera, brain, spike model or body controller is embedded.

For another species or receptor model, supply `{size, channels, lut}` to `createMapper`. Grid axes are uniformly spaced **linear RGB**, with R-major/G/B order and interleaved output channels. Sizes 2–129 and 1–16 channels are supported. Inputs are always packed sRGB `Uint8Array` values; alpha is not accepted. The engine decodes sRGB with a 256-entry table and trilinearly interpolates the three-dimensional LUT. `map` optionally takes a caller-owned Float32 output buffer. Default capacity is 1,024 pixels; maximum is 1,048,576.

## Default model

1. Decode sRGB to linear RGB.
2. Reconstruct pixel radiance using PBRT v3's Smits-style seven-basis **illuminant** reconstruction and nonnegative spectral clamp.
3. Integrate the reconstructed spectrum against Govardovskii (2000) A1 alpha/beta pigment templates, following drEye, at the reported Drosophila Rh3–Rh6 peaks of 345, 375, 437 and 508 nm.
4. Convert relative radiometric power to photon capture using wavelength weighting. Normalize each receptor to its capture under unit-energy equal-energy illumination over 300–720 nm.

The calculation is precomputed into a 33³ × 4 Float32 LUT (574,992 bytes). The runtime performs only RGB decoding and interpolation. Numerical tests compare it with a full spectral integral at primary, gray, fruit-like and 200 off-grid random colors (absolute relative-capture error below 0.003). Values are relative captures, **not voltages, firing rates, or a reconstruction of subjective fly color**.

The explicit illumination assumption is **zero incident light below 400 nm**. UV-sensitive receptors can still respond weakly through their visible sensitivity tails; this is not reconstructed UV. Their outputs are not renormalized to amplify those tails to a unit response to RGB white. RGB does not uniquely determine a spectrum. The pigment templates omit ocular screening/sensitizing pigments, polarization, receptor adaptation and phototransduction. This model provides color-sensitive input, not a validation of complete fly color vision.

The habitat independently maps these outputs to existing R7/R8 cells and supplies assumed Poisson gains. Those connectome-specific choices live outside this package.

## Sources, building and verification

`model/fly-color-model.json` records exact source revisions, URLs, file hashes, spectral assumptions and LUT hash. Third-party licenses are included. The implementation adapts [PBRT v3](https://github.com/mmp/pbrt-v3/blob/13d871faae88233b327d04cda24022b8bb0093ee/src/core/spectrum.cpp), [drEye](https://github.com/neuralsignal/dreye/blob/b36d2870ec661a34e2cda6d9c5b23ea89eb0dabb/dreye/api/filter_templates.py) and [published fly pigment peaks](https://www.nature.com/articles/s41598-020-74742-1).

From the repository root, regenerate model data with `.venv/bin/python scripts/prepare-color-model.py`; this downloads pinned source files if the reference cache is absent. Build with `source references/emsdk/emsdk_env.sh` then `bash packages/fly-color-wasm/build.sh` (Emscripten 4.0.23). The shipped binaries are ready to use. Run `npm test` inside this package, or `node --test packages/fly-color-wasm/test/*.test.mjs` from the repository root.

`node scripts/benchmark-color.mjs` compares the WASM and equivalent JavaScript lookup, including WASM copies. On the development ARM64 machine, both eyes of one fly took about 0.010 ms in WASM versus 0.013 ms in JavaScript, roughly 1.3× faster. This is a mapping-only benchmark, not a full-simulation speedup.
