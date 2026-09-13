# BANC conductance runtime

`src/index.js` exports `loadBancModel`, `createRuntime`, `WasmBrain`, `WebGPUBrain`, `WasmMuscles`, and `WasmJoints`. The host owns sensory currents and body feedback. The runtime never invents connectivity or behavior commands.

```js
const model = await loadBancModel('/banc-data/');
const runtime = await createRuntime(model, {backend: 'auto', onFallback: console.warn});
const input = new Float32Array(model.manifest.neuron_count); // pA
await runtime.brain.step(10, input, {hunger: .5, insulin: .2, akh: .6}, true);
const state = await runtime.brain.readState(new Uint32Array([0, 1]));
runtime.dispose();
```

Only one operation may be in flight per GPU instance. `step` accepts 1–128 fixed time steps and a full current vector. State is independent between instances. `readState` returns eight Float32 values per requested neuron: voltage (mV), adaptation (pA), refractory remainder (ms), cumulative spike count, exponentially filtered release (Hz-equivalent), instantaneous release (ms⁻¹), total conductance (nS), net external/adaptation current (pA). For graded cells, release is continuous and spike count stays zero; their Hz-equivalent field is not a firing rate.

`readState(indices, {includeSpikeTime:true})` adds each cell's last spike time as a ninth value. The returned array also carries `totalSpikes`, `activeEver`, and (when requested) a chronological `spikes` list of up to 16,384 retained events. Each fly records events independently even while unselected. The console displays its latest 256. Simultaneous events have no physiological ordering; an overflowing ring may retain different same-timestep subsets across parallel backends.

Use `WebGPUBrain.create(model, {shared:firstBrain})` or `new WasmBrain(core, model, {shared:firstBrain})` to share immutable wiring. Currents, voltages, kinetics, delays, counters and event history remain independent. Shared wiring remains allocated until the last owner is disposed.

`createRuntime` always initializes WASM for mechanics. Auto fallback occurs only at initialization and reports its cause. GPU device loss stops the instance. The caller must dispose any separately created muscle or joint objects as well as the brain.

The model uses incoming CSR. Offsets contain `2*(N+1)` uint32 values: chemical rows followed by electrical rows, with electrical offsets absolute into the combined edge table. Each 16-byte edge contains source uint32, conductance float32, receptor uint32 and delay-step uint32. Cell parameters are 16 Float32 values in the order recorded in `configs/banc-physiology.json`. All binaries are little-endian. Root IDs never enter the Float32 state.

See the [root README](../../README.md) for preparation, physiology coverage, equations, model limitations and validation. C++ and WGSL equations must be updated together and pass browser parity tests. Compiler/runtime notices are under `third-party-licenses`; the package follows the repository's GPL-2.0-only license.
