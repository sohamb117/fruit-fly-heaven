# fruit-fly-brain-wasm

A body-independent WebAssembly runtime for connectome-based leaky integrate-and-fire simulations. Supply a directed graph, create independent brains, send electrical signals, and read voltage or spike matrices. The runtime contains no fruit, movement rules, neuron names, dataset-specific dimensions, or happiness score.

The package runs in modern browsers, browser Workers, and Node.js 20+. No runtime dependencies or Python server are required. The `core.wasm` binary is included in the release.

Install the local release with `npm install ./releases/fruit-fly-brain-wasm-0.1.2.tgz`. For a plain browser application, serve the extracted `dist/` directory and import `dist/index.js` by URL; bare package imports require a bundler or import map.

```js
import {createBrainModule, loadConnectome} from 'fruit-fly-brain-wasm';

const module = await createBrainModule();
const data = await loadConnectome('https://your-host/connectome/');
const connectome = module.createConnectome(data);
const brains = connectome.createPopulation(100, {seed: 2026});

brains[0].setPoissonInputs({
  indices: new Uint32Array([12, 44]),
  ratesHz: new Float32Array([80, 120]),
});
for (const brain of brains) brain.step(10); // 10 ms of neural time

const matrix = module.readActivationMatrix(brains, {field: 'voltage'});
// matrix.shape: [100, neuronCount]; one independent brain per row.
// matrix.values[row * neuronCount + column] is voltage in mV.
// matrix.timesMs records the actual neural time of each row.

for (const brain of brains) brain.dispose();
connectome.dispose();
```

The neuron indices above are illustrative. Your environment must choose input and output indices from the annotations for its own dataset. Loading a measured connectome does not establish biological validity of the sensory encoding, neuron equations, or body decoder.

## Graph format

`createConnectome({neuronCount, rowOffsets, targets, weights})` accepts outgoing compressed sparse rows:

- `rowOffsets`: `Uint32Array(neuronCount + 1)`; edges from neuron `i` occupy `[rowOffsets[i], rowOffsets[i+1])`.
- `targets`: `Uint32Array(edgeCount)` of postsynaptic neuron indices.
- `weights`: signed `Float32Array(edgeCount)`. Incoming synaptic drive increments by `weights[e] * weightScaleMv`.
- Any neuron count and directed connectivity are accepted within WASM memory limits. No edges are pruned or inferred. Parallel connection rows remain additive.

`loadConnectome(baseUrl)` is an optional loader for `metadata.json`, `indptr.bin`, `targets.bin`, and `weights.bin`. Arrays are raw little-endian values of the types above. Metadata provides `neurons_per_brain` or `neuronCount`. The source dataset and its licenses are deliberately separate from this package.

The runtime automatically packs edges into four bytes when the graph has at most 262,144 neurons and every weight is an integer in [-8192, 8191]. Other graphs retain the original eight-byte target/weight representation. This is lossless storage: edge order, signed weights, and all connections are preserved. Each brain keeps hot neuron state together in 64-byte records, allocates tonic-drive storage only when needed, and avoids unnecessary threshold searches using conservative bounds and valid prior predictions. Neural timestep, arithmetic precision, and public API are unchanged.

## Inputs and outputs

- `setPoissonInputs({indices, ratesHz, amplitudesMv?})` replaces the Poisson voltage inputs. Default amplitude: 68.75 mV. Rates use a discrete Bernoulli approximation per timestep. Input streams are independently seeded by neuron and remain stable when the stepping block size changes. Duplicate active indices are rejected.
- `clearPoissonInputs()` removes all Poisson inputs.
- `injectVoltage(indices, deltaMv)` delivers instantaneous voltage pulses at the current neural time.
- `setCurrentInputs(indices, driveMv)` sets a constant drive in the membrane equation. Unmentioned neurons retain their previous drive; set a channel to zero to clear it. Values are in equivalent mV, **not electrical amperes**.
- `setRefractoryPeriod(indices, ms)` overrides the refractory duration for selected cells. For source-model sensory stimulation experiments, a zero duration can be appropriate; the runtime does not silently apply that assumption.
- `step(durationMs)` advances an exact multiple of `dtMs`. No simulation time is skipped to catch up with rendering.
- `readActivations({field, indices?})` returns an owned `Float64Array`. Fields are `voltage` (mV), `synapticDrive` (mV), and cumulative `spikeCount`. Omitting indices returns every neuron. Reading does not mutate the simulation.
- `readSpikes()` returns the recent `timesMs`, `neuronIndices`, cumulative `total`, and cumulative `dropped` count. The bounded history is explicit; increase `spikeHistoryCapacity` or sample frequently when recording a full spike train.
- `readActivationMatrix(brains, options)` returns a row-major matrix of brains × selected neurons. It includes field units and per-brain neural times. It copies data so memory growth cannot silently invalidate the returned arrays.

Readouts are electrical model states. A connectome runtime cannot supply a validated happiness, consciousness, or subjective-experience measurement.

## Configurable LIF model

`createBrain()` and `createPopulation()` accept `seed` (uint64 number/bigint) and these parameters:

```js
{
  dtMs: 0.1,
  tauMembraneMs: 20,
  tauSynapseMs: 5,
  restMv: -52,
  thresholdMv: -45,
  resetMv: -52,
  refractoryMs: 2.2,
  delayMs: 1.8,
  weightScaleMv: 0.275,
  spikeHistoryCapacity: 8192,
}
```

Between events the runtime solves `dv/dt = (rest - v + synapticDrive + externalDrive) / tauMembrane` and `dg/dt = -g / tauSynapse` analytically. Threshold checks occur on the configured timestep grid. A spike resets voltage and synaptic drive. State is frozen during refractoriness, and incoming synaptic/voltage inputs to refractory cells are ignored. Delay and refractory durations must be multiples of the timestep; `0 < tauSynapse < tauMembrane` is required.

All cells have the same configured electrical parameters except optional refractory overrides. This release does not implement conductance-based compartment models, synaptic plasticity, per-edge delays, or Hodgkin–Huxley dynamics. “Generic” means reusable across connectomes and environments within this declared model.

## Workers and 100 brains

Compile/load the WASM module once per Worker, then create many brain handles within it. Brains in the same module share immutable graph memory and have independent dynamic state. Separate module instances have separate heaps. Creating 100 literal module instances would unnecessarily duplicate the graph; worker groups are the practical default.

The included `dist/worker.js` exposes an ordered request/response protocol:

```js
// Serve the package's dist/ directory at /engine/ in your host application.
const worker = new Worker('/engine/worker.js', {type:'module'});
worker.postMessage({requestId:1, op:'init'});
// Await each response before using an id returned by it.
worker.postMessage({requestId:2, op:'loadConnectome', args:{url:connectomeUrl}});
// response.result.id identifies the loaded connectome.
worker.postMessage({requestId:3, op:'createPopulation', args:{connectomeId, count:100}});
// response.result contains independent brain ids.
worker.postMessage({requestId:4, op:'stepMany', args:{brainIds, durationMs:10}});
worker.postMessage({requestId:5, op:'matrix', args:{brainIds, options:{field:'voltage'}}});
```

Responses are `{requestId, result}` or `{requestId, error}`. `loadConnectome` also accepts `args.csr`. Brain operations take `args.id`: `setPoissonInputs` (`input`), `setCurrentInputs` / `injectVoltage` (`indices`, `values`), `setRefractoryPeriod` (`indices`, `ms`), `step` (`durationMs`), `activations` (`options`), `spikes`, and `disposeBrain`. Graph disposal uses `disposeConnectome` with `args.id`. Typed results are transferred back to the host.

The module separates neural time from wall time. **100 independent instances is a memory/isolation feature, not a real-time performance guarantee.** Performance depends on graph size, firing density, model parameters, hardware, and worker count. Keep heavy stepping and full matrices off the render thread. There is no learned surrogate, prerecorded activity, or animation fallback.

## Build, test, release

From the repository root, install and activate Emscripten 4.0.23 in `references/emsdk`, then run:

```sh
bash scripts/build-wasm.sh
node --test packages/fly-brain-wasm/test/*.test.mjs
python3 scripts/release.py
```

The release builder creates a versioned `.zip`, npm-installable `.tgz`, manifest, and SHA-256 checksums under `releases/`. The tests cover comparison against an independent dense reference, inhibitory signaling, seed reproducibility, invariance to stepping block size and observation, 100 isolated instances, activation-matrix layout, tonic and impulse inputs, memory ownership, validation, and explicit spike-history truncation.

The archive also includes the native source, JavaScript source, build script, and third-party license notices. To rebuild an extracted package, activate Emscripten 4.0.23 in your shell and run `bash build.sh` from its package directory. Repository tests and the habitat are available in the parent repository.

The local FlyWire dataset adapter and 3D bowl application are separate examples in the parent repository. The reusable runtime never downloads or bundles the FlyWire dataset automatically.

## Licensing

Runtime and wrapper: GPL-2.0-only. The electrical model defaults are informed by Shiu et al. and the Eon/FlyWire computational model; measured connectivity is not part of this release. Users must comply with the license of each dataset they load. FlyWire data used by the local example is attributed separately in its provenance manifest.
