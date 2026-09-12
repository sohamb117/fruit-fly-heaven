# fruit-fly-vision-wasm

Generic graded recurrent networks in Float32 WebAssembly. A model supplies its
incoming CSR graph, resting biases, positive time constants, input indices, and
timestep. Each network owns independent activation state while sharing immutable
model buffers. There is no built-in dataset, camera, body, or FlyWire mapping.

The kernel implements passive point neurons with rectified, instantaneous graded
synaptic release: `dx/dt = (bias + W relu(x) + input - x) / max(tau, dt)`.
Euler stepping and the timestep floor follow the published FlyVis dynamics. The
application must validate its own model, step size, input units and interpretation.
This module does not turn activation values into spikes or claim calibrated vision.

```js
import {createVisionModule} from 'fruit-fly-vision-wasm';
const runtime = await createVisionModule();
const model = runtime.createModel({rowOffsets, sources, weights, bias,
  tauSeconds, inputIndices, dtSeconds: 0.02});
const left = model.createNetwork();
const right = model.createNetwork();
left.step(new Float32Array(model.inputCount).fill(0.5), 10);
const activations = left.readActivations(); // Independent copy, in model units.
left.dispose(); right.dispose(); model.dispose();
```

`reset()` restores biases and zero elapsed time. `step(input, steps)` holds an
input vector over the requested integer number of steps. Invalid array lengths,
indices, CSR offsets, nonfinite values and nonpositive time constants are rejected.
If dynamics diverge, stepping throws; reset or discard that network. Existing
networks retain their model buffers after `model.dispose()`, but no new networks
can then be created. No heap views escape the API.

Build with Emscripten 4.0.23: `bash build.sh`. Run `npm test` for an independent
numerical reference, subthreshold transmission, network isolation and lifecycle
checks. The habitat adds full-model direction/contrast and FlyWire bridge tests.

Related model: [Lappalainen et al., 2024](https://www.nature.com/articles/s41586-024-07939-3),
[FlyVis](https://github.com/TuragaLab/flyvis). The habitat's trained model data are
separate from this artifact; its source revision and checksums are recorded there.
