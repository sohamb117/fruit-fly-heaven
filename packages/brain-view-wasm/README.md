# fruit-fly-brain-view-wasm

An independent WebAssembly module for anatomical brain viewers. It maps neural activity onto measured geometry, picks neurons, intersects meshes with planes, and samples cross-sections from 3D microscopy volumes. It contains no FlyWire data, simulation, body model, DOM, or rendering framework.

The C++ kernels compile to a custom SIMD-enabled WASM binary. A browser or Node.js loads the binary through the included ES-module wrapper. Final drawing belongs on the GPU; the example habitat uses Three.js/WebGL. WASM handles the CPU work without replacing the GPU renderer.

## Install and use

```sh
npm install ./releases/fruit-fly-brain-view-wasm-0.1.0.tgz
```

```js
import {createBrainViewModule} from 'fruit-fly-brain-view-wasm';
const view = await createBrainViewModule();
const geometry = view.createGeometry({
  positions: new Float32Array([0,0,0, 10,20,30]),
  neuronIndices: new Uint32Array([17,42]),
  neuronCount: 100,
});

// Supply real outputs from your simulator. Multiple vertices can share an ID.
const colors = geometry.updateActivity({voltage: brain.readActivations()});
const inSlice = geometry.filter({normal:[0,0,1], offset:15, halfThickness:5, mode:'slab'});
// colors: owned RGB Float32Array, one triplet per vertex.
// inSlice: owned Uint32Array of geometry vertex indices.
geometry.dispose();
```

The example above requires a brain with 100 neurons; indices are illustrative. The geometry's neuron indices must match the exact row order of the supplied simulation. Share the anatomy across any number of independent brains and supply the selected brain's activity. Keep colors separate for simultaneous views. Nothing here changes neural time or equations.

Plain browser apps can serve the extracted `dist/` directory and import `dist/index.js` by URL. Bare package imports need a bundler or import map. Keep `core.js`, `core.wasm`, and `index.js` together, or pass `wasmUrl` / `wasmBinary` to the factory. SIMD-capable WebAssembly is required.

## Geometry and activity

`createGeometry({positions, neuronIndices, neuronCount})` copies positions and their neuron-index mapping into WASM. Positions are XYZ `Float32Array` triplets in any consistent spatial unit. Each vertex has one `Uint32Array` neuron index. Neither real-world neuron IDs nor anatomical dimensions are hard-coded.

`updateActivity({voltage, lastSpikeMs?, timeMs?, restMv?, thresholdMv?})` returns RGB colors as an owned `Float32Array`. Voltage and optional last-spike times are dense `Float64Array`s, one entry per simulated neuron. Defaults are −52 mV rest and −45 mV threshold. Use a large negative finite timestamp (e.g. −1e30) for cells with no known spike. The palette turns blue below rest, green near rest, and gold near threshold; spike highlights decay with an 8 ms **neural-time** constant. This is an explicitly chosen display mapping, not a biological measurement of glow or subjective experience.

`filter({normal, offset, mode, halfThickness})` selects vertices in a half-space (`cutaway`), a slab (`slab`), or the entire geometry (`all`, default). The plane is `dot(normal, position) = offset`; half-thickness is in spatial units. Normal/offset are normalized together internally.

`pick({matrix, x, y, width, height, radius, ...clipOptions})` returns the nearest projected vertex index within a pixel radius, or −1. `matrix` is a column-major Float32 projection × view × model matrix. Cursor x/y are normalized device coordinates (−1…1, Y upward); width/height and radius use the same screen-pixel units. Behind-camera, off-screen, and clipped vertices are excluded. Map the returned vertex through your original `neuronIndices` array. Picking is vertex-based, not a ray/surface intersection.

`sliceMesh({positions, triangles, normal, offset})` returns exact triangle/plane intersection line segments as Float32 XYZ endpoint pairs. Triangle indices are Uint32 triplets. Coplanar faces and isolated point touches produce no line. It does not fabricate a filled cap or replace the source surface.

## Microscopy slices

```js
const volume = view.createVolume({
  values: microscopyUint8,
  dimensions: [sizeX,sizeY,sizeZ],
});
const grayscale = volume.samplePlane({
  origin:[0,0,50], u:[1,0,0], v:[0,1,0],
  width:sizeX, height:sizeY,
});
volume.dispose();
```

Values are a single-channel Uint8 volume in X-fastest order: `x + sizeX*(y + sizeY*z)`. Sampling uses trilinear interpolation and returns owned Uint8 pixels in row-major order. Origin is the center of output pixel (0,0); u/v are voxel-index steps per output pixel. Arbitrary oblique planes are supported. Convert world coordinates through your own volume origin and spacing. Out-of-volume samples are zero. Output dimensions are capped at 4096 per side.

## Workers, memory, and lifecycle

The included browser Worker accepts ordered `{requestId, op, args}` requests and responds with `{requestId, result}` or `{requestId, error}`. Serve its complete `dist/` directory and create `new Worker('/view-engine/worker.js', {type:'module'})`.

- `init`: optional factory options as args; call once.
- `createGeometry` / `createVolume`: constructor data as args; returns `{id}`.
- `updateActivity`, `filter`, `pick`, `samplePlane`: `{id, options}`.
- `sliceMesh`: mesh/plane data as args.
- `dispose`: `{id}`.

Typed results transfer to the caller. A worker does not import Three.js or create canvases. All read results are owned copies; memory growth does not invalidate them. Geometry and volume handles own native allocations and must be disposed. The WASM heap can grow to 1 GiB. Disposed handles reject operations.

## Build and verification

Activate Emscripten 4.0.23, then run `bash build.sh` inside this package. The archive includes source, build script, tests, and third-party notices. Run `node --test test/*.test.mjs` to verify known-volume slices, oblique interpolation, exact surface intersections, ID mapping, clipping, picking, ownership, validation, and Worker transfers.

The parent repository's `scripts/benchmark-view-wasm.mjs` benchmarks the actual prepared FlyWire display geometry using clearly labeled synthetic signal values. Browser frame rate also depends on GPU drawing, page layout, and the separate brain simulation.

Runtime and wrapper: GPL-2.0-only. Dataset licensing and attribution remain with their sources. No anatomical dataset is included in the module release.
