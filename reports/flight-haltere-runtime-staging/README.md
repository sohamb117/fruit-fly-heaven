# Per-tick external-current batching — staged only

Adds `stepSequence(inputs, internal={hunger:0, insulin:0, akh:0}, gaps=true)` to `WasmBrain` and `WebGPUBrain`. `inputs` is an array of 1–128 finite `Float32Array(neuron_count)` vectors, one per neural tick. Internal state and the gap switch are held across the sequence. It returns `undefined` synchronously on WASM, and a promise resolving to `undefined` on WebGPU. Existing `step()` method bodies are byte-for-byte unchanged. The configured neural timestep is preserved; these fixtures use 0.5 ms and the DLM profile retains its 0.1 ms ionic substeps.

The whole input schedule is validated before advancing any tick. GPU inputs are copied into the private schedule before its first dispatch. WASM also snapshots any input backed by its own HEAP before its first tick, so an earlier tick cannot overwrite a future state/current alias. Other WASM inputs are used synchronously without copies. The GPU uses one upload carrying aligned uniforms and all current vectors, one submission, and one final synchronization. Each tick copies only `neuron_count * 4` bytes into the current region before dispatch; the event-history tail is preserved. DLM's final ionic-state map provides synchronization and validation. Without DLM, one `onSubmittedWorkDone()` waits for completion.

The private CPU/GPU schedule grows to a power of two as needed, at most 128 ticks, and is reused. Each allocation is `capacity * (256 + neuron_count * 4)` bytes: 2,807,440 bytes at four ticks for 175,401 neurons, or 89,838,080 bytes at 128. There is no eager allocation. The `maxBufferSize` request preserves the original graph/uniform minimum and asks only for optional schedule capacity that the adapter supports; an adapter that fits the original model remains usable. Each sequence checks its rounded allocation against the device limit before allocating, uploading, or advancing state. A larger sequence can be rejected while `step()` and shorter sequences continue to work. The storage-binding requirement is unchanged. Disposal destroys the schedule buffer and releases its CPU array. Busy, shared device health, timestamp, declaration, and ionic-failure behavior follows the existing API.

## Validation

`result.json` and `test-results.tap` record the full verifier outcome on WASM and the existing pinned Dawn 0.6.0 Metal provider (Apple M2 Pro); the suite contains 25 tests. Six-neuron fixtures cover graded/spiking cells, several receptors, delayed chemical edges, gaps on/off, and the DLM ionic profile. Nonconstant sequences exactly match repeated `step(1)`, and constant sequences exactly match `step(N)`, within each backend, including readout state, events, counters, ionic state, and time. Adversarial WASM current/state aliases match entry-value snapshots. A separate 512-neuron fixture exposes a synthetic 192 KiB limit through wrappers around an actual adapter/device: the graph and 64-tick schedule execute on Metal, while a 128-tick schedule is rejected without changing state or allocations. Dawn retains its larger physical default despite a smaller request, so this fixture tests the declared-limit handling rather than physical hardware exhaustion. Instrumented real GPU calls confirm one upload/submission/final synchronization and current-only copies. Invalid later vectors leave state unchanged; lifecycle and failure guards are covered.

Run from the repository root:

```sh
node reports/flight-haltere-runtime-staging/verify.mjs
git apply --check reports/flight-haltere-runtime-staging/runtime.patch
```

The verifier checks source pins and unchanged original methods, runs the small-neuron tests, regenerates `runtime.patch`, and checks applicability without applying it. The patch contains only the two runtime source changes and the new optional-native test. Files under `original/` preserve the baseline; files under `staged/` are the proposed copies. Support symlinks point at pinned existing fixture/shader/WASM dependencies and are not patch changes.

No app files, native core, shaders, or simulation physics were edited. This does not demonstrate full-fly throughput or improved flight; it makes per-neural-tick current delivery available without four separate synchronization boundaries for a four-tick body block.
