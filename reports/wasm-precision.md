# Precision and fast-mode measurements

The optional Fast mode toggle measured **1.17× faster** for 100 full brains on this arm64 Mac under Node v24.19.0. Median neural-computation time fell from **7.962 s to 6.806 s** for a 100 ms neural interval. The population still runs below biological real time. This does not establish the same speedup in the browser or under other signals.

The tradeoff is substantial: fast mode produced **26.48% more spikes** during that interval. End-of-interval membrane voltage mean absolute difference was 0.4511 mV across all neurons, with a maximum difference of 153.41 mV. Many neurons are quiet, so the whole-brain mean can hide larger changes in active circuits. The sum of per-neuron absolute spike-count differences was 28.84% of the reference total. These are numerical comparisons, not biological validation.

FP32 **alone**, keeping the 0.1 ms grid, measured 0.87× reference throughput (median 9.100 s): it was slower in this test. Its population spike-count change was -0.0727%, but individual traces still diverged. A smaller numeric type does not automatically accelerate this event-driven engine. Its workload includes synaptic delivery, random memory accesses, and spike scheduling, rather than only dense arithmetic.

## What the toggle selects

- **Off / Reference:** Float64 state, 0.1 ms threshold grid, 1.8 ms delay, 2.2 ms refractory period.
- **On / Fast, approximate:** Float32 state, 1 ms threshold grid, 2 ms delay and refractory period. Membrane and synaptic time constants remain 20 ms and 5 ms.

Every one of the 138,639 neurons and 15,091,983 directed connection rows remains. Switching restarts all 100 brains, clears activity and traces, preserves the selected fly/neuron and loaded anatomy, and retains pause and sensory-toggle settings. Reference remains the default until the user opts in; the local preference persists across reloads. Trace labels use the selected timestep.

This is **reduced floating-point precision and coarser time quantization**, not INT8 weight quantization. Graph packing remains lossless. Changing the time grid also changes the discretized Poisson stimulus trains, even for the same seeds. Thus the fast-mode difference includes numeric precision, time resolution, delay/refractory changes, and stimulus discretization; it is not a measurement of quantization error alone.

## Benchmark and validation

Three trials per mode used four workers with 25 independent brains each, consecutive seeds from 2026, 100 ms warm-up, and 100 ms measured time. Workers scheduled brains in 2 ms blocks. Odor-left, odor-right and sweet input rates were 35, 45 and 150 Hz, with sensory refractory overrides of zero. Mode order alternated; modes never ran concurrently. The live browser was paused throughout timing. Loading, warm-up, activation reads, state comparison, and rendering were excluded.

Spike differences cover only the measured 100 ms interval; voltage/drive differences are snapshots at its end. The benchmark verifies finite states and repeatable state digests for each mode. Raw per-trial data, active-population readouts, exact preset parameters, and graph/WASM checksums are in [wasm-precision-100.json](wasm-precision-100.json). [wasm-precision-1.json](wasm-precision-1.json) is an earlier single-brain check and is not a substitute for the population result.

All 25 simulation/viewer tests passed. New tests check actual FP32 rounding, metadata, binary mismatch rejection, inhibitory/excitatory signaling, step partitioning, long quiet intervals, slow tonic inputs, and 100-instance Worker isolation for both precisions. The default Float64 engine also matched the previous reference at 1,600 differential checkpoints.

To reproduce with the live app paused:

```sh
node scripts/benchmark-precision.mjs 100 100 100 4 3
node scripts/check-brain-equivalence.mjs
node --test packages/fly-brain-wasm/test/*.test.mjs packages/brain-view-wasm/test/*.test.mjs
```

The generic 0.2.0 release includes both binaries: `core.wasm` and `core-f32.wasm`. Use `createBrainModule({precision:'float32'})` to choose reduced precision independently of the timestep, or omit the option for Float64. Owned activation arrays remain Float64 for API compatibility and report the underlying arithmetic precision. Source, wrapper, types, tests, build scripts, licenses, standalone binaries, and checksums are included in the local release artifacts. No registry publication was performed.
