# Browser training performance

The 2026-09-14 conservative optimization pass improves measured BANC neural and readout throughput by **22.4%**. It reduces elapsed time for that workload by **18.3%**. This is a sequential Node/WASM benchmark on the complete prepared graph, not a Safari or complete FlyBody training speed claim.

The workload has 175,401 neurons, 13,542,180 chemical edges and the ten declared DLM intrinsic cells. Each variant receives identical deterministic synthetic sensory currents and internal-state values, with 0.5 ms neural ticks and 2 ms motor readouts. Three alternating A/B repetitions measure 80 ms after a 20 ms warmup. Integrity checks are outside the timing regions.

| Component | Baseline median | Optimized median |
| --- | ---: | ---: |
| Neural step calls | 8,070.8 ms | 6,619.6 ms |
| Wing event readout | 45.8 ms | 0.69 ms |
| Neural steps plus both motor readouts | 8,147.5 ms | 6,656.3 ms |

The wing readout is approximately 67 times faster in the optimized-runtime paired measurement, but it accounts for less than 1% of the old timed path. Most execution time remains in the full-edge neural kernel. The benchmark does not include MuJoCo, body feedback, browser scheduling, initialization, rendering or network requests.

## Changes and equivalence

- The native kernels calculate unchanged receptor rise/decay factors and delay-history row addresses once per tick, retaining incoming edge order and Float32 arithmetic. No fast-math, timestep or physiology changes are used.
- The selected wing readout omits an unused whole-brain statistics scan and global spike-ring decode/sort. Its selected values and timestamps remain identical. Ordinary motor readouts retain their statistics; public readout defaults remain unchanged. WebGPU supports the same options.
- The training view retains unchanged task lists, plots and trial rows between progress messages, and reuses number formatters. Browser fixtures verify zero mutations in those unchanged sections while new/corrected results and live counters still update. This UI saving is not included in the timing table.

The native differential suite passes 11 tests covering ten cases and 281 complete-state checkpoints: all receptor types, mixed delays, graded cells, gap junctions, DLM, event-ring wrap, mutable receptor constants between calls and DLM failure handling. The complete-graph A/B also compares exact bytes at every 2 ms boundary for both neural state buffers, parameters, history, receptor kinetics, intrinsic states and the event ring. Selected motor values and decoded event packets match. All six runs end with 203,085 total spikes, 315 selected events and the same complete-state digest.

Evidence generated under `reports/training-speed-20260914/`:

- `full-graph-node.json`: samples, medians, input/graph hashes and scope.
- `optimized/equivalence.json`: differential fixture cases and checksums.
- `optimized/build.json`: compiler, flags and binary identities.

Baseline WASM: `9a1fec8f6e798f7ff3b28b8788f868dd59c376afbac5d040c1baba36986a4616`.
Candidate WASM: `b1ebbe46f986f1c2357e479f2b33163e2936526bf59d953db34f37dc854ae48d`.

## Reproduce without replacing the running engine

Preserve the pinned baseline binary before building. Choose a fresh report directory, pause other heavy computation, and run from the repository root:

```sh
mkdir -p reports/banc-speed-check/baseline
cp packages/banc-runtime/dist/core.js packages/banc-runtime/dist/core.wasm reports/banc-speed-check/baseline/
source references/emsdk/emsdk_env.sh
BANC_WASM_OUTPUT_DIR="$PWD/reports/banc-speed-check/candidate" bash packages/banc-runtime/build.sh
BANC_NATIVE_BASELINE="$PWD/reports/banc-speed-check/baseline/core.js" \
BANC_NATIVE_CANDIDATE="$PWD/reports/banc-speed-check/candidate/core.js" \
node --test packages/banc-runtime/test/native-hot-loop-equivalence.test.mjs
node scripts/benchmark-banc-kernel.mjs \
  reports/banc-speed-check/baseline reports/banc-speed-check/candidate \
  reports/banc-speed-check/full-graph.json --blocks=40 --warmup=10 --repeats=3
```

The benchmark rejects a baseline that differs from the pinned browser experiment and refuses to overwrite its output report. It starts no body simulation, leases, uploads or optimizer update. Resume the existing trainer after measuring.

## Release status

This pass is implemented in source and built as an isolated candidate. The shipped `packages/banc-runtime/dist` and hosted training run still use the original binary. Existing model/config hashes pin the implementation, so these changes cannot be slipped into the current cohort under its old identity. A production rollout must explicitly version the implementation and preserve the learned parameter vector and the old run's history; the current preparation tool accepts calibration initializers, not coordinator checkpoints. Complete browser/body timing and that continuation path remain release work.
