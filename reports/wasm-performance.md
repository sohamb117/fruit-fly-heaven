# WASM performance, 0.1.2

The full 100-brain computation was **1.91× faster** than the shipped 0.1.1 release on this local arm64 Mac / Node v24.19.0. Median wall time to advance every brain by 100 neural milliseconds fell from **19.416 s to 10.162 s**, a 47.7% reduction. This is a neural-compute benchmark; the live app also performs sensory updates, readouts, transfers, and rendering.

Three trials per engine used four workers, 25 independent brains per worker, 2 ms round-robin stepping, 100 ms warm-up per brain, and 100 ms measured neural time per brain. Baseline and optimized trials alternated order and never ran concurrently. The live browser simulation stayed paused throughout timing. Graph construction, warm-up, state hashing, and rendering were outside the timed region.

All 138,639 neurons and 15,091,983 connection rows were retained in every brain. Odor-left, odor-right, and sweet input rates were 35, 45, and 150 Hz, with consecutive seeds starting at 2026 and sensory refractory periods set to zero. Model parameters, timestep, Float64 state, input order, and synaptic accumulation order were unchanged.

Each trial ended with 9,469,700 cumulative spikes across the population. Every brain's final membrane voltages, synaptic drives, per-neuron spike counts, and retained spike history were bit-identical across versions and trials. The independent differential script also matched all fields and histories at 1,600 checkpoints over eight graph/model combinations with changing signals, fractional weights, inhibition, tonic current, and slow/near-equal time constants. All 19 simulation/viewer tests passed.

## Changes

- Lossless 18-bit target / signed 14-bit weight packing halves eligible edge storage from eight bytes to four. The measured graph saves 60,367,932 bytes per worker. Larger graphs and fractional/out-of-range weights automatically retain ordinary CSR.
- Hot neuron state resides in a single aligned 64-byte record, reducing scattered loads during delivery, decay, and scheduling.
- Tonic-drive arrays are allocated when first used, saving storage and reads for the common zero-drive case.
- A conservative upper bound rejects responses that cannot reach threshold. Valid prior crossing bounds and immediate-next-tick checks avoid repeated peak searches.
- The app's speed meter excludes pauses.

Summed WASM heap size was 1,677,721,600 bytes for the baseline and 1,512,833,024 bytes for 0.1.2 (9.8% less). These are allocated heap capacities after the workload, including allocator headroom; they are not operating-system RSS measurements.

## Limits and reproduction

This is **not a 1000× gain or real-time simulation of 100 brains**. The median optimized throughput was 0.00984× biological real time for this entire population: approximately 101.6 wall seconds per neural second. Rates, longer-running activity, worker count, hardware, browser scheduling, and rendering can change throughput. The earlier five-millisecond startup benchmark and measurements taken while another simulation was running are not valid A/B comparisons with these results.

From the repository root, with prepared data and the current WASM build present:

```sh
# Pause the browser simulation and other CPU-heavy workloads first.
node scripts/benchmark-brain-performance.mjs 100 100 100 4 3
node scripts/check-brain-equivalence.mjs
node --test packages/fly-brain-wasm/test/*.test.mjs packages/brain-view-wasm/test/*.test.mjs
```

The comparison extracts the immutable `releases/fruit-fly-brain-wasm-0.1.1.tgz` as its baseline. Exact per-trial timings, workload parameters, state digest, engine hashes, and prepared-graph hashes are in `reports/wasm-performance.json`. The updated local release is `releases/fruit-fly-brain-wasm-0.1.2.tgz` (also `.zip` and standalone `.wasm`); nothing was published to a registry.
