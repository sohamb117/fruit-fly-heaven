# Changelog

## 0.2.0

- Add explicit `float32` module selection, a separate SIMD-capable FP32 WASM artifact, and precision metadata on modules, brains, and matrices. Float64 remains the default.
- Preserve graph connectivity, API readout array types, event timestamp precision, and integer spike counters. FP32 state and arithmetic are an approximation and can alter trajectories.
- Verify requested precision against the selected binary; reject duplicate Worker initialization.
- Add precision rounding, binary mismatch, alternate timestep, long-interval, seeded partitioning, and 100-instance Worker coverage for FP32.
- The example app adds a persistent Fast mode toggle using FP32 and a 1 ms grid, with 2 ms delay/refractory periods. Switching restarts the population and clears its traces; geometry stays loaded.

## 0.1.2

- Losslessly pack eligible integer-weight graphs; retain ordinary CSR for fractional weights, large weights, or larger neuron indices.
- Keep hot neuron state in cache-aligned records and allocate tonic-drive storage on demand.
- Reject provably subthreshold responses early and reuse valid spike-time search bounds.
- Preserve the timestep, double-precision state, graph connectivity, input streams, and public API. Differential checks cover all activation fields and spike histories.
- Add regression tests for packed signed boundaries, generic graph fallbacks, and delayed allocation of tonic inputs.

## 0.1.1

- Fix unsigned negation in long-interval exponential decay. Previously, neurons quiet for more than 10,000 timesteps could produce non-finite state values.
- Add regression coverage beyond the cached exponential horizon.

## 0.1.0

Initial local release candidate: generic CSR connectome loading, configurable LIF dynamics, independent brain populations, arbitrary electrical inputs, voltage/spike matrices, deterministic per-neuron Poisson streams, and a browser Worker protocol. Includes the compiled WASM binary and TypeScript declarations.

Known limitation: whole-brain real-time performance for 100 instances has not been achieved on the development Mac. This is a computational model, not a validated biological replica.
