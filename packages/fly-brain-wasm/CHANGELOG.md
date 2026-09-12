# Changelog

## 0.1.0

Initial local release candidate: generic CSR connectome loading, configurable LIF dynamics, independent brain populations, arbitrary electrical inputs, voltage/spike matrices, deterministic per-neuron Poisson streams, and a browser Worker protocol. Includes the compiled WASM binary and TypeScript declarations.

Known limitation: whole-brain real-time performance for 100 instances has not been achieved on the development Mac. This is a computational model, not a validated biological replica.
## 0.1.1

- Fix unsigned negation in long-interval exponential decay. Previously, neurons quiet for more than 10,000 timesteps could produce non-finite state values.
- Add regression coverage beyond the cached exponential horizon.
