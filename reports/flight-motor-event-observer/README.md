# Exact wing motor events without changing the fly

The opt-in `onMotorEvents` observer recovered all **1,460 wing-motor spikes** from one live BANC/Dawn-Metal + native FlyBody episode. The matching episode without event observation produced the exact same recorded physical trajectory. Both still failed from excessive rotation at **388 ms**; neither is a flight success.

The paired evaluation records are listed in [the run record](evaluation/run-709cbeba-c183-46ce-8939-55a12b7d7e35.json). The [current analysis](analysis-v2/README.md) separates raw spike counts from the existing 50 ms rate estimate. The earlier `analysis/` directory contains the same numerical results with a subsequently corrected description of the native integration interval; use `analysis-v2/`.

## What was verified

- The 48 actual wing motor neurons were selected from the pinned BANC IO mappings. Individual motor identities and left/right assignments remain separate.
- A fresh time-zero baseline plus 194 consecutive 2 ms packets contains exactly the final cumulative spike total. Event timestamps retain the existing 0.5 ms neural resolution.
- Both runs used the same selection seed (190888), 27 parameters, model assets, backend, and grounded initial condition. The only difference was event observation.
- SHA-256 `66eb8ae0af9883914f36266018dcff3b0a262d83a5a78d684284d458c2980088` matches over 195 rows of 630 float64 values: initialization and every completed 2 ms body block. The digest covers time, qpos, qvel, activation, actuator controls, native muscle state, wingbeat phase, deployment, power, and targets. It does not independently compare every native integration step between samples.
- Active execution took 5.592 s without recording and 5.970 s with recording, excluding setup. This single pair measures observer overhead, not general performance or Safari speed.

The event reader enforces the current timing/refractory contract, validates counts and timestamps, and rejects missing intervals or overruns. It does not depend on the global recent-spike ring. Ten helper tests include comparison against per-tick output from the real WASM neuron kernel; four environment fixture tests cover the untouched default path, callback lifecycle, ownership, and cleanup. Eight offline-analysis tests check malformed or mismatched records.

## What the events reveal

During the short **(100, 280] ms** startup window, individual DLM neurons fired at **83.3–155.6 Hz** by direct spike counting. Both b1 neurons were silent. The left/right b2 neurons emitted 31/28 spikes. These are descriptive results for this one model episode, not a steady-flight physiological fit. High DLM activity therefore is not just an artifact of reading the smoothed rate field.

Reported spike phases are extrapolated from the wingbeat-table clock at the start of each body block. They are not measured hinge or muscle phases, and their concentration does not establish biological entrainment.

## Scope

The body still receives the unchanged legacy motor rates. This implementation supplies the lossless temporal observation needed to calibrate an event-to-muscle interface; it does not install an unvalidated force kernel or a stabilizing controller. Neural membrane/synaptic dynamics, muscle excitation, the public UI, and production hosting were not changed by this observer work. No optimizer jobs were leased or submitted.

The distinct development [bundle](bundle.json) retains the existing zero-adhesion benchmark and its operator-selected power 1.5 / steering 0.05 starting gains. The [plan](plan.json) pins both full-horizon-or-physical-failure evaluations. It is a diagnostic evaluation, not a production checkpoint.
