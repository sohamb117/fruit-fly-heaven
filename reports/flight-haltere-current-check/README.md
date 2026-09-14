# Isolated rate-to-current transfer check

The DC-calibrated rate interface does not successfully transmit this proposed fast haltere waveform. At0.5ms input updates, all tested bending→requested-rate→current conditions produce zero spikes, despite nonzero mechanical signals. This bridge must not be integrated as a working sensor.

Eight disconnected copies use the exact generic sensory profile present in all328 annotated haltere cells:20pF capacitance,1nS leak,−60mV rest/reset,−45mV threshold,2ms refractory period and the existing adaptation. Each represents one side and one explicitly assumed beam orientation; none is assigned to a BANC receptor identity. Existing production WASM dynamics and DC sensory-current calibration are used unchanged. There is no neural graph, muscle, body or optimizer.

The56 fixed cases compare current legacy unsigned input and proposed rectified bending input, updates every0.5 or2ms, requested-rate ceilings50/100/200Hz, and zero or±20rad/s rotation on each axis. The virtual haltere runs at236Hz and power0.8 for2s; rates are measured over the final1s. A2ms hold at the200Hz ceiling yields12–13Hz at zero rotation, whereas the0.5ms input produces none. That sampling dependence is a warning against treating longer held-current pulses as faithful transduction. The unsigned legacy input gives identical outputs for opposite rotations, as expected.

`plan.json` fixes all choices and source hashes before execution; `result.json` contains full spike times, profiles, current hashes and results. Runtime was about3s. `run.mjs` intentionally refuses to overwrite those files. The [direct mechanical-current assay](../flight-haltere-current-direct/README.md) tests the current-transfer boundary separately without changing neuronal physiology.
