# Boundary contact ablation

Removing wall and ceiling contacts prevents the retained checkpoint's original terminal spin in seed1290888. It **does not improve its504ms of qualified flight**. With boundaries disabled, the fly later descends, contacts the ground at1.49285s and fails at1.530s.

| Measurement | Original boundaries | Boundary contacts disabled |
|---|---:|---:|
| Terminal time |0.734s|1.530s|
| Angular speed at0.734s |784.01rad/s|21.27rad/s|
| Up component at0.734s |0.2060|0.9892|
| Best continuous qualified flight |0.504s|0.504s|
| Flight/landing success |false|false|

The baseline reproduces the original held-out physics digest exactly. Both arms have the same accumulated630-value physical-state digest and identical objective observations through0.728s, immediately before the baseline's first wall contact at0.72985s. The ablation changes only collision masks for the64 static walls and ceiling, once at initialization. All original task thresholds, arena termination bounds, neural inputs, parameters, ground/fruit contacts, muscles, actuators and initial conditions remain active. No pose resets or applied root forces were added.

This establishes a causal contribution of wall contact to the original terminal spin in this seed. The earlier loss of qualified flight, progressive descent and later ground collision remain unresolved. Longer survival alone is not improved flight. The ablation is diagnostic and has not been enabled in training or deployed.

The frozen plan, complete observations/contact records and runner/helper snapshots are saved beside this report. Recompute the paired analysis with `node reports/flight-boundary-ablation/analyze.mjs`; [result.json](result.json) records exact hashes and measurements. Each full-fly evaluation ran sequentially on the pinned native Dawn/Metal neural backend with native MuJoCo WASM body physics; this was not a Safari observation.
