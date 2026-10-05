# Frozen-brain interpreter diagnostic

Eight interventions were specified before execution in `plan.json`, then evaluated sequentially with one real BANC v888 network and native MuJoCo body in Safari/WebGPU. The original training UI was not redesigned; this separate development page records diagnostic evidence. No optimizer ran and nothing was uploaded to the production coordinator.

All eight trials failed by excessive rotation. None completed takeoff or accrued qualified sustained flight. `passed: true` in the raw report means the runtime, provenance, clock, pause/cancel/stop checks passed; it is not behavioral success.

| Intervention | Simulated seconds before failure | Eligible departures | Best qualified flight seconds |
| --- | ---: | ---: | ---: |
| baseline | 0.170 | 0 | 0.000 |
| baseline-repeat | 0.170 | 0 | 0.000 |
| power-1-deployment-2 | 0.278 | 0 | 0.000 |
| power-1.25-deployment-1 | 0.148 | 1 | 0.000 |
| power-1.25-deployment-2 | 0.400 | 1 | 0.000 |
| power-1.5-deployment-1 | 0.150 | 2 | 0.000 |
| power-1.5-deployment-2 | 0.236 | 0 | 0.000 |
| power-1.5-deployment-2-steering-0.05 | 0.272 | 1 | 0.000 |

The two new baseline evaluations and the previous 18-parameter zero-vector baseline match exactly in step count, return, termination, final native observation/position, and neural spike count. Thus neutral 27-parameter settings, immutable graph reuse and added telemetry preserve this baseline. Each new trial spent roughly 0.23–0.26 seconds in active setup. The old report did not separate setup from execution, so these data are not a controlled speedup benchmark.

Preview telemetry shows the baseline rear claws retaining substantial adhesion while the body pitches upward and other feet unload. At 114 ms the rear claw controls are about 0.419 and 0.529. These are native adhesion commands, not measured vertical foot forces; causality requires a controlled replay. At 56 ms, almost equal left/right requested power (0.492/0.502) produces different effective power (0.098/0.019) while the two wings cross the deployment gate. This identifies an onset amplification to investigate, not proof that it causes the later tumble.

Both iii1 steering muscles remain silent in the saved baseline samples. Their two coefficients therefore have no demonstrated influence in this episode, even though controlled native probes establish that all 27 parameters can affect the mechanics when their channels are active. Parameter movement by itself is not evidence of meaningful learning.

`analysis.json` retains exact baseline comparisons, per-trial outcomes, timing, and sampled channel peaks. Preview frames are sparse observations; their clipping/activation counts are not full native-step statistics. No candidate is promoted based on a less negative crash score or a later failure.
