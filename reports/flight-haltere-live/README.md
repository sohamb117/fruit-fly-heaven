# Haltere input integration: native A/B/C replay

The batched neural-input implementation preserves the original fly exactly when given constant inputs. The mechanical-current prior changes the trajectory but **does not improve flight** in the first full-fly check. Training remains paused.

All three arms use seed 1290888, the same 27 retained parameters, unchanged BANC graph, native body, and objective thresholds. Each runs alone until the full 8-second landing horizon or native physical failure. All three terminate with excessive rotation; none succeeds.

| Arm | Best qualified flight (s) | Survival (s) | Execution time (s) |
| --- | ---: | ---: | ---: |
| legacy | 0.504 | 0.734 | 12.37 |
| sequence | 0.504 | 0.734 | 13.62 |
| mechanical | 0.504 | 0.724 | 13.50 |

Legacy matches the pre-change v8 physics digest and every score field except wall-clock timers. Sequence-only matches legacy's entire physics digest and all 48-neuron wing event packets exactly. Batching took 10.1% longer in this single run; this is a compatibility check, not a throughput benchmark.

The mechanical arm uses signed virtual haltere loads converted directly to per-neuron current every 0.5 ms, with held body feedback every 2 ms. Its wing event packets first differ at 58 ms and recorded physics at 66 ms. Best qualified flight remains 0.504 s; survival decreases by 10 ms. The left haltere power and external current stay exactly zero; right power reaches 0.706 and recorded right current reaches 253.7 pA. This requires tracing the hDVM motor pathway before treating the sensory model as a bilateral flight feedback loop.

Orientations, sensitivity, virtual oscillation and phase are declared priors. Preserving direction in isolated cells does not establish correct anatomical tuning or corrective whole-network feedback. No training jobs were leased or uploaded, no parameters/checkpoints changed, and no production service was modified. Safari control still cannot obtain a window; these are native numerical observations.

Recompute saved-record comparisons with `node reports/flight-haltere-live/analyze.mjs`. `result.json` pins records and includes contact/qualification measurements. The three immutable bundles and plans preserve the model/parameter contracts.
