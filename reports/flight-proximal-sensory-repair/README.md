# Proximal hair-plate input correction

Fifty annotated front-leg joint-angle sensors were receiving whole-body speed and yaw through a generic fallback. Their compound anatomical labels (`coxa,front_leg` and `front_leg,trochanter`) did not match the native transducer lookup. Those labels identify proximal anatomy but do not establish a native joint axis or tuning curve.

The console builder now emits explicit exclusions for these 50 cells, retaining their graph and sensory-channel membership, raw annotations, side and parsed anatomical tokens. The encoder honors explicit exclusions with both native and aggregate feedback. This withholds unsupported added host current; recurrent neural activity remains intact.

Regeneration changed only `console/sensory-inputs.json`. Thirteen other checked files, including connectivity, neuron physiology, IDs, IO and the taste supplement, were byte-identical. The added 50 exclusions are the complete semantic difference in the regenerated sensory manifest. Eleven Python and 24 JavaScript tests passed, including the actual 50-ID join, neighboring supported receptor responses and the nonnative fallback case.

## Native regression

One full BANC network and native FlyBody, sequentially evaluated with the same seed190888, initial physical state and27 interpreter coefficients as the saved reference. The development body retains normal contacts with claw adhesion disabled. Neural execution used the pinned Dawn/Metal backend. This was a diagnostic evaluation, without optimizer updates or contribution uploads.

| Observation | Before | Corrected routing |
| --- | ---: | ---: |
| Takeoff registered | No | Yes |
| Longest qualifying flight |102ms |166ms |
| Excessive-rotation failure |388ms |480ms |
| Return |−2.334 |−1.502 |

The only changed executable/data sources between these recorded evaluations are the sensory encoder and its exclusion manifest. This is an improvement in one seed, **not stable flight or evidence of learning**. The remaining scalar rotation feedback, motor recruitment and timing limitations still require repair. Safari was unavailable, so these are native telemetry results rather than visual observation.

The improvement did not result from silencing DLM output. Over the same (100,280]ms startup window, mean raw DLM firing changed from115.6/117.8Hz (left/right) to118.9/117.8Hz. Both b1 neurons remained silent. `motor-window.json` retains every mapped wing muscle's individual count rates; these short-window observations are not a steady-state physiological fit.

`integration.json` records the regeneration checks; `comparison.json` pins the two evaluation records. The bundle, plan and raw event capture remain local alongside this report.
