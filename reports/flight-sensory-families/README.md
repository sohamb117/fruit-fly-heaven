# Frozen sensory-family recruitment diagnostic

The seven declared conditions hold intrinsic neural parameters, recurrent connectivity, gap junctions and internal state fixed. They change only the added external-current vector: all inputs, no external current, all body inputs, odor, taste, native body transducers, or broad body fallback. Each condition starts from fresh mutable BANC state and runs300 ms on native Dawn with constant input. There is no body or muscle integration.

The exact frozen200 ms context and calibrated input-current values come from the prior afferent identification. Masks cover6837 unique sensory IDs:5059 body,146 odor,539 taste and1093 visual cells. Only1776 receive added current in this context:1630 body and146 odor. Taste and vision are zero. The body mask partitions exactly into4430 native-transducer cells and629 fallback cells;1001 and629 are active respectively.

| Active input family | Cells | Requested rate range | Added current range |
|---|---:|---:|---:|
| Odor |146|39.699–40.442 Hz|19.960–20.151 pA|
| Broad body fallback |629|62.977–72.854 Hz|31.953–36.255 pA|
| Native body transducers |1001 active of4430|0–47.846 Hz|0–25.726 pA|

The fallback population contains579 antennal chordotonal cells and50 proximal front-leg hair-plate cells. Raw antennal detailed annotations are position515, direction47, vibro_position13 and missing4. All579 currently receive62.977 Hz from the fallback in this snapshot: clipped body speed contributes60 Hz and tilt contributes2.977 Hz, despite zero native antennal angle/speed feedback.

The50 hair plates have detailed function `joint_angle`: SNpp45=26, SNpp52=15 and CoHP8=9. Their body-part labels are `front_leg,trochanter` or `coxa,front_leg`. The current exact-string transducer construction misses those labels and they inherit broad self-motion drive containing body speed and yaw. These annotations do not establish a native joint axis or numerical receptive field. No annotation, model or runtime was changed for this diagnostic.

`fallback-annotations.json` preserves the exact prepared-index/raw-BANC annotation join and source hashes. `run/plan.json` preserves all masks, current-vector hashes, input ranges, internal state and declared conditions before execution. `run/source.used.mjs` matches the recorded script hash.

The all-input condition must reproduce every selected48×8 neural state sample and the full forward-state hashes of the previous300 ms conditioning prefix. Taste-only and zero-external inputs are identical here; their recorded traces and full final forward-state bytes must match exactly. All fresh initial states must also match. The observer records48 selected motor neurons every2 ms, including exact event timestamps, cumulative counts and the existing50 ms filtered rate.

Run manually:

```sh
node scripts/identify-flight-sensory-families.mjs http://127.0.0.1:7842/ reports/flight-sensory-families/run
```

`--prepare-only` verifies and writes the declaration without executing a brain. A `STOP` file in the output directory or SIGINT stops the experiment and preserves completed/partial arm artifacts. Requested rates are current-map targets, not guaranteed firing rates inside the connected graph. A DLM recruitment difference does not by itself validate sensory physiology or demonstrate flight.
