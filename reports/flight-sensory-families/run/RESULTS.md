# Sensory-family recruitment results

All seven conditions completed. The all-input neural trajectory exactly matched the previous300ms prefix; taste-only and zero-external trajectories and full forward-state bytes matched each other.

Zero added sensory current produced no DLM or DVM spikes. Every nonzero sensory family alone produced substantial DLM firing. The broad fallback remains an anatomical mapping concern, but these results do not isolate it as the unique source of strong wing power drive.

| External-current condition | DLM left raw Hz | DLM right raw Hz | DVM left raw Hz | DVM right raw Hz |
|---|---:|---:|---:|---:|
| all | 112.00 | 117.00 | 29.29 | 30.00 |
| no-external | 0.00 | 0.00 | 0.00 | 0.00 |
| body-only | 107.00 | 111.00 | 30.71 | 33.57 |
| odor-only | 111.00 | 112.00 | 33.57 | 28.57 |
| taste-only | 0.00 | 0.00 | 0.00 | 0.00 |
| native-transducer-only | 123.00 | 122.00 | 35.00 | 40.71 |
| broad-fallback-only | 113.00 | 116.00 | 30.00 | 26.43 |

Raw firing rates count events in(100,300]ms and average across the mapped neurons; they do not use the50ms rate filter. Cumulative counts were cross-checked against the exact timestamped events. Per-neuron counts, filtered rates, first spike times and full2ms group traces are in analysis.json.

The intrinsic current, recurrent graph, gap junctions and hunger/satiety modulation stayed fixed. No body, muscles, flight, optimizer or receptor-axis model was simulated. This is a sufficiency assay with constant frozen input, not a measurement of additive family contributions or a live closed-loop behavior.

The zero-external result argues against autonomous DLM firing under this specific unchanged model/state. Comparable recruitment from odor alone, native body transducers alone and fallback alone motivates examining stimulus-to-neural recruitment and the downstream rate-to-muscle interpretation before choosing a physiological change. It does not establish whether these input priors or motor firing rates are biologically appropriate.

Reproduce this analysis without simulation:

```sh
node scripts/analyze-flight-sensory-families.mjs reports/flight-sensory-families/run
```
