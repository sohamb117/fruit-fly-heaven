# Recorded afferent-response robustness

All four declared runs are complete.

Endpoint counts alone do not establish a stable early response across the two conditioned states. Inspecting intermediate2ms samples nevertheless finds sparse timing-sensitive responses; the temporal section below preserves these positive findings rather than treating zero net count as absent information. This experiment does not establish a robust multi-axis receptor projection.

Only saved neural/muscle traces were analyzed. No body integration or new neural simulation occurred.

Each run passed its duplicate-baseline gate. All available same-prefix amplitude comparisons also matched the complete conditioned snapshot, prefix muscle state, prefix trace, and baseline trace exactly.

Odd-symmetry cosine compares the positive response with the negated negative response: +1 is locally odd, negative values indicate that opposite pulse signs tend to move output in the same direction. A dash means a zero response prevents defining the cosine.

| Prefix / pulse | Population | Raw pooled spikes10ms | Spikes20ms | Spikes40ms | Filtered rate20ms | Muscle force20ms |
|---|---|---:|---:|---:|---:|---:|
| 100ms / ±5Hz (historical) | haltere/right | — | — | -0.866 | — | — |
| 100ms / ±5Hz (historical) | wing_base/left | — | -0.408 | -0.673 | -0.420 | -0.056 |
| 100ms / ±5Hz (historical) | wing_base/right | 0.707 | -0.204 | -0.285 | -0.222 | -0.450 |
| 100ms / ±1.25Hz | haltere/right | — | — | 0.000 | — | — |
| 100ms / ±1.25Hz | wing_base/left | — | — | -0.395 | 0.000 | 0.000 |
| 100ms / ±1.25Hz | wing_base/right | — | 0.000 | 0.030 | -0.020 | -0.027 |
| 100ms / ±2.5Hz | haltere/right | — | — | -0.693 | — | — |
| 100ms / ±2.5Hz | wing_base/left | — | -0.577 | -0.451 | -0.591 | -0.112 |
| 100ms / ±2.5Hz | wing_base/right | — | 0.000 | 0.000 | -0.014 | -0.069 |
| 300ms / ±1.25Hz | haltere/right | — | — | — | — | — |
| 300ms / ±1.25Hz | wing_base/left | — | — | -0.064 | 0.000 | 0.000 |
| 300ms / ±1.25Hz | wing_base/right | — | 1.000 | -0.192 | 0.995 | 1.000 |
| 300ms / ±2.5Hz | haltere/right | — | — | — | — | — |
| 300ms / ±2.5Hz | wing_base/left | — | — | -0.700 | -0.006 | -0.832 |
| 300ms / ±2.5Hz | wing_base/right | — | 1.000 | -0.836 | 0.992 | 0.918 |

Across-condition alignment below uses central difference per requested Hz.1 means the same output direction; a negative value is a reversal.

| First → second | Population | Spikes20ms | Spikes40ms | Rate20ms | Force20ms | Force20ms gain norm ratio |
|---|---|---:|---:|---:|---:|---:|
| prefix100-pulse1p25 → prefix100-pulse2p5 | haltere/right | — | 0.038 | — | — | — |
| prefix100-pulse1p25 → prefix100-pulse2p5 | wing_base/left | 0.354 | 0.010 | 0.363 | 0.588 | 0.806 |
| prefix100-pulse1p25 → prefix100-pulse2p5 | wing_base/right | 0.867 | 0.049 | 0.865 | 0.982 | 0.500 |
| prefix100-pulse2p5 → historical-prefix100-pulse5 | haltere/right | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 |
| prefix100-pulse2p5 → historical-prefix100-pulse5 | wing_base/left | 0.894 | -0.023 | 0.891 | 0.993 | 0.504 |
| prefix100-pulse2p5 → historical-prefix100-pulse5 | wing_base/right | 0.004 | 0.257 | 0.004 | 0.337 | 0.910 |
| prefix100-pulse1p25 → historical-prefix100-pulse5 | haltere/right | — | 0.000 | — | — | — |
| prefix100-pulse1p25 → historical-prefix100-pulse5 | wing_base/left | 0.316 | -0.088 | 0.323 | 0.584 | 0.406 |
| prefix100-pulse1p25 → historical-prefix100-pulse5 | wing_base/right | 0.207 | 0.090 | 0.213 | 0.361 | 0.455 |
| prefix300-pulse1p25 → prefix300-pulse2p5 | haltere/right | — | 0.775 | 0.009 | 0.223 | 0.293 |
| prefix300-pulse1p25 → prefix300-pulse2p5 | wing_base/left | — | 0.232 | -0.027 | -0.000 | 0.318 |
| prefix300-pulse1p25 → prefix300-pulse2p5 | wing_base/right | 1.000 | -0.069 | 0.999 | 0.957 | 0.522 |
| prefix100-pulse1p25 → prefix300-pulse1p25 | haltere/right | — | 0.000 | — | — | — |
| prefix100-pulse1p25 → prefix300-pulse1p25 | wing_base/left | — | 0.269 | 0.000 | 0.000 | 0.266 |
| prefix100-pulse1p25 → prefix300-pulse1p25 | wing_base/right | -0.082 | -0.088 | -0.079 | -0.261 | 0.769 |
| prefix100-pulse2p5 → prefix300-pulse2p5 | haltere/right | — | 0.000 | 0.000 | 0.000 | 0.904 |
| prefix100-pulse2p5 → prefix300-pulse2p5 | wing_base/left | 0.000 | -0.199 | 0.002 | 0.030 | 0.105 |
| prefix100-pulse2p5 → prefix300-pulse2p5 | wing_base/right | -0.071 | 0.241 | -0.088 | 0.065 | 0.803 |

The direct per-neuron state check below requires a nonzero spike-count contrast in the same MN under both100ms and300ms prefixes. "Bidirectional" additionally requires opposite effects of + and − stimulation relative to baseline in both states, with matching polarity.

| Matched amplitude | Population | Jointly changed MNs20ms | Same / opposite sign40ms | Consistently bidirectional MNs40ms |
|---|---|---:|---:|---:|
| 1.25Hz | haltere/right | 0 | 0 / 0 | 0 |
| 1.25Hz | wing_base/left | 0 | 2 / 0 | 0 |
| 1.25Hz | wing_base/right | 0 | 5 / 2 | 0 |
| 2.5Hz | haltere/right | 0 | 0 / 0 | 0 |
| 2.5Hz | wing_base/left | 0 | 2 / 2 | 0 |
| 2.5Hz | wing_base/right | 0 | 2 / 1 | 0 |

Net spike counts can hide within-window timing shifts. The following check inspects every saved2ms cumulative-count sample, including contrasts that have vanished by the window endpoint. It still cannot resolve sub2ms timing or establish wingbeat-phase coding.

| Prefix / pulse | Population | First count contrast | First raw-MN rate contrast | Transient-only count MNs20ms |
|---|---|---:|---:|---:|
| 100ms / ±5Hz | haltere/right | 16ms | 14ms | 3 |
| 100ms / ±5Hz | wing_base/left | 14ms | 12ms | 2 |
| 100ms / ±5Hz | wing_base/right | 10ms | 8ms | 10 |
| 100ms / ±1.25Hz | haltere/right | 24ms | 24ms | 0 |
| 100ms / ±1.25Hz | wing_base/left | 18ms | 16ms | 1 |
| 100ms / ±1.25Hz | wing_base/right | 14ms | 14ms | 3 |
| 100ms / ±2.5Hz | haltere/right | 16ms | 14ms | 3 |
| 100ms / ±2.5Hz | wing_base/left | 14ms | 12ms | 1 |
| 100ms / ±2.5Hz | wing_base/right | 12ms | 12ms | 6 |
| 300ms / ±1.25Hz | haltere/right | 14ms | 14ms | 2 |
| 300ms / ±1.25Hz | wing_base/left | 16ms | 16ms | 1 |
| 300ms / ±1.25Hz | wing_base/right | 8ms | 8ms | 0 |
| 300ms / ±2.5Hz | haltere/right | 8ms | 8ms | 3 |
| 300ms / ±2.5Hz | wing_base/left | 8ms | 8ms | 5 |
| 300ms / ±2.5Hz | wing_base/right | 8ms | 8ms | 2 |

Across prefixes, shared MNs with a cumulative-count contrast at *any* sampled time in the first20ms: 1.25Hz: haltere/right=0, wing_base/left=0, wing_base/right=0; 2.5Hz: haltere/right=0, wing_base/left=0, wing_base/right=3.

wing_base/right at±2.5Hz has matching transient contrast signs in both states: MN37077 (wing_steer_right:b3_muscle), first prefix at20ms and second at14ms; MN74497 (wing_power_left:dorsal_longitudinal_muscle), first prefix at16ms and second at18ms; MN139519 (wing_power_right:dorsoventral_muscle), first prefix at12ms and second at8/20ms. These are timing-sensitive contrasts, not evidence of a validated wingbeat-phase code or anatomical axis tuning.

Spike counts integrate10/20/40ms after pulse onset. The40ms window includes20ms of recovery. Filtered rate and force comparisons use the window endpoint. Normalized gain ratios include unequal realized spike recruitment; requested Hz is not guaranteed firing rate inside the graph.

These are artificial frozen-context identifications. The100/300ms prefixes change neural and muscle state together. They do not represent restored live-flight states or validate body torque, closed-loop flight, or anatomical rotation tuning. Full vectors, one-sided alignments, source hashes, and exact comparison gates are in aggregate.json.

Reproduce without a simulation:

```sh
node scripts/aggregate-flight-afferent-robustness.mjs reports/flight-afferent-identification/robustness/plan.json
```
