# Left haltere power: anatomy and routing audit

**The left hDVM motor and muscle route are present, connected, and assigned to the correct side.** No missing or swapped left power group was found. Both hDVMs have identical prepared cell physiology and traverse the same muscle kernel. The evidence supports a silent **model motor signal**, not a demonstrated anatomical defect or biologically justified silence.

**Later direct evidence:** the [selected neural-observation analysis](../../flight-haltere-neural-observation/README.md) now verifies zero left hDVM spikes in both legacy and mechanical replays, with exact physics/wing-packet equality to their unobserved references. The original routing audit below remains unchanged; its inference-only limitation is resolved by that separate trace, while biological validity remains unestablished.

| Side | Prepared neuron index | Exact BANC root ID | Motor-list position | Muscle slot | Output group |
|---|---:|---|---:|---:|---|
| Left | 97021 | 720575941546998972 | 432 | 109 | `haltere_power_left` |
| Right | 118683 | 720575941572052381 | 547 | 124 | `haltere_power_right` |

Raw metadata identifies both as `hDVM MN`, class `haltere_motor_neuron`, subclass `haltere_power_neuron`, effector `haltere`, and explicit target `haltere_dorsoventral_muscle`. Predicted transmitter is GABA but verified transmitter is glutamate for both; preparation prioritizes the verified value. Neither cell is unassigned or covered by the wing-event mask. The known muscle-target rule maps exactly these two cells. Four additional broadly labeled `haltere_power_neuron` cells lack peripheral targets and remain unassigned (listed in `result.json`); assigning them to hDVM power would require new anatomical evidence.

## What the saved run establishes

The requested legacy record contains 368 haltere observations from 0 through 0.734 s. Left power is exactly zero throughout. Right power first becomes positive at 0.042 s, peaks at **0.4832590520** at 0.420 s, and ends at 0.2769533396. Its recorded motor-event packets cover the 48 wing neurons, excluding both hDVMs, so this run does not directly save either hDVM's voltage or spike count.

The complete routing/kernel sources and compiled muscle WASM match the saved run's hashes. Each hDVM group contains one neuron. Its sampled rate becomes `clamp(rateHz/80,0,1)`, then a 1 ms muscle update with 15 ms rising/40 ms falling activation constants. Haltere group names have no native joint, so normalized length=1 and shortening velocity=0; Fmax=1. Fatigue is bounded below full exhaustion. Initial energy is 0.35 and cannot fall below 0.3486788 over this run even at maximum modeled expenditure, above the 0.1 fuel reserve. Thus force is `activation*(1-fatigue)` with a strictly positive multiplier. The group's single force is copied to the corresponding side's power without a steering, deployment, or flight gate.

Consequently, exact zero left force strongly indicates zero delivered activation and no positive sampled left hDVM rate under this pinned path. It is an inference from the recorded force and code, not a directly recorded membrane trace. A separate historical WASM capture (`reports/flight-motor-capture-wasm/result.json`) corroborates the pattern: all 68 captured left hDVM rates are 0 Hz; right reaches 49.646923 Hz and first becomes positive at neural time 42 ms. That older run lasted 0.136 s and used a different configuration/fingerprint, so it cannot replace the missing current-run hDVM trace.

## Incoming connections

Values below are prepared incoming pair-edge counts and summed release weights in **nS·ms**. The 0.1 nS·ms/contact scale and receptor assignments are model priors, not measured conductances. Both cells have zero modeled gap edges.

| Prepared receptor | Left edges / weight | Right edges / weight |
|---|---:|---:|
| nAChR, modeled fast excitation | 42 / 8.4 | 110 / 42.4 |
| GABA_A, modeled fast inhibition | 16 / 3.2 | 40 / 17.8 |
| GluCl_assumed, modeled fast inhibition | 2 / 0.3 | 14 / 2.5 |
| Serotonin, modulatory | 0 / 0 | 1 / 0.1 |
| **Total** | **60 / 11.9** | **165 / 62.8** |

| Annotated source superclass | Left edges / weight | Right edges / weight |
|---|---:|---:|
| VNC intrinsic | 44 / 8.6 | 124 / 49.6 |
| Ascending | 5 / 1.4 | 12 / 4.5 |
| Descending | 4 / 0.5 | 12 / 3.0 |
| Sensory | 4 / 0.9 | 5 / 2.2 |
| Motor | 1 / 0.1 | 4 / 0.5 |
| Ascending visceral/circulatory | 0 / 0 | 2 / 0.4 |
| Sensory ascending | 0 / 0 | 2 / 0.2 |
| Unannotated superclass | 2 / 0.4 | 4 / 2.4 |

Exact source cell-class/type aggregates and strongest source identities are in `result.json`; their raw metadata excerpts are in `metadata.json`. The largest shared excitatory source type is IN07B067: four source cells contribute 1.6 to left versus 8.8 to right. Cell-class labels are absent for 43 left-input sources and 113 right-input sources, even when their superclass/type is annotated; those absences are retained explicitly.

Neither degree, total weight, nor a simple excitation/inhibition ratio explains membrane dynamics by itself. Presynaptic activity, timing, receptor kinetics, driving force, and the generic neuron parameters still determine recruitment. These structural differences do not establish physiological validity or pathology.

## Calibration consequence

At zero left power, the declared virtual observer has zero left velocity/load and sends **zero pA to all 171 left haltere afferents**. Changing the 800 pA cap or the left orientation prior cannot change zero into an active signal. Previous mechanical sensitivity results with both organs powered do not establish behavior at this unilateral operating point. The correct diagnostic is to record both hDVM membrane/rate/spike states and muscle slots 109/124 alongside the existing power trace. No opposite-side or wing-power fallback is warranted by these data.

## Reproduce

```sh
.venv/bin/python reports/flight-haltere-live/haltere-power-audit/extract-metadata.py
node reports/flight-haltere-live/haltere-power-audit/analyze.mjs
```

These commands only read annotations, prepared graph ranges, source files and saved captures, then write the audit artifacts. No native or neural simulation, source edit, wiring change, or power fallback was performed.
