# Saved haltere neural observation

**The mechanical-current path changes neural activity substantially, but leaves the left haltere loop inactive and does not improve this flight.** The left hDVM records zero spikes in both runs. Right hDVM spikes increase from 19 to 26. Both runs retain a best maintained-flight interval of 0.504 s and fail the objective; mechanical terminates at 0.724 s versus legacy at 0.734 s. This is one fixed parameter vector and seed, not training evidence or physiological validation.

## Observer/reference checks

`analyze.mjs` verifies the exact reference files and hashes declared in each plan. Both observed runs match their respective unobserved references in:

- The complete recorded physics-digest object, including all fields and row counts.
- Every wing packet, with all 48 selected wing neurons: 368 legacy packets and 363 mechanical packets.
- Haltere feedback, saved flight history, source hashes, and every returned evaluation field except the three wall-timing measurements.

The legacy physics digest is `c8abcff173e74f7a118005b2349774b587492ac0c4447bb1b303f8af0806b9c9`. The mechanical digest is recorded with its exact reference in `result.json`. These comparisons validate the added readout as nonperturbing for these two replays. They compare recorded digests; the analysis does not rerun physics or reconstruct unrecorded dense state.

## hDVM state

Threshold is **−42 mV** for both hDVMs in the checksum-verified prepared parameters. Voltage, rate, and conductance below are **2 ms snapshots**, excluding the initial placeholder state. Their extrema are not continuous maxima or extrema over every 0.5 ms neural tick. The rate field is the runtime's smoothed rate output.

| Run / side | Index | Complete spike count | Sampled voltage range (mV) | Gap between threshold and highest sampled V (mV) | Sampled rate range (Hz) | Sampled total conductance (nS) |
|---|---:|---:|---:|---:|---:|---:|
| Legacy left | 97021 | 0 | −60 to −48.934654 | 6.934654 | 0 | 2 to 2.927539 |
| Mechanical left | 97021 | 0 | −60 to −48.096008 | 6.096008 | 0 | 2 to 2.927539 |
| Legacy right | 118683 | 19 | −60 to −42.020027 | 0.020027 | 0 to 49.209953 | 2 to 5.915625 |
| Mechanical right | 118683 | 26 | −60 to −42.036877 | 0.036877 | 0 to 65.670921 | 2 to 5.481044 |

The cumulative spike counters include intervening neural ticks: every saved value is checked for monotonicity and exact integer representation, and totals are final minus initial. Thus zero left spikes is direct evidence over the recorded episode, even though its closest approach to threshold between snapshots is unknown. Right spikes occurred despite all displayed voltage snapshots remaining below threshold; the membrane resets at a spike and sparse snapshots need not capture threshold crossing. Do not interpret its sampled voltage gap as absence of firing.

Initial state, final state, all field ranges/times, last-spike values, and per-cell summaries are retained in `result.json`. The initial conductance field is a zero placeholder and is excluded from the reported conductance ranges. The state field `externalAndIntrinsicCurrentPa` also includes intrinsic/adaptation terms; it is not the raw external transducer current.

## Sensory activity

"Active" means a cell emitted at least one spike during the recorded interval. These are cumulative event counts across the 328 checked haltere sensory identities, not instantaneous firing rates.

| Run endpoint | Side / cells | Total sensory spikes | Active cells |
|---|---|---:|---:|
| Legacy, 734 ms | Left / 171 | 515 | 89 |
| Legacy, 734 ms | Right / 157 | 520 | 106 |
| Mechanical, 724 ms | Left / 171 | 0 | 0 |
| Mechanical, 724 ms | Right / 157 | 13,960 | 157 |

At the **common 724 ms endpoint**, legacy has 499 left sensory spikes (89 active cells) and 433 right spikes (103 active cells); mechanical has 0 left and 13,960 right spikes. Right hDVM counts remain 19 versus 26 at that common endpoint. The change therefore is not explained by comparing different episode lengths.

## Current and force corroboration

Left muscle power is exactly zero in both records. The mechanical input diagnostics mark it as **present zero**, not missing, and record zero direct current and zero instantaneously driven left cells in every saved block. This agrees with the zero left sensory counters. Right muscle power peaks at 0.483259 in legacy and 0.705658 in mechanical; both first become positive at 42 ms, matching the first positive right hDVM counter sample.

The mechanical diagnostics retain only the final **1.5 ms input sample within each 2 ms body block**, not all four input vectors. Across those saved samples, the largest right transducer current is 253.723267 pA under the declared 800 pA cap; at most 86 right cells receive positive current at any one saved phase, while all 157 spike over the episode. Every recorded input's power, Omega and phase are checked against the preceding completed body state. Its timestamp must not be confused with the post-physics state in the same feedback row. Legacy has no direct mechanical-current diagnostic; this does not imply its old sensory currents were zero.

The [anatomy and routing audit](../flight-haltere-live/haltere-power-audit/README.md) identifies both correctly routed hDVM groups and their asymmetric incoming connections. This new trace resolves that audit's missing current-run neural evidence: the left hDVM is connected but unspiking in these model runs. It does not establish why it is unspiking or whether that is biologically valid. The unilateral response, orientation assignment and current cap remain unvalidated modeling choices; raising the cap cannot activate sensors whose side's mechanical power is zero.

## Reproduce

```sh
node reports/flight-haltere-neural-observation/analyze.mjs
```

This reads the saved records, pinned observer sources, prepared identities and parameters, then writes `result.json`. It performs no neural stepping, native stepping, training, uploads, or live-source edits.
