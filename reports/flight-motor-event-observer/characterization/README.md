# Event-to-native-muscle results

All ten fixed conditions completed with source and event-sequence checks intact. Synthetic kernel, excitation and activation passed the final 2 s periodicity test; no brain, body, controller or optimizer was simulated.

| Family | MN rate (Hz) | Mean excitation | Native activation | Native fatigue | Native force |
|---|---:|---:|---:|---:|---:|
| steering | 50 | 0.210 | 0.280 | 0.004 | 0.279 |
| steering | 100 | 0.392 | 0.440 | 0.091 | 0.400 |
| steering | 200 | 0.641 | 0.661 | 0.212 | 0.521 |
| dlm | 3 | 0.174 | 0.204 | 0.001 | 0.204 |
| dlm | 8 | 0.418 | 0.458 | 0.098 | 0.413 |
| dlm | 12 | 0.561 | 0.593 | 0.170 | 0.492 |
| dvm | 3 | 0.174 | 0.204 | 0.001 | 0.204 |
| dvm | 8 | 0.418 | 0.458 | 0.098 | 0.413 |
| dvm | 12 | 0.561 | 0.593 | 0.170 | 0.492 |

Means use (4000, 6000] ms, after 4 s from zero state, with length 1, shortening velocity 0, Fmax 1 and energy 1. The tested excitation, activation and force means increase with rate. Force is normalized model output. Native fatigue remains active and can drift; periodic excitation and activation do not imply stationary force. The native 15 ms rise / 40 ms fall gate also makes mean activation differ from mean excitation.

DVM matches DLM because this experiment explicitly borrows the same kernel constants and excites every unit in a mapping with the same synthetic phase. That equality is an assumption check, not independent physiology evidence.

The original 1,460 events and all 48 identities were replayed without extension. During (100, 280] ms, all ten DLM units stayed above 0.95 mean excitation in every interval; their mean excitations span 0.9930–0.9997. The left/right DLM groups produce mean excitation 0.9970/0.9978 and normalized native force 0.9844/0.9853. Thus this event prior remains nearly saturated under the recorded high DLM rates; it does not turn that neural output into a calibrated low-rate motor regime.

| Steering type | Legacy command L/R | Event mean excitation L/R | Event native force L/R |
|---|---:|---:|---:|
| b2 | 1.000/1.000 | 0.552/0.520 | 0.617/0.593 |
| i1 | 0.995/1.000 | 0.390/0.683 | 0.480/0.704 |
| i2 | 1.000/1.000 | 0.678/0.706 | 0.704/0.724 |
| iv1 | 1.000/1.000 | 0.594/0.654 | 0.642/0.690 |

The legacy values above are calculated directly from the saved 50 ms rate field using `clamp(rate/80,0,1)`, at its original 2 ms sample times. Both b2, both i2 and both iv1 commands equal 1 at every sample in this window. Their event-driven native outputs still have bilateral differences and temporal variability. This establishes retained contrast in the new assay and lost contrast at the old command clamp. It is **not a matched comparison of native force**: the old mechanical and energy history and update schedule differ, and no legacy native replay was run here. No flight or stabilization benefit is claimed.

The largest 4/8-point quadrature discrepancy was 2.630e-9 (limit 1e-7). Per-unit and per-mapping statistics, waveform samples, fatigue slopes and exact source/input hashes remain in the case files and `analysis.json`. No mean-force normalization or parameter fitting was applied.

Reproduce the offline analysis:

```sh
node scripts/analyze-wing-event-muscles.mjs reports/flight-motor-event-observer/characterization
```

The pinned characterization script can run in a new output directory:

```sh
node scripts/characterize-wing-event-muscles.mjs reports/new-output --prepare-only
node scripts/characterize-wing-event-muscles.mjs reports/new-output
```

The current event kernels, unit recruitment scale, equal pooling, DVM borrowing and native activation stage remain explicit model priors. The next proposed check is a restrained mechanical phase response with a matched no-event baseline, not a new physiological fit.
