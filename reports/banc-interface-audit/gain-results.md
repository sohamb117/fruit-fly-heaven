# Neural input and gain sensitivity

All 21 diagnostic conditions completed in 74.7 wall seconds after model load. Four gain 1 controls reproduced the earlier 200 ms assay exactly: zero difference in per-motor filtered rates, cumulative motor spikes, and total network spikes. Executed source hashes were unchanged; the browser was closed. No production parameter, muscle, native body or output-suppression change was made.

Each condition uses the same initial state and held native-posture feedback, hunger 0.65, AKH 0.65 and insulin 0. It has 200 ms input followed by 100 ms with all added input off; samples are 20 ms apart. Only chemical weights change in diagnostic copies; electrical edges are bitexact. Gains 0.5/1/2 correspond to 0.05/0.1/0.2 nS·ms per annotated contact.

The taste column is **actual cumulative spikes per directly stimulated cell divided by 0.2 s**, not requested Hz or a mean over unstimulated sugar neurons. DLM/m9/m4b columns are the existing 50 ms filtered rates at 200 ms. Total conductance/current and voltage are retained for named motor cells; these are not per-receptor conductances.

| Gain | Input | Actual taste Hz | DLM L/R filtered Hz | m9 filtered Hz | m4b filtered Hz |
|---:|---|---:|---:|---:|---:|
| 0.5 | body_taste150 | 150.9 | 0.0 / 0.0 | 0.0 | 0.0 |
| 0.5 | taste150 | 151.0 | 0.0 / 0.0 | 0.0 | 0.0 |
| 0.5 | labellar150 | 151.9 | 0.0 / 0.0 | 9.4 | 0.0 |
| 0.5 | zero | — | 0.0 / 0.0 | 0.0 | 0.0 |
| 0.5 | body_only | — | 0.0 / 0.0 | 0.0 | 0.0 |
| 0.5 | body_taste25 | 24.7 | 0.0 / 0.0 | 0.0 | 0.0 |
| 0.5 | body_taste75 | 75.6 | 0.0 / 0.0 | 0.0 | 0.0 |
| 1 | body_taste150 | 144.7 | 98.6 / 101.4 | 0.0 | 0.0 |
| 1 | taste150 | 144.6 | 88.8 / 85.6 | 0.0 | 0.0 |
| 1 | labellar150 | 142.5 | 89.6 / 89.3 | 126.1 | 18.3 |
| 1 | zero | — | 0.0 / 0.0 | 0.0 | 0.0 |
| 1 | body_only | — | 145.0 / 150.5 | 0.0 | 0.0 |
| 1 | body_taste25 | 20.4 | 124.1 / 133.9 | 0.0 | 0.0 |
| 1 | body_taste75 | 70.3 | 81.7 / 82.7 | 0.0 | 0.0 |
| 2 | body_taste150 | 133.6 | 231.8 / 237.4 | 1.2 | 0.0 |
| 2 | taste150 | 134.0 | 175.2 / 194.9 | 1.2 | 0.0 |
| 2 | labellar150 | 123.6 | 194.7 / 215.0 | 4.5 | 2.1 |
| 2 | zero | — | 191.1 / 202.7 | 1.8 | 0.5 |
| 2 | body_only | — | 239.6 / 258.8 | 0.5 | 0.5 |
| 2 | body_taste25 | 14.9 | 216.3 / 240.2 | 1.5 | 0.5 |
| 2 | body_taste75 | 57.3 | 246.4 / 264.2 | 1.4 | 0.0 |

At current gain 1, all three taste strengths with body feedback leave m9/m4a/m4b at zero **raw spikes throughout the pulse and off window**. DLM changes nonmonotonically as taste strength changes. Halving gain removes DLM responses while directly stimulated tarsal cells still fire near 151 Hz at the 150 Hz request; it does not recruit the absent reach commands. The labellar control weakens from 49 to 5 total m9 pulse spikes and from 7 to 0 m4b spikes. Doubling gain produces spontaneous DLM activity even with zero added input, so it fails the existing quiet-rest criterion. Higher gain also changes afferent firing; its comparisons are not matched spike-clamp experiments.

Raw off-window evidence at gain 1 (spikes across the five DLM cells on each side; 100 ms window):

| Prior input | Taste off spikes | DLM off spikes L/R | DLM final 20 ms spikes L/R |
|---|---:|---:|---:|
| body_taste150 | 0 | 43 / 45 | 0 / 0 |
| taste150 | 0 | 38 / 40 | 0 / 0 |
| labellar150 | 0 | 65 / 64 | 0 / 2 |
| body_only | 0 | 62 / 61 | 30 / 30 |

Off-window DLM spikes are actual new spikes, not just the rate filter decaying. Most tested taste conditions stop DLM spiking by the final 20 ms, whereas body-only continues then. This 100 ms observation does not establish a permanent attractor or long-run stability.

The data separate a reliably delivered peripheral stimulus from a strongly gain-sensitive downstream response, but do not uniquely identify biological sensory tuning, recurrent weights or cell physiology. Missing reach-MN spikes cannot be fixed by downstream force scaling. No tested gain is promoted as a biological fit or behavioral solution.

[Raw per-driven-afferent and all 805 motor counts](gain-assay.json) · [Compact pulse/off counts and semantics](gain-summary.json) · [Interface and modulation audit](README.md)

Reproduce: `node scripts/probe-banc-interface-gain.mjs` on the local server, with no competing browser/GPU jobs.
