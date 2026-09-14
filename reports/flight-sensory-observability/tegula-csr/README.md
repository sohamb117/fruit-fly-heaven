The actual prepared BANC v888 graph contains direct chemical routes from all 26 omitted tegula campaniform cells to 17 of the 24 steering motor neurons. In particular, 22 tegula cells connect directly to left b1 and 24 to right b1. This is a read-only audit of the packed incoming CSR, independently joined to raw annotations; it does not depend on transferring a published FANC/MANC connection claim to BANC.

| Steering target | Left: tegula cells / summed weight | Right: tegula cells / summed weight |
|---|---:|---:|
| b1 | 22 / 10.8 | 24 / 21.0 |
| b2 | 14 / 9.5 | 12 / 16.8 |
| b3 | 2 / 0.6 | 10 / 3.4 |
| i1 | 0 / 0 | 0 / 0 |
| i2 | 13 / 17.2 | 12 / 22.8 |
| iii1 | 0 / 0 | 0 / 0 |
| iii3 | 1 / 0.1 | 2 / 0.2 |
| iii4 | 5 / 2.0 | 7 / 3.8 |
| iv1 | 0 / 0 | 2 / 0.2 |
| iv2 | 0 / 0 | 0 / 0 |
| iv3 | 5 / 1.1 | 8 / 1.3 |
| iv4 | 1 / 0.1 | 1 / 0.2 |

Weights are **nS·ms integrated synaptic conductance**, not sustained conductance, measured current or synaptic efficacy. The table rounds actual Float32 CSR values; [result.json](result.json) retains all 141 directed edges, original indices/root IDs, exact weights and delays. Distinct presynaptic cells equal edge counts for these selected pairs. There are 1,820 outgoing chemical edges from the 26 cells overall, so activating this population would also affect other neurons.

All 26 raw records predict acetylcholine with confidence scores 0.7058–0.9294; none has a verified transmitter annotation. The existing preparation rule selects verified transmitter when present and otherwise uses the prediction. All 141 direct steering edges actually encode `nAChR`, with 0 mV reversal, 0.3 ms rise, 3 ms decay and 2 ms delay. These are the current modeled excitatory receptor assignments, not receptor-expression or electrophysiology measurements.

The b1 routes are bilateral, not exclusively ipsilateral:

| Presynaptic group | To left b1: cells / weight | To right b1: cells / weight |
|---|---:|---:|
| SNpp28 left | 8 / 4.1 | 8 / 3.5 |
| SNpp28 right | 8 / 3.2 | 8 / 9.9 |
| SNpp37 left | 1 / 0.5 | 4 / 0.9 |
| SNpp37 right | 2 / 0.9 | 2 / 1.9 |
| SNpp38 left | 2 / 1.1 | 0 / 0 |
| SNpp38 right | 1 / 1.0 | 2 / 4.8 |

Left b1 is index 75865/root 720575941521196211; right b1 is index 99458/root 720575941549822781. Their total incoming chemical weights are 323.8 and 494.8 nS·ms across 322 and 336 edges, respectively. Tegula weights are about 3.34% and 4.24% of those totals. These fractions mix receptor signs and say nothing by themselves about instantaneous voltage or whether the added input will recruit a spike.

The 26 cells are already present in `io.sensory` as `kind: load`, `body_part: wing_tegula`. They are absent from every console sensory channel, native body transducer and explicit exclusion. Their central graph remains intact; the omission concerns externally encoded sensory current, not deletion or guaranteed electrical silence. Every raw row matches its prepared root ID, cell type, side, mechanical-strain function and campaniform classification. There are 16 SNpp28, six SNpp37 and four SNpp38 cells.

This establishes a concrete missing input route to the b1 pair that remained silent in the recorded flight trials. It does not establish sufficient drive, preferred direction/phase, a strain-to-rate calibration, or stable flight after restoration. A wing-load proxy would remain an explicit sensory-model assumption. It should preserve source identity and laterality and be tested independently of motor or neuronal-profile fitting.

Reproduce without any neural, muscle or body simulation:

```sh
.venv/bin/python reports/flight-sensory-observability/tegula-csr/audit.py
```

The script checks prepared manifest hashes, physiology hash, raw-to-prepared IDs and the complete supplied 26-cell list. Its output pins raw annotations, IDs, offsets, edges, IO, sensory manifest, physiology and the preparation/CSR code.
