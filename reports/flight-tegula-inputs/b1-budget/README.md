# Why adding tegula input does not guarantee b1 recruitment

The direct tegula route is anatomically present, but its estimated mean drive is **below left b1's configured threshold budget and only modestly above right b1's before inhibition**. This makes the observed silence compatible with ordinary input balance; it does not identify a defective neuron model or justify increasing gain.

This is arithmetic on actual prepared CSR/Float32 parameters and the saved [398 ms native-Dawn tegula run](../../flight-tegula-live/feedback-evaluation/000-feedback-tegula-feedback-ed748ce4-1676-4c57-b331-251c3e0cd01f.json). Both b1 cumulative counts remain zero in all 200 motor packets. The capture contains the 48 wing motor neurons, **not actual tegula sensory spikes or b1 receptor conductances**. [result-v2.json](result-v2.json) pins the graph, parameters, raw anatomy/audit, runtime equations, sensory mapper and capture. The initial `result.json` calculation has the same numeric budget but omitted the anatomical-function field; v2 adds its verified raw-annotation provenance.

## Configured b1 cells and all incoming weights

Left b1 is index 75865/root `720575941521196211`; right b1 is index 99458/root `720575941549822781`. Both raw annotations say `tonic_wing_steering`, but both receive the generic **motor** profile: C=40 pF, leak=2 nS, rest=−60 mV, threshold=−42 mV, reset=−58 mV, refractory=2 ms, adaptation increment≈0.4 pA and decay=150 ms. Tonic current is **0 pA**, hunger gain is 0, and these neurons are spiking, not graded. Their leak time constant is 20 ms. The DLM ionic override does not include b1.

| Incoming receptor/category | Left: edges / weight | Right: edges / weight |
| --- | ---: | ---: |
| nAChR, excitatory, reversal 0 mV | 150 / 127.9 | 162 / 220.0 |
| GABA_A, inhibitory, reversal −75 mV | 164 / 193.1 | 166 / 271.9 |
| GluCl_assumed, inhibitory, reversal −75 mV | 8 / 2.8 | 7 / 2.8 |
| All excitatory chemical | 150 / 127.9 | 162 / 220.0 |
| All inhibitory chemical | 172 / 195.9 | 173 / 274.7 |
| Octopamine modulation | 0 / 0 | 1 / 0.1 |
| All chemical | 322 / 323.8 | 336 / 494.8 |

Weights are nS·ms, not sustained conductances. All listed edges have positive weight; their prepared presynaptic profiles are spiking. All other chemical receptor categories and incoming gap edges have zero entries. These totals do not reveal the actual excitatory/inhibitory activity balance.

The octopamine coefficient is ≈0.3, but the current runtime applies it to external/tonic input, not chemical conductance. The anatomical word “tonic” does not itself create ongoing current or autonomous firing in this model. No such new current is inferred or added here.

## Direct tegula budget

Every direct tegula edge uses nAChR with 0.3 ms rise, 3 ms decay and 2 ms graph delay. The runtime emits `spike / dt`, then applies two unit-DC-gain receptor filters. Therefore the long-run mean relation for actual presynaptic spike rates is:

`mean gE [nS] = sum(weight [nS·ms] × actual rate [Hz]) / 1000`.

There is no extra decay-time or timestep multiplier. At b1's threshold, leak requires `2 × (−42 + 60) = 36 pA`; nAChR supplies `42 × gE pA`. With inhibitory conductance `gI` at −75 mV, the steady balance is `42 gE = 36 + 33 gI`, before other drive/adaptation. With no inhibition the threshold equilibrium requires **0.857143 nS**; equality is not a finite-time spike guarantee.

Direct weights are asymmetric:

| Target | From left tegula: cells / weight | From right tegula: cells / weight | Total |
| --- | ---: | ---: | ---: |
| Left b1 | 11 / 5.7 | 11 / 5.1 | 22 / 10.8 |
| Right b1 | 12 / 4.4 | 12 / 16.6 | 24 / 21.0 |

Using **requested rates as an explicit stand-in for unknown actual sensory firing** gives:

| Saved input window | Requested left/right means, Hz | Estimated left b1 gE / inward current at threshold | Estimated right b1 gE / inward current at threshold |
| --- | --- | --- | --- |
| Full 0–398 ms | 46.746 / 46.640 | 0.5043 nS / 21.18 pA | 0.9799 nS / 41.16 pA |
| Predeclared 100–280 ms | 55.908 / 55.971 | 0.6041 nS / 25.37 pA | 1.1751 nS / 49.35 pA |

For the predeclared window, this leaves left b1 **10.63 pA below** the leak requirement. Right b1 has only **13.35 pA** headroom; approximately **0.405 nS** of inhibitory conductance would consume that headroom if no other excitation/current contributed. This is a conditional balance calculation, not measured inhibition. Equal sustained actual tegula firing would need about 79.37 Hz for left b1 and 40.82 Hz for right b1 to reach the no-inhibition DC threshold.

The arithmetic uses the actual causal held-input intervals and Float32 rounding at the encoder boundary. Early positive requested rates below 2 Hz are raised to the isolated mapper's 2 Hz minimum; including that floor changes full-run estimated gE to 0.5074/0.9859 nS and leaves the 100–280 ms estimate unchanged. Even the clamped mapper targets are not guaranteed full-network sensory firing rates.

These estimates do not reconstruct finite-window receptor conductance. Synaptic boundary transients, synchronized spikes, voltage–conductance covariance, recurrent excitation/inhibition and sensory response to the full graph remain unmeasured. Other tegula outputs can alter the network indirectly. The necessary evidence to resolve the remaining ambiguity is actual tegula spike counts plus b1 voltage and per-receptor conductance on a matched run; this report performs no such run and recommends no parameter change.

Reproduce the arithmetic into a new output filename, without any neural or body simulation:

```sh
node reports/flight-tegula-inputs/b1-budget/analyze.mjs result-new.json
```
