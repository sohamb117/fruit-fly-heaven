# Frozen-context BANC afferent response

The four tested organ/side populations can change wing motor output through the actual BANC graph. This establishes a usable neural pathway; it does not yet establish a stable signed feedback controller.

The native Dawn run completed all ten branches in approximately 12 seconds. Duplicate baselines matched every sampled neural/muscle trace and all 28,765,764 final forward-state bytes. Every branch began from the same checked neural snapshot. No body was integrated, no optimizer ran, and no receptor axis was assigned.

The context is artificial: the sensory encoder used a frozen native pose/feedback snapshot at 200 ms, while the brain began from fresh initialization followed by 100 ms of constant-input conditioning. Initial captured internal values were held fixed. The 28 wing-muscle readouts used native muscle dynamics and explicit Hill80/n1 steering recruitment, with frozen mechanical inputs. This is not restoration of the live flight brain at 200 ms.

| Stimulated population | Cells | Requested pulse per cell | First >1 Hz wing-rate contrast after pulse onset | First >0.0001 muscle-force contrast | Largest force contrast |
|---|---:|---:|---:|---:|---:|
| Haltere left | 171 | ±0.443 Hz | 24 ms | 28 ms | 0.0539 |
| Haltere right | 157 | ±5 Hz | 16 ms | 20 ms | 0.0390 |
| Wing base left | 62 | ±5 Hz | 14 ms | 14 ms | 0.0866 |
| Wing base right | 59 | ±5 Hz | 10 ms | 10 ms | 0.0940 |

Contrasts compare the plus and minus branches. Force is the normalized native muscle output, not measured body torque. Thresholds describe the saved traces, not statistical significance; time resolution is 2 ms. The neuronal rate readout is the production 50 ms filtered rate.

During the 20 ms pulse, stimulated-population spike totals were baseline/plus/minus: haltere left 32/33/32, haltere right 2/14/0, wing base left 57/64/47, and wing base right 56/67/47. Requested rate changes are converted to current using the existing isolated-cell calibration; they are not guarantees of those firing rates inside the connected graph. The large normalized late haltere-left response follows only one extra afferent spike, so it should not be read as a reliable high-gain linear channel.

The three populations with nonzero live angular-sense gates—haltere right and both wing bases—produce three numerically independent steering-force response columns. Their whole-response condition number is 7.26; unit-column singular values are 1.332, 1.007, and 0.461. All four groups give four independent force-response columns. Thus this experiment does not establish a neural dimensionality deficit that would immediately require finer cell-type groups.

The current encoder still drives these populations using one unsigned angular-velocity norm. Its instantaneous rotation-to-current projection therefore has at most one dimension and loses rotation sign. Independent stimulation in this assay tests potential after an explicit receptor-model change; it does not mean the running encoder already provides three signed gyro channels. Haltere-left rotation sensitivity is additionally disabled by its zero asynchronous gate in this recorded context.

The plus/minus response is substantially nonlinear. Over stimulation plus recovery, the cosine between the positive response and the negated negative response is −0.25, −0.91, −0.56, and −0.43 for the four groups. An odd linear response would give +1. Both pulse signs often move away from baseline in similar directions. Late recurrent evolution and unequal spike recruitment materially affect the central differences. These finite-amplitude contrasts are not yet validated local derivatives.

Direct cumulative wing-MN spike counts show that this issue precedes the 50 ms rate filter. In the first 20 ms after stimulation onset, pooled spike-count odd-symmetry cosines are undefined for the one-sided haltere responses, −0.408 for wing base left, and −0.204 for wing base right. By 40 ms, haltere right is −0.866. Wing base right has a more nearly odd response in the first 10 ms (+0.707), involving only two changed MNs in the positive branch and one in the negative branch. The rate filter can introduce delay and distortion, but it is not the sole source of the observed polarity instability. `directSpikeWindows` and `directSpikeTrace` preserve all raw48 counts separately from28 pooled mean-per-neuron counts.

An optional projection through the historical mechanical Jacobian is included only as a screening calculation. That plant used all steering forces 0.35, common power 0.85, unit interpreter gains, no force references, fixed nonwing joints, and a restrained level root with wingbeat-averaged measurements. These conditions differ from both this neural readout and the current body model. Power-muscle changes are omitted. The three-group projected torque condition is 64.6 over the whole response and approximately 986 during the pulse; the two wing-base mean torque columns are nearly opposite. This does not prove current-runtime torque deficiency or controllability.

Before fitting directions, test whether early response polarity survives smaller pulse amplitudes and a second relevant neural state. The present results support continuing with the three accessible anatomical groups first, while retaining finer cell-type grouping as an option if a matching operating-point assay finds inadequate torque directions. No anatomical rotation-axis labels can be inferred from these stimulation results.

Reproduce the analysis, without a neural simulation:

```sh
node scripts/analyze-flight-afferent-response.mjs reports/flight-afferent-identification/dawn-frozen-context
```

`result.json` and the ten branch files are the recorded experiment. `analysis.json` contains complete contrasts, source hashes, response-window summaries, conditioning, and the explicitly transferred mechanical screen. The assay and analysis scripts are separate; analysis does not change the recorded experiment.
