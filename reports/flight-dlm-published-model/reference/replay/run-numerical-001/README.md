# Frozen conductance comparison completed

Under identical captured synaptic conductances, the published ionic DLM membrane reference produced substantially fewer spikes than the generic BANC motor profile, but all ten cells remained active. This was one 300 ms open-loop neural replay, not a recurrent-circuit replacement, new training result or flight test. The [complete result](result.json) preserves both arms and every source/input/backend hash.

| BANC index | Generic profile, Hz | Ionic crossings, Hz | Ionic 100–200 ms, Hz | Ionic 200–300 ms, Hz |
| --- | ---: | ---: | ---: | ---: |
| 12322 | 127.78 | 50.00 | 50 | 60 |
| 31143 | 77.78 | 22.22 | 30 | 30 |
| 41465 | 105.56 | 44.44 | 50 | 50 |
| 74497 | 77.78 | 22.22 | 30 | 30 |
| 84317 | 127.78 | 44.44 | 50 | 50 |
| 127563 | 133.33 | 50.00 | 50 | 60 |
| 135445 | 94.44 | 38.89 | 40 | 50 |
| 157417 | 72.22 | 16.67 | 20 | 20 |
| 160138 | 155.56 | 61.11 | 70 | 60 |
| 173140 | 100.00 | 38.89 | 40 | 50 |
| **Mean** | **107.22** | **38.89** | **43** | **46** |

The first two rate columns use the same fixed window, (100,280] ms: 193 generic events versus 70 ionic upward crossings across ten cells. Over the entire 300 ms trace, the totals are 373 generic events and 115 ionic crossings. These are raw counts divided by duration, not the legacy 50 ms filtered-rate readout. The displayed short windows are coarse estimates; 100 ms bins have 10 Hz count resolution and are not demonstrated steady states.

The lower ionic rate is not an artifact of prolonged depolarization or the author's event-detection guard. Guarded detector counts equal the upward-crossing counts in each reported window and over the full run. Their timestamps can differ slightly when a crossing occurs within the 10 ms detection guard, so the rates above explicitly use upward crossings. No V reset or refractory clamp was imposed on the ionic model. The shortest observed crossing interval was 9.7 ms.

All ionic states were finite. Across cells, V minima were approximately −60.884 to −60.868 mV and maxima +5.140 to +5.679 mV. Each cell spent only 2.77–6.43% of the full interval above −10 mV; the longest uninterrupted period above that threshold was 1.6 ms, or 1.4 ms after 100 ms. Thus no sustained plateau above the spike-detection threshold occurred in this experiment. h and b remained within [0,1]. The later 100 ms portions continue to contain repeated spikes, so the result is not merely a delayed first event or a transient followed by silence.

The zero-input control passed: zero crossings and zero guarded detections in every cell. All ten approached approximately −61.228566 mV; over 200–300 ms the control's voltage range was less than 8×10⁻⁸ mV. Both arms used the same captured initial V and declared h=b=0.146. No tonic current, adaptation term from the generic model, or electrical coupling was added.

## Numerical acceptance and preserved failure

The separately pinned [plan-numerical.json](../plan-numerical.json) has SHA-256 `49ce96ee40f07ae6fc9912ae0ad31f0e43333e8c87d99404141d0983dc4c9a52`. Five static gate checks passed before the run: four ULP accepted, five ULP rejected, excessive absolute error rejected, nonfinite V rejected, and any non-voltage bit difference rejected.

The legacy replay exactly reproduced the bytes of the earlier [strict attempt](../run-001/result.json). Relative to the original full-BANC capture, 448 of 6,000 voltage values differ: 423 by one ULP, 25 by two ULP, with maximum absolute error 0.00000762939453125 mV. The remaining 48,000 compared float32 values—including all counts, release values and last-spike times—are bit exact. `bitwisePassed` remains false; `numericalPassed` is true. The failed original plan and result have not been rewritten.

The compiler cause has not been established. WGSL permits reassociation/fusion and does not guarantee bitwise identity of mathematically equivalent expressions in different shaders; normal-range f32 division permits 2.5 ULP error. These rules motivate an explicit numerical gate, not a claim that the exact native instruction sequence has been diagnosed. The four-ULP cap is an engineering criterion for this diagnostic, not a bound supplied by WGSL for the whole recurrence. [Primary WGSL accuracy rules](https://www.w3.org/TR/WGSL/#floating-point-accuracy), [reassociation and fusion](https://www.w3.org/TR/WGSL/#reassociation-and-fusion).

## Scope of the finding

The generic motor membrane profile materially increases firing under this frozen incoming conductance sequence. A rate filter or muscle decoder cannot explain this result because neither determines these raw neural events. The ionic reference still receives strong fluctuating excitation and inhibition and continues firing at substantial rates; changing the intrinsic membrane model alone is not a calibration of the input conductances.

The receptor gates are model-generated BANC inputs, not measured physiological conductances. Replacing DLM neurons inside the recurrent network could alter future inputs, which were intentionally held fixed here. Initial gates, no DLM gap edges, the 0.5 ms held-conductance approximation and this short record remain explicit limitations. This establishes a bounded mechanistic comparison, not a validated whole-animal DLM profile, and says nothing yet about b1/DVM physiology or stable flight.
