# First legacy replay: voltage rounding differences only

The original strict gate **failed**. `run-001` and its pinned plan remain unchanged, and the ionic arms did not execute. This is an offline audit of the saved native capture and saved replay, not a new neural simulation. Binary hashes and the complete per-field/per-cell counts are in [failure-analysis-001.json](failure-analysis-001.json).

| Compared field | Different values / 6,000 | Largest difference |
| --- | ---: | ---: |
| Voltage | 448 | 2 ULP = 0.00000762939453125 mV |
| Adaptation | 0 | Bit exact |
| Refractory time | 0 | Bit exact |
| Cumulative spike count | 0 | Bit exact |
| Filtered firing rate | 0 | Bit exact |
| Instantaneous release | 0 | Bit exact |
| Total membrane conductance | 0 | Bit exact |
| External current after adaptation | 0 | Bit exact |
| Last-spike time | 0 | Bit exact |

There are 423 voltage differences of one ULP and 25 of two ULP. All events agree in identity and time throughout the 300 ms trace. The first voltage difference is cell 173140 at 33.5 ms: captured −45.92632293701172 mV, replay −45.926326751708984 mV. There are 88 onsets of differing-voltage runs; the longest run lasts 24 samples. No growing voltage divergence or altered spike sequence occurs in this record.

## What the source audit establishes

The legacy and standalone membrane expressions use the same parameters and the same arithmetic text for gain, adaptation, implicit voltage update, refractory/reset/spike branches, release, EMA and cumulative counts. The standalone kernel receives captured post-kinetics gates directly rather than recomputing them in the preceding receptor loop. The absent DLM gap-edge contribution is exactly zero in both. Direct external input is zero, so replacing its atomic load with the captured float cannot explain the voltage-only difference. Exact total conductance, adaptation and external current further narrow the possible discrepancy to the unrecorded reversal-weighted sum or the voltage expression's evaluation.

For the first mismatching row, the previous voltage is identical (−51.261714935302734 mV). Its conductances are [10.667329788208008, 1.0134607553482056, 0.17457111179828644, 0, 0] nS; reversals are [0, −75, −75, −75, 0] mV. Both fused and separately rounded float32 multiply-add accumulation give the same reversal sum, −209.10238647460938 pA. With the captured current −0.39207932353019714 pA and total conductance 13.855361938476562 nS, the ordinary scalar float32 expression gives the captured voltage. This calculation uses one recorded row; no candidate trajectory was integrated.

That does **not** establish the actual native compiler instruction sequence. Simple fused-versus-unfused reversal summation does not explain this first discrepancy. A different compiler evaluation of division, reassociation or another intermediate in the larger versus smaller shader remains a plausible explanation. The generated Metal code or additional instrumented diagnostic would be needed to identify it precisely.

WGSL permits reassociation and fusion, does not require one rounding mode, and specifies up to 2.5 ULP error for normal-range f32 division. Identical mathematical expression text in differently optimized shaders is therefore insufficient to guarantee bitwise identical voltage evaluation. These rules support a numerical interpretation of the observed discrepancy; they do not prove its exact cause or provide a four-ULP bound for an entire recurrent computation. [WGSL floating-point accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy), [reassociation and fusion](https://www.w3.org/TR/WGSL/#reassociation-and-fusion).

## Recommended next gate

Keep this failed attempt visible. Use a new plan and result directory if adopting a revised gate:

- Require every non-voltage field, every event count and every event time to remain bit exact at every half-ms step.
- Require all voltages finite, with both an absolute error no larger than 0.0001 mV and an ordered-float distance no larger than four ULP. The absolute tolerance already exists in the author-reference numerical comparison; the ULP cap adds a much tighter requirement in this voltage range.
- Record both `bitwisePassed` and `numericalPassed`, the full per-field mismatch distribution and the prior failed run's hash. Never label a numerical pass bit exact.
- Keep all source, backend, input, continuity, no-tonic/no-gap, and zero-input control requirements unchanged. A different event or non-voltage state remains a failure; do not repeatedly widen the criterion to admit a later mismatch.

This is a proposed engineering acceptance criterion selected after inspection of a documented precision failure, not a pre-registered physiological result. It is appropriate to the immediate purpose: validate that frozen conductance timing and membrane scheduling reproduce the actual event/state dynamics closely before comparing a different ionic membrane model. The independent full-graph capture already passed its original exact gates. Changing incoming conductances, neural parameters, or the captured physical trajectory would not be an acceptable repair for this failure.

If bitwise voltage equality itself is required, the next work is a compiler/intermediate-value diagnostic or original-kernel replay with sufficient upstream state. That is separate from the biological comparison; reconstructing a presynaptic drive by numerically inverting recorded receptor gates would not restore the original arithmetic exactly.
