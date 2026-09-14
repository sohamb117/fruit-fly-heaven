# Prescribed left virtual-haltere motion: saved-record analysis

**The control recruited the previously silent left haltere sensory population and strongly increased B1 activity, but did not rescue this fixed policy.** Both 18° and 36° arms achieved 440 ms best continuous qualified flight, compared with 504 ms at 0°. All three took off and terminated with excessive rotation; none landed successfully. This is one seed (1290888), with the same 27 parameters, source assets, backend and initial state.

## Verification

- The 0° arm matches the explicitly pinned unobserved C reference: complete physics digest, all 363 wing-event packets, config, model fingerprint and native backend provenance. Its physics digest is `8a7f9ab12b53111852f1056ccd6e37b6935ba6ffe780545bb2d5cd0a2f5ff561` (363 rows × 630 values).
- All three arms have exactly equal observations, wing packets, haltere feedback and all 330 observed neural states through 100 ms.
- All commands activate at native body time `0.09999999999999715` s, within the declared 1 ns tolerance. The next input block consumes the prescribed left amplitude; its post-physics record is at approximately 102 ms. There is no extra 2 ms onset delay.
- Every recorded actual muscle-power pair equals the native post-physics pair captured before the override. Every right sensory drive equals the corresponding native right drive. The pinned, reviewed helper changes only a shallow sensory-feedback copy; it changes no native muscle state, controls, phase or velocity. Neural feedback can subsequently change native power indirectly.
- Native left hDVM power remains exactly zero throughout all three runs, despite the prescribed left sensor motion. All recorded observations have zero external applied root force.

## Matched response window: (100,720] ms

This shared 620 ms window excludes the identical pre-onset interval and avoids different termination-time denominators. “Active” means at least one observed spike in the window, rather than nonzero modeled current at a particular phase. Sensor Hz is the mean count-derived rate per cell. Motor counts are cumulative-count differences checked against B1 event records.

| Left amplitude | Left sensor spikes | Active left cells | Left sensor mean Hz/cell | hDVM spikes L/R | B1 spikes L/R | B1 count rate Hz L/R |
|---|---:|---:|---:|---:|---:|---:|
| 0° | 0 | 0/171 | 0.00 | 0/22 | 16/31 | 25.81/50.00 |
| 18° | 11,581 | 171/171 | 109.23 | 0/35 | 71/53 | 114.52/85.48 |
| 36° | 24,001 | 171/171 | 226.38 | 0/44 | 98/56 | 158.06/90.32 |

Left sensor cumulative counts first diverge at the 104 ms readout for 18° and the 102 ms readout for 36°. These are 2 ms observation boundaries. First left B1 events after onset occur at 146 ms (0°), 111 ms (18°), and 108 ms (36°). B1 identities are left **75865** and right **99458**; hDVM identities are left **97021** and right **118683**.

Right sensory activity also changes through the closed feedback loop: all 157 right cells are active, with 13,205 / 20,311 / 21,387 spikes in this window for 0° / 18° / 36°. Right hDVM drive remains endogenous. These downstream changes should not be mistaken for a direct right-side override.

The recorded 50 ms filtered B1 rates at 720 ms are L/R **16.81/31.25 Hz**, **128.87/123.58 Hz**, and **150.96/80.30 Hz**. They differ from count/window rates by definition. The event-driven muscle path uses recorded events; these filtered rates are descriptive readouts.

## Whole episodes

| Left amplitude | Episode duration | Best flight | Left sensory spikes | Active left cells | hDVM spikes L/R | B1 events L/R | B1 count rate Hz L/R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0° | 724 ms | 504 ms | 0 | 0/171 | 0/26 | 16/33 | 22.10/45.58 |
| 18° | 728 ms | 440 ms | 11,742 | 171/171 | 0/39 | 72/55 | 98.90/75.55 |
| 36° | 720 ms | 440 ms | 24,001 | 171/171 | 0/48 | 98/58 | 136.11/80.56 |

Whole-episode counts/rates use different denominators (724, 728 and 720 ms) and include the pre-onset period. The matched table above is the primary response comparison. Complete right-sensor and filtered-rate statistics for whole episodes, matched prefixes and post-onset windows are saved in [result.json](result.json).

## Interpretation and limits

The new pathway can recruit left sensory cells and alter B1 activity. This is a demonstrated response in the current ionic-DLM/event-muscle system, rather than a conclusion extrapolated from the older rate/Hill assays. It does not establish corrective signs, suitable spike phase, calibrated gain or stable feedback. More sensory/motor spiking did not produce a longer qualified flight interval here.

The intervention prescribes virtual sensor motion without commanding a native haltere actuator or modeling passive wing–haltere mechanical coupling. The amplitudes, current sensitivity and receptive orientations remain declared priors. These results neither validate a physical coupling implementation nor show that halteres are irrelevant; two fixed amplitudes failing to rescue one policy is insufficient for either conclusion. The experiment performs no training or optimization.

## Reproduce the saved-record analysis

```sh
python3 reports/flight-haltere-motion-control/analyze-records.py
```

This reads saved JSON and verifies pinned files only; it imports no model and runs no neural/body simulation. [result.json](result.json) includes source and input SHA-256 hashes, all assertions, exact timing and per-window summaries. The three raw records are under [evaluation](evaluation/).
