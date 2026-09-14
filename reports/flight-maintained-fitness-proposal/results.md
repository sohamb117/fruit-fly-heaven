# Aggregate replay of completed training records

The snapshot contains **18 completed training episodes**: 12 search evaluations and six generation-zero acceptance evaluations. No episode succeeded. No held-out evaluation was read or run. All 18 existing terminal returns reconstruct exactly from the stored scorer counters, with maximum numeric error zero. The source and artifact hashes are in `aggregate-replay.json`.

The sampled zero-score timeouts were not mostly-grounded episodes: the three incumbent acceptance runs accumulated **3.370, 3.764 and 3.526 qualified seconds**, with **0.112, 0.054 and 0.100 recorded contact seconds** respectively. Their final bouts ended before the timeout. The current scalar therefore discarded substantial flight evidence.

| Completed job | Qualified total | Best bout | Outcome | Current return | Proposed T-only return |
| --- | ---: | ---: | --- | ---: | ---: |
| g0-p2-neg | 4.108 s | 1.232 s | 5 s timeout | 1.2348 | 4.108 |
| g0-p3-pos | 2.830 s | 1.512 s | Physical failure at 3.570 s | -0.8832 | 1.830 |
| g0-a0-neg, incumbent | 3.370 s | 1.532 s | 5 s timeout | 0 | 3.370 |
| g0-a0-pos, nominee | 2.842 s | 1.510 s | Physical failure at 3.528 s | -0.8860 | 1.842 |
| g0-a1-neg, incumbent | 3.764 s | 0.972 s | 5 s timeout | 0 | 3.764 |
| g0-a2-neg, incumbent | 3.526 s | 1.252 s | 5 s timeout | 0 | 3.526 |

The proposed primary rule still selects `g0-p2-neg` as the highest-scoring member of the completed eight-job generation-zero search batch. The matched acceptance comparisons are:

| Pair | Current nominee minus incumbent | Proposed nominee minus incumbent |
| --- | ---: | ---: |
| g0-a0 | -0.8860 | -1.5280 |
| g0-a1 | -1.3032 | -2.2920 |
| g0-a2 | -1.2976 | -0.9140 |
| Mean | -1.16227 | -1.5780 |

Both scoring rules reject this nominee. The different numerical scales should not be interpreted as a measured increase in physical regression. The optional continuity blend also rejects it, with a mean difference of -1.45027; it was not selected or tuned using these results.

This evidence supports repairing the loss of earned airtime. It does **not** establish that the scoring defect caused the current optimizer's failure, that the rejected candidate should have been retained, or that a new scorer will produce stable flight. Physical trajectories, qualifications and termination times are unchanged in this arithmetic comparison. The four available generation-one search records are retained in the snapshot but an incomplete generation is not ranked as a finished experiment here.
