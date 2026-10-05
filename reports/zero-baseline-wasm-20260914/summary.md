9 / 9 validated trials; 0 missing, 0 invalid. Frozen checkpoint: generation 7.

| Condition | Seed | Score | Best flight (s) | Scored time (s) | Outcome | Wall (s) | Execution (s) |
|---|---:|---:|---:|---:|---|---:|---:|
| zero | 2490888 | -3.0000 | 0.0000 | 0.1600 | overturned | 340.6954 | 320.2486 |
| initial | 2490888 | -2.9804 | 0.0140 | 0.4100 | excessive_rotation | 281.6501 | 278.2333 |
| current | 2490888 | -2.3000 | 0.5000 | 0.9740 | excessive_rotation | 216.6547 | 212.6440 |
| zero | 2590888 | -3.0000 | 0.0000 | 0.1600 | overturned | 96.7046 | 94.7804 |
| initial | 2590888 | -2.8180 | 0.1300 | 0.8620 | excessive_rotation | 196.9110 | 193.2145 |
| current | 2590888 | -3.0000 | 0.0000 | 0.1800 | excessive_rotation | 98.5427 | 96.5532 |
| zero | 2690888 | -3.0000 | 0.0000 | 0.1600 | overturned | 96.4551 | 94.5116 |
| initial | 2690888 | -2.6500 | 0.2500 | 0.9080 | excessive_rotation | 2266.3712 | 2187.1471 |
| current | 2690888 | -2.9356 | 0.0460 | 0.2600 | excessive_rotation | 310.5825 | 276.6328 |

Matched differences (left minus right; complete pairs only):

| Comparison | Pairs | Mean score Δ | Mean best-flight Δ (s) | Mean scored-time Δ (s) |
|---|---:|---:|---:|---:|
| initial - zero | 3 / 3 | +0.1839 | +0.1313 | +0.5667 |
| current - zero | 3 / 3 | +0.2548 | +0.1820 | +0.3113 |
| current - initial | 3 / 3 | +0.0709 | +0.0507 | -0.2553 |

- Missing and invalid trials are excluded, never scored as zero.
- Differences use only matching seeds with two validated complete results.
- A safe five-second ground outcome can score above a powered crash; compare qualified flight and termination alongside reward.
- All-zero means the 672 learned decoder coefficients; fixed neural, muscle and body mechanisms remain active.
- Three seeds are a diagnostic comparison, not conclusive behavioral validation.
