# Live generation 10 versus refined decoder

All six result records validate.

| Seed | Controller | Score | Best flight (ms) | Duration (s) | Success | Reason |
|---:|---|---:|---:|---:|---|---|
| 2490888 | live-generation10 | -2.9216 | 56.0 | 0.376 | false | excessive_rotation |
| 2490888 | refined-v2 | 0.0000 | 740.0 | 5.000 | false | time_limit |
| 2590888 | live-generation10 | -2.5632 | 312.0 | 1.718 | false | excessive_rotation |
| 2590888 | refined-v2 | -2.4232 | 412.0 | 1.830 | false | excessive_rotation |
| 2690888 | live-generation10 | -2.5184 | 344.0 | 0.962 | false | excessive_rotation |
| 2690888 | refined-v2 | 0.0000 | 1016.0 | 5.000 | false | time_limit |

Matched pairs: 3/3.
live-generation10: mean score -2.6677, best flight 237.3 ms, duration 1.019 s, successes 0/3.
refined-v2: mean score -0.8077, best flight 722.7 ms, duration 3.943 s, successes 0/3.

Latest ES vector is evaluated without fitting, clipping, rescaling or output override.
Stored v2 results use the same frozen model/config/WASM and the same three seeds; they were not rerun.
All trials start airborne after 0.5 s live warm-up; takeoff, landing and feeding are not tested.
Local held-out diagnosis only; no cloud submissions, parameter promotion or deployment.
Three deterministic seeds provide a bounded comparison, not a population-level guarantee.

Coordinator reports zero parameters changed in generation 10. No separate generation-9 vector was available for an independent equality check.
