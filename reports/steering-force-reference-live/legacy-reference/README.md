# Declared steering force reference

The explicit reference implements a centered local response, but did not produce useful flight improvement in this paired check. Both native fullBANC/Dawn-Metal episodes crashed in excessive rotation near390ms. No optimizer was run or checkpoint promoted.

| Seed | Best flight, old / reference (s) | Takeoff, old / reference | Return, old / reference |
|---|---|---|---|
| 190888 | 0.102 / 0.108 | false / false | -2.334 / -2.236 |
| 290888 | 0.168 / 0.174 | true / true | -1.022 / -1.752 |

The12 tied force references come from the mean native muscle outputs over100–280ms in the earlier seed888 unstable trajectory; they are neither physiological resting values nor independently validated hover trim. The27parameter vector, legacy recruitment rule, native body and zero-adhesion reduction are otherwise unchanged. Original arithmetic is preserved when the field is absent and was covered by native parity tests. The6ms increase on each seed, with mixed rewards and unchanged takeoff count, is not evidence of successful learning.

Immutable bundle/plan and full records are beside this report. [comparison.json](comparison.json) links the separate previous native-Dawn baseline. Untouched testseeds were not used; Safari was not available for visual observation.
