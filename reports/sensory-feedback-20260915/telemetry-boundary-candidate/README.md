# Requested-rate telemetry and sampling-boundary candidate

Unapplied two-file patch based on the exact v2 bundle sources:

- `web/training/sensory-feedback.js`: top-level eye `leftHz`/`rightHz` now come from the active compact-motion summary, with `rateSource: 'requested T4/T5 rates'`; readiness also describes that active path. The previous encoder eye summary is preserved under `luminanceSummary`, and `luminanceInputEnabled:false` remains explicit. Repeating an update at the same cached input does not recursively nest the preserved summary.
- `web/training/compact-vision.js`: adds a one-nanosecond tolerance to the maximum sampling-gap comparison. It does not round or change the actual `dt` used in rate calculations.

The patch changes telemetry for the current 20 ms profile, and changes actual transduction only at the intended maximum-gap boundary where floating-point subtraction previously caused spurious resets. No other behavior change was found by these tests.

Validation:

```sh
node --test reports/sensory-feedback-20260915/telemetry-boundary-candidate/validate.test.mjs
```

All three tests passed:

1. Two isolated processes load frozen v2 sources or the candidate through the override-aware runtime. Twelve prescribed-pose samples from 0 to 120 ms at 256x128 per eye produce bit-identical complete afferent-input index/rate buffers and bit-identical compact-motion index/rate buffers. Both eyes exhibit nonzero requested motion rates. This tests transducer output, not full neural-network or body evaluation.
2. Top-level eye means equal the active requested T4/T5 means. The old zero L1/L2/L3 eye means remain accessible in the preserved summary. Repeated cached updates keep an identical finite summary.
3. The old implementation demonstrably resets on some successive exact 100 ms decimal timestamps. The candidate remains ready and produces nonzero motion rates at all those boundaries. A 100.001 ms gap still resets to zero as intended.

`baseline.capture.json` and `candidate.capture.json` preserve the exact typed-array bytes as base64 for the current-profile comparison. No original runtime files or frozen v2 bundles were edited, and no neural/body simulation was run.

Apply `telemetry-boundary.patch` after the earlier vision-contract patch. The final workspace will differ from frozen v2 in these two sources. The full v2 evaluation remains valid evidence for its exact frozen version; the tests here establish equivalence of requested sensory inputs on the current 20 ms profile, not a new full trial fingerprint.
