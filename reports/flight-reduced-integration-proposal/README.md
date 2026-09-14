# Optional reduced-body integration proposal

Staged eight-file patch only; nothing applied. The body adapters, maintained-flight environment/reset/contract/scorer, and telemetry v2 are copied from their frozen diagnostic sources. Only the three maintained-module import/asset paths are changed to canonical training paths. Wing code, XML, metadata, configuration, neural model and checkpoints are excluded.

`canonical-optional-reduced.patch` applies cleanly against the pre-integration source archive. `verify.mjs` independently checks exact source transformations, canonical dependency paths and syntax. All 37 pure fixtures pass: 8 reset, 7 body adapter, 9 scheduling and 13 scoring tests. The byte-identical telemetry module separately passes 11 tests. Full-body adapter v1 has a parent-run native parity result; telemetry v2 and reduced native evaluations are separate pending parent-run gates, not evidence supplied by this patch.

The model selection remains explicit: `full-native` or `fixed-nonwing-flight`. Reduced claw actuation is absent and reported as such; its mapped neural muscle readouts remain available. Warm-up alone holds the root; release ends those writes. The existing three grounded stages retain their original evaluator. The patch does not promote reduced flight as takeoff, landing, feeding, or learned success.

Source hashes and transformations are in `provenance.json`; exact test/path verification is in `validation.json`. Reproduce into an absent staged/output layout using `prepare.mjs --telemetry-sha256 6ba0613840fda4259a7f9b620f6ac3526b076271c354cccee4c7aaf109c6393b`, then the fixture command in provenance and `verify.mjs`. Existing files refuse overwrite.
