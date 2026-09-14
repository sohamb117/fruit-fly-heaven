# Independent full-body integration review

No source-merge blocker found for corrected patch `27260780d4b7f99e84b1a3ffd667bcdd8c1e6d47c852ae6fd021df1fb60a47ce`. All 11 assembled modules pass forced ESM syntax parsing; all 64 pure tests pass. The review executes no native or neural simulation and edits no live source.

Exact-output checks reproduce every module from the frozen scene proposal, latest root vertical-score replacements, and telemetry v2. Removing the scene overlay from the assembled scorer reproduces root scorer `a0f158bffb1b37d567f9d29f70f81b71974270b4ad9c6c1013de8e2f222bd807` byte-for-byte after its canonical import substitution. Every source hash is rechecked after testing.

The full native body, BANC model, event reader and muscle implementation stay on the existing route. The optional power source equals the previously tested amplitude implementation; its eight added compatibility tests include 2,048-step absent-profile equality and 2,048-step common-0.90 equality. The three grounded objectives retain the original evaluator verbatim. The maintained scorer intentionally changes vertical-speed measurement to contact-free 50 ms COM displacement, preserving the −1 cm/s threshold with numerical tolerance and retaining attitude, angular-rate, support, contact, failure and final-current-bout gates.

The airborne helper is the exact previously validated full-body helper. A single live BANC/event/muscle history crosses the 0.5-second warm-up/release boundary. The root hold ends at release, and the clock contract rejects later root-write count changes. Scored windows start empty; no takeoff credit or prerecorded motor stream is introduced.

Scene configuration and body metadata must both declare the same strict finite profile. The world rejects a spacious scene on reduced or non-direct dynamics; the scorer rejects missing/mismatched observation profiles or ceilings. The same profile controls habitat, collision bounds, preview floor cap and reported criteria. With the scene absent, pure surface/odor/collision comparisons remain exact. A spacious room is a distinct maintained-flight curriculum, not evidence of obstacle avoidance or original-bowl success.

Telemetry is byte-identical to v2: absent-profile outputs retain legacy equality, optional power requests match the decoder, and six reduced claw muscle readouts are explicitly separated from structurally absent native actuation. Reduced body adapters are excluded from this combined patch.

The initial proposal failed forced ESM parsing because its constructor declared `scene` twice. Node 23.7's plain `node --check file.js` returned success for that typeless-file case; the failed forced check is preserved under `before-syntax-fix/`. The corrected constructor uses a distinct profile variable. Both the author and this independent review now parse every module with `node --input-type=module --check` through stdin.

Activation still requires a fresh asset manifest, model fingerprint, exact metadata/reset hash, environment version and training/checkpoint identity covering these merged sources and the amplitude semantics. Old scene plans are not manifests for this merged scorer/telemetry. This review approves the source merge only; no native flight outcome or learning claim follows from pure tests.

Reproduce with an absent result/test log: `node reports/flight-maintained-fullbody-review/review.mjs --patch-sha256 27260780d4b7f99e84b1a3ffd667bcdd8c1e6d47c852ae6fd021df1fb60a47ce`. Exact evidence is in `result.json` and `tests.tap`.
