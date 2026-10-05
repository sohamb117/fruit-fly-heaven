# Preparing one live BANC affine comparison

Use a clone of `reports/flight-haltere-live/legacy/bundle.json`. Do not rerun `scripts/prepare-flight-development-bundle.mjs`: it reads canonical current models/configuration and resets the saved 27 initial parameters. The isolated pair must retain the current event-muscle, DLM ionic and tegula feedback configuration, with mechanical haltere feedback still absent.

`prepare-live-pair.mjs` only reads inputs and writes a new directory inside `reports/`. It never patches source, starts a server, imports a native runtime or trains. It verifies every non-model manifest asset and snapshots referenced JavaScript/WGSL/Python sources, evaluator helpers, development server/coordinator, proposed patch, bundle, capture inputs and tooling pins. Native binaries and BANC data are checked and recorded rather than duplicated. Existing output directories are rejected.

Save source bytes before applying anything:

```sh
node reports/flight-affine-live-staging/prepare-live-pair.mjs --archive-only --output=reports/flight-affine-live-source-archive
```

After the accepted calibration has produced the bare four-key `profile.json` and separate `profile-provenance.json`, record its SHA-256 as the predeclared commitment. Then prepare the pair with that literal digest and actual new paths:

```sh
node reports/flight-affine-live-staging/prepare-live-pair.mjs --output=reports/flight-affine-live-pair --profile=reports/ACCEPTED_PROFILE/profile.json --profile-sha256=RECORDED_SHA256 --calibration=reports/ACCEPTED_CALIBRATION/result.json
```

The script requires completed, source-unchanged, repeatable, accepted native trim and suitable Jacobian flags, a matching calibration plan, the original source bundle/capture, exact retained frequency, and exact T/F0/power-scale agreement. It does not fabricate acceptance for a failed run. This preparation gate does not replace the separately reviewed frozen-motor A/B gate.

## Exact source and schema changes

The existing dev server accepts **only** XML and model-metadata overrides; executable route overrides are forbidden. The evaluator also imports local `web/` modules and verifies all non-model manifest hashes. Embedding staged JavaScript in a bundle cannot substitute for the eventual one-file opt-in source patch.

Both prepared arms pin the exact bytes of `reports/flight-affine-transfer-staging/flybody-wings.proposed.js` as `/flybody-wings.js`. The expected new hash is `ab3203b24df52c438d15fd84a76b134a283eaa8e1e9eb164036aef46b1d44c06`; the old hash is `10a44cb2ed11835ed3a151081586095e6bd24527f0c20926c9a006783fec7084`. Preparation verifies the proposed file is the reviewed staging module with only its relative import restored for `web/`.

Control A retains the parent XML and metadata bytes exactly. Treatment B adds:

```text
metadata.wing_actuation.flight_operating_point = {
  schemaVersion: 1,
  reference: {left: twelve exact muscle values, right: twelve exact muscle values},
  trim:      {left: twelve exact muscle values, right: twelve exact muscle values},
  powerScale: {left: positive finite scale, right: positive finite scale}
}
```

B also records profile/calibration hashes in `metadata.diagnosticVariant.affineOperatingPoint`. The XML bytes and `metadata.xml_sha256` remain unchanged. The strict constructor rejects malformed profiles and simultaneous legacy force-reference mode. No environment loader or training config-schema changes are necessary. Config schema version remains 2; profile schema is 1.

Both configs change the wing-source asset digest, recalculate `modelFingerprint` from sorted `url:sha256\n` entries, and recalculate `configHash` from exact serialized `configText`. Only B changes the metadata asset digest. Environment versions append `-affine-source-control-v1` / `-affine-operating-point-v1`. All 27 parameter definitions, bounds, initial values and job values remain identical to the parent; stage is `landing`, seed is 1290888, horizon is 8 seconds. No gain recentering, frequency change, objective change or new feedback feature is included.

## Later execution, by the parent after diagnostic gates pass

The source patch remains a separate action. Apply only the proposed one-file change after archiving and review; do not update production config/manifest or deploy. Once applied, the old bundles intentionally fail their old source hash against this checkout. Their archived old sources and original artifacts remain available; do not rewrite those bundles to conceal the change.

Each bundle needs a separate dev-server configuration and evaluator process. Run sequentially with one fly at a time, fresh local SQLite paths and unused loopback ports. The following commands are instructions only; this staging task did not execute them:

```sh
python3 scripts/serve-training-dev.py --port 7861 --database reports/flight-affine-live-pair/control/coordinator.sqlite3 --bundle reports/flight-affine-live-pair/control/bundle.json
node scripts/evaluate-training-native.mjs http://127.0.0.1:7861/ reports/flight-affine-live-pair/control/plan.json reports/flight-affine-live-pair/control/evaluation
```

After completing and checking A, stop its server and use the same pattern for `affine/` with another unused port (for example 7862). A/B cannot be two jobs in one evaluator invocation because that invocation has one fixed config/model. Do not launch contributor training: the evaluator uses same-origin GET/HEAD only, obtains no lease and uploads no result. The server's local coordinator is only serving the isolated configuration for this run.

Plans retain native Dawn/Metal pins and capture physics digests, full flight history, all 48 wing event packets, actual haltere power, and the existing first-500-ms motor replay/force observer. The evaluator requires the planned backend; fallback is rejected. It runs to 8 seconds or native physical failure (`outside_habitat`, `excessive_rotation`, `overturned`), not an artificial short cap.

Require A's full physics digest and wing packets to reproduce the legacy reference before interpreting B. Once live sensory feedback is connected, B's neural events and muscle forces may diverge; that is expected closed-loop behavior, not frozen-motor equality. Compare complete objective state, takeoff, continuous controlled airtime, landing, termination, contacts and pre-impact motion. One predeclared seed is a diagnostic result, not trained robustness or proof of successful flight.
