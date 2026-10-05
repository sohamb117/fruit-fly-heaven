# Maintained-flight diagnostic preparation

`prepare-plans.mjs` creates three new, one-job plans. It checks the source-pinned legacy bundle, every original asset, the frozen environment/runner, the measured affine profile and its provenance before writing anything. It refuses an existing output directory. `plans-v1/ledger.json` records the hashes and archives 77 source/evidence files. No live source changes, server, optimization or native evaluation were performed during preparation.

| Plan | Start and objective | Wing transfer |
| --- | --- | --- |
| `grounded-source-control` | Original grounded reset; landing stage, 8 s or physical failure | Original |
| `maintained-baseline` | Native root held at `[0,0,3.5,1,0,0,0]` for 0.5 s of live BANC/event/muscle warm-up; then 5 s maintained flight or physical failure | Original |
| `maintained-affine` | Same warm-up and maintained-flight objective | Predeclared measured affine profile and proposed wing module |

Every job retains the exact saved 27 parameters and seed 1290888. The new objective grants no takeoff or landing credit. Warm-up preserves the neural, event, wing, and native clocks and muscle state; scoring starts empty when the root hold is released. The affine profile previously failed cold launch, so its airborne-start arm is a conditional diagnostic, not promotion or evidence of successful takeoff.

Preparation command:

```sh
node reports/flight-maintained-native-staging/prepare-plans.mjs --output=reports/flight-maintained-native-staging/plans-v1 --environment-sha256=3d3c45cf67aab2372076c9a8c46f5c15f503c906bdad9b5f0a2ebd361cf21fd3 --runner-sha256=9cee4905fe6428c46d1041851f2fc57e871bf97bfc2a57dc85857d41afd5d9cb
node reports/flight-maintained-native-staging/verify-plans.mjs reports/flight-maintained-native-staging/plans-v1
```

The commands intentionally refuse overwriting prepared artifacts. Use a new output directory after any source change.

Each `plan.json` uses `kind: maintained-native-diagnostic-plan`, an exact config SHA and sorted asset-manifest fingerprint. Overrides contain repo-relative file paths and SHA-256 hashes. All arms override the staged environment and three `/diagnostic/` modules, plus XML and metadata bytes from the saved bundle. Only the affine arm changes `/flybody-wings.js` and adds `wing_actuation.flight_operating_point`. Every override hash is in `config.assets`; `.initialCondition` exists only for maintained flight and pins the exact metadata digest. Unchanged assets are read from the existing loopback endpoint `http://127.0.0.1:7859/`; no new server or live patch is required.

`evaluate.mjs` is a separate staged native runner owned by the parent task. It verifies all plan/config/source/asset pins before and after an evaluation, blocks coordinator APIs and non-loopback fetches, and refuses an existing output directory. Run the grounded source control first. Its `expectedPhysicsDigest` is the full saved digest object, with SHA `c8abcff173e74f7a118005b2349774b587492ac0c4447bb1b303f8af0806b9c9`; the runner compares the SHA. Complete motor-history comparison is a separate saved-record check after evaluation. Hash and preparation checks alone do not establish physics parity or maintained-flight performance.
