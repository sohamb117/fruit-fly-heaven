# Sequential sensorimotor training

Implemented on 2026-09-15 using simulation-based engineering calibration. This is not measured sensory physiology or evidence that the complete behavioral sequence has been learned.

## Run

The final local bundle is `v4/sequence.bundle.json`. It contains 696 parameters: 24 sensory priors followed by the existing 672 anatomical decoder coefficients. Its configuration hash is `284e053d1a7b14e6a6fbe8d1ac08302350f90f55391c478feae61addef3bb446`; its model fingerprint is `7ec004c7f6f7e8021ccb329a2e85fe618505800dd169b0f177048ce7aa2e8b53`.

```sh
python3 scripts/serve-training-dev.py --port 7886 \
  --bundle reports/sequential-training-20260915/v4/sequence.bundle.json \
  --database reports/sequential-training-20260915/v4/coordinator.sqlite3
```

Open `http://127.0.0.1:7886/train.html` in Safari. Restarting with the same bundle and database resumes the experiment. The source builder is `scripts/prepare-sequential-training.mjs`; its existing prepared data and frozen calibration inputs are required. Large prepared data and generated bundles remain local artifacts.

The v4 page was launched in Safari with one worker at the default 60% scheduling budget. Its first job is the leg-feedback phase. The server's configuration and pinned runtime/teacher assets were verified over HTTP. The superseded v2 worker was stopped and its lease released before v4 started.

Order: leg feedback, antenna feedback, haltere feedback, wing strain feedback, visual feedback, decoder demonstration fitting, autonomous recovery, takeoff, landing. Only the active family's coordinates change. Every evaluation is assigned by the coordinator. There is no offline optimizer fallback.

Accepted improvements update the saved vector after two separate comparisons on matched seeds, even before task mastery. Phase completion also requires task success. Later phases retest previous physical tasks. Exhausting a phase's budget preserves its best accepted values and stops for review. `lastParameterUpdate` records the changed-coordinate count and update magnitude. See [the full contract](../../docs/training-sequence.md).

## Verification

- 24 coordinator tests: parameter masks, incremental acceptance, fresh validation, retention, restart, leases, exact retries and failed fits.
- 19 local HTTP tests: immutable assets, source identity, MIME types and the 696-value contract.
- 18 Firestore tests, including 13 using the local emulator; 11 Cloud Run server tests and 6 packaging tests. The exact v4 configuration initialized and resumed in a separate emulator client with identical state. The maximum batch is 15 jobs; estimated pending storage is 1.20 MB. See `firestore-storage-validation.json`.
- 11 Cloud Run deployment preflight tests, including an experiment that has already advanced to a later phase.
- Browser runtime, fitting, recovery, client and rendering tests passed. A final focused rerun passed 38 tests after correcting the demonstration collector's native substep clock.
- `v3/native-teacher-smoke.json` uses the same model fingerprint as v4 and real native MuJoCo/WASM mechanics. It verifies 2,600 correctly phased samples, a 500 ms restrained setup, a 20 ms free recovery segment, unchanged history timing and exact cleanup. Its neural packets are explicitly synthetic and empty. It establishes boundary plumbing, not fitting or autonomous recovery.
- Safari exercised the actual BANC/MuJoCo worker, updating the 3D fly, brain display and current trial values. The first five leg-feedback coordinates changed while the remaining values stayed fixed. No accepted checkpoint improvement was established during the implementation checks.

Intermediate bundles are retained for audit. The older v2 ordinary trial was used for browser observations; v4 adds the corrected demonstration clock and explicit incremental-acceptance contract. Do not resume v2 with the current coordinator.

The initial implementation checks above made no GCP changes. The sequence was subsequently deployed with eye views at the user's request; see the [production release record](../flyheaven-sequence-eyes-20260915/README.md). Its production state is independent of these local test databases.
