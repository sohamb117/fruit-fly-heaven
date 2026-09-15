# Larger search batches and bounded assignment runtimes

The scheduling update increases search rounds from two to eight perturbation pairs: **16 trials per search round**. Candidate comparisons and fresh validation keep their existing seeds, thresholds and counts.

The coordinator records a versioned scheduling policy, its effective generation and history separately from the immutable scientific configuration. The existing configuration hash remains `284e053d1a7b14e6a6fbe8d1ac08302350f90f55391c478feae61addef3bb446`, model fingerprint remains `7ec004c7f6f7e8021ccb329a2e85fe618505800dd169b0f177048ce7aa2e8b53`, and Firestore namespace remains `training_sequences/banc888-sensorimotor-wasm-20260915` in `flyheaven`. Reproducing the search requires both the configuration and the recorded scheduling policy.

An explicit atomic migration appends deterministic pairs to an active search round and preserves existing accepted rows and checkpoint values. If comparison has already begun, its assignments stay intact and the larger pool starts with the next search round. Compatible server code must receive all traffic before this migration; the older server assumes four jobs and cannot read an expanded search batch.

## Assignment limits

- The renewable inactivity lease remains 180 seconds, renewed every 60 seconds.
- Each attempt also has a fixed runtime deadline: 1,200 wall seconds per five assigned simulation seconds, with a 600-second minimum.
- The deployed horizons therefore allow 20 minutes for maintained flight/recovery, 12 for takeoff, 32 for landing and 60 for the three-episode decoder fit. Rest and pauses count toward the deadline.
- Existing leases receive one recorded migration grace window because their original start time was not stored. Repeating the migration does not extend that window.
- Heartbeat, result and lease selection all enforce the deadline. Reassignment retains the candidate and seed but creates a new lease token and attempt deadline.
- The overdue owner waits 60 seconds before reclaiming that same trial; it can receive other work meanwhile. Another browser can claim the overdue trial immediately.
- Updated clients recover from `assignment_timeout`, `lease_expired` and `stale_lease` automatically. Other conflicts and invalid results remain errors. Late heartbeat replies cannot discard an in-flight accepted upload, and Stop/Pause remain authoritative.

The scheduler still waits for each batch to complete. This update bounds individual attempts and supplies more search work; it does not turn the algorithm into an asynchronous optimizer.

## Deployment and validation

Live at **https://flytrain.morisoba.moe/train.html** on revision `fly-training-r6d8663150cbc`, with 100% traffic. Cloud Build `dc7e02cd-154f-4a04-85c7-bbb783eac0ab` produced image `us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:5d147ee8d53b5888f8b3ba5f927bcc83515c493a244d227fba84ee45a9a74c36` from build context `dist/cloudrun/scheduling-20260915`, build ID `ff1582946858e5df`.

The live migration appended 12 jobs to generation zero, recorded policy revision one and granted the one existing lease its migration grace deadline. At the 08:03 UTC audit, all three previously completed Firestore rows matched their earlier hashes exactly, the checkpoint values/config hash were unchanged, and a fourth result had been accepted. The batch had 16 jobs, and new assignments carried a 1,200-second fixed deadline. All 15 public HTTPS checks passed, and the deployed client matched the packaged source byte for byte.

Safari was refreshed while idle and resumed trial three from the expanded batch. Its five displayed active sensory values matched the new Firestore assignment, and the native preview clock advanced from 0.2 to 0.4 seconds. The user's 100% intensity, Low preview quality and Fly view were restored. The three earlier accepted results, the fourth completion during deployment, and this new partial trial are distinct pieces of evidence; no completed evaluation on the refreshed Safari client is claimed here.

The browser bundle includes the existing Fly, Brain and Eyes views. Eye images and neural/physics execution are unchanged. The client update only changes assignment recovery and API error handling.

All 180 focused tests passed: 31 SQLite sequence tests, 29 Firestore tests (including 23 against the local emulator), 32 Cloud Run tests and 88 client tests. Local HTTP regression tests exercise all 16 leases, continued heartbeat renewal up to the hard deadline, late-result rejection, reassignment to another browser and exactly-once acceptance. SQLite and real local Firestore emulator tests cover migration, state preservation, restart, concurrency and runtime limits. Client tests cover cancellation, recovery, Pause/Stop and upload races. These are protocol checks, not evidence of better flight behavior. No production trial was deliberately allowed to time out for verification.

Deployment receipts and live preservation checks are recorded alongside this file. `before-completed-records.json` contains hashes of previously accepted Firestore rows, never their lease tokens or contributor identities. Public status/checkpoint snapshots are in `before-status.json` and `before-checkpoint.json`.
