# Managed training service on Cloud Run

Cloud Run serves the browser bundle and coordinator; contributors' browsers run BANC and the full FlyBody/MuJoCo body in WASM. Firestore stores progress independently of container lifetime. The live entry is **[flytrain.morisoba.moe/train.html](https://flytrain.morisoba.moe/train.html)**: **Start** runs one assigned fly locally, **Pause** suspends it, and **Stop** releases unfinished work. Completed contributions go to the GCP coordinator. Opening the page alone starts no simulation. The Google service alias is `https://fly-training-yekt6i27nq-uc.a.run.app`.

## Current release: sequential training and Eyes

The subsequent [scheduling update](../../reports/flyheaven-scheduling-20260915/README.md), revision `fly-training-r6d8663150cbc`, now serves 100% of traffic. It keeps this experiment and namespace, increases search batches to 16 trials, and adds absolute per-attempt deadlines. Its release receipt records the immutable image digest and migration. The revision below introduced the sequence and Eyes views.

The initial sequence/Eyes release on 2026-09-15 promoted revision `fly-training-r0b7a72087eed` with image digest `sha256:324f99174bd6edab893415c19d7aad9a716a9c41a576c3bac74499bb3a527388`. Cloud Build `7b2ad0fd-5cff-4b06-a6cf-474c43577d96` built context `dist/cloudrun/sequence-eyes-20260915`, whose build ID is `a9050b10eaf98530`.

The deployed [v4 experiment](../../reports/sequential-training-20260915/v4/config.json) has config hash `284e053d1a7b14e6a6fbe8d1ac08302350f90f55391c478feae61addef3bb446` and model fingerprint `7ec004c7f6f7e8021ccb329a2e85fe618505800dd169b0f177048ce7aa2e8b53`. Its 696 values are 24 sensory calibration parameters followed by the existing 672 motor-decoder coefficients. The sequence runs leg, antenna, haltere, wing-strain and visual calibration, then teacher demonstration fitting and autonomous recovery, takeoff and landing. Only the active family's coefficients change. Teacher demonstrations do not count as autonomous success; fitted candidates require separate autonomous comparisons. Accepted progress and phase completion have distinct gates, and exhausting a phase's budget stops for review. See [the sequence contract](../../docs/training-sequence.md).

**Eyes** displays the actual left/right retinal images already consumed by the sensory pipeline, labeled with their capture time. Observation is off by default, copies at most twice per second, and stops when the view is hidden or disabled. It performs no additional camera renders, neural steps or physics steps. Images travel only from the local worker to the page; they are absent from result uploads and Firestore. The [Brain view](../../docs/training-brain-view.md), current-trial parameter table and native 3D preview remain available.

Release verification passed all 15 public transport checks. Local Safari verified actual Eyes images and Pause/Resume/Stop. Public Safari completed a full five-second scored trial, which Firestore accepted at 2026-09-15 07:28 UTC with all 696 applied values matching the assignment and the expected WASM provenance. Safari displayed one uploaded result and began trial two automatically. The first result scored 0.0588 without task success; the checkpoint remained unchanged pending candidate comparisons. See the [release evidence](../../reports/flyheaven-sequence-eyes-20260915/README.md).

| Setting | Value |
| --- | --- |
| Project | `flyheaven` (`753887928769`) |
| Region / service | `us-central1` / `fly-training` |
| Runtime | Request-based billing, 1 vCPU, 512 MiB, concurrency 8 |
| Capacity | Minimum 0, maximum 2 instances |
| Runtime identity | `fly-training-runtime@flyheaven.iam.gserviceaccount.com` |
| Runtime project role | `roles/datastore.user` only |
| Firestore | Native mode, Standard edition, `(default)` database in `us-central1`, deletion protection enabled |
| Current sequence namespace | `training_sequences/banc888-sensorimotor-wasm-20260915` |
| Previous browser decoder history, retained | `training_runs/banc888-motor-decoder-wasm-20260914` — 104 accepted results, generation 10 at cohort switch |
| Native decoder history, retained | `training_runs/banc888-motor-decoder-v1-20260914` — 17 accepted results, generation 1 |
| Earlier 14-parameter history, retained | `training_runs/banc888-v1-20260913` |
| Image repository | `us-central1-docker.pkg.dev/flyheaven/fly-training` |

The service APIs, database, runtime identity and image repository are provisioned. The unused VM infrastructure in `flyheaven` was removed; the [VM deployment](../gcp/README.md) is an alternative, not this domain's target.

## Build a matching image

Run from the repository root with the BANC/body prerequisites and pinned Python dependencies described in [reproducibility](../../docs/reproducibility.md). The sequence builder also requires its frozen structural bundle and native teacher calibration inputs. Generated bundles and large prepared data are local artifacts. To create a different experiment, choose an unused output directory:

```sh
node scripts/prepare-sequential-training.mjs \
  --output=reports/sequential-training-NEW
```

For the deployed experiment, reuse the exact v4 bundle rather than regenerating its configuration. Package it with the current presentation files, then create a fresh Cloud Run build context. `BUILD` is a new directory/tag for each release:

```sh
node scripts/package-training-client.mjs \
  --experiment-bundle=reports/sequential-training-20260915/v4/sequence.bundle.json \
  --output-dir=dist/training-sequence-client-BUILD
.venv/bin/python deploy/cloudrun/package.py \
  --bundle dist/training-sequence-client-BUILD/fruit-fly-training-client-284e053d1a7b \
  --config reports/sequential-training-20260915/v4/config.json \
  --output dist/cloudrun/BUILD
```

The deployed client archive is `dist/training-sequence-eyes-client/fruit-fly-training-client-284e053d1a7b.zip`. Packaging verified 111 archive entries and 110 public HTTP files. The experiment verifier accepts only its declared model/source/catalog/teacher overrides and checks every scientific asset against the pinned hashes. Presentation files, including Eyes, are recorded separately in the package manifest. A new scientific configuration requires a fresh namespace; updating only presentation can retain the current config and history.

The extracted client joins the same GCP pool. Serve it on localhost without starting a local coordinator:

```sh
cd dist/training-sequence-client-BUILD/fruit-fly-training-client-284e053d1a7b
python3 serve.py --port 7842
```

Open `http://127.0.0.1:7842/train.html` and press **Start**. The client verifies its model/config identity against `https://flytrain.morisoba.moe`; a mismatch or unavailable coordinator prevents training. **Download checkpoint** fetches the selected server vector, independently of any running assignment. Old tabs from the previous cohort must reload.

Back in the repository root, submit the prepared image build:

```sh
gcloud builds submit dist/cloudrun/BUILD \
  --config dist/cloudrun/BUILD/cloudbuild.yaml \
  --substitutions _IMAGE=us-central1-docker.pkg.dev/flyheaven/fly-training/web:BUILD \
  --service-account projects/flyheaven/serviceAccounts/fly-training-builder@flyheaven.iam.gserviceaccount.com \
  --project=flyheaven
```

The dedicated build identity `fly-training-builder@flyheaven.iam.gserviceaccount.com` has `roles/logging.logWriter` on the project, `roles/artifactregistry.writer` on the `fly-training` repository, and `roles/storage.objectViewer` on the `flyheaven_cloudbuild` source bucket. Build logs go to Cloud Logging. See [Google's user-specified build identity instructions](https://docs.cloud.google.com/build/docs/securing-builds/configure-user-specified-service-accounts).

[`package.py`](package.py) verifies the contributor manifest and exact model/config hashes, copies an allowlist of public files, and creates MIME/hash manifests with deterministic gzip variants. Server code, dependencies and configuration stay separate from public files. Local SQLite databases, credentials and repository metadata are excluded. `build-manifest.json` records the packaged sources and hashes.

Record the immutable digest from the successful build, or inspect the tagged image:

```sh
gcloud artifacts docker images describe \
  us-central1-docker.pkg.dev/flyheaven/fly-training/web:BUILD \
  --project=flyheaven --format='value(image_summary.fully_qualified_digest)'
```

## Preserve or initialize training history

The current coordinator is [`firestore_sequential.py`](../../scripts/firestore_sequential.py). It stores metadata/state plus `jobs`, `checkpoints` and `contributors` under `training_sequences/RUN_ID`. Each transaction reads the current bounded batch and commits its jobs, state, counters and phase checkpoint together. Its in-memory SQLite interpreter is disposable; Firestore remains the sole durable authority. A different config/model is rejected for an existing namespace. The server selects this coordinator from `trainingSequence` and opens it with `initialize=False`.

The production namespace is already initialized. These are the operator commands used to validate a matching configuration locally and deliberately initialize a sequence using application-default credentials:

```sh
.venv/bin/python scripts/firestore_sequential.py \
  --config reports/sequential-training-20260915/v4/config.json \
  --run-id banc888-sensorimotor-wasm-20260915 --project=flyheaven \
  --lease-timeout-seconds=180 --dry-run
.venv/bin/python scripts/firestore_sequential.py \
  --config reports/sequential-training-20260915/v4/config.json \
  --run-id banc888-sensorimotor-wasm-20260915 --project=flyheaven \
  --lease-timeout-seconds=180 --initialize
```

`--dry-run` creates no Firestore client. `--initialize` requires an explicit project; repeating it with the exact existing identity verifies that namespace rather than resetting progress. New experimental configurations need a new run ID. Startup never creates or repairs missing history implicitly.

The old 672-parameter WASM run remains under `training_runs/banc888-motor-decoder-wasm-20260914`, with 104 accepted results and generation 10 at the cohort switch. Its [checkpoint was archived before release](../../reports/sequential-training-20260915/cloudrun-old-checkpoint-before-release.json). The native 17-result/generation-1 run and earlier 14-parameter run also remain separate. No old scores, leases or phase progress were imported into the new sequence. WASM and Dawn evaluations must not share an execution cohort.

For a **legacy `training_runs` migration only**, [`firestore_coordinator.py`](../../scripts/firestore_coordinator.py) still supports stopped SQLite histories. Stop source writers and make an online backup using the [historical VM procedure](../gcp/README.md#verify-and-maintain), then validate locally:

```sh
.venv/bin/python scripts/firestore_coordinator.py \
  data/training/source-snapshot.sqlite3 \
  --config PATH_TO_MATCHING_CONFIG.json --project=flyheaven \
  --run-id NEW_MATCHING_LEGACY_RUN_ID --dry-run
```

Repeating without `--dry-run` imports that validated legacy snapshot. It checks historical hashes/transitions, preserves its records, resumes the same source digest after interruption, and refuses to overwrite a different active run. This importer is not the initializer for the sequence namespace. Keep private backups and verify counts/checkpoint identity before retiring any source.

## Deploy the image

After history is ready, deploy using the full image digest rather than a mutable tag. Replace `DIGEST` with the 64-character SHA256 returned by the build:

```sh
uv run --offline python deploy/cloudrun/deploy.py \
  --image us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:DIGEST \
  --run-id banc888-sensorimotor-wasm-20260915 --project=flyheaven
```

[`deploy.py`](deploy.py) updates the existing service with the capacity settings above, CPU throttling and startup boost, an HTTP `/healthz` startup probe, HTTP/1 to the container, and the dedicated runtime identity. It configures the domain and the returned `run.app` URL as exact allowed public origins. The service also supports the four documented localhost contributor origins. Bootstrap a missing service separately before using this guarded update command.

The deployer sets `TRAINING_LEASE_TIMEOUT_SECONDS=180` independently of the pinned scientific config. Each lease lasts at most three minutes from the most recent heartbeat; active browsers renew once a minute. The sequence's separate scheduling policy also imposes an absolute attempt deadline that renewal cannot extend: 20 wall minutes per five assigned simulation seconds, with a ten-minute minimum. Longer tasks scale with their assigned horizon; decoder fitting uses the sum of its three episode horizons. The browser also stops a worker after three active minutes without advancing neural or physics clocks. Manual Pause suspends that progress timer and lease renewal, but does not extend the server deadline. Hidden tabs continue running. Updated clients automatically abandon rejected expired attempts and seek new work; existing pages need one reload to receive this client behavior.

Scheduling changes are explicit administration and are never available through the public HTTP API. First deploy a revision that understands the versioned policy and variable search batch sizes. After that exact revision receives 100% traffic, update the existing run atomically:

```sh
.venv/bin/python scripts/firestore_sequential.py \
  --config reports/sequential-training-20260915/v4/config.json \
  --run-id banc888-sensorimotor-wasm-20260915 --project flyheaven \
  --lease-timeout-seconds 180 --set-scheduling \
  --search-pairs 8 --trial-timeout-seconds 1200
```

For a read-only capacity check, replace `--set-scheduling` with `--dry-run`. The live mutation appends missing pairs if the current phase is still searching; otherwise the larger pool starts at the next search round. Accepted results and checkpoint values stay intact. Policy revisions/history are saved with the run and returned in status/checkpoint; the original scientific config hash remains unchanged. Legacy leases receive one recorded grace window, and retrying the same policy does not extend it. After expansion, rollback requires a policy-compatible server: an older revision assumes exactly four search jobs and rejects the expanded batch.

Each update creates a uniquely named revision with no traffic and a temporary tag. The helper checks that revision's public config, coordinator status, effective lease timeout, checkpoint identities and parameter schema, then explicitly assigns 100% traffic to that exact revision and removes the tag. A failed preflight leaves the old traffic split intact. It never selects `LATEST`, including when another deployment runs concurrently, and confirms the returned traffic allocation before reporting success. The result includes the revision, config hash, model fingerprint and effective lease timeout. The public Google frontend `/healthz` route is not used for preflight; the container startup probe performs that readiness check. Custom-domain activation remains a separate step.

The startup probe validates the current coordinator history as well as the static bundle, including sequence phase/checkpoint agreement. An incompatible namespace cannot pass readiness. For retained legacy guarded-search histories, validation preserves exact stored parameters and hashes; seeded Gaussian regeneration alone permits four floating-point representable steps of platform rounding drift. The macOS-to-Linux migration fixture measured at most two such steps. Stored pair noise, parameter reconstruction, job identities, comparison assignments and acceptance decisions remain exact.

## Activate the Google-managed domain

`morisoba.moe` is already verified for the authenticated operator. The mapping below already exists; use `describe` to inspect it. To recreate it in a fresh deployment, run `create` after the service is ready:

```sh
gcloud beta run domain-mappings create \
  --service=fly-training --domain=flytrain.morisoba.moe \
  --region=us-central1 --project=flyheaven
gcloud beta run domain-mappings describe \
  --domain=flytrain.morisoba.moe --region=us-central1 --project=flyheaven \
  --format='yaml(status.resourceRecords,status.conditions)'
```

Apply the exact records returned by Google in the existing Cloudflare DNS zone. The created mapping requires a DNS-only `flytrain` CNAME to `ghs.googlehosted.com.`; no additional ownership TXT was requested. Google marked the certificate provisioned on 2026-09-13 at 22:04:10 UTC. Verify an actual HTTPS response as well as the mapping conditions before treating activation as complete. Do not point this domain at either former VM address. Cloud Run provisions and renews the certificate after DNS validation. See [Google's domain mapping instructions](https://docs.cloud.google.com/run/docs/mapping-custom-domains).

## Verify a release

After domain activation, check the canonical HTTPS service:

```sh
curl --fail https://flytrain.morisoba.moe/api/training/status
curl --fail --output heaven-checkpoint.json \
  https://flytrain.morisoba.moe/api/training/checkpoint
gcloud run services describe fly-training --project=flyheaven --region=us-central1
TRAINING_CONFIG=reports/sequential-training-20260915/v4/config.json \
  TRAINING_URL=https://flytrain.morisoba.moe \
  node scripts/verify-cloudrun-host.mjs
```

Use `/api/training/status` for external health verification. The internal startup probe uses `/healthz`; the Google frontend historically returned 404 for that exact public path while application routes worked. Compare config/model hashes, all 696 checkpoint names/values, the sequence phase/role, generation and accepted counts. Transport verification checks isolation, CORS, fresh checkpoints and oversized requests without starting a simulation. It defaults to the separate canonical config unless `TRAINING_CONFIG` is explicit.

In a real browser, verify Start, actual 3D/Brain/Eyes views, Pause/Resume/Stop and fresh checkpoint download. Complete an assigned episode and confirm durable acceptance with matching applied parameters and WASM provenance separately. Eyes should show the last consumed retinal capture time, not a newly rendered observer camera. Changing tabs disables image observation but does not stop training; manual Pause/Stop remain explicit.

At this release, 15 public transport checks and local Safari Eyes/Pause/Resume/Stop checks passed. The first public sequence trial completed and was accepted; its read-only Firestore audit verified assignment equality, all 696 bounded values, the phase mask, neural WASM pin, native body backend and full scored horizon. No checkpoint improvement had yet been accepted. The prior cohort's accepted trials below are historical evidence only. Accepted contributions establish recorded browser-reported work, not remote attestation, successful flight or measured biological calibration. Accepted request bodies remain capped at 262144 bytes; the local fit-upload fixture used 95644 bytes, while full image and regression matrices remain in the browser.

## Checkpoints and private full-history backups

The public checkpoint endpoint downloads the current parameter vector and model/config identity. It does not export the BANC graph or private contribution history. A Cloud Run filesystem is not the durable database; Firestore is the source of truth.

For full history, use Google's managed Firestore export with an operator-owned private Cloud Storage bucket in `us-central1`. The bucket is not created by these deployment tools. After configuring the bucket and operator permissions, choose a fresh export prefix:

```sh
gcloud firestore export gs://YOUR_PRIVATE_BACKUP_BUCKET/training/YYYYMMDD-HHMMSS \
  --database='(default)' --project=flyheaven
gcloud storage cp --recursive \
  gs://YOUR_PRIVATE_BACKUP_BUCKET/training/YYYYMMDD-HHMMSS \
  ./backups/ --project=flyheaven
```

This exports the entire database, including legacy generations/jobs/contributors and sequence jobs/checkpoints/contributors. Files include contributor identifiers and lease tokens, so keep them private and outside Git. A live export can include changes made while it runs; quiesce writers if a stable training snapshot is required. Wait for completion before downloading or restoring. Exports incur document reads and bucket storage charges. The format is a managed database export, not checkpoint JSON. Consult [Firestore export/import permissions and restore instructions](https://docs.cloud.google.com/firestore/native/docs/manage-data/export-import); do not restore over an active run without a deliberate migration plan.

## Historical release evidence

The following records describe the previous `training_runs` cohort. Their revision traffic, trial counts and pending checks are snapshots from those releases, not the current sequence deployment.

The previous browser cohort was [`reports/motor-decoder-v1/browser-wasm-001`](../../reports/motor-decoder-v1/browser-wasm-001/README.md), with config hash `1f3b0935af592a3edf283f02ae2d7eb570fa5c2a6acd225b429b6624d6ed328f`. The lease-recovery revision was `fly-training-r53af464ae124`, image digest `sha256:339f8528fc3157acdf5777e2867b0f0d6f512cbef65bacf4c6886a8ea51a9cc4`, which served 100% of traffic at that release. Cloud Build `1e09cd82-a214-4701-8b43-29aae9838eb8` built release `7f6cc712e832f7b0` from `dist/cloudrun/lease-recovery-20260914` on 2026-09-15 UTC (September 14 in New York).

The lease-recovery release caps abandoned assignments at 180 seconds since their last renewal, with browser heartbeats every 60 seconds. The client terminates and releases a worker after three active minutes without advancing simulation clocks; a healthy trial can renew for any duration. This addresses the [abandoned final search job](../../reports/flyheaven-lease-recovery-20260914/diagnosis.json) that held the batch for 30 minutes. All 102 focused client, server and real Firestore emulator tests, the worker browser fixture, and 15 public transport checks passed. Public assets differ from the preceding release only in `training/client.js`; the scientific config, all 59 pinned assets and Firestore namespace remain unchanged. Post-deployment status showed 35 accepted evaluations, three leased jobs and two pending jobs. See the [verification record](../../reports/flyheaven-lease-recovery-20260914/verification.json). Existing pages receive the client watchdog on reload.

The preceding Brain release (`fly-training-racb4e5e40435`, Cloud Build `a8380fef-7901-4a22-af0a-7802cceef278`) added the [Brain view](../../docs/training-brain-view.md) and current-trial parameter table. It samples 4,096 measured BANC neuron anchors at most twice per second, without additional neural steps. Parameters show only the assigned values used by the current fly; saved checkpoint values do not appear in this table. The config, model fingerprint, neural WASM and Firestore namespace are unchanged. [Public asset checks](../../reports/flyheaven-brain-view-20260914/public-assets.json) matched all eight new or updated UI/observer files to the release manifest. Safari displayed Running, a 4,096-neuron activity sample with the neural clock advancing from 0.1 to 0.2 seconds, and populated current values for trial 15 in generation 2. These checks verify live telemetry; they do not establish improved flight or a completed trial on this revision. See the [validation record](../../reports/flyheaven-brain-view-20260914/verification.json) for test scope. A live Safari Stop check was still pending at that historical release checkpoint.

The initial browser release on 2026-09-14 was revision `fly-training-rbe4741deef91`, image digest `sha256:afe0b3ae02363158edb42b9ab6fc35c37cb3d9b3d9c067444f59417e3e1ef651`, which received 100% of service traffic. Public assets and coordinator identities matched, and all 15 transport checks passed. Safari Start, Pause, Resume, the low-resolution 3D preview and a generation-zero download containing 672 parameters were verified. The first actual Safari WASM trial completed and was accepted into Firestore; the [read-only result audit](../../reports/flyheaven-decoder-deployment-20260914/browser-wasm-audit-002.json) passed every check. The generation-zero checkpoint remained unchanged at that time, and stable flight was not achieved.

The subsequent background-training update was revision `fly-training-r7812ed676b60`, image `sha256:f2d21866f252d64c3199d43af284e6dfc0b812884c2d155a3270c93d00b735d9`, promoted to 100% of traffic. It changed only the public client: switching tabs no longer pauses computation or lease renewal. Manual Pause/Stop still work, while hidden previews avoid drawing. The model/config identity and existing Firestore history were preserved; all 15 transport checks passed with three accepted results present after promotion. The [live Safari check](../../reports/flyheaven-tab-background-20260914/safari-background-verification.json) captured advancing simulation frames while the tab was hidden and unpaused.

The status/timeout update was revision `fly-training-r57ef0c0c4717`, image `sha256:e7eee3aca42d61b943d930f6c825d23531868001097b1f0383e675102b492e8d`, promoted to 100% of traffic before later releases. Waiting for an assigned trial now displays **Waiting for work**, and a network timeout reports a connection error instead of silently leaving an inactive loop labeled Running. Only `training/client.js` and `training/view.js` changed in that public bundle; the run, physics, lease policy and model/config identity remained unchanged. All 52 focused client tests, the browser control fixture and 15 transport checks passed, with 13 accepted results and the generation-one checkpoint preserved. The [incident record](../../reports/flyheaven-waiting-fix-20260914/diagnosis.json) documents an idle Safari session blocked by the last reserved job in a generation. It automatically reclaimed that job about two seconds after expiry and advanced trial 9 without a restart. That existing session was left running; already-open pages receive a new client on their next reload.

The subsequent coordinator hardening was revision `fly-training-r2579cd13b393`, image `sha256:4cd7c771b8c3dbb0f71fdc3dff29835daa69ccdad780e667b11cb65c6daf47bb`. It requires the browser's reported applied parameter vector and assignment identity to match the leased candidate. The [result audit](../../reports/flyheaven-decoder-deployment-20260914/browser-wasm-audit-sync-hardened-006.json) passed for 20 accepted results at generation 2. This validation remained in subsequent releases of that cohort; no historical results or checkpoints were reset.

The first accepted trial's Firestore audit at 2026-09-14 19:54:45 UTC verified the exact WASM pin and equality of the applied vector/hash with its assignment. Its candidate changed all 672 coefficients relative to the initial vector (24 power, 648 steering; maximum absolute change 0.0064765787001395125, L2 change 0.02735765139639696). After 0.5 seconds of warmup, it scored 0.222 seconds before `excessive_rotation`, returning −2.9524 with 0.034 seconds of best qualified flight. Execution took 109.7049 seconds; total evaluation wall time was 302.4048 seconds including pauses. The browser automatically began trial two. No candidate-retention comparison had finished: the saved generation-zero checkpoint still changed 0 of 672 coefficients. Accepted contribution means a valid recorded evaluation, not successful flight or an improved checkpoint.

## Local checks

```sh
.venv/bin/python -m unittest discover -s deploy/cloudrun -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests -p 'test_sequential_training.py'
.venv/bin/python -m unittest discover -s tests -p 'test_training_coordinator.py'
```

These cover packaging/HTTP, the sequence protocol and the retained legacy coordinator. Firestore emulator checks and actual browser execution remain distinct; see [bounded validation](../../docs/reproducibility.md#bounded-validation). Keep infrastructure results separate from evidence that the full behavioral target has been learned.
