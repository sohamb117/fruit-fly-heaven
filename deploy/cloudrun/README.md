# Managed training service on Cloud Run

Cloud Run serves the browser bundle and coordinator; contributors' browsers run BANC and the FlyBody/MuJoCo body in WASM. Firestore stores progress independently of container lifetime. The live entry is **[flytrain.morisoba.moe/train.html](https://flytrain.morisoba.moe/train.html)**: **Start** runs one assigned fly locally, **Pause** suspends it, and **Stop** releases unfinished work. Every completed contribution goes to the GCP coordinator. Opening the page alone starts no simulation. The Google service alias is `https://fly-training-yekt6i27nq-uc.a.run.app`.

The live browser cohort is [`reports/motor-decoder-v1/browser-wasm-001`](../../reports/motor-decoder-v1/browser-wasm-001/README.md), with config hash `1f3b0935af592a3edf283f02ae2d7eb570fa5c2a6acd225b429b6624d6ed328f`. On 2026-09-14, revision `fly-training-rbe4741deef91`, image digest `sha256:afe0b3ae02363158edb42b9ab6fc35c37cb3d9b3d9c067444f59417e3e1ef651`, received 100% of service traffic. Public assets and coordinator identities matched, and all 15 transport checks passed. Safari Start, Pause, Resume, the low-resolution 3D preview and a generation-zero download containing 672 parameters were verified. The first actual Safari WASM trial completed and was accepted into Firestore; the [read-only result audit](../../reports/flyheaven-decoder-deployment-20260914/browser-wasm-audit-002.json) passed every check. The generation-zero checkpoint remained unchanged, and stable flight was not achieved.

| Setting | Value |
| --- | --- |
| Project | `flyheaven` (`753887928769`) |
| Region / service | `us-central1` / `fly-training` |
| Runtime | Request-based billing, 1 vCPU, 512 MiB, concurrency 8 |
| Capacity | Minimum 0, maximum 2 instances |
| Runtime identity | `fly-training-runtime@flyheaven.iam.gserviceaccount.com` |
| Runtime project role | `roles/datastore.user` only |
| Firestore | Native mode, Standard edition, `(default)` database in `us-central1`, deletion protection enabled |
| Browser WASM namespace | `training_runs/banc888-motor-decoder-wasm-20260914` |
| Native decoder history, retained | `training_runs/banc888-motor-decoder-v1-20260914` — 17 accepted results, generation 1 |
| Earlier 14-parameter history, retained | `training_runs/banc888-v1-20260913` |
| Image repository | `us-central1-docker.pkg.dev/flyheaven/fly-training` |

The service APIs, database, runtime identity and image repository have been provisioned. The unused VM infrastructure in `flyheaven` was removed. The [VM deployment](../gcp/README.md) remains an alternative for other operators; it is not this domain's target.

## Build a matching image

Run from the repository root after preparing the pinned assets as described in [reproducibility](../../docs/reproducibility.md). Choose a fresh `BUILD` directory/tag for each release; the packager refuses to overwrite an existing output directory.

The browser cohort trains 672 decoder coefficients. Its initial values match the final native checkpoint, but it starts a new generation-zero history with no inherited scores or fitness. WASM and Dawn results must never be combined. The schema-2 guard still compares the best search candidate against the incumbent on three fresh paired seeds before replacing a checkpoint.

To prepare a new browser experiment after the ordinary BANC/body prerequisites, use an unused output directory and select WASM explicitly. This path requires no native Dawn tooling:

```sh
node scripts/prepare-motor-decoder-experiment.mjs \
  reports/motor-decoder-v1/NEW_BROWSER_RUN --backend=wasm
```

Package the exact already-prepared release configuration and physical model overrides. The bundle verifier permits only the two model overrides and requires all executable assets to match the experiment's pinned hashes:

```sh
node scripts/package-training-client.mjs \
  --experiment-bundle=reports/motor-decoder-v1/browser-wasm-001/bundle.json
uv run --offline python deploy/cloudrun/package.py \
  --bundle dist/training-client/fruit-fly-training-client-1f3b0935af59 \
  --config reports/motor-decoder-v1/browser-wasm-001/config.json \
  --output dist/cloudrun/BUILD
```

The resulting local client joins the same GCP pool. Serve the verified extracted bundle on localhost; it does not start a local coordinator:

```sh
cd dist/training-client/fruit-fly-training-client-1f3b0935af59
python3 serve.py --port 7842
```

Open `http://127.0.0.1:7842/train.html` and press **Start**. The client verifies its model/config identity against `https://flytrain.morisoba.moe`; a mismatch or unavailable coordinator prevents training, with no unshared fallback. **Download checkpoint** fetches the currently selected server checkpoint. Separate recorded-frame telemetry never changes scores or checkpoint selection.

Back in the repository root, submit the prepared image build:

```sh
gcloud builds submit dist/cloudrun/BUILD \
  --config dist/cloudrun/BUILD/cloudbuild.yaml \
  --substitutions _IMAGE=us-central1-docker.pkg.dev/flyheaven/fly-training/web:BUILD \
  --service-account projects/flyheaven/serviceAccounts/fly-training-builder@flyheaven.iam.gserviceaccount.com \
  --project=flyheaven
```

The dedicated build identity `fly-training-builder@flyheaven.iam.gserviceaccount.com` has `roles/logging.logWriter` on the project, `roles/artifactregistry.writer` on the `fly-training` repository, and `roles/storage.objectViewer` on the `flyheaven_cloudbuild` source bucket. The build configuration writes logs to Cloud Logging. It does not use the runtime identity or the Compute default service account. See [Google's user-specified build identity instructions](https://docs.cloud.google.com/build/docs/securing-builds/configure-user-specified-service-accounts).

[`package.py`](package.py) verifies the contributor manifest and canonical model/config hashes, copies an allowlist of public files, and creates a MIME/hash manifest with deterministic gzip variants. Server code, dependencies and configuration are separate from public files. Local SQLite databases, credentials and repository metadata are not included in the image. `build-manifest.json` records the packaged sources and their hashes.

Record the immutable digest from the successful build, or inspect the tagged image:

```sh
gcloud artifacts docker images describe \
  us-central1-docker.pkg.dev/flyheaven/fly-training/web:BUILD \
  --project=flyheaven --format='value(image_summary.fully_qualified_digest)'
```

## Preserve or initialize training history

[`firestore_coordinator.py`](../../scripts/firestore_coordinator.py) stores run metadata plus `generations`, `jobs` and `contributors` subcollections; bounded preview geometry is kept separately in `telemetry`. Transactions control assignment ownership, idempotent submissions and generation advancement across Cloud Run instances. A different model/config is rejected for an existing run ID. Updating an image with the same identity preserves progress; a new experiment needs a new run namespace.

Initialize the browser cohort with its exact `browser-wasm-001/config.json` and `run_id="banc888-motor-decoder-wasm-20260914"`, using the operator API `FirestoreCoordinator(..., initialize=True)`. The selected initial vector is configuration, not an imported native training history. Preserve the native decoder's 17-result, generation-1 namespace and the earlier 14-parameter namespace independently; do not migrate either one's scores into WASM.

For migration from the earlier SQLite coordinator, first stop source writers and make an online SQLite backup using the [historical VM backup procedure](../gcp/README.md#verify-and-maintain). Keep the original backup privately. Install the pinned Firestore dependencies and use authenticated operator application-default credentials, then validate the snapshot without contacting Firestore:

```sh
uv run --with-requirements requirements-cloudrun.txt python scripts/firestore_coordinator.py \
  data/training/source-snapshot.sqlite3 \
  --config PATH_TO_MATCHING_CONFIG.json --project=flyheaven \
  --run-id NEW_MATCHING_RUN_ID --dry-run
```

To import that validated snapshot, repeat the command without `--dry-run`. The importer preserves generations, results, contributors and leases, checks historical hashes and optimizer transitions, and verifies imported records before marking the run ready. It resumes the same source digest after interruption and refuses to overwrite a different active run. Source progress accepted after the snapshot is not part of the import. Do not retire the source until the imported counts and checkpoint match.

The serving process expects an initialized, ready namespace; it must not silently replace missing state with a fresh experiment. A deliberate new experiment can use `FirestoreCoordinator(..., initialize=True)` through the internal research API; that is separate from migrating the existing run.

## Deploy the image

After history is ready, deploy using the full image digest rather than a mutable tag. Replace `DIGEST` with the 64-character SHA256 returned by the build:

```sh
uv run --offline python deploy/cloudrun/deploy.py \
  --image us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:DIGEST \
  --run-id banc888-motor-decoder-wasm-20260914 --project=flyheaven
```

[`deploy.py`](deploy.py) updates the existing service with the capacity settings above, CPU throttling and startup boost, an HTTP `/healthz` startup probe, HTTP/1 to the container, and the dedicated runtime identity. It configures the domain and the returned `run.app` URL as exact allowed public origins. The service also supports the four documented localhost contributor origins. Bootstrap a missing service separately before using this guarded update command.

Each update creates a uniquely named revision with no traffic and a temporary tag. The helper checks that revision's public config, coordinator status and checkpoint identities and parameter schema, then explicitly assigns 100% traffic to that exact revision and removes the tag. A failed preflight leaves the old traffic split intact. It never selects `LATEST`, including when another deployment runs concurrently, and confirms the returned traffic allocation before reporting success. The result includes the revision, config hash and model fingerprint. The public Google frontend `/healthz` route is not used for preflight; the container startup probe performs that readiness check. Custom-domain activation remains a separate step.

The startup probe validates the current coordinator history as well as the static bundle, so an incompatible imported job batch cannot pass readiness. Guarded search history preserves exact stored parameters and hashes; seeded Gaussian regeneration alone permits four floating-point representable steps of platform rounding drift. The macOS-to-Linux migration fixture measured at most two such steps. Stored pair noise, parameter reconstruction, job identities, comparison assignments and acceptance decisions remain exact.

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

Before DNS is ready, use the `run.app` URL returned by `deploy.py`. After activation, verify the canonical domain with certificate verification enabled:

```sh
curl --fail https://flytrain.morisoba.moe/api/training/status
curl --fail --output heaven-checkpoint.json \
  https://flytrain.morisoba.moe/api/training/checkpoint
gcloud run services describe fly-training --project=flyheaven --region=us-central1
```

Use `/api/training/status` for external health verification. The internal Cloud Run startup probe uses `/healthz`; public requests to that exact path returned a Google-front-end 404 during deployment even while the application routes worked. Compare model/config fingerprints, generation and accepted-result counts with the source snapshot. In a real browser, check the preview, run and submit one genuine episode, and verify the contribution persists across a new Cloud Run instance/revision. Checkpoint download always fetches current server state, including before a visitor starts training. Infrastructure checks do not establish successful fly behavior.

The transport verifier checks public file isolation, CORS, fresh checkpoints and oversized requests without starting a simulation. Set `TRAINING_CONFIG` to the exact experiment configuration; otherwise it defaults to the separate canonical profile in `web/training/config.json`. Actual browser execution is a separate check:

```sh
TRAINING_CONFIG=reports/motor-decoder-v1/browser-wasm-001/config.json \
  TRAINING_URL=https://flytrain.morisoba.moe \
  node scripts/verify-cloudrun-host.mjs
```

For browser WASM release verification, press **Start** in a real browser, observe the actual 3D preview, complete one assigned episode, and confirm the hosted accepted-result count increases with matching WASM provenance. Check **Pause**, **Stop** and fresh checkpoint download as separate controls. The live release passed the 15 transport checks and Safari Start/Pause/Resume, low-resolution 3D preview, fresh checkpoint download and one completed, hosted-accepted browser episode. Persistence of that result across a revision and Stop verification remain pending. The older `verify-gcp-training.mjs` currently asserts WebGPU and must be adapted before it can certify this WASM release. Browser automation prerequisites are described in the reproducibility guide. Rejected oversized uploads are drained for at most 1 MiB or ten seconds before the 413 response; accepted bodies remain capped at 262144 bytes.

The first accepted trial's Firestore audit at 2026-09-14 19:54:45 UTC verified the exact WASM pin and equality of the applied vector/hash with its assignment. Its candidate changed all 672 coefficients relative to the initial vector (24 power, 648 steering; maximum absolute change 0.0064765787001395125, L2 change 0.02735765139639696). After 0.5 seconds of warmup, it scored 0.222 seconds before `excessive_rotation`, returning −2.9524 with 0.034 seconds of best qualified flight. Execution took 109.7049 seconds; total evaluation wall time was 302.4048 seconds including pauses. The browser automatically began trial two. No candidate-retention comparison had finished: the saved generation-zero checkpoint still changed 0 of 672 coefficients. Accepted contribution means a valid recorded evaluation, not successful flight or an improved checkpoint.

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

This exports the entire database, including nested generations/jobs/contributors. Files include contributor identifiers and lease tokens, so keep them private and outside Git. A live export can include changes made while it runs; quiesce writers if a stable training snapshot is required. Wait for completion before downloading or restoring. Exports incur document reads and bucket storage charges. The format is a managed database export, not checkpoint JSON. Consult [Firestore export/import permissions and restore instructions](https://docs.cloud.google.com/firestore/native/docs/manage-data/export-import); do not restore over an active run without a deliberate migration plan.

## Local checks

```sh
uv run --offline python -m unittest discover -s deploy/cloudrun -p 'test_*.py'
uv run --offline python -m unittest discover -s tests -p 'test_training_coordinator.py'
```

The first suite checks packaging and the managed HTTP server; the second checks the shared coordinator protocol and optimizer. Firestore emulator tests and live deployment checks are distinct; see [bounded validation](../../docs/reproducibility.md#bounded-validation). Keep their results separate from evidence that the full behavioral target has been learned.
