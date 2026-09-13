# Managed training service on Cloud Run

Cloud Run is the selected hosting target. It serves the browser bundle and coordinator; contributors' computers run BANC and native FlyBody. Firestore stores shared progress independently of container lifetime. The public domain is **`https://flytrain.morisoba.moe`**. Its Google domain mapping was created on 2026-09-13; DNS and certificate verification are still pending. Until activation, the ready service is available at **`https://fly-training-yekt6i27nq-uc.a.run.app`**. Do not treat the custom domain as live until the checks below pass.

| Setting | Value |
| --- | --- |
| Project | `flyheaven` (`753887928769`) |
| Region / service | `us-central1` / `fly-training` |
| Runtime | Request-based billing, 1 vCPU, 512 MiB, concurrency 8 |
| Capacity | Minimum 0, maximum 2 instances |
| Runtime identity | `fly-training-runtime@flyheaven.iam.gserviceaccount.com` |
| Runtime project role | `roles/datastore.user` only |
| Firestore | Native mode, Standard edition, `(default)` database in `us-central1`, deletion protection enabled |
| Training namespace | `training_runs/banc888-v1-20260913` |
| Image repository | `us-central1-docker.pkg.dev/flyheaven/fly-training` |

The service APIs, database, runtime identity and image repository have been provisioned. The unused VM infrastructure in `flyheaven` was removed. The [VM deployment](../gcp/README.md) remains an alternative for other operators; it is not this domain's target.

## Build a matching image

Run from the repository root after preparing the pinned assets as described in [reproducibility](../../docs/reproducibility.md). Choose a fresh `BUILD` directory/tag for each release; the packager refuses to overwrite an existing output directory.

```sh
node scripts/package-training-client.mjs
uv run --offline python deploy/cloudrun/package.py \
  --bundle dist/training-client/fruit-fly-training-client-1bad7d5c80d5 \
  --output dist/cloudrun/BUILD

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

[`firestore_coordinator.py`](../../scripts/firestore_coordinator.py) stores run metadata plus `generations`, `jobs` and `contributors` subcollections. Transactions control assignment ownership, idempotent submissions and generation advancement across Cloud Run instances. A different model/config is rejected for an existing run ID. Updating an image with the same identity preserves progress; a new experiment needs a new run namespace.

For migration from the earlier SQLite coordinator, first stop source writers and make an online SQLite backup using the [historical VM backup procedure](../gcp/README.md#verify-and-maintain). Keep the original backup privately. Install the pinned Firestore dependencies and use authenticated operator application-default credentials, then validate the snapshot without contacting Firestore:

```sh
uv run --with-requirements requirements-cloudrun.txt python scripts/firestore_coordinator.py \
  data/training/source-snapshot.sqlite3 \
  --config web/training/config.json --project=flyheaven \
  --run-id banc888-v1-20260913 --dry-run
```

To import that validated snapshot, repeat the command without `--dry-run`. The importer preserves generations, results, contributors and leases, checks historical hashes and optimizer transitions, and verifies imported records before marking the run ready. It resumes the same source digest after interruption and refuses to overwrite a different active run. Source progress accepted after the snapshot is not part of the import. Do not retire the source until the imported counts and checkpoint match.

The serving process expects an initialized, ready namespace; it must not silently replace missing state with a fresh experiment. A deliberate new experiment can use `FirestoreCoordinator(..., initialize=True)` through the internal research API; that is separate from migrating the existing run.

## Deploy the image

After history is ready, deploy using the full image digest rather than a mutable tag. Replace `DIGEST` with the 64-character SHA256 returned by the build:

```sh
uv run --offline python deploy/cloudrun/deploy.py \
  --image us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:DIGEST \
  --run-id banc888-v1-20260913 --project=flyheaven
```

[`deploy.py`](deploy.py) applies the capacity settings above, CPU throttling and startup boost, an HTTP `/healthz` startup probe, HTTP/1 to the container, and the dedicated runtime identity. It configures the domain and the returned `run.app` URL as exact allowed public origins. The service also supports the four documented localhost contributor origins. The command returns the actual service URL; successful deployment is separate from activating the custom domain.

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

The transport verifier checks public file isolation, CORS, fresh checkpoints and oversized requests without starting a simulation. The browser verifier separately runs and submits one real trial:

```sh
TRAINING_URL=https://fly-training-yekt6i27nq-uc.a.run.app \
  node scripts/verify-cloudrun-host.mjs
TRAINING_URL=https://fly-training-yekt6i27nq-uc.a.run.app/train.html \
  node scripts/verify-gcp-training.mjs
```

Use the canonical domain after its certificate is ready. The browser verifier needs Playwright and Chrome as described in the reproducibility guide. Rejected oversized uploads are drained for at most 1 MiB or ten seconds before the 413 response; accepted bodies remain capped at 262144 bytes.

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
