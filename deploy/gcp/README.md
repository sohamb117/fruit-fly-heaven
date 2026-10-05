# Historical VM deployment alternative

The selected deployment is now [Cloud Run with Firestore](../cloudrun/README.md) in project `flyheaven`, with intended domain `https://flytrain.morisoba.moe`. The unused VM resources in that project were removed. This guide preserves the earlier Caddy/SQLite implementation for another operator; its example commands do not describe the active target.

One Debian 12 `e2-micro` VM serves the static browser training bundle and the Python coordinator. Contributor devices run all neural and physical simulation; the VM has no GPU. The SQLite database and Caddy certificate storage persist on the VM's standard persistent disk. Root owns read-only application and public files; separate `flytraining` and `caddy` service accounts own their state directories.

## Build the exact deployment

Regenerate the contributor bundle after UI changes. This does not change the model/config identity unless the canonical manifest itself changes.

```sh
node scripts/package-training-client.mjs
uv run --offline python deploy/gcp/package.py \
  --bundle dist/training-client/fruit-fly-training-client-1bad7d5c80d5 \
  --database data/training/coordinator.sqlite3
```

Use `--database` to bootstrap a new server with existing progress. Omit it when updating a server that already has its persistent database, or when starting a new experiment. It uses SQLite's online backup API to include committed WAL transactions and validates the snapshot's config/model identity. The snapshot retains current generations, results and leases. Source writers may continue, but only the point-in-time backup is deployed; stop local training first if no subsequent progress should be omitted. Deployment never replaces a preexisting server database.

The packager copies only files explicitly listed in the verified contributor manifest, the coordinator, the canonical config and the listed deployment helpers. Local `serve.py`, README, repository metadata and secrets are excluded from public files. Required licenses and corresponding runtime source notices are retained. Large compressible public files receive deterministic gzip copies for Caddy to serve without runtime compression. Private bootstrap SQLite state is outside the web root. A deployment manifest, archive SHA256 and consistency checks accompany the archive.

## Install on a dedicated VM

Reserve a **Standard-tier static external IPv4** and create the VM in an eligible Free Tier region such as `us-central1`. Use `e2-micro`, Debian 12 and a 10 GB `pd-standard` boot disk. Open ingress TCP 80/443. Keep SSH behind GCP IAP or an operator-restricted rule; ports 7850 and 2019 must not be public. Attach no service-account identity unless another operation requires one.

Extract the archive on the VM after checking its SHA256. The pinned Caddy release is 2.11.4. Its verified `linux_amd64` archive SHA256 is `527fbf917c39189a1e3b31d34fa955601680b2d5c8055d2a87b8b9588dec7bb9`. This matches the GitHub release asset metadata; the downloaded bytes also match the official release's SHA512 checksum. The installer requires the SHA256 explicitly and refuses a mismatch. [Caddy release and checksums](https://github.com/caddyserver/caddy/releases/tag/v2.11.4).

```sh
sha256sum --check fly-training-gcp-DEPLOYMENT_ID.tar.gz.sha256
tar -xzf fly-training-gcp-DEPLOYMENT_ID.tar.gz
sudo PUBLIC_ORIGIN=https://training-vm.example.com \
  CADDY_VERSION=2.11.4 \
  CADDY_SHA256=527fbf917c39189a1e3b31d34fa955601680b2d5c8055d2a87b8b9588dec7bb9 \
  bash fly-training/install.sh
```

For this optional VM deployment, replace `training-vm.example.com` with an operator-owned hostname and point its DNS at that VM's reserved address before starting Caddy. Do not use this VM setup for `flytrain.morisoba.moe`; that domain targets Google-managed Cloud Run.

Caddy obtains and renews a public certificate for the domain automatically and redirects HTTP to HTTPS. Preserve `/var/lib/caddy` and keep ports 80/443 reachable. The domain configuration uses [Caddy automatic HTTPS](https://caddyserver.com/docs/automatic-https); the former public-IP certificate configuration is retained only in historical deployment reports. `PUBLIC_ORIGIN` must have no trailing slash.

Both page and coordinator use that one HTTPS origin. The page always shares: it checks coordinator metadata on load, and Start runs only assigned shared jobs. A failed coordinator connection never starts an unshared run. The supplied local client targets `https://flytrain.morisoba.moe`; an operator using this alternative VM pool must update that local-client default in its matching build. CORS explicitly permits the public origin and the four localhost origins on ports 7842/7843. The proxy exposes exactly six training endpoints, caps request bodies at 262144 bytes and bounds upstream connections/timeouts. Caddy's admin endpoint and Python port bind only to loopback. No public administrative API exists. Anonymous results remain unverified; these controls do not attest contributor execution.

## Verify and maintain

```sh
curl --fail https://training-vm.example.com/api/training/status
curl --fail https://training-vm.example.com/api/training/checkpoint
sudo systemctl status fly-training caddy --no-pager
sudo journalctl -u fly-training -u caddy -n 50 --no-pager
```

Confirm both model and config hashes before inviting contributors. Confirm a browser can load the model, run an episode, submit a result and resume after a restart. Validate the low-resolution preview visually; HTTP success alone is not behavioral evidence.

The coordinator database is `/var/lib/fly-training/coordinator.sqlite3` on the VM's retained persistent disk. `generations.center` stores each generation's trainable parameter vector; `jobs` stores contribution history. **Download checkpoint** in the page, or [the direct checkpoint URL](https://training-vm.example.com/api/training/checkpoint), downloads the latest generation as JSON with model/config fingerprints. Each download fetches current server state. Frozen model assets are referenced, not duplicated in this small parameter checkpoint.

```sh
curl --fail --output heaven-checkpoint.json https://training-vm.example.com/api/training/checkpoint
```

To back up the full history without stopping training:

```sh
sudo -u flytraining python3 /opt/fly-training/current/tools/backup.py \
  /var/lib/fly-training/coordinator.sqlite3 \
  /var/lib/fly-training/backup-YYYYMMDD.sqlite3
```

Download backups and keep a separate copy; a backup on the same boot disk does not protect against deleting the disk. Do not copy only a live main SQLite file. Ordinary service restarts preserve all progress. Keep the same static address when restarting the VM so users retain the same origin/local checkpoints. Stopping the VM does not stop persistent disk or reserved IPv4 charges. Delete the VM, reserved address and disk only when deliberately retiring the service.

New releases are stored under `/opt/fly-training/releases` and `/srv/fly-training/releases`; `current` links select the installed release. Reinstalling a matching config preserves the live database. A different model/config fails before services are stopped; changing a shared experiment requires an explicit migration/new database decision.
