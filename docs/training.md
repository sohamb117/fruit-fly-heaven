# Training and shared contributions

The shared training console evaluates one real BANC v888 network and native FlyBody at a time. Its route remains **brain → VNC → motor neurons → muscles → body**. Every Start joins the shared GCP pool; there is no unshared mode or sharing switch. Opening the page checks coordinator metadata but starts no compute. The original observation console remains in the repository at `/`.

The live host is **Cloud Run with Firestore** in project `flyheaven`, at **[flytrain.morisoba.moe/train.html](https://flytrain.morisoba.moe/train.html)**. DNS, Google-managed HTTPS, browser connection and checkpoint downloads passed verification on 2026-09-13. See the [managed deployment guide](../deploy/cloudrun/README.md).

This is an implemented parameter-search system, not a completed behavioral controller. Autonomous food localization → approach → landing → probing/feeding → takeoff/flight has not been established. Shared checkpoints are explicitly **unverified**.

## Run locally

Run these commands from the repository root. If the prepared data and runtime assets are already present, no model rebuild is needed.

```sh
uv run --offline python scripts/serve.py --port 7842
```

Open `http://127.0.0.1:7842/train.html`. Port `7843` also works if `7842` is occupied; use the same port in the browser URL. This server binds to the local machine only.

1. The page automatically checks the GCP coordinator at `https://flytrain.morisoba.moe`. Set **Intensity** and preview quality, then press **Start**. The first start loads and checks assets and calibrates the sensory interface. A failed connection never falls back to local unshared work.
2. The neural model requests WebGPU and falls back to the same BANC model in WASM if initialization is unavailable. Native body dynamics use MuJoCo WASM. Runtime details remain in diagnostics rather than the public interface.
3. Use **Pause**, **Resume**, or **Stop** to control computation. Hiding the tab pauses training. Intensity describes scheduling duty cycle, not a measured percentage of your computer's total CPU/GPU capacity. Turning the preview off does not reduce neural or physical simulation fidelity.
4. **Download checkpoint** fetches the latest saved candidate directly from GCP, including its parameter vector, generation and model/config identity. It works before training starts and never substitutes a local cached checkpoint.
5. Local storage retains counters and unsent completed results for retry. Accepted results and shared checkpoints are saved on GCP. The coordinator assigns the task stage; the contributor cannot select an unshared task.

The shared coordinator combines eight paired exploration episodes into each generation. Shared results do not automatically promote the curriculum; independent scientific validation is required. The internal research client retains separate local search/test APIs, but the contributor UI disables those paths. The current stages are posture (1 simulated second), approach (3), feeding (3), flight (2), landing (3), and the complete sequence (8). Episode duration is simulated time, so a run can take substantially longer in wall-clock time.

## Start an isolated development coordinator

In a second terminal, with the same finalized build:

```sh
uv run --offline python scripts/training_coordinator.py \
  --config web/training/config.json \
  --database data/training/coordinator.sqlite3 \
  --port 7850
```

This development coordinator defaults to `127.0.0.1:7850`. It schedules work and aggregates returns; contributors' browsers execute the neural and physical episodes. Its SQLite database persists leases, results, generation progress, and candidate checkpoints across restarts. Restart with the same command and database to resume. A database made with another config hash or model fingerprint is rejected; use a separate database for a different build. The managed deployment uses the same protocol with transactional Firestore storage instead of this local SQLite file.

Check the service without allocating a job:

```sh
curl --fail http://127.0.0.1:7850/api/training/status
curl --fail http://127.0.0.1:7850/api/training/checkpoint
```

This localhost coordinator command is for isolated development/API tests. The contributor UI has a fixed coordinator: its own origin when hosted on public HTTPS, or `https://flytrain.morisoba.moe` when served locally. It checks config/model identity automatically, and **Start** leases and executes work. A separate development pool needs an explicitly configured client build; the public UI provides no coordinator switch.

Each shared generation contains four positive/negative pairs, or eight jobs. Both signs of a pair use the same episode seed. Multiple contributors receive distinct leased jobs, and each contributor has at most one active lease. The coordinator specifies the stage and duration; the page cannot override them. The default shared stage is posture, and shared reports never automatically promote the curriculum.

The UI distinguishes episodes completed on this computer from results accepted by the coordinator. Exact retries of an accepted result are idempotent. Expired, reassigned, incompatible, or altered results are rejected and are not counted as contributions. Completed but unsent results are retained locally for retry against their original coordinator. There is no unshared fallback when submission fails.

Leases last 30 minutes and renew every 60 seconds while a shared episode is running and unpaused. Paused or hidden tabs stop renewing. A pause longer than the lease can invalidate the result; resuming computation does not resurrect an expired lease. **Stop** releases incomplete work. A completed result awaiting upload keeps its original lease until normal expiry so a later Start can retry submission; temporary upload failure does not proactively invalidate that result. Disconnected leases still expire on the server.

## Download checkpoints and full history

The managed service stores progress in project `flyheaven`, Firestore `(default)` in `us-central1`, under `training_runs/banc888-v1-20260913`. Run metadata and the `generations`, `jobs` and `contributors` subcollections preserve candidates, assignments and accepted results across Cloud Run restarts, scale-to-zero and new revisions. The container filesystem is not the database. The [storage and migration guide](../deploy/cloudrun/README.md#preserve-or-initialize-training-history) describes importing earlier SQLite progress without resetting the experiment.

Use **Download checkpoint** or the direct JSON download:

```sh
curl --fail --output heaven-checkpoint.json https://flytrain.morisoba.moe/api/training/checkpoint
```

The JSON contains the latest generation's 14 trainable parameters, generation, and frozen model/config fingerprints. It is not a copy of the entire BANC graph or native body assets. Downloads are uncached, and shared candidates remain unverified.

For complete history, use the [private Firestore export procedure](../deploy/cloudrun/README.md#checkpoints-and-private-full-history-backups). It requires an operator-owned private Cloud Storage bucket and exports all nested records. The public endpoint exposes only the current parameter checkpoint. Full exports include contributor identifiers and lease tokens and remain private. Earlier VM/SQLite backup instructions are retained only in the [alternative VM guide](../deploy/gcp/README.md#verify-and-maintain).

## Contribute from another computer

Open `https://flytrain.morisoba.moe/train.html` and press **Start**. The page runs simulations on your own computer and sends their results to the GCP pool. The matching downloaded client joins the same pool automatically through this domain. The deployed `run.app` page also connects to the same stored run.

### Ready-to-copy contributor archive

The generated [contributor ZIP](../dist/training-client/fruit-fly-training-client-1bad7d5c80d5.zip) includes the prepared model and browser client, so recipients do not need to clone the repository, download BANC data, install Node, or compile WASM. Its [SHA256 sidecar](../dist/training-client/fruit-fly-training-client-1bad7d5c80d5.zip.sha256), [file manifest](../dist/training-client/fruit-fly-training-client-1bad7d5c80d5.manifest.json), and [packaging verification](../dist/training-client/fruit-fly-training-client-1bad7d5c80d5.verification.json) are beside it under `dist/training-client/`. These are local generated artifacts, not public download URLs; an operator can distribute the ZIP and checksum together.

Extract the ZIP completely, open a terminal in its folder, and run:

```sh
python3 serve.py --port 7842
# Windows with the Python launcher: py -3 serve.py --port 7842
# With uv installed: uv run --offline python serve.py --port 7842
```

Python 3.9 or newer is sufficient; the server uses only the standard library. It verifies every bundled file before binding to localhost. Open `http://127.0.0.1:7842/train.html` and press **Start**. It automatically uses the GCP pool. Do not open the HTML through `file://`. This archive contains the shared training client and labels its navigation accordingly. The repository's original observation UI remains unchanged.

To build a new matching archive from an already prepared checkout:

```sh
node scripts/package-training-client.mjs
```

The builder needs Node, `uv`/Python for ZIP creation and verification, the prepared runtime/model files, and access to the pinned upstream MuJoCo license text. It copies a strict file allowlist: the canonical training manifest's assets, every prepared graph file referenced by the BANC manifest, the explicit training UI, required presentation assets, and source/license notices. It never copies whole data, dependency, or repository directories. Every model and graph hash must match; it also rejects sources changed during the build. Output names use the first 12 characters of the config hash. The current archive has 71 entries and is about 59 MB compressed, 247 MB of payload before compression. ZIP CRC/decompressed hashes and HTTP hashes/MIME types/isolation headers are verified without starting the neural model.

A separate [real bundle-worker smoke](../reports/training-bundle-smoke/result.json) loaded this archive's WebGPU BANC network and native MuJoCo body, advanced 200 ms with 419,301 neural spikes, and passed pause, cancellation and stop checks without errors. This verifies the packaged runtime paths and execution; it does not establish learned behavior or completion of a curriculum stage.

The selected [Cloud Run deployment](../deploy/cloudrun/README.md) serves the page and coordinator on one Google-managed HTTPS origin, with durable Firestore state. It uses request-based billing, 1 vCPU, 512 MiB, concurrency 8, and zero to two instances. Contributor simulation still runs in the browser. The browser requires HTTPS for non-loopback coordinator URLs.

For the optional VM/container deployment below, a reverse proxy provides HTTPS and forwards `/api/training/*` to the Python coordinator. Merely binding that development server to `0.0.0.0` does not create a public HTTPS service.

Contributors can run the matching site locally on port `7842` or `7843` and press Start to join the GCP pool. A separately hosted training site uses its own HTTPS origin for its coordinator and must serve the matching assets. A different operator must update the local-client default in its build.

The coordinator's default CORS allowlist is exactly:

```text
http://localhost:7842
http://localhost:7843
http://127.0.0.1:7842
http://127.0.0.1:7843
```

For another site origin, repeat `--allow-origin` as needed. Supplying any origins replaces the defaults, so include the localhost origins too if they should remain usable. Origins contain the scheme, host and port, without a path or trailing slash. For example, an operator with an actual site at `https://training.example.org` could add `--allow-origin https://training.example.org` alongside the desired localhost entries. That example domain is not a deployed service.

CORS controls which browser origins can call the API; it does not authenticate contributors. The coordinator accepts anonymous, untrusted reported returns. Lease tokens protect assignment ownership and retry semantics, but do not prove that a contributor ran the simulation. An internet-facing operator must supply its own access policy and traffic limits. The local implementation caps request size, checks finite bounded numbers and provenance, and rejects stale/forged lease identities; it does not turn remote reports into trusted scientific evidence.

### Alternative self-hosted container deployment

[`deploy/training/compose.yaml`](../deploy/training/compose.yaml) builds the stdlib Python coordinator and runs Caddy as its HTTPS proxy. The coordinator's port `7850` is exposed only inside the Compose network. Only Caddy publishes ports `80` and `443`. The database and Caddy's certificate/configuration storage use separate persistent volumes. The image contains the coordinator and pinned config, not the model assets or training website; contributors still need the matching local site.

On an operator-provided server with Docker Compose, copy this matching checkout and configure a real domain:

```sh
cp deploy/training/.env.example deploy/training/.env
# Edit deploy/training/.env: replace coordinator.example with your real DNS name.
docker compose --env-file deploy/training/.env \
  -f deploy/training/compose.yaml config --quiet
docker compose --env-file deploy/training/.env \
  -f deploy/training/compose.yaml up --build -d
docker compose --env-file deploy/training/.env \
  -f deploy/training/compose.yaml logs --tail 50 coordinator caddy
```

The reserved `coordinator.example` name is a placeholder. For public HTTPS, point the real domain's A/AAAA records at that server and make its ports `80` and `443` reachable. Caddy provisions and renews the certificate using its persistent storage. These requirements follow [Caddy's automatic HTTPS documentation](https://caddyserver.com/docs/automatic-https).

Check `https://YOUR-DOMAIN/api/training/status` and compare both hashes with the contributor build before inviting contributors. A custom local contributor build must set `https://YOUR-DOMAIN` as its fixed coordinator; the supplied local build targets `https://flytrain.morisoba.moe`. Compose waits for the coordinator's HTTP health check before starting Caddy, using its [service health dependency](https://docs.docker.com/compose/how-tos/startup-order/).

`docker compose ... stop` and ordinary `docker compose ... down` preserve the named database volume. `down --volumes` deletes it. To migrate existing local results, stop writers and make a consistent SQLite backup before populating the container volume; copying only a live SQLite main file can omit WAL transactions. A new model/config requires a separate database/volume, as with the non-container server. The supplied deployment retains the four localhost CORS origins; a hosted frontend needs an explicit coordinator command override containing its exact `--allow-origin` entries.

These files package a deployment; creating them does not provision a server, configure DNS, expose this machine, or start a public service.

## Prepare and pin the assets

Fresh clones may lack the ignored BANC data and installed MuJoCo package. The preparation tools require the pinned Python dependencies, Node/npm, and the pinned upstream FlyBody source. Run from the repository root; do not clone over an existing `references/flybody` checkout.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-banc.txt
uv run --python .venv/bin/python python scripts/prepare-banc.py --download

git clone https://github.com/TuragaLab/flybody.git references/flybody
git -C references/flybody checkout d015e9bfe441bd90ae431bac24c55cb74bdbce26
npm ci --prefix packages/flybody-runtime
uv run --python .venv/bin/python python scripts/prepare-flybody-runtime.py
uv run --python .venv/bin/python python scripts/prepare-banc-console.py

node scripts/prepare-training-manifest.mjs
```

The BANC downloader checks its source lock. The training loader checks the prepared graph buffers against the pinned BANC manifest, checks the sensory supplement's graph identities, and verifies the training asset hashes. Anatomy preparation is optional for separate anatomy views and is not needed for this training preview. Checked-in BANC `dist/core.js` and `dist/core.wasm` avoid needing a compiler for ordinary use. If changing native BANC runtime code, rebuild it with the repository's Emscripten setup before rebuilding the training manifest:

```sh
source references/emsdk/emsdk_env.sh
bash packages/banc-runtime/build.sh
node scripts/prepare-training-manifest.mjs
```

[`prepare-training-manifest.mjs`](../scripts/prepare-training-manifest.mjs) pins the executable environment, source dependencies, native model, runtime binaries, and model manifests in [`web/training/config.json`](../web/training/config.json). `modelFingerprint` hashes the ordered URL/digest manifest; `configHash` hashes the exact config file bytes. The prepared BANC manifest in turn pins its graph buffers. A pending or inconsistent manifest cannot start the coordinator.

Rebuild this manifest after changes to the training environment, neural/body implementation, shaders, runtime binaries, sensory assets, or physical model. Stop active runs before changing a served build, then distribute matching code/config/assets and use a new Firestore run namespace or development SQLite database. **Do not regenerate a contributor's config merely to bypass a mismatch with someone else's coordinator.** Install the operator's exact matching build instead. Even a whitespace change to the config changes its hash.

## What is being optimized

Version 1 applies 14 bounded parameters in log space, with each multiplier equal to `exp(theta)` and initial `theta = 0`:

| Parameter family | Count | Target |
| --- | ---: | --- |
| Synaptic gain | 2 | Excitatory and inhibitory chemical weights |
| Neural leak scaling | 5 | Wing, leg, probing and grip motor classes; descending neurons |
| Sensory gain | 3 | Taste, odor and body transduction |
| Motor-to-muscle scale | 4 | Wing, leg, probing and grip muscles |

The connectome topology and native physical model remain fixed. These broad learned gains are hypotheses, not cell-specific measurements or calibrated biological physiology. Reward and task goals score episodes; they do not write limb trajectories or body forces. Stage-dependent placement and initial velocity are recorded initial conditions, not a flight controller.

The optimizer uses four antithetic pairs, Gaussian perturbation scale `sigma = 0.25`, learning rate `0.035`, and maximum coordinate update `0.15`. For each coordinate, it adds

```text
clamp(learningRate / (2 * pairCount * sigma)
      * sum((return_plus - return_minus) * noise), -0.15, 0.15)
```

to the baseline log parameter and clips to its configured bounds. Returns must be finite and within `[-10, 10]`. The server deterministically generates jobs and sends explicit parameter arrays and seeds; contributor-side random-number implementations do not choose shared perturbations. Shared aggregation updates the candidate only after all eight jobs are accepted, always with status `unverified`.

`vision: false` is deliberate in this version. The model receives odor, taste and body feedback; the low-resolution 3D preview is an observer view, not retinal input. Neural integration is fixed at `0.5 ms` and the neural/body exchange block at `2 ms`. Budget and preview controls do not change these values. This version does not establish visual food localization or fix the known rigid-wing/contact and unsigned sensory-feedback limitations.

## Validation and limits

The two configured selection seeds are repeatedly used to compare local candidates. They are not an untouched final test. The internal research API `TrainingClient.evaluateCheckpoint()` uses three separate configured test seeds, never used in the local update/selection loop. Repeatedly choosing models based on those test results also compromises their independence; stronger claims require additional predeclared seeds, initial conditions and independent reproductions.

`locally-validated` means the selected stage's implemented criteria passed on selection seeds. `locally-tested` means they passed on the separate local test seeds. Validation and curriculum badges belong to the current parameter vector; changing parameters clears earlier passes instead of carrying success forward to an unevaluated candidate. Evaluating a shared checkpoint adopts it for local evaluation and clears previous local curriculum claims. Neither label establishes biological realism, robust full-sequence control, or generalization to different hardware/backends or body models. Shared result provenance is checked for consistency, not independently attested execution. Export and independently evaluate shared candidates before deciding whether to retain them or create a later-stage experiment.

The coordinator's focused tests exercise HTTP/CORS, leases, heartbeat/expiry, retries, incompatible and nonfinite inputs, concurrency, restart persistence, raw config hashes, and agreement with the browser's ES update/checkpoint format:

```sh
uv run --offline python -W error::ResourceWarning \
  -m unittest discover -s tests -p 'test_training_coordinator.py'
```

These tests use synthetic objectives to validate infrastructure and algebra. Passing them is not evidence that a trained fly completes any behavioral stage.

The earlier VM deployment passed trusted public HTTPS, a real browser WebGPU contribution and native preview, checkpoint persistence after service restart, and private-path rejection; that [historical evidence](../reports/gcp-deployment/README.md) does not validate the managed deployment. The [Cloud Run guide](../deploy/cloudrun/README.md#verify-a-release) lists its separate release checks. Current UI checks are `node web/test/training-hosted-view.browser.mjs` and `node scripts/verify-training-ui-idle.mjs`; `node scripts/verify-gcp-training.mjs` runs one real contribution against a deployed pool. The earlier local UI/evaluation reports record the prior optional-sharing interface; the current contributor page exposes shared jobs only. Internal research APIs still support independent evaluation.
