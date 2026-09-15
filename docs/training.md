# Training and shared contributions

The live [sequential sensorimotor experiment](training-sequence.md) trains 24 structural feedback parameters by family, fits the existing 672-coefficient motor decoder, then trains recovery, takeoff and landing. It runs at **[flytrain.morisoba.moe/train.html](https://flytrain.morisoba.moe/train.html)** on Cloud Run with Firestore in project `flyheaven`. Its configuration hash is `284e053d1a7b14e6a6fbe8d1ac08302350f90f55391c478feae61addef3bb446`; durable state is in `training_sequences/banc888-sensorimotor-wasm-20260915`. See the [release record](../reports/flyheaven-sequence-eyes-20260915/README.md).

The [scheduling update](../reports/flyheaven-scheduling-20260915/README.md) increases search rounds to 16 trials and records that policy separately from the original configuration. Each attempt has an absolute deadline of 20 wall minutes per five assigned simulation seconds, with a ten-minute minimum. Heartbeats cannot extend it; overdue trials can be reassigned, and updated clients automatically request new work. Existing results and checkpoint values remain in the same run.

The live browser entry evaluates one BANC v888 network and FlyBody/MuJoCo body in WASM at a time. Its route remains **brain → VNC → motor neurons → muscles → body**. **Start** executes a coordinator-assigned trial on the visitor's computer; **Pause** suspends computation, and **Stop** releases unfinished work. Every completed contribution goes to the GCP pool; there is no unshared mode or sharing switch. Opening the page checks metadata but starts no simulation. The original observation console remains in the repository at `/`.

**Fly**, **Brain** and **Eyes** inspect the current trial. Eyes shows separate 256 × 128 images from the existing sensory cameras, copied at most twice per second only while selected and visible. It performs no extra camera simulation, and images are never uploaded. The parameter table shows the assigned values actually used by the current fly. See [observation details](training-brain-view.md).

Each phase changes only its declared parameter family. Fresh paired comparisons can accept incremental improvements before task mastery; advancing a phase also requires task success. Later updates retest previous physical tasks. An exhausted budget stops for review and preserves the best accepted values. This is simulation-based engineering calibration, not measured sensory physiology or established behavioral success.

## Previous browser cohort

The previous browser release was [`reports/motor-decoder-v1/browser-wasm-001`](../reports/motor-decoder-v1/browser-wasm-001/README.md), config hash `1f3b0935af592a3edf283f02ae2d7eb570fa5c2a6acd225b429b6624d6ed328f`, for namespace `training_runs/banc888-motor-decoder-wasm-20260914`. Its 104 accepted results and generation-10 checkpoint were retained when the sequence went live. The following evidence describes that earlier cohort; its scores are not imported into the new sequence. On 2026-09-14, revision `fly-training-rbe4741deef91` received 100% of traffic; public and coordinator config identities matched and all 15 transport checks passed. Safari Start/Pause/Resume, the low-resolution 3D preview and a 672-parameter generation-zero checkpoint download were verified. One actual Safari WASM trial completed and was accepted into Firestore, with every check passing in the [read-only result audit](../reports/flyheaven-decoder-deployment-20260914/browser-wasm-audit-002.json). See the [managed deployment guide](../deploy/cloudrun/README.md).

This cohort searches **672 individual wing motor-decoder coefficients** during `maintained_flight`: 24 power weights and 648 steering coefficients. Its initial values match the final native checkpoint, but the WASM run starts at generation 0 with zero inherited results or fitness. The native decoder namespace `banc888-motor-decoder-v1-20260914` is retained with 17 accepted results at generation 1; the earlier 14-parameter namespace `banc888-v1-20260913` is also retained. Different execution backends and config identities never share scores.

The browser run uses the schema-2 acceptance guard. Four search trials (two positive/negative pairs) nominate a candidate; three fresh candidate/incumbent seed pairs then determine whether to retain it. Only a strictly positive mean paired improvement replaces the incumbent. The page uses the pinned BANC WASM binary directly and does not request WebGPU or require native Dawn tooling. GCP schedules work and saves results and checkpoints; it does not simulate the fly.

The first accepted trial, audited at 2026-09-14 19:54:45 UTC, used the exact assigned vector and WASM pin. Its candidate differed from the initial vector in all 672 coefficients (24 power, 648 steering; maximum absolute change 0.00647658, L2 change 0.02735765). It ended with `excessive_rotation` after 0.222 scored seconds following 0.5 seconds of warmup: return −2.9524, best qualified flight 0.034 seconds, and no task success. Execution took 109.7049 seconds; evaluation wall time including pauses was 302.4048 seconds. The browser automatically started trial two. The saved generation-zero checkpoint still differed in **0 of 672 coefficients**, because candidate-retention comparisons had not finished. This verifies browser computation and hosted result storage; it does not show stable flight or a retained improvement.

This is an implemented parameter-search system, not a completed behavioral controller. Autonomous food localization → approach → landing → probing/feeding → takeoff/flight has not been established. Shared checkpoints are explicitly **unverified**.

## Earlier experiments and native diagnostics

The earlier canonical `banc-flybody-flight-interpreter27-v6` profile uses an eight-second takeoff, maintained-airtime and landing episode. It is a separate 27-parameter experiment, not the browser WASM entry above. Criteria version 3 ranks physical failures by measured task progress minus a constant penalty. The archive evidence below describes those earlier builds; their scores and checkpoints cannot be resumed under a different objective or backend.

The earlier development-bundle builder can separately enable individual wing motor events (`--wing-events`), the ten-cell DLM conductance model (`--dlm-ionic`), and native wing-load input to 26 tegula sensory neurons (`--tegula-load`). The current decoder experiment declares its event, DLM and tegula priors explicitly in its own pinned configuration. These preserve the BANC graph and remain model assumptions. See the [DLM implementation and numerical evidence](../reports/flight-dlm-runtime/README.md), [actual DLM flight comparison](../reports/flight-dlm-live/comparison/README.md), and [tegula anatomy and load measurements](../reports/flight-tegula-inputs/README.md). Neither numerical agreement nor a complete signal route establishes stable flight.

For native research runs, `scripts/serve-training-dev.py` serves a prepared development bundle and durable local coordinator. `scripts/contribute-training-native.mjs` executes assigned jobs and uploads them to its selected coordinator. HTTP is limited to loopback; HTTPS requires an explicit backend plan, and a Dawn plan pins its loader and package lock. This is a separate native workflow, not a prerequisite for browser contribution. Native Dawn and browser WASM are not assumed numerically interchangeable. The [v8 guarded experiment](../reports/flight-development-v8/README.md) introduced the candidate-retention guard now used by the browser cohort.

The completed [v6 held-out comparison](../reports/flight-development-v6/held-out-comparison/README.md) separates the optimizer's center from its best training candidate. The center regressed from 130 ms to 111 ms mean qualified flight. The candidate selected using training scores improved all three new seeds to 338/354/352 ms, but every trial still crashed. The [v7 comparison](../reports/flight-development-v7/held-out-comparison/README.md) starts from that exact candidate and uses smaller frequency, deployment and power search steps with the same neural/body environment and criteria version 3. Its final center again regressed, from 336 ms to 223 ms on three fresh seeds. The best training candidate averaged 369 ms, with mixed paired changes of −84/+164/+18 ms. Every held-out episode still crashed; neither run established sustained flight or landing.

The completed [v8 guarded run](../reports/flight-development-v8/README.md) rejected both proposed updates after 28 real evaluations. It preserved all 27 saved coordinates. Six independent evaluations reproduced the retained checkpoint exactly across three new seeds, averaging 392 ms qualified flight with no sustained-flight or landing success. This demonstrates correct rejection and checkpoint retention, not new learning or stable flight. Search was then paused to investigate the steering signal and its physiological input path.

Subsequent [native-step contact traces](../reports/flight-boundary-diagnostic/README.md) reproduce all three retained-checkpoint trajectories exactly. Their main flight bouts lose qualification before any later environmental contact. A [paired boundary-contact ablation](../reports/flight-boundary-ablation/README.md) prevents one seed's original wall-triggered spin and extends survival from734ms to1.530s, but leaves qualified flight unchanged at504ms and ends in a ground crash. This separates the collision failure from the earlier flight-control limitation; disabled boundary collisions are not a training fix.

The opt-in [haltere input A/B/C check](../reports/flight-haltere-live/README.md) adds direct mechanical-current inputs at the existing0.5ms neural timestep. `stepSequence` batches them in one WebGPU submission; constant-input replay matches the prior physics digest and all wing motor events exactly. The mechanical prior changes the trajectory but does not improve the first seed's504ms qualified flight. The left hDVM pathway supplies zero haltere muscle power in that replay. Receptor orientations, current sensitivity and virtual motion remain explicit unmeasured priors; the canonical configuration does not enable this pathway. That diagnostic did not justify resuming training by itself.

The native contributor attaches its latest actual preview frame to lease heartbeats about every five wall-clock seconds and flushes a final frame before submitting a result. This bounded geometry telemetry is separate from scores and checkpoint history; it starts no additional simulation. The monitor labels frames as recorded and may show the same pose until the next native frame arrives. Heartbeats serialize requests and drain before lease release; invalid preview geometry is omitted while ordinary lease renewal continues.

## Maintained-flight task

The browser cohort uses `maintained_flight`. It places the full articulated body at root pose `[0,0,3.5,1,0,0,0]`, runs 0.5 seconds of live BANC, event and muscle dynamics while holding only the root, then releases it for up to five scored seconds. Neural/native clocks and histories continue across release; no root correction is applied afterward. This stage awards no takeoff or landing credit. Earlier `takeoff`, `flight` and `landing` stages retain their grounded starts.

Maintained-flight criteria version 2 measures vertical speed from the COM height change across a fresh, contact-free 50 ms window. The threshold remains −1 cm/s, with a 1e−9 cm/s numerical tolerance; the existing support, attitude, power and 20 ms angular-speed RMS gates remain active. Contact clears the flight bout and its support window. Success requires reaching the full five-second horizon with at least one continuous qualified second immediately beforehand; merely surviving five seconds does not pass.

The browser experiment uses the explicit `spacious-maintained-flight-v1` scene: 50 cm radius, 50 cm ceiling and a floor capped beyond 6.5 cm, preserving central fruit and odor. Config and model metadata declare the same profile. It retains all articulated body joints and zero phenomenological claw adhesion, with normal collisions and friction. The optional structurally reduced body is not used.

### Earlier amplitude experiments

The earlier 27-parameter amplitude experiment opts into `wing_actuation.power_transfer = {schemaVersion:1, profile:"activation-amplitude-v1", activationGain:1.999999638880142}`. Its first learned multiplier becomes an amplitude after fixed activation normalization: `exp(theta[0]) * clamp(rawMuscleForce * activationGain)`, bounded at one. The remaining 26 parameter meanings stay unchanged. This historical interpreter is distinct from the current 672-coefficient decoder, whose normalized power bypasses that transfer.

The [local run declaration](../reports/flight-maintained-amplitude-local-v2/declaration.json) records the run started on 2026-09-14 at `http://127.0.0.1:7862/`, using a pinned Dawn/Metal contributor and the standard coordinator. Its initial amplitude is 0.80, frequency multiplier 1, and other 25 coordinates come from the explicit migration vector. Search uses sigma 0.10 with amplitude/deployment/frequency scales 0.10/0.25/0.05 and steering scales 1. Two guarded generations allow at most 28 accepted evaluations: eight search jobs plus three matched candidate/incumbent pairs per generation. A rejected proposal preserves all 27 saved coordinates. At launch no learned update had been established; checkpoints remain unverified. See the [run recipe](../reports/flight-maintained-training-plan/README.md) for the fixed seeds, source archive and reserved final comparison.

The separate amplitude-0.81 diagnostic survived five scored seconds without contacts and stayed within 23.39° of level. Under the v2 score it had a 2.668-second best qualified bout and 4.25 seconds total qualified flight, but ended descending at −1.746 cm/s and **did not succeed**. It is evidence of a useful mechanical operating region, not trained success or completion of the food-to-feeding flight sequence.

Inspect that archived amplitude run without changing its frozen sources:

```sh
.venv/bin/python scripts/report-flight-training.py reports/flight-maintained-amplitude-local-v2/coordinator.sqlite3 --last-generations 2 --output reports/flight-maintained-amplitude-local-v2/training-progress.json
```

## Run locally

Use the matching extracted contributor archive. For the generated bundle already present in this checkout, run from the repository root:

```sh
cd dist/training-sequence-eyes-client/fruit-fly-training-client-284e053d1a7b
python3 serve.py --port 7842
```

Open `http://127.0.0.1:7842/train.html`. Port `7843` also works if `7842` is occupied; use the same port in the browser URL. This server binds to the local machine only.

1. The page automatically checks the GCP coordinator at `https://flytrain.morisoba.moe`. Set **Intensity** and preview quality, then press **Start**. The first start loads and checks assets and calibrates the sensory interface. A failed connection never falls back to local unshared work.
2. This cohort explicitly runs the pinned BANC neural binary in WASM; body dynamics also use MuJoCo WASM. It does not fall back to or mix in Dawn/WebGPU evaluations. Runtime details stay in diagnostics.
3. Use **Pause**, **Resume**, or **Stop** to control computation. Training continues when you switch tabs. Intensity describes scheduling duty cycle, not a measured percentage of your computer's total CPU/GPU capacity. Turning the preview off does not reduce neural or physical simulation fidelity.
4. **Download checkpoint** fetches the latest saved candidate directly from GCP, including its parameter vector, generation and model/config identity. It works before training starts and never substitutes a local cached checkpoint.
5. Local storage retains counters and unsent completed results for retry. Accepted results and shared checkpoints are saved on GCP. The coordinator assigns the task stage; the contributor cannot select an unshared task.

The sequence assigns four search trials, followed by paired comparisons and fresh acceptance/retention checks. Decoder fitting instead assigns demonstration trajectories before autonomous evaluation. Task horizons are three to eight simulated seconds; wall-clock execution can be much longer. Food localization, approach and feeding remain outside this sequence.

## Earlier standalone coordinator recipe

For the current sequence, use the [sequential development recipe](training-sequence.md#build-and-run). The command below is retained for the previous 672-parameter cohort and cannot serve the new 696-parameter configuration.

In a second terminal, with the same finalized build:

```sh
uv run --offline python scripts/training_coordinator.py \
  --config reports/motor-decoder-v1/browser-wasm-001/config.json \
  --database reports/motor-decoder-v1/isolated-wasm.sqlite3 \
  --port 7850
```

This development coordinator defaults to `127.0.0.1:7850`. It schedules work and aggregates returns; contributors' browsers execute the neural and physical episodes. Its SQLite database persists leases, results, generation progress, and candidate checkpoints across restarts. Restart with the same command and database to resume. A database made with another config hash or model fingerprint is rejected; use a separate database for a different build. The managed deployment uses the same protocol with transactional Firestore storage instead of this local SQLite file.

Check the service without allocating a job:

```sh
curl --fail http://127.0.0.1:7850/api/training/status
curl --fail http://127.0.0.1:7850/api/training/checkpoint
```

This localhost coordinator command is for isolated development/API tests. The contributor UI has a fixed coordinator: its own origin when hosted on public HTTPS, or `https://flytrain.morisoba.moe` when served locally. It checks config/model identity automatically, and **Start** leases and executes work. A separate development pool needs an explicitly configured client build; the public UI provides no coordinator switch.

The browser cohort's two positive/negative search pairs share a seed within each pair. Multiple contributors receive distinct leased jobs, and each contributor has at most one active lease. The coordinator fixes the `maintained_flight` stage, duration and exact parameter vector; the page cannot override them. After all four search results arrive, it assigns the three paired acceptance seeds.

The UI distinguishes episodes completed on this computer from results accepted by the coordinator. Exact retries of an accepted result are idempotent. Expired, reassigned, incompatible, or altered results are rejected and are not counted as contributions. Completed but unsent results are retained locally for retry against their original coordinator. There is no unshared fallback when submission fails.

Leases last 30 minutes and renew every 60 seconds while an episode is running and unpaused. Switching tabs keeps training and lease renewal active; only preview drawing and idle status polling are suspended while hidden. Manual **Pause** stops computation and lease renewal. A pause longer than the lease can invalidate the result; resuming computation does not resurrect an expired lease. **Stop** releases incomplete work. A completed result awaiting upload keeps its original lease until normal expiry so a later Start can retry submission; temporary upload failure does not proactively invalidate that result. Disconnected leases still expire on the server.

## Download checkpoints and full history

The browser cohort's durable location is project `flyheaven`, Firestore `(default)` in `us-central1`, under `training_runs/banc888-motor-decoder-wasm-20260914`. Run metadata and `generations`, `jobs` and `contributors` preserve candidates, assignments and accepted results across Cloud Run restarts, scale-to-zero and new revisions; preview telemetry is separate. The native decoder's 17-result, generation-1 history remains in `training_runs/banc888-motor-decoder-v1-20260914`, and the earlier 14-parameter run remains in `training_runs/banc888-v1-20260913`. The container filesystem is not the database. See the [storage guide](../deploy/cloudrun/README.md#preserve-or-initialize-training-history).

Use **Download checkpoint** or the direct JSON download:

```sh
curl --fail --output heaven-checkpoint.json https://flytrain.morisoba.moe/api/training/checkpoint
```

The JSON contains the deployed build's trainable parameter vector, generation, and frozen model/config fingerprints. The live browser release contains 672 decoder coefficients; verify the config hash above before treating a download as this cohort's checkpoint. Older deployments have different contracts. The checkpoint is not a copy of the entire BANC graph or body assets. Downloads are uncached, and candidates remain unverified.

For complete history, use the [private Firestore export procedure](../deploy/cloudrun/README.md#checkpoints-and-private-full-history-backups). It requires an operator-owned private Cloud Storage bucket and exports all nested records. The public endpoint exposes only the current parameter checkpoint. Full exports include contributor identifiers and lease tokens and remain private. Earlier VM/SQLite backup instructions are retained only in the [alternative VM guide](../deploy/gcp/README.md#verify-and-maintain).

## Contribute from another computer

Open `https://flytrain.morisoba.moe/train.html` and press **Start**. The page runs simulations on your own computer and sends their results to the GCP pool. The matching downloaded client joins the same pool automatically through this domain. The deployed `run.app` page also connects to the same stored run.

### Ready-to-copy contributor archive

The generated [contributor ZIP](../dist/training-client/fruit-fly-training-client-1f3b0935af59.zip) includes the prepared model and browser client, so recipients do not need to clone the repository, download BANC data, install Node, or compile WASM. Its [SHA256 sidecar](../dist/training-client/fruit-fly-training-client-1f3b0935af59.zip.sha256), [file manifest](../dist/training-client/fruit-fly-training-client-1f3b0935af59.manifest.json), and [packaging verification](../dist/training-client/fruit-fly-training-client-1f3b0935af59.verification.json) are beside it under `dist/training-client/`. These are local generated artifacts, not public download URLs; an operator can distribute the ZIP and checksum together. Packaging verification does not establish that this release is live or that a browser trial passed.

Extract the ZIP completely, open a terminal in its folder, and run:

```sh
python3 serve.py --port 7842
# Windows with the Python launcher: py -3 serve.py --port 7842
# With uv installed: uv run --offline python serve.py --port 7842
```

Python 3.9 or newer is sufficient; the server uses only the standard library. It verifies every bundled file before binding to localhost. Open `http://127.0.0.1:7842/train.html` and press **Start**. It automatically uses the matching live GCP pool. Do not open the HTML through `file://`. The archive contains the training client; the repository's original observation UI remains unchanged.

To build a new matching archive from an already prepared checkout:

```sh
node scripts/package-training-client.mjs \
  --experiment-bundle=reports/motor-decoder-v1/browser-wasm-001/bundle.json
```

The builder needs Node, `uv`/Python for ZIP creation and verification, prepared runtime/model files, and the pinned upstream MuJoCo license text. It copies an allowlist of experiment assets, graph buffers, training UI and source/license notices. Every model and graph hash must match; sources changed during packaging are rejected. Output names use the first 12 config-hash characters. This archive has 92 entries, about 59 MB compressed and 248 MB before compression. Archive and HTTP integrity checks run without starting the neural model.

An [earlier bundle-worker smoke](../reports/training-bundle-smoke/result.json) loaded a different archive's WebGPU BANC network and MuJoCo body, advanced 200 ms, and checked pause/cancellation/stop. That historical result does not verify the new browser WASM archive or establish learned behavior.

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

node scripts/prepare-motor-decoder-experiment.mjs \
  reports/motor-decoder-v1/NEW_BROWSER_RUN --backend=wasm
```

Choose an unused output directory. `--backend=wasm` requires no Dawn loader, package lock or installation. It creates a new experiment; contributors should use the operator's already-matching archive instead of generating their own config to bypass an identity mismatch.

The BANC downloader checks its source lock. The training loader checks graph buffers and source/model assets against their pinned manifests. Anatomy preparation is not needed for this preview. Checked-in BANC `dist/core.js` and `dist/core.wasm` avoid needing a compiler for ordinary use. If changing native BANC runtime code, rebuild it with the repository's Emscripten setup before preparing another experiment:

```sh
source references/emsdk/emsdk_env.sh
bash packages/banc-runtime/build.sh
node scripts/prepare-motor-decoder-experiment.mjs \
  reports/motor-decoder-v1/ANOTHER_BROWSER_RUN --backend=wasm
```

The experiment builder pins executable sources, model overrides, runtime binaries and model manifests in its generated `config.json`. The older [`prepare-training-manifest.mjs`](../scripts/prepare-training-manifest.mjs) still prepares the separate canonical profile in `web/training/config.json`. In both cases, `modelFingerprint` hashes the ordered URL/digest manifest and `configHash` hashes the exact config bytes. The BANC manifest pins its graph buffers. A pending or inconsistent manifest cannot start the coordinator.

Rebuild this manifest after changes to the training environment, neural/body implementation, shaders, runtime binaries, sensory assets, or physical model. Stop active runs before changing a served build, then distribute matching code/config/assets and use a new Firestore run namespace or development SQLite database. **Do not regenerate a contributor's config merely to bypass a mismatch with someone else's coordinator.** Install the operator's exact matching build instead. Even a whitespace change to the config changes its hash.

## What is being optimized

The live browser cohort optimizes **672 bounded motor-decoder coefficients**, not neural weights:

| Parameter family | Count | Inputs and output |
| --- | ---: | --- |
| Power weights | 24 | Individual DLM/DVM motor excitation to ipsilateral wing power |
| Steering coefficients | 648 | 24 steering MNs × 3 axes × 3 lags (0/1/4 ms) × 3 phase bases (constant/sine/cosine) |

The coefficients use individual event-derived excitation and wing phase. The decoder receives no root pose, target or reward. Neural physiology, synapses, sensory gains, muscle dynamics, the existing wingbeat generator and native mechanics stay fixed. These fitted coefficients remain engineering hypotheses; changing them is not proof of improved flight or identified biological physiology.

### Earlier 27-parameter interpreter

The historical interpreter applies **27 bounded parameters** in log space, with each multiplier equal to `exp(theta)`. Its canonical defaults use `theta = 0`; separately pinned development runs declare other initial vectors:

| Parameter family | Count | Target |
| --- | ---: | --- |
| Power | 1 | Muscle force to wingbeat power |
| Deployment time | 1 | Wing deployment time constant |
| Frequency | 1 | Wingbeat frequency multiplier |
| Steering mean angle | 12 | One gain per steering muscle type |
| Steering stroke amplitude | 12 | One gain per steering muscle type |

The twelve types are b1, b2, b3, i1, i2, iii1, iii3, iii4, iv1, iv2, iv3, and iv4. Left/right homologues share a learned coefficient, while their muscle forces stay separate. Fixed anatomical axis/sign bases remain modeling assumptions; these two coefficients per type cannot learn arbitrary phase or torque directions. There are no redundant trainable global steering gains.

BANC neuron physiology, chemical weights, gap junctions, sensory transduction, native muscle dynamics, other limb mappings, and the native physical model remain fixed. Body feedback still changes neural activity within an episode. These learned interpreter gains are hypotheses, not cell-specific measurements or calibrated biological physiology. Reward and task goals score episodes; they do not write limb trajectories or body forces. All three flight tasks start grounded; landing cannot receive an airborne reset that bypasses takeoff.

The objective requires observed foot support before takeoff, a rise above the initial body height, and controlled powered flight. The full-cycle task then requires at least one continuous second of flight followed by a stable landing on loaded feet. Standing still cannot complete the task; a ballistic hop, repeated brief hops, or body/wing contact cannot substitute for sustained flight and a foot-supported landing. Terminal crashes remain penalized even after earlier flight milestones. These criteria are explicit engineering targets, not measured biological thresholds.

Version 5 recognizes a foot-origin departure even if its first airborne sample has low upward velocity or the wings deploy later. A brief foot recontact starts a fresh flight bout; it cannot accumulate airtime across hops. Body/wing contact invalidates the previous support history. Confirmation still requires powered ascent, clearance, airborne acceleration consistent with lift, and continuous controlled flight. Angular control uses a 20 ms RMS magnitude at the existing 20 rad/s limit, while attitude and catastrophic rotation are checked every sample. This is a sampled recoil-tolerant criterion, not a measurement of all wingbeat harmonics. A frozen 195-step native regression receives 184 ms flight credit but still fails its later crash at 390 ms.

For a separate reduced flight benchmark, `scripts/prepare-flight-development-bundle.mjs` writes immutable model overrides under `reports/`. It disables the uncalibrated phenomenological claw adhesion while preserving normal contacts and friction. The full/default physical model stays unchanged. Its power 1.5 and steering 0.05 starting values are operator-selected diagnostic controls, not learned values. This benchmark cannot validate gripping, climbing, or the omitted adhesion mechanism. Its distinct configuration hash, model fingerprint and environment version prevent pooling with the full model.

An optional model metadata field, `motor_excitation.steering = {kind: "hill", halfActivationHz: 80, exponent: 1}`, replaces the `clamp(rateHz / 80)` recruitment rule only for the 24 wing-steering mappings. It preserves separate left/right signals; power muscles and every other muscle retain their original rule. Missing metadata preserves the old implementation byte-for-byte. The curve constants are explicit modeling priors, outside the 27 learned gains, and are not measured physiology.

Another optional field, `wing_actuation.steering_force_reference`, declares all twelve per-type force references in `[0,1]`. The interpreter then uses `gain × (force − reference)` for both bias and amplitude. At that declared operating point it produces the calibrated wing table exactly, independently of gain; left/right deviations remain separate. Without the field, the original absolute-force arithmetic is preserved. The published steering derivative uses activity deviations, but its numerical activity baselines are not transferable WASM forces. A reference must be declared and tested for this model; it is not an extra learned parameter or a measured physiological baseline. The bundle builder's `--force-reference=...` accepts a pinned reference derived from the frozen native force calibration, separately from any optional Hill candidate.

`scripts/calibrate-steering-recruitment.mjs` derives tied per-type gain corrections from a frozen, exactly reproduced native muscle trajectory. Passing its candidate JSON as the bundle builder's second argument creates a separately pinned opt-in model. The correction matches aggregate mean muscle force before the steering basis over 100–280 ms; individual sides, startup, temporal variation and live feedback can still differ. Both mappings require paired live evaluation before interpreting this as a flight improvement. The canonical model does not opt in automatically.

The [paired live evaluation](../reports/steering-recruitment-live/README.md) did not justify adopting this curve: one selection seed matched the old flight metrics and the other lost takeoff and crashed earlier. The preceding [one-generation training run](../reports/flight-development-v5/README.md) changed 10 of 27 parameters but also worsened both selection episodes. Those runs are stopped; parameter movement and input desaturation alone are not evidence that this flight system learns.

The [sensory observability audit](../reports/flight-sensory-observability/README.md) finds identical full encoder outputs for opposite angular velocities at a fixed native airborne state. Current rotation priors preserve organ/side identity but discard direction; later motion or contacts can still provide indirect differences. The separate [classical controller assay](../reports/flight-classical-control/README.md) maintains the reduced plant within 8.03 degrees of level for 1.5 seconds using bounded synthetic steering forces and direct diagnostic feedback. It establishes actuator control authority, not autonomous BANC flight, takeoff, landing, or realistic muscle dynamics.

The [paired muscle-delay assay](../reports/flight-muscle-latency/README.md) reuses that exact controller and initial plant state. The direct-force trajectory reproduces the earlier result exactly. Adding the actual 28 native muscles, with prepared trim activation and explicit fatigue compensation, still passes the 1.5-second control: maximum tilt is 6.68 degrees. Adding the existing nominal 50 ms rate smoother before those muscles produces 112.62 degrees of tilt and fails. This isolates a concrete sensitivity to the current output filtering; it does not prove every controller with that delay must fail. These are synthetic controller inputs, not BANC flight.

The [afferent robustness checks](../reports/flight-afferent-identification/robustness/AGGREGATE.md) retain some early motor spike-timing responses across two conditioned neural states, but do not establish a dependable three-axis feedback map. Endpoint spike counts can hide transient timing changes. The [primary physiology review](../reports/flight-sensory-observability/motor-timing-review.md) supports preserving individual spikes and distinguishing power from steering muscles; it does not supply a universal measured steering-force kernel. Optimizing the existing 27 gains cannot restore discarded timing, assign missing sensory directions, or calibrate the fixed neural physiology.

The [proximal sensory correction](../reports/flight-proximal-sensory-repair/README.md) removes unsupported generic body-motion input from50 explicitly annotated proximal hair-plate sensors while preserving their graph membership. Its single-seed native regression increased qualifying flight from102ms to166ms and registered takeoff, but still failed from excessive rotation at480ms. This local result is not stable flight or a trained-policy improvement. The [frozen sensory-family assay](../reports/flight-sensory-families/run/RESULTS.md) also shows strong DLM recruitment from odor input alone and native body transducers alone; the fallback is not the unique cause of high motor firing.

`createWingMotorEventReader` provides guarded event observation using cumulative counts and last-spike timestamps at the existing 0.5 ms neural / 2 ms body exchange. It accepts only spiking neurons with at least 2 ms refractory time, checks every interval and count, and emits no historical events when first initialized. It does not depend on the global recent-spike ring, convert rates into artificial spikes, or change muscle excitation. Event transport and a validated event-to-force model are separate requirements.

The optional developer `onMotorEvents` evaluation hook records this signal before each body block. Its [live paired check](../reports/flight-motor-event-observer/README.md) recovered all 1,460 wing-MN events and matched the recorded physics digest exactly with observation disabled. Both runs still crashed at 388 ms. Raw counts in the 100–280 ms startup window showed individual DLM rates of 83.3–155.6 Hz and no b1 spikes, so neural calibration also remains unresolved. The evaluator's per-job `captureMotorEvents` and `capturePhysicsDigest` options are diagnostic recording only; the contributor interface and optimizer do not use them.

Read-only native evaluation supports an explicitly planned `dawn-metal` backend through the isolated [diagnostic tooling](../reports/native-webgpu-tooling/README.md). Its plans pin the loader and package lock; fallback results are rejected. On the paired legacy cases it took 14.35 active wall seconds per simulated second versus 97.05 for WASM, excluding setup. Native Metal, Safari and WASM outcomes must be identified separately: the observed neural trajectories are not numerically interchangeable. No production dependency or contributor backend was changed by this diagnostic installation.

`scripts/serve-training-dev.py --bundle reports/flight-development-v5/bundle.json --database reports/flight-development-v5/coordinator.sqlite3 --port 7849` serves this benchmark through the existing loopback coordinator and original training UI. `node scripts/contribute-training-native.mjs http://127.0.0.1:7849/ reports/flight-development-v5/contributor 8` evaluates eight assigned jobs with the actual WASM neural backend and native MuJoCo. Every result goes to that coordinator; this is a development contributor, not a separate optimizer or a Safari/WebGPU result. Creating `STOP` inside the contributor report directory cancels its current job without submitting an incomplete result. Parameter updates must still be checked with the reporter and then compared on held-out episodes.

Flight support is inferred from the native whole-body center-of-mass velocity over a 50 ms contact-free window, excluding the ground's launch impulse. This includes moving wing and limb mass; thorax motion alone could mistake internal recoil for lift. Foot support uses upward world-space contact force on distinct tarsal feet, with separate body/wing collision counts. Success is checked at the complete episode horizon, so an early hop or landing followed by a crash cannot pass. Finite failed attempts retain bounded earned flight progress minus a severity penalty, giving the optimizer a signal even when no candidate completes the task.

The earlier schema-1 optimizer uses four antithetic pairs, Gaussian perturbation scale `sigma = 0.25`, learning rate `0.035`, and maximum coordinate update `0.15`. For each coordinate, it adds

```text
clamp(learningRate / (2 * pairCount * sigma)
      * sum((return_plus - return_minus)
            * (assigned_plus - assigned_minus) / (2 * sigma)), -0.15, 0.15)
```

to the baseline log parameter and clips to its configured bounds. Using the actual assigned separation accounts for clipping near parameter bounds and matches the browser implementation. Returns must be finite and within `[-10, 10]`. The server deterministically generates jobs and sends explicit parameter arrays and seeds; contributor-side random-number implementations do not choose shared perturbations. Shared aggregation updates the candidate only after all eight jobs are accepted, always with status `unverified`.

Each parameter may declare `searchScale` in `(0, 1]`; omission is exactly equivalent to1. Assigned perturbations become `sigma * searchScale * noise`. The update still uses the actual assigned separation, so this also preconditions local updates by approximately `searchScale²`; the physical-coordinate update cap is unchanged. Search scales are configuration values, not additional learned parameters. Changing them changes the configuration hash and requires a new coordinator namespace. The local v7 run uses0.5 for power and deployment,0.15 for frequency, and1 for the24 steering coefficients. Existing configurations without scales retain their exact sampling and update behavior.

### Current guarded checkpoint selection

The **schema-2 acceptance guard** nominates the highest-scoring search vector, with lexical job-ID tie-breaking. The browser cohort uses two antithetic pairs, `sigma = 0.02`, power search scale `0.1` and steering search scale `0.05`. The coordinator compares the nominee and incumbent on three fresh matched training seeds. Only a strictly positive mean paired return replaces the incumbent; ties and regressions retain it. Bounds/scales apply to search, but the legacy gradient's `learningRate` and `maximumUpdate` do not constrain this sampled-candidate nomination.

During comparison, `/checkpoint` returns the incumbent. Phase transitions, results, comparisons and the selected successor are stored atomically. A browser-cohort generation normally uses 10 evaluations: four search plus six comparisons. Earlier four-pair runs used 14. An exactly identical proposal skips comparison. Rejections still complete a generation and are distinct from stalled or incomplete work. These comparisons are training data, not held-out validation; checkpoints remain `unverified`.

The browser cohort declares `optimizer.acceptance.nativeExecution = {backend:"wasm", moduleSha256:...}`, where the hash must equal the pinned `/banc-engine/dist/core.wasm` asset. Result provenance must identify `backend:"wasm"`, `neuralEngine:"wasm"` and the exact `wasmExecution` pin, without fabricated native WebGPU metadata. Termination/cancellation flags and the 0.5 ms neural / 2 ms body timing contract are checked; partial or infrastructure-failed evaluations are rejected. The earlier Dawn guard remains supported under its separate strict loader/package-lock pins and namespace. The two backends never share fitness.

`vision: false` is deliberate in this version. The model receives odor, taste and body feedback; the low-resolution 3D preview is an observer view, not retinal input. Neural integration is fixed at `0.5 ms` and the neural/body exchange block at `2 ms`. Budget and preview controls do not change these values. This version does not establish visual food localization or fix the known rigid-wing/contact and unsigned sensory-feedback limitations.

## Development flight checks

Stop active contributors before changing the served model, then regenerate the manifest. Use a new database for this 27-parameter contract. The development server serves the original training UI and its local coordinator together:

```sh
node scripts/prepare-training-manifest.mjs
.venv/bin/python -B scripts/serve-training-dev.py --port 7845 \
  --database reports/training-flight-27/coordinator.sqlite3
```

Open `http://127.0.0.1:7845/train.html` in Safari. This explicitly marked development build evaluates coordinator-assigned jobs and preserves results in the specified SQLite file. The public client still has no coordinator switch or unshared training mode.

Before starting optimization, check that every interpreter coefficient affects native actuation, then evaluate equal-wing mechanical controls and the complete neural/body baseline. A passing control-sensitivity test is not evidence of takeoff or stable flight. Diagnostic rig constraints and synthetic muscle drive must never be presented as autonomous neural behavior.

During training, inspect completed generation updates and behavioral results independently:

```sh
.venv/bin/python -B scripts/report-flight-training.py \
  reports/training-flight-27/coordinator.sqlite3 --last-generations 2 \
  --output reports/training-flight-27/progress.json
```

The report includes every named parameter delta, the number changed, and update norms. It distinguishes unfinished generations, real changes, equal-score no-update, and other zero updates. A changed parameter vector or a less severe crash does not establish learning; compare takeoff, sustained airtime, and landing on fresh evaluation episodes.

## Validation and limits

The two configured selection seeds are repeatedly used to compare local candidates. They are not an untouched final test. The internal research API `TrainingClient.evaluateCheckpoint()` uses three separate configured test seeds, never used in the local update/selection loop. Repeatedly choosing models based on those test results also compromises their independence; stronger claims require additional predeclared seeds, initial conditions and independent reproductions.

`locally-validated` means the selected stage's implemented criteria passed on selection seeds. `locally-tested` means they passed on the separate local test seeds. Validation and curriculum badges belong to the current parameter vector; changing parameters clears earlier passes instead of carrying success forward to an unevaluated candidate. Evaluating a shared checkpoint adopts it for local evaluation and clears previous local curriculum claims. Neither label establishes biological realism, robust full-sequence control, or generalization to different hardware/backends or body models. Shared result provenance is checked for consistency, not independently attested execution. Export and independently evaluate shared candidates before deciding whether to retain them or create a later-stage experiment.

The coordinator's focused tests exercise HTTP/CORS, leases, heartbeat/expiry, retries, incompatible and nonfinite inputs, concurrency, restart persistence, raw config hashes, and agreement with the browser's ES update/checkpoint format:

```sh
uv run --offline python -W error::ResourceWarning \
  -m unittest discover -s tests -p 'test_training_coordinator.py'
```

These tests use synthetic objectives to validate infrastructure and algebra. Passing them is not evidence that a trained fly completes any behavioral stage.

The earlier VM deployment passed HTTPS, a browser WebGPU contribution and checkpoint persistence checks; that [historical evidence](../reports/gcp-deployment/README.md) does not validate the new WASM release. The [Cloud Run guide](../deploy/cloudrun/README.md#verify-a-release) separates transport checks from an actual browser WASM trial. The older `verify-gcp-training.mjs` asserts WebGPU and is not a WASM release verifier. UI-only checks also cannot substitute for an accepted real episode. The contributor page exposes coordinator-assigned jobs only; internal research APIs retain independent evaluation support.
