# Reproduce the implementation

The repository contains the BANC/FlyBody implementation, the original 3D console, the shared training client, and deployment tools. It includes the small BANC WASM runtime, native model descriptions, reference controller assets, tests, source locks, and licenses. The BANC datasets, installed dependencies, training databases, distribution archives, and large experiment captures are generated or stored separately.

The target remains brain → VNC → motor neurons → muscles → body, with food localization → approach → landing → probing/feeding → takeoff/flight. Passing software tests or replaying a reference controller does not establish that the BANC controller achieves that sequence.

## Prepare a fresh clone

Run from the repository root with `uv`, Node, and npm installed:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-banc.txt
npm ci --prefix packages/flybody-runtime
uv run --python .venv/bin/python python scripts/prepare-banc.py --download
uv run --python .venv/bin/python python scripts/prepare-banc-console.py
```

The BANC downloader checks the upstream files against [`configs/banc-v888.lock.json`](../configs/banc-v888.lock.json). Preparation uses the physiology configuration and the included sensory annotation evidence to create `data/prepared/banc888/`, including its console mappings. Source data and prepared buffers are ignored by Git. They are required for native training and the BANC tests that inspect real neuron identities.

`npm ci` installs MuJoCo 3.13.0 from the included lockfile. Its JavaScript and WASM files live under `packages/flybody-runtime/node_modules/` and are not committed. The native XML, metadata, wing calibration, and small native/reference fixtures in `models/` are included, so ordinary use does not require regenerating them or downloading the upstream FlyBody checkout.

Start the local site after preparation:

```sh
uv run --python .venv/bin/python python scripts/serve.py --port 7842
```

Open `http://127.0.0.1:7842/train.html` for shared training or `http://127.0.0.1:7842/?population=1` for the observation console. The training page targets the GCP coordinator at `https://flytrain.morisoba.moe` and waits for Start before loading the model and computing. That domain's activation is pending; local contributions need it to be reachable. Opening a local page does not start a new local training pool. The observation console's separate anatomical view additionally requires `scripts/prepare-banc-anatomy.py`; anatomy is unnecessary for the training preview.

The canonical training config pins the exact runtime and model hashes. To verify and package that existing build:

```sh
node scripts/package-training-client.mjs
```

This creates a verified contributor archive under ignored `dist/training-client/`. It requires the prepared assets and access to the pinned MuJoCo license text. A mismatch must be resolved with the operator's matching build before joining its pool. Do not regenerate the config merely to bypass a mismatch with a live coordinator. See [training instructions](training.md) for contribution, checkpoint download, and deployment details.

## Rebuild model or runtime assets deliberately

The BANC runtime's `packages/banc-runtime/dist/core.js`, `core.wasm`, and `neural.wgsl` are intentionally versioned runtime artifacts. Together they are about 26 KB. Their C++ and shader sources, build script, and third-party notices are included. Install the pinned compiler only when rebuilding native code:

```sh
git clone https://github.com/emscripten-core/emsdk.git references/emsdk
uv run --python .venv/bin/python python references/emsdk/emsdk.py install 4.0.23
uv run --python .venv/bin/python python references/emsdk/emsdk.py activate 4.0.23
source references/emsdk/emsdk_env.sh
bash packages/banc-runtime/build.sh
```

To regenerate the native FlyBody model from its pinned upstream source:

```sh
git clone https://github.com/TuragaLab/flybody.git references/flybody
git -C references/flybody checkout d015e9bfe441bd90ae431bac24c55cb74bdbce26
uv run --python .venv/bin/python python scripts/prepare-flybody-runtime.py
```

Use an existing checkout at the stated revision instead of cloning over it. Model preparation uses the included wing calibration and downloads the experimental wing-cycle archive if absent. `scripts/distill-flybody.py` regenerates the optional older reduced-body model; its XML references meshes in the pinned upstream checkout. The main native `flybody-mujoco.xml` is self-contained.

The reference flight policy, trajectory metadata, native reference XML, and fixtures in `models/` support the console's comparison mode and associated tests. They are runtime assets, not training checkpoints. Their source/license records and export tools are included. Regenerating the learned reference policy requires its separate Python 3.11/TensorFlow environment documented in `scripts/export-flybody-policy.py` and the baseline requirements/lock files; it is not required for ordinary BANC training. Wing calibration reproduction uses the pinned FlyRanch checkout described in the [wing repair summary](../reports/flybody-flight-repair.md#reproduction).

After an intentional change to neural code, body physics, sensory mappings, model data, or runtime binaries, rebuild the training manifest:

```sh
node scripts/prepare-training-manifest.mjs
node scripts/package-training-client.mjs
```

A changed config or model fingerprint defines a different training build and requires a matching coordinator storage namespace: a new Firestore run for managed hosting or a separate SQLite database for development. Preserve earlier history rather than repurposing it for another model identity. Root `dist/` archives are ignored; package-level BANC runtime artifacts remain versioned.

## Reproduce the managed deployment

The selected target is Cloud Run service `fly-training` in project `flyheaven`, region `us-central1`, with transactional Firestore storage. Follow the [Cloud Run build, migration and deployment guide](../deploy/cloudrun/README.md) to package the allowlisted contributor files, build an image, record its immutable digest, migrate compatible history and deploy it. The canonical model/config identity and `build-manifest.json` identify the inputs independently of mutable image tags.

The coordinator state lives in Firestore `(default)` at `training_runs/banc888-v1-20260913`, including nested generation, job and contributor records. Container restarts and new revisions do not reset it. The runtime identity has `roles/datastore.user`; no credentials or training database are baked into the image. The [older VM/SQLite deployment](../deploy/gcp/README.md) is retained as a separate option, not the active hosting target.

## Bounded validation

After preparation, the following checks run locally without paid compute or a long behavior-training run:

The complete Python suite also imports the legacy FlyWire ctypes bridge. On macOS, compile its library before running that suite:

```sh
mkdir -p build
c++ -O3 -std=c++17 -dynamiclib -fPIC src/brain.cpp -o build/libflybrain.dylib
```

This ignored native library is separate from the browser's BANC WASM runtime. The legacy Python bridge currently expects the macOS `.dylib` path. On other platforms, the BANC preparation and training coordinator tests can be selected separately without importing `test_brain.py`.

```sh
node --test packages/banc-runtime/test/*.test.mjs scripts/benchmark-transplant-metrics.test.mjs
node --test --test-force-exit --test-concurrency=1 web/test/*.test.mjs
uv run --python .venv/bin/python python -m unittest discover -s tests -p 'test_*.py'
uv run --offline python -m unittest discover -s deploy/cloudrun -p 'test_*.py'
uv run --offline python -m unittest discover -s deploy/gcp -p 'test_*.py'
node scripts/test-flybody-reference-trajectory.mjs
```

`banc-ground-sense.test.mjs` reads the generated BANC sensory mappings; `banc-proboscis.test.mjs` reads generated `io.json`. The runtime embodiment test reads the included `models/flybody-reduced.json`. Other core unit tests use small fixtures and require no historical report captures. Existing FlyWire behavior tests explicitly skip if their separate old prepared dataset is absent; [the preserved FlyWire README](../README-flywire.md) documents that preparation.

The hosted UI regression `web/test/training-hosted-view.browser.mjs` uses intercepted metadata, starts no model, and checks mandatory sharing, failed connections, checkpoint downloads, and layouts. It needs Playwright and an installed Chrome executable; set `PLAYWRIGHT_MODULE` and `CHROME_PATH` for their locations. The live `scripts/verify-gcp-training.mjs` check is separate: it executes and submits a real episode to GCP.

The managed deployment package/server tests run locally; they do not prove Google IAM, Firestore transactions or custom-domain certificate readiness. Install `requirements-cloudrun.txt` for Firestore-backed tools. The Firestore test module requires the loopback emulator and refuses non-loopback endpoints; a missing emulator is not a passing integration test. Its exact setup and commands are in the [Firestore validation record](../reports/cloudrun-firestore/README.md). The [managed release checks](../deploy/cloudrun/README.md#verify-a-release) use `scripts/verify-cloudrun-host.mjs` for public HTTPS/API checks without compute and `scripts/verify-gcp-training.mjs` for a real contribution. Historical VM deployment results do not substitute for those checks.

`scripts/verify-training-ui.mjs` and `scripts/verify-training-evaluation.mjs` preserve the earlier optional-sharing UI harnesses. They refer to removed opt-in/import/evaluation controls and are not current UI acceptance tests. Use the hosted UI regression, `scripts/verify-training-ui-idle.mjs`, and the GCP verifier for the current shared-only page. Internal independent research APIs remain available for scientific validation; they are not exposed as an unshared contributor mode.

## Historical evidence and archived inputs

Written experiment summaries under `reports/` are retained. Raw JSON/JSONL captures, images, video, native arrays, logs, database backups, and generated browser reports are ignored by default. Compact provenance or regression evidence may be explicitly selected for version control. Some links in older summaries refer to local artifacts from those runs; a missing raw report in a fresh clone does not imply that the corresponding check ran or passed there.

The ordinary unit suites do not depend on historical captures. These analysis tools do, and need the original archived inputs restored at their recorded paths:

| Analysis tool | Historical input |
|---|---|
| `scripts/audit-onset-contact-geometry.mjs` | Static module import `reports/flybody-onset-causal/sources/body-world.js`, plus its captured onset data. |
| `scripts/verify-banana-endpoint-surface.mjs` | Static module import `reports/observation-60min-20260913/source-snapshots/body-world-before-endpoint-fix.js` and `reports/observation-60min-20260913/frames.jsonl`. |
| `scripts/verify-fruit-camera-occlusion.mjs` | The same recorded `frames.jsonl`, including frame 80. |
| `scripts/verify-flybody-habitat-collision.mjs` | `reports/flybody-onset-causal/contact-geometry.json`. |
| `scripts/verify-flybody-wing-pose.mjs` | `reports/flybody-wing-landmarks.json`. |
| `scripts/verify-banc-assay-input-parity.mjs` | Recorded full and posture neural assays under `reports/banc-sensory-recruitment/`. |
| `scripts/verify-task-monitor.mjs` | Recorded flight traces in `reports/flybody-flight-live-audit.json` and `reports/flybody-flight-repair-final-live.json`. |
| Onset/wing replay, plotting, and interface fitting tools | Their recorded capture, source snapshot, and candidate files; inspect the tool's arguments and its experiment summary. |

Retain each historical capture with its original source hashes and snapshots when archiving it. New recordings from changed code are new experiments and cannot substitute for an older run's evidence. Some tools import a source snapshot before reading command-line arguments, so restoring only the large JSON capture is insufficient.

Training databases, Firestore exports and downloaded checkpoints also stay outside Git. The public checkpoint is a small parameter snapshot, not full training history. Use the [documented Firestore export procedure](../deploy/cloudrun/README.md#checkpoints-and-private-full-history-backups) for private managed-history backups; full exports include contributor identities and lease tokens. Preserve SQLite snapshots only for local or historical VM runs. Git source history is not a backup of training progress.
