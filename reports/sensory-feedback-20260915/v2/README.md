# Frozen sensory feedback v2

Four local bundles are available: `off.bundle.json`, `vision.bundle.json`, `airflow.bundle.json`, and `vision-airflow.bundle.json`. All embed the same 51 exact JS runtime sources and frozen original body XML/metadata. Only the declared visual/antennal feedback differs across variants. BANC binaries/data, native body binary, fitted-v2 motor coefficients and checkpoint identity remain unchanged.

The common model fingerprint is `2c0bc5e9d9a7c294ae400c0ee2f1412b720f68dd78e7bcbe2d09467d8d47e55b`; each config has its own hash in `manifest.json`. Vision variants truthfully declare `vision:true` and set `visionFeedback.camera.textureAntialias:true`. Environment versions use `sensory-feedback-v2`. Notes explicitly describe local-only operation.

Reproduce exact bytes (input archive/source SHA checks are mandatory; the script refuses to overwrite differing existing evidence):

```sh
node reports/sensory-feedback-20260915/build-v2-bundles.mjs
```

The generator was run twice with identical output. It preserves the original archive creation metadata and records a fixed v2 version timestamp to make the resulting bundles byte-reproducible.

All four bundles passed imports and configuration checks through `installFeedbackRuntime`, including the original 672-neuron decoder identity contract, all asset checksums, truthful flag/profile pairing, and rejection after flipping the vision flag. No brain/body environment was instantiated and no simulation was run by these checks.

```sh
node reports/sensory-feedback-20260915/validate-v2-bundle.mjs off
node reports/sensory-feedback-20260915/validate-v2-bundle.mjs vision
node reports/sensory-feedback-20260915/validate-v2-bundle.mjs airflow
node reports/sensory-feedback-20260915/validate-v2-bundle.mjs vision-airflow
```

After v1 finishes, workspace runtime JS can match v2 by applying the existing seven-file contract candidate patch and then copying `reports/sensory-feedback-20260915/retinal-sensor-antialias.js` to `web/training/retinal-sensor.js`. Exactly four runtime JS files differ from the workspace snapshot used by this review:

| Runtime target | Exact replacement source under `reports/sensory-feedback-20260915/` |
| --- | --- |
| `web/training/config-schema.js` | `vision-contract-candidate/web/training/config-schema.js` |
| `web/training/episode.js` | `vision-contract-candidate/web/training/episode.js` |
| `web/training/sensory-feedback.js` | `vision-contract-candidate/web/training/sensory-feedback.js` |
| `web/training/retinal-sensor.js` | `retinal-sensor-antialias.js` |

`workspace-runtime-differences.json` gives each old/new hash. The other changes in the seven-file candidate patch cover tests and build tooling, not runtime math. The ordinary experiment generator will also need its version marker, antialias option and notes updated if it is to generate future v2-equivalent experiments; these frozen bundles are built by the separate reproducible generator above. Preserve all existing v1 bundles/results. Do not replace the canonical old 27-parameter training configuration with a diagnostic bundle.

Frozen body XML and metadata continue to be bundle overrides; matching these runtime JS files does not authorize replacing the checkout's body assets. Serve/evaluate these bundles through the override-aware local runtime.
