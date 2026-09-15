# Browser package handoff

The explicit structural experiment can now be passed to the existing browser client packager. The packager retains the supplied configuration bytes and copies its hash-pinned runtime JavaScript, native XML/metadata, and leg catalog. The ordinary UI and coordinator connection behavior remain unchanged.

No package was built during this change because packaging copies the prepared graph and WASM dependencies and creates a ZIP. The commands below are the handoff for a later build with sufficient disk space:

```sh
node scripts/package-training-client.mjs \
  --experiment-bundle=reports/structural-proprioception-20260915/v2/structural.bundle.json \
  --output-dir=dist/structural-proprioception

uv run --offline python \
  dist/structural-proprioception/fruit-fly-training-client-ecce268f712a/serve.py \
  --port 7843

open -a Safari http://127.0.0.1:7843/train.html
```

The output directory name uses the first twelve characters of the experiment's configuration hash. A differently generated configuration will have a different directory name; the packager prints the resulting paths.

**Starting training requires the page's coordinator to serve this exact configuration and model fingerprint.** This candidate has not been deployed to that coordinator. Until it matches, the existing synchronization checks reject the candidate. Packaging does not redirect the public run, start training, change checkpoints, or provide an independent local-training mode.

`--model-bundle` is optional and accepts an already packaged model with the same configuration identity. It is not an overlay from an older run. The separate `scripts/serve-training-dev.py --bundle` interface still supports only its original body-asset bundles; use the packaged `serve.py` for this handoff.

Owned experiment assets use a narrow allowlist: the 52 supported runtime JavaScript modules, the two native body files, and `banc-leg-proprioception-v1.json`. Every owned asset must have a matching SHA-256 entry in the exact configuration. UI/coordinator files, arbitrary data, paths outside the allowlist, and native binary overrides are rejected. External prepared graph and WASM dependencies must still match their pinned hashes; packaging performs its existing import, served-file, and archive checks.

Validation: `node --test web/test/training-experiment-package.test.mjs web/test/training-brain-package.test.mjs` passes 14 tests using small fixtures. Read-only parsing of all three actual v2 bundles confirms 52 owned executable sources per bundle and unchanged configuration bytes. No full package build, Safari execution, coordinator change, or deployment was performed.
