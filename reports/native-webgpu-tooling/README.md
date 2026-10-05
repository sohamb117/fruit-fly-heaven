# Native Metal compatibility tooling

This directory is an isolated installation of the official `webgpu@0.6.0` npm package from `dawn-gpu/node-webgpu`. It is not a production dependency and does not change canonical simulation assets.

The registry tarball's SHA-512 was verified before installation. It is 34,417,631 bytes compressed and 95,162,161 bytes unpacked, including a prebuilt macOS universal `dawn.node`. The package's postinstall and imported utility files were inspected: its only installation action is checking/removing the macOS quarantine attribute. Installation used `--ignore-scripts`, so this action did not run. No source build, extra binary download, or quarantine modification occurred.

The root agent installed with:

```sh
npm install --ignore-scripts --no-audit --no-fund --save-exact webgpu@0.6.0 --cache=.npm-cache --fetch-retries=0 --fetch-timeout=20000
```

The lock resolves `webgpu` 0.6.0, `@webgpu/types` 0.1.72, `debug` 4.4.3, and `ms` 2.1.3. The installation, audit tarball, and dedicated npm cache occupy about 157 MiB together.

## Verified compatibility

`compatibility.json` records the successful root-executed probe, including source/native-binary hashes:

- Node v23.7.0, macOS 26.3, arm64.
- Hardware Apple M2 Pro, Metal 3; `isFallbackAdapter: false`.
- Real GPU compute transformed `[1, 2, 4, 8]` to `[3, 5, 9, 17]`.
- The existing, unchanged `WebGPUBrain` and WGSL shader ran a four-neuron fixture for 64 steps / 32 simulated milliseconds, including chemical/electrical edges and a graded neuron. GPU readback returned finite state and two spikes.
- No full BANC graph, behavior, training, or speed claim is established by this compatibility probe.

The subagent's restricted execution context returned no Metal adapter; it did not attempt a fallback. This is retained in `backend-sandbox-check.json`. The root execution context successfully ran the same package on hardware.

## Diagnostic integration

```js
import {installNativeWebGPU} from './backend.mjs';

const native = await installNativeWebGPU();
// navigator.gpu and WebGPU globals now use the pinned native Metal provider.
// Record native.provenance alongside the evaluation's backend/model pins.
// Destroy the environment and all GPU devices before cleanup:
native.uninstall();
```

The loader verifies the exact package version, lockfile registry integrity, and installed package/index/native-binary hashes before import. It forces `backend=metal` and rejects a missing or fallback adapter. It preserves original globals for cleanup and releases its own GPU reference on uninstall. It does not install a model loader or network/file-fetch adapter.

Node's fetch cannot read `file:` shader URLs. `probe.mjs` serves only its exact source-pinned local `neural.wgsl` through a narrow file-read adapter; a coordinator evaluator must instead map that exact URL to its pinned same-origin shader asset. No broader file/network fallback is required.

Run the bounded probe from the repository root with an unused output filename:

```sh
/Users/soham/.nvm/versions/node/v23.7.0/bin/node reports/native-webgpu-tooling/probe.mjs --output=compatibility-repeat.json
```

`backend-check.mjs` is an optional adapter install/uninstall check without compute or neural evaluation. The initial loader was also accepted by the root's explicit Dawn evaluator. Neither script runs training or accepts a software/WASM fallback.
