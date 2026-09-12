# Local release artifacts

- **Simulation:** `fruit-fly-brain-wasm-0.1.2.tgz` / `.zip` / `.wasm`.
- **Anatomical viewer kernels:** `fruit-fly-brain-view-wasm-0.1.0.tgz` / `.zip` / `.wasm`.

Use the archives for JavaScript wrappers, TypeScript declarations, source, build scripts, tests, and license notices. Standalone binaries expose the C ABI and normally run through their matching wrapper. `SHA256SUMS` covers the versioned artifacts and manifests. Nothing has been published to an external registry.

Simulation 0.1.1 supersedes 0.1.0: it fixes non-finite voltages after a neuron remains quiet for more than 10,000 timesteps. The old artifacts remain for version history.

Simulation 0.1.2 adds lossless compact connectivity, improved state locality, and faster threshold scheduling. See `reports/wasm-performance.json` and `reports/wasm-performance.md` in the repository for measured 100-brain performance and reproducible comparisons against 0.1.1. Earlier versioned artifacts remain unchanged.
