# Training brain view

The **Brain** tab displays 4,096 measured BANC v888 neuron anchors from the current fly, colored by its simulated membrane voltage and recent spikes. Selecting a point shows its neuron identity, voltage and filtered rate. These are representative anatomical points, not full neuron skeletons. Missing anatomical locations are omitted. Every neuron and connection remains in the simulation.

**Parameters** shows only the current trial's assigned decoder values, with search and pagination. It does not substitute the coordinator checkpoint. Values stay fixed during a trial and change when a new assignment starts. A stopped or completed trial retains its last readings until the next trial starts.

Observation runs only while the Brain tab is visible and preview quality is enabled. The worker samples at most twice per wall second after an existing neural read. It copies selected voltages, rates and last-spike timestamps into owned transfer buffers; it never steps the brain, reads the full network, or scans connections. Hidden tabs stop observation while training continues. Observer errors affect only the preview.

The separate `observer-worker.js` entry wraps the released worker controller and installs a read-only `WasmBrain.readState` observer. The pinned worker, environment and neural kernel bytes remain unchanged. Tests compare native states, spike events and original read counts with observation enabled and disabled, as well as trial transitions, cancellation and error handling.

Generate local display data from the prepared BANC graph and anatomy:

```sh
node scripts/prepare-training-brain.mjs \
  --config=reports/motor-decoder-v1/browser-wasm-001/config.json \
  --output=web/training/brain-sample.json
```

The generator verifies source checksums and every anatomy ID against `ids.bin` before selecting points. Packaging generates this display asset automatically. It is outside the scientific model fingerprint because it cannot affect the simulation.

When local model sources contain unreleased changes, preserve the released simulation with a verified contributor bundle:

```sh
node scripts/package-training-client.mjs \
  --experiment-bundle=reports/motor-decoder-v1/browser-wasm-001/bundle.json \
  --model-bundle=dist/training-client/fruit-fly-training-client-1f3b0935af59 \
  --output-dir=dist/training-brain-client
```

The frozen bundle supplies pinned model assets, graph files and matching native sources. Current UI and observer sources are packaged separately. Identity, missing-file and checksum mismatches fail packaging; there is no fallback to edited model files.
