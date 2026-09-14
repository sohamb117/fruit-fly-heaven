# Actual BANC/WASM motor capture

This diagnostic ran the current production training environment with the real BANC WASM neural backend and native MuJoCo WASM body. Safari computer control was unavailable (`cgWindowNotFound`); no browser was replaced with a headless substitute. Node resolves the existing browser asset prefixes to the same checksum-verified workspace files.

Reproduce with `node scripts/capture-training-native.mjs reports/NEW_CAPTURE/result.json`. The output path must not already exist. This is a diagnostic, not a coordinator contribution or training run.

The baseline failed with `excessive_rotation` at 0.136 simulated seconds. All 805 motor-neuron rates were captured for each of its 68 two-millisecond body steps, alongside full initial integration/muscle/interpreter state and exact poststep positions/velocities. The capture uses binary float32/float64 encoding and preserves signed zero.

`result.json` contains the capture at `evaluation.motorReplay`. The complete report is 2.24 MiB. Runtime/model fingerprint: `54d24152ef8cba656445fe166f7a54c2b745d51aa2b1da78b94bdb4fea736e53`. Config hash: `1306df6fd9af1b022ed1c90d47b40c75ab1d65fc534676914c8693f63b32bb63`.

The previous Safari/WebGPU baseline failed at 0.170 seconds, so this is explicitly a different neural-backend realization. Mechanical counterfactuals must use this exact recorded signal sequence and pass a byte-identical native replay gate before drawing conclusions. A successful capture is not successful flight.
