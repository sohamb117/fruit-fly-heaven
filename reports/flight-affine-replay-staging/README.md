# Frozen event-aware motor replay staging

**Prepared and frozen for the parent-owned native baseline run.** Source/data preflight and three pure fixture tests pass. This agent has not initialized MuJoCo, run a body or neural model, or changed any live source. A native baseline result is still required before any affine comparison.

The capture is [the current 500 ms event/native replay record](../flight-affine-source-capture/evaluation/000-current-event-transfer-capture-28388a1f-ae4b-4a25-8295-2b1e852a2258.json), SHA-256 `ee736149551da39bd824e6988a7930badae37c026eb41d9ab8add36e5f80805e`. It contains 250 complete 805-MN rate vectors, a separate initialized event-adapter snapshot, exact 48-unit event packets, and 251 full native muscle samples. [prepare-validation.json](prepare-validation.json) pins these scripts and records preparation evidence.

Preflight only (default; native modules are not imported):

```sh
node reports/flight-affine-replay-staging/replay-event-motor.mjs --capture reports/flight-affine-source-capture/evaluation/000-current-event-transfer-capture-28388a1f-ae4b-4a25-8295-2b1e852a2258.json
```

Parent-authorized native baseline:

```sh
node reports/flight-affine-replay-staging/replay-event-motor.mjs --capture reports/flight-affine-source-capture/evaluation/000-current-event-transfer-capture-28388a1f-ae4b-4a25-8295-2b1e852a2258.json --execute --output reports/flight-affine-replay-staging/baseline-result.json
```

The result path must be new and inside `reports/`. No counterfactual or approximation fallback is implemented. Errors preserve a failed result and exit nonzero. The companion `run-<runId>.json` supplies exact config text; `--run` can specify it explicitly. Twenty-four mechanics/module/WASM/io dependencies must match both captured and configured hashes before execution and after a successful replay. Embedded scene XML, exact heightfield, metadata and food-contact resolver reconstruct the captured native world.

## Restore and scheduling contract

1. Construct the native body and allocate event muscles while the body remains fresh. Enable with the exact snapshot contract config and its ionic event contract.
2. Restore the full native integration vector, including solver warmstart/equality state, around one initial forward to rebuild caches. Restore packed full-muscle WASM state, public muscle state/input, activation, rest posture, internal/monitor state, wing interpolation state and wrapper values.
3. Restore the already-initialized event-adapter snapshot and separate native 28-muscle state/input. Verify restored packed state bytes and adapter equality. **Do not reaccept packet0.**
4. Feed the next recorded event packet immediately before each 2 ms body call. Keep 0.1 ms ionic timestamps verbatim. The existing body implementation retains its two causal 1 ms muscle boundaries and held force semantics.
5. Keep all 805 recorded rates fixed; nonwing routes use those rates, while wing routes use the exact captured event packets. Packet rates are cross-checked against the corresponding 805-MN readout. Food targets are restored from each captured block.

Every completed block must match qpos and qvel **byte for byte**. All full-muscle activation/fatigue/force values, raw wing drive, mechanical wing power, wing phase and internal-state fields must match exactly. Float32 force samples were saved as JSON arrays, so their numeric gate deliberately cannot distinguish +0 from -0; it has no magnitude tolerance. Authoritative packed native integration, initial muscle WASM/readback state and qpos/qvel do preserve signed zero. Fresh initial event input/public-state arrays are checked as exact finite Float32 values.

The replay also checks the restored initial objective observation and records COM/support observations after each matched block. It adds no root force, reward/controller action, neural rerun, sensory feedback adaptation, or optimizer. Recorded trajectories remain frozen at the motor interface; later affine comparisons would therefore be mechanical counterfactuals, not live closed-loop flight.

Pure fixtures cover packed Float32/64 comparisons including signed zero and adjacent representable values; invalid intervals, missing baseline/force samples and mismatched rates; and packet0 omission plus preservation of a 1.7 ms ionic event. These validate parser/control flow only, not native replay parity.
