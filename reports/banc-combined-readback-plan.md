# Bounded combined GPU readback plan

Status: design only. No production files changed and no browser workload run. Apply only after the matched benchmark finishes and only if its result justifies an optimization. This plan preserves neural mathematics, neural dt, delayed history, sensory sampling, every selected-neuron trace point and the 2 ms neural/body handshake.

## Current cost and scope

Each selected BANC block calls `step(..., {traceIndex})`, awaits a trace staging map, then calls `readState(...)`, submits another command buffer and awaits another staging map. At dt=0.5 ms the block contains four neural dispatches and four 32-byte trace copies. At dt=1 ms it contains two. Full inspection reads 6,314,436 state bytes, 5,488 aggregate bytes and 131,080 event bytes for 175,401 neurons; compact motor readback is 53,136 bytes. The proposed first change retains those payloads and shaders. It coalesces the readbacks; compacting full inspection is a separate future change.

Current `webgpu.js` SHA256 e076a35780bfba9a0edf5afda548bd25ca6d8877589c7107155f84042715e431 matches every available observation source manifest since 09:44 UTC. Current worker SHA256 0fe21f63e647411f0ada283d0e38c13a32ad8749eed9671970f6dc5875d957f5 matches every load since 09:59 UTC. The newest app change is only precision labeling. No new full-state hot-path regression was established.

## API and files

Add an additive API to `packages/banc-runtime/src/webgpu.js`:

```js
await brain.stepAndRead(steps, input, internal, gaps, {
  traceIndex: selectedIndexOrNull,
  indices: readIdsOrAllIds,
  includeSpikeTime: falseOrTrue,
}); // -> { trace: Float32Array | undefined, state: Float32Array }
```

`state` must retain exactly the existing readState field layout and its `totalSpikes`, `activeEver` and optional `spikes` properties. `trace` must retain exactly the existing steps×8 values. Keep existing `step` and `readState` public behavior available. The worker uses its existing sequential step/read fallback on WASM; a WASM/kernel change is unnecessary.

Factor the current read-cache acquisition, command encoding and mapped-result decoding into private helpers shared with readState. Reuse the two existing read caches. Each staging buffer reserves an additional 4,096 bytes (128 steps × 8 floats) for traces. Let `traceOffset = align8(stateAndAggregateBytes + eventBytes)` and reserve enough storage for that offset plus the full 4,096-byte capacity. The extra per-brain staging capacity is approximately 8 KiB across both caches.

In one encoder, preserve every existing neural dispatch and its state-parity trace copy. Copy each trace to `cache.staging` at `traceOffset + k*32`; the destination remains MAP_READ|COPY_DST. Append the unchanged readout dispatch using parity `(startingTick + steps) % 2`, then copy state/aggregates/events into their original staging offsets. Submit once, map once, decode both outputs into independent owned arrays and unmap. Set tick only after successful completion, as in the existing implementation. Keep validation before submission, busy ownership throughout the operation, and finally-block unmapping/error cleanup. Do not change spike-ring writes, copy order relative to neural dispatches, or arithmetic.

## Worker integration and inspection timing

Refactor `inspect` so it can summarize an already supplied state. Separate the supplied state's layout (`dense/full`, stride 8/9) from whether an inspection should be emitted; selection may change during an awaited operation.

At block start, choose the read layout with the existing selected-fly/requested-inspection/100 ms wall deadline predicate. Invoke stepAndRead with the exact current trace selection. After it resolves, recheck the current selection, request and deadline predicate. If a full inspection became due while the GPU was working but only compact state was requested, obtain the full readState at the same completed neural time before continuing. This rare fallback preserves inspection readiness and avoids postponing a request or deadline by another neural/body block. It uses two readbacks, as the current path does. If selection changed away, summarize the supplied layout for motor output but do not emit obsolete inspection data.

Retain the existing trace-index/selected-fly equality checks before appending trace points. A new inspection request arriving during an await must remain pending unless that exact request was served; use a captured request serial if necessary. Pause must still finish the started cohort and honor the existing body barrier. No additional neural step may occur between the combined read and body acknowledgement.

The optimization is strongest when full inspection is already due at block start or only compact output is needed. If neural execution alone crosses the 100 ms deadline on every block, the fallback may eliminate little overhead. This is a measurement question, not an assumed speedup.

## Required parity and safety checks

1. Use two independently stateful GPU brains sharing immutable graph buffers. Compare old step+readState with stepAndRead under the same fixed input/current/internal-state replay. Cover both dt modes, step counts 1/2/4/127/128, both final parities, first/last trace neuron, compact/reordered/full indices, and spike-time readout on/off.
2. Require bit-identical returned state, selected trace, motor rates, spike counts and last-spike times. Compare sorted spike events while below ring capacity; add a deterministic single-workgroup ring-wrap fixture. Large overlapping ring reservations can already differ in tie ordering and must not be mistaken for new neural-state differences.
3. Alternate compact/full caches repeatedly; verify old result arrays are unchanged after subsequent calls and mapped buffers are always unmapped. Check invalid indices/input/steps, concurrent operation rejection, disposal and device-loss errors without ticking or leaking resources.
4. Extend the isolated worker scheduling test for late selection/neuron changes, a deadline crossing while awaiting the combined call, pause during a cohort, paused inspection and body acknowledgement. Every motor/body update remains one completed 2 ms neural block, with every selected trace timestep retained.
5. Run a short fixed-input full-BANC comparison after the small fixtures. Then run the one-fly full-console UI/runtime checks and the same matched benchmark. Record queue submissions/maps separately: the common path should fall from two to one per block. Report observed throughput and variability; do not infer speedup from fewer maps alone.

Do not combine this change with neural shader rewrites, fewer contacts, reduced sensory cadence, altered precision, hidden body assistance or disabled inspection. Keep full-state packing reduction and other allocation work separate so any performance or parity change remains attributable.
