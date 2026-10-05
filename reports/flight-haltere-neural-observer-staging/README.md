# Haltere neural observer diagnostic

Ready for the parent-owned exact replay check. Nine fixture tests pass; [validation.json](validation.json) records source hashes. This helper is outside the live application and changes no source, neural tick, motor signal or physics.

```js
import {installHaltereNeuralObserver} from './haltere-neural-observer.mjs';
const observer = installHaltereNeuralObserver({
  BrainClass: WebGPUBrain,
  neuronCount: model.manifest.neuron_count,
  wingIndices,           // exactly 48 sorted, explicit identities
  haltereIndices,        // exactly 328 unique, explicit identities
  haltereMotorIndices: [97021, 118683],
  onSample({timeMs, stride, state, indices, haltereIndices, haltereMotorIndices}) {
    // Synchronously persist/copy diagnostics. All supplied arrays are owned.
    // state is 330 x 9: 328 sensory rows, then the two hDVM motor rows.
  },
});
try {
  // Existing evaluation, including its existing wing readState calls.
} finally {
  observer.restore();
}
```

Only a caller Uint32Array whose contents exactly equal the 48 sorted wing indices, with `includeSpikeTime === true`, is expanded. The original readState executes once with 48 + 330 distinct indices. The returned wing prefix preserves its Float32 bytes and the exact original property descriptors/values for `totalSpikes`, `activeEver`, and `spikes`. The global spike-history object is returned unchanged and is not passed to the observer. Unmatched calls preserve even the original promise identity.

The extra state is 11,880 bytes per matching read. The existing gather/map and global event-history read are reused; there is no second GPU call or wait. The normal two-cache pattern becomes 805 x 8 motor rows and 378 x 9 event/diagnostic rows. This does increase gather/output size and callback copying, so it is diagnostic overhead rather than a free observation.

Inputs are copied and validated before installation: index counts, integer/range bounds, sorted wing selection, disjoint identity sets, exact hDVM identities, and a replaceable own prototype method. The helper rejects duplicate installation and a mismatched model count. Restore is idempotent and reinstates the exact original method descriptor. If another owner replaces the method during installation, restore reports that conflict rather than erasing that owner's change. Restoring while a read is pending suppresses its callback but still returns the correctly sliced wing result.

Callback errors propagate visibly; restore remains the caller's responsibility in `finally`. Callbacks must be synchronous and receive no brain handle. Tests cover metadata/byte parity (including negative zero and a NaN payload), unchanged unmatched calls, owned arrays, source identity mutation, restoration, pending reads, and errors. These tests do not prove real GPU/physics nonperturbation: the parent will compare complete event history and physics digests with the unobserved run.
