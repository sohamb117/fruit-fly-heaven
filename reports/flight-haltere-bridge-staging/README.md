# Haltere mechanical-current mapper — staged only

`staged/web/banc-haltere.js` adds an explicit opt-in mapper for the **328 currently annotated haltere transducers**: 171 left, 157 right. Every profile cell includes its exact index, root ID, organ, side, type, and orientation. Construction validates the complete set against prepared IDs and sensory annotations; missing, duplicate, mismatched, or wrongly sided rows are rejected. `staged/web/virtual-haltere.js` is a byte-identical copy of the frozen mechanical prototype.

`orientation-prior.json` is a separately labeled **arbitrary, exchangeable orientation prior**. The generator sorts exact root IDs numerically within each organ/side/type, then assigns π/4, 3π/4, 5π/4, and 7π/4 round robin, restarting each group. It uses no graph edges, motor output, rewards, or flight behavior. This is not anatomical directional knowledge.

For each side, the frozen virtual observer consumes that side's existing normalized haltere muscle power, native root-local angular velocity, and the existing wing phase/frequency. Phase advances by `2*pi*frequency*elapsedSeconds` within a held 2 ms body interval; angular velocity and muscle power remain held. The frozen observer's π phase offset and amplitude `(pi/4)*power` remain mechanical priors. Native prepared halteres have no moving joints.

The direct transduction is `I_pA = maxCurrentPa*c/(1+c)`, where `c = max(0, signedBending/scale)` and the side-specific fixed reference is `scale = mass*|neutralCOMLever|²*(pi/4)*(2*pi*236)²`. The default **800 pA is an unmeasured sensitivity prior**, configurable at construction. The mapper returns current directly; it does not request a firing rate or invoke a DC rate-to-current conversion. Baseline oscillatory bending is retained alongside signed Coriolis bending.

## API

```js
const mapper = createHaltereCurrentMapper({
  enabled: true, profile, sensoryManifest, preparedIds, geometries,
  maxCurrentPa: 800,
});
const sample = {
  halterePower: [leftPower, rightPower],
  omegaRootRadS: [x, y, z],
  wingPhaseRadians, wingFrequencyHz, elapsedSeconds: 0.0005,
};
const currentsPa = mapper.currents(sample); // Float32Array(328), .diagnostics
const diagnostics = mapper.writeInto(reusedDenseCurrentVector, sample);
const copiedDenseVector = mapper.apply(baseCurrentVector, sample);
```

`mapper.indices` and `mapper.cells` are immutable and define the output order. `writeInto` validates vector length and computes/validates all 328 finite outputs before replacing only those indices; it neither allocates nor copies a full-neuron vector. Non-haltere values are untouched. `apply` is the copied convenience and attaches `.haltereDiagnostics`.

Missing/null side power yields zero on that side, with `powerStatus: "missing-zeroed"`. Numeric zero is separately recorded as `"present"` and also produces exact zero current. Missing power never borrows wing, steering, or opposite-side drive. Supplied nonfinite/out-of-range power is rejected before target mutation.

## Evidence and reproduction

```sh
node reports/flight-haltere-bridge-staging/generate-profile.mjs
node reports/flight-haltere-bridge-staging/verify.mjs
git apply --check reports/flight-haltere-bridge-staging/bridge.patch
```

The 12 pure-module tests pass: complete identities; deterministic prior generation; direct signed-moment/current equality for every cell; phase sampling; Omega sign sensitivity; per-side power isolation; explicit missing versus zero; pA saturation; unit conversion; atomic writes and preservation of non-haltere values; owned configuration. `result.json` records source/artifact hashes, exact group/orientation counts, test evidence, and four illustrative current samples. These are mechanical mapping samples, not neural or fly simulations.

`bridge.patch` adds only the two modules. No environment, graph, gap junction, reward, native physics, or motor-force changes were made by this task. Mapping tests do not establish useful flight behavior or measured receptor physiology.
