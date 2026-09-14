# Staged DLM intrinsic model — ready for parent review

Everything in this directory is isolated. The live runtime, body, prepared BANC data, configuration, binaries, and deployment were not edited. `dlm-runtime.patch` is a source-only patch against the originals in `source-snapshot.json`; `git apply --check` passed. `staged-manifest.json` records before/after source hashes and the separately built staged WASM artifacts.

## Selection and integration

The public exports `DLM_CELLS`, `DLM_PROFILE` and `intrinsicLayout` are in `packages/banc-runtime/src/index.js`. Selection is explicit:

```js
const selectedModel = {
  ...base,
  manifest: {
    ...base.manifest,
    intrinsic_models: {
      schema: 1,
      profile: 'dlm-snl-2023-v1',
      cells: DLM_CELLS.map(cell => ({...cell})),
      ionic_step_ms: 0.1,
      initial_gates: {h: 0.146, b: 0.146},
      event_policy: 'threshold-10ms-guard',
    },
  },
};
const eventContract = intrinsicLayout(selectedModel).eventContract;
```

Pass this same declaration to `createWingMotorEventReader({...existing, eventContract})` and `createWingEventExcitation({...existing, eventContract})`. Both brain wrappers also expose `.eventContract`. Parent owns environment/body/evaluator glue; none is included in this patch. Construct/select the brain model before enabling the event effector. A separate model identity is necessary for the existing shared-resource cache.

All ten real IDs must join the IO motor-neuron records and both five-unit DLM mappings, including root ID, cell type, region, target and side. The synthetic fixture exemption permits at most 64 cells with explicit `fixtureN` IDs. Unknown options, altered frozen constants, graded DLMs, and incoming DLM gap edges reject. The actual graph is retained; no gap junction or tonic current is fabricated. Changed active model declarations reject, including when requesting a shared graph.

With no `intrinsic_models`, buffer sizes, the original `neural.wgsl`, original `_br_step`, motor event grid, and effector snapshot schema remain unchanged. The optional route uses `neural-dlm.wgsl` and `_br_step_dlm`. The build script copies both shaders and exports both functions.

## Model and state contract

The new equations use V/h/b with five 0.1 ms RK4 substeps per 0.5 ms graph tick. The first five actual receptor conductances are held over that tick and each synaptic `g*(E-V)` is reevaluated at every RK stage. Generic adaptation/reset/refractory are absent for selected cells. Existing external current, gain and hormonal-current terms remain; no paper tonic stimulus is added to BANC. Incoming DLM electrical coupling is deliberately unsupported in this first profile because the actual prepared graph has none.

The fixed parameters are from [Hürkey et al., Nature (2023)](https://doi.org/10.1038/s41586-023-06099-0), [open paper and Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10232364/). Units are mV, ms, nS, pA and pF. Production equation expressions were newly written from these equations; they do not import the separately isolated authors' CC-BY-NC code/reference. This is not a formal clean-room claim. The paper is CC BY 4.0; retain the attribution. Reuse/distribution of the separate [author software](https://zenodo.org/records/7740678) remains subject to its own CC-BY-NC 4.0 license.

The published implementation's `V > -10 mV`, 10 ms detection guard is retained as an event convention. It never clamps or resets voltage and is not claimed to be a biological firing-rate ceiling. Graph delivery is still quantized to 0.5 ms, while recorded DLM event times retain their 0.1 ms substep.

The normal state8/readState9 interfaces remain. DLM state slots 1/2 are zero (no LIF adaptation/physical refractory). Slot 6 reports instantaneous total ionic-channel plus synaptic conductance, not differential input conductance. Slot 7 is the actual external/tonic/hormonal current, without LIF adaptation subtraction. The usual count, filtered rate and release semantics stay intact.

A sparse private tail has four floats per DLM: h, b, integer detector guard, sticky numerical failure. `.readIntrinsicState()` returns an owned copy in selected-cell order. An immutable slot table follows packed parameters. At the real model size this adds 701,604 bytes to packed storage and 160 bytes of private ionic state per brain. GPU packed storage is shared; WASM's existing private packed allocation remains private. The opt-in GPU step reads a 160-byte failure/state tail once per block before returning; default steps add no readback. End-to-end performance has not been measured.

Raw float32 event times accept at most 3 ULP from the correctly rounded canonical decimal value, following the declared WGSL division bound. Neighbouring error regions must not overlap (`6 * maxNeighbourULP < 0.1 ms`). Off-grid values outside that bound and ambiguous large clocks reject. Returned events and snapshots require exact canonical JS `subtick / 10` times. Generic events remain exact half-ms times. The optional runtime refuses advancement to 262,144 ms, before its bounded timestamp contract becomes ambiguous. Counts and per-cell event intervals are checked independently.

Effector snapshots stay schema 1 without the option; the option uses schema 2 and includes the timing contract. DLM count history uses the 10 ms guard and 0.1 ms first possible event. Kernels, excitation priors, force conversion and native body scheduling are unchanged.

## Evidence and limits

- `unit-tests.txt`: **53/53** focused native/model/reader/effector/lifecycle tests pass. `legacy-hash-test.txt` separately passes the existing pre-extension neural trace hash. Actual IO joins are read-only validation, not a whole-network run.
- `integration-001.json`: actual Metal, five-cell-or-smaller integration fixtures. Unselected mixed chemical/gap cells match the separate legacy shader bit-for-bit. Chunking, private h/b, shared graph disposal, stale model rejection and numerical failure isolation pass.
- `numeric-004.json`: final source-pinned quiet 1 s and near-onset 2 s, five-cell fixtures compared with saved **actual author Brian2** output. All original local and per-cell event gates pass. The quiet condition is silent; the near-onset condition has 12 events for each cell, at most one 0.1 ms substep from the author output. Local five-substep errors are <=2.49e-5 mV versus author and <=3.25e-5 mV GPU/WASM; h/b errors are <=2.50e-7.
- `frozen-002.json`: final source-pinned 300 ms/ten-cell frozen incoming conductance replay. All **115 events** match the validated f64 reference at identical per-cell 0.1 ms times on both backends. Local voltage errors are <=2.54e-5 mV versus reference and <=2.77e-5 mV GPU/WASM; gates <=2.99e-7.

Free-running voltage traces are **not bit-identical**. The near-onset case has maximum unaligned GPU/WASM voltage difference 2.783 mV and h difference .04663 near spikes over 2 s. The frozen replay has maximum voltage error about .01025 mV versus f64. These full errors remain in the reports. The original plan required local numerical bounds and free-running finite gates/per-cell event counts/times; it did not require a continuous free-running alignment bound. An extra, overly strict free-alignment diagnostic remains visibly failed; it was not silently relaxed. These results validate the selected cell implementation and timing interface, not whole-fly flight or the biological calibration of incoming BANC drive.

Historical failed attempts are preserved: `numeric-001.json` stopped on the incorrect assumption that GPU division always stores the correctly rounded decimal; `numeric-002.json` also paired near-simultaneous events by global list position. Subsequent reports pair the nth event of each cell, retaining raw traces/errors. No full BANC or body simulation was run for this staging task.

Broad discovery of all preexisting package tests also encountered unrelated untracked `CalibratedWasmMuscles` experiments requiring an inactive `_muscle_step_configured` ABI, and missing staged embodiment test dependencies. Those experiments were not integrated or deleted. Focused tests above cover the changed route and active legacy ABI.

## Apply/build and repeat

From repository root, after parent review:

```sh
git apply --check reports/flight-dlm-runtime/dlm-runtime.patch
git apply reports/flight-dlm-runtime/dlm-runtime.patch
PATH="$PWD/references/emsdk/upstream/emscripten:$PATH" bash packages/banc-runtime/build.sh
```

The staged build is already available at `staged/packages/banc-runtime/dist/`; generated artifacts are not part of the source patch. The parent must explicitly add/pin `cell-models.js` and `neural-dlm.wgsl` in runtime/development bundle manifests and allow the exact new shader URL in native evaluator fetch mapping. Shader selection is a dynamic fetch, so import scanning alone will not discover it. Recompute the model/config/runtime fingerprint after opt-in selection; do not reuse the baseline hash.

Focused tests in staging:

```sh
node --test reports/flight-dlm-runtime/staged/packages/banc-runtime/test/dlm-model.test.mjs reports/flight-dlm-runtime/staged/packages/banc-runtime/test/physiology.test.mjs reports/flight-dlm-runtime/staged/packages/banc-runtime/test/motor-events.test.mjs reports/flight-dlm-runtime/staged/packages/banc-runtime/test/webgpu-lifecycle.test.mjs reports/flight-dlm-runtime/staged/web/test/flybody-wing-event-excitation.test.mjs reports/flight-dlm-runtime/staged/web/test/flybody-wing-event-dlm.test.mjs
```

Native Metal numerical rechecks require the already installed isolated `reports/native-webgpu-tooling` package and immutable reference artifacts. Choose new output filenames; runners refuse overwrite:

```sh
node reports/flight-dlm-runtime/validate-numeric.mjs numeric-new.json
node reports/flight-dlm-runtime/validate-frozen.mjs frozen-new.json
```
