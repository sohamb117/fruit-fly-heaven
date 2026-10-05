# Staged haltere environment scheduling review

**No blocking scheduling or data-handoff defect found.** The focused fixture suite passes **10/10**. Reviewed source hashes and exact command are in [scheduling-validation.json](scheduling-validation.json). Live environment bytes matched the reviewed staged environment when this evidence was saved. Source edits were limited to this report directory.

Run:

```sh
node --test reports/flight-haltere-environment-staging/environment-scheduling.test.mjs
```

The test harness imports the actual original/staged environment, actual staged haltere current mapper and mechanical helper, and actual motor event reader. Neural state, native body and 805-cell identity fixtures are deliberately synthetic. It does not run BANC, WASM physics or a GPU. Pinned native-derived geometry is read as static data. This establishes integration behavior, not neural response or flight stabilization.

## Verified handoff

- With both additions disabled, current vectors, neural readouts, event-free body inputs and operation order match the original environment fixture exactly.
- Sequence-only mode produces the same four tick currents as the old held-input step. Four distinct dense buffers are reused only after the awaited delivery finishes. The default 805 × 8 motor read remains unchanged.
- Enabled mode samples the block-start root-local angular velocity, existing wing phase/frequency and same-side haltere power. The four current vectors correspond to offsets 0, 0.5, 1.0 and 1.5 ms. Current computation does not advance the real wing oscillator or native body.
- The next block uses the updated body state. All 328 selected haltere entries replace legacy rate-derived current, including zero-power entries; unrelated sensory and zero-input entries remain unchanged. Tests compare each selected Float32 value to the real bridge output at the declared phase.
- Zero initial power gives zero current despite nonzero legacy input. One active side does not recruit the missing side. A changed interpreter frequency reaches the transducer directly.
- The four neural ticks precede the existing motor/event read. The body consumes its event packet before the next native update. Baseline and body-time packet conventions remain unchanged.
- Invalid sequence flags and corrupted identities reject before allocation. Missing sequence support and invalid later-side power reject before physics and release the allocated brain/world.

## Sampling and model interpretation

This is causal block sampling: at body time t, the available body angular velocity and muscle power are held while a prescribed virtual phase advances through the next 2 ms neural block. It does not sample actual within-block body recoil or update the physical body four times. Consequently, 0.5 ms input delivery improves representation of the modeled oscillation but does not increase the mechanical feedback sampling rate beyond 500 Hz. At the approximately 236 Hz carrier, the neural waveform has about 8.5 samples per cycle; the held body context has about 2.1 updates per cycle.

The virtual phase is tied to the existing target-table oscillator with the helper's fixed pi offset. This is a declared phase prior, not measured haltere motion or wing/haltere phase. Same-side asynchronous haltere muscle force controls amplitude; zero force is intentionally zero even if wings are powered. Initial native haltere power is zero, so this pathway has no initial output until the motor/muscle route recruits it.

The geometry, oscillation axis, fixed beam frame, per-cell orientations and 800 pA sensitivity are explicit modeling priors. Currents preserve signed mechanical projections before compression, but neither the arbitrary per-type orientation assignment nor the network's corrective response is established by these tests. The point-mass model holds its amplitude envelope within each block and omits envelope derivatives and additional inertial terms by design.

Frame diagnostics record the **last input sample** of the previous neural block (offset 1.5 ms), its source body time, source phase and held angular velocity. At a post-physics frame that source is intentionally older than frame time; it is not the instantaneous current at the frame boundary. The diagnostic therefore must not be interpreted as a complete per-tick current trace.

The actual A/B/C experiment remains necessary: legacy versus sequence-only establishes numerical compatibility on the real neural backend; the phasic-current arm measures behavior of the new declared input model. No flight claim follows from this fixture suite.
