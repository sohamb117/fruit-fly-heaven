# Frozen affine transfer counterfactual

Prepared for source review and a later root-owned run using a measured operating-point file. Four pure fixture tests pass. Data/source preflight passes against the real 500 ms capture and its **passed 250-step exact native baseline**, using a schema-only profile fixture. No native counterfactual has been run by this agent, and no live sources were edited.

```sh
node reports/flight-affine-replay-staging/replay-affine-candidate.mjs --capture reports/flight-affine-source-capture/evaluation/000-current-event-transfer-capture-28388a1f-ae4b-4a25-8295-2b1e852a2258.json --operating-point MEASURED_PROFILE.json
```

Default is preflight. Parent-authorized execution requires `--execute --output reports/NEW_RESULT.json`. `--baseline` can identify the passed baseline explicitly; it defaults to this directory's `baseline-result.json`. Existing output files cannot be overwritten. `fixture-operating-point.json` is only an unexecuted input-schema fixture and must not be represented as a measured calibration.

The entrypoint requires matching capture SHA-256, config/model fingerprints, source hashes, baseline script/helper hashes, and all 250 passed baseline steps. It pins all three affine transfer files (`build.mjs`, `flybody-wings.mjs`, `install.mjs`), the counterfactual helper and the unchanged scorer, plus the 24 captured mechanics dependencies. There is no approximate baseline bypass.

## Intervention and audit

Only after exact initial native-state restoration, full initial muscle-state checks and exact initial-observation matching does it call `installFlightOperatingPoint` on the existing wings instance. This is a **combined static trim and power remapping**. No optimizer, desired body trajectory, root correction or online controller is added.

The 805 motor rates and 48 exact event packets remain frozen. All neuron identities, each packet's identity hash and every rate vector's binary hash are saved. A second pure event adapter verifies identical event-excitation kernel state at every 2 ms boundary; it does not drive the body. The real event-muscle WASM state must match its public readback and single full-muscle merge; hidden full-kernel wing rows must remain silent.

The 28 actual wing forces are compared with the recorded baseline at every 2 ms boundary. Force, activation and fatigue differences are reported, not rejected. Fuel may change through internal-state or ingestion feedback; the saved record therefore includes actual and baseline internal state. No equivalence claim is made merely because motor inputs are frozen.

A native-call guard permits pose/velocity changes only during `mj_step`; it checks exact qpos/qvel values between calls and rejects `mj_forward`, reset, `mj_setState` and split-step calls after restoration. It also rejects nonzero externally applied forces. This checks boundary writes in the pinned code; it is not an adversarial memory-write monitor. No qpos/qvel reset is used to keep the candidate stable.

Every wing update records the min/max of **T + gain × (F − F0)** for each side, muscle type and bias/amplitude coordinate, plus counts below zero or above one. These are wing-shape basis coordinates, not negative native muscle forces. No coordinate clamp is added. Existing shape, joint-target and servo clamps remain in the pinned transfer implementation.

## Outcomes

Per-block output includes COM position/height, up vector projection, angular speed, vertical speed, support/contact measures, native wing forces and internal state. The same scorer evaluates both the saved baseline observations and the candidate, retaining the original 8 s objective horizon. The 500 ms replay is a censored prefix, not a new shortened task. If the scorer fails early, that score stays terminal while the mechanical diagnostic continues through the available motor stream unless native execution fails; partial reports retain completed samples and coordinate ranges.

The physical candidate is expected to diverge from the baseline trajectory, so candidate qpos/qvel equality is not required. Success or failure under frozen inputs does not establish the behavior of a reconnected live neural feedback loop.

Preparation evidence and source hashes: [counterfactual-prepare-validation.json](counterfactual-prepare-validation.json). Transfer/scorer pins: [counterfactual-source-pins.json](counterfactual-source-pins.json).
