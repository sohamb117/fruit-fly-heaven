# Frozen power-only ablation

Prepared for root-owned execution; **no native run performed during staging**.
Four pure tests pass and the recorded-input preflight passes. The baseline is
the existing exact 250-step replay of the same 500 ms capture, not an approximate
or newly generated baseline. See [preflight.json](preflight.json) and
[source-pins.json](source-pins.json).

The sole transfer change is:

```
requested_left  = clamp(raw_left  * retainedPowerGain * 0.516569885701581, 0, 1)
requested_right = clamp(raw_right * retainedPowerGain * 0.4913795054978524, 0, 1)
```

The existing deployment ramp still gates actual power. Zero raw power remains
zero. Steering retains the exact legacy iteration order and arithmetic
`force * basis * retainedGain`; neither T nor F0 enters steering. The retained
27-parameter interpreter is not tuned or replaced. The staged wing class is a
copy of the pinned legacy source with exactly two textual changes: its relative
import path and requested-power expression. Only its `step` is installed on the
already restored wings instance; no body state is reconstructed by that install.

The scales come from the accepted measured profile:
`commonTrimPower / (recordedSideRawPowerMean * retainedPowerGain)`.
Preflight verifies the profile, provenance, accepted/repeated calibration,
source-capture hashes, retained interpreter and that exact derivation. The
profile's `trim` and `reference` fields are **not applied**. Power measured with
the restrained unit-gain steering trim does not imply one body weight when
combined with this replay's different legacy steering waveform.

All 805 rate values and 48 exact event streams are frozen for 250 × 2 ms blocks.
The existing helpers restore the full initial native integration state and both
muscle kernels. A separate pure event adapter checks event-kernel identity;
actual 28 wing activation/fatigue/force values are compared with the saved
baseline. They are not forced equal. The native-write guard rejects pose or
velocity edits between `mj_step` calls, forward/reset/state calls after restore,
and external applied forces. No controller, BANC execution, neural feedback,
optimizer, arena change or root correction is added.

Every executed wing update audits both sides' requested and actual power,
including clipping, and verifies the actual steering residual arrays against
the exact legacy sum. The result uses `powerOnlyAudit` and explicit
`affineTrimApplied:false` / `forceReferenceApplied:false` fields. It never reports
affine force coordinates as applied. The unchanged scorer retains the original
objective horizon; 500 ms is only the captured prefix. A score can terminate
while the mechanical replay continues through the available frozen inputs.

Pure validation covered source-copy identity, unit-scale byte parity over 2500
varied updates, measured-scale parity against independently precomputed
requested powers over another 2500 updates, untouched steering inputs, execution
audit rejection, scale ownership, malformed inputs, incompatible mode rejection,
and restoration of the original method. These are numerical wing-interpreter
fixtures, not body or native-muscle simulations.

```sh
node --test reports/flight-affine-power-ablation-staging/power-only.test.mjs
node reports/flight-affine-power-ablation-staging/replay-power-only.mjs --capture reports/flight-affine-source-capture/evaluation/000-current-event-transfer-capture-28388a1f-ae4b-4a25-8295-2b1e852a2258.json --operating-point reports/flight-affine-transfer-staging/measured-profile/profile.json
```

After root review, one-body execution:

```sh
node reports/flight-affine-power-ablation-staging/replay-power-only.mjs --capture reports/flight-affine-source-capture/evaluation/000-current-event-transfer-capture-28388a1f-ae4b-4a25-8295-2b1e852a2258.json --operating-point reports/flight-affine-transfer-staging/measured-profile/profile.json --execute --output reports/flight-affine-power-ablation-staging/result.json
```

Existing output files are rejected. Executable sources are frozen after the
passing preflight. No live runtime source or prepared canonical data changed.
