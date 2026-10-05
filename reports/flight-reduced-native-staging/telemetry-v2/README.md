# Telemetry v2, staged only

Fixes the reduced-model ready-preview failure without changing simulation control. The strict `fixed-nonwing-flight-v1` model retains six claw muscle readouts; native adhesion control, force and gain are `null`, with `claws.nativeActuation.available=false` and reason `structurally-absent`. Missing claws on an unrecognized/full model still fail. No actuator-zero fallback is used.

The optional `activation-amplitude-v1` power readout now uses the exact decoder expression `powerGain * clamp(drive * activationGain)`. Legacy absent-profile output remains deep-equal, including ready frames and nonfinite reporting. The new saturated 0.90 amplitude is reported as 0.90, rather than the previous incorrect raw-force product.

Eleven pure tests pass: seven existing telemetry fixtures plus legacy parity, reduced absence/identity, malformed-layout refusal, and exact request/effective-power comparisons on both model variants. No native/brain steps, server, live source edits or optimizer work. The prior reduced allocation failed before physics; this stage does not claim a completed reduced run.

`build.mjs` creates immutable staged/original copies and exact substitutions. `validate.mjs` checks reversible changes and runs the fixtures; output paths refuse overwrites. `source-pins.json` and `validation.json` commit the evidence.
