# Telemetry v2 diagnostic plans

Prepared only. No native evaluation, server, optimizer, checkpoint or live source change was performed.

The three plans preserve their version 1 model, reset, scorer, native wing decoder, neural settings and all 27 job parameters. The only changed source asset is flight telemetry: strict reduced-body support explicitly marks absent native claw actuation while retaining the six muscle readouts; requested power reflects the activation-amplitude-v1 decoder. Full-body physics and motor history must still match the pinned reference. Preview telemetry values may legitimately differ.

The version 1 reduced allocation failed in ready-preview before physics on a missing claw actuator. It did not produce a completed reduced simulation. Re-run the full-adapter-equivalence gate before either reduced arm. These are diagnostics, not optimizer updates or saved training checkpoints.

Reproduce preparation into an absent plans-v2 directory:

`node reports/flight-reduced-native-staging/prepare-plans-v2.mjs --telemetry-sha256 6ba0613840fda4259a7f9b620f6ac3526b076271c354cccee4c7aaf109c6393b --validation-sha256 268f3a4a1e045d3ca0d787ab21c684b19676368be5219219e13eafcb888dda3b`

All prior artifacts remain unchanged. See ledger.json and verification.json for exact commitments.
