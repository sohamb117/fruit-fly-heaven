# Pinned motor-interface causal diagnostic

This is a fixed nine-condition diagnostic in the real closed BANC/FlyBody loop. It does not optimize parameters, submit coordinator results, select a winner, or establish learned flight. The separate coordinator remains the only adaptive training path.

`plan.json` pins the exact active configuration hash, model fingerprint, ordered 18-parameter contract, harness source hashes, and all trial vectors. Every trial uses seed 888 and the complete eight-second landing objective. The full neural/body loop remains active; freezing numerical neural parameters does not freeze neural activity or replay motor output.

The two interventions are physical wing-power gain `{0.75, 1, 1.25}` and deployment-time scale `{0.5, 1, 2}`. They enter their existing log-parameter slots. All other parameter values are zero, so frequency, steering, neural, sensory, leg, probe and grip gains remain one. The baseline `(1, 1)` runs first, followed by all other combinations. No physics, decoder or task criteria are changed by the harness.

Start the dedicated loopback server:

```sh
.venv/bin/python -B scripts/serve-training-safari-check.py \
  --port 7846 \
  --report-dir reports/flight-calibration-v3 \
  --plan reports/flight-calibration-v3/plan.json
```

Open `http://127.0.0.1:7846/test/training-safari-check.html` in Safari and press **Run native check**. Opening the page alone starts no simulation. Pause the separate training contributor before this diagnostic so only one fly is evaluated at a time. The server rejects a plan whose config/model identity no longer matches the checkout; the page also rechecks the config around every trial.

`result.json` is saved throughout the diagnostic. Its `trials` array retains each named job, full result/provenance, runtime checks, and preview frames, including physical failures. Every native evaluation uses the complete horizon unless the real failure criteria stop it. Simulation errors or identity mismatches stop the diagnostic with earlier results retained.

The preview remains 320 × 180 at three frames per wall-clock second. Each trial retains its first 79 frames plus the latest/final frame, at most 80 frames. Omission counters are recorded; these frames are not exhaustive physics-step evidence. The original no-plan single-baseline and pause/cancel/stop checks remain available.

Compare actual clean departure, measured clearance, continuous qualified flight, and preimpact rotation. A less negative crash penalty or longer time before a crash is not by itself evidence of learned flight. There is no automatic promotion in this diagnostic.
