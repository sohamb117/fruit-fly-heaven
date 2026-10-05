# Motor interface calibration before behavioral search

This experiment keeps the BANC v888 neural model fixed while identifying the muscle and body interface. The route remains brain → VNC → motor neurons → muscles → native FlyBody. Goals and scores are observers: they do not inject root forces, reset the fly during an episode, or supply an attitude controller.

The existing hosted v1 run has not been replaced. Its 14-parameter checkpoints retain their original identity and remain unverified. The research configuration is separate: [`configs/training-interface-v2.json`](../configs/training-interface-v2.json). Do not relabel an old checkpoint or regenerate its manifest to make it fit this experiment.

## What changed

[`web/motor-interface.js`](../web/motor-interface.js) defines named physical interface values, rather than applying another global neural gain:

| Control | Scope |
| --- | --- |
| DLM and DVM rate scales | Separate full-excitation reference rates for the two power-muscle groups |
| Steering rate scale | Steering muscles, independently of power muscles and halteres |
| Power-muscle rise/fall times | Per-muscle WASM activation kinetics |
| Deployment time and beating threshold | Opening dynamics and the existing deployment-to-beating gate |
| Continuous opening span | Optional smooth recruitment above the old 0.01 power onset |
| Folded/active servo scales | Torque-command sensitivity, interpolated by actual deployment |
| Front, middle and hind adhesion scales | Claw force capacity, without changing claw MN firing |

These are engineering hypotheses, not measured physiology. Default values preserve legacy output exactly, including the old discontinuous deployment rule when the new span is zero. The native ABI keeps the old muscle and neural exports intact; pre-rebuild golden traces and native-body comparisons check exact default parity. Each profile is fixed before an episode and owned by that body. No per-step parameter fitting or root feedback is introduced.

The initial experiment varies only **hind adhesion** and **continuous opening span**. The other controls are available for targeted identification, not all thrown into one search. Slowing deployment alone was rejected as a general fix by the controlled sweeps. A nonzero opening span fixes the command discontinuity; it does not by itself establish stable high-power flight.

## Reproduce the mechanical assays

The [motor transfer report](../reports/motor-wing-calibration/README.md) records controlled input rates, native restrained/free-body scenes, torques, work, activation, fatigue, target/actual joint motion, root kinematics and aerodynamic forces. The [takeoff report](../reports/takeoff-calibration/README.md) records matched actual-fruit replays and claw/wing ablations.

```sh
node --test scripts/test-motor-wing-transfer.mjs
node scripts/audit-motor-wing-transfer.mjs --pilot=true
node scripts/audit-motor-wing-transfer.mjs
node scripts/audit-takeoff-coordination.mjs
```

Recorded-input replay checks source identities. A replay from an older build must fail after executable sources change; do not remove its hash guards to turn it into current evidence. Reports include the hashes under which their measurements were made. Instrumentation passing means the measurement protocol worked, not that the body was stable or biologically calibrated.

## Run independent BANC evaluations

Use the local static development server. The harness creates an inert observer page and calls the real training worker with an experiment-specific config. It blocks coordinator API calls, runs one fly at a time, retains a low-resolution 3D preview, and records matched baseline/candidate results. This is an operator validation tool, not an alternate contributor mode.

```sh
uv run --offline python scripts/serve.py --port 7842
```

In a second terminal:

```sh
node scripts/prepare-training-manifest.mjs --config=configs/training-interface-v2.json
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
  node scripts/run-interface-experiment.mjs \
  --candidates='[{"name":"hind_025","parameters":{"hindGripScale":0.25}},{"name":"smooth_hind_025","parameters":{"hindGripScale":0.25,"deploymentPowerSpan":0.1}}]'
```

`PLAYWRIGHT_MODULE` is optional if Playwright resolves normally. `CHROME_PATH` can select a local Chromium executable; the default is macOS Google Chrome. `--site=http://127.0.0.1:7843` changes the development port. `--headed=true` shows the live observer. The default report directory is `reports/interface-training-validation`; JSON results, compressed native frames and final screenshots are retained there. The default held-out seeds come from the experiment config; final test seeds remain unused. Changing the model during evaluation fails asset/config verification.

The worker records actual backend, model/config identities, parameters, initial condition, exact physical/neural time, neural activity, outcome and wall time. `passed` on the harness report means the real experiment completed correctly. Behavioral success and promotion are separate fields; a successfully measured failure is still a failure of the fly.

## Behavior and promotion gates

V2 posture requires a seeded angular disturbance, recovery within one simulated second, and a final 0.75-second interval with at least three feet supporting 80% of bodyweight, bounded translational speed and bounded rotation. Contact forces are transformed from native contact frames; a wing or thorax collision cannot count as supporting feet.

Flight requires a powered upward departure after measured foot support and a full continuous second of upright, bounded-RMS-rotation flight with kinematic support and descent limits. Alternating signed angular velocities cannot average into a pass. Landing requires qualified prior flight, a slow approach, and sustained loaded-foot contact on food. Feeding requires actual new intake while probing. The sequence checks localization → approach → landing → probing → feeding → takeoff → flight in order. All stages must survive their full horizon; transient successes do not terminate them early.

These thresholds are explicit research criteria, not biological measurements. Localization is still the existing odor/heading proxy; vision remains off. An isolated v2 landing episode is intentionally unavailable until a separately validated flight-start fixture exists. The sequence starts on the ground and cannot claim landing from a cold airborne reset.

`createOperatorPromotionGate()` accepts evidence registered by the independent operator process. It checks a matched mechanical protocol and candidate identity, separate held-out seeds, every required prior stage, and a positive improvement over a matched baseline. Anonymous JSON claiming `trustedExecution: true` is insufficient. Passing this gate still does not establish biological validity.

The current runner explicitly supplies a failed mechanical gate: the assays have not validated the actuation parameters against independent physical constraints, and the tested candidates also fail full neural behavior. Open-loop tumbling alone does not prove the mechanics are invalid; a balanced wingbeat may still require neural feedback for stability. The gate cannot auto-promote a candidate merely because a task return improves. Contributor/coordinator and deployment paths reject the v2 experiment until an explicit validated release workflow is implemented. The canonical hosted v1 config is left unchanged; executable edits in this working checkout require the separate experiment manifest and must not be submitted to the existing hosted run.
