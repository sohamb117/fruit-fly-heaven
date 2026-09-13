# Browser training and shared contributions — 2026-09-13

The implemented training path ran real BANC v888 neural dynamics and native FlyBody/MuJoCo in a browser worker. It did not replace the brain with a scripted locomotion controller. The new `/train.html` page keeps a low-resolution observer preview separate from simulation fidelity.

The [integration record](result.json) contains a complete local ES generation: two baseline selection evaluations, eight perturbation evaluations, and two candidate selection evaluations. All twelve completed. The selected mean return improved from **−5.011 to −4.935**, but both selected-candidate evaluations failed the posture goal. The parameter update and exported checkpoint are real; this is not evidence of reliable behavior.

Two independent Chromium browser profiles then ran four real shared episodes each, sequentially on this computer. Eight accepted returns produced coordinator generation one and a changed parameter vector. The profiles had distinct contributor identities. This verifies the multi-client protocol on one machine, not a deployment across physical computers or an internet-hosted service. One exploration episode passed the one-second posture criterion. The shared aggregate remained unverified.

The [separate checkpoint evaluation](../training-independent-checkpoint/result.json) imported that shared candidate and ran all three configured test seeds. **Zero of three passed**; all terminated for excessive rotation, with mean return **−4.793333**. No selection or update used those test returns. Autonomous localization, approach, landing, feeding and flight remain unestablished.

## Interface and lifecycle evidence

- Real native pose frames drove the 320 × 180 preview at up to 3 fps. [Shared-client screenshot](contributor-1.png).
- Pause held the preview/body time steady; Resume continued. Stop released worker memory. Checkpoint download/import and reload restored the selected vector without starting compute.
- [Idle desktop/mobile checks](../training-ui-idle/result.json) covered 1440, 390 and 320 px widths, consent controls and zero workers, model requests or WebGL contexts before Start.
- Visual inspection found the SVG reward chart still hidden after results; this was fixed with reflected attribute updates. The browser regression checks empty → populated → empty history. The [recorded local-history replay](../training-ui-idle/recorded-local-history.png) displays the twelve actual local results; its source hash is recorded in the idle report. The earlier `local-training.png` capture predates that display fix.
- Current client regression checks cover local/shared state separation, stale Stop/restart responses, mismatched worker provenance, cancelled evaluations, retired workers, heartbeat ownership, checkpoint evidence and validation badges.

All **35 JavaScript tests and 25 coordinator tests** passed. Coordinator tests use synthetic objectives only for infrastructure/algebra and separately exercise real HTTP/CORS and SQLite restart behavior. They are distinct from the 23 real scored episodes above and the [native worker smoke](../training-worker-smoke/README.md).

The self-contained contributor ZIP was also tested outside the checkout's serving layout. Its standalone Python server served every payload with matching hashes, then the [bundle runtime smoke](../training-bundle-smoke/result.json) loaded the actual WebGPU BANC model and native body, advanced 200 ms with 419,301 neural spikes, and passed pause/cancel/stop checks. It had no browser errors. The ZIP contains 71 entries, is 58,854,351 bytes, and has SHA256 `a7ff53db00c61d9d3c7d77f683456f07af14d2143b1ac9b1cca5542726e10b66`. This confirms local bundle execution; a public shared coordinator still requires deployment.

## Runtime and provenance

All real browser runs reported `webgpu` for BANC and `mujoco-wasm` for the body. Neural steps were 0.5 ms; neural/body feedback exchanged every 2 ms. Local training advanced 3.456 simulated seconds in 92.695 wall seconds including initialization, pause and orchestration (~0.037× real time). This is an observed run, not a controlled performance comparison. The preview rate is not the physical simulation rate. Safari was opened for the user; automated compute and visual checks used Chromium.

The model fingerprint was `8bfd755893d540ed11e249c36ad7ae73d6a955c78bc57a3899ad18cc8c2bee13` and exact config hash was `1bad7d5c80d5129dcbe4f94b66fc725ec343053fb50bf70769c4a8e75510d00e`. The 39-asset environment manifest stayed fixed throughout these runs; the prepared graph manifest transitively pins the graph buffers. UI/client lifecycle fixes do not change the neural/body environment fingerprint.

Reproduce with the local site and coordinator running, then set `PLAYWRIGHT_MODULE` to an installed Playwright module and run `node scripts/verify-training-ui.mjs` and `node scripts/verify-training-evaluation.mjs`. These execute real computation and append accepted jobs to the selected coordinator; use a separate database for an isolated reproduction. See [setup and deployment](../../docs/training.md).
