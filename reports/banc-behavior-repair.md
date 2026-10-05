# Why the fly appeared inactive

The prior default ran 100 full brains with bodies tied to the slowest neural clock. Even a one-fly diagnostic showed only 0.069 body lengths of sampled travel over 1.2 simulated seconds. The fly began on fruit and transferred food, but never became airborne.

The muscle model also made takeoff impossible at the default hungry state: its force multiplier was 0.545, giving at most 713 cm/s² of wing lift against 981 cm/s² gravity. The earlier actuator test used a fully fed body and missed this failure.

The repaired default starts one fly while retaining the original population controls and saved selections. Muscle force is now fuel-limited near depletion, with an explicit assumed reserve threshold, and initial foot support uses the actual terrain. Selected-neuron traces retain every timestep with one GPU readback per block. Startup settling and tiny on-food shuffles no longer count as landing or approach.

In the corrected direct-mode run, the original on-fruit pose produced proboscis contact and a small food transfer, takeoff at approximately 36 ms of body time, and 150 ms of continuous flight by approximately 186 ms. No movement commands were injected into motor neurons or supplied by a behavior program. However, sustained wing activity carried the fly into the ceiling and arena boundary. This is movement through the neural/muscle path, **not completion of the behavioral target**. Food localization and controlled landing remain unresolved, and the tiny intake is not a completed meal or satiety response.

Disconnecting motor coupling at 520 ms of neural time removed wing force and returned the body to the bowl while neurons continued firing. At 1272 ms, wing force was below 1e-8, the body was supported and motor-neuron rates remained active. Passive momentum and joint relaxation persisted briefly after disconnection.

Validation: 21 runtime/mechanics tests, GPU/WASM per-timestep trace parity, the original console UI test, and 36 legacy assertions passed. The full legacy runner needed `--test-force-exit` because the existing circuit-diagnostics test sometimes kept its process alive after all assertions completed; its isolated rerun passed without that flag.

Evidence: [before](banc-behavior-before.json), [after](banc-behavior-after.json), [motor disconnection and source hashes](banc-behavior-disconnection.json), [summary](banc-behavior-repair.json). Before/after observation windows differ and are not paired performance benchmarks.

Reproduce with the local server running:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --ms=500 --disconnect-ms=750
```
