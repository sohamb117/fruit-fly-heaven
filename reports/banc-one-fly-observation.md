# Observed behavior of one BANC fly

**The behavioral target is not met.** Two trials ran sequentially, with exactly one fly, the original 3D console, actual rendered eyes, Reference physiology, direct motor-to-muscle coupling and the neural body clock. No behavior controller or actuator commands were added during either observation.

| Initial condition | Observed neural time | What happened |
|---|---:|---|
| Original pose on fruit | 6.006 s | Brief proboscis contact and a tiny food transfer, then takeoff at about 36 ms of body time. The fly reached the ceiling by the 180 ms sample and the wall by 400 ms. It stayed against the boundaries, descended after about 4.6 s, contacted the bowl around 5.18 s and nearly stopped. It did not return to food. |
| On the bowl, away from food | 2.038 s | The fly also took off immediately, reached the ceiling by 176 ms and the wall by 312 ms. It made no food contact and ingested nothing. This did not demonstrate food localization or a directed approach. |

The off-food trial changes only the initial pose through the test harness: world x = 47, z = −15, heading = π/2. Neural parameters, connectivity, muscles, food and sensory processing are unchanged. The normal on-fruit pose does not test food localization.

The cause of the first trial's descent is visible in the recorded muscle signals. At 4.526 s, modeled wing lift was about 987 cm/s²; at 4.604 s it fell to about 967 cm/s², below gravity at 981 cm/s². All four wing-muscle groups remained essentially fully activated, with motor rates above the adapter's 80 Hz saturation threshold. The model's fatigue reduced force despite continuing neural drive. This is not evidence of a neural decision to land. Up to seven of the 22 reduced joints also reached their limits, indicating poor coordination of the current muscle commands.

The first trial transferred only 0.000354 normalized food units. Hunger rose from 0.650 to 0.656 over the run; it did not complete a meal or produce a satiety transition. The final motion label said “Walking,” but measured speed was only 0.018 body lengths/s. The label alone is insufficient evidence of locomotion.

The existing geometric event monitor also reported “approach” while the body was moving along the wall. That event is not accepted as successful food seeking. Raw position, contact, intake, muscle and neural measurements take precedence over event labels.

The remaining behavioral work is to calibrate sensory-dependent recruitment and inhibition of wing motor output, coordinated leg-muscle activity, muscle force/fatigue and the feeding/internal-state transition. The completed wiring and passing runtime/UI checks do not establish those behaviors.

[View the off-food screenshot sequence](banc-one-fly-off-food.mp4), played at approximately one quarter of neural speed. It uses held screenshots sampled during the actual 3D run, without motion interpolation; it is not a continuous screen recording.

Evidence: [6-second on-fruit run](banc-one-fly-observation.json), [off-food run](banc-one-fly-off-food.json), [summary](banc-one-fly-observation-summary.json). Both reports record source hashes and captured no application errors. Wall times were approximately 172 s and 50 s respectively; this was behavior observation, not an isolated performance benchmark.

Reproduce one trial at a time:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --population=1 --ms=6000 --wall-ms=360000
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --population=1 --start=off-food --follow=true --ms=2000
```
