The v6/v7 recordings show intermittently inactive steering-gain coordinates, without demonstrating a missing body-control axis. No simulator, database or network calls were used for this analysis.

[result.json](result.json) summarizes 82 episodes and 338 saved frames. Of those frames, 210 are powered and contact-free; 95 also have a positive current qualified-flight counter. [qualified-frames.json](qualified-frames.json) lists those exact 95 original file/frame indices, full-precision times, parameter vectors and muscle forces. The filter is `time > 0`, `environmentContacts === 0`, `wingPower > 0.1`, and `flightSeconds > 0`; it does not additionally require confirmed takeoff. Counts are 6 v6 training, 7 v6 held-out, 49 v7 training and 33 v7 held-out samples.

| Shared log-gain coordinates, zero-based | Qualified samples with both homologous muscle forces exactly zero |
|---|---:|
| b1 bias/amplitude: 3–4 | 20/95 |
| iii1 bias/amplitude: 13–14 | 77/95 |
| iii3 bias/amplitude: 15–16 | 26/95 |

Right iii3 has exactly zero saved requested rate, activation and force in all 338 frames. Left iii3 sometimes has nonzero force, reaching 0.015282748267054558 in the qualified sample set. Since left/right homologues share gains, neither iii3 coordinate is globally inactive. No entire muscle-type gain pair is zero across the complete qualified sample set. The other nine types have nonzero forces on both sides in every selected qualified sample; a nonzero value alone does not establish meaningful control authority.

The current `web/flybody-wings.js` and `web/training/flight-parameters.js` hashes match both recorded bundle pins. Neither bundle specifies `steering_force_reference`, so the relevant interpreter terms are gain times force, with no additive offset. At fixed muscle force, each log-gain derivative is its own summand: `F × exp(logGain) × basis`. Both gains have zero direct sensitivity when both homologous forces are zero. Gain-only search cannot directly create missing right-iii3 events, independently rescale the two sides, or learn event phase/sign tuning. Changes elsewhere may still alter later neural activity through feedback.

The analytical **unclipped wing-shape residual Jacobian** has rank 12 at all 210 powered, contact-free samples, using a relative singular-value threshold of `1e-10`. It is a 12×24 matrix: left/right three mean-angle and three amplitude residuals, versus 24 shared steering log gains. This is a local calculation with recorded forces held fixed. Its rows are not body forces or torques. Rank does not establish attainable directions at parameter bounds, after residual/target/actuator clipping, or over time; it does not establish body or closed-loop controllability. Bias/amplitude rows also have different modeling meanings, so singular-value ratios are numerical diagnostics rather than physical authority scores.

The supported implication is that iii1/iii3 gains often have absent or weak direct excitation in these recordings. It would exceed the evidence to mark their shared coordinates universally inactive or infer that roll, pitch or yaw control is impossible. Sparse previews do not prove continuous silence between samples.

[inputs.json](inputs.json) pins all 82 record files, both bundles, both interpreter sources and the analysis script, and records Python/NumPy versions. The reproducible [analyze.py](analyze.py) verifies bundle/source agreement, checks the expected record/frame counts and computes the statistics and singular values. From the repository root, this command performs a read-only recomputation and prints the summary:

```sh
.venv/bin/python reports/flight-development-v8-staging/trainability-review/analyze.py
```

To regenerate the three derived JSON artifacts in a separate directory:

```sh
.venv/bin/python reports/flight-development-v8-staging/trainability-review/analyze.py --output /private/tmp/flight-trainability-review
```
