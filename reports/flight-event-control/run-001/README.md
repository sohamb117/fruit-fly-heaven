The frozen classical feedback controller kept the event-driven native body within 7.20 degrees of upright for the full 1.5 second free-flight assay. The matching fixed-trim event arm reached 43.52 degrees. This is a mechanical positive control using synthetic events and prepared flight conditions; it does not demonstrate BANC flight, takeoff, landing, or biological calibration.

| Arm | Maximum tilt (deg) | Final filtered angular speed (rad/s) | COM rise (cm) | Wing control clipped steps | Declared result |
|---|---:|---:|---:|---:|---|
| historical_native | 6.679 | 2.295 | 0.007 | 18.76% | pass |
| causal_native | 6.641 | 2.273 | 0.005 | 18.76% | pass |
| event_closed_loop | 7.198 | 2.378 | 0.833 | 18.80% | pass |
| event_open_loop | 43.519 | 2.968 | -0.765 | 19.07% | fail |

All arms ran 30,000 native steps (50 microseconds each), with no contacts, instability warnings, applied root forces, or root pose writes after release. The historical native arm exactly matched the earlier latency assay's 750 saved 2 ms samples, final state and metrics. This is not a claim of hashing every native step.

The causal native arm changes the muscle update to the completed 1 ms interval boundary and performs similarly to the historical update. The event arm passes through 48 quantized synthetic unit event streams, per-unit kernels, 28 mapped excitations and the unchanged native activation/fatigue/force kernel. Its maximum force tracking error is 0.562 and RMS error 0.123; 2.52% of event-rate commands clip to the declared maximum. The approximately 18.8% wing-control clipping in the successful arms remains a material feature of this operating point.

The rate inverse uses actual native mean activation, not mean excitation. Eleven nonzero rates from 2 to 400 Hz each receive 3 seconds of muscle warm-up and a 1 second measurement, with the same quantized event generator and within-muscle unit staggering used during release. Every activation waveform matches its 1-second-earlier waveform exactly at Float32 precision. Means are nondecreasing; high-rate DLM/DVM values plateau. Fatigue evolves in this measurement, but only activation is inverted; the current-fatigue force inverse is separate. This lookup is a diagnostic feedforward map, not reward fitting or a neural component.

Timing review found no future body-feedback use: current body measurements produce held commands at each 2 ms boundary, packet events depend only on that command and previous generator state, and the event/native state advances after each completed 1 ms physical interval. Prior force drives the interval. A packet may contain its full upcoming 2 ms event schedule, but the adapter consumes only events at or before the completed boundary.

All arms share a 100 ms restrained physical warm-up and prepared native trim activation with zero fatigue. Event arms additionally prepare 2 seconds of kernel history without evolving native muscles. Thus this result does not test cold onset. The existing classical controller has access to body orientation, angular velocity, height and vertical velocity. Its gains and physical allocation Jacobian are frozen from the earlier diagnostic; BANC is not executed. Horizontal position is uncontrolled (event feedback drift 2.168 cm). The permissive COM criterion is absolute rise below 10 cm; this is not precise hover.

Results: [result.json](result.json), [offline analysis](analysis.json), [executed source](diagnostic-source.used.mjs). MuJoCo 3.13.0; result SHA256 fb24967dfea4027fe6a09071c15eb355a004aa64a152a976e47e84efa13b2e6a; executed source SHA256 e1b8072ac7cc9948a42d63a0c6a8343088f975150aa5fa840d23cb5b57b811b1; reduced XML SHA256 016145f213906218cb0efab33cc24baf8f45fcb7cc6fbe1714a7b7e33ecdcaa5; native muscle WASM SHA256 bcbc77058e9f310a8faba078d16fdff2fdb3bea9d1836d80f80cda2598451d6a. Full dependency hashes are in both JSON files. Recreate only this offline summary with `node scripts/analyze-flight-event-controls.mjs`.
