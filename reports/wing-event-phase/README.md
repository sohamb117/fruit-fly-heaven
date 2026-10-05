A single left-b1 event produces different early fluid-torque impulses at four oscillator phases, while its full native muscle waveform is identical. All four 120 ms net impulses retain the same roll-positive, pitch-negative, yaw-positive directions at this operating point. Phase changes the transient; this assay does not identify a biological preferred phase or show phase alone reversing a steering command.

| Actual event clock phase (deg) | 0–20 ms impulse: roll / pitch / yaw | 0–120 ms impulse: roll / pitch / yaw |
|---:|---:|---:|
| 342.75 | 1.3130 / -0.2547 / 1.1749 | 4.1132 / -1.3044 / 4.1078 |
| 110.09 | 0.8960 / -0.5180 / 1.0898 | 4.2218 / -1.2921 / 4.1444 |
| 194.98 | 0.9084 / -0.5959 / 1.1216 | 4.2240 / -1.2697 / 4.1175 |
| 257.86 | 1.4462 / -0.3027 / 1.1403 | 4.1608 / -1.2641 / 4.1077 |

Impulse units in the table are 10^-6 g cm^2/s. Across phases, the range divided by absolute mean is roll 48.23%, pitch 81.68%, yaw 7.52% in the first 20 ms, and roll 2.65%, pitch 3.14%, yaw 0.89% over 120 ms. These are four deterministic probes, not statistical confidence intervals.

All eight arms completed. Each pulse has its own phase-matched no-event baseline; initial selected native/wing state hashes and body prefixes through 21 ms agree within every pair. The four selected kernel/excitation/activation/fatigue/force waveforms have the same binary Float64 hash. These finite body-state hashes use JSON serialization of the explicitly listed fields, not the entire native integration state. All recorded fluid-wrench differences are zero through 21 ms, with no contacts or instability warnings.

The event occurs at 20 ms. First nonzero native force is 0.018634 at 21 ms, and peak force 0.157515 is at 30 ms. First fluid-wrench contrast appears in 21.05–21.1 ms, 21.05–21.1 ms, 21.05–21.1 ms, 21.05–21.1 ms for the table rows. Force remains 0.011608 at the 140 ms endpoint (7.37% of peak), so these are finite-window impulses rather than complete impulse responses.

The root is explicitly restrained at level pose; six wing joints remain dynamic. Only left b1 MN75865 uses the event/native muscle route, from zero initial kernel and muscle states. Its baseline force is zero; the other 23 steering channels and common power hold historical synthetic trim forces. This differs from the historical hover operating point. Wing servo controls clip during 18.86–19.29% of native steps, with the same fraction within each matched pair; the measured transfer includes that nonlinear actuation. No brain, feedback controller, gain fitting, free flight, or root angular response is measured.

The 235.813447 Hz table clock runs continuously after its pre-warm initialization. Quantized 0.5 ms starting offsets yield actual event phases shown above, with target-quadrant errors from -17.25 to +20.09 degrees. One millisecond of causal force delay spans about 84.89 clock degrees; event phase, first force phase, last generated target phase, and peak-force phase are recorded separately. None is asserted to equal measured wing angle.

Wrench extraction uses the native fluid-force cache immediately after each 50 microsecond step, rotating local root torque to world axes and shifting to the matching COM, then rotating torque into body axes. It excludes actuator torque and the unmeasured restraint reaction. The result is not total torque or angular acceleration.

Results: [result.json](result.json), [offline analysis](analysis.json), [predeclared plan](plan.json), [executed source](source.used.mjs). MuJoCo 3.13.0; result SHA256 b4cb35ecfaac0578262fa6c86432afa2997c38dadc261361dd9f1f5dcaf4e8de; plan SHA256 ef0c2473bd5a9ac608ea1efb24cb0720b74ffc37b65ef4126f70b1ef7389831e; executed source SHA256 361033810474ef4739a06fe884604d96f23dfe7c5ebd967afc000853983c63d8; reduced XML SHA256 016145f213906218cb0efab33cc24baf8f45fcb7cc6fbe1714a7b7e33ecdcaa5. Full source and per-arm hashes are in the JSON files. Recreate only this offline summary with `node scripts/analyze-flight-event-controls.mjs`.
