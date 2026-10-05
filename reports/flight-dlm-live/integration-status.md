The opt-in DLM model is integrated and executes successfully through the actual BANC → native event-muscle → FlyBody loop. It has not produced stable flight. Both matched episodes fail from excessive rotation after about 0.4 seconds.

The profile-disabled run exactly reproduces the previous event-interface run on all nine declared regression gates: assignment parameters, native initial conditions/observation, initial frame, full physics digest, every motor-event packet, behavioral result, behavioral metrics, and final frame. Both current arms use identical runtime/backend sources, seed 190888, all 27 interpreter parameters, native XML, event-excitation priors, and objective. Only the declared DLM intrinsic-model configuration and its diagnostic metadata differ. All comparison gates passed. See [the complete comparison](comparison/comparison.json) and [readable report](comparison/README.md).

| Actual result | Profile disabled | DLM ionic profile |
|---|---:|---:|
| Takeoff time | 76 ms | 84 ms |
| Longest qualified flight | 174 ms | 170 ms |
| Terminal time | 392 ms | 398 ms |
| Termination | excessive rotation | excessive rotation |
| Return | −1.478 | −1.119 |
| Left DLM raw rate, (100,280] ms | 110.00 Hz | 37.78 Hz |
| Right DLM raw rate, (100,280] ms | 107.78 Hz | 41.11 Hz |
| DLM events in that window, all ten cells | 196 | 71 |
| B1 events, each side, entire episode | 0 | 0 |
| Active wall time / simulated second | 15.42 s | 17.28 s |

The first recorded neural/count difference occurs at 30 ms. Per-cell raw rates are derived from actual events and checked against cumulative counter differences at the window endpoints. The ionic DLM range in that window is 33.33–44.44 Hz; this is a substantial reduction, not validation of biological recruitment. Later neural differences include feedback from two diverging bodies, so they cannot be interpreted as a frozen-input cell assay.

The reduction in firing did **not** yield comparably low wing drive. At the ionic arm's available 278 ms preview, left/right DLM excitation was .903/.899, activation .908/.919, and normalized force .897/.909. The combined wing muscle drives were .845/.879, and requested power was 1 on both sides. At the 398 ms final sample, DLM excitation was .943/.945 and both requested powers remained 1. These are sparse native measurements, not time averages. The baseline's intermediate sample is at 314 ms and therefore is not a matched-time force comparison.

The source still computes requested wing power as `clamp(sideDrive * powerGain)`. This diagnostic uses powerGain 1.5, so a sideDrive above 2/3 reaches maximum requested power. The recorded ionic side drives exceed that threshold. Thus the new cell model changes neural activity, while the remaining recruitment/combined-drive path still occupies the maximum-power region at the observed samples. Lower mean DLM rate alone has not corrected orientation control. The slightly higher return is not an improvement in maintained flight or evidence of learning.

The ionic arm costs about 12.1% more active wall time per simulated second in this pair. This is not a repeated performance benchmark: different body trajectories and different termination times are normalized but not controlled away. Setup was .209/.212 s; active execution was 6.046/6.878 s. Neither arm is Safari: both used the pinned native Dawn/Metal backend.

The three live built artifacts match the validated staged build byte-for-byte; hashes are recorded in [artifact-verification.json](artifact-verification.json). The parent reported 72 focused live tests passing. Staged equation, event-timing, graph-delivery, ownership and numerical controls are documented in [the runtime handoff](../flight-dlm-runtime/README.md). No simulation, configuration mutation, or optimizer update was performed by this offline comparison.
