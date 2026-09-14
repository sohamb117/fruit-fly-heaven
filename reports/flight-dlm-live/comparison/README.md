The disabled DLM option reproduces the prior event-interface actual run on every declared regression gate. The baseline/ionic comparison passes its assignment, initial-state, source and selective-config checks. These are two closed-loop episodes of one fixed seed, with no training update.

| Arm | Terminal time (s) | Takeoff | Longest qualified flight (s) | Return | Termination |
|---|---:|---|---:|---:|---|
| baseline-dlm-profile | 0.392 | yes | 0.174 | -1.478 | excessive_rotation |
| ionic-dlm-profile | 0.398 | yes | 0.170 | -1.119 | excessive_rotation |

Baseline regression: parameters=true, initialCondition=true, initialObservation=true, initialFrame=true, physicsDigest=true, everyMotorEventPacket=true, evaluationBehavior=true, behavioralMetrics=true, finalFrame=true. The digest covers initialization and each completed 2 ms body block; it does not certify the native integration steps between those samples. Preview schedules depend on wall time, so intermediate frames are not a parity gate.

Raw DLM rates in (100,280] ms: baseline-dlm-profile: left 110.000 Hz, right 107.778 Hz; ionic-dlm-profile: left 37.778 Hz, right 41.111 Hz. Per-cell rates: baseline-dlm-profile [12322: 133.333 Hz, 31143: 77.778 Hz, 41465: 105.556 Hz, 74497: 83.333 Hz, 84317: 127.778 Hz, 127563: 127.778 Hz, 135445: 100.000 Hz, 157417: 83.333 Hz, 160138: 150.000 Hz, 173140: 100.000 Hz]; ionic-dlm-profile [12322: 38.889 Hz, 31143: 38.889 Hz, 41465: 44.444 Hz, 74497: 33.333 Hz, 84317: 38.889 Hz, 127563: 38.889 Hz, 135445: 44.444 Hz, 157417: 33.333 Hz, 160138: 38.889 Hz, 173140: 44.444 Hz]. B1 full-run counts: baseline-dlm-profile: left 0 events, right 0 events; ionic-dlm-profile: left 0 events, right 0 events. Counts are checked against cumulative counter differences at both window endpoints. Filtered rates remain separately labeled in JSON.

First neural packet difference: 30 ms; first count difference: 30 ms. This comparison changes DLM intrinsic dynamics before the body receives each episode's evolving signals. Later differences include sensory feedback from the diverged bodies; it is not a fixed-input isolated-membrane replay.

Selectivity checks: nativeXmlIdentical=true, metadataDiffersOnlyByDeclaredIntrinsicModel=true, intrinsicModelMatchesBetweenConfigAndMetadata=true, baselineHasNoIntrinsicOption=true, eventPriorsUnchanged=true, onlyDlmCellsSelected=true, otherConfigurationIdentical=true, onlyChangedAssetIsMetadata=true. The JSON records the actual ten-cell profile, unchanged event-excitation priors, full source/config hashes and all 27 parameters. Native XML, contact/adhesion treatment and the objective must remain identical.

Active execution cost: baseline 15.422 and ionic 17.282 wall seconds per simulated second; ionic/baseline ratio 1.121. Raw execution: 6.046 and 6.878 s; setup: 0.209 and 0.212 s. This is a single normalized episode comparison, not a repeated performance benchmark or a pure kernel-overhead measurement.

Actual excitation, activation, fatigue, muscle force, wing controls/actuator forces, root motion and foot/contact observations are preserved at each available preview. Sample times: baseline-dlm-profile [0.000, 0.314, 0.392] s; ionic-dlm-profile [0.000, 0.278, 0.398] s. These sparse samples do not establish dense force curves or matched-window force averages. In particular, rate80-clamp equivalents are not actual event-mode excitation.

A longer short flight or lower motor rate alone is not stable flight, verified biological calibration, or evidence of learning. Any success/failure claim here uses the unchanged full-horizon objective. No simulation was executed by this comparison script.
