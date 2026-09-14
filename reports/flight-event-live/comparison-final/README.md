The saved corrected baseline was reproduced exactly on all declared regression gates. The opt-in event comparison has matching assignments, initial conditions and recorded runtime/backend sources. This is one actual BANC/native-body seed, with no training or optimizer update.

| Arm | Terminal time (s) | Takeoff | Longest qualifying flight (s) | Return | Termination |
|---|---:|---|---:|---:|---|
| baseline-wing-interface | 0.480 | yes | 0.166 | -1.502 | excessive_rotation |
| event-wing-interface | 0.392 | yes | 0.174 | -1.478 | excessive_rotation |

Regression gates: parameters=true, initialCondition=true, initialObservation=true, initialFrame=true, physicsDigest=true, everyMotorEventPacket=true, evaluationBehavior=true, behavioralMetrics=true, finalFrame=true. The physics digest covers initialization plus completed 2 ms blocks (241 rows of 630 Float64 values), including qpos/qvel, controls, muscle state and wing state; it does not cover every native 50 microsecond step. Preview schedules may differ with wall time, so only initial/final preview equality is a parity gate.

Raw DLM recruitment over the same (100,280] ms window: baseline-wing-interface: left 118.889 Hz, right 117.778 Hz; event-wing-interface: left 110.000 Hz, right 107.778 Hz. B1 full-run event counts: baseline-wing-interface: left 0 events, right 0 events; event-wing-interface: left 0 events, right 0 events. These are actual cumulative event counts, not filtered-rate estimates. Complete per-type bilateral counts, raw/filtered rates and old-clamp equivalents are in [comparison.json](comparison.json).

First baseline/event neural packet difference: 64 ms; first cumulative-count difference: 78 ms. Once body feedback diverges, these are two live closed-loop trajectories, not a matched fixed-input motor replay.

Source selectivity passed: both arms used the same runtime/backend sources and byte-identical native XML. Metadata differs only by the explicit event-prior diagnostic tag; the corresponding config option, version/fingerprint and explanatory note differ. Other configuration, neural graph/profile, sensory mapping and all 27 parameters remain identical. The baseline is the existing no-adhesion development benchmark with power gain 1.5 and tied steering gains 0.05, not a freshly learned or physiological trim.

Active execution cost was 15.260 seconds per simulated second for the baseline and 15.442 for events (1.189% more in this pair). Raw execution times were 7.325 and 6.053 seconds. The shorter event run must not be called faster from raw duration alone. This single normalized comparison excludes setup and does not isolate implementation overhead from different body trajectories.

Actual wing power, excitation, activation, fatigue and force are retained for every available preview in the JSON. Their sampling times are baseline-wing-interface [0.000, 0.316, 0.480] s; event-wing-interface [0.000, 0.314, 0.392] s. These sparse samples do not justify dense force curves or matched-window force means. The rate80-clamp equivalents are labeled command comparisons and are not the event arm's actual excitation.

The initial state is native grounded stance with the same seeded disturbance. See the JSON for foot support, contacts, final COM/rotation state, takeoff/landing diagnostics and exact source/config hashes. Sustained flight and landing criteria remain unchanged. A short increase in flight time is not stable flight, biological validation or evidence of learning.
