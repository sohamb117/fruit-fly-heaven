# Recorded motor mechanical replay

Capture: synthetic-native-motor-replay-fixture; neural backend synthetic-no-neural-model. Native MuJoCo 3.13.0; 20 recorded 2ms blocks (0.04s). [Full source-pinned results](result.json). 

Bit-exact qpos/qvel baseline gate: **PASS**. Behavioral interpretation allowed: **false**. This is a synthetic plumbing fixture, not actual BANC behavior.

No direct qfrc_applied/xfrc_applied injection or pose resets occur during replay. Every case starts from the same complete state. The factorial mechanical interventions retain recorded motor rates; the separately identified steering-rate ablation zeros only the annotated steering rates. Native contacts, muscle length/velocity effects, internal state and biomechanics evolve freely. Claw scalar actuator output and transmitted world force are reported separately. Events are sampled every 2ms.

All table metrics use only the recorded horizon.

| Case | Recorded COM rise (cm) | Minimum up | Maximum angular speed (rad/s) | First inversion (s) | Final loaded feet | Error |
|---|---:|---:|---:|---:|---:|---|
| baseline | -0.0061 | 0.9731 | 8.63 | none | 6 | none |

Hold-last extension: 0s, applied only to counterfactual diagnostic runs and explicitly labeled in each sample. Its total-horizon summaries are separate diagnosticTotalMetrics fields. Counterfactual motion would alter sensory input in a live brain; that neural response is not simulated.
