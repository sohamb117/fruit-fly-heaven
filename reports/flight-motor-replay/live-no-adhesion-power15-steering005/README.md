# Recorded motor mechanical replay

Capture: training-native-motor-replay; neural backend wasm. Native MuJoCo 3.13.0; 195 recorded 2ms blocks (0.39s). [Full source-pinned results](result.json). Actual BANC/WASM and native MuJoCo. Distinct from the Safari/WebGPU baseline; no coordinator or optimization.

Bit-exact qpos/qvel baseline gate: **PASS**. Behavioral interpretation allowed: **true**. 

No direct qfrc_applied/xfrc_applied injection or pose resets occur during replay. Every case starts from the same complete state. The factorial mechanical interventions retain recorded motor rates; the separately identified steering-rate ablation zeros only the annotated steering rates. Native contacts, muscle length/velocity effects, internal state and biomechanics evolve freely. Claw scalar actuator output and transmitted world force are reported separately. Events are sampled every 2ms.

All table metrics use only the recorded horizon.

| Case | Recorded COM rise (cm) | Minimum up | Maximum angular speed (rad/s) | First inversion (s) | Final loaded feet | Error |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1.7286 | -0.1792 | 577.16 | 0.390 | 0 | none |

Hold-last extension: 0s, applied only to counterfactual diagnostic runs and explicitly labeled in each sample. Its total-horizon summaries are separate diagnosticTotalMetrics fields. Counterfactual motion would alter sensory input in a live brain; that neural response is not simulated.
