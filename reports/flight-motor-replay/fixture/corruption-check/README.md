# Recorded motor mechanical replay

Capture: synthetic-native-motor-replay-fixture. Native MuJoCo 3.13.0; 20 recorded 2ms blocks (0.04s). [Full source-pinned results](result.json).

Bit-exact qpos/qvel baseline gate: **FAIL**. Behavioral interpretation allowed: **false**. This is a synthetic plumbing fixture, not actual BANC behavior.

No external root forces or pose resets occur during replay. Each intervention reuses the same complete initial state and captured neural sequence. Native contacts, muscle length/velocity effects, internal state and biomechanics evolve freely. Claw scalar actuator output and transmitted world force are reported separately. Events are sampled every 2ms.

| Case | Recorded COM rise (cm) | Minimum up | Maximum angular speed (rad/s) | First inversion (s) | Final loaded feet | Error |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.0000 | 0.9783 | 0.00 | none | 6 | none |

Hold-last extension: 0s, applied only to counterfactual diagnostic runs and explicitly labeled in each sample. Counterfactual motion would alter sensory input in a live brain; this replay deliberately holds the captured motor sequence fixed.
