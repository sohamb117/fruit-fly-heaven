# Recorded motor mechanical replay

Capture: synthetic-native-motor-replay-fixture. Native MuJoCo 3.13.0; 20 recorded 2ms blocks (0.04s). [Full source-pinned results](result.json).

Bit-exact qpos/qvel baseline gate: **PASS**. Behavioral interpretation allowed: **false**. This is a synthetic plumbing fixture, not actual BANC behavior.

No external root forces or pose resets occur during replay. Each intervention reuses the same complete initial state and captured neural sequence. Native contacts, muscle length/velocity effects, internal state and biomechanics evolve freely. Claw scalar actuator output and transmitted world force are reported separately. Events are sampled every 2ms.

| Case | Recorded COM rise (cm) | Minimum up | Maximum angular speed (rad/s) | First inversion (s) | Final loaded feet | Error |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | -0.0061 | 0.9731 | 8.63 | none | 6 | none |
| symmetric_wing_drive | -0.0061 | 0.9734 | 8.66 | none | 6 | none |
| zero_steering_force | -0.0064 | 0.9729 | 8.68 | none | 6 | none |
| zero_steering_force+symmetric_wing_drive | -0.0063 | 0.9732 | 8.69 | none | 6 | none |
| no_claw_gain | -0.0116 | 0.9612 | 6.52 | none | 2 | none |
| no_claw_gain+symmetric_wing_drive | -0.0116 | 0.9611 | 6.74 | none | 3 | none |
| no_claw_gain+zero_steering_force | -0.0117 | 0.9612 | 6.58 | none | 3 | none |
| no_claw_gain+zero_steering_force+symmetric_wing_drive | -0.0116 | 0.9615 | 6.58 | none | 1 | none |
| power_x1.5 | -0.0043 | 0.9734 | 11.95 | none | 6 | none |
| symmetric_wing_drive+power_x1.5 | -0.0043 | 0.9734 | 12.10 | none | 6 | none |
| zero_steering_force+power_x1.5 | -0.0046 | 0.9732 | 12.01 | none | 6 | none |
| zero_steering_force+symmetric_wing_drive+power_x1.5 | -0.0045 | 0.9732 | 12.20 | none | 6 | none |
| no_claw_gain+power_x1.5 | -0.0098 | 0.9602 | 6.60 | none | 1 | none |
| no_claw_gain+symmetric_wing_drive+power_x1.5 | -0.0099 | 0.9608 | 6.52 | none | 0 | none |
| no_claw_gain+zero_steering_force+power_x1.5 | -0.0096 | 0.9612 | 7.47 | none | 1 | none |
| no_claw_gain+zero_steering_force+symmetric_wing_drive+power_x1.5 | -0.0099 | 0.9612 | 8.51 | none | 2 | none |
| zero_steering_rates | -0.0064 | 0.9729 | 8.68 | none | 6 | none |

Hold-last extension: 0s, applied only to counterfactual diagnostic runs and explicitly labeled in each sample. Counterfactual motion would alter sensory input in a live brain; this replay deliberately holds the captured motor sequence fixed.
