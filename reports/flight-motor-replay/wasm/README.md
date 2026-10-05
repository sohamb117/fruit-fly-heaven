# Recorded motor mechanical replay

Capture: training-native-motor-replay; neural backend wasm. Native MuJoCo 3.13.0; 68 recorded 2ms blocks (0.136s). [Full source-pinned results](result.json). Actual BANC/WASM and native MuJoCo. Distinct from the Safari/WebGPU baseline; no coordinator or optimization.

Bit-exact qpos/qvel baseline gate: **PASS**. Behavioral interpretation allowed: **true**. 

No direct qfrc_applied/xfrc_applied injection or pose resets occur during replay. Every case starts from the same complete state. The factorial mechanical interventions retain recorded motor rates; the separately identified steering-rate ablation zeros only the annotated steering rates. Native contacts, muscle length/velocity effects, internal state and biomechanics evolve freely. Claw scalar actuator output and transmitted world force are reported separately. Events are sampled every 2ms.

All table metrics use only the recorded horizon.

| Case | Recorded COM rise (cm) | Minimum up | Maximum angular speed (rad/s) | First inversion (s) | Final loaded feet | Error |
|---|---:|---:|---:|---:|---:|---|
| baseline | -0.0516 | -0.6558 | 454.87 | 0.132 | 0 | none |
| symmetric_wing_drive | -0.0631 | 0.2580 | 57.90 | none | 1 | none |
| zero_steering_force | -0.0377 | 0.7776 | 31.51 | none | 2 | none |
| zero_steering_force+symmetric_wing_drive | -0.0852 | 0.8673 | 27.60 | none | 2 | none |
| no_claw_gain | -0.2003 | 0.9119 | 31.98 | none | 0 | none |
| no_claw_gain+symmetric_wing_drive | -0.1947 | 0.9261 | 29.47 | none | 0 | none |
| no_claw_gain+zero_steering_force | -0.3877 | 0.6523 | 31.69 | none | 0 | none |
| no_claw_gain+zero_steering_force+symmetric_wing_drive | -0.3713 | 0.6749 | 29.44 | none | 0 | none |
| power_x1.5 | -0.4089 | -0.3844 | 69.28 | 0.114 | 0 | none |
| symmetric_wing_drive+power_x1.5 | -0.4037 | -0.3630 | 66.23 | 0.114 | 0 | none |
| zero_steering_force+power_x1.5 | -0.2428 | -0.2915 | 53.09 | 0.124 | 0 | none |
| zero_steering_force+symmetric_wing_drive+power_x1.5 | -0.2866 | -0.5403 | 71.02 | 0.118 | 0 | none |
| no_claw_gain+power_x1.5 | 0.4109 | 0.7891 | 32.31 | none | 0 | none |
| no_claw_gain+symmetric_wing_drive+power_x1.5 | 0.4073 | 0.7737 | 33.74 | none | 0 | none |
| no_claw_gain+zero_steering_force+power_x1.5 | 0.1735 | 0.9480 | 21.49 | none | 0 | none |
| no_claw_gain+zero_steering_force+symmetric_wing_drive+power_x1.5 | 0.1760 | 0.9492 | 21.43 | none | 0 | none |
| zero_steering_rates | -0.0377 | 0.7776 | 31.51 | none | 2 | none |

Hold-last extension: 0s, applied only to counterfactual diagnostic runs and explicitly labeled in each sample. Its total-horizon summaries are separate diagnosticTotalMetrics fields. Counterfactual motion would alter sensory input in a live brain; that neural response is not simulated.
