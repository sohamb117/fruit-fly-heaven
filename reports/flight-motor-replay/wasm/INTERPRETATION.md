# Matched motor replay findings

This is one actual BANC **WASM** motor sequence, seed 888, covering 136ms. It is distinct from the Safari/WebGPU sequence. The native baseline reproduces all 68 recorded qpos and qvel checkpoints with identical float64 bytes. No root force or pose correction is applied during any run. The [full results](result.json) preserve the capture-file hash, current model/config hashes, interventions and 2ms samples.

| Mechanical intervention | Maximum angular speed | Minimum up | COM change at 136ms | Outcome within the recorded interval |
|---|---:|---:|---:|---|
| Baseline | 454.87rad/s | −0.6558 | −0.0516cm | Inverts at 132ms |
| Zero claw gain only | 31.98rad/s | 0.9119 | −0.2003cm | Releases support at 70ms; remains upright but descends |
| Zero steering force only | 31.51rad/s | 0.7776 | −0.0377cm | Still supported by two feet |
| Power ×1.5 only | 69.28rad/s | −0.3844 | −0.4089cm | Inverts at 114ms |
| Zero claw gain + power ×1.5 | 32.31rad/s | 0.7891 | +0.4109cm | Upward departure without inversion |
| Zero claw gain + zero steering force + power ×1.5 | 21.49rad/s | 0.9480 | +0.1735cm | Releases support at 68ms; final up 0.9993, vertical speed +7.47cm/s |

Adding equal left/right wing drive to the final combination gives similar results: COM +0.1760cm, minimum up 0.9492 and maximum angular speed 21.43rad/s. Zeroing steering rates and zeroing the steering forces produce identical observed root/wing trajectories and foot loads at all 68 boundaries in this recording. Every raw motor rate stays unchanged in the force-only, grip and power interventions; the separate rate-zero control changes only the annotated steering rates.

At 100ms the baseline has only the two rear feet loaded. The rear claw scalar actuator outputs are 0.467 and 0.538 bodyweights, but their transmitted **downward world forces** are 0.329 and 0.520 bodyweights. Scalar adhesion output is not itself a measured vertical force. The replay computes the transmitted force from native sparse actuator moment rows multiplied by actuator force. The six claws' summed root generalized forces match the native root `qfrc_actuator` exactly at every sample across all 17 cases.

The evidence supports an interaction between grip, wing power and steering in this onset. Grip removal alone avoids the catastrophic rotation but does not supply adequate lift. Increasing power with grip retained worsens inversion. Suppressing steering helps the higher-power grip-release trial, but worsens descent when grip is released at the baseline power. These results do not justify permanently deleting adhesion or steering.

The current 27 wing parameters cannot directly release the claw actuators. A model that retains grip while increasing wing power therefore constrains what those parameters can achieve. Grip recruitment/release and steering calibration deserve explicit treatment before interpreting this onset as a neural-learning failure.

The recording ends after 136ms. None of these results establishes one second of maintained flight, successful landing, generalization to another neural sequence, or Safari behavior. No held-last-input extension was run for this report.
