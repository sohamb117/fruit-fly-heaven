# Exact live-variant departure replay

All 195 native qpos/qvel checkpoints are byte-identical. The initial and all 36 captured preview observations also agree exactly. Every 2 ms observation is in training-observations.json. This is the live BANC/WASM diagnostic with zero adhesion gain, power gain 1.5 and steering gains 0.05; it is not the canonical production body or Safari/WebGPU.

Contact-free at 70 ms, rear-right foot recontact at 72 ms (0.160 bodyweights), clear at 74/76 ms, front-right foot brush at 78 ms (0.03975 bodyweights), clear from 80 through 388 ms. At 390 ms two nonfoot contacts accompany excessive rotation.

Through 280 ms, angular-speed median is 8.23, p95 20.43, maximum 23.84 and RMS 10.53 rad/s; 9 of 140 samples exceed 20. Minimum up is 0.8695 (29.6 degrees tilt), and COM rise is 1.806 cm. These are finite-horizon observations, not maintained-flight or landing validation.

| Time (ms) | Contacts | Foot load (BW) | COM vz (cm/s) | Angular speed (rad/s) | Trailing RMS (rad/s) |
|---|---|---|---|---|---|
| 60 | 4 | 1.3909 | -0.335 | 9.86 | 8.73 |
| 62 | 3 | 0.6255 | 0.190 | 9.97 | 9.27 |
| 64 | 3 | 0.5897 | 0.205 | 6.89 | 9.51 |
| 66 | 3 | 0.2249 | 0.774 | 2.13 | 9.52 |
| 68 | 1 | 0.1833 | 0.211 | 8.23 | 9.85 |
| 70 | 0 | 0.0000 | 0.849 | 2.75 | 9.87 |
| 72 | 1 | 0.1600 | 0.443 | 8.98 | 10.26 |
| 74 | 0 | 0.0000 | 0.661 | 6.87 | 10.44 |
| 76 | 0 | 0.0000 | 0.237 | 4.41 | 8.28 |
| 78 | 1 | 0.0398 | 0.419 | 9.81 | 7.53 |
| 80 | 0 | 0.0000 | 0.212 | 3.87 | 6.97 |
| 82 | 0 | 0.0000 | 0.113 | 9.31 | 6.88 |
| 84 | 0 | 0.0000 | 0.129 | 6.34 | 6.82 |
| 86 | 0 | 0.0000 | -0.174 | 9.22 | 7.39 |
| 88 | 0 | 0.0000 | 0.273 | 5.77 | 7.15 |
| 90 | 0 | 0.0000 | -0.049 | 8.01 | 7.54 |
| 92 | 0 | 0.0000 | 0.856 | 10.04 | 7.67 |
| 94 | 0 | 0.0000 | 0.636 | 18.61 | 9.42 |
| 96 | 0 | 0.0000 | 1.848 | 7.42 | 9.61 |
| 98 | 0 | 0.0000 | 1.517 | 20.43 | 11.15 |
| 100 | 0 | 0.0000 | 2.730 | 7.68 | 11.35 |

Statistics are of the actual 2 ms observer samples. The trailing 20 ms magnitude RMS is unsegmented; samples near a departure may include pre-departure points. Signed mean angular velocity uses the rotating free-joint frame and is not a fixed-world angular displacement or complete wingbeat torque measurement.
