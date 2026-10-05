# Flight scoring repair

The native trajectory is unchanged. Exact replay reproduced all 195 recorded qpos/qvel checkpoints byte for byte and matched the initial plus all 36 captured preview observations. The source is the explicitly reduced no-adhesion, power 1.5, steering 0.05 live BANC/WASM diagnostic; it is not Safari/WebGPU or an optimizer result.

Contact-free samples begin at 70 ms, followed by brief rear-right contact at 72 ms and a low-load front-right brush at 78 ms. The final departure is at 80 ms, with upward velocity only 0.212 cm/s. The old scorer required its first airborne sample to exceed 0.5 cm/s and permanently ignored the subsequent ascent. Instantaneous wingbeat recoil also repeatedly crossed 20 rad/s despite a 20 ms RMS below 16.71 rad/s during the early ascent.

The repaired scorer authorizes attempts from recent measured foot support, requires powered ascent later in the same uninterrupted bout, and measures angular-speed magnitude with a 20 ms RMS window. Body impacts invalidate support history. The 50 ms airborne acceleration window excludes ground impulses and is measured independently of attitude qualification. Contacts still reset continuous airtime. Instantaneous attitude and catastrophic rotation limits remain in force.

| Same recorded trajectory | Previous scorer | Repaired scorer |
|---|---:|---:|
| Confirmed takeoff | no | yes, departure 80 ms |
| Best controlled flight | 0 ms | 184 ms |
| Final outcome | crash at 390 ms | crash at 390 ms |
| Full task success | no | no |
| Return | -4.04518 | -1.49952 |

`rescore.json` contains every score state before/after. `previous-flight-objective.mjs` preserves the old observer. The frozen native observation fixture and adversarial tests reject ballistic hops, accumulated short hops, body contact, unpowered ascent, sustained spin, and failed full horizons. Recognizing valid partial progress is a scoring repair, not evidence of learned or maintained flight.
