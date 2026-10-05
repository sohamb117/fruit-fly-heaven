# Reduced flight learning check

This is a local development experiment, not a production deployment or a validated flight controller. One fly is evaluated at a time through the actual BANC/WASM and native MuJoCo environment. The existing durable coordinator assigns every training candidate and receives every completed result.

The model omits phenomenological claw adhesion while retaining normal foot contact and friction. The canonical full body model is unchanged. The initial power gain is 1.5 and the 24 steering gains are 0.05, selected from prior causal controls, not learned. Neural physiology, sensory transduction, native muscle dynamics, aerodynamics and wingbeat tables are fixed. All 27 trainable coefficients belong to the wing interpreter.

- Bundle: `bundle.json`
- Config hash: `d62ca36f5bbff7ea1b11af418219f7f00b6ea6ffcb162d2df26c73172025e0fb`
- Model fingerprint: `f4ed6414618c2395257728f5b584af217b773ad46d510ef4d4bcf79269bd5bfd`
- Database: `coordinator.sqlite3`
- Full parameter deltas and trial measurements: `progress.json`
- Saved native evaluations and sparse frames: `contributor/`

The first generation completed eight episodes (four antithetic pairs). The stored checkpoint changed 10 of 27 parameters, with L2 update 0.129054 and maximum absolute log-coordinate change 0.088060. Each of the other 17 received a negative proposed update that was clipped at its existing lower bound; they were not silently frozen.

One exploration episode confirmed takeoff with 182 ms of controlled flight. All eight eventually failed with excessive rotation. The total simulated duration was 2.226 seconds; active evaluation took 223.82 wall seconds on the WASM neural backend, plus 1.54 seconds of per-episode setup. These numbers exclude process/model initialization and do not measure Safari/WebGPU throughput.

Checkpoint movement is established, but generation 1 is worse on both paired selection seeds. Optimization was stopped after this generation. `comparison.json` contains the completed comparisons; the final test seeds remain unused. These comparisons do not mutate the optimizer or submit validation observations as training results.

| Selection seed | Initial best flight | Generation 1 best flight | Initial / generation 1 takeoff |
|---|---:|---:|---|
| 190888 | 164 ms | 104 ms | yes / no |
| 290888 | 150 ms | 106 ms | yes / no |

Every comparison eventually failed with excessive rotation. Mean qualified airtime fell from 157 ms to 105 ms, and mean return fell from -1.41956 to -2.33655. This is not evidence of improved flight. It motivates further mechanical/interface calibration, rather than promoting the changed checkpoint.

Reproduce the development service and contribution:

```sh
python3 scripts/serve-training-dev.py --bundle reports/flight-development-v5/bundle.json --database reports/flight-development-v5/coordinator.sqlite3 --port 7849
node scripts/contribute-training-native.mjs http://127.0.0.1:7849/ reports/flight-development-v5/contributor-next 8
python3 scripts/report-flight-training.py reports/flight-development-v5/coordinator.sqlite3 --output reports/flight-development-v5/progress.json
```

The next contribution resumes the current generation from the coordinator. It does not replay generation 0. Keep every run's model/configuration namespace separate; do not import this checkpoint into the full-model run.
