# Guarded flight training

The two-generation run completed all 28 assigned episodes. Both nominees lost their paired comparisons and were rejected. All 27 saved coordinates stayed exactly unchanged. There were 24 confirmed takeoffs, no sustained-flight/landing successes, and every episode ended in excessive rotation. This validates checkpoint retention under real evaluations; it does not establish a new flight improvement.

| Generation | Nominee | Nominee mean qualified flight | Incumbent mean qualified flight | Mean return difference | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | g0-p3-pos | 247 ms | 359 ms | −0.338 | Reject |
| 1 | g1-p2-pos | 280 ms | 401 ms | −0.362 | Reject |

These comparisons are training data. The [complete training report](training-final.json) records the decisions and saved parameter differences. The [independent held-out evaluation](held-out-comparison/README.md) passed its provenance, assignment and history checks: the initial and final checkpoints produced identical results on all three reserved seeds, with 328/504/344 ms qualified flight (392 ms mean). All six evaluations confirmed takeoff and then crashed; none achieved sustained flight or landing. The unchanged behavior matches the exactly retained parameter vector. No additional search is running.

This local experiment tests checkpoint selection with the same neural and body dynamics as v7. It starts from the exact v6 selected candidate, which remained the more consistent checkpoint after the v7 comparison. Neither starting checkpoint nor any previous trial has completed sustained flight and landing.

Each generation evaluates eight search candidates. The best trial then faces the incumbent on three fresh matched seeds. Its mean return must improve before the saved parameters change; otherwise the incumbent is retained. These comparisons are training data. Two generations are followed by six independent evaluations of the initial and final checkpoints on three reserved seeds.

Only one full BANC/FlyBody episode runs at a time. Native Dawn/Metal executes the neural model and MuJoCo WASM executes the body. This is a local experiment; it has not been deployed and is not a Safari compute measurement.

The frozen inputs are `bundle.json`, `backend-plan.json`, `plan.json`, and `validation-protocol.json`. Their configuration hash is `64ef429834ca902cdac4f2e83d2a8b1e0104cc134701760c6858448cd9ab614a`; the model fingerprint is `f536f20ed9295ab6ae1f672739b9ad8c80ef3320bc0a53450da7ffeb45249182`. `prepare.py` creates them once and rejects overwrites. The SQLite database preserves assignments, comparison decisions and all 27 parameter coordinates.

After preparation, start or resume the matching coordinator:

```sh
.venv/bin/python scripts/serve-training-dev.py --port 7858 \
  --database reports/flight-development-v8/coordinator.sqlite3 \
  --bundle reports/flight-development-v8/bundle.json
```

In another terminal, run the contributor. Its generation limit prevents a restart from evaluating generation 2 or later:

```sh
node scripts/contribute-training-native.mjs http://127.0.0.1:7858/ \
  reports/flight-development-v8/contributions 28 \
  reports/flight-development-v8/backend-plan.json
```

Inspect actual saved parameter changes and behavior without allocating a job:

```sh
.venv/bin/python scripts/report-flight-training.py \
  reports/flight-development-v8/coordinator.sqlite3 --last-generations 3
```

After the contributor exits at generation 2, follow the [independent evaluation procedure](../flight-development-v8-staging/validation/README.md). Acceptance alone does not establish a flight improvement.
