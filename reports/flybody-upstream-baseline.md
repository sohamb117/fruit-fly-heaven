# Released FlyBody flight controller: reproduced locally

The existing published setup works. The released, deterministic FlyBody flight
policy completed **1.1988 seconds of continuous free flight** on the upstream
straight-flight reference, with no applied root forces or root actuator.
This reproduces a trained controller baseline; it does not establish BANC-driven
flight, takeoff, landing, or food seeking.

| Measurement | Original episode | Extended horizon |
|---|---:|---:|
| Simulated flight | 0.5988 s | 1.1988 s |
| Control steps | 2,994 | 5,994 |
| Mean upstream reward | 0.902825 | 0.902922 |
| Maximum reference position error | 0.407 mm | 0.407 mm |
| Minimum root height | 10.118 mm | 10.118 mm |
| Maximum orientation error from reference | 11.714° | 11.714° |
| Applied root force/torque | 0 | 0 |
| Direct root actuator force | 0 | 0 |
| Early failure | none | none |
| Wall time, policy and environment, no rendering | 9.451 s | 18.914 s |

The original upstream task is configured with a 0.6 s horizon and reserves six
control samples for reference previews, so it completes at 0.5988 s. The longer
run changes only the task and environment time limits to 1.2 s. Both runs use
seed 42, one physical fly, the official 20 cm/s straight reference, 50 µs physics,
and 200 µs control. The upstream reference ghost cannot collide with the fly.

The policy takes privileged future reference positions/orientations and
proprioceptive/vestibular observations. Its 12 outputs operate the published
head/abdomen/wing actuators and wingbeat-frequency generator. The task starts
airborne with forward velocity, disables the legs and floor contacts, and does
not implement muscle-specific BANC physiology. There is no additional stabilizer
or fitted local wing adapter in these runs.

Evidence: [original episode](flybody-upstream-baseline.json),
[continuous episode](flybody-upstream-baseline-continuous.json),
[native/Node/Chromium replay](flybody-reference-replay.json).

## Exact native-to-WASM reference

`models/flybody-flight-reference.xml` preserves the upstream task's body frames,
joints, actuators, sensors, primitive contact/fluid geoms, and dynamics. Only
non-colliding visual mesh geoms and their assets are removed. Their exact
compiled inertias are inserted explicitly. The native mesh-free model has zero
qpos/qvel and mean sensor error across the first ten control steps compared with
the full visual model.

`models/flybody-flight-reference.json` contains the initial state, observation
and action mappings, original wingbeat tables and phase, reference trajectory,
and model hashes. The accompanying fixtures contain original TensorFlow outputs,
full native controls, all states, and observations for independent replay.

Observation timing matters: the original dm_control task uses `mj_step2` then
`mj_step1`. Its ghost pose write marks the physics state dirty, and the first
sensor read after substep zero calls `mj_forward`. Reproducing this refresh is
necessary to match accelerometer inputs, which are averaged over four substeps.

## Reproduction

Source checkout: [TuragaLab/flybody](https://github.com/TuragaLab/flybody), revision
`d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
Released assets: [trained policies](https://ndownloader.figshare.com/files/44815195)
and [flight data and wing waveform](https://ndownloader.figshare.com/files/51196859).
The source code is Apache-2.0; published data/checkpoint licensing is separate.

Expected extracted files:

- `data/raw/flybody-policy/flight/saved_model.pb` and its `variables/` directory.
- `data/raw/flybody-baseline/wing_pattern_fmech.npy`.

```sh
uv venv --python 3.11 /tmp/flybody-baseline-venv
uv pip install --python /tmp/flybody-baseline-venv/bin/python -r scripts/flybody-baseline-lock.txt
/tmp/flybody-baseline-venv/bin/python scripts/run-flybody-baseline.py --episodes 1
/tmp/flybody-baseline-venv/bin/python scripts/run-flybody-baseline.py --episodes 1 --duration 1.2 --output reports/flybody-upstream-baseline-continuous.json --fixtures reports/flybody-upstream-policy-fixtures-continuous.npz
/tmp/flybody-baseline-venv/bin/python scripts/export-flybody-flight-reference.py
node scripts/verify-flybody-reference-replay.mjs
```

The browser replay expects the normal local server on port 7842 and the existing
Playwright installation; `--node-only` runs only the Node WASM replay. TensorFlow
2.15.1 and TensorFlow Probability 0.23.0 support local Apple Silicon inference.
The loader aliases the two old serialized distribution type names to their
current classes; it does not modify weights or graph operations.
