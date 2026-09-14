# Individual motor-unit decoder calibration

The opt-in decoder preserves the identities of 48 prepared wing motor neurons.
Its 672 coefficients describe a bounded numerical interface to wing controls;
they are not measured muscle physiology. Offline calibration fits paired input
and control records. Native force, body response and closed-loop flight require
separate validation before a fitted checkpoint is used for training.

The decoder has 24 nonnegative power weights in `[0, 4]`, initially 1, and 648
signed steering coefficients in `[-0.25, 0.25]`, initially 0. Input units are in
the exact sorted prepared motor-index order. Each steering unit influences only
the three ipsilateral wing-control axes. Its nine causal features combine
excitation at lags 0, 1 and 4 ms with `1`, `sin(phase)` and `cos(phase)`.
Power features divide each individual excitation by twice its muscle-family
size: five DLM or seven DVM units on that side. This recovers the initial
half-DLM-mean plus half-DVM-mean weighting without discarding unit identities.

The runtime contract defines the authoritative parameter ordering. The fitter
uses `decoder.advance()` and `decoder.features()` directly; it does not copy the
FIR or phase calculations. Controls are subsequently clipped to `[0, 1]` for
power and `[-0.25, 0.25]` for steering. These coefficients neither change BANC
connectivity nor establish a new muscle or aerodynamic model.

## Live BANC integration

The opt-in training contract is `banc-masked-motor-decoder-v1`. It freezes neural
parameters and requires the explicit individual wing-event interface. The body
advances the decoder once per completed 1 ms excitation interval and samples
its output at the existing 200 microsecond wing-control cadence. The decoder
receives individual excitation histories and wingbeat phase; body orientation,
task targets and reward are not decoder inputs.

Prepared BANC identities establish the neuron, muscle group and wing side. The
learned association from each steering muscle to that wing's three native
actuator axes is an engineering assumption, not an identified anatomical
moment arm. The learned output is in native actuator-control units, not newtons
or newton-metres. Each steering residual is multiplied by that side's deployed
power and the resulting actuator command is clipped to the native range.

This first implementation covers the wings. Existing nonwing muscle controls,
the full native articulated body, aerodynamic model and wingbeat tables remain
in use. Individual wing excitation replaces the old aggregated wing-force
input for this decoder; it does not turn the wing actuators into Hill muscles.
Zero decoded drive removes powered beating and learned steering, but the
existing folding/position-restoring servo can still exert torque. The new mode
bypasses the legacy per-muscle steering basis and its optional force reference.

## Reproduce the local experiment

Install the repository's prepared BANC/body assets and native dependencies
first. Native coordinated evaluation additionally uses the existing pinned
Dawn/Metal installation described in `reports/native-webgpu-tooling/README.md`.
It is a local macOS backend; Safari and WASM are not assumed numerically
interchangeable with it.

Prepare a new directory, then start the development coordinator:

```sh
node scripts/prepare-motor-decoder-experiment.mjs reports/motor-decoder-v1/run-002
.venv/bin/python scripts/serve-training-dev.py \
  --port 7872 \
  --database reports/motor-decoder-v1/run-002/coordinator.sqlite3 \
  --bundle reports/motor-decoder-v1/run-002/bundle.json
```

In a second terminal, evaluate one fly at a time:

```sh
node scripts/contribute-training-native.mjs http://127.0.0.1:7872/ \
  reports/motor-decoder-v1/run-002/contributions 12 \
  reports/motor-decoder-v1/run-002/backend-plan.json
```

The plan stops after one completed generation: four search evaluations followed
by six candidate/incumbent evaluations on three matching seeds. Only a positive
mean paired score difference changes the incumbent. This small comparison is
an update guard, not statistical proof of generalization. The SQLite database
retains assignments and results; `contributions/checkpoint.json` and
`contributions/parameter-updates.jsonl` record the selected vector and exact
parameter changes. Do not edit pinned runtime files during an evaluation.

This fixture starts airborne after 500 ms of live BANC warmup with a restrained
root. The restraint ends before scoring. It measures maintained flight and
awards no takeoff or landing credit. Claw adhesion is disabled in the fixture;
ordinary collision and friction remain. No gripping or climbing claim follows.
The preparation command leaves the canonical training profile and deployment
unchanged.

Preparation reads the compact `configs/training-motor-decoder-v1.json` template
and pinned repository models, not an earlier experiment bundle. To initialize
a new run from a fitted decoder, append `--initial=PATH_TO_CHECKPOINT` to the
preparation command. The checkpoint must match the exact IO and decoder
contract hashes and all 672 bounds. Its bytes and provenance are archived as
an unverified initialization; fitting does not bypass behavioral evaluation.

Run the independent mechanical routing assay with:

```sh
node scripts/test-motor-decoder-native.mjs reports/motor-decoder-v1/native-check.json
```

It uses synthetic motor events and a restrained root to measure native wrench
changes. Passing establishes neuron/side selectivity, signed responses,
phase sensitivity and finite integration; it does not establish stable free
flight or a physiological calibration.

## Fit paired records

Run from the repository root with a new output directory:

```sh
node scripts/calibrate-motor-decoder.mjs \
  --dataset=reports/motor-decoder-v1/paired-native.json \
  --output=reports/motor-decoder-v1/native-fit-001
```

Optional arguments are `--io=PATH`, `--initial=checkpoint.json`, `--ridge=1e-5`,
`--sweeps=1000` and `--tolerance=1e-9`. There is no implicit dataset, simulator
allocation, optimizer upload or checkpoint promotion. Existing output
directories are rejected.

The input JSON has this structure; shortenings below are illustrative:

```json
{
  "schemaVersion": 1,
  "kind": "paired-motor-decoder-calibration",
  "intervalMs": 1,
  "targetSpace": "pre-clamp-decoder",
  "ioSha256": "SHA256_OF_EXACT_PREPARED_IO_BYTES",
  "indices": ["48 exact sorted numeric motor indices"],
  "provenance": {
    "kind": "native-control-calibration",
    "description": "How independently calibrated commands were obtained",
    "sources": {"reports/example/measurement.json": "SHA256"}
  },
  "trials": [{
    "id": "trial-001",
    "groupId": "paired-experiment-001",
    "split": "train",
    "history": [],
    "rows": [{
      "step": 1,
      "unitExcitation": ["48 numeric excitations in [0,1]"],
      "wingPhaseRadians": 0.4,
      "targetSteering": [[0.01, -0.02, 0], [0, 0.01, -0.01]],
      "targetPower": [0.8, 0.8]
    }]
  }]
}
```

Provide at least one training and one validation trial. `history` contains zero
to five immediately preceding 1 ms excitation vectors, oldest first. An empty
history explicitly starts from zero decoder history. Each row advances the
decoder once with its completed 1 ms excitation, then evaluates features at
that row's phase. Rows must be contiguous; do not silently drop uninteresting
intervals or interpolate spikes. Initial-history rows have no fitted target.

`targetSteering` contains raw decoder commands in axis order yaw, roll, pitch,
before the decoder's output clamp. `targetPower` is optional, but must occur on
every row or none; omitting it preserves the initial 24 power coefficients
exactly. A clipped measured control is not an unclipped target: clipping loses
information about coefficients beyond the bound. Converting a measured wrench
to a control target needs a separately documented inverse/mechanical
calibration; the fitter does not invent one.

Use a common `groupId` for dependent trials, including a paired baseline and
perturbation, repeated captures of the same sequence, and overlapping segments.
All members of a group must share a split. Keep entire independent sequences,
phases, frequencies or stimulus amplitudes for validation. Randomly splitting
adjacent time rows leaks the FIR history and overstates generalization.

The objective is equal-weight mean squared control error per training trial
plus ridge distance from the initial coefficient vector. Coordinate descent
enforces bounds during fitting. Validation targets never determine updates or
stopping. Choose hyperparameters on a separate development split if comparing
multiple fits; the provided validation set then ceases to be a final test.

## Artifacts and limits

Each output directory stores the exact dataset, decoder contract, an
`unverified` checkpoint and a report. The report includes source/IO/data hashes,
training and validation errors before and after fitting, per-trial errors,
clipped-control errors, convergence, bound hits, unexcited columns and a
weighted numerical feature-rank estimate. That estimate is a pivoted-QR span
check, not a singular-value decomposition or proof of biological
identifiability. Correlated inputs can leave many coefficient combinations
equivalent despite low prediction error.

`completed` means the offline procedure finished; inspect the checkpoint's
`converged` field as well. A sweep cap can stop an improving fit before the
specified numerical tolerance. Increase a predeclared sweep budget based on
training convergence, without selecting it to improve validation scores.

No fitted parameter is automatically physiological. Unit gains, downstream
muscle force scales, moment arms and redundant contributions can trade off.
Native controlled stimulation should test signed left/right responses across
phase and amplitude, with fixed mechanical initial states and independently
measured forces or accelerations. Hold all paired records together when
splitting them. A small steady-state wrench Jacobian alone cannot identify all
672 temporal coefficients.

## Explicit synthetic plumbing check

```sh
node scripts/calibrate-motor-decoder.mjs \
  --self-test --output=reports/motor-decoder-v1/synthetic-001
```

This creates deterministic independent input trials and targets from a known
synthetic decoder, fits four trials and evaluates two unseen trials. It checks
signed/boxed fitting, an unexcited coefficient, invalid data, grouped splitting,
agreement between linear features and executed controls, all 672 coefficient
updates, and at least 95% reduction in held-out raw prediction error. Outputs
are explicitly marked `synthetic-teacher`; success establishes fitting
plumbing, not native mechanics, anatomical validity or stable flight.

The completed `reports/motor-decoder-v1/synthetic-003` check used
`--sweeps=5000`. All eight output fits converged, all 672 coefficients changed,
and held-out raw RMSE decreased from `0.0533531` to `0.000214295` in approximately
5 seconds. The observed feature spans were 108/108 steering features per side
and 12/12 power features per side. The earlier `synthetic-001` artifact preserves
the default 1000-sweep run, which improved predictions but reached its steering
iteration limit. Neither artifact contains native or neural simulation.
The final check also rejects unsupported initial-checkpoint schemas, profiles,
IO/contract hashes and vector lengths.

## Completed native and BANC checks

`reports/motor-decoder-v1/result.json` records the implementation checks and the
first one-fly experiment. All 33 focused JavaScript tests, eight focused Python
tests and eight native mechanical gates passed. Decoder sampling alone took
about 10.4 microseconds per call on this machine; this excludes neural and body
simulation. The final native assay reproduced the original measured wrenches
exactly after the optional legacy-force-reference bypass was added.

The live BANC experiment completed four search evaluations and six comparison
evaluations. Its nominated candidate obtained these qualified flight durations
on the three matching comparison seeds:

| Seed | Candidate | Incumbent |
| --- | ---: | ---: |
| 3559293007 | 0.094 s | 0.044 s |
| 3632537482 | 0.082 s | 0.152 s |
| 3705781957 | 0.244 s | 0.252 s |

Every trial terminated for excessive rotation. Mean score was -2.804 for the
candidate versus -2.790933 for the incumbent, so the coordinator rejected the
proposal and changed **0 of 672 checkpoint parameters**. The best search trial
had 0.402 s of qualified flight, but that result did not generalize across this
small comparison. This is functioning evaluation/update plumbing, not learned
stable flight. The ten evaluations took approximately 313 active wall seconds.

The baseline-only input audit found 46/48 units fired across 1.17 s. Both iii3
units were silent, making 54 coefficient features exactly unexcited. Additional
sparse channels brought the number of features below a declared `1e-12`
magnitude threshold to 135 during the scored window. This is a coverage check,
not a rank estimate or evidence that all remaining coefficients can be
independently identified. No real-kinematic paired dataset has yet been fitted.

The BANC experiment retains its original source fingerprints. The final
compatibility fix bypasses optional legacy steering-reference metadata, which
was absent in that experiment; the final native assay and focused regressions
cover the fix. A newly prepared run pins the final code. Historical bundles
must not be rehashed in place to disguise a code change.

To inspect completed native poses without allocating another simulated fly:

```sh
node scripts/prepare-motor-decoder-preview.mjs
.venv/bin/python -m http.server 7873 --bind 127.0.0.1 \
  --directory reports/motor-decoder-v1/preview
```

Open `http://127.0.0.1:7873/` in Safari. This report uses the existing theme and
low-resolution 3D renderer. It plays discrete recorded poses, without invented
interpolation, and labels the recording explicitly. Rerun the generator and
use Refresh recordings to include newly completed trials. This does not modify
the product UI or run a training worker.
