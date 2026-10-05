# Local all-zero decoder evaluation

This is a fixed-parameter diagnostic, not an optimizer run. The same production
WASM evaluator tests three decoder vectors against three configured held-out
seeds, one fly at a time in Safari. No diagnostic result is submitted to the
training coordinator.

## Conditions

- **Zero:** all 672 learned motor-decoder coefficients are exactly zero.
- **Initial:** the configuration's original vector: 24 power weights equal to
  one and 648 steering coefficients equal to zero.
- **Current:** the generation-7 checkpoint fetched at 2026-09-15 01:27:43 UTC and
  frozen in `checkpoint.json`. Later cloud updates do not change these jobs.

Seeds are 2490888, 2590888, and 2690888. Each seed runs zero, initial, then
current. These are fixed-condition evaluations; their parameters do not update.

Zeroing the decoder does not silence BANC or remove every actuator force. Neural
activity, sensory feedback, muscle dynamics, wing folding servos, leg controls,
contacts, gravity, and the native body remain active. It removes decoded powered
wing output in this model. Actual returned parameter vectors and sampled
effective-power commands are checked separately.

## Evaluator and objective

The original worker and model files come from
`dist/training-lease-client/fruit-fly-training-client-1f3b0935af59`. All 95 manifest
files and 59 model asset pins were checked before launch. The config hash is
`1f3b0935af592a3edf283f02ae2d7eb570fa5c2a6acd225b429b6624d6ed328f`.
Run identities are recorded in `input-verification.json`, `run-identity.json`,
and `plan.json`. Per-file hashes and sizes are in the frozen bundle's
`bundle-manifest.json`.

Each evaluation uses a fresh neural/body state, the declared 2 ms feedback
blocks, 0.5 seconds of live warm-up with the body root restrained, and then up
to 5 seconds of unrestrained scored time. No warm-up time counts as flight.
Neural execution and MuJoCo physics both use WASM. The local runner requests
100% duty and a 1 Hz low-resolution observer preview; it does not replace the
worker's scheduler, physics, or score implementation.

Qualified flight requires no environmental contact, wing power above 0.1,
uprightness within 45 degrees, low angular RMS, sufficient inferred support,
and limited downward COM velocity. It is stricter than simply remaining above
the ground. A physical failure scores `min(3, 1.4 * bestFlightSeconds - 3)`.
Success requires reaching the five-second horizon with a final uninterrupted
qualified flight bout of at least one second. This objective does not evaluate
takeoff, landing, food localization, or feeding.

The seeds perturb the grounded stance used before airborne placement. The
subsequent reset standardizes root position, attitude, and velocity, while leg
posture, actuator state, and the saved rest pose retain seed-dependent
differences. These differences can affect proprioception during warm-up. Neural
initialization itself is deterministic and receives no independent job RNG seed.
The comparison therefore covers three stance-dependent starting conditions in
one fixed scene, not broad disturbances or biological variability.

## Evidence and reproduction

- `results/<jobId>.json`: complete original worker result, including actual
  parameters, provenance, physical outcome, clocks, and score evidence.
- `results/<jobId>.frames.json`: at most 200 saved observer samples, with actual
  simulation timestamps. Samples are not a complete physics-step trace.
- `events.jsonl`, `status.json`: local job progression and latest status.
- `summarize.py`: validates identities, vectors, clocks, completeness, and reward
  arithmetic before calculating matched comparisons. Missing/invalid trials are
  excluded, never assigned fabricated zero scores.
- `telemetry_summary.py`: summarizes recorded power/contact/posture telemetry;
  first sampled contact is not necessarily exact contact onset.

From the repository root:

```sh
.venv/bin/python -B reports/zero-baseline-wasm-20260914/summarize.py
.venv/bin/python -B reports/zero-baseline-wasm-20260914/summarize.py --format json
.venv/bin/python -B reports/zero-baseline-wasm-20260914/telemetry_summary.py
```

The loopback-only runner can be started with:

```sh
node reports/zero-baseline-wasm-20260914/server.mjs --port=7880
```

Open `http://127.0.0.1:7880/` in Safari. Start resumes only missing jobs; it does
not overwrite validated completed results. A new repetition should use a new
evidence directory and explicitly prepared plan. Do not restart the server
during an active job.

## Observation limits

Some trials experienced background suspension/throttling in Safari. Moving the
running local tab into its own window initially restored progress without
reloading the worker. Later, the third initial-control trial slowed again while
Safari's window was inaccessible to the UI tools, but local progress continued.
Wall times include these delays and are not comparable performance benchmarks.
Simulation-time outcomes remain the comparison basis.
Recorded telemetry supports the behavioral analysis; it should not be described
as a continuous visual observation of trials whose preview was hidden.

Three paired seeds are a diagnostic. A better score establishes improvement on
these initial conditions under this objective, not successful general flight or
validation of the underlying biological assumptions.
