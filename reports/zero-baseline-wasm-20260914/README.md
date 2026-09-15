# All-zero decoder evaluation

**All three requested zero-baseline evaluations completed.** Every returned
decoder vector contained exactly 672 zero coefficients. All three trials had
zero qualified flight and scored **−3**, terminating **160 ms after release**
for remaining inverted for 100 ms.

The brain and body still ran. Sampled decoded power, deployment, and effective
wing power stayed zero. Wing folding servos and other fixed body mechanisms
remained active, so this is not a zero-force body or a silent-neuron control.

## Results

All nine planned zero/initial/generation-7 comparisons are complete and
validated. The table reports the **longest uninterrupted qualified flight bout**,
which is stricter than simply being above the ground.

| Seed | Zero: score / best flight | Initial: score / best flight | Generation 7: score / best flight |
|---|---:|---:|---:|
| 2490888 | −3.0000 / 0 ms | −2.9804 / 14 ms | −2.3000 / 500 ms |
| 2590888 | −3.0000 / 0 ms | −2.8180 / 130 ms | −3.0000 / 0 ms |
| 2690888 | −3.0000 / 0 ms | −2.6500 / 250 ms | −2.9356 / 46 ms |

All completed powered controls failed for excessive rotation. None reached the
five-second horizon or met the maintained-flight success condition.

The three matched initial/current comparisons have mean scores **−2.8161 and
−2.7452**, respectively, and mean best bouts of **131.3 ms and 182.0 ms**.
Generation 7 improves one seed (2490888) and regresses on two (2590888 and
2690888). Its mean score improvement of 0.0709 and mean best-bout improvement
of 50.7 ms do not establish reliable improvement: all six powered controls
still fail from excessive rotation.

## What the baseline establishes

Zeroing these coefficients disables decoded powered wing actuation as intended
and produces worse maintained-flight scores than all three initial-vector
controls. The coefficients have a real effect on the measured motion. This
does not validate the biological decoder assumptions or prove that the current
training procedure will converge to reliable flight.

The first generation-7 trial did generate lift: sampled COM height rose from
about 3.4 cm to 4.6 cm during its longest qualifying bout. It then lost stability
and failed at 974 ms. On the second seed, generation 7 earned no qualified flight
and failed at 180 ms, compared with 862 ms for the initial controller.

The zero trials fell onto food and overturned. Their first saved food-contact
samples occur at 56–60 ms. These are sampled telemetry observations, not exact
contact-onset measurements or continuous visual observations.

## Final Safari completion

The third initial-vector trial eventually resumed and completed at 0.908 scored
seconds, with a 250 ms best bout and score −2.65. Its long wall time includes
the suspension described in the protocol.

The final generation-7 trial completed in Safari at **2026-09-15
02:57:44.469 UTC** (the loopback server's result event). It scored **−2.9356**,
with **46 ms** of qualified flight, and failed for excessive rotation at
**260 ms** after release. The returned wall time is 310.58248 seconds.

The server recorded a fresh begin at **02:52:27.381 UTC** and readiness at
**02:52:33.731 UTC**, following the earlier unfinished begin at 02:30:49.557.
This is the later Safari run; it is not claimed to be an uninterrupted resume
of the earlier attempt. Earlier computer-use attempts reported
`cgWindowNotFound`. The recorded events do not identify the cause of the pauses.

The original result in `results/current-seed-2690888.json` was received through
the local Safari harness and saved before the server's result event. Its 189
saved frames include 70 scored samples. No Node result was copied into this
directory. An independent Node WASM run saved at **02:50:07.576 UTC** in
`../flight-decoder-feasibility-20260914/live-generation7-2690888-a-2690888.evaluation.json`
has the same score, 46 ms best bout and 260 ms termination, but a different wall
time (113.535686 seconds) and small floating-point differences in final physical
observations. This one matching outcome is not a general browser-equivalence
claim.

The loopback runner at <http://127.0.0.1:7880/> now reports zero remaining jobs
and no active trial. Completed results are persisted independently of browser
state.

## Files and validation

- [Protocol](PROTOCOL.md): model, conditions, objective, seed semantics, and
  reproduction commands.
- [Validated summary](summary.md) and [machine-readable summary](summary.json):
  nine complete trials, zero missing, zero invalid.
- [Telemetry summary](telemetry-summary.json): sampled power, contact timing,
  final physical state, and all-zero command checks.
- [Frozen checkpoint](checkpoint.json), [plan](plan.json), and
  [input verification](input-verification.json).
- `results/`: original WASM results and at most 200 saved observer frames per
  completed trial.

Each counted result matches the assigned vector, seed, configuration, model and
WASM identities. Warm-up and scored neural/native clocks match. The recorded
reward agrees with the physical outcome and best-flight counter. No counted
result is partial, cancelled, or a simulation error. The local diagnostic does
not request cloud leases or submit results to production training.

Regenerate the summaries from the completed evidence:

```sh
.venv/bin/python -B reports/zero-baseline-wasm-20260914/summarize.py > reports/zero-baseline-wasm-20260914/summary.md
.venv/bin/python -B reports/zero-baseline-wasm-20260914/summarize.py --format json > reports/zero-baseline-wasm-20260914/summary.json
.venv/bin/python -B reports/zero-baseline-wasm-20260914/telemetry_summary.py > reports/zero-baseline-wasm-20260914/telemetry-summary.json
```

Wall-clock timings include Safari background delays and are not a performance
benchmark. These three configured held-out seeds vary inherited leg stance in
one fixed scene; they do not vary neural initialization, root release pose,
internal state, or external disturbances.
