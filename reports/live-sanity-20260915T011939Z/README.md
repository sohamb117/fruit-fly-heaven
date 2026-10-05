# Live training audit — September 14, 2026

Snapshot: 2026-09-15T01:20:43.248328+00:00 (approximately 9:20 PM New York time). Read-only Firestore transaction and public endpoint downloads.

The recorded data are internally consistent. Parameter selection is functioning, but the data do not yet demonstrate sustained flight or a reliable improvement from the initial controller.

## Training state

- 70 accepted evaluations in the audited snapshot; generation 7 has started.
- Six retained checkpoint updates; one rejected proposal. The rejected generation preserved its center exactly.
- All 70 applied vectors match the assigned vector and hash; all use the pinned WASM version.
- All 672 current coefficients differ from their initial values. Each accepted whole-vector perturbation changed all 672, so this is not evidence that every coefficient independently learned.
- Power weights range from 0.987489 to 1.013408 (initially 1); steering values range from -0.008384 to 0.007290 (initially 0).
- Config, model and run identity are unchanged.

## Recorded behavior

| Metric | Result |
| --- | ---: |
| Successful trials | 0 / 70 |
| Excessive-rotation terminations | 70 / 70 |
| Median longest qualified flight bout | 0.182 s |
| Longest qualified bout observed | 0.596 s, in generation 0 |
| Median scored simulation duration | 0.797 s |
| Longest scored simulation duration | 2.594 s |
| Median elapsed evaluation wall time | 240.784 s |

Every recorded score agrees with `1.4 * bestFlightSeconds - 3`, and every final angular speed exceeds the 300 rad/s termination threshold. All tested clocks, counters, finite-state flags, release records, absence of external-force flags, and restraint-release flags are consistent. This checks recorded evidence, not an independent rerun.

First ten versus most recent ten audited trials: mean score -2.69396 to -2.69116, and mean best bout 0.2186 to 0.2206 seconds. This is essentially flat. These groups contain different candidates and seeds, so the comparison is descriptive rather than a controlled estimate of learning.

## Paired checkpoint comparisons

Each comparison evaluates the nominated candidate and incumbent on three matching fresh seeds. Positive mean difference retains the candidate; any positive mean is sufficient.

| Generation | Candidate minus incumbent | Seeds won | Decision |
| --- | ---: | ---: | --- |
| 0 | +0.248267 | 2 / 3 | accepted |
| 1 | +0.034533 | 2 / 3 | accepted |
| 2 | +0.010267 | 2 / 3 | accepted |
| 3 | +0.124133 | 2 / 3 | accepted |
| 4 | +0.238933 | 2 / 3 | accepted |
| 5 | -0.189467 | 0 / 3 | rejected |
| 6 | +0.227733 | 2 / 3 | accepted |

The latest accepted proposal improved its paired mean by 0.227733 score units, equivalent to 0.162667 seconds of best qualifying flight under this failure-only reward. It won two seeds and lost one. This is encouraging local evidence; three seeds and repeated selection do not establish final-controller generalization. No exact parameter-vector/seed combination in this snapshot was independently repeated.

## Interpretation limits

- These trials assess maintenance after an imposed airborne release and 0.5 seconds of restrained live warmup. Takeoff, landing and food behavior are not evaluated. All 70 release root poses and velocities are identical; other seeded internal/body state may still differ.
- All-zero controls have not been run. A safe descent followed by survival to the time limit can score 0, exceeding current crash scores. The observed 70 trials all crashed, so this loophole is potential rather than an observed explanation of current scores.
- A matched, held-out seed comparison of initial, current retained, and all-zero controllers is still the useful next behavioral test. The requested baseline remains on hold.

## Service health

The coordinator reports a 180-second lease timeout. At the first queue read there were three leased jobs and one pending job; another evaluation subsequently completed and available work was assigned. Two heartbeat requests returned HTTP 500 at 01:20:22 and 01:20:27 UTC; subsequent heartbeat success and additional accepted results demonstrate continued operation, but do not explain the errors. The current handler omits exception detail. A local mock reproduced a possible exception-masking path when Firestore rollback fails during an API error; production logs do not establish that as the cause. See `recent-heartbeats.json` and the lease snapshots for bounded observations.

## Files

- `audit.json`: assignment, provenance, history and checkpoint integrity checks.
- `statistics.json`: behavior and timing aggregates.
- `measurements.json`: sanitized per-evaluation measurements and generation decisions.
- `checkpoint.json`: downloaded current checkpoint.
- `leases*.json`: timestamped assignment observations.

No source code, deployments, leases or experiment results were changed by this audit.
