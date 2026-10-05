# Cross-platform assignment reproduction

Cloud Build `d3352d9c-d782-4d2e-aa61-fefe0f25135c` ran the same frozen coordinator
and sanitized assignment fixture on `python:3.11-slim-bookworm`, using project
`flyheaven` and its `fly-training-builder` service account. It performed no
deployment, simulation, API mutation or database import.

| Check | Mac Python 3.13.15, arm64 | Linux Python 3.11.16, x86_64 / glibc 2.36 |
|---|---:|---:|
| Regenerated noise values different from stored | 0 / 5,376 | 182 / 5,376 |
| Maximum noise difference | 0 ULP | 2 ULP |
| Regenerated parameter values different | 0 / 5,376 | 178 / 5,376 |
| Maximum parameter difference | 0 ULP | 2 ULP |
| Parameters reconstructed from stored noise | All bit-exact | All bit-exact |
| Hashes of parameters reconstructed from stored noise | All exact | All exact |
| Job identities and seeds | All exact | All exact |

The eight search assignments cover four independent 672-value noise vectors;
positive/negative partners repeat their shared noise in the comparison counts.
The Python source of `random.Random.gauss` has the same SHA256 on both platforms.
The result is consistent with platform math-library rounding, rather than a
different Python Gaussian algorithm. Individual math-library calls were not
isolated in this experiment.

The decisive result is that **stored-noise reconstruction preserves the exact
assigned parameters and their canonical hashes on Linux**. Parameter, hash,
seed, identity and paired-noise consistency checks do not need to be relaxed.
Only comparison with newly generated Gaussian values needs to account for the
observed platform rounding. Two ULPs covers this fixture; it is not a universal
bound for every Python release, math library or future draw.

`comparison.json` contains the concise result and a few numeric examples;
`mac-summary.json` and `linux-summary.json` preserve the individual summaries.
The source hash is pinned in `fixture.json`. This snapshot predates the
coordinator repair, so its raw regeneration measurement is independent of any
new tolerance.

The fixture was extracted read-only from the local `source.sqlite3`, selecting
only the config/hash, two generation centers, and search job IDs, signs, seeds,
noise, parameters and parameter hashes. No SQLite file, tokens, contributors,
results or acceptance metadata were copied. `.gcloudignore` uploads only six
explicit files. Full acceptance-history validation is outside this fixture
because scientific results were intentionally excluded.

To rerun the local comparison:

```sh
.venv/bin/python -B reports/flyheaven-decoder-deployment-20260914/cross-platform-check/check.py
```

The local summaries are excluded from Cloud Build uploads. The Cloud Build
configuration only runs `python -B check.py` and emits one compact
`CROSS_PLATFORM_SUMMARY=` JSON log line, without printing full vectors.
