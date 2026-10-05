# Pre-audit one-fly performance after controller reference integration

This historical measurement predates the extended visual audit and its contact-memory, sensory and rendering fixes. The measured BANC path was within the requested factor-of-ten neural-throughput target against the retained FlyWire backend. This measurement includes the corrected per-neural-block body feedback, haltere motor mappings, original 3D renderer, rendered eyes, graded/color vision and open anatomy inspection. It does not measure the optional published-controller reference trial.

| Neural mode | FlyWire neural speed | BANC neural speed | BANC slowdown | FlyWire display FPS | BANC display FPS |
|---|---:|---:|---:|---:|---:|
| Reference | 0.0807× real time | 0.0341× real time | 2.37× | 53.5 | 59.9 |
| Fast | 0.0945× real time | 0.0365× real time | 2.59× | 14.8 | 50.8 |

All four samples passed the harness's sensory, anatomy, neural-progress and error checks. These are one 20-second measurement window per setting on the shared Apple M2 Pro, with one fly and a fresh browser per sample. They establish the requested order of magnitude in this run, not identical performance or real-time neural simulation. Display frame rates varied substantially, particularly for the old Fast backend. Source and graph hashes, raw counters, memory measurements and settings are retained in [the machine-readable report](transplant-performance-reference-reuse.json).

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs \
  node scripts/benchmark-transplant.mjs --populations=1 --seconds=20 \
  --output=reports/transplant-performance-reference-reuse.json
```

The separate released flight reference completed 1.1988 simulated seconds in 4.84 wall seconds in Chromium (0.248× real time, excluding rendering and BANC). It uses a learned controller and starts airborne, with floor contacts disabled. [Flight reference result](flybody-reference-browser.json). This faster standalone body trial is not substituted for the default whole-brain benchmark above.
