Prepared candidate only. The running renderer at `web/training/retinal-sensor.js` was not edited; its SHA-256 remains `0b4cba21ffdfc1ad48a73177df3dd0822868648eba80600ebe5d2688d953a545`.

The candidate derives each pixel's world-space footprint from analytic camera-ray differentials and the existing surface normal. Each sinusoidal texture component receives the corresponding local box-average response plus a smooth cutoff before pixel Nyquist. This accounts for distance and grazing angle without casting additional rays. Products of sinusoids are decomposed into their sum/difference frequencies before filtering.

`textureAntialias` defaults to **false** for old-profile parity. The next explicitly versioned retinal profile should pass **`textureAntialias: true`** through the existing sensor settings. This changes sensory samples, so the new profile needs a new source/config snapshot and its own short assay; it must not relabel the running v1 results.

`antialias-comparison.png` shows the exact saved release pose (native time 0.509999999999952 s): columns are original, analytic candidate, and offline 16-sample-per-pixel reference; rows are left and right eyes. The filter removes the strong wall/floor moiré while preserving resolved fruit detail. Geometry silhouettes remain point-sampled.

- Original versus reference RGB MSE: **43.216**.
- Candidate versus reference RGB MSE: **16.284**, a **62.3% reduction**.
- Native pose benchmark, six measured frames after two warmups: **85.12 ms original / 84.41 ms candidate**, during other running processes. This supports similar cost, not a reliable speedup claim.
- Both versions cast **65,536 total rays**; geometry hit counts are identical.
- Disabled candidate is byte-identical to the original RGB/luminance images.
- Three focused tests pass: disabled identity/geometry, cutoff behavior, and footprint agreement with finite ray-plane derivatives including grazing stretch.

After the pinned runs are complete, apply the checked patch from the repository root:

```sh
git apply --check reports/sensory-feedback-20260915/retinal-antialias.patch
git apply reports/sensory-feedback-20260915/retinal-antialias.patch
node --test web/test/compact-retina.test.mjs
node --test reports/sensory-feedback-20260915/antialias-check.mjs
```

The patch only changes `web/training/retinal-sensor.js`. Profile schema/config metadata, `config.vision`, and new pinned evaluation snapshots remain the root agent's integration work.

Reproduction uses the existing saved body frame only; no neural or physics step is run. `antialias-check.mjs` regenerates PNG/PPM comparisons and `antialias-report.json`, including input/module hashes. The high-resolution reference is an offline diagnostic and is never used by the sensory renderer.

Limits: this filters procedural texture, not geometric silhouettes, shadows or occlusion. Surface curvature and angular wall texture use a local tangent/phase approximation. The deliberate pre-Nyquist cutoff is smoother than a finite box-filtered reference. This is an optical engineering improvement, not a validated fly photoreceptor model.
