# Spacious maintained-flight curriculum — staged only

This profile expands the locomotion task to a finite **50 cm nominal radius and 50 cm ceiling**. It is a separate curriculum; passing it would not establish obstacle avoidance, takeoff/landing, height regulation, or success in the original small bowl. No controller, body force, new sensory cue, neural change or native run is introduced here.

Config `maintainedScene` and body metadata `maintained_scene` must both contain exactly:

```json
{"schemaVersion":1,"profile":"spacious-maintained-flight-v1","radiusCm":50,"ceilingCm":50,"floorCapRadiusCm":6.5}
```

Version1 accepts only those finite dimensions and owns/freezes the validated record. Missing on both sides means legacy behavior; disagreement, malformed records, reduced bodies, or non-maintained configuration is rejected. Adding the metadata field changes its hash and therefore the reset's `bodyVariantHash`, config hash and model fingerprint. Model XML, all native body joints/masses, event muscles and neural graph remain unchanged.

The original central bowl is `z=.15+.037*r²` in native centimetres. Its analytic surface, normals and sampled heightfield vertices are unchanged through radius6.5cm; beyond that the floor is flat at1.71325cm. The existing257×257 grid, extent±6.6cm and original height normalization are retained. Four static slabs surround that square out to±51cm. This avoids a larger/coarser heightfield. At the cap, triangulation interpolates across the kink over at most one original grid cell; the analytic cap and exact vertex parity do not imply identical continuous interpolation in that thin boundary band. Apron/heightfield seams can differ by Float32 height rounding and require the later native static geometry check.

The original six fruit objects,189 fruit collision meshes and odor function are unchanged everywhere. This deliberately preserves neural sensory drive before trajectories diverge. Remaining fruit and floor contacts can still end flight; this is a spacious continuation task, not an empty room. Walls retain64segments and.05cm half-thickness, at49.9cm centre radius (49.85cm inner face), spanning z−1..50.15cm. The ceiling is at50.15cm, retaining the old.15cm physical offset. The nominal scored radius is50cm; the unchanged.2cm ceiling allowance applies. All scene solids remain worldbody geoms.

`maintained-flight-objective.js` changes only explicit spatial-bound selection and matching observation-profile validation. The v1 instantaneous vertical-speed gate remains **−1cm/s**. Minimum upcos45°,20rad/s RMS over20ms, inferred support≥.8 over50ms, wing power>.1, zero environmental contact,300rad/s catastrophic limit,.1s overturn limit, minimum height−.5cm, five scored seconds, and current continuous≥1s success are unchanged. The.5s live warmup and3.5cm release are byte-identical. Root's proposed later v2 height-window scorer is intentionally not included in this geometry stage.

Preview packets report the new bounds and cap; the optional renderer includes a radial ring exactly at the cap. The measured observation carries `curriculumScene`; the scorer rejects an omitted/mismatched profile instead of silently using broader bounds. The existing standalone native runner separately rebuilds its saved raw trajectory observation without that extra field: an offline replay must take the explicit profile from the saved plan/config and annotate its input. The actual environment scorer receives it directly.

Validation is pure JavaScript. Eleven tests pass: strict profile/metadata agreement; complete legacy habitat/collision parity; central floor and odor parity; finite walls/apron coverage and identical fruit meshes; per-step legacy scorer parity; preserved flight-quality gates/current-bout success; malformed observation rejection; reversible source deltas; syntax of every runtime module; exact runner delta; and resolution of new canonical modules without creating live files. `validation.json` and `test-results.tap` preserve the results. The read-only plan audit checks all104 source pins and59 assets for every plan. Native compile/contact and live-trajectory validation remain for root after review. No simulation was run by this stage.

```sh
node reports/flight-spacious-curriculum-staging/prepare-sources.mjs
node --import ./reports/flight-spacious-curriculum-staging/test-loader.mjs --test reports/flight-spacious-curriculum-staging/scene.test.mjs reports/flight-spacious-curriculum-staging/runner.test.mjs
node reports/flight-spacious-curriculum-staging/audit-plans.mjs
```

The final three independent one-job plans are in **`plans-v1-final/`**: amplitude.90/frequency log−.03719178739112225, amplitude.80/frequency log0, and amplitude.81/frequency log0. They retain the same other25 parameters and fresh initial neural model/seed. The live warmup can produce different later neural/muscle states because physical sensory feedback depends on the arm; paired initial conditions are not a claim of identical neural trajectories. Root may execute at most one body at a time; preparation itself starts nothing.

The report-local `evaluate.mjs` differs from the existing strict runner only by resolving exact pinned canonical override URLs before Node's file-existence check. All source/hash, backend, loopback, no-coordinator, one-job and native-write guards remain intact. The old `plans-v1/` and `plans-v1-resolved/` are superseded preparation artifacts: the former had the new-module resolution problem, and the latter preceded a name-only duplicate-local syntax fix in `FlyBodyWorld`. Their old pins deliberately fail against final source. Use only the final plan below:

```sh
node reports/flight-spacious-curriculum-staging/evaluate.mjs reports/flight-spacious-curriculum-staging/plans-v1-final/amplitude-0p80-frequency-zero.plan.json reports/flight-spacious-curriculum-staging/amplitude-0p80-frequency-zero-run1
```

Final .80 plan SHA256: `b2e8e778a4acdfd1960edfd6d605759c6c1181bfe712048d6e835ee881360701`. Config SHA256: `439ca67a9f8a4ad7d2efb38a1bb89a3b47096b2b368fe7b36876e5a882722a25`. Model fingerprint: `8d7fa4960c8cfc090ea385dc364f7ade2974ffb23617f890b0dabcdc309fc08e`. Runner SHA256: `9279d4548d9439683cabcdc96b4febc62aa5e647f17d8460f6f40ed153ede9a3`.
