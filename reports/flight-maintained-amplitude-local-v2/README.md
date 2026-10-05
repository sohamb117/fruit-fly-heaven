# Full-body maintained-flight experiment

**Paused on 2026-09-14 at 16:02 UTC for architectural review.** There are 19 accepted evaluations, one rejected generation and zero saved-coordinate changes. The active twentieth evaluation was cancelled and not uploaded; its lease was released. The second generation and reserved final comparison are incomplete. See `pause-record.json`, `checkpoint-paused.json` and `training-paused.json`. Do not interpret this as a completed two-generation experiment or automatically resume it.

This is the executed local experiment following the mechanical transfer-function repair. The preparation snapshot is `declaration.json`; its `databaseCreated: false` and `simulationsExecuted: 0` describe preparation time, not the subsequent run.

The canonical implementation adds an explicit amplitude-normalized wing-power profile, a live airborne warmup and release, a spacious flight scene, and a five-second maintained-flight objective. The full native body remains articulated. BANC motor events drive the existing muscles and native wing mechanics. No root correction is applied after release. This is a modeled calibration, not a claim that the transfer function is measured physiology.

Run conditions:

- One native MuJoCo/WASM fly with actual Dawn/Metal neural execution.
- 0.5 seconds of unscored live warmup with the root restrained, followed by up to five scored seconds fully released.
- Initial amplitude 0.80 and frequency multiplier 1; 27 interpreter coefficients.
- Two coordinator generations, each with eight search jobs and up to six paired acceptance jobs. A result upload is not a checkpoint promotion.
- The initial and final saved vectors are compared afterward on three reserved paired seeds. Those seeds do not enter search or acceptance.

The implementation intentionally tests maintained flight from an airborne reset. It does not establish takeoff, landing, food localization, or feeding. Success requires reaching the full horizon and finishing with at least one second of uninterrupted qualified flight. Best or total airtime alone does not establish success.

## Evidence

- `checkpoint-initial.json`: exact starting vector.
- `contributions/*.json`: assigned vectors, actual evaluations, timing, sparse preview frames and backend provenance.
- `contributions/checkpoint.json`: latest saved coordinator checkpoint.
- `contributions/parameter-updates.jsonl`: actual saved-coordinate deltas after completed generations, including zero deltas for rejections.
- `training-progress.json`: read-only database summary; inspect its timestamp before treating it as current.
- `actual-loop-initial-audit.json`: independent first-five-evaluation audit, with all 70 checks passing.
- `outcome-audit.py`: independent reconstruction of assignments, acceptance decisions, saved-coordinate changes and the reserved final comparison; usage is in `outcome-audit.README.md`.
- `sources/` and the hashes in `declaration.json`: archived small sources; large graph arrays and native binaries are pinned in place as described by the declaration.

Monitor without allocating another fly:

```sh
.venv/bin/python scripts/report-flight-training.py reports/flight-maintained-amplitude-local-v2/coordinator.sqlite3 --last-generations 2 --output reports/flight-maintained-amplitude-local-v2/training-progress.json
```

The exact start, stop and final-comparison recipe is in `../flight-maintained-training-plan/README.md`. The run uses the local coordinator at port 7862. Production hosting is unchanged.

## Interpretation limits

The current coordinator chooses the highest-scoring tested vector and evaluates it against the incumbent on three matched seeds. It does not use an ES gradient for this guarded configuration. A changed coefficient does not establish that the corresponding muscle caused the improvement. Some steering inputs were inactive or very small in the recorded diagnostic windows.

The seeds vary inherited stance geometry. The airborne root pose, root velocity and neural initial state are fixed. Frequency also changes the oscillator phase at the fixed 500 ms release: a one-standard-deviation frequency perturbation changes accumulated release phase by about 3.7 radians. A successful result in this setup therefore still needs phase-robustness testing before claiming general stabilization. The proposed follow-up is documented separately in `../flight-release-phase-protocol/README.md`; it has not been executed here.

Return depends on the terminal continuous bout for full-horizon episodes and on the best preceding bout minus a penalty for physical failures. It is not identical to total qualified airtime or survival time. Report those measures together.
