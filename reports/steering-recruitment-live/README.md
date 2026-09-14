# Paired steering recruitment evaluation

The opt-in Hill curve did not improve flight. Both arms used the real BANC/WASM brain and native MuJoCo, with zero claw adhesion, the same code, starting conditions and two previously used selection seeds. All four runs ended in excessive rotation; none succeeded. No optimizer was run.

| Seed | Best flight, legacy / Hill (s) | Takeoff, legacy / Hill | Crash time, legacy / Hill (s) | Return, legacy / Hill |
|---|---|---|---|---|
| 190888 | 0.164 / 0.164 | true / true | 0.384 / 0.384 | -1.508 / -1.508 |
| 290888 | 0.150 / 0.018 | true / false | 0.390 / 0.226 | -1.331 / -3.706 |

The first pair has equal coarse reward metrics but different final poses. The second pair is worse under Hill recruitment. The new mapping remains a diagnostic option; canonical metadata does not enable it.

The Hill candidate uses tied per-type gains matching aggregate mean native muscle force over 100–280 ms of the frozen seed888 capture. Individual side means, startup, temporal variation and live feedback can differ. This does not isolate temporal variability alone. Power, deployment and frequency gains are unchanged. Untouched test seeds were not used.

Each arm stores an immutable bundle, exact plan and per-evaluation records. [comparison.json](comparison.json) records result hashes and vectors. The legacy rerun reproduced the prior v5 generation0 scores and flight metrics exactly on both seeds. These are instrumented native results, not Safari visual observations.

## Separate native Dawn/Metal comparison

Both arms were then run with the same full BANC graph through pinned webgpu0.6.0/Metal on the Apple M2 Pro. This is not Safari and is not assumed numerically equivalent to WASM. All four runs crashed.

| Seed | Best flight, legacy / Hill (s) | Takeoff, legacy / Hill | Crash time, legacy / Hill (s) | Return, legacy / Hill |
|---|---|---|---|---|
| 190888 | 0.102 / 0.108 | false / false | 0.388 / 0.386 | -2.334 / -1.853 |
| 290888 | 0.168 / 0.168 | true / true | 0.384 / 0.388 | -1.022 / -1.088 |

The6ms improvement on one seed and unchanged flight on the other do not establish stable flight or learning. The baseline active execution cost was 14.35 wall seconds per simulated second versus 97.05 for WASM (6.76× faster), excluding setup. This is an observed end-to-end result for these short failing trajectories, not a general GPU benchmark. [comparison-dawn-metal.json](comparison-dawn-metal.json) pins artifacts and backend provenance.
