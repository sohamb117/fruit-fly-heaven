# Live BANC airborne-start diagnostics

These are completed native Dawn/Metal + MuJoCo runs of one fly, not optimizer updates. Each maintained run uses the complete native body, the live BANC/event/muscle feedback loop, a predeclared 500 ms root-held warm-up, and up to five scored seconds after release. All nonwing joints remain dynamic. No root corrections or applied external forces occur after release. All completed runs have 10,000 warm-up native steps and an unchanged cumulative root-write count of 20,001 after release.

The original grounded evaluator is preserved. `grounded-control-run1` reproduces its reference physics SHA **c8abcff173e74f7a118005b2349774b587492ac0c4447bb1b303f8af0806b9c9**, all 368 motor packets exactly, 0.504 s best flight, and the original 0.734 s physical failure. Thus the diagnostic loader and observer did not change that control's physical trajectory.

| Maintained arm | Best continuous qualified flight | First sampled environment contact | End | Result |
| --- | ---: | ---: | ---: | --- |
| Original retained interpreter | 0.324 s | 0.374 s | 0.524 s | Excessive rotation |
| Ground-calibrated affine transfer | 0.164 s | 0.644 s | 0.652 s | Excessive rotation |
| Original, power log gain 0 | 0 | 0.162 s | 0.192 s | Excessive rotation |
| Original, power log gain 0.1 | 0.164 s | 0.706 s | 0.864 s | Excessive rotation |
| Original, power log gain 0.2 | 0.342 s | 0.396 s | 0.522 s | Excessive rotation |
| Original, power log gain 0.3 | 0.334 s | 0.394 s | 0.480 s | Excessive rotation |

All times in this table begin at release. Each `*-run1/result.json` contains its exact plan/config/source commitments, motor packets, 2 ms native state trajectory, frames, final score and restraint audit. The associated `summary.json` is produced by `analyze.mjs`, which reconstructs the maintained scorer exactly from saved observations. A physical failure completes a diagnostic but is not task success. No arm completed the five-second flight target.

The original airborne control remained qualified until the first sampled contact at a COM height of 5.609 cm, next to the 5.6 cm ceiling. Its first sampled tilt beyond 45 degrees occurred at 0.380 s, after contact. This differs from the grounded experiment and does not establish an initial free-flight attitude failure. The lower-power 0.1 arm lost its vertical-speed criterion at 0.214 s, then contacted the bowl much later; its sampled tilt remained below 45 degrees throughout. The 0.2 arm first exceeded 45 degrees at 0.392 s, just before first sampled contact. Two-millisecond observations do not exclude shorter intervening contacts.

In the original control's contact-free scored samples, raw left/right power drive spans 0.70119–0.84762. The retained log gain 0.693147 saturates every sample. Holding those recorded inputs fixed, every log gain at or above 0.35498 produces the same requested power. This is a transfer-function calculation on captured inputs, not a counterfactual full-brain simulation. Live parameter changes also change the subsequent sensory/neural history.

The affine normalization was derived from a grounded force window and did not transfer cleanly to the airborne input distribution. It introduced persistent excess left power in the observed early flight interval, alongside its steering offsets. The separate torque review quantifies this; the combined live experiment cannot attribute the subsequent yaw exclusively to either change. The affine module remains staged and unpromoted.

The saved stable event-control trajectory was checked separately against instantaneous and 50 ms averaged vertical-speed gates. It passes both, with the same 1.452 s motion-qualified interval. That reconstruction is not a full task score, and does not justify changing the current reward thresholds. See `../flight-event-control-scoring-audit/README.md`.

At 2026-09-14 14:46 UTC, a read-only SQLite check confirmed 28 previously completed training jobs, eight pending jobs, no leased jobs, and **zero changes to the 27 saved coordinates**. The scalar sweep is explicitly diagnostic. Training and production remain unchanged. Safari automation still returned `cgWindowNotFound`; these results were inspected numerically, not visually in Safari.
