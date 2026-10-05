# Flight decoder feasibility

The unchanged full native body can sustain flight through the existing decoded-control boundary. A diagnostic controller with direct body-state feedback passed all three live-BANC teacher trials: five scored seconds, 4.952 seconds of uninterrupted qualified flight after the initial measurement window, no scored environment contacts, and score 9.9328.

Fitting the existing 672-coefficient neural decoder substantially improved the three held-out comparisons, but **none of the fitted trials completed the maintained-flight goal**. The mean longest qualified bout increased from **131 ms initially** and **182 ms at generation 7** to **723 ms after fitting**. This is a measured improvement on these three cases, not proof of reliable flight or statistical generalization.

## Closed-loop results

Each row uses the same seed and frozen physical/neural configuration. “Best flight” means the longest uninterrupted bout satisfying the original contact, orientation, angular-rate, vertical-speed and support criteria.

| Seed | Initial best flight | Generation 7 best flight | Fitted best flight | Total fitted qualified airtime | Fitted outcome |
|---:|---:|---:|---:|---:|---|
| 2490888 | 14 ms | 500 ms | **740 ms** | 2.510 s | Reached 5 s; goal not met |
| 2590888 | 130 ms | 0 ms | **412 ms** | 0.574 s | Excessive rotation at 1.830 s |
| 2690888 | 250 ms | 46 ms | **1,016 ms** | 2.868 s | Reached 5 s; goal not met |

The fitted decoder improves the best bout on every initial and generation-7 comparison. Its scores were 0, −2.4232 and 0, respectively. A score of zero at the time limit must not be read as successful flight: both five-second survivors had zero current qualified airtime at the final observation.

The initial and generation-7 baselines all failed before one second. All nine zero/initial/generation-7 Safari evaluations are complete and validated in [the baseline report](../zero-baseline-wasm-20260914/README.md). Separate Node replays reproduced the initial seed-2490888 and generation-7 seed-2690888 scores, failure timesteps and best-flight counters; floating-point physical states are not assumed bit-identical across hosts.

## What the fit learned

The teacher trials used seeds 190888 and 290888 for fitting, and 490888 for validation. Closed-loop test seeds 2490888, 2590888 and 2690888 did not enter fitting. The teacher preserved live BANC feedback and nonwing controls; its explicitly labeled intervention replaced decoded wing commands using body pose and velocity.

The fitted controller uses the original anatomically masked decoder, with no oracle output replacement and no new body-state input. **618 of 672 coefficients changed**. The other 54 were unexcited steering coefficients retained at their initial values. Connectivity, neural parameters, effector masks, feature history, phase mechanism and physical parameters remained fixed.

The first coordinate fit hit its iteration cap. A direct/active-set quadratic refinement solved the same bounded ridge objective around the original initial vector, with ridge 1e−5. All eight outputs passed the full-gradient KKT check; maximum residual was 2.78e−17. This establishes convergence for the recorded command-imitation objective, not a global optimum of flight behavior.

Held-out command RMSE across all five 0.2 ms wing-control phases decreased from **0.02990514 to 0.00568041**, about **81%**. This combined metric mixes normalized power and steering-control units; per-output errors are recorded separately. Predictions and targets did not clip on the teacher trajectories. Actual feature reconstruction and original decoder outputs were checked, including the four phases omitted from the one-millisecond fitting samples.

## Remaining failure

The retained early qualification failures were caused by downward speed below −1 cm/s while attitude, angular RMS, powered wings and inferred support still passed. This differs from the earlier immediate catastrophic wing-actuation failure.

Seed 2590888 descended from about 3.48 cm to 1.74 cm by 1.42 s while remaining upright. It then contacted the floor and later exceeded the 300 rad/s angular-speed limit. The other two stayed upright through five seconds but descended into intermittent contact. Their scored contact totals were 164 ms and 200 ms, which reset the qualification windows. They ended just after contact, rather than in a sustained qualifying flight bout.

This points to insufficient closed-loop vertical regulation. A good fit on oracle-controlled trajectories does not guarantee the same commands on states reached by the decoder itself. The teacher dataset also covers scored flight, not the fitted decoder's initial 0.5 s warm-up. These tests do not distinguish conclusively between inadequate trajectory coverage, insufficient information in the current neural features, and a limitation of the decoder class. They do not prove that no better 672-vector can work.

The next useful experiment is to fit and optimize on decoder-generated descent/recovery trajectories, starting with power and lift regulation, while keeping the biological input boundary explicit. Further reduction of offline hover-prediction error alone is insufficient evidence of control success.

## Scope and evidence

- Every trial began airborne after the same unscored 0.5 s root restraint. No takeoff, landing, feeding or food-localization success was established.
- The complete native body retained all 50 nonroot joints, original contacts and actuator limits, and the original 50 μs timestep. No applied external forces or direct scored root writes were introduced.
- Main feasibility trials ran the pinned browser WASM modules in Node. The separate baseline page was restored and completed in Safari. No diagnostic results were submitted to the cloud coordinator and no production files or deployment were changed for this experiment.
- Qualification/contact counters come from the original 2 ms scorer. Saved previews retain the first 199 frames and the final frame; later substep contact timing and continuous visual behavior cannot be reconstructed from those sparse images.
- The classical controller is an engineering positive control, not a biological controller. The fit is simulator system identification, not validation of real muscle physiology.

[Validated machine-readable summary](summary.json) · [Refined checkpoint](refinement-v2/checkpoint.json) · [Fit report](refinement-v2/report.json) · [All-phase validation](additional-phase-v2-report.json) · [Physical postmortem](postmortem-fitted-v2.json) · [Protocol and frozen identities](PROTOCOL.md)

Regenerate the numerical summary with `.venv/bin/python -B reports/flight-decoder-feasibility-20260914/summarize-feasibility.py`. Original evaluation, capture and preview files remain alongside this report; the protocol and executable diagnostic scripts describe their reproduction.
