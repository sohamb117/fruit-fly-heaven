# Decoder feasibility — draft

**COMPLETE: all three student evaluations are validated.**

Nine Safari baseline evaluations are verified. The student comparison uses the same three seeds; oracle teacher seeds are reported separately.

| Seed | Controller | Score | Best flight (ms) | Duration (s) | Success | Reason |
|---:|---|---:|---:|---:|---|---|
| 2490888 | zero | -3.0000 | 0.0 | 0.160 | False | overturned |
| 2490888 | initial | -2.9804 | 14.0 | 0.410 | False | excessive_rotation |
| 2490888 | generation7 | -2.3000 | 500.0 | 0.974 | False | excessive_rotation |
| 2590888 | zero | -3.0000 | 0.0 | 0.160 | False | overturned |
| 2590888 | initial | -2.8180 | 130.0 | 0.862 | False | excessive_rotation |
| 2590888 | generation7 | -3.0000 | 0.0 | 0.180 | False | excessive_rotation |
| 2690888 | zero | -3.0000 | 0.0 | 0.160 | False | overturned |
| 2690888 | initial | -2.6500 | 250.0 | 0.908 | False | excessive_rotation |
| 2690888 | generation7 | -2.9356 | 46.0 | 0.260 | False | excessive_rotation |
| 2490888 | refined-v2 | 0.0000 | 740.0 | 5.000 | False | time_limit |
| 2590888 | refined-v2 | -2.4232 | 412.0 | 1.830 | False | excessive_rotation |
| 2690888 | refined-v2 | 0.0000 | 1016.0 | 5.000 | False | time_limit |

## Live-BANC oracle teacher

| Seed | Controller | Score | Best flight (ms) | Duration (s) | Success | Reason |
|---:|---|---:|---:|---:|---|---|
| 190888 | oracle | 9.9328 | 4952.0 | 5.000 | True | stage_success |
| 290888 | oracle | 9.9328 | 4952.0 | 5.000 | True | stage_success |
| 490888 | oracle | 9.9328 | 4952.0 | 5.000 | True | stage_success |

The oracle reads body pose/velocity and replaces wing outputs. It tests the physical control interface, not learned autonomous decoding.

## Fit and paired comparison

618/672 coefficients changed from initial. The bounded ridge fit converged (maximum full-gradient KKT residual 2.78e-17). Held-out all-phase teacher RMSE: 0.02990514 → 0.00568041.

Paired means use only 3/3 seeds with complete student results: [2490888, 2590888, 2690888].

- initial: score -2.8161; best flight 131.3 ms; duration 0.727 s; successes 0/3.
- generation7: score -2.7452; best flight 182.0 ms; duration 0.471 s; successes 0/3.
- refined-v2: score -0.8077; best flight 722.7 ms; duration 3.943 s; successes 0/3.

Oracle uses explicit body-state feedback and replaces wing outputs; live BANC and all nonwing physics remain active.
Refined students use the unchanged anatomically masked decoder with the complete refined 672-vector and no output override.
All trials start airborne after the same unscored 0.5-second warm-up; no takeoff or landing was demonstrated.
Quadratic KKT convergence establishes the regularized teacher-imitation optimum, not a global behavioral optimum or reliable closed-loop flight.
Teacher seeds are separate from the three student/Safari comparison seeds. Safari and Node are distinct hosts; exact dynamics are not assumed identical.

These are recorded results and telemetry, not continuous visual observations. Rebuild: `.venv/bin/python -B reports/flight-decoder-feasibility-20260914/summarize-feasibility.py`.
