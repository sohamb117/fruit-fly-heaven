All six held-out evaluations pass the source, configuration, assignment, backend, complete-result and paired initial-state gates. This primary comparison uses exact stored g0/g2 incumbents on three seeds excluded from training and guard evaluations.

Guard history: generation 0: rejected, 8 search + 6 guard evaluations; generation 1: rejected, 8 search + 6 guard evaluations. These guard scores selected the center and are training evidence.

| Seed | Qualified flight g0 → g2 (s) | Takeoff | End time (s) | Score | Termination |
|---|---:|---|---:|---:|---|
| 1190888 | 0.328 → 0.328 | True → True | 0.722 → 0.722 | -0.0160 → -0.0160 | excessive_rotation → excessive_rotation |
| 1290888 | 0.504 → 0.504 | True → True | 0.734 → 0.734 | 0.5120 → 0.5120 | excessive_rotation → excessive_rotation |
| 1390888 | 0.344 → 0.344 | True → True | 1.092 → 1.092 | 0.0320 → 0.0320 | excessive_rotation → excessive_rotation |

return: mean difference +0.000000, 0 increased / 0 decreased / 3 unchanged; bestFlightSeconds: mean difference +0.000000, 0 increased / 0 decreased / 3 unchanged; simSeconds: mean difference +0.000000, 0 increased / 0 decreased / 3 unchanged; landingSeconds: mean difference +0.000000, 0 increased / 0 decreased / 3 unchanged.

g0: successes 0/3, takeoffs 3/3; g2: successes 0/3, takeoffs 3/3.

A guarded center change is not itself a learning claim. Three new held-out seeds provide a limited check; all failures remain in the report. See [comparison.json](comparison.json) for exact values, audited acceptance decisions, diagnostics and provenance. No simulation or training update ran in this analysis.
