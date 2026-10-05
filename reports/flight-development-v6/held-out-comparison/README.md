All nine held-out evaluations pass the pinned configuration, source, backend, assignment, horizon/physical-failure and paired initial-state checks. The primary comparison is generation 4 versus the exact generation-0 center. A pre-held-out amendment adds the single accepted candidate with highest training score (ties by lexical job ID) as a separate secondary head. Objective criteria 3 and the three seeds stay fixed.

The selected training candidate is g3-p0-neg, with training score 0.06200000000000072. It is not the generation-4 center.

| Seed | Candidate | Qualified flight g0 → candidate (s) | Takeoff | End time (s) | Score | Termination |
|---|---|---:|---|---:|---:|---|
| 490888 | g4 | 0.136 → 0.112 | False → False | 0.402 → 0.318 | -0.7787 → -1.1707 | excessive_rotation → excessive_rotation |
| 490888 | training-best | 0.136 → 0.338 | False → True | 0.402 → 0.686 | -0.7787 → 0.0140 | excessive_rotation → excessive_rotation |
| 590888 | g4 | 0.082 → 0.110 | False → False | 0.416 → 0.326 | -1.6607 → -1.2033 | excessive_rotation → excessive_rotation |
| 590888 | training-best | 0.082 → 0.354 | False → True | 0.416 → 1.106 | -1.6607 → 0.0620 | excessive_rotation → excessive_rotation |
| 690888 | g4 | 0.172 → 0.110 | True → False | 0.392 → 0.314 | -0.4840 → -1.2033 | excessive_rotation → excessive_rotation |
| 690888 | training-best | 0.172 → 0.352 | True → True | 0.392 → 0.720 | -0.4840 → 0.0560 | excessive_rotation → excessive_rotation |

primary_center: return: mean paired difference -0.218000, 1 increased / 2 decreased / 0 unchanged; bestFlightSeconds: mean paired difference -0.019333, 1 increased / 2 decreased / 0 unchanged; simSeconds: mean paired difference -0.084000, 0 increased / 3 decreased / 0 unchanged; landingSeconds: mean paired difference +0.000000, 0 increased / 0 decreased / 3 unchanged.

secondary_selected_training_candidate: return: mean paired difference +1.018444, 3 increased / 0 decreased / 0 unchanged; bestFlightSeconds: mean paired difference +0.218000, 3 increased / 0 decreased / 0 unchanged; simSeconds: mean paired difference +0.434000, 3 increased / 0 decreased / 0 unchanged; landingSeconds: mean paired difference +0.000000, 0 increased / 0 decreased / 3 unchanged.

g0: successes 0/3, takeoffs 1/3; g4: successes 0/3, takeoffs 0/3; training-best: successes 0/3, takeoffs 3/3.

The exact parameter center changed in 20 coordinates. This is reported separately from behavior and is not a learning claim. Three held-out seeds provide a limited paired check; they do not establish generalization or biological accuracy. A later crash without increased controlled flight is not successful flight.

Full raw differences, criteria-3 diagnostics, source hashes and native digests are in [comparison.json](comparison.json). No simulation or training update was run by this report.
