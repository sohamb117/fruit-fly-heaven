All nine held-out evaluations pass the pinned configuration, source, backend, assignment, horizon/physical-failure and paired initial-state checks. The primary comparison is generation 4 versus the exact generation-0 center. A pre-held-out amendment adds the single accepted candidate with highest training score (ties by lexical job ID) as a separate secondary head. Objective criteria 3 and the three seeds stay fixed.

The selected training candidate is g3-p1-neg, with training score 0.7460000000000013. It is not the generation-4 center.

| Seed | Candidate | Qualified flight g0 → candidate (s) | Takeoff | End time (s) | Score | Termination |
|---|---|---:|---|---:|---:|---|
| 790888 | g4 | 0.326 → 0.214 | True → True | 1.438 → 0.410 | -0.0220 → -0.3580 | excessive_rotation → excessive_rotation |
| 790888 | training-best | 0.326 → 0.242 | True → True | 1.438 → 1.066 | -0.0220 → -0.2740 | excessive_rotation → excessive_rotation |
| 890888 | g4 | 0.332 → 0.238 | True → True | 0.702 → 0.414 | -0.0040 → -0.2860 | excessive_rotation → excessive_rotation |
| 890888 | training-best | 0.332 → 0.496 | True → True | 0.702 → 2.286 | -0.0040 → 0.4880 | excessive_rotation → excessive_rotation |
| 990888 | g4 | 0.350 → 0.218 | True → True | 1.106 → 0.412 | 0.0500 → -0.3460 | excessive_rotation → excessive_rotation |
| 990888 | training-best | 0.350 → 0.368 | True → True | 1.106 → 1.708 | 0.0500 → 0.1040 | excessive_rotation → excessive_rotation |

primary_center: return: mean paired difference -0.338000, 0 increased / 3 decreased / 0 unchanged; bestFlightSeconds: mean paired difference -0.112667, 0 increased / 3 decreased / 0 unchanged; simSeconds: mean paired difference -0.670000, 0 increased / 3 decreased / 0 unchanged; landingSeconds: mean paired difference +0.000000, 0 increased / 0 decreased / 3 unchanged.

secondary_selected_training_candidate: return: mean paired difference +0.098000, 2 increased / 1 decreased / 0 unchanged; bestFlightSeconds: mean paired difference +0.032667, 2 increased / 1 decreased / 0 unchanged; simSeconds: mean paired difference +0.604667, 2 increased / 1 decreased / 0 unchanged; landingSeconds: mean paired difference +0.000000, 0 increased / 0 decreased / 3 unchanged.

g0: successes 0/3, takeoffs 3/3; g4: successes 0/3, takeoffs 3/3; training-best: successes 0/3, takeoffs 3/3.

The exact parameter center changed in 25 coordinates. This is reported separately from behavior and is not a learning claim. Three held-out seeds provide a limited paired check; they do not establish generalization or biological accuracy. A later crash without increased controlled flight is not successful flight.

Full raw differences, criteria-3 diagnostics, source hashes and native digests are in [comparison.json](comparison.json). No simulation or training update was run by this report.
