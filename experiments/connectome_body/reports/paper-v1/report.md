# Ten-experiment compatibility study

Purpose: experiment. 0 terminal training runs out of 24180 planned unique conditions.

| Experiment | Training conditions | Runnable inputs | Post-training jobs |
|---|---:|---:|---:|
| 1: Encoder-decoder architecture and performance-capacity Pareto frontier | 12936 | 9363 | 0 |
| 10: Nested structural prediction with whole pairs held out | 0 | 0 | 4 |
| 2: Architecture x connectome x body/task; minimum width, depth, rank and capacity | 3240 | 2385 | 0 |
| 3: Biology versus parameter- and compute-matched recurrence and no brain | 2520 | 1890 | 0 |
| 4: Original, community, degree, direction-shuffled and N/M-random wiring | 3600 | 2700 | 0 |
| 5: Initialization x permitted-edge plasticity, with rewired controls | 4800 | 3600 | 0 |
| 6: Crossed connectome-body effects and matched-pair interaction | 4392 | 3060 | 0 |
| 7: Connectome x task and task-relevant versus equal-size subgraphs | 480 | 360 | 0 |
| 8: Post-training causal substrate interventions | 0 | 0 | 7200 |
| 9: Distribution shifts, zero-shot task transfer and fine-tuning | 648 | 486 | 9720 |

Censored thresholds are not successful runs. Confidence intervals resample training seeds. A compiled condition is not a measurement.

Machine-readable estimands, paired comparisons, missing cells, and causal/transfer results are in `paper-report.json`.
