# Full BANC structural-adapter benchmark (mps)

Full stateless E/D, learned structural ports, frozen recurrence; no MuJoCo or optimizer step

CUDA instrumented peak; MPS allocated/driver memory sampled after each decision and backward, not an instrumented peak; process RSS high-water is cumulative

| Batch | BPTT | Inference transitions/s | Forward+backward transitions/s | Allocated GiB |
|---:|---:|---:|---:|---:|
| 1 | 8 | 714.07 | 320.32 | 0.271 |
| 1 | 32 | 745.69 | 349.45 | 0.396 |
| 1 | 64 | 747.94 | 356.76 | 0.564 |
| 4 | 8 | 789.23 | 385.27 | 0.272 |
| 4 | 32 | 792.91 | 391.88 | 0.402 |
| 4 | 64 | 795.77 | 392.50 | 0.575 |
| 16 | 8 | 794.00 | 403.05 | 0.407 |
| 16 | 32 | 796.64 | 398.52 | 0.927 |
| 16 | 64 | 796.81 | 397.67 | 1.620 |
