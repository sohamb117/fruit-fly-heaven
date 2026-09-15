# Full BANC structural-adapter benchmark (cpu)

Full stateless E/D, learned structural ports, frozen recurrence; no MuJoCo or optimizer step

CUDA instrumented peak; MPS allocated/driver memory sampled after each decision and backward, not an instrumented peak; process RSS high-water is cumulative

| Batch | BPTT | Inference transitions/s | Forward+backward transitions/s | Allocated GiB |
|---:|---:|---:|---:|---:|
| 4 | 32 | 73.65 | 37.01 | n/a |
