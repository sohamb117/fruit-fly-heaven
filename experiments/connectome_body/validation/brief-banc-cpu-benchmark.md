# Sparse recurrent benchmark

Device: **cpu**. Graph: 175,401 neurons / 13,542,180 edges. Float32.
No CUDA result is implied by a CPU run. Rates count individual recurrent substeps across the batch.

| Batch | BPTT | Inference transitions/s | Training forward/s | Backward/s | Combined training/s | Peak VRAM GiB | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6.75 | 6.75 | 6.76 | 3.38 | n/a | passed |
| 1 | 16 | 6.76 | 6.75 | 6.79 | 3.38 | n/a | passed |
| 4 | 8 | 23.69 | 23.66 | 24.31 | 11.99 | n/a | passed |
| 4 | 16 | 24.22 | 24.42 | 24.49 | 12.23 | n/a | passed |
