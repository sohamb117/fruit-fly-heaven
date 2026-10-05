# Original 3D console: BANC versus FlyWire

The restored BANC console meets the requested same-order-of-magnitude throughput threshold in all four measured settings. BANC renders at 59.3–60.0 frames/s; display frame rate is separate from simulated neural speed.

Measured September 12, 2026 (local time), on an Apple M2 Pro with 16 GiB RAM, Chrome 152.0.7977.83 and an Apple Metal WebGPU adapter. FlyWire uses its original WASM backend. Both use the original 3D console with real binocular RGB rendering, FlyVis, color, all senses, direct movement, neural body clock and open anatomy inspection at 1440 × 1000.

| Flies | Mode | FlyWire simulated s/wall s | BANC simulated s/wall s | BANC slowdown | BANC display frames/s |
|---:|---|---:|---:|---:|---:|
| 1 | Reference | 0.09135 | 0.05312 | 1.72x | 60.0 |
| 1 | Fast | 0.10799 | 0.08186 | 1.32x | 60.0 |
| 100 | Reference | 0.00735 | 0.00129 | 5.69x | 59.7 |
| 100 | Fast | 0.00954 | 0.00228 | 4.18x | 59.3 |

Each result is one 20-second wall-time sample after at least 20 ms of neural progress by every fly and another 2 seconds of wall warmup. Cohort speed follows the slowest fly's neural clock. Every sample had active rendered eyes, graded vision and color processing for the entire population, working anatomy and no captured application errors. These are short-window throughput measurements, not proof of sustained long-horizon speed, numerical stability or a completed autonomous food-to-flight cycle.

The FlyWire samples came from the earlier complete paired run in the same session. BANC was rerun after adding retained per-fly spike history and fixing direct-mode root orientation; the shared console, original FlyWire worker/core and graph inputs were unchanged. [Raw samples and per-backend source hashes](transplant-performance.json) preserve the two snapshots.

At 100 flies, BANC reports about 3.81 GB of neural GPU buffers in Reference and 3.67 GB in Fast, including independent physiology state and shared immutable graphs. Chrome descendant RSS is also recorded, but sums shared pages and is not unique physical memory. Graph sharing, delay histories sized to actual delays and GPU motor/counter reductions keep the full population usable while retaining selected-neuron inspection and spike history.

Reproduce with the local server running:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --seconds=20
```

[UI validation](transplant-ui-validation.json) separately covers the original panels, anatomy/microscopy, traces, popouts, selection, pause, sensory switches, movement modes, body clocks and population changes. [GPU/WASM validation](banc-browser-validation.json) covers independent brain state, shared graph lifetime and retained spike history. The physiological parameters remain explicit priors; the complete behavioral sequence still requires calibration and validation.
