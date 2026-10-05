# One-fly speed after ground-feedback repair

Apple M2 Pro, Chrome 152, original 3D console, open anatomy inspection, rendered binocular vision, all senses enabled, direct movement and neural body clock. Each measurement uses one fly. BANC uses WebGPU neural computation and native MuJoCo WASM mechanics; the old FlyWire comparison uses its original WASM neural engine and body.

| Trial | Mode | Old neural seconds / wall second | BANC neural seconds / wall second | BANC slowdown | Old display FPS | BANC display FPS |
|---|---|---:|---:|---:|---:|---:|
| Initial | Reference | 0.04607 | 0.03612 | 1.28× | 45.0 | 33.0 |
| Initial | Fast | 0.06191 | 0.04600 | 1.35× | 17.5 | 4.9 |
| Repeat | Fast | 0.07476 | 0.05345 | 1.40× | 15.7 | 32.9 |

All runs meet the requested factor-of-ten neural-throughput threshold. This does not mean real-time simulation or equal display performance. The first Fast run had a large rendering slowdown; the repeat did not reproduce it. Other user applications remained active on the shared machine, so these short windows do not establish steady frame rate or a cause for the variation. No runtime errors occurred, and rendered eyes, graded vision, color processing and anatomical inspection remained active in every run.

Raw settings, source hashes, memory observations and timings: [initial comparison](flybody-ground-performance.json), [Fast repeat](flybody-ground-fast-repeat.json). Behavior and its remaining failures are reported separately in [ground-feedback observation](flybody-ground-feedback.md). These performance measurements were made in Chrome; Safari was opened for viewing, not benchmarked.

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=1 --seconds=20 --output=reports/flybody-ground-performance.json
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=1 --modes=fast --seconds=20 --output=reports/flybody-ground-fast-repeat.json
```
