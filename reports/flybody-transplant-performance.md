# One-fly native FlyBody performance

The BANC + native MuJoCo body remains within the requested factor-of-ten end-to-end slowdown threshold versus the old FlyWire system on this Apple M2 Pro. This is a sequential, one-fly test with the original 3D renderer, binocular RGB, FlyVis, color processing, neural body clock, direct movement and open anatomy inspection. Each sample measures 20 wall seconds after warm-up. All samples have active eyes/graded/color processing, neural progress, anatomy rendering, and no reported errors.

| Mode | Old neural seconds/wall second | BANC neural seconds/wall second | Slowdown | Old / BANC display FPS |
|---|---:|---:|---:|---:|
| Reference | 0.0512× | 0.0410× | 1.25× | 44.5 / 38.5 |
| Fast | 0.0826× | 0.0576× | 1.43× | 8.3 / 52.6 |

These are single-window measurements, not isolated-machine estimates or real-time simulation. Existing user applications, including Safari, were left running. The old Fast frame rate was unusually low in this sample; do not generalize its 8.3 FPS as a baseline. Throughput and display FPS measure different things. No 100-fly body-performance claim is made for the new solver.

[Raw measurements and source hashes](flybody-transplant-performance.json). Subsequent UI-only changes corrected the body description and added explicit viewing-link options for follow camera, movement mode and body clock. Neither the solver nor the benchmark settings changed.

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=1 --modes=reference,fast --datasets=flywire,banc --seconds=20 --trials=1 --output=reports/flybody-transplant-performance.json
```
