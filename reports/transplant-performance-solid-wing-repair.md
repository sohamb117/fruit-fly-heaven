# Matched whole-product throughput

The current BANC build meets the factor-of-ten throughput threshold against the retained FlyWire system in every accepted mode and paired trial. 8/8 measurements passed the harness checks. Source hashes were unchanged through the measurement run. This measures execution speed; it does not establish successful behavior or real-time simulation.

Current production COM/contact/sensory build; excludes experimental wing-flex prototype. [Measurement context](transplant-performance-solid-wing-repair-context.json).

2 20-second measurement windows per mode and dataset, using the original 3D UI with binocular rendering, FlyVis, color, every sensory switch, direct motor coupling, the neural body clock and open anatomy inspection. FlyWire retains its kinematic body; BANC uses muscles and synchronized native MuJoCo. These are the intended product workloads, not equal numerical workloads.

| Flies | Mode | FlyWire neural/real time | BANC neural/real time | BANC slowdown | Paired slowdown range | BANC sim time per 20 s wall | BANC callbacks/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | reference | 0.06381× | 0.03868× | 1.65× | 1.56–1.74× | 0.774 s | 51.9 |
| 1 | fast | 0.07667× | 0.04563× | 1.68× | 1.64–1.72× | 0.913 s | 57.4 |

Throughput is simulated neural milliseconds divided by elapsed wall-clock milliseconds. The table also converts each BANC rate into simulated seconds per measurement window. These rates remain below real time even when the UI redraws frequently. The callback column counts requestAnimationFrame callbacks and is not a guarantee of distinct presented frames. Renderer and anatomy liveness are checked separately.

Measured on Apple M2 Pro (darwin/arm64), starting 2026-09-13T19:40:13.268Z. Short paired windows are subject to shared-machine load and trial variation. A prior report is not a controlled before/after performance experiment.

[Complete measurements, effective model settings, source hashes and validity checks](transplant-performance-solid-wing-repair.json).

Reproduce:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=1 --modes=reference,fast --datasets=flywire,banc --seconds=20 --trials=2 --output=reports/transplant-performance-solid-wing-repair.json
```
