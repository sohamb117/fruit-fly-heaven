# Post-observation one-fly performance

The current BANC build meets the requested factor-of-ten throughput threshold against the retained FlyWire system in this short-window comparison. All eight measurements passed the liveness, configuration, sensory, inspection and source-consistency checks. This does not establish successful behavior or real-time simulation.

Two 20-second measurement windows per mode and dataset; one fly; original 3D UI with binocular rendering, FlyVis, color, every sensory switch, direct motor coupling, neural body clock and open anatomy inspection. The observer browser was closed before timing. Models intentionally differ: FlyWire uses its original kinematic body, while BANC uses muscles and synchronized native MuJoCo. Reference timesteps also differ (FlyWire 0.1 ms/Float64; BANC 0.5 ms/Float32); both Fast modes use 1 ms/Float32.

| Mode | FlyWire median neural/real time | BANC median neural/real time | BANC slowdown | Paired slowdown range | BANC animation callbacks/s |
|---|---:|---:|---:|---:|---:|
| Reference | 0.01748× | 0.01399× | 1.25× | 1.20–1.29× | 53.5 |
| Fast | 0.03120× | 0.02254× | 1.38× | 0.80–1.95× | 55.0 |

Each paired trial is also within the 10× threshold. Below 1× slowdown means BANC was faster in that pair. The callback metric counts requestAnimationFrame callbacks; it is not a guarantee of distinct presented frames. Renderer and anatomy counters independently advanced in every accepted window.

This was a shared-machine test on an Apple M2 Pro. Other browser workloads remained active; [load snapshot](transplant-performance-visual-audit-context.json). The second FlyWire Fast trial had 7.48 animation callbacks/s while advancing its neural/body clocks, compared with 41.37 in its first trial, so Fast-mode variance is material. BANC callbacks ranged 52.99–56.15/s across the four samples. No claim of improvement over older reports follows from these noisy shared-machine measurements.

[Complete measurements, controls, checks and source hashes](transplant-performance-visual-audit.json) · [200-frame visual audit](observation-60min-20260913/README.md)

Reproduce:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=1 --modes=reference,fast --datasets=flywire,banc --seconds=20 --trials=2 --output=reports/transplant-performance-visual-audit.json
```
