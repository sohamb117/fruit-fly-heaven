# Frozen-neural BANC interface validation

**No candidate passed. No candidate was promoted or deployed.** Six matched, real BANC v888 evaluations completed on WebGPU with native MuJoCo/WASM muscles. This is evidence that the current candidate changes do not rescue behavior, not evidence of trained recovery or flight.

The intended posture horizon was two simulated seconds, beginning with a seeded 9 rad/s angular disturbance. Each trial terminated when instantaneous angular speed exceeded the explicit 300 rad/s failure limit. Neural parameters, synapses, sensory gains, graph topology and native body model remained fixed. The two changed values were hind-claw force capacity and the optional continuous deployment span.

| Interface | Held-out seed | Failure time (s) | Maximum angular speed (rad/s) | Outcome |
| --- | ---: | ---: | ---: | --- |
| Legacy defaults | 190888 | 0.126 | 546.36 | Excessive rotation |
| Legacy defaults | 290888 | 0.178 | 323.01 | Excessive rotation |
| Hind adhesion ×0.25 | 190888 | 0.144 | 325.63 | Excessive rotation |
| Hind adhesion ×0.25 | 290888 | 0.396 | 354.28 | Excessive rotation |
| Hind adhesion ×0.25, opening span 0.1 | 190888 | 0.136 | 308.11 | Excessive rotation |
| Hind adhesion ×0.25, opening span 0.1 | 290888 | 0.156 | 1532.61 | Excessive rotation |

All returns were −10, all success flags false, and all six final observations had zero supporting foot load. Both candidates received `mechanical_calibration_required` from the operator gate. A short replay surviving longer on one seed does not establish a calibrated controller or justify starting a broader search.

The independent runner wrote `result.json`, six compressed frame streams and six final screenshots. The [trajectory contact sheet](trajectory-contact-sheet.png) shows four actual native poses per trial, sampled from 84 retained preview frames. The frames show wing recruitment followed by loss of support and rotation away from the fruit. Rendering is observational; simulation clocks and landmarks come from the native body. Neural and physical clocks matched on all final frames, neural activity was nonzero, and no user-applied root force was present.

The initial observer diagnostic is retained separately in [the superseded diagnostic directory](../interface-training-observer-diagnostic/README.md). It found that geometric penetration counts omitted force-bearing contacts inside MuJoCo's soft-contact margin. The corrected observer counts native force-bearing contacts, and all six trials above reached real behavioral failure rather than `invalid_observation`. The runner now treats invalid observations as instrumentation failures.

## Exact identity and reproduction

- Config: `configs/training-interface-v2.json`
- Config SHA256: `40412edab59ab89155dc3d3a5031eb85f96fc11fc76c4ace99c31875f96e94d6`
- Model fingerprint: `4db4297e05933cf811fc47c8a90ccd1ff2eccb241e74bb0c0aa9c634928fb981`
- Validation seeds: `190888`, `290888`; final test seeds remain unused.
- Backend: WebGPU BANC, native `mujoco-wasm` body and configured WASM muscles.
- UI: original training/habitat pages were not used or changed by the experiment.

With the local static server running, execute:

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
  node scripts/run-interface-experiment.mjs \
  --candidates='[{"name":"hind_025","parameters":{"hindGripScale":0.25}},{"name":"smooth_hind_025","parameters":{"hindGripScale":0.25,"deploymentPowerSpan":0.1}}]'
```

See the [calibration guide](../../docs/motor-interface-calibration.md) for asset pinning, constraints and the distinction between an executed experiment, mechanical calibration, behavioral success and promotion. Browser/WebGPU reduction ordering can introduce small numeric differences; compare outcomes and recorded identities, not cross-device bitwise neural equality.

## Decision

Retain the continuous-onset branch as an experimental option, with legacy defaults unchanged. It removes a demonstrable command discontinuity but does not repair high-drive flight. Reduced hind adhesion is also insufficient. Neither change earns release into contributor training. The remaining target is independently constrained wing actuation and adequate neural feedback, assessed together in matched full neural evaluations. Open-loop instability alone does not establish a bad physical model or justify requiring a passively stable fly. Increasing training duration or weakening the success criteria would not resolve the failures observed here.
