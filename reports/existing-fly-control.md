# Reusing demonstrated fly control

Stable simulated walking and flight already have released implementations. The current BANC build imported the FlyBody mechanical model and wing cycle, but replaced the trained controller with a custom motor-force adapter. That is not a reproduction of published FlyBody control, and stable perching does not validate flight.

## Available systems

| System | Demonstrated capability | Interface and relevance |
|---|---|---|
| [FlyBody](https://github.com/TuragaLab/flybody), [Nature 2025](https://doi.org/10.1038/s41586-025-09029-4) | Walking, maneuvering flight, and vision-guided flight; released trained policies | Learned low-level policy plus experimental wingbeat generator, driven by body observations and a desired trajectory. Use the released controller and original simulator as the reference/teacher. |
| [NeuroMechFly v2 / FlyGym](https://neuromechfly.org/), [Nature Methods 2024](https://doi.org/10.1038/s41592-024-02497-y) | Terrain walking, adhesion, odor navigation, and connectome-constrained visual following | Existing locomotor controllers, sensory models, and a browser MuJoCo/WASM implementation. Useful for the ground-control baseline. |
| [Eon's embodied fly](https://eon.systems/updates/embodied-brain-emulation) | Reported brain/body integration for walking, grooming, and feeding | Explicitly uses selected descending outputs and learned body controllers, with hand-chosen mappings. It does not implement the full VNC/motor-neuron hierarchy. |
| [FlyGM](https://arxiv.org/abs/2602.17997), [project page](https://lnsgroup.cc/research/FlyGM/) | Connectome-structured learned walking and flight controllers | Imitates FlyBody experts, then trains with PPO. Neurons have learned vector states and a learned action decoder, rather than the BANC conductance physiology. Project page currently labels code “Coming soon”. |
| [Pugliese et al., 2026](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v2), [code](https://github.com/smpuglie/Pugliese_2026) | Connectome-derived VNC motor rhythms, including BANC | A published rate-equation CPG assay with morphology-scaled neuronal parameters. A useful neural-dynamics calibration reference; it does not supply embodied flight. Its shipped BANC snapshot predates v888 and needs ID verification. |

These systems make the integration feasible. They do not establish that importing anatomical connectivity and assigning generic cell parameters automatically recovers coordinated locomotion.

## Reproduced locally

The released FlyBody policy passed its [original 0.6-second flight task](flybody-upstream-baseline.json) and a [continuous 1.2-second horizon](flybody-upstream-baseline-continuous.json) on this machine. The latter completed 5,994 control steps (1.1988 simulated seconds), with maximum trajectory error 0.407 mm, minimum thorax height 1.012 cm and mean reward 0.903. Externally applied root forces and direct root actuator forces were zero. These are flight-controller reference tests, not BANC behavior or takeoff/landing tests.

The released 104→256→256→256→12 policy has also been exported with its original learned weights. The [JavaScript implementation](../web/flybody-policy.js) matches original TensorFlow outputs on 43 fixtures, including real flight observations, with maximum absolute difference 5.67e−7 in Node and Chromium. Chromium inference averaged 0.223 ms over 1,000 calls. [Inference verification](flybody-policy-validation.json).

The complete [controller and physics loop](../web/flybody-flight-reference.js) now runs in MuJoCo WASM. In Chromium it completed the same 5,994 control steps with maximum reference error 0.407 mm and zero external root forces in 4.84 wall seconds (0.248× real time, excluding rendering and BANC). It implements the original wingbeat lookup, action rescaling, sensor averaging, reference observations, failure conditions and successful termination. The original model retains its dynamics and compiled inertias; only visual mesh geometry is removed. [Browser flight result and sampled poses](flybody-reference-browser.json), [native/WASM replay](flybody-reference-replay.json), [disconnection control](flybody-reference-validation.json).

The optional [reference in the original 3D console](http://127.0.0.1:7842/?controller=flybody-reference&follow=1) displays one fly. Its default trajectory now brakes over 0.3 seconds and holds a bounded hover, rather than flying straight out of the visual habitat. The controller passed [60 simulated seconds](flybody-bounded-hover-60s.json) with maximum displacement 3.027 cm and zero external/root actuator force; [8.019 seconds in the original UI](flybody-bounded-ui.json) also passed. It explicitly labels the learned controller, prescribed airborne trajectory, disabled ground contacts and BANC observer role. The original short straight fixture remains available with `&trajectory=straight`. This reference does not demonstrate the requested neural-to-muscle coupling.

The [common-body reuse investigation](flybody-contact-reuse.md) reproduced both walking and flight on the same physical parameter set. A prescribed descent followed by a contact-triggered walking-policy handoff produced a reproducible [landing/walking teacher](flybody-landing-teacher/README.md), with exact physical state continuity and zero external forces. A naive walking-to-flight handoff still tumbles. These provide mechanical/controller baselines and teacher data; they do not establish BANC control or an autonomous complete behavioral cycle.

The current default BANC build was separately benchmarked against the old FlyWire backend. One-fly throughput was 2.37× slower in Reference and 2.59× slower in Fast, within the requested factor-of-ten threshold; BANC display throughput was 51–60 FPS. [Current matched whole-console benchmark](transplant-performance-reference-reuse.md).

## Integration sequence

1. **Completed:** reproduce one released FlyBody controller with its original model, input ordering, initial pose, timestep and wingbeat generator.
2. **Completed:** export the exact policy, verify numerical parity, and port the reference to WASM and the existing 3D renderer. The reference is explicitly labeled as learned control.
3. Use that baseline as a teacher and reference for fitting the BANC sensory/neuromuscular interface. Preserve the intended brain → VNC → motor neurons → muscles route in the mechanistic mode. A descending-command controller is a useful comparison, but does not demonstrate that route.
4. Validate takeoff, landing and feeding separately. FlyBody's standard flight task starts airborne, disables leg actuation and normally disables floor contacts. Its successful flight rollout alone cannot establish those transitions.

## Corrections in the current experimental integration

- Body feedback now advances after each neural block, rather than waiting for a screen redraw; workers await the resulting body state.
- The retraction interpretation was corrected from I1 to III1 after checking the original Drosophila recording paper. Previous perching measurements remain historical evidence for the old adapter.
- Annotated haltere motor targets are now retained at the muscle boundary. Their sensory tuning and physiological calibration remain unfinished.

The FlyBody source code is Apache-2.0. Its released trained-policy data have their own GPL-3.0-or-later license according to the Figshare data record. The exported weights retain that provenance and the [GPL-3.0 text](../models/FlyBody-policy-LICENSE).

## Repeat browser and integration checks

```sh
node scripts/verify-flybody-policy.mjs
node scripts/verify-flybody-reference.mjs
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/verify-flybody-reference-browser.mjs
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/verify-flybody-reference-ui.mjs
```

Native environment and export commands are documented in [the upstream baseline reproduction](flybody-upstream-baseline.md).
