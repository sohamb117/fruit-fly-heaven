# Ground feedback repair — one fly

The floor collision solver already existed, but its contact information was not reaching the neural inputs faithfully. Load, joint movement and whole-body movement were blended into the same left/right rates. Three legs on each side received identical inputs, leg bristles used a head-relative obstacle probe, and the haltere signal could disappear when the legs unloaded.

Direct mode now sends native per-leg normal forces into the corresponding BANC load afferents. Position, velocity, leg-shaft contact and haltere/wing-base rotation use separate transducers. Leg identity and modality come from BANC annotations; their numerical tuning remains an explicit prior. The original UI channels summarize these separate neuron inputs. The original assisted comparison retains its kinematic sensory adapter.

The visual thorax-to-root offset now rotates with the physical body, and the leg endpoints follow native claw sites. Feeding requires actual native mouth contact at food, plus probing and pumping. A body merely standing or hovering over fruit cannot ingest through the former distance proxy. Initial foot posture is adapted to terrain once, and leg target excursions are limited to 0.35 radians around that posture. These initialization and actuator settings are modeling assumptions, not learned posture control.

## Verified

- [Native MuJoCo/WASM checks](flybody-runtime-validation.json): supported contact produces positive leg load; lifting the body clears all six loads and touch despite a moving joint. Airborne probing and pumping transfers no food. Native/WASM trajectory error remains below 4.69e-8. Isolated motor stimulation, disconnection, claw actuation and wing actuation pass.
- [Original console checks](transplant-ui-validation.json): one fly, original windows/controls, rendered binocular vision, BANC anatomy, pause, popouts and modes pass. Thorax and leg coordinates agree with native physics under upright, tilted and inverted poses to within 8.15e-15 display units. This checks landmark registration, not exact visual/collision surface equivalence.
- All 41 web tests, 22 physiology/muscle tests and four importer tests pass. Five web tests specifically cover per-leg sensing, unloaded movement, airborne rotation, separate cell inputs and the assisted comparison. An initial concurrent Node test run stalled; the complete serial run passed, and the stalled test also passed independently.

## Observed behavior

[Final live telemetry](flybody-ground-feedback-final.json) records one fly in the original 3D habitat. It ran 2.040 neural/body seconds over 55.71 wall seconds, with motor coupling removed at 1.516 seconds. The run includes rendered eye feedback and the full BANC brain/VNC graph, not replayed motor commands. Some regression tests shared the machine during observation, so its elapsed time is not a controlled throughput benchmark.

All 11 sampled airborne states before disconnection had zero leg load. Ground contact was detected in five sampled states; the remaining samples include near-surface states without loaded contacts. Joint/rotation sensing continued in the air. No food was ingested and no browser error occurred.

**The fly still tumbles.** The maximum sampled tilt was 2.79 radians and angular speed 927.8 rad/s. Horizontal speed reached 576.7 mm/s. The only event was an orientation proxy during uncontrolled movement; this is not successful localization. After motor disconnection, maximum muscle force decayed to 2.20e-6 and wing power to 1.81e-6. Gravity, passive actuator targets and contact continued to move the body at 5.04 mm/s in the final sample.

Correct ground inputs do not establish a corrective neural response. Stable posture, coordinated walking, controlled landing and the requested complete feeding/flight sequence remain unresolved. Mechanical replays isolated large wing/floor impulses. Increased friction, common wing power, an implicit integrator and a wing-contact torque limit did not reliably prevent active tumbling over a one-second replay; those experimental changes are not enabled in the default body.

## Reproduce

```sh
.venv/bin/python scripts/prepare-flybody-runtime.py
.venv/bin/python scripts/prepare-banc-console.py
.venv/bin/python scripts/serve.py --port 7842
node scripts/verify-flybody-runtime.mjs
node --test --test-concurrency=1 --test-timeout=30000 web/test/*.test.mjs
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/verify-transplant-ui.mjs --one-fly
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --population=1 --ms=1500 --disconnect-ms=500 --follow=true --output=reports/flybody-ground-feedback-final.json
```

The latest [end-to-end speed comparison](flybody-ground-performance.md) measures a 1.28–1.40× neural-throughput slowdown relative to the old system. It compares one fly at a time, with the same original UI, open anatomy inspection and sensory settings on both backends. Frame rate varied across the short measurement windows; this is not a stable-frame-rate guarantee.
