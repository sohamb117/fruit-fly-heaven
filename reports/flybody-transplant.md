# Native FlyBody transplant: one-fly observation

The default console now runs actual FlyBody joints, contact dynamics and wing aerodynamics in MuJoCo WASM. The original 3D console and FlyWire comparison remain available. BANC brain/VNC connectivity drives annotated motor neurons, modeled muscle force and native actuators. The actuator adapter is a modeling assumption; it is not a learned locomotion controller.

## Mechanical validation

[Native/WASM validation](flybody-runtime-validation.json) compares a fixed-control native MuJoCo trajectory with WASM (maximum state difference 4.69e-8), checks original body mass/inertia, isolated identified joint stimulation, motor disconnection, claw actuation and wing motion with no externally applied root forces. This establishes the engine connection, not useful behavior. [Console validation](transplant-ui-validation.json) exercises the original 3D instruments, eyes, anatomy, popouts, pause, movement modes, clock and controls with one fly. All 30 legacy regression tests, 22 physiology/muscle tests and four importer tests pass.

## Observed behavior

The runs below predate the [ground-feedback repair](flybody-ground-feedback.md). The distance-based mouth proxy has since been replaced by native mouth contact; the earlier intake is historical and is not evidence for the current feeding boundary.

[Raw telemetry](flybody-one-fly-final.json) and [sampled 3D video](flybody-one-fly-final.mp4) record one fly starting at the original on-fruit pose. The run advanced 2.378 neural/body seconds in about 61 wall seconds. Motor coupling was disabled at 2.050 seconds. The clip samples roughly two frames per simulated second; it is not a continuous high-frame-rate recording.

The fly probes and briefly transfers 0.00098759 normalized food units into its crop under the former geometric mouth-contact proxy; this is not resolved labellar contact or validated feeding. It then tumbles, leaves and regains contact repeatedly, and moves around the bowl without controlled navigation. The largest sampled horizontal speed was 752.1 mm/s. No approach, controlled food landing or sustained flight was established. The monitor's orientation and upward-departure events can occur during tumbling; they are not evidence of successful localization or intentional takeoff.

After disconnection, maximum muscle force decayed to 0.000263, mean wing power to 0.000183, and horizontal speed from 61.02 to 0.094 mm/s. Gravity, contact and native position-actuator resting targets continue to act when neural drive is removed. No browser/runtime errors were recorded.

A separate [off-food start](flybody-one-fly-off-food.json) ran for 1.032 neural seconds, again with one fly. It recorded no intake (0) and no approach event. The body tumbled across the bowl and briefly contacted fruit; the orientation/landing monitor labels do not establish controlled localization or landing. No runtime errors occurred.

**The direct connection is implemented. The requested autonomous food-localization → approach → landing → probing/feeding → takeoff/flight sequence remains unachieved.** The remaining work is calibrated sensory/physiology and muscle-to-actuator control, including stable posture and wing steering, tested against held-out identified-cell/body measurements. A connectome and a working physics solver alone do not supply that calibration.

## Reproduce

```sh
node scripts/verify-flybody-runtime.mjs
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/verify-transplant-ui.mjs --one-fly
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node scripts/diagnose-banc-behavior.mjs --population=1 --ms=2000 --disconnect-ms=300 --follow=true --output=reports/flybody-one-fly-final.json
```

The server runs at `http://127.0.0.1:7842/?population=1&movement=direct&clock=neural&follow=1`. The viewing link selects one fly, direct movement, neural time and the existing follow camera. The raw reports include source hashes and the current numerical observations. The earlier surrogate-body observations are historical and are not benchmarks of this replacement.
