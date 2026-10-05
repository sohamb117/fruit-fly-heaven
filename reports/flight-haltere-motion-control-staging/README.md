# Externally prescribed left virtual-haltere motion

Diagnostic-only sensor-motion positive control. It is **not brain-autonomous flight**, a physical actuator, a passive-coupling implementation, or a training candidate. No live files are changed.

```js
import {prescribeLeftHaltereSensorMotion} from './prescribe-left-haltere-sensor-motion.mjs';
// Call after native world.copyPose has refreshed fly.feedback, before the next
// sensory-current sampling block. Reapply after each later world copy.
const diagnostic = prescribeLeftHaltereSensorMotion({body, fly}, {
  onsetSeconds: 0.1,
  leftAmplitudeRadians: 18 * Math.PI / 180, // or 36 degrees
});
```

Before onset the helper leaves all inputs and aliases untouched. At `body.time >= onsetSeconds - 1e-9`, it assigns only a shallow copy of `fly.feedback`, containing a new `halterePower` array. This explicit one-nanosecond tolerance prevents floating-point accumulation just below the 100 ms boundary from delaying onset by an entire 2 ms block; it permits at most 1 ns of early application. Left power becomes the declared amplitude divided by pi/4; right power is copied from the actual current `body.halterePower[1]`. Thus 18/36 degrees correspond to .4/.8 in this transducer's amplitude convention. The helper neither changes phase nor commands motion in native physics.

The returned diagnostic has kind `externally-prescribed-left-virtual-haltere-motion`, `active`, owned two-element arrays `actualMusclePower` and `transducerAmplitudeEquivalentPower`, and scalar `amplitudeRadians` (configured left amplitude), `onsetSeconds`, and `bodyTimeSeconds`. Before onset, the equivalent power pair is the actual native pair; `active` distinguishes whether the configured amplitude has been applied. Diagnostic arrays do not alias native or applied feedback arrays.

Seven tests pass, covering the exact onset boundary and its roundoff tolerance, pre-onset aliases, native muscles/controls/phase/velocity remaining unchanged, right-side freshness, declared amplitude endpoints and invalid-input transactionality. See [validation.json](validation.json) for source hashes. These fixture checks do not establish a physiological amplitude, passive transmission or flight performance.
