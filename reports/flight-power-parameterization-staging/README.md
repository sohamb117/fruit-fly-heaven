# Optional activation / amplitude power parameterization — staged only

The proposed wing source preserves the legacy calculation when `wing_actuation.power_transfer` is absent. The optional, strictly validated metadata is:

```json
{"schemaVersion":1,"profile":"activation-amplitude-v1","activationGain":1.999999638880142}
```

For each side it changes the request from `clamp(rawForce * powerGain)` to `powerGain * clamp(rawForce * activationGain)`. `activationGain` is an owned, frozen, positive finite numerical normalization. Its proposed value retains the old `exp(.693147)` activation gain; this is a modeling prior, not measured biology. The existing learned power coordinate becomes amplitude after that clamp. Under this profile its physical gain must be positive and at most one; invalid updates fail before interpreter state changes. Validation and permitted gains remain unchanged without the profile.

The staged migration sets power log-gain maximum to `0` and initial value to `log(.90)` (`-0.10536051565782628`), preserving its minimum/search scale and the complete other 26 parameter definitions and initial values. The changed label states the new meaning. `parameter-migration.json` carries the complete proposed parameter list, old and new vectors, source identity, and profile. It is a configuration fragment, not a runnable configuration: no plan, canonical manifest, fingerprint, checkpoint or optimizer history is changed. A future integration needs a new config/model identity and fresh training state; the historical 27-vector is not silently reused.

At the migrated initial value, the dynamic wing calculation is exactly the existing common `.90` diagnostic: `.90 * clamp(rawForce * 1.999999638880142)`. For saturated nonzero force, varying the log-amplitude now varies requested power. Zero input still requests zero. Bilateral inputs remain independent. The existing deployment threshold (`requested > .01`), ramp, target tables, residual calculation and servo clipping remain in place. Consequently this parameter affects wing-table interpolation and steering's final `residual * power` term; monotonic requested power does **not** establish monotonic aerodynamic force or improved flight.

`staged/web/flybody-wings.js` retains its canonical `./training/flight-parameters.js` import. That dependency is copied unchanged solely to make the staged source importable. No runtime, native kernel, model XML, body state, task state, neural state or feedback/controller path is added.

Prepare and validate using pure JavaScript only:

```sh
node reports/flight-power-parameterization-staging/build.mjs
node --test reports/flight-power-parameterization-staging/power-parameterization.test.mjs
```

The tests compare all dynamic wing buffers, tables, phase, frequency and control outputs byte-for-byte over 2,048 varied timesteps, separately for profile-absent legacy parity and profile `.90` versus the existing common `.90` variant. Interpreter power gain itself differs by the explicit migration. Additional tests cover schema/ownership, transactional rejection of old gains, requested-amplitude sensitivity, zero input, deployment and the 27-coordinate migration. These are numerical interpreter checks; no native or full-fly simulation is performed.

All **8 tests passed** on Node 23.7.0. `validation.json` records the final source/artifact hashes and both 2,048-step parity gates; `test-results.tap` preserves the output. Live source and the baseline input pins still match their original hashes.
