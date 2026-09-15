# Decoder feasibility diagnostic

This local experiment tests two different claims:

1. The existing decoded wing-control boundary can sustain flight in the full native body.
2. The existing 672-coefficient motor-neuron decoder can reproduce a successful controller from its permitted neural inputs and remain stable in closed loop.

No production code, coordinator state, cloud candidates, neural connectivity, physical parameters, contacts, actuator limits or joints are changed. The unchanged pinned browser WASM modules run in Node for these diagnostics. This is not a Safari runtime-equivalence test.

## Frozen system

- Bundle: `dist/training-lease-client/fruit-fly-training-client-1f3b0935af59`.
- Configuration SHA256: `1f3b0935af592a3edf283f02ae2d7eb570fa5c2a6acd225b429b6624d6ed328f`.
- Model fingerprint: `9ff0d93ef893be939bd9717b3469325e33207d8c0ee2c2f8c5fc4124c0677fc7`.
- 95 manifest files and 59 configuration asset hashes are verified before execution.
- Full native body: 57 generalized positions, 56 velocities, all 50 nonroot joints retained; native timestep 50 microseconds.
- Original two-millisecond body/neural blocks, one-millisecond MN activation/history updates, and 0.2-millisecond wing-control updates.
- Original 0.5-second airborne warm-up restrains the root only during setup. The root is released for five scored seconds. Applied external-force arrays remain zero; the fixture checks there are no direct root writes after release.

## Calibration and oracle

`calibrate.mjs --calibrate` probes 20 phase-basis coefficients: two bounded wing powers and three steering axes per side, each with constant/sine/cosine components. The emitted commands still obey the original [0,1] power and [-0.25,0.25] steering bounds and original power scaling.

The restrained full-body probe uses zero nonwing motor-neuron rates and synthetic empty wing event packets. It measures the final 100 milliseconds of native aerodynamic force/torque, a finite phase-conditioned average. Calibration is neither free flight nor a live BANC result. The model and its nonwing joints remain dynamic.

A bounded allocator uses the measured local force/torque Jacobian. A diagnostic quaternion/height feedback controller requests corrections from root state. This direct state access is an explicit intervention at the decoded-control boundary, not a proposed replacement for biological proprioception or the BANC network.

`calibrate.mjs --screen` runs a constant-trim control and three feedback settings (natural frequency 12, 24, 36 rad/s; damping ratio 0.9) on calibration seed 190888. These are development trials. A successful setting must also be evaluated with the full live BANC input stream before counting as a positive control for the complete system.

## Success and scope

Record both the unmodified maintained-flight objective and continuous qualified airtime. The production objective calls a five-second trial successful when its final continuous qualified bout is at least one second. The stronger diagnostic target is an uninterrupted bout through the horizon after the initial 50 ms measurement window (approximately 4.952 scored seconds), without environment contact or physical failure. Report lesser outcomes as lesser outcomes.

Native `ncon` includes self contacts. Generic native contact counts must not be presented as environment contacts. The original scored environment-contact measurements remain separate.

This begins airborne. It does not establish takeoff, landing, food localization, feeding, biological fidelity, or reliable long-duration flight.

## Decoder fitting and validation

Keep the original decoder feature clock, motor-neuron identities, input mask, phase features and all 672 parameter bounds. Fit only supervised command targets collected with live BANC, using two teacher seed groups for fitting and a separate third group for validation. Preserve five preceding one-millisecond excitation rows and explicitly account for the final completed but unapplied interval.

Report offline raw and clipped command error separately from closed-loop physical outcomes. A low prediction error alone is not a successful decoder. Evaluate a frozen fitted vector on additional seed groups; never update it from held-out outcomes. Seed variation changes the inherited body stance under the original reset, not independent neural randomization.

If the first oracle fails, that is failure of the tested controller, not proof the actuator interface is uncontrollable. If a first supervised fit fails, that is failure of that fit and dataset, not proof no decoder parameters can work.
