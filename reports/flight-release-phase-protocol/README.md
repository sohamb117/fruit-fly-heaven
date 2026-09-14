# Fixed-release-phase diagnostic proposal

Status: report only. No diagnostic runtime, model, configuration, native episode or neural episode was created or executed. The current two-generation run and its reserved held-out evaluation must finish before this proposal is considered. This proposal does not authorize a mid-run change.

## What the existing records establish

The current oscillator starts at zero (`web/flybody-wings.js:62`). Its phase advances after each wing target calculation by `2*pi*frequencyHz*0.0002`, modulo `2*pi` (`web/flybody-physics.js:223`, `web/flybody-wings.js:127`). The live warmup lasts exactly 500 ms: 10,000 native 50 us steps and 2,500 oscillator updates. No phase reset occurs at release.

The metadata frequency is **235.813447 Hz**. Nominal warmup spans **117.9067235 cycles**, with release clock phase **5.697111772874852 rad**. Current frequency perturbations have log SD `0.1 * 0.05 = 0.005`; one positive SD changes accumulated phase by **3.7134247892390304 rad (0.5910099110 cycles)**, and one negative SD by **-3.694904005836363 rad**. These are unwrapped changes; phase is periodic.

`quantification.json` pins the inspected source/configuration and all eight completed generation-zero search records. A scalar recurrence using their exact frequency and update count reproduces each recorded release clock phase with **zero numeric difference**. This calculation did not instantiate a body or brain. The records' native release time is 0.4999999999999531 s because of floating-point accumulation; their logical warmup remains 500 ms.

Clock phase is not a measured joint angle. The final warmup target used the preceding clock phase, and native wing angles/velocities can lag it. The reset keeps the root pose fixed and root velocity zero during warmup, while leaving all joints and neural/muscle history dynamic. Release can therefore occur at different internal wing motion and momentum states. The source and records establish that frequency changes this initialization; they do not establish how much of its reward effect is caused by it.

## Primary protocol: vary only initial oscillator phase

Reference the final saved generation-two 27-vector without refitting, and use the already consumed training seed **1474186734**. This seed is absent from both reserved validation lists in the active configuration. Pin the final checkpoint and its full vector before creating any diagnostic job. All five primary arms use that exact vector, seed, native/backend provenance, scene, sensory configuration, intrinsic models and event contract.

Run these arms sequentially, with a fresh BANC/body instance per arm:

| Arm | Initial phase offset | Purpose |
| --- | ---: | --- |
| A | Existing unmodified zero | Reference |
| B | Explicit zero | Verify the diagnostic initialization is behavior preserving |
| C | pi/2 | First phase intervention |
| D | pi | Second phase intervention |
| E | 3*pi/2 | Third phase intervention |

An existing arm A can be reused only if its full vector, seed, source/backend identity, starting state and recorded digest match exactly. Otherwise run A; do not treat a nearby historical vector as equivalent. Arm B must match A's complete 2 ms qpos/qvel digest, wing-event history and evaluation outcome before interpreting C–E.

The smallest proposed intervention uses the already writable oscillator state. In a future diagnostic-only `createWorld` setup, after setting interpreter parameters and before the existing final initial pose copy, event packet zero or warmup, assign the declared offset once to `body.wings.phase`. Set the presentation/sensory alias `body.wingPhase` to the same number at that point. The latter is not a second dynamical control: it prevents initial feedback from reporting the old phase until the first native step synchronizes the alias. Reuse the existing pose-copy path; no additional forwarding, placement or pose reset is needed. Validate finite offset in [0, 2*pi), body/native time zero and empty event history before assignment.

This is a proposed diagnostic hook, not an existing public configuration option. Do not add phase to the 27 learned parameters or change the active initial-condition schema. A future implementation must live in a separately pinned diagnostic source/configuration namespace and never submit its results to the coordinator.

Preserve exactly:

- 500 ms of the same live, closed-loop BANC warmup, followed by five scored seconds or existing physical termination;
- root pose [0, 0, 3.5, 1, 0, 0, 0], the existing warmup-only restraint and its cumulative write audit;
- all neural, event, muscle and native clocks, kernels, parameters and their ordering;
- all initial native qpos/qvel, activation, internal state and seeded stance/rest geometry;
- the same oscillator frequency throughout warmup and scoring;
- no further manual phase assignments, direct root writes, native state resets or applied external forces after initialization/release.

The known assignment does not introduce a phase-dependent force or feedback controller. It changes which point of the existing fixed waveform is reached when the unchanged warmup restraint ends.

## Measurements and interpretation

Record the declared initial phase; actual oscillator phase, target, native wing qpos/qvel, whole-body COM velocity and available subtree angular momentum at release; and the existing event/muscle/native clock and root-write audits. If using MuJoCo `subtree_angmom`, label it rigid-body momentum about the subtree COM, excluding joint armature. Compare the phase recurrence with the measured release phase using a tight numerical tolerance rather than changing the integration clock. Do not force the joints onto the oscillator target.

Keep the existing return and success criteria, but report survival, contacts, best/current/total qualified airtime, angle, angular-rate RMS, support and 50 ms vertical-speed gate separately. Show early 0–100 ms after release and subsequent behavior separately. Publish all four phase outcomes; report range and mean without choosing the best phase as a new model parameter.

If otherwise identical arms differ substantially, the fixed-frequency model is sensitive to this initial phase and the warmup state it induces. That is evidence that a fixed-phase frequency comparison can be misleading, not proof that all frequency effects are artifacts. Similar behavior at four phases only bounds this small probe; it does not prove global phase robustness.

**Important limit:** an initial phase offset changes the entire closed-loop warmup. Wing loads, sensory currents and resulting neural/muscle states may diverge before release. This protocol tests sensitivity to the full phase-conditioned initialization. It does not isolate an instantaneous mechanical release phase while holding the warmed brain and body state constant. No saved neuronal state is transplanted between arms.

## Optional second question, outside the fixed-27 protocol

Holding all 27 coefficients fixed cannot estimate a sustained-frequency main effect. If a frequency comparison is still needed after the phase-only result, separately declare a two-frequency by four-phase comparison, keeping the other 26 coefficients and the training seed identical. Choose the two frequencies before seeing those new results. For each frequency f and common target release phase Phi_j, initialize once at `wrap(Phi_j - 2*pi*f*0.5)`; use the same 500 ms warmup and no subsequent phase reset. Verify the actual release phase from the discrete recurrence.

Comparing frequency means and their phase interaction would separate phase-specific ranking from a frequency effect that persists across these initialized phases. It would still include the frequency-conditioned live warmup, not isolate a pure mechanical frequency response. This extension deliberately relaxes the fixed-27 requirement and is **not part of the primary proposed runs**. No such jobs have been prepared or scheduled.
