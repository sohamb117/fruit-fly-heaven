# Opt-in wing event → muscle interface

The [48-unit event-to-excitation adapter](../../web/flybody-wing-event-excitation.js) is implemented, with [focused numerical tests](../../web/test/flybody-wing-event-excitation.test.mjs). It consumes exact motor events; the 50 ms rate estimate remains diagnostic only. It uses no rate/80 conversion, desired body state, assigned corrective spike phase, or oscillator reset. This remains a phenomenological effector hypothesis, not a fitted Drosophila force model.

The [native muscle characterization](characterization/README.md), [four matched restrained phase probes](../wing-event-phase/README.md), and [four-arm free-body positive control](../flight-event-control/run-001/README.md) have completed with source-pinned results. The synthetic event feedback controller kept maximum tilt at 7.20 degrees over 1.5 seconds; the fixed-trim event arm reached 43.52 degrees. The recorded BANC DLM events still nearly saturate the declared excitation prior, and one b1 event's phase changes its early torque transient. These findings validate bounded mechanical/numerical capabilities, not biological calibration or BANC-controlled flight.

Actual BANC event-path integration has now completed its [first paired native evaluation](../flight-event-live/comparison-final/INTERPRETATION.md). The feature-off baseline reproduced its prior physical digest and every motor packet exactly. The event arm took off and qualified for 174 ms of flight versus 166 ms, but hit excessive rotation earlier, at 392 ms versus 480 ms. The implementation is exercised; **stable BANC flight remains unvalidated**. This is one fixed seed, with no optimizer update or successful landing. The [current diagnosis map](../flight-event-control/README.md) separates this boundary from DLM cell physiology and missing signed sensory feedback.

The [latency assay](../flight-muscle-latency/README.md) supports testing this boundary: the fixed classical controller passed with native 15/40 ms muscle gating (maximum tilt 6.68°), but failed with an additional 50 ms rate smoother (112.62°). That result does not validate the event kernels below or guarantee that sparse event control will work.

## States, equations and initial priors

Keep separate states for the **24 steering MNs, 10 DLM MNs and 14 DVM MNs**, joined by the pinned IO indices. Do not pool events before applying each unit's nonlinear excitation rule. Times below are in milliseconds.

For each neuron, maintain two states `d,r`. At its event, increment both by one. Between events decay them analytically with `tauDecay,tauRise`. Define:

```text
tPeak = tauRise*tauDecay/(tauDecay-tauRise) * log(tauDecay/tauRise)
N = exp(-tPeak/tauDecay) - exp(-tPeak/tauRise)
c(t) = max(0, [d(t)-r(t)]/N)
excitation(t) = -expm1(-log(2)*c(t))
```

An isolated event has kernel peak 1 and excitation peak 0.5. Additional events add kernel state; excitation approaches 1 smoothly. The nonnegative guard handles rounding, not spike clipping. Kernel amplitudes and the common `log(2)` recruitment gain are dimensionless modeling choices. Record high-excitation fractions; do not normalize each observed train or lower its gain to conceal excessive DLM firing.

| Family | Rise / decay | Interpretation |
|---|---:|---|
| All synchronous steering types | 1 / 5 ms | Explicit initial fast excitation prior, identical across types and sides. These are not measured force constants or inferred from afferent-to-EMG latency. |
| DLM | 6.2 / 82 ms | Calcium-like prior using the reported DLM fitted constants; choosing this peak-normalized double exponential and its amplitude remains a modeling assumption. |
| DVM | 6.2 / 82 ms | Separate configurable family, initially borrowing DLM values by an explicit homology assumption. No DVM-specific measurement is claimed. |

The DLM measurements concern GCaMP calcium and wingbeat-frequency responses, not force. Steering phase-dependent action is documented, but there is no validated common twitch kernel for all twelve mapped types. See the [bounded primary-source review](../flight-sensory-observability/motor-timing-review.md) and [Hürkey et al. 2023](https://www.nature.com/articles/s41586-023-06099-0). The retained native 15 ms activation / 40 ms deactivation gate remains a further phenomenological stage; its combined response must not be called a 6.2 ms force rise.

For each original muscle mapping, average **unit excitation after the nonlinearity**, with equal weights as an explicit default. Send that excitation to the real native kernel, which continues to compute activation, fatigue, length/velocity dependence, energy dependence and force. Existing DLM/DVM group-force averaging into left/right wing power stays fixed. Individual calcium-like states remain visible; fatigue remains a muscle-group state, not a modeled state for every fibre. Nonwing muscles retain their rate-based path. The opt-in body integration owns 28 wing muscle states and merges them into the original mapping order exactly once for force, effort and telemetry.

## Interval integration and causality

Accept only the existing reader's validated consecutive 2 ms packets. First baseline emits no historical events. Preserve the 0.5 ms event grid, identities, and `(fromTimeMs,timeMs]` boundary convention. Zero-event intervals still advance decay. Snapshots must include all kernel states, pending boundary events, muscle states and the last consumed timestamp.

For each native **1 ms** muscle interval `[a,b]`, split integration at every event. Its contribution to the kernel integral is analytic. For an event at `s<b`, with `u0=max(0,a-s)` and `u1=b-s`:

```text
integral K = {tauDecay*[exp(-u0/tauDecay)-exp(-u1/tauDecay)]
            - tauRise*[exp(-u0/tauRise)-exp(-u1/tauRise)]}/N
```

The mean excitation is the integral of the nonlinear excitation, not the nonlinearity of mean kernel state. Use 8-point Gauss–Legendre quadrature on each event-separated segment; compare against 4 points and fail if the interval-mean discrepancy exceeds `1e-7`. Validate against 16 points offline. Segment endpoints have zero measure: an event exactly at `b` contributes nothing to `[a,b]`, then seeds the next interval.

**Apply the native update at the interval's right boundary.** Integrate the body through `[a,b]` with the previously available muscle force; at `b`, advance the native muscle kernel by 1 ms using the completed interval's mean excitation, and use its resulting force thereafter. Applying this average at `a` would allow a later event to act early. This causal sample/hold adds up to 1 ms latency and changes the update phase relative to today's advance-before-body loop; isolate that difference in the control comparison below. No event in the second millisecond may affect the first, and no future body observation enters the adapter.

The implemented excitation API is `createWingEventExcitation({io, config})`, then `accept(packet)`, `finishInterval(endMs)`, `readState()` and `snapshot()/restore()`. The native body integration uses `body.enableWingMotorEvents(config)`, `body.acceptWingMotorEvents(packet)` and an owned diagnostic `body.readWingMotorEvents()` readout. Missing/duplicated packets and interval overruns are rejected. Preserve 28 forces/states in explicit original mapping order plus 48 per-unit diagnostics, and leave the legacy mode unchanged.

## Completed finite validation protocol before a live BANC candidate

1. **Transport and numerical tests, no body:** single events at 0.5/1/1.5/2 ms, exact endpoints, empty intervals, replay partitions, snapshots and simultaneous/staggered DLM trains. Check integral area, causal prefix equality, quadrature convergence, zero-input decay and no double counting. Events at different subinterval times must remain distinguishable; each unit's contribution must be retained. Compare per-unit nonlinear pooling with the deliberately incorrect pooled-input operation as a regression test.
2. **Fixed excitation characterization, no body:** steering periodic trains at 50/100/200 Hz; DLM/DVM at 3/8/12 Hz; also replay the already captured 48-unit events unchanged. Report excitation, native activation, fatigue and delivered force separately. The observed 83–156 Hz startup DLM activity remains visible as a neural-calibration problem; this adapter must not rescale it into a nominal physiological range.
3. **Four restrained native phase probes:** one steering muscle, identical event count and interval, four realized phases of the existing wingbeat clock. Keep all other trim inputs fixed. Measure whole-body COM fluid torque with the established conventions. Report the actual quantized phases. A flat or incorrect phase response reveals a limitation of this effector/basis combination; do not invent a preferred phase to pass the test.
4. **Four fixed-controller 1.5 s releases, one body at a time:** historical native-excitation path; the same native excitation with the causal right-boundary schedule; synthetic event-driven closed loop; paired event-driven open loop. Reuse the established reduced plant, gains, trim and pass criterion. No gain search. For the event positive control only, a declared synthetic pulse generator can use a monotone activation-versus-periodic-rate lookup: three family/group curves, fixed grid `0,2,5,10,20,40,80,120,160,200,300,400 Hz`, preserving the 2.5 ms refractory bound. Convert desired force to required activation using current fatigue and all force multipliers before the lookup; log unattainable commands. This diagnostic event generator is not BANC or a proposed biological controller.

Use the same 100 ms prepared physical warm state for the release comparisons, with initial native force at trim and fatigue zero. Initialize event states from a recorded deterministic periodic prehistory; save its phase, complete states and hashes. This is a controlled prepared state, not biological steady flight. Compare the causal excitation arm against the event arm to separate scheduling, discrete control authority and kernel latency. If a fixed-controller arm fails, report it without retuning in this stage.

Only after these checks should an opt-in live candidate consume actual BANC events. Pin the new interface configuration, IO, source, model, force reference and interpreter gains. Do not change intrinsic firing, receptor signs, wing-force centering or gains in that same comparison. A successful plant positive control establishes compatibility of this effector hypothesis with the reduced mechanics; it does not establish biological calibration or successful BANC flight.
