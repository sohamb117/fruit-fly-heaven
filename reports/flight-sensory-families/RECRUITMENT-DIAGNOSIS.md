# Recruitment diagnosis and next experiment

This is a read-only audit of the unchanged prepared graph/physiology and saved native-Dawn traces. No neural or body simulation was run, and no model, runtime, input mapping or training parameter was changed. `audit-recruitment.py` reproduces `recruitment-audit.json`; that file pins every data/source input it used. The old frozen sensory context remains the subject even if a newer console manifest excludes the 50 proximal hairplate cells.

**The evidence favors uncalibrated network-to-motor recruitment over a demonstrated unit error. It does not yet establish positive recurrent saturation.** Similar DLM means arise from very different amounts of total network spiking, and the DLMs are not at their hard refractory limit. The next useful test is an exact replay of measured DLM excitatory/inhibitory conductances into the existing motor neuron and the separately validated ionic reference, before choosing a broader stimulus sweep or changing production physiology.

## What the implementation actually means

- `scripts/prepare-banc.py` multiplies contact count by 0.1 **nS·ms/contact**. A spike is released as `1/dt` in **ms⁻¹**, not Hz, and the two unit-DC-gain receptor filters turn that product into conductance in **nS**. For an isolated event, the infinite discrete-time integral of the filtered conductance is the integrated edge weight. The ×1000 factor occurs only in the separate Hz readout. Both `native/core.cpp` and `src/neural.wgsl` implement this convention. No obvious factor-of-1000 or sign error was found; this is source inspection, not a new numerical impulse test.
- Chemical weights are nonnegative; postsynaptic reversal potentials determine sign: nAChR/explicit cation override 0 mV, GABA/GluCl/HisCl −75 mV. All glutamate currently defaults to inhibitory `GluCl_assumed`; receptor overrides are empty. This is an explicit receptor prior, not measured target-cell receptor expression. Unknown transmitter edges are retained with zero weight.
- Monoamine gates modify `(external current + tonic current)`, not chemical conductance. The DLMs have zero external input, tonic current and hormone coefficient, so their configured octopamine sensitivity does **not** directly amplify their incoming synaptic drive. State slot 7 is negative adaptation in these DLM traces; it excludes chemical driving current and must not be mistaken for total neuronal input.
- All 10 DLMs use the same generic motor profile: C=40 pF, leak=2 nS, rest=−60 mV, threshold=−42 mV, reset=−58 mV, refractory=2 ms, adaptation increment=0.4 pA and decay time=150 ms. Passive current to threshold is 36 pA; steady mean spike-triggered adaptation at 110 Hz would be about 6.6 pA. The profile is explicitly an uncalibrated prior. Its discrete refractory ceiling is 400 Hz, not 100–110 Hz.

## Actual DLM connectivity and recorded state

Each DLM has 426–559 chemical inputs, total integrated weight 414.8–631.6 nS·ms. Per-cell excitatory nAChR sums are 252.1–393.4; inhibitory GABA sums 158.7–238.0; inhibitory glutamate sums 1.9–15.8 nS·ms. No currently declared body, odor, taste or vision input neuron directly contacts a DLM. The strongest inputs are mostly VNC interneurons, especially annotated `IN19B043`; their actual activity has not been recorded in this assay.

An illustrative DC calculation, **assuming every presynaptic input fires at the same constant mean rate**, places the passive DLM threshold at only 3.75–6.86 Hz of common presynaptic activity, before adaptation. That demonstrates the gain implied by the selected weights and leak; it is not an estimate of actual input rates. At a hypothetical 100 Hz in every source, total membrane-synapse conductance would be about 41–63 nS. Per-source refractory-ceiling DC bounds are 166–251 nS; the much looser instantaneous positive-filter bounds are 828–1,257 nS. These are algebraic limits, not observed activity, and exclude the 2 nS leak. The machine-readable report retains each receptor separately.

All 10 DLM transmitter labels are **unverified predictions of GABA**, with recorded prediction scores 0.383–0.5664. Therefore their 39–66 outgoing CNS edges are currently inhibitory GABA_A; summed outgoing weights are 6.8–12.3 nS·ms, with a largest single output edge of 1.2 nS·ms. Those sums are distributed over recipient cells and are not one conductance acting on the DLM. Some outputs return to its presynaptic interneurons, but a direct excitatory DLM feedback loop is not present under this mapping. The central transmitter prediction does not control the separate positive muscle-excitation mapping. This is a priority for annotation/receptor validation, **not** evidence authorizing an arbitrary sign flip. No DLM has a prepared electrical edge.

In the saved all-input run, individual DLMs fire 80–160 Hz over (100,300] ms. Their mean total conductance is 9.98–15.16 nS, including leak, with sampled peaks of 16.64–25.95 nS. Minimum exact interspike intervals are 3.5–4.5 ms; none reaches the 2.5 ms discrete refractory floor. Zero external current leaves each DLM at exactly 2 nS conductance with no spikes. Other neurons still spike in that zero-current graph.

The all-input and odor-only graphs emit respectively 450,744 and 445,566 spikes during (100,300] ms; body-only emits 118,668, native-transducer-only 122,161 and fallback-only 117,967. Nevertheless all recruit DLM near the same broad range. This suggests a convergent or nonlinear motor pathway but does not prove a positive-feedback attractor, refractory saturation or physiological correctness. Excitation and inhibition cannot be separated from the saved total-conductance field.

## Sensory conversion is not a full-network rate clamp

`web/banc-sensory-current.js` inverts an **isolated, disconnected WASM neuron** with zero hormone state, at least 1 s settling and a 4 s measurement window. The experiment instead starts a recurrent network fresh, observes its first 300 ms and retains hunger=0.65, AKH=0.65, insulin=0. Its olfactory profiles therefore receive another 2.535 pA of hormone current, in addition to the roughly 20 pA mapped odor current. Internal and recurrent drive were intentionally not subtracted by the mapper.

The saved 300 ms reference already demonstrates different target and actual rates:

| Rotation afferents | Requested Hz | Actual raw mean Hz, (100,300] ms | Silent cells |
|---|---:|---:|---:|
| Haltere left | 2.885 | 4.561 | 96/171 |
| Haltere right | 16.552 | 1.624 | 134/157 |
| Wing base left | 47.846 | 46.855 | 3/62 |
| Wing base right | 47.846 | 48.305 | 4/59 |

These are true count differences, not the 50 ms filtered output. A finite 200 ms counting window has 5 Hz single-cell resolution and is not proof of steady state. Within a group, cells receiving identical requested rates can differ substantially. The other odor/fallback/body afferents were not read out, so their achieved rates remain unknown. The mapper also raises positive requests below 2 Hz to 2 Hz: 45 of 1,776 active requested inputs meet that condition here. This discontinuity is real but cannot by itself explain the odor-only result, whose requested rates are near 40 Hz.

The historical rest calibration selected 0.1 nS·ms/contact using only 200 ms of zero external input followed by 200 ms of an older nominal 150 Hz sugar-current heuristic. At the selected weight, its sugar arm already recruited asynchronous wing neurons (mean 47.09, maximum 94.74 Hz filtered output). That report lacks the exact modern graph/source/parameter/group pins and DLM-specific counts. It establishes neither general sensory selectivity nor calibrated DLM rates, and its sugar stimulus is not equivalent to the current mapper.

## Minimum next deterministic tests, in order

1. **Capture and replay actual DLM receptor conductances.** Rerun the pinned saved all-input context solely to observe the 10 DLMs, with baseline byte parity against the existing 2 ms traces and final forward-state hashes. Capture all five membrane receptor decay gates after their update on every 0.5 ms neural step, plus the gain-adjusted external/tonic/hormone term *before* subtracting legacy adaptation. Record complete initial DLM state and receptor kinetics, units, step timing and sources. DLM electrical input is zero in this graph. Retain the other four receptor gates/gain if convenient for auditing, even though the direct DLM current is zero.
2. **First require exact legacy reconstruction.** Replay the recorded conductance using the unchanged float32 threshold/reset/adaptation update and the original order of operations. Require exact full-state and event agreement, not merely similar average rate. Then apply the same conductance histories as `Σ g_r(t)·(E_r−V)` to the validated ionic reference at its 0.1 ms substeps, with an explicitly declared hold convention. Do not transfer the legacy adaptation current, add the publication's illustrative tonic current, copy a net current evaluated at the old voltage, or insert electrical coupling into the first comparison. Native loop outputs remain frozen. Low ionic rates under the same input would implicate generic motor excitability; persistently strong firing would implicate the incoming drive or both. Either outcome is diagnostic, not evidence of improved flight or a calibrated whole brain.
3. **Only if needed, distinguish sustained stimulation from network persistence.** A later predeclared fixed-current dose test can use fractions 0, 0.5, 0.75, 1 and 1.25 of the same captured pA array, with one 100 ms-on/200 ms-off branch. Read all stimulated sensory cells, DLM presynaptic cells and DLMs; preserve all intrinsic/hormone terms. Current scaling and requested-rate scaling are different interventions and must not be mixed. Compare actual afferent counts, receptor-specific DLM conductances, ISIs and post-offset decay. Persistence above the matched zero-input trajectory indicates state-dependent memory; it alone does not prove positive recurrence because slow gates and intrinsic dynamics also retain state. A recurrence intervention should be designed only after identifying the active upstream pathway. **This sweep is not implemented or queued.**

If the afferent readout is badly off target, small disconnected Dawn-versus-WASM checks at the exact saved pA values, first-300 ms versus steady-state windows, and actual versus zero hormones can separate backend/startup/internal-state effects. Existing physiology tests already cover sign and delay, so broad repeated testing is not the next step. No proposed change should be fitted solely to low DLM output or flight reward without preserving the intended sensory-to-motor pathway.

Reproduce the completed read-only audit:

```sh
.venv/bin/python reports/flight-sensory-families/audit-recruitment.py
```
