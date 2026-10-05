# Current BANC interface audit

The anatomical route is connected, but its transfer functions are not jointly calibrated. Current production also contains an **ineffective chemical neuromodulator gain**: its configured coefficients multiply zero input. This audit reads source, prepared parameters and recorded results only. It runs no new simulation and changes no production equation. [Exact counts, intersections, source hashes and protocol](audit.json).

## A concrete functional gap

Both neural kernels compute `current = (external + tonic) * modulation_gain + hormone_term - adaptation`. Dopamine, octopamine, serotonin and tyramine receptor gates affect that gain; their channels do not independently enter the fast conductance sum. See [WASM equations](../../packages/banc-runtime/native/core.cpp:22), [gain/current](../../packages/banc-runtime/native/core.cpp:33), and [WebGPU counterpart](../../packages/banc-runtime/src/neural.wgsl:44).

| Exact current configuration | Cells |
|---|---:|
| Nonzero chemical modulation gain coefficients | 5,357 |
| Motor profile, octopamine coefficient ≈0.3 | 805 |
| Kenyon profile, dopamine coefficient ≈0.2 | 4,552 |
| External sensory-input indices | 18,232 |
| Gain-bearing cells receiving external input | **0** |
| Gain-bearing cells with nonzero tonic current | **0** |
| Motor cells receiving external input | **0** |

The worker creates a zeroed input array and writes only encoder indices. Every gain-bearing cell therefore has `external + tonic = 0`; every other cell has gain 1. Changing chemical modulation cannot change the membrane-current term in this production configuration. This is a source-level proof, not a claim that neuromodulator neurons or kinetic gates are absent. Hunger/insulin/AKH modulation of 3,007 olfactory-profile cells, crop/metabolism and muscle fuel remain active. A replacement equation needs a defined physiological target; moving the multiplier arbitrarily would add another uncalibrated assumption.

## What is sourced versus assumed

| Boundary | Implemented or sourced | Remaining assumptions |
|---|---|---|
| Network and cell identity | BANC v888: 175,401 neurons, 13,542,180 chemical edges, 42,199,458 contacts; annotated or predicted transmitter identities | Connectivity alone does not provide membrane physiology or functional synaptic strength. Unknown-transmitter edges retain identity with zero conductance: 427,715 edges. |
| Cell physiology | Conductance neurons, adaptation/refractory periods, graded versus spiking profiles are implemented | Seven broad profiles; all 805 motor cells share one profile; no per-cell overrides. Parameters are explicitly unmeasured priors. |
| Synapses/receptors/gaps | Receptor-specific kinetic channels, chemical modulation gates, and two bidirectional GF–PSI correspondences exist | One 0.1 nS·ms/contact scale and 2 ms delay; transmitter-to-receptor defaults including inhibitory glutamate; no postsynaptic overrides. The two electrical conductances are assumed 0.02 nS. The rest calibration covers 200 ms, not the behavior sequence. |
| Sensory identity/input | Real retinal rendering and native organ contact; checked taste identity, side and leg mapping; unsupported auditory static input excluded | Taste contact is 150 Hz, odor is `3 + 65*concentration`, and body transduction gains are priors. The rate-to-current lookup matches isolated configured cells; full-network input is **added current**, not a clamped sensory firing rate. |
| Motor boundary | 454 annotated motor cells map to 135 groups; explicit antagonists and native joint-sign checks; native physics follows actuator commands | The 351 unassigned cells are not automatically relevant missing controls. Group-mean rate divided by **80 Hz** supplies every muscle's excitation. Rates already have a 50 ms filter; generic muscle activation/deactivation are 15/40 ms, with normalized maximum force 1. |
| Muscle-to-motion | Hill-like force/length/velocity/fatigue dynamics and native MuJoCo mechanics are implemented | Equal group strengths and normalized moment-arm surrogates; legs use antagonist differences to command position servos over at most 0.35 rad from rest, rather than measured muscle insertions. Proboscis coupling strengths are also relative priors. |
| Flight/internal state | Published-pattern-derived 235.813447 Hz oscillator, COM-corrected aerodynamic moment, bilateral muscle-dependent steering, hunger/crop/insulin/AKH state | Rate-to-force, deployment and steering gains remain modeled. Rotation sensing is an unsigned organ/side scalar, with no native haltere strain or signed phase tuning. Internal-state dynamics are not fitted to measured feeding-state responses. |

Code: [physiology](../../configs/banc-physiology.json:3), [edge preparation](../../scripts/prepare-banc.py:96), [isolated current calibration](../../web/banc-sensory-current.js:112), [sensory gains](../../web/sensory-encoder.js:7), [native body rates](../../web/banc-ground-sense.js:47), [motor and servo boundary](../../web/flybody-physics.js:73), [muscle kinetics](../../packages/banc-runtime/native/core.cpp:60), [proboscis assumptions](../../web/banc-proboscis.js:4), [rotation limits](../../web/banc-ground-sense.js:22), [internal state](../../web/banc/embodiment.js:9).

## What the current failure establishes

In the source-matched [live trace](../flybody-solid-wing-repair/current-live/behavior.json), DLM left/right are already 89.94/97.67 Hz at 48 ms, exceeding the generic full-excitation threshold. At 136 ms they are 124.44/130.27 Hz and muscle activation is approximately 0.998. The tested m9, m4a and m4b reach-motor outputs, pump and proboscis remain zero at both times. The first wing impact is later, at 138.05 ms, with the body already tilted about 66 degrees. Existing [causal contact replay](../flybody-wing-correction/README.md:40) establishes that the impact produces the subsequent full inversion and launch in that trajectory; it does not explain the earlier command selection or tilt.

This is inappropriate recruitment for the requested grounded probing/feeding sequence. It is **not** evidence that the biological fly intentionally chooses flight, nor proof of which numerical interface caused that pattern. No food bearing or task-stage controller enters the native motor decoder. The fixed oscillator is an asynchronous muscle bridge, not the released learned flight policy.

The [held-posture neural assay](../banc-sensory-recruitment/README.md:16) removes mechanical feedback as a cause of the missing command: six-tarsus sugar alone gives DLM 88.8/85.6 Hz and zero m9/m4b after 200 ms. All 15 tarsal pairs also leave these reach channels silent. Verified labellar stimulation can recruit them. These are open-loop input experiments, not a natural contact sequence; the 625-configuration mouth-only grid also fails to reach food from the initial held body/leg posture. Reducing motor force cannot create absent m9/m4 spikes, and raising their gain cannot resolve unreachable posture.

## Executed discriminating assay

The bounded neural-only assay is now complete: **21 conditions in 74.7 wall seconds**, with all four current-gain baselines reproducing the earlier result exactly and all executed source hashes unchanged. It crossed chemical gain 0.5/1/2 with zero input, actual body only, body plus six-tarsal taste at 25/75/150 Hz, six-tarsal 150 Hz alone and bilateral labellar 150 Hz alone. Each run used 200 ms stimulation plus 100 ms all-input-off and recorded every 20 ms. Electrical edges remained unchanged. No muscle/native replay or production tuning was performed.

At current gain, all three body-plus-taste strengths leave m9/m4a/m4b at zero raw spikes, despite strong and nonmonotonic DLM recruitment. Halving gain removes DLM but also weakens the labellar positive response and still fails to recruit tarsal probing output. Doubling gain produces DLM activity even with zero added input. Thus neither simply weakening taste nor uniformly rescaling synapses is supported as a solution. Actual sensory firing is measured, not clamped, and itself changes with recurrent gain.

[Full results and raw off-decay summary](gain-results.md) · [Raw per-cell measurements and copied-model hashes](gain-assay.json) · [Compact spike summaries](gain-summary.json). The stored motor conductance/current fields are total state quantities, not per-receptor currents.

The next evidence needed is an independently constrained stimulus-to-afferent and identified reflex-circuit response target; fit these interfaces jointly against rest, stimulus and recovery rather than selecting parameters because a desired behavior appears. Missing motor spikes cannot be repaired by force scaling, and observed numerical sensitivity is not a biological calibration.
