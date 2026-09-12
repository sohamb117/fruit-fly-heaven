# Baseline diagnosis before graded vision and behavior flight

This report describes the earlier all-spiking visual path and direct actuator controller. The habitat now adds a trained graded visual network and a behavior-mode flight program. See [current validation](embodiment-validation.json); the historical measurements below are preserved.

The quiet output is present inside the neural model. It is not merely a hidden animation or a rate rounded down by the body decoder. The current brain receives sensory stimulation, but its visual pathway fails to carry substantial activity onward and its selected forward-walking neurons receive strong inhibition.

## Controlled full-connectome probes

`scripts/diagnose-circuits.mjs 1000 3` runs 11 conditions on three independent seeds each, using the full 138,639-cell graph and reference Float64 WASM parameters. Each trial lasts 1,000 neural ms; rates use the 200–1,000 ms interval. [Raw measurements, per-cell voltages, traces, conditions, seeds, and graph hashes](circuit-diagnosis.json) accompany this report.

These are controlled input probes, not recordings of the live bowl. Baseline stimuli are bilateral odor at 56 Hz, sugar at 150 Hz, mapped photoreceptors at 16 Hz, self-motion input proxies at 15 Hz, and antennal proxies at 2 Hz. Other body-input rates are zero. There is no body motion or closed-loop feedback in these experiments. Seeds and per-neuron random streams are matched across conditions. Rates below are means across the three trials unless specified otherwise.

### 1. Activity largely stops in the early visual pathway

- The 6,244 stimulated photoreceptors fire at **15.481 Hz** on average.
- The 3,503 L1/L2 cells average **0.0321 Hz**.
- The selected Mi1/Tm3 relays emit **zero spikes**; Tm1/Tm2 average **0.0043 Hz**.
- All **12,245 T4/T5 cells emit zero spikes** during the measurement windows.
- Increasing photoreceptor stimulation to 60 Hz raises L1/L2 activity to **1.6695 Hz**, but T4/T5 remain silent.
- A moving sinusoidal grating with 2–60 Hz retinal drive, updated every 50 neural ms, also leaves T4/T5 silent in all three trials. This is a stress stimulus, not calibrated fly optics or a full direction-selectivity test.

The kernel transmits only spikes between cells. A subthreshold voltage response in a modeled early visual neuron therefore stops there. This is a concrete mismatch to graded signaling: a published connectome-constrained visual model uses non-spiking dynamics and graded synapses, with parameters constrained by a visual task. Simply attaching images to this LIF model does not reproduce those dynamics. [Visual model and methods](https://www.nature.com/articles/s41586-024-07939-3).

There is also a sign issue to investigate before calibrating vision. For mapped R1–6 → L1/L2 connections, the upstream matrix contains **9,429 positive and 2,209 negative rows**, with signed weight totals **+97,527 and −28,172**. Those are preserved model weights, not verified retinal physiology. Photoreceptor-to-lamina transmission is histaminergic and inhibitory, and communication is graded. This audit is recorded by `prepare-circuit-probe.py`; no sign flips were applied. Correcting signs alone would not repair missing graded release or resting activity. [Primary photoreceptor physiology study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4801898/).

### 2. Forward-walking cells are inhibited

All four DNg97/DNg100 cells are silent in all three baseline windows. Their group-mean endpoint voltages range from **−126.1 to −103.5 mV**, versus the model threshold of **−45 mV**. These are values of the unconstrained current-based LIF model, not physiological membrane-potential measurements.

The weighted presynaptic-rate ranking identifies CB0890, CB0677, and DNge054 among strong inhibitory contributors. In paired perturbations:

- Removing sugar raises the group-mean endpoint voltage in every seed, but only one trial develops sparse walking spikes: **1.875 Hz**, versus **0 Hz** in the other two.
- Silencing the two incoming CB0890 cells depolarizes the walking population but produces **no walking spikes** in any trial.
- Silencing the incoming CB0890, CB0677, and DNge054 cells together produces **0.9375, 0.3125, and 1.25 Hz**, respectively. This partially releases inhibition but does not establish sustained walking drive.
- Removing odor or the body-input proxy alone does not produce walking spikes in these windows.

Silencing is an artificial −1,000 mV constant drive applied to the named source cells in fresh diagnostic brains. Their measured spikes are checked in the raw report. These experiments establish effects in this model; they do not establish that these cells physiologically prevent walking in a real fly.

The group rates above use an 800 ms arithmetic mean. The habitat uses a 100 ms exponentially smoothed rate, so an occasional spike can cause a short pulse even when the longer-window mean is below its 2 Hz dead zone. Disabling sugar therefore cannot be promised to produce either sustained walking or exact stillness.

### 3. Wing cells are quiet, but can fire

All 25 DNg02 wing-power cells are silent in the baseline conditions. Their endpoint voltages are near or below the model resting potential, rather than crossing threshold. In the diagnostic positive control, directly stimulating the four walking and 25 wing cells at 80 Hz yields walking means of **66.6–71.3 Hz** and wing means of **65.5–71.6 Hz**. The unchanged decoder produces full forward and wing commands from those means. Separate body tests verify that these commands produce translation and lift.

This positive control verifies the runtime, readout, and decoder path. It is artificial stimulation, not sensory-evoked behavior. No forced activation was added to the live habitat.

## Live verification

With rendered eyes and body feedback enabled, Fly 001's inspector showed approximately **15.38 Hz** in the photoreceptors, **0.037 Hz** in L1/L2, and **0 Hz** across T4/T5. All four forward cells and the inspected 13 left-wing cells were silent. Removing sugar in the running habitat eliminated proboscis activity and reduced the CB0890 contribution; DNg100 became less hyperpolarized but did not begin sustained walking in the observed interval. Sugar was restored after the check.

The live inspector exposes these observations per fly, including the measurement window and individual cell voltages. Its input rankings are signed weight × presynaptic spikes/s, not measured current and not causal proof. Delays and refractory losses are excluded from that ranking; the separate net synaptic-drive readout comes directly from WASM.

## What the diagnosis supports

The next model repair should address graded visual transmission, retinal neurotransmitter signs, and cell-type-specific resting activity and response calibration, then validate ON/OFF responses and motion selectivity under controlled stimuli. The walking pathway also needs calibrated sensory drive and excitation/inhibition balance; the present body-sense boundary proxy has not solved that problem. None of these experiments supports inferring hunger, motivation, happiness, or a natural decision to remain still.

The brain-only scope can be retained while improving these components. No ventral nerve cord was added, and no graph, neural parameter, or actuator gain was changed by this diagnostic work.
