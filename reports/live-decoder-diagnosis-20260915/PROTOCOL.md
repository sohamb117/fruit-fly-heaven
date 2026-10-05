# Live checkpoint comparison and decoder diagnosis

The user requested a comparison with the live checkpoint and a diagnosis of CNS signal noise versus poor adaptation. This is a local diagnostic experiment; no candidate, result, lease, or checkpoint is written to the coordinator.

## Frozen comparison

`checkpoint.json`, `status.json`, and `fetch.json` preserve the read-only generation-10 snapshot, retrieval times, URLs, response headers and hashes. The live model/configuration match the previous 672-coefficient experiment. Compare exact live weights on seeds 2490888, 2590888 and 2690888 with the already completed fitted-v2 trials under that identical frozen model, scorer, seed and warm-up. Do not relabel a new generation as the evaluated snapshot.

The three seeds are matched body reset variants, not independent neural datasets. The sample is small. Five-second evaluations start airborne and provide no takeoff or landing evidence.

## Causal controls

Begin with seed 2590888, the fitted trial that loses height, contacts the floor and subsequently tumbles. Replicate informative controls on the other matched seeds as needed. Keep the complete original 0.5 s warm-up unchanged. Interventions begin only at scored release.

- **Identity:** instrument the fitted decoder without changing any commands. Require the known score, failure time and flight counters to reproduce before interpreting interventions.
- **Smooth power input:** apply a causal 20 ms exponential filter to the 24 power motor-neuron excitation channels. Preserve the 24 steering channels, phase features, weights and all native physics. This tests sensitivity to fast fluctuations in the power input. Added delay and using an unrefitted filter limit interpretation of a negative result.
- **Calibrated constant power:** replace just the two power outputs with the pre-existing restrained-body trim values, retaining live BANC-driven learned steering. This changes both mean and variation, so success alone would not distinguish bias from noise.
- **Optional matched mean power:** hold each side at its measured baseline precontact mean to remove fluctuations without using the teacher trim mean. Record the exact source and averaging interval.
- **Optional shadow power feedback:** replace only power with the earlier full-state PD teacher's power output; keep learned steering. This is an explicitly privileged diagnostic, not a biological controller or a deployable decoder checkpoint.

The root harness saves every 1 ms excitation/first-phase command and every 2 ms scored physical observation, including loaded environment contacts. All modes also calculate the same read-only PD shadow for desired-command comparison. The identity run checks that this instrumentation does not materially change the original result. No scored root writes or externally applied forces are allowed.

## Offline and architectural audit

Use only the two original training trajectories to construct signal summaries or a phase-only command baseline; retain the third for validation. Report actual filtered excitation variability separately from raw spikes. Compare offline imitation error with closed-loop results instead of treating one as proof of the other.

Inspect the active frozen sensory configuration, including whether directional haltere currents or vision are enabled. Source-code presence alone is not evidence that a sensory route is active. Distinguish missing feedback, decoder expressiveness, data coverage, gain bias and fast fluctuations. These experiments cannot identify real fly physiology or rule out all CNS problems.

## Same-input capacity question

Mechanical power interventions cannot distinguish absent corrective information from an inadequate readout class. The final analysis therefore compares readouts on identical saved raw motor-neuron streams, using the four completed control trajectories. It introduces no additional physical simulation.

Use only even-ms command records, paired with the exact contemporaneous pre-command body observation. Start at 50 ms so all 0/1/4/10/20/50 ms histories are available. Primary fitting, inner model selection and outer scoring exclude every sample at or after the first recorded contact. Hold out one whole control trajectory at a time, with inner trajectory-held-out regularization selection and training-only preprocessing. All four trajectories share one reset seed; they are not independent seeds or animals.

Compare the original class, bias and sign/bounds ablations, masked causal linear/nonlinear histories, and an independently labeled body-state benchmark. Report improvement over train-only nominal power/phase baselines. Separately decompose held-out nominal offsets and temporal corrections for evaluation; do not use test targets to change model predictions. Negative finite-probe results cannot establish absence of information. Shadow targets represent one state controller and are not unique flight-valid controls.

## Execution

Use unchanged pinned browser WASM in local Node processes, with at most two concurrent single-fly simulations. Original report files and production source remain unchanged. Record any launch failures separately; an unstarted or incomplete trial is not a behavioral result.
