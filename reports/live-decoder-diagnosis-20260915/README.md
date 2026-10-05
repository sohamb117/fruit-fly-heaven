# Live checkpoint and CNS-to-decoder diagnosis

The question is whether corrective information is missing from the simulated CNS motor output, or present but inaccessible to the current adapter structure. Changing power gain alone does not distinguish those explanations. This report separates the live-checkpoint comparison, mechanical controls, and offline information/capacity probes.

## Live checkpoint comparison

The read-only snapshot from `https://flytrain.morisoba.moe/api/training/checkpoint` is generation 10, with the same frozen configuration, model, WASM and 672-parameter contract as the prior local fit. Retrieval time and response hashes are in [fetch.json](fetch.json). Exact live parameters were evaluated on the same three seeds as the stored fitted-v2 trials; no clipping, fitting, rescaling, or control override was applied to these generation-10 evaluations.

| Seed | Generation 10 best qualifying flight | Fitted-v2 best qualifying flight |
|---:|---:|---:|
| 2490888 | 56 ms | 740 ms |
| 2590888 | 312 ms | 412 ms |
| 2690888 | 344 ms | 1,016 ms |
| Mean | 237 ms | 723 ms |

Fitted v2 wins score and best uninterrupted qualifying flight on all three matched cases. Neither checkpoint passes the maintained-flight goal on any of them. Two fitted trials last five seconds with intermittent qualification; every generation-10 trial terminates early. These are three deterministic body-reset variants, not a statistical guarantee of generalization. [Validated comparison](comparison.json), [full table](comparison.md).

## Mechanical controls: useful localization, not an information verdict

All controls use seed 2590888 and the original fitted-v2 vector. Every control retains the identical original 0.5 s live warm-up. Interventions begin only at scored release. Learned steering commands remain unchanged by the intervention, although subsequent neural activity and body trajectories can diverge through feedback. Physics, contacts, actuator limits and all native joints remain fixed.

| Control | Best qualifying flight | Duration | First environment contact | Goal |
|---|---:|---:|---:|---|
| Instrumented unchanged decoder | 0.412 s | 1.830 s | 1.426 s | Failed |
| Additional 20 ms filter on power-MN excitation | 0.042 s | 0.746 s | 0.314 s | Failed |
| Two power outputs fixed at prior calibrated trim | 4.952 s | 5.000 s | None | Passed |
| Multiply both decoded powers by 1.02 | 2.330 s | 5.000 s | None | Not met |

The identity run reproduced the original score, failure time, clocks, parameters, spike count and full initial/final observations exactly. [Parity evidence](identity-parity.json).

Fixed power is an explicitly non-neural intervention on those two outputs, not a proposed permanent controller. It held the values at `[0.8138653348406052, 0.8125142811723722]`. This demonstrates that the learned steering plus unchanged CNS/physics can maintain this particular flight with adequate power; it does not prove the CNS carries the information needed to regulate that power.

The 2% increase preserved changing neural-derived power. It prevented contact throughout the five seconds and yielded 4.398 s of total qualifying flight, but its final uninterrupted qualifying bout was only 0.026 s, so the original goal remained unmet. The fly ended upright at approximately 5.21 cm height. This is a material physical improvement, not a passed goal or a validated deployable checkpoint. Applying the increase only after release also differs from changing actual parameter values before warm-up.

Extra smoothing reduced command step variation but caused earlier descent and foot contact; large rotation followed contact. It also changed bilateral balance and the subsequent neural trajectory. It rejects this particular filter, not the general hypothesis that some signal variation is harmful. [Smoothing postmortem](smoothing-postmortem.md), [all control traces summarized](control-analysis.json).

## Signal and feedback observations

In the matched 0.1–1.0 s interval, the original fitted decoder produces mean powers of approximately `[0.81468, 0.81416]` on the two training-teacher input streams, but `[0.80674, 0.80564]` on the failed student's stream. Lower DLM excitation accounts for almost all of the shift. Power standard deviation is comparable between those streams; there is no conspicuous increase in variability. The state-aware shadow controller requests about `[0.82370, 0.82230]` on the descending student's states. Those shadow powers are part of a jointly allocated controller, not proven optimal isolated power targets. [Exact accounting](distribution-shift.json).

On the original narrow hover demonstrations, a train-only constant-power/phase-steering model predicts held-out teacher commands better than the fitted neural decoder. That reveals how much of the original imitation gain came from nominal operating conditions. It does not establish that constant power is sufficient for varied behavior or justify disconnecting neural flight gating. [Offline signal audit](offline-audit.md).

The active configuration has vision disabled and no directional haltere-current option enabled. Its rotation encoder supplies angular-speed magnitude and loses sign/axis. No direct height or signed vertical-speed afferent is encoded. Tegula load, joint, odor and contact feedback may still carry indirect state information. These are concrete input limitations, not proof that all corrective information is absent from the recurrent network or selected motor neurons.

## Same-input capacity comparison

The [capacity probe](decoder-capacity.json) holds each recorded raw MN input stream fixed while fitting different readout classes. Primary fitting, inner validation and outer scoring use only samples before the first recorded contact. Each outer fold holds out one complete control trajectory; all four trajectories share seed 2590888. Regularization and preprocessing use training trajectories only. Labels are the read-only shadow controller's contemporaneous corrective commands, not the control actually applied by an intervention.

Models include the exact bounded current class, an added-bias variant, causal 50 ms masked linear and nonlinear histories, a train-only constant-power/phase baseline, and separately privileged body-state/history benchmarks. The latter establish predictability from physical state. Motor-only models receive neither body state, elapsed time, intervention identity nor contact labels. Richer probes are diagnostic function classes, not deployable or biologically validated muscle models.

After removing each held-out trajectory's nominal constant/phase component for evaluation, the improvement in predicting changing power corrections is small and inconsistent:

| Readout | Power temporal skill beyond nominal baseline | Steering temporal skill |
|---|---:|---:|
| Current class, refitted | −160.7% | −3.1% |
| Current class plus bias | −0.1% | −1.0% |
| Current features plus bias, coefficient bounds removed | −12.1% | −1.0% |
| Same masked MNs, 50 ms linear histories | +4.0% | −5.0% |
| Same masked MNs, 50 ms nonlinear histories | +5.8% | −6.6% |
| Privileged body history, linear | +99.1% | +97.8% |

Skill is `1 − prediction SSE / nominal-baseline SSE`, after evaluation-only separation of nominal offsets and changing corrections. Zero means no improvement on the nominal baseline; negative means worse. Held-out targets are not used to fit or alter predictions. Model selection uses only inner training folds. The richer MN models' earlier 22–23% aggregate power-error improvement includes better prediction of mean offsets; it must not be described as 22–23% recovery of changing corrections.

Linear-history power skill is −24%, +15%, +14%, and −249% on the identity, smoothing, gain and trim trajectories respectively. The last trajectory has little remaining temporal target variation. This is limited, heterogeneous signal-prediction evidence, not a reliable new controller. Removing bounds without adding history gives no aggregate improvement, so sign/bounds relaxation alone does not explain the small history advantage.

**There is no clean binary CNS-versus-adapter verdict from these data.** The tests establish a calibration vulnerability in the existing adapter. They also fail to show that modestly richer readouts reliably recover the teacher's corrections from the selected MNs. Together with the active sensory limitations, that supports investigating what corrective information reaches the motor outputs, instead of assuming that increasing parameter count alone solves the problem. It does not prove the information is absent: longer recurrent histories, other neural populations, different data, or other valid flight commands remain untested.

The privileged body benchmark is expected to do well because these labels come from a body-state controller. Its role is to verify learnability and timing of the chosen targets. Those targets are one controller's commands, not the unique commands capable of flight. The power-gain and constant-power controls therefore cannot be used to declare the entire CNS adequate or inadequate, and failed imitation cannot prove that no neural controller would fly.

[Readable capacity results](decoder-capacity.md) · [Primary folds](decoder-capacity.json) · [Sign/bounds ablation](capacity-sign-ablation.json) · [Temporal-versus-offset decomposition](capacity-dynamics.json).

## Scope

All physical trials start airborne; none establishes takeoff, landing, feeding or food localization. Main evaluations run unchanged pinned browser WASM locally in Node. No production source, live checkpoint, lease or coordinator result was changed. The four information-probe trajectories are correlated interventions on one reset seed, and failed finite probes cannot prove information-theoretic absence of a signal. Even a successful offline richer readout would still require closed-loop validation.
