# Same-input decoder capacity comparison

**The current adapter is not extracting all the power-related information available in these MN traces, but the additional information demonstrated here is modest and inconsistent. This experiment does not establish whether the remaining bottleneck is CNS/sensory information or a still-insufficient decoder class.** Steering remains unresolved.

These are offline predictions of the same diagnostic corrective commands from the same recorded inputs, with whole trajectories held out. The labels are one diagnostic controller's commands, not the unique commands capable of valid flight. No tested model was deployed or run in closed loop. Four control trajectories share seed 2590888: identity, extra power smoothing, 1.02 power gain, and fixed trim power. Their respective pre-contact sample counts are 688, 132, 2,475, and 2,475. All fitting and regularization selection use only pre-first-contact samples; post-contact predictions are separate diagnostics.

Every model sees the same raw per-MN excitation samples. Power uses its 12 ipsilateral DLM/DVM MNs; steering uses its 12 ipsilateral steering MNs. No time, seed, intervention identity, contact, body state, or controller output enters an MN probe. Current features are reproduced from the original contract. History probes use causal lags 0/1/4/10/20/50 ms. The nonlinear probe adds 64 fixed tanh features, without projecting phase into the nonlinear map. Only steering receives the original constant/sine/cosine wing-phase bases.

| Model | Held-out power RMSE | Power improvement beyond train-only nominal baseline | Within-trajectory power correction improvement |
|---|---:|---:|---:|
| Constant power / phase steering baseline | 0.014928 | 0% | 0% |
| Fixed local v2 decoder | 0.034413 | −431.5% | −264.1% |
| Refit exact bounded current class | 0.025953 | −202.3% | −160.7% |
| Current class plus constant/phase bias | 0.014931 | −0.05% | −0.06% |
| Current features plus bias, coefficient bounds removed | 0.015005 | −1.0% | −12.1% |
| Masked MN linear history | 0.013119 | **+22.8%** | **+4.0%** |
| Masked MN nonlinear history | 0.013187 | **+22.0%** | **+5.8%** |
| Separate body-state/history linear benchmark | 0.001315 | +99.2% | +99.1% |

“Improvement” is reduction in held-out squared error, rather than reduction in RMSE. The primary denominator is error from the phase-only baseline fitted exclusively on training trajectories. All folds weight each training trajectory equally. Outer validation holds out one complete control trajectory; inner trajectory validation selects ridge regularization from the fixed grid. All preprocessing uses training data only.

The correction column additionally removes each held-out trajectory's own constant-power or phase-steering component from **both** its predictions and targets for an evaluation-only error decomposition. It never changes a prediction, coefficient, regularization choice, or primary result. This matters because most of the history models' primary improvement comes from predicting the nominal offset between trajectories, rather than correctly following every time-varying correction.

The temporal result varies considerably:

| Held-out trajectory | Linear-history primary power improvement | Linear-history within-trajectory improvement |
|---|---:|---:|
| Identity | −107.2% | −24.2% |
| Extra smoothing | +31.0% | +14.9% |
| 1.02 power gain | +41.7% | +14.1% |
| Fixed trim | +30.2% | −249.1% |

Trim has particularly small target fluctuations, making extra prediction variation costly. Its improved primary score comes mainly from predicting its mean power offset. The richer models therefore provide a limited positive signal, not a robust control solution.

The matched coefficient-bound ablation uses exactly the current-plus-bias features, scaling, objective, folds, and regularization selection, removing only MN coefficient bounds. It does not improve aggregate power prediction. The history result consequently is not explained simply by permitting negative power coefficients, though it still cannot isolate every modeling choice or establish that a particular history length is optimal.

For steering, linear/nonlinear MN history probes have −7.2% / −9.5% primary improvement and −5.0% / −6.6% within-trajectory improvement. They do not establish additional useful steering decoding on these held-out trajectories. Fixed v2 does better on the gain and trim trajectories individually, but no positive aggregate steering result follows. Failure of these finite probes does not establish that steering information is absent from the CNS or from other neurons/history lengths.

The separate body benchmark uses canonical rotation, raw and causally filtered angular velocity, height error, signed vertical speed, and up, with causal history. It uses the same ridge/random-feature procedure. Its strong performance verifies that corrective targets are learnable with the relevant privileged state. It is not a biologically validated MN decoder. A pure replay of the original state-controller mathematics reproduces **all 125,760 stored shadow coefficients exactly** from the chosen pre-command body observations, confirming that the body benchmark does not accidentally use future observations.

The dataset limits the conclusion. All four trajectories have one seed and different actuator interventions; hidden actuation or body-state differences can make the MN-to-correction relationship ambiguous. The smooth trajectory contributes only 132 eligible samples. Correlated samples do not justify thousands of independent observations or inferential confidence intervals. Together with the separate sensory-route audit, the large gap to the body benchmark makes an input/feedback bottleneck worth investigating; it does not establish binary causation, information absence, or that a bigger decoder cannot work. A stronger distinction would require independent-seed perturbation/recovery trajectories and a broader tested decoder family. None of the unrestricted readout probes is deployed or biophysically validated, and offline prediction does not establish closed-loop success.

Artifacts:

- [Primary results and all fold selections](decoder-capacity.json), [predictions](decoder-capacity-predictions.json), [script](decoder-capacity.py).
- [Matched bounds ablation](capacity-sign-ablation.json), [ablation script](capacity-sign-ablation.py).
- [Offset versus dynamic correction decomposition](capacity-dynamics.json).
- [Exact label/body alignment verification](capacity-alignment.json), [alignment script](capacity-alignment.mjs).

The primary probe checked 100,608 original decoder output values and all bounded-fit KKT conditions (maximum residual 3.4×10⁻¹⁶). The unbounded ablation checked linear-system stationarity to 1.4×10⁻¹⁵. Reproduction uses NumPy plus the standard library and the pinned original JavaScript controller; no packages are installed. Output files use exclusive creation. No neural/physics steps, cloud writes, production modifications, or deployable checkpoints were produced.
