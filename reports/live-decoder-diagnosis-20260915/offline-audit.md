# Offline neural signal and decoder audit

The existing evidence favors a mismatch between the learned power mapping and the signals/state encountered in closed loop. It does not establish that the CNS is intrinsically too noisy. The fitted decoder substantially improves nominal teacher-command prediction, but a train-only phase baseline predicts these stable hover commands better, and the failed student's mean decoded power decreases before contact while the diagnostic state controller asks for more power.

This audit executes no neural or physics steps and does not change weights. The only newly fitted model is the explicitly non-neural, 20-coefficient diagnostic phase baseline, trained on seeds 190888 and 290888. Seed 490888 is held out. Original decoder samples, original feature digests, source identities, and exact power-mean accounting are checked.

| Held-out teacher inputs, first recorded phase each millisecond | Power RMSE | Steering RMSE | Combined RMSE |
|---|---:|---:|---:|
| Initial decoder | 0.058909 | 0.005897 | 0.029894 |
| Live generation 10 | 0.058841 | 0.013514 | 0.031663 |
| Local fitted v2 | 0.010869 | 0.001832 | 0.005661 |
| Fitted v2, causal 20 ms power smoothing | 0.009127 | 0.001832 | 0.004831 |
| Fitted v2, power inputs held at training means | 0.002276 | 0.001832 | 0.001952 |
| Constant power plus steering phase baseline, trained without neural input | 0.002266 | 0.001866 | 0.001974 |

The phase baseline uses two constant powers and six steering axes with constant/sine/cosine coefficients. Its coefficients respect the diagnostic command bounds; validation targets are not used to fit it. It shows that nominal controls explain most of these narrow hover targets. It does **not** show that constant drive is a valid replacement for neural flight gating, or that neural information would be unhelpful in richer behaviors. Offline controls reuse successful teacher-conditioned inputs and cannot establish closed-loop performance. Combined RMSE mixes normalized power and steering coordinates; the separate columns are more interpretable.

Power signals already pass through event kernels with 6.2 ms rise and 82 ms decay. Across the three teachers, DLM excitation has mean 0.930 and per-unit temporal SD RMS 0.037; DVM has mean 0.698 and SD RMS 0.141. Power-unit lag-one-millisecond correlations are approximately 0.996–0.997. Steering excitation is faster: lag-one-millisecond correlation 0.902, lag-ten-millisecond 0.091. Additional 20 ms filtering therefore tests additional smoothing of filtered motor excitation, not removal of raw spike noise. Three DVM units spend more than 93% of their samples above 0.99 excitation; this is not evidence that all MN firing or all decoder power is saturated.

The best training-selected single power feature correlates only about 0.062 with the teacher's power correction remaining after constant power. Corresponding held-out correlations are similarly small. Steering residual correlations are also modest, with training magnitudes at most 0.155 among the reported strongest features. These are descriptive marginal correlations, not information-theoretic or causal limits; correlated multivariate features and unobserved behaviors remain possible. The original fit's 108-column steering designs have numerical rank 90 per side, while the 12-column power designs have rank 12. Numerical rank does not establish biological parameter identifiability.

The matched 0.1–1.0 s comparison uses 901 student samples and 1,802 samples from the two training teachers, before any student ground contact:

| Quantity | Left power | Right power |
|---|---:|---:|
| Fitted decoder on teacher inputs, mean | 0.814679 | 0.814158 |
| Fitted decoder on student inputs, mean | 0.806742 | 0.805643 |
| Mean shift | −0.007937 | −0.008514 |
| DLM contribution to shift | −0.008361 | −0.008018 |
| DVM contribution to shift | +0.000424 | −0.000496 |
| Student shadow state-controller request, mean | 0.823705 | 0.822302 |
| Teacher-input decoded power SD | 0.008206 | 0.013289 |
| Student-input decoded power SD | 0.009898 | 0.011823 |

The mean DLM input decreases from [0.923283, 0.932403] to [0.909557, 0.918243]. Because this decoder's power outputs are an unclipped linear combination in the observed range, the per-MN mean shifts account for the command changes to less than 2×10⁻¹⁵. The student does not exhibit a conspicuous increase in power variability in this window; its mean drive falls, while its shadow recovery request is about 0.017 above the actual drive. Actual-versus-shadow power correlation is negative on each side. The first recorded ground contact is 1.426 s. Different seeds and controllers mean the input shift cannot be assigned separately to initial neural randomness, feedback, or body-state divergence from this comparison alone. The shadow powers are two components of a jointly allocated 20-coordinate controller, so executing only those powers need not reproduce the full state controller.

The current decoder has 24 nonnegative power weights, no additive power intercept, and no power history beyond the already filtered current excitation. Steering has 648 ipsilateral coefficients using 0/1/4 ms history and fixed wing-phase bases. A model without an intercept must synthesize nominal power from the active MN levels, making its baseline susceptible to changes in those levels. The fitted solution assigns zero weight to seven DVM inputs and relatively high weights to several nearly constant DVM inputs; this is consistent with fitting nominal drive, without proving that this is the solver's unique or causal strategy.

The enabled sensory route has a material feedback limitation. This pinned config has vision disabled, no `haltereFeedback`, and no `neuralInputSequence`; the directional haltere implementation is not active. The active rotation prior supplies an organ/side scalar using angular-speed magnitude, discarding rotation sign and axis. All 1,016 active self-motion afferents use annotated transducers, with 50 excluded; none use the broader self-motion fallback. The 579 antenna afferents use the broad fallback, which includes unsigned total speed and tilt magnitude. Tegula aerodynamic load feedback is enabled. No direct height or signed vertical-speed afferent is encoded. Odor, contact, load, and joint signals can still contain indirect state information. Pure fixtures confirm rotation-sign/axis ambiguity when other inputs are held fixed; they do not assert that complete sensory vectors are identical on different moving trajectories.

The root agent's causal controls are the appropriate next discriminator: compare the same failing seed with additional power smoothing, fixed calibrated trim power, and a small multiplicative power gain while retaining learned steering and identical warmup. Constant trim remains a diagnostic intervention. If needed, compare signed state-feedback power with the full state controller while acknowledging coupled steering allocation. Successful candidates should be repeated on independent seeds. Longer term, perturbation/recovery teacher data and an explicit choice of enabled sensory transduction are more informative than simply extending training on steady hover targets.

Reproduce in a fresh report destination, or remove only newly generated result files intentionally before rerunning; scripts use exclusive creation and will not overwrite results:

```sh
node reports/live-decoder-diagnosis-20260915/signal-audit.mjs
node reports/live-decoder-diagnosis-20260915/distribution-shift.mjs
```

Results: [signal-audit.json](signal-audit.json), [distribution-shift.json](distribution-shift.json). Checks include 15,000 original feature digests, 1,080,000 sample-versus-feature values, 14,640 recorded identity replay values, and exact power mean decomposition. No original provenance-pinned files or production files were modified by this audit.

Source anchors in the pinned bundle: [environment.js:43](../../dist/training-lease-client/fruit-fly-training-client-1f3b0935af59/training/environment.js#L43), [environment.js:332](../../dist/training-lease-client/fruit-fly-training-client-1f3b0935af59/training/environment.js#L332), [sensory-encoder.js:28](../../dist/training-lease-client/fruit-fly-training-client-1f3b0935af59/sensory-encoder.js#L28), [sensory-encoder.js:99](../../dist/training-lease-client/fruit-fly-training-client-1f3b0935af59/sensory-encoder.js#L99), [banc-ground-sense.js:21](../../dist/training-lease-client/fruit-fly-training-client-1f3b0935af59/banc-ground-sense.js#L21), [motor-decoder.js:182](../../dist/training-lease-client/fruit-fly-training-client-1f3b0935af59/motor-decoder.js#L182).
