# Affine wing transfer: bounded read-only review

The proposed transfer is a reasonable **frozen-motor diagnostic**, provided its calibration and replay gates pass. It can test whether an unsuitable mechanical operating point contributes to the current trajectory. It does not establish physiological muscle baselines, dynamic flight stability, or a trained takeoff/landing policy. No native/neural runs or live source edits were performed for this review.

## Define the transfer precisely

For each side, let `B` be the existing 6 × 12 steering basis. Its first three rows alter mean wing angles; its last three alter waveform amplitudes. The existing 24 steering gains comprise **two gains per type**, tied across sides. Thus `B*G` is shorthand for row-block weighting: bias gains multiply rows 0–2, amplitude gains rows 3–5. The proposed six coefficients are

```
r_bias = B_bias * T + B_bias * diag(g_bias) * (F - F0)
r_amp  = B_amp  * T + B_amp  * diag(g_amp)  * (F - F0)
```

`T` and `F0` each need twelve explicit values **per side**. The existing `steering_force_reference` accepts only twelve shared values and cannot express this. An opt-in, owned, validated profile is appropriate; reject simultaneous legacy and affine reference modes. Keep `T` outside the learned gains, retain signed deviations, and do not clamp `F-F0` as though it were a nonnegative force. At `F=F0`, the coefficients are `B*T` independently of steering gains.

Likewise, do **not** introduce a runtime `[0,1]` clamp on `T + G*(F-F0)`: these are force-equivalent basis coordinates, not actual native muscle forces, and that clamp would change the affine hypothesis. Record separate bias/amplitude coordinate extrema at each causal 1 ms force handoff, including startup. If claiming interpolation within the measured `[0,1]` trim domain, predeclare a gate that rejects/reports candidates outside it. Otherwise label the comparison as extrapolation. Range containment alone does not establish local-Jacobian accuracy; a fitted T near a bound can leave the domain even with the current small gains.

At zero steering input with async wing power still present, the coefficients are generally `B*T - B_G*F0`, not zero. That offset is a declared numerical transfer assumption. It must not be called neural output or measured biological resting tension.

## Power normalization and clamps

The proposed side scale `s = trimPower / (rawPowerMean * retainedPowerGain)` is suitable for the same-27-parameter diagnostic. Apply it **before** the requested-power clamp:

```
requested = clamp(rawPower * retainedPowerGain * s, 0, 1)
actualPower = min(requested, existingDeploymentRamp)
```

The saved means are 0.7666709813 / 0.8059740725, with power gain 1.9999996389. Therefore `s_left = 0.6521703605 * trimPower` and `s_right = 0.6203674626 * trimPower`. Existing mean table powers are already 0.9999098255 / 0.9998983105. Scaling after that saturation would not map the raw reference means to the trim. Reject zero/nonfinite calibration denominators rather than introducing a power floor. This is a side-specific normalization assumption: it deliberately maps unequal recorded mean drives to a common reference power.

Zero raw async power still produces zero requested and actual power, including after deployment. The steering residual is multiplied by actual power and vanishes. **Zero power does not guarantee zero native actuator torque**: the existing target/deployment/neutral-restoration servo remains. Also, the mean of a clamped, deployment-limited time series need not equal the value at its mean. Measure achieved power separately.

Retain and audit all current nonlinearities: requested-power `[0,1]` clamp; deployment threshold `requested > 0.01`; deployment ramp above 0.85; per-axis combined shape residual `[-0.15,0.15]` before power multiplication; final joint-range clamps; and actuator error clamps `[-1,1]`. Scaling power moves the raw-drive deployment threshold as well as amplitude. Calibrate through these exact operations, and report shape/target/servo clipping and local fit residuals. A clipped fit can conceal lost sensitivity.

## Conditioning and the current 27 coordinates

The embedded basis has rank 6, with singular values `[0.1129443, 0.0963540, 0.0646772, 0.0380374, 0.0215019, 0.00633734]` and raw condition 17.822. With current bias/amplitude gains, rank remains 6 and raw condition is 15.567. These are static coefficient-space results with differently scaled row meanings, **not** physical-wrench conditioning. Twelve forces per side still have a six-dimensional nullspace before native mechanics; constrained attainable directions can be poorer. Record the measured native wrench Jacobian, its declared normalization/conditioning, bound-active controls, and measured residual at the chosen trim. Regularization selects a numerical solution; it does not identify anatomical forces.

The current power gain is at its permitted upper bound (~2). Deployment scale is 0.6934201 and frequency scale 0.9634913. All steering gains are 0.0500000–0.0697336; eight of 24 sit exactly at their configured lower log bounds. Allowed steering gains are approximately `[0.05,2]`, strictly positive: they cannot reverse a basis direction or reach zero exactly. The global power scale has no upward search headroom in the frozen coordinate system. A later reset to relative gain 1 would require an explicitly versioned coordinate/bounds change; do not silently do it in this A/B.

At `F=F0`, steering-gain sensitivity is zero. Away from it, log-gain sensitivity is proportional to `gain * (F-F0)`. Both iii1 forces are identically zero throughout the 100–500 ms window, so their two shared gains are unobservable in this window; left b1 and right iii3 are also individually silent. A static trim does not make all 27 parameters effective or independently identifiable. These facts do not prove that those neurons are physiologically abnormal.

## Gates before an interpretable A/B

1. **Measured calibration:** use the bundle's actual full body, 227.204212186366 Hz, 50 µs native steps, 200 µs wing updates, stated nonwing posture, exact basis and clamps. Freeze the measured `T`, common trim power, profile, bounds, tolerances and source hashes before release. Calibration input is not a completed calibration result.
2. **Scope of “trim”:** the current staged plan constrains vertical fluid force and three COM fluid moments only. Horizontal force is logged but unconstrained; hinge/restraint reactions are excluded. Call this a four-target aerodynamic trim. It does not by itself establish full hover equilibrium. Frozen nonwing coordinates and a restrained root also cannot demonstrate release stability or reject momentum-driven transients.
3. **Exact baseline gate:** replay the recorded 805 MN rates and 48 wing event streams with their original timestamps and causal 1 ms muscle-force handoff, restored native integration state, muscle/event state, body clock, wing phase/deployment and internal state. Require baseline qpos/qvel and muscle/power readbacks to match before comparing variants. The F0 snapshots are not a replacement for that timed input stream. With live BANC feedback reconnected, this is no longer a frozen-motor comparison.
4. **One declared intervention:** hold the same 27 coordinates, physics, initial state, input stream, food schedule, duration and scoring across arms; change only the declared affine profile and its power units. This estimates the combined trim-plus-power-remapping effect. It cannot separate those components without further predeclared arms. Frozen MN input does not automatically mean every body-dependent muscle force is frozen; verify the wing-force stream or name the input boundary precisely.
5. **Legacy equality:** when the profile is absent, retain original iteration order and arithmetic. An algebraically “equivalent” rewrite or implicit identity profile is not a byte-equality guarantee. Compare representative outputs and the exact baseline replay. The currently staged branch preserves the absent-profile expressions and its pure equality test passes.

F0 is explicitly the equal-weight mean of 201 **post-body snapshots** at 100, 102, …, 500 ms. It is a chosen trajectory reference, not an exact continuous-time average, a 1 ms held-command average, a hover baseline, or resting physiology. The source reducer documents this correctly. Keep both successes and failed trim acceptance visible; do not relabel an unmet target as hover.

## Evidence checked

Read `web/flybody-wings.js`, `web/flybody-physics.js`, `web/training/flight-parameters.js`, the legacy bundle, capture/reducer, and available affine staging modules. Read-only JSON assertions independently reproduced F0 and raw-power means, verified the source/bundle hashes, and confirmed the entire source-capture physics digest and motor-event array equal the unobserved legacy reference. NumPy SVD was applied only to the saved 6 × 12 matrix; no native/neural simulator was initialized.

Independently ran the final `node --test reports/flight-affine-transfer-staging/transfer.test.mjs`: **8/8 pass**, using the exact embedded bundle metadata. They cover 2,048-step absent-profile equality, F0-to-trim equality, zero power both initially and after deployment, signed bilateral gain effects, atomic profile rejection/ownership, explicit pre-clamp power scaling, and legacy-reference parity/exclusivity. These are pure wing-transform tests, not native dynamics validation.

Saved-record inspection of `flight-affine-replay-staging/baseline-result.json` reports the exact native baseline gate passed all 250 steps; this review did not rerun it. `flight-affine-calibration-staging/captured-plan/result.json` has `completed:false`, `trimAccepted:false`; its failure record reports one contact violating the contact-free calibration invariant. Thus an accepted T remains an unresolved blocker. Do not continue the same contact-free protocol by silently removing its invariant or hiding the collision.

| Artifact | SHA-256 |
|---|---|
| `flight-haltere-live/legacy/bundle.json` | `4e76394a9b8fd6106dd8f86f1019fe484fd6b15f61d8cdc75a22d1e75bda0140` |
| Bundle embedded XML | `697aadc8a4df253537f1a03a786ece20fa57f22883d55ea24320a59c15b14583` |
| Bundle embedded metadata | `8d389067f771d9cad002dda4f806a352fd5806ba94e30125db2f96bce4a8a1ae` |
| `web/flybody-wings.js` | `10a44cb2ed11835ed3a151081586095e6bd24527f0c20926c9a006783fec7084` |
| `web/training/flight-parameters.js` | `6392fcab849eef6f2fdfc20d5eab0cd246b63705caedb114d7a09caebc1f385d` |
| `flight-affine-source-capture/calibration-input.json` | `2165a1f19f6ff22cadd9070b01cd937719b363448d590dfc661557075f72f467` |
| Source capture JSON | `ee736149551da39bd824e6988a7930badae37c026eb41d9ab8add36e5f80805e` |
| Source and legacy physics digest | `c8abcff173e74f7a118005b2349774b587492ac0c4447bb1b303f8af0806b9c9` |
| Reviewed staged `flybody-wings.mjs` | `a8ed8d5836cb41fc5f6f6a99263d0f57227a759e1aab7eb223e3eefc7dea97de` |
| Reviewed staged `install.mjs` | `11f50a150edad4971b4e594ace522ce47f542c12f0f2f52c928bdf4b12dca06c` |
| Reviewed staged `transfer.test.mjs` | `75e36dc44c91563745ad7f4ed62b2eed49a02ba99fece6f52e8f298cd45f122b` |

Paths in this table are relative to `reports/` unless prefixed `web/`; reviewed staged files are under `flight-affine-transfer-staging/`. Working-tree model files are not interchangeable with the bundle's embedded assets. No accepted T was available at review time.
