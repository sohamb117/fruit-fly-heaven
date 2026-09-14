The frozen virtual-haltere observer preserves three independent components of a **constant root-local angular velocity over a full cycle** when both sides have nonzero drive. Conditioning is substantially worse at low drive, and the Coriolis signal is small relative to baseline bending. These are mechanical-input results only; no neurons, receptor current, gap junctions, muscle response or body dynamics were executed.

The exact response matrices and row labels are in [response-matrices.json](response-matrices.json). [result.json](result.json) contains SVD values, normalized Gram matrices, per-side spectra, sampled moment extrema and sampling comparisons. [collect.mjs](collect.mjs) calls the frozen pure helper; [analyze.py](analyze.py) performs NumPy linear algebra on saved matrices. All priors match the parent [prototype](../README.md), including the fixed root-attachment beam frame.

Each matrix row is one side, phase and modeled receptive orientation. Its three columns are

`J[row,k] = (signedProjection(+1 rad/s along root axis k) − signedProjection(−1 rad/s along root axis k))/2`.

The four orientations are 45°, 135°, 225°, 315°. Both sides use the same listed power; frequency is 236 Hz. No magnitude, rectification or neuron assignment enters this derivative. Matrix units are **g cm²/s² per (rad/s)**. Singular values divided by `sqrt(number of rows)` are reported for comparison across phase grids. Rank uses relative cutoff `largest singular value × 1e−10`.

| Power | Phases | Bilateral rank | Condition number | RMS-normalized singular values |
|---|---:|---:|---:|---|
| 0 | 8 / 32 / 128 | 0 | undefined | 0, 0, 0 |
| 0.25 | 8 | 3 | 63.8928 | 5.84479e−8, 4.20811e−8, 9.14781e−10 |
| 0.25 | 32 | 3 | 63.8907 | 5.84479e−8, 4.20811e−8, 9.14811e−10 |
| 0.25 | 128 | 3 | 63.8907 | 5.84479e−8, 4.20811e−8, 9.14811e−10 |
| 1 | 8 | 3 | 18.1512 | 2.12133e−7, 1.64408e−7, 1.16869e−8 |
| 1 | 32 | 3 | 17.9736 | 2.11985e−7, 1.64552e−7, 1.17942e−8 |
| 1 | 128 | 3 | 17.9736 | 2.11985e−7, 1.64552e−7, 1.17942e−8 |

The relative Frobenius difference in `JᵀJ/rows` between 8 and 128 phases is **4.05e−7 at power0.25** and **0.001528 at power1**. The 32-versus-128 difference is below **1.1e−14**. The weakest singular value at power1 is underestimated by about **0.91%** with eight phases. This is convergence of a uniform full-cycle phase calculation, not proof that real-time held input preserves the signal.

At 128 phases, each side alone is technically rank3, but condition numbers are **7125.7 left / 8130.5 right at power0.25**, and **447.2 / 510.3 at power1**. Thus a binary unilateral rank3 statement would conceal severe sensitivity imbalance. This exact geometry includes small COM components along its modeled oscillation axes and uses constructed fixed beam frames; it is not an ideal planar sensor or a measurement of biological unilateral observability. Bilateral spectra are better conditioned than either side alone.

Sampled scale at 20 rad/s, using the 128-phase grid; all entries below are **g cm²/s²**. Coriolis norm minima are numerically zero (≤3.5e−22) at turning phases. Zero-power baseline and Coriolis moments are exactly zero within the prototype's restricted mechanics.

| Power / side | Baseline moment norm min–max | Maximum Coriolis moment norm: Ωx / Ωy / Ωz =20 |
|---|---|---|
| 0.25 left | 6.01030e−7 – 1.05307e−4 | 1.69271e−6 / 1.71092e−6 / 1.60077e−6 |
| 0.25 right | 5.46634e−7 – 1.09292e−4 | 1.73855e−6 / 1.77219e−6 / 1.68407e−6 |
| 1 left | 9.61648e−6 – 4.21227e−4 | 7.12355e−6 / 8.38220e−6 / 7.08874e−6 |
| 1 right | 8.74614e−6 – 4.37168e−4 | 7.28761e−6 / 8.68492e−6 / 7.47667e−6 |

| Power / side | Baseline signed-projection range across four orientations | Maximum absolute Coriolis projection: Ωx / Ωy / Ωz =20 |
|---|---|---|
| 0.25 left | ±7.48842e−5 | 1.19737e−6 / 1.21010e−6 / 1.13239e−6 |
| 0.25 right | ±7.76642e−5 | 1.22971e−6 / 1.25322e−6 / 1.19120e−6 |
| 1 left | ±3.03901e−4 | 4.98634e−6 / 5.69271e−6 / 4.90519e−6 |
| 1 right | ±3.14634e−4 | 5.10258e−6 / 5.89226e−6 / 5.16745e−6 |

The projected rotation-dependent peaks are approximately **1.5–1.9% of the baseline projected peak** for these axis probes. A transducer that saturates on baseline bending, removes temporal structure, or cannot resolve small waveform changes may discard that information. The present calculation deliberately selects no current scale, threshold, gain or desired motor correction. The full per-orientation extrema and RMS values, including total-load projections at20rad/s, are retained in `result.json`.

Minima/maxima are **sampled extrema**, not certified continuous extrema. Negative20rad/s reverses the Coriolis vectors/projections but preserves their norms. Maximum odd/even decomposition residual is **1.63e−19**, and the independently evaluated20rad/s projection differs from20× the ±1rad/s response by at most **1.94e−18**. The collector made **10,080 pure analytical evaluations**, without advancing native or neural state.

The uniform grids have phase-equivalent spacings **0.52966 / 0.13242 / 0.03310 ms**. Actual0.5ms and2ms steps at236Hz advance **42.48° and169.92°**, respectively, and do not coincide with those uniform phase grids. These matrices therefore do not validate a2ms hold, a0.5ms pulse-current implementation, spike timing, or any aliasing-induced firing. They also presume Ω and drive remain constant through the cycle; conditioning does not measure instantaneous three-axis observability or robustness to noise/model mismatch.

Reproduce:

```sh
node reports/flight-haltere-prototype/observability/collect.mjs
.venv/bin/python reports/flight-haltere-prototype/observability/analyze.py
```

The collector rejects changes to the frozen helper/geometry hashes. Its output pins those sources and the parent ledger; the analysis pins the response matrices and its own source. No production or parent-prototype source was changed for this artifact.
