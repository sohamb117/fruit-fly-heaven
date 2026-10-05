# Published DLM neuron model: source and integration boundary

Read-only research, 2026-09-14. No dependencies were installed, no simulation was run, and no model/runtime configuration was changed.

The model in Hürkey et al. (2023) is available as author code. It is a **three-state ionic-conductance model**, not an adaptive integrate-and-fire parameter set. Its reported desynchronization mechanism depends on the single-cell phase-response curve near a homoclinic/saddle-node-loop onset, together with weak electrical coupling. A fitted firing rate alone does not reproduce that mechanism.

## Exact sources

- [Paper, Nature DOI10.1038/s41586-023-06099-0](https://www.nature.com/articles/s41586-023-06099-0), Methods: “Computational modelling and equations”; Fig.3 and Extended Data Fig.9. [PMC version](https://pmc.ncbi.nlm.nih.gov/articles/PMC10232364/).
- [Author code, Zenodo7740678, v1.0.0](https://zenodo.org/records/7740678), archive `drosophila_wing_cpg.zip`; [direct download](https://zenodo.org/records/7740678/files/drosophila_wing_cpg.zip?download=1). Published MD5 `abcfdfa63d0d7c15f948df080d288ec9` was verified before reading.
- [Supplementary Tables2–4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06099-0/MediaObjects/41586_2023_6099_MOESM1_ESM.pdf): common constants, gate equations, and regime-specific conductance/input. Saved as `supplementary-tables.pdf` with extracted text alongside it.
- [Experimental electrophysiology and imaging data, Zenodo7737730](https://doi.org/10.5281/zenodo.7737730). Not downloaded here. The code archive also contains experimental spike-train arrays, which were not retained in this bounded source extraction.

Only small source/configuration files were extracted under `source/drosophila_wing_cpg/`; the 36.3MB archive was read in memory and not retained. `source-provenance.json` records its SHA-256 and the extracted files' hashes. `paper.html` is the retrieved paper page. The original 2015 Berger/Crook formulation had inconsistencies acknowledged by the 2023 authors; use the corrected 2023 equations/code rather than reconstructing constants from the older article.

| Local author file | What it specifies |
| --- | --- |
| `source/drosophila_wing_cpg/cfg/Berger_SNL.json` | Exact three ODEs, gates, and near-SNL parameter set |
| `source/drosophila_wing_cpg/cfg/Berger_SNIC.json` and `Berger_Hopf.json` | Excitability-class controls |
| `source/drosophila_wing_cpg/cfg/ggap_hom.npy` and `ggap_het.npy` | Homogeneous scalar and heterogeneous 5×5 coupling matrix, multiplied by nS by the calling code |
| `source/drosophila_wing_cpg/utils/sim.py` | Continuous integration, electrical-current sum, input override, spike detection, noise |
| `source/drosophila_wing_cpg/py/F3B_sim_exampletrace.py` | Deterministic reference:5 neurons, RK4,100µs step,5s, selected initial phases |
| `source/drosophila_wing_cpg/py/F3D_sim_weakstrongGJ.py` | Noisy weak/strong coupling comparison |
| `source/drosophila_wing_cpg/py/F3H_sim_hetSNLSNIC.py` | Heterogeneous coupling with SNL versus SNIC neurons |
| `source/drosophila_wing_cpg/py/F3J_sim_hethom.py` | Long noisy homogeneous/heterogeneous sequence comparison |
| `source/drosophila_wing_cpg/utils/random_initial_conditions.py` | Initial V,h,b sampled from a previously simulated uncoupled neuron's limit cycle |
| `source/drosophila_wing_cpg/utils/PRC_estimation_helper.py` and `couplingfunction_helper.py` | Perturbation-based phase-response/coupling-function analysis |

## Equations and constants

The dynamic variables are membrane voltage V, sodium inactivation fraction h, and Shab potassium activation b. Sodium activation m is instantaneous. In consistent units:

```
C dV/dt = Iin + Σj gij(Vj − Vi)
          − gL(V − EL) − gNa m∞(V)^3 (1 − h)(V − ENa)
          − gShab b^4(V − EK)
dh/dt = [h∞(V) − h]/τh(V)
db/dt = [b∞(V) − b]/τb(V)

p∞(V) = 1/[1 + exp(−Q zp(V − vp))],  p ∈ {m,h,b}
τp(V) = exp[−Q zp γp(V − vp)] /
        {rp[1 + exp(−Q zp(V − vp))]},  p ∈ {h,b}
```

| Quantity | Published value |
| --- | ---: |
| C |130pF |
| gL |8.624nS |
| gNa |431.2nS |
| EL, EK, ENa |−60,−72,+55mV |
| Q |0.0392/mV |
| zm, vm |3,−33mV |
| zh, vh |5.2,−39.14mV |
| zb, vb |1.1056,−42.14mV |
| rh, rb |0.2/ms each |
| γh, γb |0.38 each |

| Regime | gShab | Constant Iin |
| --- | ---: | ---: |
| HOM/SNL |137.68216nS |108.75pA |
| SNIC |215.6nS |175pA |
| Hopf |344.96nS |330pA |

The input currents are **selected model operating points near the corresponding onset**, not measured universal premotor currents or values to add automatically to the BANC chemical drive. The paper treats premotor cholinergic excitation as driving tonic spiking. The code's chemical-synapse option is unused in the paper's five-cell experiments.

There is **no additional spike-triggered adaptation current, calcium-dependent potassium current, threshold reset, or forced post-spike voltage clamp** in this minimal model. Sodium inactivation and voltage-dependent Shab activation provide its ionic dynamics. The paper explicitly notes that a pronounced after-hyperpolarization is absent from the minimal model and is not necessary for its desynchronization result.

`utils/sim.py` detects spikes above −10mV and uses a10ms Brian refractory condition. No ODE is marked “unless refractory” and no reset is supplied: this is an event-detection guard, not a10ms membrane-voltage clamp. It should not be copied as a substitute for fitting our current threshold/reset neuron.

## Electrical coupling and experimental constraints

The modeled circuit has all five neurons connected bidirectionally with linear, nonrectifying currents gij(Vj−Vi). The diagonal is zero. All units use identical intrinsic parameters within a simulation.

| Condition | Conductance per undirected pair |
| --- | ---: |
| Homogeneous weak reference |43.5pS |
| Heterogeneous MN1–MN2 and MN3–MN4 |86.59pS |
| Heterogeneous remaining MN1–4 pairs |38.27pS |
| Heterogeneous each MN5 connection |27.19pS |
| Homogeneous strong-coupling control |3nS |

The exact saved heterogeneous nS values are0.08658631126375121,0.03826934377953759, and0.027187499803804395; the homogeneous scalar is0.043499999686087025nS. Preserve those source values for numerical reproduction, while reporting sensible precision in explanatory text.

Measured pair coupling coefficients were0.023±0.003 for MN1–MN2/MN3–MN4 and0.010±0.0027 for other MN1–4 combinations. The model preserves the measured **relative strengths**, fixes the mean conductance, and assigns MN5 a coupling half the mean of the other connections. Its absolute conductances and MN5 value are model choices, not direct measurements of each BANC electrical edge. The source's passive two-cell relation is CC=gGap/(gGap+gL); a43.5pS model pair therefore has CC≈0.00502.

Deterministic Fig.3B/C integration uses RK4 at100µs. Noisy Fig.3D/H/J uses Heun at3µs with white-current-noise amplitude approximately0.949pA√ms; this is encoded as3e−8µA√s. Fig.3D/H uses60s runs; Fig.3J uses300s runs, generally10 repetitions. Most runs initialize each cell at a randomly sampled phase of a single-cell periodic trajectory. Fig.3B instead chooses phases[0.9,0.6,0.5,0.7,0.8] to illustrate a preferred sequence. That selected initialization must not be mistaken for spontaneous ordering from identical resting states.

The authors explicitly do not include firing-frequency homeostasis and accept implausible frequencies in part of their coupling sweep. The published model is consequently a mechanistic excitability/coupling reference, not a ready-made complete flight controller or a guarantee of3–12Hz under arbitrary BANC input.

## Compatibility with the present BANC runtime

Current `configs/banc-physiology.json` assigns all motor neurons C40pF, leak2nS, threshold−42mV, reset−58mV, refractory2ms, and an exponentially decaying150ms adaptation current incremented by0.4pA per spike. `packages/banc-runtime/src/neural.wgsl` has the corresponding linear conductance/leak update plus reset. It has no Na/Shab gates or continuous action-potential waveform. Its listed electrical edges are currently two assumed GF–PSI pairs, not this DLM circuit.

The common physical units are compatible: pF, nS, mV, pA, and ms. The dynamics are not. Changing C/leak/adaptation could fit selected rate-current observations phenomenologically, but it does not port the three-state equations or establish the required phase-response curve and electrical-coupling behavior. The most literal small implementation requires two extra ionic states per selected DLM neuron and a separate continuous integration rule for those cells. Our0.5ms neural step is not a validated numerical substitute for the published100µs RK4 reference; convergence must be checked for a selective/substepped implementation.

Prepared `io.json` identifies ten DLM MNs: four labeled `DLM1-4` plus one `DLM5` for each annotated side. It does **not** identify which of the four `DLM1-4` entries corresponds to physiological MN1,2,3,or4. Do not assign the two stronger pairs by arbitrary index order. Muscle-side versus soma-side conventions, especially for the contralateral MN5 soma, must also be reconciled before choosing exact BANC edge endpoints. Homogeneous coupling is an explicit published toy-circuit control that does not require inventing that internal ordering.

## Smallest defensible next path

1. Make a separately pinned five-neuron reference fixture from the author HOM/SNL equations and homogeneous weak coupling, retaining source units, input, continuous dynamics, and declared initialization. Compare uncoupled and coupled rate/phase responses, not flight reward. This is the positive control for the neuronal mechanism; no such simulation was run during this review.
2. If retaining LIF is a priority, fit a named DLM-specific approximation to the published/current-clamp f–I data, then independently test its PRC and weak/strong-coupling outcomes. Call it an approximation until those checks pass. Do not infer adequacy from reducing115Hz to a target rate.
3. For a mechanistic port, opt in only the ten verified DLM cells to the new ionic profile. Preserve BANC chemical inputs and outputs, with explicit current/conductance units; calibrate the incoming drive against the reference's f–I response. Do not blindly add108.75pA atop that input. Add electrical edges as separately sourced assumptions after the identity/side crosswalk is resolved, using the homogeneous graph first as a declared diagnostic if necessary.
4. Keep DVM and synchronous steering separate. This paper does not provide their neuron profiles and cannot explain or repair silent b1 cells by itself. Retain raw event observation and validate the DLM event-to-calcium/muscle boundary independently before resuming body-level optimization.

The actual captured115–118Hz mean DLM rate motivates this investigation, but no published constant above was selected or changed to fit that trajectory or desired flight behavior.
