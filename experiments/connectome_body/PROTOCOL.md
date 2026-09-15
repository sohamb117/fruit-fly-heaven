# Experimental protocol

Status: implemented experiment suite, with engineering validation. Successful learning and biological rankings remain to be measured; [VALIDATION.md](VALIDATION.md) records the actual evidence.

## Questions and estimands

1. **Generic interface engineering:** can the same adapter family, retrained independently, obtain control from different frozen graphs? Report held-out success/reward, learning curves, experience, stability, and the smallest successful tested size. Accepting a graph file is not successful embodiment.
2. **Value of connectivity:** compare every real graph with a retrained degree null and matched frozen random RNN, then with exact adapter-only and conventional trainable RNN policies. The real-minus-null learning/experience difference is the main topology contrast.
3. **Compatibility with FlyBody:** compare capacity and experience curves across connectomes, conditional on data coverage and common assumed dynamics. A fly–insect–vertebrate ordering would be consistent with an embodiment prior. A single body and few specimens do not identify a connectome-by-morphology interaction or a causal effect of phylogenetic distance.

For fixed optimizer/procedure \(\mathcal A\), actor ceiling \(P\), training seed \(s\), and port/null draw \(g\), define

\[
\widehat E_{\mathcal A,P,s,g}(C,B,\tau)
=\text{first evaluation step certified to meet success threshold }\tau.
\]

Primary certification requires success ≥0.8 on the same 20 validation scenarios at **three consecutive evaluations**, 20,000 training interactions apart. Record both the first crossing and the later certification step. Runs that do not certify by one million interactions are right-censored at that budget. Keep these failures in the analysis; do not average only successful learners. Certification is an empirical persistence rule, not a population confidence guarantee.

This estimates a distribution for a specified training procedure, rather than the unobservable global minimum over all weights and learning algorithms. Evaluation interactions are separately recorded and included in total experience. Validation selects checkpoints; final test scenarios do not.

## Generic interface

For observation dimension \(O\), action dimension \(A\), hidden width \(w\), and \(K=16\) channels, use

\[
z_t=W_{E2}\tanh(W_{E1}o_t+b_{E1})+b_{E2},\qquad
\mu_t=W_{D2}\tanh(W_{D1}r_t+b_{D1})+b_{D2}.
\]

E and D are feedforward and contain no learned recurrent state. Including all biases and action log-standard-deviations,

\[
P_{E,D}=(O+2K+A+2)w+K+2A.
\]

Choose the largest integer width under the declared ceiling. At \(O=286,A=59,K=16\), the 5k ceiling yields \(w=12\) and **4,682 parameters**. Neuron count does not enter this expression. Plot actual trainable counts, alongside the ceiling used for comparisons.

For each graph/draw, uniformly permute neurons into disjoint input and output halves. Assign balanced channel buckets and fixed ±1 projection signs. Input neurons receive their channel's scalar; readout sums signed output activity divided by the square root of its bucket population. Storage scales as O(N), rather than a dense N×K trainable projection.

Input/output assignments and per-node dynamical signs are held fixed in paired topology nulls. Randomness for ports, signs, rewiring, actor initialization, and scenarios uses separate seed namespaces. No anatomy labels, homologous neurons, motor maps, or body-specific graph edits are required. At least 2K nodes are needed. A graph with no effective input-to-output paths cannot control the body through this interface; preflight detects an absent encoder gradient.

The channel bottleneck can miss useful graph modes. A shared K sweep tests that limitation. Do not select favorable port seeds or a different K for one species. To support generalization beyond the development graphs, freeze the family/settings before introducing future releases.

Requiring an edge prevents a literal bypass. A large E/D pair could still implement most of the controller while using the graph as a communication channel. Capacity sweeps and independently trained no-brain/random controls are needed to distinguish that case from useful biological structure.

## Frozen sparse dynamics

Let \(A_{ij}\) count connections from presynaptic i to postsynaptic j. Transform weights with \(T(A)=\log(1+A)\), form incoming/outgoing strength diagonals, and use source-sign matrix S:

\[
W_C=D_{in}^{-1/2}T(A)^\top D_{out}^{-1/2}S.
\]

Zero-degree neurons have zero connections. This operator has Euclidean norm at most one. For each of two substeps per 10 ms policy interval,

\[
h\leftarrow(1-\alpha)h+\alpha\tanh(\gamma W_Ch+\beta Q_{in}z_t),
\quad \alpha=1-\exp[-\Delta t/(2\tau_n)],
\quad r_t=Q_{out}h.
\]

Primary settings: gain γ=0.9, input gain β=1, neural time constant τₙ=20 ms. The γ<1 regime is contractive in recurrent state, and tanh/leak bounds activity. This standardization can also suppress biological attractors. The gain-1.1 sensitivity tests a noncontractive regime with bounded activity.

Primary signs mark exactly the rounded 20% of nodes inhibitory using a seeded, topology-independent assignment shared by paired nulls. This controls unequal annotation availability. It tests structural connectivity under a common **synthetic sign/rate model**, not native animal physiology. BANC's separate annotated sensitivity assumes acetylcholine positive and GABA/glutamate/histamine negative, retaining seeded signs for unknown labels. Transmitters alone do not establish postsynaptic receptor physiology.

CSR multiplication caches both orientations. Its fixed-matrix backward operation propagates gradients into state and E without training any connectome weights. Episode resets clear state and cut previous-episode gradients. PPO differentiates through neural dynamics, not through MuJoCo.

## Controls and fairness

| Comparison | Matched properties | Changed properties |
|---|---|---|
| Real vs degree-shuffled | N, M, signs, ports, in/out degree, signed incoming degree, per-source/global weight multisets, actor capacity | Targets, motifs, communities, incoming weighted strengths, detailed dynamics |
| Real vs frozen random RNN | N, M, original isolates, signs, ports, global weights, actor capacity | Degree sequence, weighted strengths; extra isolates may arise |
| Real vs adapter-only | Identical E/D architecture and trainable count | E channels go directly to D; no brain or memory |
| Real vs adapter-GRU | Total actor ceiling and E/D family | Trainable K-dimensional GRU consumes some of the ceiling, reducing E/D width |
| Real vs conventional GRU | Actor ceiling, observations/actions, optimizer, experience | Direct observation→GRU→action, with recurrent width maximized under the ceiling |
| Real vs no-edges | Same adapter count and disjoint ports | Only a constant decoder output can act |

The degree null uses directed double-edge swaps between same-sign sources, rejecting autapses/duplicates. Specify ten attempts per edge and log accepted swaps and changed-target fraction. This finite chain is not claimed to sample uniformly or have proven mixing. The random RNN samples directed edges among the original active nodes, preserving original isolates and the global weight multiset.

The frozen random RNN matches large recurrent-state/graph resources. The trainable GRU is the standard learned-policy baseline. Matching a small RNN's trainable parameters to an adapter does not also match a 175k-dimensional frozen state, so disclose graph N/M, actor state size, critic count, and measured compute. The shared critic is a separate two-layer, 64-wide MLP; it never supplies features or state to the actor.

Adapter-only parity means the substrate is unnecessary at that capacity/task. It does **not** by itself establish statistical overfitting. Overfitting is evaluated through held-out scenarios, harder distributions, and training-start replay gaps. High-capacity adapters may compensate for substrate mismatch; a capacity curve tests this possibility.

## Body and tasks

Use pinned upstream FlyBody's full walking configuration, stock actuators, and disabled wings. Directed locomotion and target reaching are the primary learning tasks. Standing balance/recovery is retained for calibration. Flight/landing are not implemented tasks in this version.

The first 4,096-interaction conventional RNN pilot met the balance success criterion at initialization, despite improving its reward during training. This makes balance success unsuitable for the principal adaptation-cost claim. Keep that calibration result visible rather than interpreting it as learned embodiment. The main study uses seeds 100–104, distinct from pilot seeds 0–1 and sensitivity seeds 10–12; those seeds define separate evaluation scenario cohorts as well as training randomization.

| Observation block | Dimensions |
|---|---:|
| Joint position / π; joint velocity / 50 rad s⁻¹ | 102 + 102 |
| Actuator activation | 59 |
| Body-frame world-up, linear velocity, angular velocity | 3 + 3 + 3 |
| Thorax height | 1 |
| Body-frame target displacement; commanded velocity | 2 + 2 |
| Task identity; transformed touch forces | 3 + 6 |

Scale observations by fixed constants and clip to [-10,10]; do not introduce learned normalization or connectome-specific sensors. Affinely map 59 normalized actions into native actuator limits. FlyBody uses centimetres, grams, and seconds. Physics dt is 0.2 ms; the native 2 ms action interval repeats five times per policy decision.

Episodes last 200 interactions (2 seconds). Initial heading/tilt, target/command, and velocity kicks are randomized. Training/validation/test have separate deterministic seed namespaces. Validation/test share the training distribution. OOD expands heading/bearing to the full circle, increases target distances/speeds, and doubles kicks.

| Task | Dense progress term | Required throughout final 0.25 seconds |
|---|---|---|
| `balance` | exp[-(speed/1 cm s⁻¹)² − (origin distance/0.25 cm)²] | Upright >0.85, speed <0.5 cm/s, origin distance <0.15 cm, height >0.07 cm |
| `walk` | exp[-(velocity error/1.5 cm s⁻¹)²] | Upright >0.8, velocity error <0.75 cm/s, height >0.07 cm |
| `reach` | exp[-(target distance/0.5 cm)²] | Upright >0.8, distance <0.15 cm, speed <0.5 cm/s, height >0.07 cm |

Reward multiplies progress by upright, Gaussian height centered at 0.1278 cm, and `1 − 0.05 × mean(action²)`. Upright below 0.2 or height below 0.035 cm terminates with failure. Numerical physics failures terminate with zero reward and remain in the denominator. Score is return divided by the full horizon, so early falls do not inflate mean reward.

Before a full study, the feasibility pilot must demonstrate that a sufficiently large conventional policy can learn each task. If all policies fail, topology conclusions are unsupported: inspect control scaling, task difficulty, optimizer, and signal strength, change the shared protocol, and rerun affected conditions with a new identity. Do not quietly drop tasks or add a privileged controller to one substrate.

## Optimization and selection

Primary PPO: four environments, 128 rollout steps, sequence length 16, burn-in up to 16, four epochs, Adam 3e-4, clip ratio 0.2, discount 0.99, GAE 0.95, gradient clipping 0.5. Exploration uses entropy of the pre-tanh Gaussian; do not describe it as transformed-action entropy. Store unsquashed actions for stable log probabilities.

Time limits bootstrap values; terminations do not. GAE never crosses an episode boundary. Sequence minibatches preserve time order. No-gradient burn-in refreshes stored rollout states before truncated BPTT; this is an approximate recurrent PPO procedure, not full-episode optimization.

Evaluate at zero, every cadence, and the exact final budget. Select checkpoints by validation success, breaking ties with score. Continue training to the full budget after certification. Test/OOD/replay/interventions use the selected checkpoint. Log counts, source/graph/port hashes, gradients, state saturation, and throughput.

Freeze primary settings after a separately labeled feasibility pilot and before final test access. Per-connectome architecture search, favorable neuron selection, reward changes, and extra training violate a primary comparison. Any shared hyperparameter search must have equal trials/experience across substrates and its additional cost reported.

## Dependence and generalization checks

At the selected checkpoint, evaluate identical test scenarios intact and under:

- **Silence:** zero the latent before D; for adapter-only, zero E's latent.
- **Per-step reset:** erase recurrent state before each policy decision. Two within-step neural updates still allow some edge transmission, isolating longer memory.
- **No recurrence:** remove graph propagation; disjoint output neurons remain zero from their initial state.
- **Readout permutation:** permute activities within the output half, preserving which neurons are accessible.
- **Acute rewiring:** apply the paired degree-null graph without retraining E/D.

Acute drops show intervention sensitivity and may involve distribution shift. They do not show that a real graph is intrinsically more useful: that requires independently retrained topology controls. Training-start replay minus held-out score measures an empirical generalization gap, separate from dependence on the substrate.

## Analysis

Training seeds are sampling units; port/null draws are nested, and episodes are repeated measurements. Weight training seeds equally, bootstrap entire clusters, and retain draws together. Use weighted Kaplan–Meier and restricted mean experience to the common cap. Plot learning curves, held-out/OOD success against actual actor parameters, and restricted mean experience against capacity.

Pair real/null comparisons by graph, task, ceiling, training seed, and port/null draw. Brain-free baselines pair by training seed and are reused across draw comparisons with clustering. Cross-connectome comparisons first average draws within each seed. Release/subset fingerprints and protocol settings separate incompatible groups. Five seeds still have limited precision; intervals are exploratory and unadjusted for multiple comparisons. A confirmatory multiplicity/power plan should be frozen separately.

The descriptive capacity frontier is the smallest tested actor size with mean test success ≥τ and seed-averaged certified fraction ≥0.8. Otherwise report no success in the tested grid. Do not assume monotonic performance or infer a global optimal size.

Audit coverage, isolates/components, degree/weight definitions, sign availability, and reconstruction quality. Run equal-N/equal-M subsampling alongside full graphs, recognizing that subsampling destroys real circuitry. Examine rankings across common gain/timescale/port/weight/sign sweeps. Size, sex, developmental stage, coverage, and assumed dynamics can confound cross-connectome comparisons. A crossed multiple-connectome × multiple-body experiment is needed to directly estimate a morphology interaction; it is outside this fixed-FlyBody suite.

## Related work and evidence boundary

[Jin et al., arXiv:2602.17997v3](https://arxiv.org/abs/2602.17997v3) already report connectomic graph policies controlling a biomechanical fly with sample-efficiency comparisons. The proposed contribution here is the shared constrained interface, capacity/experience curves, strong brain-free controls, and cross-connectome comparison. Do not claim the first connectome-controlled fly. This description is based on their abstract and is not a reproduction claim.

[FlyBody](https://github.com/TuragaLab/flybody/tree/d015e9bfe441bd90ae431bac24c55cb74bdbce26) supplies the body. [BANC](https://github.com/sjcabs/fly_connectome_data_tutorial/blob/main/data/dataset_documentation/banc_data.md) and [MaleCNS](https://male-cns.janelia.org/download/) supply structural graphs. Port mappings, signs, rate dynamics, time constants, and learned functional decoding are benchmark assumptions. Successful control would not establish recovery of natural animal sensorimotor physiology.
