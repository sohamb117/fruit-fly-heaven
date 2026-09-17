# Experimental Plan: Measuring Connectome–Body Compatibility Through Embodied Learning

## Research objective

Determine whether biological connectome structure provides a useful architectural prior for embodied control, whether that advantage depends on compatibility between the connectome and body, and how much learned interface capacity is required to bridge them.

The central claim to test is:

> Connectome–body compatibility is measurable as a reduction in both the interface complexity and optimization complexity required to learn embodied control.

The controller is:

$$
o_t
\xrightarrow{E_\theta}
x_t
\xrightarrow{C_{A,W}}
h_t
\xrightarrow{D_\phi}
a_t
$$

where:

* \(E_\theta\) is the trainable sensory encoder.
* \(A\) is the fixed connectome adjacency structure.
* \(W\) contains trainable weights on permitted edges.
* \(D_\phi\) is the trainable motor decoder.
* No connections outside \(A\) are introduced.

The primary outcome is environment interactions required to reach a predefined task-success threshold. Secondary outcomes are learning-curve AUC, success rate, final performance, gradient updates, compute-to-threshold, robustness, and minimum successful adapter capacity.

---

## Experimental factors

### Neural substrates

* BANC
* MaleCNS
* C. elegans connectome
* Zebrafish connectome
* Standard RNN
* GRU
* Degree-preserving rewired connectome
* Community-preserving rewired connectome
* Node-and-edge-matched random recurrent graph
* Adapter-only controller

### Bodies and task families

* Fly body: hover, controlled flight, walking, limb control
* Worm body: locomotion, steering, posture stabilization
* Fish body: swimming, heading control, depth stabilization
* Engineered multirotor drone: the twenty-task battery below

The drone is an additional engineered embodiment with no biological native-pair
label. It does not replace the crossed fly/worm/fish study or enter its native-pair
diagonal contrast. Every substrate also receives the temporal diagnostic battery
below, through the same stateless interface and substrate dynamics.

Each body receives matched and mismatched connectomes. Performance is compared within each body, while the complete connectome × body matrix is used to estimate compatibility.

### Encoder–decoder families

1. Linear encoder + linear decoder
2. MLP encoder + linear decoder
3. Linear encoder + MLP decoder
4. MLP encoder + MLP decoder
5. Low-rank adapters
6. Sparse structured adapters
7. Anatomically informed mappings

All generic adapters are stateless. Anatomical mappings are treated as a separate oracle condition rather than as the primary generic interface.

### Plasticity regimes

1. Adapters train; substrate weights remain frozen
2. Substrate weights train; adapters remain fixed
3. Encoder only trains
4. Decoder only trains
5. Adapters and permitted substrate weights train jointly

---

# Experiment 1: Encoder–decoder architecture

## Research question

Which stateless encoder–decoder architecture provides the best general interface between neural substrates and embodied systems?

## Null hypothesis \(H_0\)

After controlling for parameter count and bottleneck dimension, encoder–decoder architecture does not materially affect learning efficiency or success.

## Alternative hypothesis \(H_1\)

Encoder–decoder architecture materially affects learning efficiency, and a particular architecture offers a superior performance–complexity tradeoff.

## Experimental design

Train every encoder–decoder family across biological connectomes, artificial recurrent controls, and embodied tasks. Compare both parameter-matched adapters and capacity sweeps.

The preferred interface is the smallest architecture on the Pareto frontier of:

$$
\text{sample efficiency},
\quad
\text{success rate},
\quad
\text{final performance},
\quad
\text{adapter size}.
$$

## Ideal outcome

A small stateless adapter—preferably linear or a shallow MLP—matches or exceeds larger interfaces across most substrates. This establishes a generic interface that cannot independently implement the full controller.

---

# Experiment 2: Interface dependence on connectome–body compatibility

## Research question

Does the optimal encoder–decoder depend on which connectome is attached to which body?

## Null hypothesis \(H_0\)

Adapter rankings are invariant across connectome–body pairs. Adapter effects and substrate effects are additive.

## Alternative hypothesis \(H_1\)

Adapter effectiveness interacts with connectome–body compatibility. Compatible pairs require simpler interfaces, while mismatched pairs require more expressive translation.

## Experimental design

Measure the interaction:

$$
\text{adapter architecture}
\times
\text{connectome}
\times
\text{body/task}.
$$

For each pair, estimate the minimum adapter width, depth, rank, and parameter count required to reach the task-success threshold.

## Ideal outcome

Biologically compatible pairs succeed with linear, low-rank, or small MLP adapters, while mismatched pairs require substantially larger nonlinear adapters or fail entirely.

This would support interface complexity as a quantitative measure of connectome–body compatibility.

---

# Experiment 3: Biological topology versus artificial recurrence

## Research question

Does connectome topology make embodied control easier to learn than conventional recurrent architectures?

## Null hypothesis \(H_0\)

After controlling for parameter count, sparsity, state dimension, and training procedure, biological connectomes provide no advantage over RNNs, GRUs, or random recurrent graphs.

## Alternative hypothesis \(H_1\)

Biological connectome topology reduces the sample or optimization complexity of learning embodied control.

## Experimental design

Compare each connectome against:

* Parameter-matched RNN
* Compute-matched RNN
* Parameter-matched GRU
* Sparse random recurrent graph
* Adapter-only MLP

Use identical observations, actions, objectives, training budgets, and adapter families.

## Ideal outcome

Connectome-constrained controllers reach task thresholds faster, succeed more reliably, or require fewer trainable parameters than artificial recurrent controls, particularly on compatible bodies and tasks.

---

# Experiment 4: Specificity of biological wiring

## Research question

Is any connectome advantage caused by the specific biological arrangement of edges rather than generic recurrence or graph statistics?

## Null hypothesis \(H_0\)

Connectome performance is explained by neuron count, edge count, sparsity, degree distribution, or community structure.

## Alternative hypothesis \(H_1\)

The specific biological wiring provides an additional trainability advantage beyond these coarse properties.

## Experimental design

For every connectome, compare:

1. Original topology
2. Community-preserving rewiring
3. Degree-preserving rewiring
4. Direction-shuffled topology
5. Edge-count-matched random graph

Use identical neuron dynamics, adapters, initialization scales, and learning procedures.

## Ideal outcome

Performance follows a graded ordering:

$$
\text{biological topology}
>
\text{community-preserved}
>
\text{degree-preserved}
>
\text{random}.
$$

This would attribute the advantage to progressively finer biological organization.

---

# Experiment 5: Topology, initialization, and plasticity

## Research question

Which part of the connectomic prior is useful: topology, biological weights and signs, or the ability to adapt synapses?

## Null hypothesis \(H_0\)

Biological weight initialization and connectome-constrained plasticity provide no benefit beyond the adjacency mask or learned adapters.

## Alternative hypothesis \(H_1\)

Topology, biological initialization, and synaptic plasticity make separable contributions to trainability.

## Experimental design

Within each biological topology, compare:

* Biological synaptic magnitudes and signs
* Sign-preserving randomized magnitudes
* Shuffled biological magnitudes
* Fully random permitted-edge initialization
* Frozen substrate
* Trainable permitted-edge weights
* Joint adapter and substrate training

## Ideal outcome

The topology-only condition outperforms rewired controls, biological initialization accelerates early learning, and joint plasticity produces the strongest final controller.

A particularly strong result would be:

$$
\text{biological topology + biological initialization + plasticity}
>
\text{biological topology + random initialization}
>
\text{rewired topology}.
$$

---

# Experiment 6: Matched versus mismatched connectome–body pairs

## Research question

Do connectomes learn disproportionately well when controlling biologically compatible bodies?

## Null hypothesis \(H_0\)

Connectome and body effects are additive. There is no matched-pair advantage:

$$
Y_{C,B}=\mu+\alpha_C+\beta_B+\epsilon.
$$

## Alternative hypothesis \(H_1\)

Matched connectome–body pairs exhibit a positive compatibility interaction:

$$
Y_{C,B}
=
\mu+\alpha_C+\beta_B+
\delta\mathbf{1}[C\text{ matches }B]+\epsilon,
\qquad
\delta>0.
$$

## Experimental design

Construct a crossed connectome × body matrix. Train every connectome on every body using the same generic adapter families. Compare each pairing against its expected performance after controlling for connectome-wide and body-wide difficulty.

Compatibility is measured through:

* Residual learning-curve AUC
* Residual steps-to-threshold
* Minimum successful adapter capacity
* Success probability
* Final normalized performance

## Ideal outcome

Matched pairs occupy the diagonal of the performance matrix:

$$
\begin{array}{c|ccc}
 & \text{Fly body} & \text{Worm body} & \text{Fish body}\\
\hline
\text{Fly connectome} & \mathbf{high} & low & low\\
\text{Worm connectome} & low & \mathbf{high} & low\\
\text{Fish connectome} & low & low & \mathbf{high}
\end{array}
$$

The strongest evidence would be that matched pairs both learn faster and require smaller interfaces.

---

# Experiment 7: Task-specific compatibility

## Research question

Does compatibility operate at the level of an entire body or at the level of particular sensorimotor functions?

## Null hypothesis \(H_0\)

A connectome’s relative ranking remains constant across tasks.

## Alternative hypothesis \(H_1\)

Connectome advantages depend on whether the task engages structures relevant to their biological specialization.

## Experimental design

Within each body, compare stabilization, locomotion, steering, and appendage-control tasks. Evaluate whole connectomes and task-relevant subgraphs against equally sized irrelevant or randomized subgraphs.

Measure:

$$
\text{connectome}
\times
\text{task}
$$

interactions.

## Ideal outcome

Task-relevant connectome structures produce disproportionate advantages on corresponding control problems. For example, fly motor circuitry provides a larger advantage for flight or articulated limb coordination than for generic stabilization.

---

# Experiment 8: Causal substrate utilization

## Research question

Does the trained controller genuinely depend on connectome computation, or can the adapters bypass it?

## Null hypothesis \(H_0\)

Behavior is primarily implemented by the adapters; structured substrate activity is unnecessary.

## Alternative hypothesis \(H_1\)

The trained policy causally depends on structured temporal activity within the connectome.

## Experimental design

Apply post-training interventions:

* Zero substrate activity
* Replace activity with its temporal mean
* Shuffle activity across time
* Permute neuron identities
* Remove recurrent state
* Lesion task-relevant subnetworks
* Replace the substrate with a matched random graph without retraining

## Ideal outcome

Normal performance depends on intact, temporally structured connectome activity. Targeted lesions produce task-specific deficits, while generic adapter outputs cannot preserve behavior after substrate disruption.

---

# Experiment 9: Transfer and robustness

## Research question

Does connectome–body compatibility produce reusable control structure rather than narrow optimization gains?

## Null hypothesis \(H_0\)

Connectome advantages are limited to the training distribution.

## Alternative hypothesis \(H_1\)

Compatible connectomes improve zero-shot transfer, fine-tuning efficiency, and robustness.

## Experimental design

Evaluate trained controllers under:

* New target speeds and trajectories
* Sensor noise and delay
* Altered mass and inertia
* Aerodynamic or friction changes
* Actuator weakness or failure
* Unseen terrain or current conditions
* Transfer between related control tasks

## Ideal outcome

Compatible connectome–body pairs preserve performance under perturbation and adapt to new tasks with fewer updates than mismatched connectomes, rewired graphs, and standard recurrent networks.

---

# Experiment 10: Unified compatibility model

## Research question

Can connectome–body compatibility be predicted from measurable structural and interface properties?

## Null hypothesis \(H_0\)

Performance variation cannot be systematically explained by structural compatibility measures.

## Alternative hypothesis \(H_1\)

Structural correspondence predicts interface requirements and learning complexity across unseen connectome–body pairs.

## Experimental design

Model training outcomes using:

* Connectome size and sparsity
* Degree distribution
* Modularity and hierarchy
* Recurrent path structure
* Sensory-to-motor path lengths
* Bilateral symmetry
* Motif distributions
* Adapter capacity
* Body sensor–actuator dimensionality
* Task dynamics

Fit the compatibility model on a subset of connectome–body pairs and evaluate its predictions on held-out pairs.

## Ideal outcome

The model predicts which unseen pairings will learn efficiently and how much adapter capacity they will require. Predictive power remains after controlling for connectome size, body difficulty, and parameter count.

---

# Primary hypotheses

### H1: Interface hypothesis

A small stateless encoder–decoder can provide a general interface across connectomes and bodies.

### H2: Structural-prior hypothesis

Biological topology reduces embodied-control training complexity relative to matched artificial recurrent systems.

### H3: Compatibility hypothesis

The benefit of biological topology is greatest when connectome and body are biologically or functionally compatible.

### H4: Interface-complexity hypothesis

Compatible pairs require smaller, simpler encoder–decoder interfaces.

### H5: Plasticity hypothesis

Connectome-constrained synaptic plasticity improves control while preserving the causal contribution of biological topology.

### H6: Generalization hypothesis

Compatibility improves transfer and robustness, not merely training-set performance.

---

# Target evidence for the strongest paper

The strongest result would establish all of the following:

1. Biological connectomes outperform matched RNN, GRU, and rewired controls on embodied-learning complexity.
2. The advantage survives parameter, compute, sparsity, and initialization controls.
3. Matched connectome–body pairs outperform their expected performance from connectome and body main effects.
4. Matched pairs require materially smaller encoder–decoder interfaces.
5. The controller fails under substrate disruption, demonstrating causal use of connectome dynamics.
6. Compatible pairs transfer and withstand perturbations better than controls.
7. Structural measurements predict trainability on held-out pairings.
8. Substrate effects replicate on an engineered drone across feedback, tracking,
   disturbance, memory and actuator-fault tasks.
9. Independently measured temporal capabilities predict specific drone-control
   strengths on held-out substrates beyond size, capacity and general performance.

The resulting central conclusion would be:

> Biological connectomes are not merely viable recurrent controllers. Their evolved structure provides a measurable, body-dependent inductive bias that reduces both the interface complexity and learning complexity required for embodied control.

These are hypotheses and target evidence, not conclusions assumed by the
implementation. Equal performance of biological and artificial controls is a
valid result.

---

# Additional benchmark D: Drone-control repertoire

Use one fixed six-degree-of-freedom multirotor model, rotor layout, observation
schema and physics step across substrates. The primary model is a redundant
six-rotor rigid body; report thrust/torque allocation and single-motor feasibility
checks. Failure tasks must respect available control authority. Qualification
establishes simulated mechanical consistency, not fidelity to a commercial drone.

Policies command individual normalized motor thrusts. There is no hidden learned
stabilizer, state estimator or recurrent adapter. A privileged analytic controller
is allowed only as a named physics/task qualification reference. Policy state
observations pass through the declared delay/noise/dropout path; clean simulator
state is evaluator information only. Goal cues and range/color features have
explicit semantics. Range observations and symbolic color cues do not establish
image perception.

| Task | Objective/capability | Primary measurements |
| --- | --- | --- |
| Attitude stabilization | Recover target roll, pitch and yaw | Angular error, recovery time, crash rate |
| Hover stabilization | Hold 3D position and orientation | Position/orientation error, survival |
| Setpoint switching | Change position, altitude and heading | Settling time, overshoot, steady-state error |
| Waypoint navigation | Reach sequential 3D waypoints | Completion, path length, elapsed time |
| Trajectory tracking | Circles, figure-eights, spirals and splines | Tracking error, phase lag, smoothness |
| Moving-target pursuit | Follow a partly unpredictable target | Distance, interception rate |
| Wind-gust recovery | Recover from external impulses | Recovery, displacement, crashes |
| Continuous turbulence | Fly under correlated wind | Tracking, action variance, survival |
| Sensor-delay control | Control with delayed pose and velocity | Performance versus delay |
| Sensor-dropout control | Fly during observation loss | Survival and drift during dropout |
| Noisy-sensor control | Corrupted pose and velocity signals | Performance versus noise |
| Actuator-delay control | Variable command latency | Stability versus delay |
| Motor-degradation recovery | One motor loses thrust | Recovery and residual tracking error |
| Motor-failure recovery | One motor disabled | Controlled landing, survival, feasibility |
| Payload variation | Unseen mass, inertia and center of mass | Performance across payloads |
| Energy-efficient flight | Track while limiting thrust expenditure | Power proxy per distance plus tracking |
| Color-conditioned navigation | Approach cued color, avoid the other | Correct-target rate, wrong-target contacts |
| Obstacle avoidance | Navigate clutter using range observations | Collision, completion time |
| Precision landing | Land on a designated platform | Landing success, touchdown velocity |
| Takeoff–navigate–land | Complete a multi-phase episode | Full-mission and phase success |

**Core six capability groups:** attitude, hover, trajectory, gusts, sensor delay
**and** dropout (two conditions), and motor degradation. Implement all twenty;
the core subset is the first bounded compute pass.

Cross each task with all biological substrates, degree/community-preserving
rewires and matched fixed sparse reservoirs, parameter-matched learned RNN/GRU
and no-brain controls. Retain stateless adapter families and all declared
plasticity regimes. Use paired seeds, fixed optimization and P={5k,20k,80k}.
Report actual adapter and total trainable counts separately when edges train,
including matching error. Measure J(C,drone,P,N,task), experience/capacity to
threshold, AUC, wall time and physical metrics. Severity curves and success
criteria are specified before training. Unreached thresholds are right-censored.
Crashes and failed seeds remain in results. Validation selects weights; held-out
test scenarios never tune the controller. Split generators use distinct namespaces.

---

# Additional benchmark T: Temporal capabilities

Use scalar/low-dimensional streams, small stateless encoders and linear (or
explicitly small) readouts through the same structural ports and dynamics.
Refit weights for each task/substrate. Only the substrate retains state: no
history stacking, target histories, temporal preprocessing or teacher-forcing
channels in the adapter. Task-defining cues are explicit current inputs.
Autonomous continuation receives zero input after its cue and no target feedback.

| Task family | Required behavior | Measurement |
| --- | --- | --- |
| Linear memory capacity | Recall independent input at delays k | Per-delay R², squared correlation and summed capacity |
| Delayed impulse recall | Reproduce a delayed pulse | Amplitude and timing error |
| Delayed match-to-sample | Compare with an earlier sample | Balanced accuracy |
| Temporal XOR | XOR separated bits | Balanced accuracy versus separation |
| Delayed parity | Parity over preceding k bits | Balanced accuracy versus window |
| Running sum/integration | Accumulate a rolling window | Normalized error versus window |
| Leaky integration | Estimate decaying input history | Error versus time constant |
| Multi-timescale integration | Short and long histories together | Per-timescale normalized error |
| NARMA-10 | Predict nonlinear input-driven dynamics | NMSE and R² |
| NARMA-20/30 | Longer-order declared variants | NMSE/R² with exact equations recorded |
| Mackey–Glass prediction | Forecast a delayed chaotic system | Error versus horizon and persistence baseline |
| Lorenz prediction | Forecast a multivariate chaotic state | Per-coordinate error versus horizon |
| Frequency discrimination | Recognize frequency with random phase/amplitude | Balanced accuracy |
| Phase discrimination | Compare phase to an explicit reference | Balanced accuracy |
| Rhythm continuation | Continue after oscillatory input stops | Error and phase drift |
| Missing-step prediction | Reconstruct a masked pattern element | Error at masked positions only |
| Change-point detection | Detect changes in signal statistics | Precision/recall, false alarms, detection delay |
| Context-dependent integration | Integrate the cued input channel | Error with distractors/context switches |
| Selective memory | Store cued input, reject distractors | Recall error at query times |
| Temporal order recognition | Identify which stimulus came first | Balanced accuracy at queries |
| Variable-delay response | Respond after a cue-dependent interval | Event precision/recall, timing error |
| Noisy sequence reconstruction | Denoise a latent signal | Error versus observation baseline |
| System identification | Infer unknown dynamics from history | Parameter error on unseen systems |
| Inverse dynamics | Infer action from an observed transition | Action reconstruction error |

**Core eight:** linear memory, temporal XOR, NARMA-10, multi-timescale integration,
frequency, context-dependent integration, Mackey–Glass and change-point detection.
NARMA-20 and NARMA-30 are distinct configured variants of the longer-order family.
Record coefficients/stability policy; never silently clip a diverging series.

Use immutable content-hashed complete trajectories, identical across substrates.
Fit scaling on training data only. Mask burn-in and query/recall periods, avoiding
credit for trivial blank timesteps. TBPTT carries state and detaches gradients at
declared boundaries. Count unique training input timesteps, repeated optimization
exposures, updates and compute separately; these are not drone interactions.
Checkpoint optimizer, RNG, data/split identities and sample cursor for exact resume.
Include real/rewired/fixed-random/learned-RNN/GRU/no-brain conditions and exact
encoder/port/readout/substrate parameter accounting. A nonlinear-readout
sensitivity track must be labeled separately from the primary linear readout.

---

# Additional analysis M: Temporal capabilities predict control strengths

| Temporal capability | Predicted drone strength |
| --- | --- |
| Short-delay memory | Attitude and hover |
| Long-delay memory | Sensor delay and dropout |
| Nonlinear temporal computation | Coupled tracking and motor-fault recovery |
| Frequency selectivity | Oscillatory tracking and disturbance rejection |
| Multi-timescale integration | Trajectory tracking and navigation |
| Context gating | Setpoint switching and color-conditioned navigation |
| Chaotic forecasting | Turbulence and moving-target pursuit |
| Change-point detection | Gust and motor-fault response |

These are predeclared associations, not assumed mechanisms. Fit on development
substrates; hold out entire **source substrates**, with their seeds, capacities,
plasticity regimes and null realizations kept in one fold. Null draws/episodes
are not independent biological replicates. Use nested group-held-out tuning and
training-fold scaling. Aggregate training seeds within matched cells for model
fitting and report seed uncertainty separately from substrate uncertainty.

Compare a baseline controlling size, edges, capacity and task difficulty with a
diagnostic-augmented model. Test specificity using within-cell task-relative
performance (removing global drone ability), target versus non-target contrasts
and group-level permutations; correct across the predeclared association family.
Diagnostic/drone runs must match graph, dynamics, interface and plasticity settings;
weights and task data are independent. Correlation does not establish causality;
state/edge interventions supply complementary evidence. Too few independent
substrates or incomplete contrasts must be reported as unidentifiable.

Paper targets: substrate × diagnostic and substrate × drone heatmaps, adaptation
frontiers, delay/dropout/fault severity curves, and held-out diagnostic-to-control
prediction with task-specific contrasts. Missing results stay missing. The
original biological compatibility analysis remains separate; the drone tests
transferable functional priors.
