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

The resulting central conclusion would be:

> Biological connectomes are not merely viable recurrent controllers. Their evolved structure provides a measurable, body-dependent inductive bias that reduces both the interface complexity and learning complexity required for embodied control.
