# Measurement protocol and interpretation

The authoritative [experimental target](../GOAL.md) defines ten biological
experiments and the additional drone, temporal and diagnostic-to-control
benchmarks. Its alternative
hypotheses and ideal outcomes are predictions, not acceptance conditions for the
software or conclusions supported by the current checks.

## Unit of comparison

Measure `J(C, B, P, N, task)` on frozen evaluation policies. Here `P` is the
**actual** trainable interface size and `N` is the number of training environment
decisions, summed over parallel environments. All methods controlling the same
body/task receive the same observations, actions, physics, rewards, scenario
distribution, optimizer protocol and experience ceiling.

The primary adapter/joint regimes train the entire allocated interface. In
encoder-only, decoder-only and fixed-interface regimes, record allocated and
trainable interface counts separately; zero trainable interface weights do not
mean zero interface architecture capacity.

The main outcome is experience to 80% development success, with three consecutive
evaluation points required for confirmation. Report both the first crossing and
the later certification time. A never-reached threshold is right-censored at the
budget, not assigned an invented crossing. Evaluation interactions are separately
counted and included in total environment use. They can exceed training
interactions and must be included in compute estimates.

Secondary outcomes are normalized score/success AUC, held-out final and selected
policy scores, success probability, optimizer steps and training wall time to
threshold, robustness, and minimum successful interface size/width/depth/rank.
Capacity frontiers retain unsuccessful capacities and do not assume performance
is monotone or architectures are nested.

Decision intervals differ across bodies. Direct experience counts are primarily
within-body comparisons; retain physical time and control interval in structural
descriptors and model body/task effects in crossed analyses. A single global
average reward is not the compatibility estimand.

## Controller contract

```
observation -> stateless E -> K inputs -> fixed/learned C -> K readouts -> stateless D -> action
```

Only the neural substrate holds controller recurrence. Generic adapters contain
no recurrent layers, observation histories, phase counters or direct
observation-to-decoder connections. Physical state, actuator activation and
delayed/noisy sensor state belong to the body/environment and are checkpointed.

Generic ports use twelve within-graph structural features: log in/out degree
and strength, reciprocal fraction, PageRank, neighbor degree summaries,
two-step return probability and isolation. The feature list and preprocessing
are fixed across substrates. Two learned `K x 12` query arrays select top-support
populations. Queries, E and D are counted in the adapter budget; there are no
learned parameters proportional to neuron count in the generic interface.
Input and output candidate populations are disjoint seeded halves. Sparse
attention weights are differentiable within the selected support; support
indices may change between updates. L2-normalized ports keep population size
from trivially changing signal scale.

All generic families use the same ports: linear/linear, MLP/linear, linear/MLP,
MLP/MLP, low-rank and sparse structured. Sparse maps train only their permitted
coefficients. Low-rank maps train factor matrices. Anatomical mappings are a
separate fixed-population oracle with explicit provenance and no claim of
cross-species neuron homology. No unused coefficients are added to fill a budget.
The planner records allocation error and merges ceilings that create the exact
same controller.

For `W[dst,src]`, define positive permitted-edge magnitudes `m`, incoming strengths
`d_in` and outgoing strengths `d_out`:

```
W_hat[dst,src] = sign[src] * m[dst,src] / sqrt(d_in[dst] * d_out[src])
alpha = 1 - exp(-control_dt / (neural_substeps * tau_seconds))
h <- (1-alpha) h + alpha tanh(gain * W_hat h + input_gain * neural_input)
```

This is a deterministic strength normalization with an operator-norm bound, not
an estimate that every graph has identical spectral radius or memory capacity.
The default gain is 0.9, time constant 20 ms and two neural substeps per decision.
Graph-dependent dynamics remain part of the measured prior. A null result under
this conservative rate rule does not establish that biological circuitry is
useless under other dynamical rules.

Frozen conditions never change biological recurrent weights. Plastic conditions
learn one softplus magnitude per existing edge, keep source signs fixed, and
recompute the same differentiable normalization. No new edges are introduced.
The five regimes enable exactly the adapters, permitted edges, encoder, decoder,
or adapters plus permitted edges. Joint plasticity can have millions more learned
weights than the frozen condition; compare it with corresponding matched controls
and report both adapter and total actor parameters.

## Graph controls and sign uncertainty

Real, community-rewired, degree-rewired, direction-shuffled and N/M-matched random
graphs share neuron dynamics, observation/action interfaces and training rules.
The main degree/community nulls preserve source signs, in/out degree, incoming
signed degree and global weights. With `preserve_null_strengths: true`, swaps are
also stratified by exact source weight: each neuron's in/out strength is retained.
Community swaps additionally retain each neuron's degrees to each community.

The swap budget is ten attempts per edge. Each run records accepted swaps and
the fraction of destinations changed. Finite swap chains are not assumed to
mix uniformly. A null with little change is a weak intervention and must be
reported as such. Direction shuffling preserves the undirected skeleton and
reciprocal pairs, but not directed degree/strength. The random recurrent graph
matches neuron count, edge count, weight multiset, signs and gain rule; it is
sparse, not a dense RNN.

Default topology comparisons use a common seeded 20% inhibitory source assignment
because annotation coverage differs sharply across releases. Experiment 5
separately uses available annotations and explicitly imputes unknown signs,
crossing native magnitudes, randomized magnitudes, shuffled magnitudes and random
source identities with all five plasticity regimes and real/rewired topology.
Randomized source signs retain the original overall E/I count. Insect
transmitter-to-sign assumptions and missing physiological information remain
limitations. Cook chemical weights and fly synapse counts are different source
measurements. Neither is treated as a directly measured conductance.

Conventional RNN/GRU controls match **total trainable actor parameters**, not
biological neuron count. A common allocation rule leaves at least half the
ceiling for recurrence where feasible. Their recurrence is the standard learned
cell, not the rate-connectome rule. The N/M-matched reservoir is the control that
holds state dimension and sparse connectivity budget fixed. These controls answer
different questions; no single dense RNN can generally match parameters, neurons,
sparsity and compute simultaneously.

Compute matching uses synchronized measured policy collection plus TBPTT/Adam
cost, a common critic and an explicitly labeled regression surrogate. The
default matching tolerance is 20%, with timing CV also required below tolerance.
Record actual end-to-end training and physics time separately. CPU measurements
cannot establish a CUDA match, and a FLOP estimate cannot replace a measurement.

## Bodies and task definitions

| Body | Tasks | Decision interval | Default horizon | Observations / actions |
|---|---|---:|---:|---:|
| FlyBody flight | hover, controlled flight | 0.2 ms | 5,000 | 104 / 12 |
| FlyBody ground | walking | 2 ms | 5,000 | 286 / 59 |
| FlyBody ground | limb control | 2 ms | 5,000 | 418 / 59 |
| Published worm mechanics | locomotion, steering, posture | 10 ms | 1,000 | 173 / 48 |
| simZFish-derived 3D extension | swimming, heading, depth | 5 ms | 1,000 | 34 / 10 |

Fly flight retains the shared wing-pattern actuator abstraction. Ground tasks
use native articulated actuators. No neural-substrate-specific motor mapping is
embedded in a body. Target and success definitions are fixed in `bodies.py` and
`worm_body.py` and pinned by the method/source identity.

The worm uses the mechanics accompanying the
[published neuromechanical model](https://pmc.ncbi.nlm.nih.gov/articles/PMC6158225/),
from [this pinned source repository](https://github.com/edizquierdo/RoyalSociety2018).
Its agar model is overdamped and planar: inertia, water current and terrain slope
are not meaningful interventions there. Friction changes alter the actual drag
constants. Muscle activation is filtered by the original mechanics. Steering
starts with a nonzero target turn; posture holds a nonzero prescribed bend.
No clock, oscillator or original neural controller is exposed to the actor.

The fish retains collision morphology from
[simZFish](https://ponyo.epfl.ch/proj/zebrafish/simzfish), associated with
[the 2025 embodied-circuit study](https://doi.org/10.1126/scirobotics.adv4408).
The released demonstration operates at the water surface. The paper extension
is fully submerged in an unbounded fluid and adds two pectoral fins, each with
sweep and pitch actuation. Buoyancy depends only on displaced volume, density,
gravity and pose, never target depth. All translation and turning must arise
from joint commands and fluid forces. Native length units are 1 cm; body length
is 0.4 native units (4 mm). Position/velocity tasks use the body's center of mass
and velocity averaged over the physical control interval; attitude uses the
head frame. Depth targets begin away from the initial depth, preventing passive
initialization from satisfying the task.

The fish extension's fluid model, fin dimensions and servo gains require future
empirical validation. Opposite commanded fins can move the body up and down in
the implemented mechanics; this engineering check is not a validation of living
zebrafish behavior. The morphology manifest records the source values, the added
assumptions and a minimal correction to one source inertia tensor. Do not label
results as native simZFish without these qualifications.

## Training, held-out data and failures

The first compute pass follows [PAPER_LEARNING_CORE.md](PAPER_LEARNING_CORE.md):
BC-only, PPO-only and BC→PPO in the 120-cell regime interaction, and BC→PPO in the
five-seed crossed-body core. BC uses one sealed external-expert dataset per task,
contiguous recurrent sequences and validation action error for checkpoint
selection. BC-only and BC→PPO share the exact selected policy. Experts receive the
student observation/action interface; they contain no experimental connectome.
Teacher qualification precedes the primary dataset and failed qualification
blocks that body's BC arms. Low action error alone is not control performance.

Every PPO stage uses the same recurrent PPO loop, stateless common critic,
fixed Gaussian exploration scale, GAE, clipping, gradient clipping and Adam.
The default four environments collect 128-step rollouts, optimize 16-step
sequences with 16-step burn-in, and use four epochs. Training stops at the exact
declared interaction count. BC→PPO resets optimizer/critic/episode state and
retains the entire PPO-only interaction budget. Report unique expert samples,
repeated exposures, teacher-generation interactions, BC and PPO optimizer steps,
and measured compute separately. Equal-compute comparisons include pretraining
cost and use only validation points observed before a shared cutoff. Scientific
evaluation is always closed-loop without teacher assistance. DAgger remains a
documented follow-up rather than an unimplemented fourth arm counted as executed.

Scenario seeds pair initial conditions, disturbances and targets across
conditions. Training, development, test and OOD streams have distinct namespaces.
PPO checkpoint selection uses development success then score; BC uses development
expert-action MSE and separately reports closed-loop development/test/OOD scores.
Held-out evaluation cannot influence adapter design, hyperparameters or model
selection. Related-task fine-tuning initializes the selected controller,
resets its optimizer, and counts new target-task experience separately from the
full source-training budget. Corresponding scratch controls use the same new-task
budget. Transfer requires identical named observation and actuator semantics.

Use BANC/FlyBody and synthetic tasks to settle interface and optimization choices.
Freeze code, feature family, normalization, task definitions, capacity grid,
hyperparameter-search rules and evaluation protocol before collecting MaleCNS or
Fish1 performance. Preparing their graphs and annotation files is not a performance
evaluation. A change after seeing held-out results defines a new study version;
retain the earlier version and label the new analysis exploratory.

Numerical nonfiniteness is an explicitly identified unusable condition and is
retained using the declared intention-to-treat zero-score convention. File,
dependency, device and resource errors remain unfinished execution problems;
they cannot silently become poor performance. A user interruption remains
resumable. Configuration/data/code drift prevents exact resume rather than
silently starting a different experiment.

## Causal tests, statistics and limits

Post-training tests include zero activity, development-only mean activity,
causal resampling of prior states, neuron permutations, per-step state reset,
removal of recurrent edges, task-associated lesions and acute matched-random
replacement without retraining. The causal time-shuffle uses past states only;
it is not a noncausal permutation containing future trial information. Acute
graph replacement keeps learned interfaces and the learned magnitude multiset,
then recomputes normalization. Report each intervention's scope separately.

Use training seeds as the replication/cluster unit, not evaluation episodes or
multiple null draws. The new learning core has five paired seeds; the archived
full catalog has three seeds and needs two more for headline comparisons. Report paired effects,
seed-cluster confidence intervals, curves and censored frontiers. With only
three seeds, inferential resolution is coarse. The compatibility coefficient
controls connectome and body/task main effects; architecture and task interactions
are estimated only when their factorial cells and replication identify them.

Also estimate the native-pair interaction in the paired difference between real
and matched-null performance. Pair by connectome, body, task, adapter family and
budget, bottleneck, initialization, sign rule, plasticity, experience budget and
training/null seed before fitting the crossed model. Report separate contrasts
for degree-preserving, community-preserving and N-and-M-matched nulls. This
distinguishes a native-pair advantage shared by generic recurrence from an excess
benefit of biological topology. Missing pairs remain explicitly unmeasured.

The structural prediction model holds out entire connectome–body pairs, including
all their seeds, tasks and capacities. Inner tuning also holds out whole pairs.
Scaling and missing-value imputation use training folds only. The baseline
controls size, body/task difficulty and trainable capacity. The structural model
adds graph/body descriptors and their interactions. There are only twelve coarse
biological pairs in the four-connectome/three-body matrix: treat predictive
claims and intervals accordingly. BANC and MaleCNS are same-species replication,
not independent species samples.

Biological advantage, native-pair advantage, adapter compensation and transfer
benefits are separate hypotheses. A working pipeline, a qualifying body, a small
synthetic test or low imitation error does not establish any of them. If the
adapter-only or matched random controls perform as well, report that result and
change the scientific conclusion rather than selecting favorable tasks after
looking at held-out performance.

## Engineered drone and temporal additions

The authoritative GOAL additions D/T/M are implemented and specified in
[PAPER_BATTERY.md](PAPER_BATTERY.md). They retain the common stateless controller,
fixed adjacency, explicit plasticity and exact parameter accounting. The drone
uses the common recurrent PPO learner; the temporal battery uses supervised
TBPTT with immutable shared streams and a primary linear readout. Training
experience units are reported separately, never pooled into one frontier.

The drone does not enter the biological native-pair contrast. Diagnostic-to-drone
prediction holds out whole source reconstructions and all their nulls/replicates,
uses training-fold preprocessing/tuning, and predicts task-relative performance
to remove general controller ability. The default catalog's three qualified
biological reconstructions are insufficient for the four-group minimum: this
analysis will explicitly remain unidentifiable until an additional appropriate
source and complete outcomes exist. Performance does not cure input qualification.
