# Implementation map for the ten experiments

The authoritative scientific requirements are in [experiments/GOAL.md](../GOAL.md).
Recheck that file for every implementation and execution decision. The frozen
measurement rules and limitations are in [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md);
commands are in [PAPER_README.md](PAPER_README.md). All modules below live in
`connectome_body/compatibility/`. A complete catalog covers all experiments, but
its missing-input records and unmeasured reports are not completed trials.

| Experiment | Implemented comparison and output | Main modules |
|---|---|---|
| 1. Adapter architecture | Linear/linear, MLP/linear, linear/MLP, MLP/MLP, low-rank and sparse structured interfaces; anatomical mappings reported separately. Exact active parameter counts and capacity/Pareto frontiers. | `adapters.py`, `controller.py`, `design.py`, `analysis.py` |
| 2. Interface dependence | Architecture × connectome × body/task interaction; smallest successful tested width, depth, rank and parameter count. Unsupported or unidentified cells remain explicit. | `study.py`, `paper_analysis.py`, `analysis.py` |
| 3. Artificial recurrence | Total-trainable-parameter-matched RNN and GRU, measured-training-step-matched RNN, sparse N-and-M-matched reservoir and stateless adapter-only controls. Match records retain residual parameter/state differences. | `design.py`, `benchmark.py`, `controller.py`, `paper_analysis.py` |
| 4. Wiring specificity | Real, weighted-degree-preserving, community-preserving, direction-shuffled and N-and-M-matched random graphs. Null audits include edge overlap, swap acceptance and preserved statistics. | `topology.py`, `study.py`, `paper_analysis.py` |
| 5. Initialization/plasticity | Biological magnitudes, sign-preserving random magnitudes, shuffled magnitudes and randomized signed initialization, crossed with five plasticity regimes on real and rewired adjacency. | `config.py`, `dynamics.py`, `controller.py`, `paper_analysis.py` |
| 6. Matched/mismatched bodies | Full biological connectome × body matrix; main-effect-adjusted native-pair interaction, censored experience/capacity frontiers and native-pair interaction in real-minus-null performance. | `study.py`, `analysis.py`, `paper_analysis.py` |
| 7. Task specialization | Whole graph, declared task-associated subgraph and equal-size random-node selection with randomized wiring controls; optional irrelevant anatomy requires explicit annotation. | `anatomy.py`, `subgraphs.py`, `study.py`, `paper_analysis.py` |
| 8. Causal use | Zero/mean activity, causal time resampling, neuron permutation, state reset, recurrent-edge removal, task-associated lesions and acute matched-random replacement; no retraining. | `interventions.py`, `evaluation.py`, `runner.py` |
| 9. Robustness/transfer | Held-out targets, sensor noise/delay, mass/inertia, drag/friction, weak/failed actuators, terrain/current shifts; related-task zero-shot transfer, bounded fine-tuning and equal-new-experience scratch controls. | `robustness.py`, `bodies.py`, `runner.py`, `study.py` |
| 10. Predicting compatibility | Graph, anatomical, interface and body descriptors; nested held-out-whole-pair prediction against a size/capacity/body-difficulty baseline. Missing annotations are not fabricated. | `predictors.py`, `runner.py`, `paper_analysis.py` |

## Shared controller and learner

The generic observation encoder and action decoder have no recurrent state.
Learned structural queries address a fixed number of disjoint input/output
ports, so their trainable count does not grow with neuron count. The only actor
memory is the substrate state. Anatomical oracles use separately supplied,
provenance-backed groups and cannot enter the generic-interface result.

The sparse substrate uses one leaky-tanh rule, deterministic gain normalization
and fixed adjacency. The five plasticity regimes train adapters, permitted-edge
weights, encoder only, decoder only, or adapters and permitted-edge weights
jointly. Nonedges never become trainable parameters. Common recurrent PPO,
observation/action semantics, rollout budgets, evaluation seeds and checkpoint
selection apply to all controller conditions. The critic is separate from the
deployed actor and has the same architecture in every condition.

Graph/control matching is deliberately explicit: no-brain tests interface
capacity, fixed sparse reservoirs test generic recurrence, learned RNN/GRU
controllers test a conventional learned architecture, and graph nulls test
which wiring statistics matter. No single control simultaneously matches every
dimension; analysis retains the matching dimension for each comparison.

## Bodies and tasks

| Body | Tasks | Backend and qualification |
|---|---|---|
| Fly | Hover, controlled flight, walking, articulated limb control | Pinned FlyBody/MuJoCo. Flight uses the same wing-pattern generator for every controller. |
| Worm | Locomotion, steering, posture stabilization | Published 50-segment mechanical solver with a native bridge and 48 differential muscle controls. Direction, steering and posture probes are saved independently of learning. |
| Fish | Swimming, heading control, depth stabilization | Pinned simZFish morphology in a documented 3D MuJoCo extension with six tail and four fin actuators. Buoyancy and fluid forces are physical-model terms; targets never inject thrust. |

The worm and fish qualification files verify finite dynamics and mechanical
control authority before primary training is enabled. They do not validate a
biological nervous-system model or establish a learning advantage. The fish
extension's fins, fluid coefficients, underwater condition and small audited
inertia repair are modeling assumptions, not features validated by the original
published simulator.

## Inputs and execution boundaries

Prepared primary graphs are BANC, MaleCNS and Cook 2019 chemical C. elegans.
Fish1's complete v700 export and soma-root import are local, but their coverage
audit blocks primary whole-connectome interpretation: 0.7994% of raw synapses
remain between selected roots, with 60.4053% isolated roots. The common interface
and gradients work on this graph across all three bodies; that does not resolve
the reconstruction confound. A separate 197-cell, binary HMI circuit is available
for a declared secondary comparison. [PAPER_DATA.md](PAPER_DATA.md) records the
source checks and scope. Anatomical descriptors and task-module files have
independent provenance and checksums.

The planner writes immutable conditions and dependency jobs. The worker requires
an explicit run limit and can use a plan-pinned selection; it never provisions
cloud resources. Current selections cover the crossed-body core, architecture,
capacity and five-regime initialization experiments. Matching references keep
RNN/GRU/no-brain controls attached to their biological comparisons. Selection
reports also count development/test simulation, which can cost more than training.

Atomic checkpoints include actor, critic, optimizer, neural state, body state,
random-number streams, scenario counters and evaluation selection. Exact resume
requires identical code, data, configuration and environment. Numerical failure
is retained as a declared unusable condition; I/O or device failures stay pending.

## Evidence and figures

Each completed run yields learning curves with environment interactions,
optimizer updates and measured training wall time. Analysis computes censored
experience-to-threshold, actual-capacity frontiers, paired effects and seed-cluster
intervals. Reports expose missing controls and unidentified interactions instead
of filling gaps with estimates. Native-pair models require a complete crossed
matrix; structural prediction holds out whole connectome–body pairs at both
tuning and test time.

Standalone PNG/PDF figures include capacity curves, real-minus-null topology
advantage, cross-substrate gaps, learned-controller comparisons, causal losses,
robustness losses and adjusted compatibility matrices. Software-fixture figures
are visibly marked and cannot be read as experimental evidence.

Full-BANC local CPU benchmarks validate sparse forward/backward execution,
including gradients through all 13,542,180 permitted edges. They establish
feasibility only. CUDA calibration and substantive embodied training remain
separate execution steps; no proposed biological advantage has been established
by preparing this suite.

## Additional drone and temporal paper targets

The latest [GOAL expansion](../GOAL.md#additional-benchmark-d-drone-control-repertoire)
adds an engineered drone alongside the biological bodies, a full temporal
capability battery and a diagnostic-to-control model. [PAPER_BATTERY.md](PAPER_BATTERY.md)
records exact model assumptions, metrics, paper targets and bounded commands.

| Addition | Implementation |
| --- | --- |
| 20 drone tasks and severity grids | `drone.py`, common `bodies.py`/`training.py` contract |
| 25 temporal variants across 24 requested families | `temporal.py`, immutable split-separated data |
| Checkpointed diagnostic TBPTT with small linear readout | `temporal_training.py`, same `Controller` and exact parameter reports |
| Biological/rewired/random/RNN/GRU/no-brain matrices | `battery.py`, shared execution IDs prevent duplicate control training |
| Zero-shot severity and acute causal evaluations | `battery_evaluation.py`, frozen selected policy and paired scenarios |
| Capability heatmaps, censored grid frontiers and paired effects | `battery_analysis.py` |
| Source-group-held-out prediction and specificity | Nested group CV, balanced source permutations, eight-hypothesis Holm correction |

The native-drone match indicator is undefined and never enters the biological
compatibility diagonal. Diagnostic correlations are mechanistic hypotheses;
acute lesions provide separate causal evidence. Missing runs are not scored.

## Training regime interaction and staged core

The first execution pass now includes the [120-cell BC/PPO interaction](PAPER_LEARNING_CORE.md)
and a five-seed, three-connectome × two-body BC→PPO core. Main figures are
real-minus-rewired and BANC-minus-GRU gaps under each regime, their seed-paired
changes relative to PPO-only, expert action error versus post-BC closed-loop
performance, and sample/compute frontiers including pretraining cost. Only a
short, source-pinned GPU qualification is authorized before the full campaign.
