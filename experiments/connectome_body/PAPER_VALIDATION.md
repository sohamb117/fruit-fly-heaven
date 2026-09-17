# Expanded-study implementation status — 2026-09-16

The authoritative target is [experiments/GOAL.md](../GOAL.md). This report concerns
its ten-experiment, four-connectome, three-body study. No scientific training
conditions from the expanded study have completed, so no biological advantage
or compatibility result is established.

## Implemented scope

- All seven adapter families, including the separate anatomical oracle; generic
  encoders and decoders are stateless.
- All five plasticity regimes with fixed adjacency and trainable magnitudes only
  on permitted edges.
- Real, degree-rewired, community-rewired, direction-shuffled and matched random
  graphs; parameter-matched RNN/GRU, empirical compute-match preparation and
  adapter-only controls.
- All ten native task interfaces: four fly, three worm and three fish tasks.
- Checkpointed recurrent PPO, post-training causal interventions, robustness and
  related-task transfer, censored adaptation frontiers, crossed-body interactions
  and held-out-whole-pair prediction.

The [implementation map](PAPER_EXPERIMENTS.md) connects each GOAL experiment to
its modules and outputs. [PAPER_README.md](PAPER_README.md) contains preparation,
selection and execution commands; [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) specifies
the estimands, controls and limitations.

## Biological inputs

BANC, MaleCNS and Cook 2019 chemical C. elegans graphs are prepared, with pinned
fingerprints, anatomical controls and task-associated selections. Fish1 access
is configured, materialization v700 is pinned, and all 29,474,316 physical
synapses have been downloaded. The separate label-table export and final
graph import are still running.

The Fish1 exporter verifies complete counted signed-ID intervals, unique
annotation IDs and page checksums. Its selected population consists of nonzero
segmentation roots with exactly one soma annotation, retaining unknown molecular
types and disconnected selected roots. This is not a claim that every selected
neuron is fully proofread. Labels audit signs; unique physical synapses determine
connection weights.

Fish1 anatomical oracle and task-module conditions require provenance-backed
sensory/motor or functional annotations. E/I labels do not provide that evidence.

## Validation and execution

The current full suite passed **229 tests, zero failures and zero skips**, in
43.81 seconds. This includes unordered Fish1 pagination, interrupted export/resume,
lossless IDs and control retention under capacity filters. Ruff passes. The one
warning is PyTorch's sparse-CSR beta notice. [Test log](validation/paper-goal-study-20260916/pytest.log),
[JUnit record](validation/paper-goal-study-20260916/tests.xml).

Native worm and fish mechanical qualification already passed; their evidence is
retained in [worm qualification](data/paper-bodies/worm/qualification-v1.json)
and [fish qualification](data/paper-bodies/fish/qualification-v1.json). These
checks establish modeled control authority and finite mechanics, not biological
fidelity or successful learned policies. The fish remains the documented
simZFish-derived 3D extension selected for this study.

The expanded selections cross all four connectomes with fly hover, worm
locomotion and fish swimming. They include architecture, capacity and
initialization/plasticity comparisons and retain explicit RNN/GRU/no-brain
matching references. A new immutable plan will incorporate the completed Fish1
graph. A condition catalog or selection does not launch training.

Target-GPU throughput and compute-matched RNN widths still require measurement
on the execution hardware. The Docker recipe has not been built or CUDA-tested.
The local CPU preflight passed and the unavailable-CUDA probe failed explicitly
within its deadline. No new cloud resource was provisioned or started.

## Preserved earlier evidence

The [earlier validation snapshot](validation/paper-implementation-20260916/PAPER_VALIDATION.md),
its source archive, `paper-plan-v1`, old BANC-only selections, benchmarks and
downloaded checkpoints remain available for provenance. They are not current
expanded-study results, and the old BANC milestone is not being repeated.
