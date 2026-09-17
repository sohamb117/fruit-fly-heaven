# Connectome–body compatibility experiments

The authoritative target is [experiments/GOAL.md](../GOAL.md). Check it before
choosing work, freezing a study, launching conditions or interpreting results.
`PAPER_PLAN.md` is an archived copy, not the source of truth.

The implementation of the ten-experiment study is under
`connectome_body/compatibility/`. Run it with
`uv run --no-sync python -m connectome_body.compatibility` from this directory.
It has a separate CLI and does not replace the original FlyBody imitation or
`adaptation/` campaigns. Current work concerns the crossed connectome-by-body
study and its interface and plasticity comparisons. Earlier BANC-only work is
historical evidence, not a prerequisite to repeat.

The suite implements stateless sensory/motor adapters, frozen or permitted-edge
plastic connectomes, parameter-matched RNN/GRU controls, empirically matched RNN
controls, adapter-only controls, all ten tasks, post-training interventions,
transfer, and held-out-pair analysis. A compiled catalog is not a scientific
result or an instruction to run every condition.

## Start here

1. Read [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) for the fixed estimands,
   normalization, matching rules and model limitations.
2. Read [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for the implementation map.
3. Inspect the current [study configuration](configs/paper/study.json).
4. Prepare inputs and compile an immutable plan; inspect `coverage.json` and
   `prerequisites.json` before launching a bounded worker.

The shared environment already has Python 3.12, PyTorch 2.8.0, MuJoCo 3.13.0 and
the dependencies pinned in `pyproject.toml`/`uv.lock`. Do not synchronize or alter
an environment while an existing campaign is using it. On a fresh checkout,
install `uv`, Python 3.12 and a C++17 compiler, then use `uv sync --frozen` before
starting work. Subsequent commands below use `--no-sync`. The paper's sparse
backend supports CPU and CUDA; it explicitly rejects MPS. CUDA performance
must be measured on the actual target hardware.

On an existing GPU machine, check initialization and both frozen/plastic sparse
gradients before preparing a long run:

```sh
uv run --no-sync python scripts/paper_preflight.py --device cuda --timeout 60 --output data/paper-calibration/runtime-preflight.json
```

The probe runs in a child process with a deadline and records an explicit
failure if CUDA hangs or is unavailable. `--device cpu` checks a local CPU;
`--expected-gpu 'RTX 4090'` additionally checks an intended allocation. This tiny
probe is separate from the full-connectome throughput benchmark below.

`Dockerfile.paper` packages the pinned Python environment, native C++ compiler
and paper CLI for an already allocated Linux host. Build with
`docker build -f Dockerfile.paper -t connectome-paper .`. Bind persistent
directories at `/bench/data`, `/bench/runs`, `/bench/references` and
`/bench/reports`, and enable the NVIDIA runtime for CUDA. Run preparation and
compile a new plan **inside that machine** so paths, native binaries and device
calibrations match. The image does not include data or credentials and its
default command only displays help. The Docker recipe requires a Linux build
and CUDA validation before it can be considered deployment-tested.

The pinned upstream FlyBody source and flight assets use the existing project
preparation commands:

```sh
uv run --no-sync python -m connectome_body.cli bootstrap
uv run --no-sync python -m connectome_body.adaptation.cli assets --download
uv run --no-sync python -m connectome_body.cli prepare banc --download
uv run --no-sync python -m connectome_body.cli prepare malecns --download
```

The flight assets supply the common wing-pattern generator. The native paper
controller does not need teacher actions: it uses recurrent PPO across all bodies.

## Biological and physical inputs

Prepared local graph directories are:

| Graph | Directory | Units | Directed edges | Current role |
|---|---|---:|---:|---|
| BANC v888 | `data/graphs/banc` | 175,401 neurons | 13,542,180 | Primary |
| MaleCNS | `data/graphs/malecns` | 165,122 neurons | 25,563,096 | Primary |
| Cook 2019 chemical hermaphrodite | `data/graphs/celegans-cook2019-chemical` | 302 neurons | 3,671 | Primary |
| Fish1 v700 soma-root candidate | `data/graphs/fish1` | 178,976 soma roots | 157,014 | Primary use blocked by reconstruction coverage |
| Fish1 curated HMI circuit | `data/graphs/fish1-hmi-binary` | 197 source-selected cells | 293 | Separate secondary option; binary adjacency |

Graph fingerprints and release membership are in `configs/paper/study.json` and
each graph's manifest. Cook edge weights are EM serial-section extent, not
synapse counts. The default magnitude transform is therefore a common numerical
rule, not a claim that the source weights have equivalent physical units.

The complete Fish1 export is local, but the soma-root graph retains only 235,608
of 29,474,316 physical synapses (0.7994%) and has 108,111 isolated roots (60.4053%).
An independent uint64 endpoint audit agrees with the importer, and six sampled
incoming/outgoing counts match CAVE exactly. The original [Fish1 paper](https://vcg.seas.harvard.edu/publications/20250615-zbrain/paper)
describes a fragmented automatic segmentation dominated by soma-attached dendrites.
See [PAPER_DATA.md](PAPER_DATA.md) for the observed coverage and interpretation.
The planner records a failed graph-use qualification for the primary Fish1 arm;
having a valid sparse graph file is insufficient to establish a comparable whole
connectome. The file remains available for explicitly scoped exploratory use.

Prepare the two additional bodies and their mechanical checks:

```sh
uv run --no-sync python -m connectome_body.compatibility prepare-worm-body
uv run --no-sync python -m connectome_body.compatibility prepare-fish-body
uv run --no-sync python -m connectome_body.compatibility qualify-body --body worm --manifest data/paper-bodies/worm/manifest.json --output data/paper-bodies/worm/qualification-v1.json
uv run --no-sync python -m connectome_body.compatibility qualify-body --body fish --manifest data/paper-bodies/fish/manifest.json --output data/paper-bodies/fish/qualification-v1.json
```

The worm executes the pinned published 50-segment mechanical solver with 48 muscle
commands; its original neural circuit is excluded. The fish is an explicitly
documented 3D extension of simZFish's collision morphology, masses and inertias.
It uses MuJoCo hydrodynamics, hydrostatic buoyancy and four added pectoral-fin
degrees of freedom. It is **not numerically equivalent to published simZFish or
an empirically validated zebrafish digital twin**. Its source, assumptions and
one small inertia feasibility correction are retained in its manifest.

The mechanical qualification checks actual actuator-driven propulsion in both
directions, steering, loss of propulsion after actuator failure, worm posture
tracking and fish vertical actuation. It is separate from learned-policy
evaluation. The planner requires a current passing qualification for these
native bodies. A changed body implementation requires a new qualification path
and a new plan; keep old evidence intact.

Prepare Cook data on a new machine:

```sh
uv run --no-sync python -m connectome_body.compatibility prepare-celegans
```

Prepare structural communities once per graph. For example:

```sh
uv run --no-sync python -m connectome_body.compatibility communities --graph data/graphs/banc --output data/paper-annotations/banc/communities.json
```

`prepare-anatomy` reads release-pinned BANC/MaleCNS annotation feathers or the
Cook importer's `cell-types.json`. These privileged annotations are used only
for anatomical oracles, anatomical descriptors and declared subgraph hypotheses.
They are never fed into a generic adapter. Example:

```sh
uv run --no-sync python -m connectome_body.compatibility prepare-anatomy --kind celegans --graph data/graphs/celegans-cook2019-chemical --annotations data/graphs/celegans-cook2019-chemical/cell-types.json --output data/paper-annotations/celegans --subgraphs-output data/paper-subgraphs/celegans
```

The current BANC, MaleCNS and Cook preparations include anatomical ports and ten
task-associated module selections each. Cross-species task assignments are
preregistered functional hypotheses, not neuron homologies or established task
specificity. Inspect each selection's provenance before interpreting Experiment 7.

## Fish1 access

Follow the [official Fish1 programmatic access instructions](https://fish1-release.storage.googleapis.com/programmatic.html)
and configure CAVE authentication locally. Do not put a token in a run
configuration, command-line argument, repository or report. Choose and record
an exact materialization version, membership rule and reconstruction coverage.
The documentation's example version is not automatically the study version.

```sh
uv run --no-sync --with-requirements scripts/fish1-export-requirements.txt python -m connectome_body.compatibility export-fish1 --version MATERIALIZATION_INTEGER --coverage 'Describe the selected reconstruction and proofreading coverage' --membership single_soma_roots --output data/raw/fish1-versioned-export
uv run --no-sync python -m connectome_body.compatibility import-fish1 --export data/raw/fish1-versioned-export --output data/graphs/fish1
```

The exporter verifies the live segmentation endpoint, uses the fixed
materialization, preserves uint64 root IDs and signed int64 annotation IDs
losslessly, checks page counts and duplicate IDs, and resumes from checksummed
Parquet pages. It partitions signed annotation-ID ranges until each query
returns its complete counted interval. [CAVE does not guarantee row order](https://www.caveconnecto.me/CAVEclient/tutorials/materialization/),
so offsets and sorted-looking previews cannot establish complete coverage.
Cached pages are reused only after unique-ID and complete-prefix count checks.
Physical synapses and reference labels are exported separately:
connection weights count physical synapses even when labels are missing or
duplicated. Counts are requested from unjoined tables because the Fish1 service
does not reliably count reference-table joins. The importer excludes zero
and multiple-soma roots, preserves disconnected selected neurons and reports
annotation coverage. It never remaps roots to a newer segmentation.

After import, pin the resulting graph fingerprint in a new study configuration
and review neuronal membership and reconstruction coverage.
Prepare communities and provenance-backed anatomical/functional selections where
annotations support them. The generic interface needs none of these labels.
Missing Fish1 oracle or task-module inputs remain explicit prerequisites; the
suite does not invent sensory/motor identities from excitatory/inhibitory labels.

Record the structural measurements used by GOAL Experiment 10 before training:

```sh
uv run --no-sync python -m connectome_body.compatibility graph-features --graph data/graphs/fish1 --communities data/paper-annotations/fish1/communities.json --output data/paper-annotations/fish1/structural-features-v1.json
```

For BANC, MaleCNS and Cook, also supply the corresponding `--anatomy` file.
The measurement includes degree/strength distributions, directed modularity,
strong components, sampled paths and conditional motifs. Sensory/motor paths
and bilateral correspondence remain unknown when annotations are absent. The
prediction jobs recompute these descriptors from the inputs pinned in the plan.
The current MaleCNS and Cook graph imports have no annotated E/I signs; their
`available`-sign conditions therefore use the declared imputation rule entirely.
They cannot establish an advantage from biological signs.

The published HMI circuit is an additional, smaller input:

```sh
uv run --no-sync python -m connectome_body.compatibility prepare-fish1-hmi
```

It selects the 197 cells with the two declared axon/dendrite reconstruction
labels in the pinned source workbook, retains selected isolates and uses the
union of directed input/output annotations. Weights are binary because those
records do not supply shared physical-synapse IDs with which to deduplicate
multiplicity. This is a circuit-level option, not a silent substitute for the
whole-volume Fish1 row in the primary study. A circuit comparison needs its own
declared population and size/coverage controls.

## Compile the current study and select an expanded experiment

```sh
uv run --no-sync python -m connectome_body.compatibility plan --study configs/paper/study.json --output runs/paper-plan-v2
uv run --no-sync python -m connectome_body.compatibility status --plan runs/paper-plan-v2/plan.json
```

The plan includes all ten experiments, exact parameter counts, input checksums,
body dimensions, immutable run configurations, shared-baseline references and
post-training dependencies. Unsupported capacities are recorded as infeasible,
not padded with unused parameters. Missing inputs block their conditions rather
than silently selecting smaller or synthetic substitutes.

The full catalog has thousands of training configurations. The default
million-interaction learning budget and repeated evaluations require staged
execution. Current selections cross four connectomes with fly hover, worm
locomotion and fish swimming:

- `cross_body_core.json`: 20k interfaces, real/rewired/random topology, frozen
  versus jointly trained permitted weights.
- `cross_body_capacity.json`: the same crossed comparison at 5k/20k/80k.
- `cross_body_architectures.json`: all seven interface families at 20k.
- `plasticity_initialization.json`: all five plasticity regimes and declared
  initializations, using the measured-sign sensitivity convention.

Selections follow explicit matching references to retain RNN, GRU, random and
adapter-only controls even when their internal adapter allocations differ.
Compute-matched controls retain their hardware-calibration prerequisites.
Inspect selected statuses and interaction totals before execution. The old
`banc_hover_mvp.json` is retained for historical reproduction only.

```sh
uv run --no-sync python -m connectome_body.compatibility select --plan runs/paper-plan-v2/plan.json --filters configs/paper/cross_body_core.json --output runs/paper-plan-v2/cross-body-core.json
uv run --no-sync python -m connectome_body.compatibility select --plan runs/paper-plan-v2/plan.json --filters configs/paper/cross_body_architectures.json --output runs/paper-plan-v2/cross-body-architectures.json
```

The selection includes both training interactions and an upper bound on all
development/test interactions. To execute at most one selected condition for a
bounded interval on the current machine:

```sh
uv run --no-sync python -m connectome_body.compatibility worker --plan runs/paper-plan-v2/plan.json --condition CONDITION_ID_FROM_SELECTION --max-runs 1 --max-seconds 1800
```

For a ready condition ID from the plan:

```sh
uv run --no-sync python -m connectome_body.compatibility benchmark --plan runs/paper-plan-v2/plan.json --condition CONDITION_ID --output data/paper-calibration/benchmark.json --batches 1 4 16 --lengths 8 16 --repeats 5 --warmup 2
uv run --no-sync python -m connectome_body.compatibility worker --plan runs/paper-plan-v2/plan.json --condition CONDITION_ID --max-runs 1 --max-seconds 1800
```

Workers run only on the current machine or an already allocated machine. No
command rents, starts or stops a pod. `--max-seconds` pauses at a saved rollout
boundary; allow for one rollout, optimization and scheduled evaluation beyond
that time. It is not a provider billing cutoff. Repeating the worker command
resumes its atomic checkpoint using the same plan and environment.

The benchmark measures sparse forward transitions, TBPTT/Adam, peak CUDA memory,
batch and sequence scaling, and a common critic. Its regression objective is a
compute surrogate; it is not an embodied-learning result. It excludes physics,
so end-to-end throughput must also be measured from training logs.

To prepare the compute-matched RNN control on the target GPU:

```sh
uv run --no-sync python -m connectome_body.compatibility compute-match --plan runs/paper-plan-v2/plan.json --condition CONDITION_ID --hidden-sizes 32 64 128 256 512 1024 --output data/paper-calibration/compute-matches.json
```

Only a measured width within the declared tolerance and timing stability is
accepted. A miss remains `no_match`; expand the preregistered width search without
looking at rewards. Recompile to a new plan after adding calibrations. Results
from a different device, precision, batch, sequence length or source revision
are not accepted as compute matches.

## Results and checkpoint safety

Each training directory contains `manifest.json`, `latest.pt`, `best.pt`,
`learning_curve.json`, update logs and a terminal result or explicit failure
record. A checkpoint retains actor, critic, optimizer, neural/physical state,
RNGs, scenario counters and evaluation selection state. Source data and body
assets remain external, checksummed dependencies. Copy the run directory and
its referenced inputs before discarding an allocated machine.

`best.pt` is selected using development episodes only. Post-training jobs use
that exact policy, common held-out episode seeds and no further learning except
explicitly labeled fine-tuning. Run bounded jobs, then generate the report:

```sh
uv run --no-sync python -m connectome_body.compatibility jobs --plan runs/paper-plan-v2/plan.json --max-jobs 8 --experiment 8
uv run --no-sync python -m connectome_body.compatibility report --plan runs/paper-plan-v2/plan.json --output reports/paper-v2
```

Unfinished conditions, missing controls, numerical failures and unidentifiable
effects remain visible. An I/O or device failure is not counted as evidence
against a connectome. Native-body software tests and short sparse benchmarks
establish implementation behavior, not any of the paper's proposed advantages.

## Validation

```sh
uv run --no-sync ruff check connectome_body/compatibility tests/test_compatibility_*.py
uv run --no-sync pytest -q tests/test_compatibility_*.py
```

Native tests require the pinned local bodies and FlyBody assets. Skips must be
reported; a fixture-only test pass does not certify a missing native backend.
Use the saved validation report for the exact environment, checks and remaining
execution prerequisites.

## Drone and temporal benchmark expansion

[GOAL.md](../GOAL.md) now includes all twenty drone tasks, the full temporal
battery, and held-out capability-to-control predictions. Use
[PAPER_BATTERY.md](PAPER_BATTERY.md) for the complete runbook and model/measurement
semantics. The entry point is `python -m connectome_body.compatibility.battery`;
`configs/paper/battery.json` defines the matrix. It uses the existing controller,
nulls, parameter accounting and checkpoint infrastructure. Compilation and data
preparation are local; workers require explicit run/time bounds and never create
cloud resources. The earlier biological plans remain archived snapshots; source
changes require freshly compiled plans before execution.
