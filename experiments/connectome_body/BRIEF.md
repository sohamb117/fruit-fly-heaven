# Generalized connectome–body adaptation

This protocol implements the hover-first experimental brief. It lives in `connectome_body/adaptation/` and is separate from the earlier ground-task/PPO protocol. It tests whether frozen biological topology reduces the trainable interface capacity and experience needed to control one fixed FlyBody embodiment. The architecture is generic; weights are fitted separately for every graph and seed.

The implementation, source data, and engineering evidence are available locally. Full BANC has passed all nine forward/backward benchmark cases on this Mac's **Apple M2 Pro GPU**, using a sparse Metal backend. The local MVP can use this GPU; CUDA is not required. The 24 scientific MVP runs have **not** been completed. [BRIEF_VALIDATION.md](BRIEF_VALIDATION.md) distinguishes completed checks from pending scientific results.

## Primary experimental object

Measure held-out performance `J(C, P, N)` across adapter ceilings and cumulative DAgger experience, conditional on a common offline teacher dataset. Record both success and horizon-normalized hover score. Derive the first tested validation threshold crossing `E_tau`, the smallest tested capacity reaching held-out mean success `P_tau`, and learning-curve area. Unsuccessful learners remain censored at the largest tested budget. There is no interpolation between grid points.

These are empirical frontiers under the fixed learner, not global minima over all adapter weights or possible training algorithms.

Report:

1. Real graph versus the brain-free adapter: useful frozen computation under a limited interface budget.
2. Real versus degree/strength-preserving rewiring: higher-order topology benefit.
3. Real versus a matched fixed sparse RNN: benefit beyond generic reservoir computation.
4. A conventional trainable GRU: a learned-controller reference under the same total trainable-parameter ceiling. It is not an established performance ceiling until adequately trained.
5. Biological excess benefit by connectome, `J(real) - J(rewired)`, and its differences across species.
6. Substrate gaps across increasing capacity and acute intervention effects.

With one body, cross-substrate results concern **FlyBody adaptation affinity**. They do not identify universal brain–body compatibility, phylogenetic effects, or the mechanism of biological motor control. Size, reconstruction completeness, developmental stage, assumed physiology, and compute remain possible explanations. A null/no-brain match is a substantive negative result; it does not by itself show statistical overfitting.

## Frozen interface specification

```text
104 standardized FlyBody observations
  → stateless Eθ: Linear → tanh → Linear
  → K=16 latent input channels
  → learned sparse structural input queries
  → frozen N-neuron rate dynamics, h
  → learned sparse structural output queries
  → K=16 readout channels
  → stateless Dφ: Linear → tanh → Linear → tanh
  → 12 normalized native flight actuator commands
```

Only `h` retains controller state. E/D have no recurrence, history window, residual observation-to-action path, per-neuron learned embeddings, or anatomical labels. The conventional GRU is an explicitly separate baseline. The fixed WPG and native actuator dynamics belong to the standardized body in every condition.

Each neuron receives 12 structural features: log input/output degree, log input/output strength, reciprocal outgoing fraction, log PageRank, four directional neighbor-degree averages, two-step random-walk return probability, and an isolate indicator. Within-graph standardization and clipping use no task data. PageRank runs a fixed 100 iterations and reports its residual. Features are recomputed on the actual randomized graph. Expensive clustering/k-core features are not part of this version.

Each of the K input/output queries has 12 learned coefficients: **384 trainable port parameters**, independent of N. Top-256 scoring neurons per query receive softmax weights normalized to unit L2 norm. Small graphs clamp support to half their neurons and require at least 2K neurons. One seeded random partition allocates disjoint input/output populations; every readout signal must traverse an edge. Ports may overlap within one population. Top-k selection is piecewise differentiable through selected scores. Tied structural features need not uniquely identify neurons.

For each neural substep:

```text
h_next = (1 - alpha) * h + alpha * tanh(gain * W_normalized @ h + input_gain * i)
alpha = 1 - exp(-control_dt / (substeps * tau))
```

Primary settings are gain 0.9, tau 2 ms, two substeps per 0.2 ms body decision, input gain 1, and log1p raw synapse counts. Directed source signs obey a common seeded 20% inhibitory Dale assignment for all graphs. This is an explicit modeling assumption, not measured biological physiology. Transmitter-derived signs are reserved for a separately identified sensitivity analysis because annotation coverage differs across datasets.

Normalization is `D_in^-1/2 T(A)^T D_out^-1/2 S`, with operator norm bounded by 1. This provides a common gain bound; it does **not** make every graph's spectral radius identical. All synaptic weights, graph features, and state dynamics are frozen. The sparse backward pass uses a cached CSR transpose.

CPU/CUDA use PyTorch sparse CSR. Apple GPUs use a custom FP32 CSR multiplication and its cached transpose through [PyTorch's public Metal shader API](https://docs.pytorch.org/docs/2.8/generated/torch.mps.compile_shader.html). The Metal implementation stores sparse arrays, never a dense N×N matrix, and has no CPU fallback. CPU/Metal state and input-gradient agreement is checked before GPU profiling. Training uses FP32 everywhere; validation MSE accumulation uses FP32 on MPS and FP64 on CPU/CUDA. Device/runtime is recorded in the method identity, so the primary comparison must keep it fixed.

All query and E/D weights count toward the ceiling. Width is the largest integer fitting it; unused capacity is reported, not padded with irrelevant parameters. The no-brain adapter directly connects E to D through the same 16-dimensional bottleneck and spends its entire ceiling on useful E/D weights. The GRU matches total trainable capacity, not neuron count.

| Ceiling | Graph + adapter | Adapter-only | Learned GRU |
|---:|---:|---:|---:|
| 5,000 | 4,912 | 4,978 | 4,809 |
| 20,000 | 19,912 | 19,978 | 19,749 |
| 80,000 | 79,912 | 79,978 | 79,689 |

These counts apply to the native 104-observation/12-action schema. Graph size changes frozen computation and memory, not trainable capacity. Report those resource costs alongside parameter savings.

## Substrates and controls

The existing canonical importer retains neuron IDs, isolated listed neurons, directed edges, raw positive weights, sign annotations, and source fingerprints. It removes autapses and sums duplicate edges consistently. No species-specific controller code is required. [DATA.md](DATA.md) documents the common interchange and provenance requirements.

| Condition | Frozen substrate |
|---|---|
| `real` | Pinned biological graph |
| `degree_shuffled` | Directed endpoint swaps restricted to equal raw weight and equal source sign |
| `matched_random` | Sparse simple directed random graph, same N/M, sign assignment and global raw weight multiset |
| `adapter_only` | No substrate or controller state; direct K-channel bottleneck |
| `trainable_gru` | Conventional learned recurrent controller, same total parameter ceiling |
| `no_edges` | Additional disconnected-network control; not one of the four MVP curves |

The weighted rewiring preserves each neuron's input/output degree and strength, signed input degree/strength, source weight multisets, and global weights. Ten attempted swaps per edge are fixed. The report records accepted swaps and the changed-target fraction. A finite restricted chain is not guaranteed to mix uniformly; graphs admitting no such swaps fail explicitly instead of silently using a weaker null. The random reservoir matches global statistics, not the biological degree sequence.

BANC is already prepared at 175,401 neurons and 13,542,180 directed edges. MaleCNS is prepared but is held out from this protocol's adapter tuning. Fish1 requires a pinned neuron/synapse export with coverage and materialization provenance; its imaging volume must not be described as a complete proofread whole-CNS graph. Cockroach and C. elegans are later extensions. Source locks and current unresolved entries are in [configs/sources.json](configs/sources.json).

## Native hover task and teacher

The body uses the pinned native `FlightImitationWBPG` flight configuration, released wing pattern, stock aerodynamics, and native residual joint/frequency action semantics. Actions modulate the same WPG and actuators for every controller. There is no learned stabilizer in the body. The teacher is used only to provide training labels.

Primary episodes last one second: 5,000 controller decisions, each containing four 50 μs physics steps. Reference COM is fixed; initial orientation/velocity vary. Two paired velocity disturbances occur at 25% and 60% of the horizon. Test cohorts are separate from training and validation; the OOD cohort doubles disturbance amplitudes. Observations include current proprioception and the fixed target reference, with fixed scaling and clipping. Hover success requires survival and a final 100 ms dwell inside position/orientation/velocity tolerances. Metrics include score, success, survival, drift, orientation error, recovery time/censoring, control magnitude, and numerical failures.

The released flight teacher and wing pattern come from [FlyBody's versioned research assets](https://doi.org/10.25378/janelia.25309105.v4), associated with [the FlyBody paper](https://www.nature.com/articles/s41586-025-09029-4). Asset SHA-256s are pinned; this asset release lists GPL 3.0+. The source/model licenses remain distinct. Teacher weights and datasets stay outside Git.

The saved TensorFlow teacher is converted once to a frozen PyTorch mean-action policy. Its numerical equivalence is checked against the original SavedModel. Qualification requires at least eight ≥0.5-second episodes, teacher success ≥87.5%, and WPG zero modulation success <50%. Published teacher pretraining is additional external prior information whose original training cost is not measured here.

## Training, accounting, and recovery

The immutable common cache contains 32 training, 8 validation, and 8 test teacher trajectories: **160,000 / 40,000 / 40,000 interactions** for this qualified teacher. Complete trajectories, terminal boundaries, executed actions, scenarios, and file digests are stored. Split overlap and schema mismatches are rejected.

Every run starts with 1,000 offline updates, batch 4, BPTT length 32, burn-in 64, Adam at 0.001, and gradient clipping at 1. DAgger adds exactly 5,000 and then 15,000 retained interactions, giving the grid **N_online = 0, 5k, 20k**. The student acts (`beta=0`); the teacher labels each visited state once. Each DAgger stage adds 500 updates, warm-starting weights and resetting Adam by the same rule for every condition. Validation action MSE selects weights within each stage. Closed-loop validation selects the best stage within each experience prefix.

All prefix selections are fixed before held-out evaluation. Neither test trajectories nor test success select weights or experience budgets. Final evaluation queries no teacher. Acute tests apply frozen weights with recurrence removed, state reset each decision, state permutation, output-port disruption, or silenced readout. State permutation is restricted to the output population, preventing the intervention itself from creating an input/output bypass.

Report common teacher interactions, additional on-policy interactions, optimizer updates, labeled sample presentations, burn-in presentations, evaluation interactions, and elapsed training time separately. `E_tau` is conditional on the shared offline data; `N_total = 160k + N_online` is available alongside it. Offline cache size is the available unique dataset, not a claim that every stored transition was sampled. The optimizer samples trajectories uniformly in this version, then selects a labeled window; short episodes therefore have greater per-transition sampling probability. This rule is shared and must remain fixed after the method freeze.

Optimizer checkpoints include model, Adam, RNG/sampling state, exact update count, selected weights, and histories. Offline continuation is tested bitwise on CPU. Collection commits immutable episodes and journals progress; interrupted uncommitted episodes replay from the same scenario/mixing seed. Retained experience stays exact. Extra simulator work caused by interrupted episode replay is reported as an upper bound, separately from retained learning experience. Episode files already committed before interruption are reused. Advisory locks reject concurrent writers. File/code/graph/body/budget identity changes reject resume.

## Run the BANC MVP

Commands below run from `experiments/connectome_body`. Use `uv` throughout. On a new machine, prepare the pinned BANC data and FlyBody source first:

```sh
uv sync --frozen
uv run --frozen cbbench bootstrap
uv run --frozen cbbench prepare banc --download
uv run --frozen python -m connectome_body.adaptation.cli assets --download
uv run --frozen --with tensorflow==2.18.1 --with tensorflow-probability==0.25.0 \
  --with tf-keras==2.18.0 python scripts/convert_flight_teacher.py
```

TensorFlow is only needed for conversion/verification, not subsequent training. On this workspace those assets, source, BANC graph, conversion, qualification, and common caches already exist; do not recreate immutable destinations.

The full-graph profile is already complete on this Mac. The command below creates the canonical report on a fresh checkout; choose an unused `--output` path to repeat it here, because reports are immutable.

```sh
uv run --frozen python -m connectome_body.adaptation.benchmark \
  --graph data/graphs/banc --device mps --batches 1 4 16 --lengths 8 32 64 \
  --repeats 3 --warmup 1 --output validation/brief-banc-mps-benchmark.json
```

The [local GPU table](validation/brief-banc-mps-benchmark.md) reports 320–403 combined forward/backward transitions per second through the actual 4,912-parameter structural adapter and full 175,401-neuron, 13,542,180-edge BANC graph. A transition means one whole-graph recurrent substep for one batch member; each body decision uses two. The largest case, batch 16 / BPTT 64, recorded 1.62 GiB of sampled tensor allocations and 2.13 GiB of sampled driver memory. MPS counters are sampled, not instrumented memory peaks. The benchmark includes learned structural-port allocation; it excludes optimizer steps, burn-in, validation, MuJoCo, and data loading, so it is not a complete campaign wall-time estimate.

For CPU profiling use `--device cpu --threads 6`. A [matched CPU case](validation/brief-banc-cpu-matched-benchmark.json), batch 4 / BPTT 32, measured 37.01 combined transitions/s against the GPU's 391.88: about 10.6× faster on the GPU for this workload. CUDA can use the same benchmark with `--device cuda` and a distinct output file; it remains untested here. Requested device unavailability fails explicitly. The earlier [CPU recurrence table](validation/brief-banc-cpu-benchmark.md) uses fixed ports and is a different benchmark; do not interpret it as a directly matched speedup comparison.

`benchmarks/structural_interface.py --graph data/graphs/banc --output validation/structural-check.json` additionally checks learned-port gradients and full-size rewired/random construction. Run it through `uv run --frozen python`; the completed local CPU check is [recorded here](validation/brief-banc-structural-interface.json).

On a new worker, qualify and collect the common data, or use verified copies with matching fingerprints:

```sh
uv run --frozen python -m connectome_body.adaptation.cli teacher-check \
  --config configs/brief/mvp_local.json --output validation/hover-teacher-qualification.json
uv run --frozen python -m connectome_body.adaptation.cli collect --split train --episodes 32 \
  --config configs/brief/mvp_local.json --output data/hover-v1/train
uv run --frozen python -m connectome_body.adaptation.cli collect --split validation --episodes 8 \
  --config configs/brief/mvp_local.json --output data/hover-v1/validation
uv run --frozen python -m connectome_body.adaptation.cli collect --split test --episodes 8 \
  --config configs/brief/mvp_local.json --output data/hover-v1/test
```

The local matrix is prepared at `runs/brief-mvp-local/plan.json`. Start here with the first `worker` command below. The `plan` command is for a fresh destination; it performs readiness checks without starting training and refuses to overwrite an existing plan.

```sh
uv run --frozen python -m connectome_body.adaptation.cli plan \
  --study configs/brief/mvp_local.json --output runs/brief-mvp-local
```

Run the first cell, then resume the remaining cells and analyze:

```sh
uv run --frozen python -m connectome_body.adaptation.cli worker \
  --plan runs/brief-mvp-local/plan.json --max-runs 1
uv run --frozen python -m connectome_body.adaptation.cli worker --plan runs/brief-mvp-local/plan.json
uv run --frozen python -m connectome_body.adaptation.cli analyze \
  --runs runs/brief-mvp-local --output runs/brief-mvp-local-analysis
```

The matrix is 4 conditions × 2 ceilings (5k/80k) × 3 seeds. It adds at most 480,000 retained DAgger interactions across all runs, plus shared teacher data, optimization, and evaluation. The learned-GRU reference is a separate [9-run local matrix](configs/brief/learned_rnn_local.json). Generate plans inside the intended runtime so paths and Python/dependency identities match. Use one worker on this Mac's single GPU. Re-run the same worker command to resume. Plans with missing data, qualification, or matching GPU evidence refuse to start; once prerequisites arrive, generate a fresh plan directory. CUDA versions of the study files remain available without the `_local` suffix.

The [existing Docker recipe](Dockerfile) can run this package with `--entrypoint /bench/.venv/bin/python ... -m connectome_body.adaptation.cli`; it has not been validated on CUDA here. Avoid transferring local plans with machine-specific paths. Preserve datasets, qualified teacher, source snapshot, checkpoints, and reports when moving workers.

## Prespecified expansion gates

The MVP advancement rule is a compute-allocation decision: all 24 cells must be complete scientific runs; low-capacity BANC must achieve mean held-out success ≥0.8; versus each of rewired/random/no-brain, it must show either a mean paired score gain ≥0.05 with at least two positive seed pairs, or ≥20% capped online-experience saving with every BANC seed reaching threshold. This exploratory rule is not a significance test. Negative results remain in the report.

After a recorded go decision:

```sh
uv run --frozen python -m connectome_body.adaptation.cli plan \
  --study configs/brief/capacity_local.json --output runs/brief-capacity-local
uv run --frozen python -m connectome_body.adaptation.cli worker --plan runs/brief-capacity-local/plan.json
uv run --frozen python -m connectome_body.adaptation.cli freeze \
  --study configs/brief/capacity_local.json --decision runs/brief-mvp-local-analysis/decision.json \
  --output runs/brief-local-method-freeze.json
uv run --frozen python -m connectome_body.adaptation.cli plan \
  --study configs/brief/heldout_local.json --output runs/brief-heldout-local
```

Capacity expansion adds 20k and reuses matching MVP runs. The held-out matrix includes BANC/MaleCNS/Fish1, 5k/20k/80k, and five seeds: 150 cells with the no-brain baseline shared across datasets. Matching BANC capacity runs can be reused. It cannot launch until the scientific MVP decision is verified, the method is frozen, and every required graph is available. Changing the architecture, dynamics, optimization, body, or evaluation after seeing a held-out substrate requires a new protocol rather than an unmarked adaptation of this one.

Target following, landing/walking, 320k capacity, additional sign/gain/port ablations, and additional species remain gated extensions. This implementation deliberately stops task expansion at hover until the MVP resolves the first scientific decision.

## Analysis and local verification

Analysis emits `frontier.csv`, `summary.json`, `report.md`, `decision.json`, and standalone PNG/PDF capacity, experience, topology advantage, cross-substrate compensation/affinity, and intervention figures when the required cells exist. It reports planned/incomplete cells, exact parameters, paired training-seed effects, 95% seed-bootstrap intervals, censored experience, and AUC. Episodes are not counted as independent training replicates. Three/five seeds give exploratory uncertainty; an interval containing zero does not establish equivalence. The ±0.05 compensation band is reported explicitly.

```sh
uv run --frozen pytest -q
uv run --frozen python scripts/validate_adaptation.py \
  --output runs/native-check --device mps --full-banc
```

The validation script exercises all five controller types on native FlyBody with a tiny graph, complete offline/DAgger/evaluation/lesion flow, and optionally the full BANC graph. Its short horizons and tiny update counts are labeled engineering evidence and cannot open the scientific MVP gate. Synthetic temporal sanity tasks are available through `temporal-data`; these isolate memory without simulator cost.
