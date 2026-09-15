# Connectome–FlyBody benchmark

A runnable suite for a **common learned interface to frozen connectomes**, with stock FlyBody as the body. Encoder/decoder weights are learned separately for each substrate; architecture, parameter ceilings, observations/actions, tasks, PPO procedure, and experience budgets are shared.

```text
FlyBody observations (286)
  → learned encoder → 16 fixed input channels
  → frozen sparse connectome dynamics
  → 16 fixed output channels → learned decoder
  → FlyBody actuator commands (59)
```

Input and output neurons are disjoint: observations must cross a graph edge to reach the decoder. No homologous neurons, anatomical motor labels, or pretrained controller are required. The 5,000-parameter ceiling gives **4,682 trainable actor parameters** for any sufficiently large graph. Graph size still affects memory and computation.

Read the [experimental protocol](PROTOCOL.md), [data requirements](DATA.md), and [validation evidence](VALIDATION.md). The suite is implemented; scientific learning curves and biological conclusions remain to be obtained.

## Comparisons and tasks

| Condition | Learned part | Comparison |
|---|---|---|
| `real` | Encoder and decoder | Frozen biological graph under common assumed dynamics |
| `degree_shuffled` | Same adapters | Directed degree/signed-degree preserving topology null |
| `matched_random` | Same adapters | Frozen random RNN matched for N, M, signs, and global weight multiset |
| `no_edges` | Same adapters | Disconnected input/output populations; decoder can only learn a constant action |
| `adapter_only` | Identical encoder and decoder | Exact brain-free adapter, with encoder channels feeding decoder directly |
| `adapter_gru` | Adapters and a GRU | Recurrent brain-free adapter within the same total actor ceiling |
| `trainable_gru` | Observation→GRU→action policy | Conventional trainable RNN within the same actor ceiling |

Tasks are **standing balance/recovery**, **commanded ground locomotion**, and **target reaching with a final dwell**. All use the stock FlyBody walking configuration, with wings disabled, and the same observation/action schema.

## Install and validate

From the repository root:

```sh
cd experiments/connectome_body
uv sync --frozen
uv run --frozen cbbench bootstrap
uv run --frozen pytest -q
```

Python 3.12, dependencies, and FlyBody source are pinned. No display or OpenGL rendering is needed. `bash scripts/validate.sh` installs, verifies source, lints, and tests. Set `UV_CACHE_DIR` to a writable directory if needed.

## Prepare data

Both fly graphs are already prepared in this workspace. For a fresh checkout, download and prepare them with:

```sh
uv run --frozen cbbench data-status
uv run --frozen cbbench prepare banc --download
uv run --frozen cbbench prepare malecns --download
```

| Prepared snapshot | Neurons | Directed edges | Status |
|---|---:|---:|---|
| BANC v888 / v3 | 175,401 | 13,542,180 | Prepared and verified in this workspace |
| MaleCNS v1.0 / confidence 0.5 / traced neurons | 165,122 | 25,563,096 | Prepared and verified in this workspace |
| Fish1 zebrafish | — | — | Requires a pinned neuron/synapse export |
| Cockroach | — | — | Suitable release not yet verified |

An existing BANC raw directory containing `meta.feather` and `edges-v3.feather` can be passed with `--raw /absolute/path`. Source bytes must match [the lock](configs/sources.json). Prepared graphs are immutable: rerunning preparation against an existing destination intentionally fails. Use a new directory for a new release. Data, environments, downloaded references, and runs are ignored by Git.

## Check actual graphs and training

```sh
uv run --frozen cbbench preflight --graph data/graphs/banc --output runs/banc-preflight.json
uv run --frozen cbbench train-smoke --graph data/graphs/banc --output runs/banc-smoke
uv run --frozen cbbench train-smoke --graph data/graphs/malecns --task walk --output runs/malecns-smoke
uv run --frozen cbbench train-smoke --substrate adapter_only --output runs/adapter-only-smoke
uv run --frozen cbbench train-smoke --substrate trainable_gru --output runs/gru-smoke
```

Preflight checks finite activity/actions and a gradient through frozen edges. Training smoke performs 32 real interactions, checkpointing, validation selection, held-out evaluation, and causal interventions. Both are engineering checks; smokes do not establish task learnability or a biological advantage. Scientific reports exclude them by default. Use a fresh output path, or `--resume` with unchanged code/configuration.

## Plan and launch

Planning writes all configurations, graph fingerprints, readiness, and interaction counts without launching training:

```sh
uv run --frozen cbbench plan --study configs/pilot.json --output runs/pilot
uv run --frozen cbbench worker --plan runs/pilot/plan.json --max-runs 1
uv run --frozen cbbench worker --plan runs/pilot/plan.json
uv run --frozen cbbench analyze --runs runs/pilot --output runs/pilot-analysis
```

The pilot has **132 runs and 540,672 training interactions**: both fly datasets, all controls, three tasks, two capacities, two training seeds, and 4,096 interactions per run. It is a feasibility pilot, not a powered species comparison.

The [full study](configs/study.json) specifies four datasets, **walking and reaching** as the two primary tasks, four graph conditions, three brain-free baselines, actor ceilings of **5k, 20k, 100k, 500k**, five training seeds, three port/null draws, and one million training interactions per run. It has **2,040 runs / 2.04 billion training interactions**, plus evaluation. Baselines are not duplicated for datasets or port draws.

Balance remains a calibration task: the first conventional RNN pilot already met its success criterion before learning, so it cannot carry the main sample-efficiency claim. Pilot seeds 0–1, sensitivity seeds 10–12, and primary seeds 100–104 generate separate training/validation/test scenario cohorts.

```sh
# Records missing Fish1/cockroach data and refuses to launch the incomplete matrix.
uv run --frozen cbbench plan --study configs/study.json --output runs/full-study

# Explicitly scoped to the two prepared fly releases: 1,080 runs / 1.08B interactions.
uv run --frozen cbbench plan --study configs/study.json --datasets banc malecns --output runs/fly-study
```

These are planned budgets, not completed experiments. Profile a full graph before allocating the campaign; equal interaction budgets do not imply equal wall time.

### GPU workers and resumption

CPU is the explicit default. Generate a new plan with a recorded CUDA override:

```sh
uv run --frozen python scripts/device_plan.py --study configs/pilot.json --device cuda --output runs/pilot-cuda
uv run --frozen cbbench preflight --graph data/graphs/banc --device cuda --output runs/banc-cuda-preflight.json
bash scripts/worker.sh runs/pilot-cuda/plan.json 0 1
```

For four workers, use shard indices 0–3 with shard count 4. Assign one GPU per process using `CUDA_VISIBLE_DEVICES` as appropriate. Each worker has independent environments and checkpoints; a run lock prevents simultaneous writes. Keep the shard count fixed on restart. Completed runs are skipped, and interrupted runs resume from their latest complete checkpoint. Use a filesystem supporting advisory locks, or separate worker output directories.

Single-run debugging/resumption:

```sh
uv run --frozen cbbench train --config runs/pilot/configs/banc-balance-real-p5000-s0-g0.json \
  --output runs/one-run --stop-after-updates 2
uv run --frozen cbbench train --config runs/pilot/configs/banc-balance-real-p5000-s0-g0.json \
  --output runs/one-run --resume
```

Checkpoints include actor, critic, optimizer, neural state, full MuJoCo integration state, scenario/RNG state, counters, and history. Graph/body/code/configuration/Python/dependency identity must match. CPU fixture continuation is tested bitwise; cross-device bitwise equivalence is not asserted. Unavailable CUDA raises an error; there is no device fallback. CUDA and the [Docker recipe](Dockerfile) need validation on the intended worker.

Build the Docker image from this directory. Generate plans inside the container so paths and versions match:

```sh
docker build -t connectome-body-bench .
docker run --rm --gpus all -v "$PWD/data:/bench/data:ro" -v "$PWD/runs:/bench/runs" \
  --entrypoint /bench/.venv/bin/python connectome-body-bench \
  scripts/device_plan.py --study configs/pilot.json --device cuda --output runs/container-pilot
docker run --rm --gpus all -v "$PWD/data:/bench/data:ro" -v "$PWD/runs:/bench/runs" \
  connectome-body-bench worker --plan runs/container-pilot/plan.json
```

## Sensitivities and new connectomes

The [dynamics sweep](configs/dynamics_sensitivity.json) varies common gain, timescale, input drive, port count, and weight transform. The [sign sweep](configs/sign_sensitivity.json) compares common random Dale signs with BANC transmitter-derived assumptions. Each uses 200k interactions/run and is analyzed separately from the main study.

```sh
uv run --frozen cbbench plan --study configs/dynamics_sensitivity.json --output runs/dynamics
uv run --frozen cbbench plan --study configs/sign_sensitivity.json --output runs/signs
```

Equal-N/equal-M sensitivity uses only existing edges within uniformly sampled neurons:

```sh
uv run --frozen cbbench matched-subgraph --graph data/graphs/banc \
  --neurons 100000 --edges 3000000 --seed 0 --output data/graphs/banc-size0
uv run --frozen cbbench matched-subgraph --graph data/graphs/malecns \
  --neurons 100000 --edges 3000000 --seed 0 --output data/graphs/malecns-size0
uv run --frozen cbbench plan --study configs/pilot.json \
  --datasets banc-size0 malecns-size0 --output runs/size0
```

Repeat predeclared subset seeds in distinct directories. If the induced subset has too few edges, choose a lower common target across datasets before inspecting performance. Subsampling removes real circuit structure; interpret alongside full-graph results.

For any future release, supply a neuron table, directed weighted edge table, and completed provenance:

```sh
uv run --frozen cbbench import-edges --nodes /path/nodes.csv --edges /path/edges.csv \
  --provenance /path/provenance.json --output data/graphs/fish1
```

[DATA.md](DATA.md) specifies the interchange and Fish1 snapshot requirements. No species-specific adapter code is needed.

## Outputs

Runs write `manifest.json`, `config.json`, `latest.pt`, `best.pt`, `learning_curve.json`, `updates.json`, `episodes.json`, and `result.json`. Interrupted/failed training also records `failure.json`. Immutable plans reject changed code or graphs.

Analysis writes `runs.csv`, `summary.json`, `report.md`, and standalone PNG/PDF learning/capacity figures. Metrics include held-out/OOD success, reward, generalization gaps, falls, numerical failures, seed-clustered intervals, real-minus-null/RNN effects, cross-connectome contrasts, censored experience, and the smallest successful tested capacity.

```sh
# Only for explicitly labeled validation artifacts:
uv run --frozen cbbench analyze --runs runs/validation --output runs/validation-analysis --include-validation
```

Duplicate run identities count once. Different code, releases, dynamics, devices, body/training configurations stay separate. Missing/incomplete runs remain visible; learners that fail to reach threshold are retained in the experience statistic.
