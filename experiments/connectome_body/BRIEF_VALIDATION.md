# Local validation of the adaptation suite

Validated on 2026-09-15 on an **Apple M2 Pro, 10 CPU cores, 16 GiB unified memory**, macOS 26.3, Python 3.12.1, and PyTorch 2.8.0. The full BANC graph runs locally on the Apple GPU with sparse Metal kernels. No cloud GPU was provisioned.

The local 24-run MVP and separate 9-run learned-GRU reference pass launch prerequisites. The 24-run MVP was subsequently launched; [CAMPAIGN.md](CAMPAIGN.md) links its live status. No complete scientific comparison is available yet. The evidence below establishes sparse computation, differentiation, interface learning on simple temporal data, and native FlyBody integration; it does not establish a biological topology advantage or learned BANC hover competence.

The implemented protocol and launch commands are in [BRIEF.md](BRIEF.md). Machine-readable evidence, source/archive identities, and local result links are collected in [brief-local-evidence.json](validation/brief-local-evidence.json).

## Full BANC on the local GPU

The canonical graph contains **175,401 neurons and 13,542,180 directed edges**. Its fingerprint is `a19030096ae9a19e80bf52e2423b9640a40e1867fcf330e17572a89cb7354415`.

The measured controller is the actual generic interface: 104 observations → 16 learned structural ports → full frozen BANC → 16 readout ports → 12 actions. It has **4,912 trainable parameters**, including 384 port-query coefficients. Synaptic weights are frozen. The rate rule uses two neural substeps per body decision, tau 2 ms, and gain 0.9.

All nine combinations of batch `{1, 4, 16}` and BPTT length `{8, 32, 64}` passed. Each cell has one warm-up and three measured repetitions. Tables report medians of phase timings.

| Batch | BPTT length | Inference transitions/s | Forward + backward transitions/s | Sampled tensor allocations |
|---:|---:|---:|---:|---:|
| 1 | 8 | 714.07 | 320.32 | 0.271 GiB |
| 4 | 32 | 792.91 | 391.88 | 0.402 GiB |
| 16 | 64 | 796.81 | 397.67 | 1.620 GiB |

A transition is one **whole-graph neural substep for one batch member**. Divide the transition rate by two for aggregate body-decision equivalents under this configuration. These timings include E/D and structural-port computation; they exclude MuJoCo, optimizer steps, burn-in, dataset loading, and validation. They are not measurements of complete campaign throughput.

The largest cell recorded 1.620 GiB of MPS tensor allocations and 2.133 GiB of driver memory. These are sampled unified-memory counters, not instrumented peak VRAM; driver memory includes tensor allocations and must not be added to them. The complete [nine-cell table](validation/brief-banc-mps-benchmark.md) and [raw repetitions](validation/brief-banc-mps-benchmark.json) are saved.

At batch 4 / BPTT 32, the same full-adapter benchmark on six CPU threads achieved **37.01** combined transitions/s. The GPU achieved **391.88**, about **10.6×** faster in this matched configuration. The [CPU report](validation/brief-banc-cpu-matched-benchmark.json) also contains all three repetitions. The earlier single-thread [fixed-port CPU benchmark](validation/brief-banc-cpu-benchmark.json) measures a different workload and is retained as historical engineering evidence.

Before GPU profiling, the full recurrent system was compared against CPU CSR for four body decisions at batch two. Maximum state difference was **8.94×10⁻⁸**; relative input-gradient error was **3.87×10⁻⁸**. Every benchmark cell also checked finite input and adapter gradients. The Metal backend uses sparse CSR arrays and a cached transpose for differentiation, without densifying the graph or falling back to CPU computation.

Reproduce on this machine with a fresh output filename:

```sh
uv run --frozen python -m connectome_body.adaptation.benchmark \
  --graph data/graphs/banc --device mps --batches 1 4 16 --lengths 8 32 64 \
  --repeats 3 --warmup 1 --output validation/banc-mps-repeat.json
```

Run commands from `experiments/connectome_body`. Add `--offline --cache-dir /private/tmp/connectome-body-uv` to use the already cached environment. CPU and CUDA remain explicit alternative devices; CUDA has not been tested in this workspace.

## Teacher and immutable trajectories

The pinned released FlyBody teacher was converted to a frozen PyTorch mean-action policy. Comparison against the original TensorFlow SavedModel on 64 observations passed, with maximum absolute action error **1.1623×10⁻⁶**. The [conversion record](data/teacher-assets/torch-flight/conversion-check.json) identifies the exact teacher.

Qualification used eight paired, disturbed, one-second native hover episodes:

| Controller | Successes | Mean hover score | Mean survival fraction |
|---|---:|---:|---:|
| Released teacher | 8 / 8 | 0.98532 | 1.00000 |
| WPG with zero modulation | 0 / 8 | 0.01331 | 0.02733 |

Neither condition had numerical failures. The teacher's mean position error was 0.01927 cm and mean orientation error 0.04718 radians. This qualifies the task/teacher pair; it is not a result for a learned connectome controller. Full per-episode measurements are in [hover-teacher-qualification.json](validation/hover-teacher-qualification.json).

The common immutable cache contains 32 / 8 / 8 complete train/validation/test trajectories, totaling **160,000 / 40,000 / 40,000 interactions**. Data, teacher, body, and file fingerprints are recorded in [brief-data-manifests.json](validation/brief-data-manifests.json). Teacher weights and caches are local ignored assets, not included in Git or the source archive.

## Interface and learning checks

The full-size CPU structural check built real BANC, its degree/strength-preserving null, and its matched sparse random reservoir through the same 4,912-parameter interface. All three produced finite, nonzero encoder and input/output query gradients. The rewiring attempted 135,421,800 swaps and accepted 134,058,411; 99.992% of edge targets changed. Exact degree/strength, sign, and weight preservation checks passed. Acceptance and target change do not establish uniform mixing of the restricted null. See [brief-banc-structural-interface.json](validation/brief-banc-structural-interface.json).

On a balanced delayed binary-cue fixture with identical final observations, a small recurrent substrate and a trainable GRU each achieved 100% held-out sign accuracy; the stateless no-brain adapter achieved 50%. A full-BANC adapter also learned this simple cue-retention task after 64 updates, with 128 available unique training interactions and 2,048 labeled sample presentations. These are learnability checks on a small cue alphabet, with no biological-versus-random inference. The results and their earlier source snapshot are preserved in [brief-engineering-evidence.json](validation/brief-engineering-evidence.json).

Exact native-interface parameter counts are independently recorded in [brief-parameter-accounting.json](validation/brief-parameter-accounting.json). No-brain and trained-GRU controllers spend their full available ceilings on useful trainable weights; they do not receive padding parameters to manufacture a match.

## Native FlyBody integration on MPS

[The local integration run](runs/brief-native-validation-mps/validation-summary.json) completed all five conditions: real, rewired, fixed random reservoir, adapter-only, and learned GRU. Each exercised four offline updates, two DAgger stages with two updates each, exactly 32 retained student interactions, frozen validation/test/OOD evaluation, acute interventions, checkpointing, and analysis.

The same script additionally completed full BANC with the native 104-observation / 12-action interface. Its two-update offline check used 16 labeled sample presentations and achieved validation action MSE 0.014738. It then evaluated the frozen controller in native FlyBody. See the [full-BANC result](runs/brief-native-validation-mps/full-banc/result.json) and [offline-stage record](runs/brief-native-validation-mps/full-banc/stage-00/result.json).

These checks use a 32-decision horizon, too short to satisfy the primary 100 ms success dwell. All recorded hover successes are zero. They validate execution, accounting, and report generation; they do not estimate scientific hover performance. Analysis labels them as validation fixtures, excludes them by default, and correctly leaves the MVP advancement decision false. The generated experience figure was visually inspected.

```sh
uv run --frozen python scripts/validate_adaptation.py \
  --output runs/native-check-repeat --device mps --full-banc
```

Use a fresh destination, or `--resume` with the same configuration and source. The current code plus tests/configuration/runtime lock are preserved in [brief-local-sources.zip](validation/brief-local-sources.zip). Earlier CPU smoke checkpoints have separate `brief-native-v1-sources.zip` and `brief-native-v2-sources.zip` snapshots; strict source identity intentionally prevents silently loading them with changed code.

## Automated checks and campaign readiness

**60 tests passed in 16.82 seconds.** Ruff checks, formatting checks on 45 Python files, and `git diff --check` passed. The sole test warning was PyTorch's standard sparse-CSR beta notice. Complete command output is saved in [brief-local-checks.json](validation/brief-local-checks.json).

Checks cover structural-query gradients and disjoint supports, budget independence from neuron count, frozen recurrence, null invariants, temporal no-brain controls, 10,000-node sparse differentiation, cached-graph identity, immutable-data integrity, selection/accounting, exact CPU training resumption, native MuJoCo/WPG checkpoint restoration, interrupted DAgger recovery, and frontier statistics. Four MPS tests cover directed sparse output/gradient parity, empty rows, noncontiguous input, learned query gradients, and training continuation.

The [launch readiness audit](validation/brief-launch-readiness.json) supersedes the earlier local and CUDA-only audits. The first local launch stopped before training because JSON integers such as `2` and equivalent floats such as `2.0` produced different task hashes. Input configuration now uses canonical numeric types, and a new native test verifies the actual planned body fingerprint against the qualified teacher. The numerical task, physiology code, teacher data, and recurrent benchmark identities are unchanged. After this correction, **62 tests passed in 18.12 seconds**, with lint and formatting checks also passing; see [brief-launch-checks.json](validation/brief-launch-checks.json).

| Matrix | Cells | Current status |
|---|---:|---|
| Local BANC MVP | 24 | Launched sequentially on the local Apple GPU |
| Local learned-GRU reference | 9 | Ready; no scientific runs launched |
| BANC capacity expansion | 36, including 24 reused MVP cells | Gated on scientific MVP advancement |
| BANC / MaleCNS / Fish1 | 150 | Gated on Fish1 export, MVP advancement, and frozen method |

The current MVP plan is [runs/brief-mvp-local-v2/plan.json](runs/brief-mvp-local-v2/plan.json). It has an active supervisor. If the campaign has stopped, resume with:

```sh
uv run --frozen python scripts/run_adaptation_campaign.py \
  --plan runs/brief-mvp-local-v2/plan.json
```

Use one worker on this Mac's GPU. The supervisor resumes unfinished work, skips verified completed results, and analyzes the final matrix. The GPU benchmark establishes feasibility; the first scientific cell will establish actual training/evaluation wall time. No full-campaign time or biological effect has been inferred from the microbenchmark.
