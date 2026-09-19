# Imitation and reinforcement learning core

The execution priority is the final section of [GOAL.md](../GOAL.md).
[learning_core.json](configs/paper/learning_core.json) now schedules only the five-seed BC→PPO cross-body core.
`learning_regimes.json` retains the deferred 120-cell training-regime comparison.
The linear-memory diagnostic remains planned but is excluded from this initial
embodied worker by `launch_regimes: ["bc_ppo"]`. The full
paper plans remain separate and must be recompiled after this source change.

Prepare graph inputs **on CPU before paid GPU time**. For example, compile a
local plan and warm its qualification inputs (six topology/seed preparations
shared across bodies and plasticity). The default cache is `data/graphs/.prepared`.

```sh
uv run --no-sync python -m connectome_body.compatibility.learning_study plan --config configs/paper/learning_core.json --output runs/learning-core-local-v2
uv run --no-sync python scripts/prepare_learning_core.py --plan runs/learning-core-local-v2/plan.json --output validation/preparation-v2.json
```

Cache entries bind graph fingerprints, topology/initialization settings, seeds,
implementation hashes and dependency versions. Arrays are checksummed, locked and
published atomically; learned weights and recurrent state are never shared. Keep
the cache with the approved public graph inputs when transferring to the pod.
Changing a relevant implementation or graph creates a new cache identity.

Compile a **new remote plan** with the same source after transferring the prepared
inputs; plan paths are machine-specific. The active config uses a new immutable
`data/learning-core-v2` demonstration namespace because runtime code changed.
Preserve old data, plans and checkpoints; do not edit their stored identities to
resume them under new code. The historical source is saved beside downloaded
qualification artifacts.

Qualification now covers all six source/body pairs first, then GRU/RNN/MLP, then
joint and rewired conditions. Every case has a 180-second cap inside a 1200-second
global cap. Failures are recorded while other cases continue. `--resume` skips
verified complete cases and resumes atomic checkpoints for interrupted cases,
but requires identical source/plan/qualification code. Qualification refuses
missing preparation caches rather than spending GPU budget regenerating them.

Use an independent supervisor for any paid launch. The following command runs
only against an **already running** pod with dependencies, inputs and the remote
plan ready; it never creates, starts or deletes a pod. Resolve the current SSH
host/port from RunPod, verify its host key, and substitute those values below.

```sh
uv run --no-sync python scripts/qualification_ssh_job.py \
  --pod-id POD_ID --host SSH_HOST --port SSH_PORT --ssh-key /path/to/ssh-key \
  --plan runs/learning-core-cuda-v2 --run runs/learning-qualification-cuda-v2 \
  --output validation/qualification-download-v2 \
  --training-seconds 1200 --hard-seconds 1500
```

Before allocating compute, configure a locally scoped `RUNPOD_API_KEY` or
runpodctl key with Pods read/write access. Missing credentials prevent the worker
from launching. Keys stay on the supervisor host, are removed from the worker's
environment and are never written to experiment outputs. The supervisor arms a
detached watchdog **before** SSH. The watchdog stops the exact pod boot on command
completion, missed supervisor heartbeat, or the hard deadline; retries API
errors; and verifies `EXITED` with no runtime. A changed pod boot is never stopped
by an old watchdog. The parent also attempts verified shutdown if its guard dies.
The worker checks `RUNPOD_POD_ID`, archives checkpoints even after a qualification
failure, downloads the archive, and verifies its checksum before finishing.

Run the supervisor on an **always-on host**: a sleeping laptop cannot enforce a
wall-clock deadline while asleep. Network/control-plane outages can delay stop
verification; the watchdog keeps retrying and does not claim success. This is an
external API watchdog, not a RunPod server-side scheduling feature. Its automated
tests use fake provider responses and an actual detached local process; a live
shutdown test remains required before trusting a paid deployment.
[RunPod's stop documentation](https://docs.runpod.io/pods/manage-pods#stop-a-pod)
explains the provider operation and retained volume storage.

Keep all remote artifacts on `/workspace`. The hard deadline includes archiving
and transfer, and wins if either stalls. Checkpoints remain on the persistent
volume for later retrieval; storage billing continues after compute stops.
`guard/guard.json` records the reason, retry count and verified provider state.
Full scientific training remains a separate budget, requiring a complete passing
qualification for that exact source and plan. Do not interpret the qualification's
64-step episodes (12.8 ms for FlyBody) as control-performance evidence.

A BC-only cell's `policy_id-bc/best.pt` is also the initialization of its
`policy_id-bc_ppo` cell. BC completion and checkpoint hash are checked before PPO;
PPO cannot silently start from scratch when BC is missing. BC-only has no PPO
stage. PPO-only never opens the expert data. Conventional RNN/GRU cores always
train; they match either the frozen-interface or joint-edge biological capacity.

Teacher data are generated once per task, not once per substrate or seed. Failed
expert qualification remains explicit. No-brain controls use the same data and
20k interface ceiling. Repeated anchored conventional controls share an execution
when their actual configuration is identical.

The report writes test/OOD episode metrics, BC error, PPO AUC and thresholds,
seed variance, and both requested interactions. Common-compute comparisons use
prior measured scores only and show missing coverage instead of extrapolation.
DAgger is documented but deliberately not scheduled in the first pass.

## Training repair candidates

The September 18–19 development checks are described in
[`validation/training-repairs-20260918/STATUS.md`](validation/training-repairs-20260918/STATUS.md)
and the authoritative `../GOAL.md`. They preserve the original run and are not
pooled with its results. The optional implementation now supports fixed train-only
observation/action scaling, standardized BC loss, constant-action initialization,
physical-time PPO horizons and exact conditional-KL stopping. Scaling statistics
are non-trainable checkpoint buffers; actor, critic and BC→PPO transfer use the
same observation transformation. Parameter budgets and the memory constraint
remain unchanged.

`scripts/qualify_training_repairs.py` performs an isolated, resumable BC→PPO
candidate run against the existing shared expert data. Its supervisor bounds a
window inside the original paid deadline, temporarily holds new old-plan
admissions, restores scheduling on exit, and never stops the provider pod.
Use `--run-name` to resume the same checkpoints in a newly named scheduling
window. A time-limited BC checkpoint is unfinished; it is not a PPO result.

The first action-scaled candidate reduced BANC imitation error but did not
achieve hover success during the initial BC window. Its frequency-response probe
and the unchanged-size linear-readout probe are mechanistic diagnostics, not
evidence of biological advantage. Promotion requires closed-loop validation of
the complete repaired BC→PPO sequence, not merely passing implementation tests.
