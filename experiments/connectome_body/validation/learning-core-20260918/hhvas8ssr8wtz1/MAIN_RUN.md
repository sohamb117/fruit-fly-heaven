# BC to PPO main training window

Pod: `hhvas8ssr8wtz1`, one RTX 4090. This is the full-budget cross-body core,
not a qualification run. The frozen plan contains 270 comparison cells and
210 distinct training pipelines after shared controls are deduplicated.

- Start: September 18, 2026, 16:08:34 UTC (12:08:34 PM EDT).
- Worker deadline: September 19, 2026, 16:08:34 UTC (12:08:34 PM EDT).
- Checkpoint grace begins ten minutes before the deadline.
- Authorized window: 24 hours, up to $17.76 compute at $0.74/hour.
- **The user will stop the pod manually at the deadline.** Ending the worker
  window does not stop GPU billing. There is no automatic provider shutdown.

Six concurrent workers are admitted with at most two large connectome jobs.
Repeated launch checks showed 100% GPU utilization. The scheduler queues the
remaining work as workers finish; this window does not promise completion of
all 210 pipelines.

All 30 deterministic graph/topology/seed caches were uploaded and verified as
cache hits against the frozen plan. All five seeds are enabled in the running
scheduler. No workers were restarted to enable the remaining seeds.

## Scientific scope

BANC, MaleCNS and C. elegans control FlyBody hover and published worm locomotion.
The plan includes real and degree-preserving rewired topology, adapters-only
and joint permitted-edge plasticity, parameter-matched GRU/RNN controls, and
adapter-only MLP controls, with paired seeds 0 through 4. All students use the
same qualified expert trajectories for their task. Each pipeline has five BC
epochs followed by a fixed one-million-interaction PPO budget. The broader
training-regime comparison and temporal diagnostic are deferred.

The full-duration expert datasets passed their qualification checks. Their
fingerprints are:

- Hover: `6e16a7037f45c103090ad1445c9e9b739c0c3ef3876b131b96769127fdffa6f0`
- Worm: `f45aa9c6f1a0236627302c9c4a437353b483f92e7fa69732b9aaa89fffe3e4b3`

## Runtime and artifacts

Runtime root on the pod:
`/workspace/qualification-isolated-20260918/experiments/connectome_body`

Relative to that root:

- Frozen plan: `runs/learning-core-cuda-v2/plan.json`
- Scheduler: `scripts/parallel_learning_core.py`
- Status and logs: `runs/main-parallel-20260918/`
- Checkpoints and results: `runs/learning-core-cuda-v2/runs/`
- Qualification: `runs/learning-qualification-cuda-v2/qualification.json`
- Expert preparation report: `validation/main-preparation-20260918/datasets.json`

Public input graphs, deterministic caches, and common expert trajectories are
under `/workspace/experiments/connectome_body/data/`, as pinned in the plan.
The isolated runtime avoids the source/environment replacement observed during
the earlier migration. No identity checks were relaxed.

Plan fingerprint:
`8cb73fec7ede3446f480fddef595e7925df0f6fc028aff11109b66aade8e8b64`

Source fingerprint:
`683ff8b711d4c46519e01b06bfea4b6b882d3dc6d3a902d762a91f362165a309`

All 18 qualification cases passed; all 36 selected qualification checkpoints
were downloaded and hash-verified locally. These are execution checks, not
evidence of a biological performance advantage. Main-run scientific results
must be taken from the full-training outputs and paired evaluations.

The initial main-training snapshot was also downloaded locally: 12 checkpoints
plus launch/status, expert-dataset and graph-cache provenance. Every included
file was hash-verified. `main-initial-snapshot-20260918.tar.gz` has SHA-256
`bac83b40e7c46817e3095aee6feca65b8fda76b2c46048f2b72698fff847bc61`.
This is a partial snapshot; active workers continue saving newer checkpoints
on the pod. `main-status.json` is the captured status, not a live view.

## Priority control window after training-loop validation

The user requested direct validation of the training loop and prioritized MLP/GRU
controls before expanding the comparison. `scripts/priority_learning_controls.py`
started at epoch `1789770073.8353827` with two original frozen-plan conditions:

- Matched GRU, hover, seed 0: `4d0dbec884577da83120f530-bc_ppo` (PID 13285).
- Adapter-only MLP, hover, seed 0: `e2ecf7db63b53727fab0bcea-bc_ppo` (PID 13286).

Their logs/status are in `runs/priority-controls-20260918/` under the active
runtime. Each has a 1,200-second cooperative checkpoint limit; the supervisor's
hard cutoff is epoch `1789771873.8352897`. It restores the previous scheduler
admission limit on completion or failure, including timeout. During this window
`control.json` has `max_workers=0` to pause NEW admissions; the original six
workers continue. This does not stop the pod or extend its original deadline.
The full scientific budgets remain unchanged and paused controls can resume.

The supervisor's restoration behavior passed two local subprocess tests covering
successful and failed children. The loop itself passed 54 local tests and three
full-size CUDA gradient audits; see `../RECURRENT_LOOP_VALIDATION.md`.
