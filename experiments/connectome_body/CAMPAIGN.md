# Local BANC hover campaign

The scientific MVP was launched on **2026-09-15 at 14:02 EDT**, on the Mac's Apple M2 Pro GPU. This document records the launch; the [live status file](runs/brief-mvp-local-v2/campaign-status.json) is authoritative for current progress and is refreshed every 30 seconds while a cell runs.

The campaign covers **24 cells**: real BANC, degree/strength-preserving rewired BANC, a matched fixed sparse RNN, and the adapter-only controller; 5k and 80k parameter ceilings; seeds 0, 1, and 2. It uses the existing immutable teacher trajectories and fixed training/evaluation protocol. The separate learned-GRU campaign is prepared but is not queued in this MVP.

Each cell performs 1,000 offline updates, DAgger at cumulative 5,000 and 20,000 retained interactions with 500 further updates per stage, validation-based selection, held-out/OOD evaluation, and acute interventions. Checkpoints are saved every 25 optimizer updates. A completed cell must pass the worker's existing identity checks before the next one starts.

| Artifact | Location |
|---|---|
| Live status, current stage, checkpointed updates | [campaign-status.json](runs/brief-mvp-local-v2/campaign-status.json) |
| Exact launch command and provenance | [campaign-launch.json](runs/brief-mvp-local-v2/campaign-launch.json) |
| Cell start/finish events and failures | [campaign.log](runs/brief-mvp-local-v2/campaign.log) |
| First cell's detailed log | [banc-real-p5000-s0.log](runs/brief-mvp-local-v2/logs/banc-real-p5000-s0.log) |
| Immutable 24-cell plan | [plan.json](runs/brief-mvp-local-v2/plan.json) |
| Launcher | [run_adaptation_campaign.py](scripts/run_adaptation_campaign.py) |

The supervisor executes one cell at a time in a fresh process, so completed cells release their GPU allocations. It runs independently of this chat, inhibits idle sleep for its lifetime, and stops if a worker fails. It releases the sleep inhibitor when it exits. macOS lid closure, forced sleep, logout/reboot, or process termination can still interrupt execution; checkpoint and dataset journals support resumption.

After all 24 cells finish, analysis is generated automatically at `runs/brief-mvp-local-v2-analysis/`, including the four capacity curves, experience frontiers, topology effects, interventions, and the MVP advancement decision. The campaign does not automatically launch additional species or change the method in response to results.

To inspect progress from `experiments/connectome_body`:

```sh
cat runs/brief-mvp-local-v2/campaign-status.json
tail -n 30 runs/brief-mvp-local-v2/campaign.log
```

If the supervisor has stopped, resume using the same plan:

```sh
uv run --frozen --offline --cache-dir /private/tmp/connectome-body-uv \
  python scripts/run_adaptation_campaign.py \
  --plan runs/brief-mvp-local-v2/plan.json
```

An advisory lock prevents duplicate supervisors. Keep implementation files, dataset files, and immutable plan/configuration files fixed while the campaign runs. If interrupted, resumption restores unfinished training and skips verified completed cells. The supervisor PID and worker PID are recorded in the live status file.

The original `runs/brief-mvp-local` attempt failed before training due to `2` versus `2.0` JSON serialization in the body fingerprint. Version 2 canonicalizes input types; its actual native body fingerprint matches the existing teacher and trajectory cache. Numerical settings, neural dynamics, and physical simulation code are unchanged. The failed attempt and log are retained, and [62 regression tests passed](validation/brief-launch-checks.json) before the corrected launch.

Wall time for the full scientific matrix has not yet been measured. The first completed cell will supply a full training/evaluation timing; sparse-kernel benchmark throughput alone is insufficient for that estimate.
