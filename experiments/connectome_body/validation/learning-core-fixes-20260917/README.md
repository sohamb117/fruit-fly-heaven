# Local fixes and interpretation — 2026-09-17

No pod was launched, restarted, or provisioned during this fix pass.

## What the earlier run established

The three completed smoke cases each used two BC updates, 128 expert samples,
two PPO updates, and 16 PPO decisions. Their 64-step FlyBody evaluations lasted
0.0128 simulated seconds. They do not establish sustained control, topology
benefit, or cross-body generalization.

| Condition | Initial BC validation MSE | Final BC validation MSE | Relative reduction | Held-out success |
| --- | ---: | ---: | ---: | ---: |
| Real BANC, adapters | 0.01283734 | 0.01278753 | 0.388% | 0/1 |
| Real BANC, joint | 0.01283734 | 0.01278752 | 0.388% | 0/1 |
| Rewired BANC, adapters | 0.01281644 | 0.01278909 | 0.213% | 0/1 |

All three PPO runs selected step zero (their BC initialization). These values are
descriptive smoke output, not comparative scientific evidence. Machine-readable
values are in `prior-smoke-results.json`. Old checkpoints and their source bundle
remain in `../learning-core-20260917/gpu-qualification/`.

## Implemented changes

- Persistent, checksummed CPU topology/feature/initialization cache, with atomic
  publication and a per-key lock. Shared across bodies, frozen/joint plasticity,
  and BC/PPO/resume; learned parameters and neural state remain independent.
- CPU preparation CLI. Paid qualification refuses cache misses.
- All six source/body pairs come first, then conventional controls, then
  topology/plasticity ablations. Per-case and overall deadlines; failures do not
  prevent other cases while time remains. Strict source/plan/script/checkpoint
  validation on qualification resume.
- Separate detached pod-stop watchdog. Requires local credentials before worker
  launch; verifies it is armed; stops on deadline, completion or lost heartbeat;
  retries control-plane failures; confirms provider `EXITED` with no runtime.
  Checks pod boot identity to avoid stopping a later job. Parent stop fallback if
  the watchdog dies. No pod creation/start/deletion capability.
- SSH driver verifies the remote pod ID, archives even failed qualification
  attempts, retrieves and hashes artifacts, and rejects stale success records.
- Phase/setup timing logs. BC curves now preserve cumulative elapsed time rather
  than overwriting it with evaluation duration. The old BC curve `wall_seconds`
  values must not be used as cumulative compute observations.
- Current scope remains BC→PPO only, two embodied tasks and five paired seeds;
  the training-regime comparison and temporal runs remain deferred.

## Measured preparation improvement

Full BANC degree-rewired preparation (175,401 neurons, 13,542,180 edges, ten swap
attempts per edge, strength preservation), measured locally:

- Cold preparation and cache publication: **73.0247 s**.
- Warm, checksum-verified cache load: **0.1828 s**.
- Approximately **400×** for this preparation operation, not end-to-end training.

See `banc-cache-benchmark.json`. The benchmark created the local cache under
`data/graphs/.prepared`; it performed no learning or GPU/cloud work.

## Verification

- `compatibility-tests.log`: 257 regression tests passed after caching and the
  first launch fixes.
- `final-focused-tests.log`: 21 focused checkpoint/resume, cache, launch and BC
  timing tests passed after the cumulative-clock correction.
- `launch-tests.log`: 19 final launch/watchdog/transfer checks passed, including
  an actual detached local process with a fake provider and simulated SSH/SCP.
- Cached/uncached actions and gradients are bit-identical on test graphs for
  real/rewired and frozen/joint conditions. Cache corruption is rejected.
- Ruff, shell syntax and `git diff --check` passed.
- Compiled local plan: `runs/learning-core-fixed-local-v1/plan.json`; only
  `bc_ppo` is enabled, with hover and worm locomotion, seeds 0–4, 270 comparison
  cells and 18 qualification cases. No worker was launched.

Runtime source fingerprint:
`683ff8b711d4c46519e01b06bfea4b6b882d3dc6d3a902d762a91f362165a309`.
Local plan fingerprint:
`e9b17ce41bd2a19e77aa3345da9923a44386fb35aa9e15e9dcc829062bfbb8df`.

The shutdown guard was **not tested against a live provider**. Before any future
paid deployment it needs a Pods read/write API key, an always-on supervising
host, and live shutdown verification. A sleeping supervisor or API outage can
delay stopping; retries persist and are never reported as verified success.
See `../../PAPER_LEARNING_CORE.md` for the updated workflow and provider docs.
