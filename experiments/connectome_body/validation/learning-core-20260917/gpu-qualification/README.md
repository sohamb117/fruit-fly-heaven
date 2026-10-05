# BC-to-PPO cross-body qualification — 2026-09-17

The current launch scope is BC→PPO only: BANC, MaleCNS and C. elegans across
FlyBody hover and published worm locomotion; real/degree-rewired topology;
adapters-only/joint plasticity; learned GRU/RNN and adapter-only controls;
one MLP/linear adapter family and five paired seeds. The training-regime study
and temporal training are deferred. No full-budget scientific run was launched.

Pod: `2tmnut86nzr086`, RTX 4090, listed compute price $0.74/hour.
Remote plan: `runs/learning-core-cuda-v1/plan.json`.
Plan fingerprint: `55a0471755ae5700520d7dc6c47eb1a6e75d2d5ea032c061e5811ae4f0899207`.
Runtime source: `2090df96dcef6943276803ca29d9485b50adeaab0ea7027c7b826a1f019ea020`.

## Measured outcome

The qualification started at 20:55:34 UTC and stopped after 1200.77 seconds.
Three of eighteen cases completed and passed. The fourth reached the deadline.
The qualification gate is **not passed**; remaining cases are untested.

| Substrate and condition | Body | Outcome | Case wall seconds | Peak allocated GPU MiB |
| --- | --- | --- | ---: | ---: |
| BANC real, adapters only | FlyBody | passed | 81.35 | 636 |
| BANC real, joint plasticity | FlyBody | passed | 191.57 | 1654 |
| BANC rewired, adapters only | FlyBody | passed | 457.88 | 636 |
| BANC rewired, joint plasticity | FlyBody | deadline after BC and one PPO update | censored | not recorded |

Each passed case completed two BC and two PPO updates, interrupted/resumed each
stage, verified finite actor coefficients, and verified the exact selected BC
checkpoint used for PPO initialization. These are smoke checks, not evidence
for learning performance, topology benefit or cross-body generalization.

BC optimization used about 0.54–2.77 seconds per completed case. Most case wall
time was outside those updates. Controller construction regenerates topology
and graph features at each BC/PPO/resume call; the rewired frozen case took
about 5.6 times the real frozen case. Cache/measure deterministic preprocessing
and distribute qualification coverage across bodies before another GPU attempt.
Do not extrapolate full-training throughput from the tiny PPO timing samples.

## Artifacts and shutdown

`cross-body-qualification-20260917.tar.gz` contains all qualification logs,
datasets, manifests, checkpoints (including the interrupted fourth case),
the remote plan, and setup logs. Its remote and local SHA-256 agree:

`5ba7d68d0b59c1b65036e0b2e63a04d947d956d5b130ce9866e5e4b903b19bb7`

The extracted artifacts are under `extracted/`. Six completed-case BC/PPO
checkpoint hashes were individually checked against their qualification records
in `local-checkpoint-verification.json`.

`source.tar.gz` is the uploaded source bundle; `deployed-qualifier.py` is the
subsequent eighteen-case qualifier that superseded the version in that bundle.
Its SHA-256 is `e6615dbd56aa003e2b51fd4ea4254c25013799c3480fd86fc1d799d536cbdd90`.

The training deadline did not stop the pod. Compute remained billable for roughly
an additional hour while the pod was idle. This was a shutdown-control failure.
After the archive download and hash verification, the pod was stopped; a live
RunPod read confirmed `EXITED` with no runtime by 22:22:09 UTC. Persistent disk
storage remains. A verified independent pod-stop deadline must be installed
before another paid launch; a training-process timeout alone is insufficient.
