# Validation record

Validated on 2026-09-15 with Python 3.12.1, macOS 26.3 arm64, and CPU execution. Dependencies are pinned in [uv.lock](uv.lock). Native checks use [FlyBody revision d015e9b](https://github.com/TuragaLab/flybody/tree/d015e9bfe441bd90ae431bac24c55cb74bdbce26), the stock walking body, 286 observations, and 59 actions.

**The software suite and full-graph integration have been exercised. Successful learned walking/reaching, a biological connectivity advantage, and a cross-species ranking have not been established.**

## Automated checks

The full suite passed **35 tests in 8.42 seconds** before the final checkpoint recovery repair. After that repair, all **17 tests in `test_training_and_analysis.py` passed in 14.60 seconds**, including the added regression for a crash after the final checkpoint but before derived JSON logs were written. The unchanged graph/policy and native body tests had already passed; these are separate test invocations, not a reported single 36-test run.

Coverage includes:

- Directed graph import, lossless neuron identifiers, membership/provenance checks, and graph fingerprints.
- Seeded null reproducibility, exact degree/sign/weight invariants, isolated nodes, and rejection of duplicate/self edges.
- Adapter capacity accounting, disjoint ports, absent connectivity, frozen graph weights, and sparse backward agreement with a dense reference.
- PPO episode boundaries, termination versus timeout bootstrap, finite optimization, exact CPU checkpoint continuation, and recovery of selected checkpoints and derived logs.
- Native MuJoCo state restoration for all three tasks, including integration state and the next trajectory.
- Plan readiness/identity checks, completed-run skipping, report generation, duplicate identities, seed clustering, censoring, and capacity summaries.

Ruff and shell syntax checks passed. PyTorch emits its sparse-CSR beta warning. No CUDA, Docker, or cross-device reproducibility test was performed.

Reproduce the current tests from this directory with:

```sh
uv run --frozen ruff check connectome_body tests scripts
uv run --frozen pytest -q
```

## Full graph checks

These counts describe the prepared snapshots after the filters in [DATA.md](DATA.md), not all segmented objects in the original releases.

| Prepared graph | Neurons | Directed weighted edges | Encoder gradient norm | Eight-step forward / backward |
|---|---:|---:|---:|---:|
| BANC v888 / v3 | 175,401 | 13,542,180 | 2.1683 × 10⁻⁷ | 2.44 s / 2.24 s |
| MaleCNS v1.0 / confidence 0.5 / traced neurons | 165,122 | 25,563,096 | 1.2821 × 10⁻⁷ | 4.50 s / 4.16 s |

Both preflights used one environment, a 5k actor ceiling, **4,682 actual trainable actor parameters**, and **zero learned connectome parameters**. Activities, actions, and gradients were finite. The nonzero gradient verifies an encoder-to-action path through frozen edges; it does not establish useful control. Timing is a local CPU measurement, not a hardware-independent benchmark.

Exact settings, source digests, canonical graph fingerprints, and port fingerprints are in [BANC preflight](validation/banc-preflight.json) and [MaleCNS preflight](validation/malecns-preflight.json).

## Native training smoke checks

Five runs completed actual native FlyBody rollout, PPO, checkpointing, validation selection, held-out evaluation, and applicable interventions:

| Substrate | Task | Actual actor parameters | Training interactions | Trainer wall time |
|---|---|---:|---:|---:|
| BANC, real graph | Balance | 4,682 | 32 | 243.54 s |
| MaleCNS, real graph | Walking | 4,682 | 32 | 282.53 s |
| Adapter only, no brain | Balance | 4,682 | 32 | 1.73 s |
| Adapter plus trainable GRU, no brain | Balance | 4,798 | 32 | 1.97 s |
| Conventional trainable GRU | Balance | 4,808 | 32 | 1.97 s |

Each has a separate 22,593-parameter critic and one held-out episode. Smoke episodes last only 32 decisions; their results are engineering evidence. No held-out numerical failures occurred. Trainer wall time includes evaluation and interventions but excludes initial model construction.

Both graph runs completed silence, per-step reset, no-recurrence, output permutation, and acute degree-rewiring evaluations. The full BANC rewiring accepted 91,281,528 of 135,421,800 proposed swaps; MaleCNS accepted 171,640,508 of 255,630,960. These are completed null-generation checks, not independently trained null performance comparisons.

The [machine-readable summary](validation/native-smokes.json) records result hashes and local run paths under `runs/validation/v2/`. Reports and PNG/PDF figures are in `runs/validation/v2-analysis/`. Analysis includes smoke evidence only when explicitly requested with `--include-validation`.

Two earlier diagnostic graph runs were stopped during a slow null-construction implementation and marked as superseded in their local `stopped.json` records. Their checkpoints were retained. The replacement completed runs had bitwise identical actor and critic states after the same 32 training interactions. The final null implementation also retains seeded golden-output tests.

## Initial task calibration

Three conventional-GRU runs completed **4,096 training interactions each**, with a 50k ceiling, 48,924 actual actor parameters, training seed 0, full 200-decision episodes, and eight final test episodes per task. This is one seed and **three of the pilot's 132 planned runs**.

| Task | Initial validation success | Final selected-policy test success | Test score | Selected checkpoint |
|---|---:|---:|---:|---:|
| Balance | 1.00 | 1.00 | 0.8869 | 3,072 interactions |
| Walking | 0.00 | 0.00 | 0.1183 | 4,096 interactions |
| Reaching | 0.00 | 0.00 | 0.0581 | 0 interactions |

Score is episode return divided by the full horizon. The reaching policy selected its initialization; completing optimization does not imply improvement. Walking and reaching did not certify success and remain censored at 4,096 interactions. These short runs do not show that the tasks are unlearnable.

Balance met its success criterion before learning, although its reward improved during training. Accordingly, **balance is calibration only; walking and reaching are the primary study tasks**. The main study uses training seeds 100–104 and separate scenario cohorts from the pilot. Before interpreting topology effects, a longer conventional-policy feasibility run must establish that the primary tasks can be learned under the shared protocol.

The [pilot summary](validation/pilot-calibration.json) includes exact scores, experience events/censoring, result hashes, and local paths. Full reports and figures are in `runs/launch/pilot-analysis/`. No connectome-versus-RNN effect can be estimated from these three RNN-only runs.

## Source identity and current plans

The native checks and three calibration runs used code fingerprint:

```text
5a82565622e9694e59d351cf0a645e19cde596b322b143fe6a04e9d6af9e52e2
```

Their exact package, tests, configuration, and dependency lock are preserved in [runtime-source-5a825656.zip](validation/runtime-source-5a825656.zip), with archive SHA-256 `b81e3bf0a6e105a3dba06756e7630fc064478f142bbd7ca88ceba7bf36ffe88c`.

The current code fingerprint is:

```text
2c52744ee3411a0c2c84a2015e00a3f7acc89364afcb3ed842949b9ec6718a68
```

The only runtime change after those native runs restores `learning_curve.json` and `updates.json` from the authoritative checkpoint when resuming. It does not change the policy, body, optimizer, or graph dynamics. The recovery tests above validate that final repair. Run identities deliberately remain distinct across code versions; earlier runs are not silently relabeled or resumed with modified code.

Current plans were generated under `runs/current/` using the current code and verified prepared graph fingerprints:

| Plan | Runs | Training interactions | Readiness |
|---|---:|---:|---|
| `pilot/plan.json` | 132 | 540,672 | Data ready |
| `fly-study/plan.json` | 1,080 | 1,080,000,000 | Explicit two-fly scope; data ready |
| `full-study/plan.json` | 2,040 | 2,040,000,000 | Requires Fish1 and cockroach graphs |
| `dynamics/plan.json` | 288 | 57,600,000 | Data ready |
| `signs/plan.json` | 36 | 7,200,000 | Data ready |

These are launch plans, not completed campaigns; evaluation adds further interactions. No worker was left running. Use the commands in [README.md](README.md) to generate fresh plans on another machine, with its recorded device and paths. The local data, downloaded body source, environment, run directories, and large checkpoints are excluded from Git; the small validation summaries and source archive are included in the suite.

## Outstanding scientific work

Establish primary-task learnability, then execute the capacity/experience comparisons with multiple seeds and retrained real/null/RNN/no-brain policies. Validate CUDA on the intended worker before the full campaign. Obtain a pinned Fish1 neuron/synapse export and verify a suitable cockroach release before launching the four-dataset matrix. The importer accepts the documented interchange, but neither missing dataset has been represented by a synthetic substitute.

The suite can test a generic interface under common assumed dynamics. Its current results do not establish native neural physiology, arbitrary-connectome control, statistical overfitting, or a connectome–body compatibility ordering. The [protocol](PROTOCOL.md) specifies the controls needed to evaluate those narrower claims.
