# Local implementation validation — 2026-09-16

Implementation checks passed. The paper experiment catalog has no completed scientific training runs; this report makes no claim of biological superiority or connectome–body compatibility.

## Saved checkpoint

[BANC update 8,475](../../checkpoints/banc-4090-update-008475/latest.pt) is local. All five entries in its `SHA256SUMS` pass. The checkpoint includes adapter/optimizer/RNG state; source graphs and the original runtime remain external dependencies.

## Checks and artifacts

- **223 tests passed, zero failures, zero skips**, including the existing suite and new sparse-gradient, native-body, checkpoint-resume, null-control, planning and analysis checks. [JUnit](validation/paper-implementation-20260916/tests.xml), [test log](validation/paper-implementation-20260916/pytest.log).
- Ruff passed. The one pytest warning is PyTorch's sparse-CSR beta notice, not a failed check.
- Native worm and fish mechanical qualification passed, with finite forward/reverse movement, steering and loss of propulsion under actuator failure. Worm posture and fish vertical actuation also passed. [Worm probes](data/paper-bodies/worm/qualification-v1.json), [fish probes](data/paper-bodies/fish/qualification-v1.json).
- Full native task interfaces instantiate: four FlyBody tasks, three worm tasks and three fish tasks.
- Standalone plot layout was inspected on explicitly marked software fixtures. Fixture curves are not research results.
- Existing tracked files, dependency pins and legacy campaign code were unchanged. No cloud resources were provisioned or started.

## Frozen catalog and initial selection

The [plan](runs/paper-plan-v1/plan.json) contains **24,180 unique conditions** and **16,924 post-training jobs**: 17,670 conditions are ready, 198 have infeasible capacities, and 6,312 await inputs. All ten experiment sections are present in the [unmeasured report](reports/paper-v1/paper-report.json). The older `paper-plan-draft-20260916` is a superseded draft.

The [BANC hover selection](runs/paper-plan-v1/banc-hover-mvp.json) contains exactly 24 ready conditions: real, degree-rewired, N-and-M-matched fixed random and adapter-only, at nominal 5k/80k capacity, with three paired seeds. It launches nothing.

- Training allocation: 24,000,000 environment interactions.
- Development/test allocation: at most 158,400,000 additional interactions, before separate post-training jobs.
- Worker time limits stop at checkpoint boundaries, not at a provider billing deadline. Review this substantial evaluation allocation before scheduling the pilot.

## Full-BANC local CPU feasibility

Apple M2 Pro CPU, one Torch thread, float32, 175,401 neurons and 13,542,180 directed edges, two neural substeps per body decision. The adapter has 4,912 trainable parameters. Joint training additionally differentiates through every permitted edge. Measurements include two timed repeats after one warm-up; they exclude physics and use a synthetic regression loss.

| Plasticity | Batch | TBPTT length | Forward transitions/s | Optimized transitions/s | Process peak GiB |
|---|---:|---:|---:|---:|---:|
| adapters | 1 | 2 | 3.32 | 1.51 | not recorded |
| adapters | 4 | 8 | 11.90 | 4.93 | 1.99 |
| joint | 1 | 2 | 2.42 | 0.87 | not recorded |
| joint | 4 | 8 | 11.00 | 3.05 | 2.06 |

These are controller-throughput observations, not achieved PPO training rates or evidence of useful control. Raw records and the exact batch-4 benchmark source archive are in [data/paper-calibration](data/paper-calibration). CUDA throughput and empirical RNN compute matches still need measurement on the target hardware.

## Remaining execution inputs

Fish1 requires authenticated CAVE access, a chosen materialization and declared reconstruction coverage. The exporter/importer and failure/resume checks are implemented; no synthetic fish graph substitutes for the missing release. Fish1 anatomical oracle/module conditions additionally require provenance-backed labels.

The fish body is explicitly a simZFish-derived 3D extension, with documented fin/fluid parameters and an audited inertia repair. Native mechanical checks do not establish biological fidelity. BANC, MaleCNS and Cook graphs and their prepared anatomical controls are available locally.

Running the scientific study, obtaining Fish1 access and measuring GPU compute matches remain execution work. No results from the earlier imitation campaign are counted as evidence for this new protocol.

## Reproducibility handoff

Implementation fingerprint: `6c4890361816a4d361dd453d3b0f51c4ba7be51f762c159b2e0b509a9b21b675`.
Plan fingerprint: `57e274a1ffb49a5defb60da6f7fbe314dcdbccb33ee25f531b8e99ca11fc5c2f`.

[Source archive](validation/paper-implementation-20260916/sources-6c4890361816a4d361dd453d3b0f51c4ba7be51f762c159b2e0b509a9b21b675.tar.gz) contains code, dependency pins, paper configurations, tests and documentation. Large graph arrays, body/flight assets and run artifacts are external, checksummed dependencies; use the preparation commands in [PAPER_README.md](PAPER_README.md). [Machine-readable validation](validation/paper-implementation-20260916/validation.json) records file hashes and the environment.
