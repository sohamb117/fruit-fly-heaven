# Expanded-study implementation status — 2026-09-17

The authoritative target is [experiments/GOAL.md](../GOAL.md). Its ten biological
experiments now have the requested drone-control battery, temporal diagnostics
and diagnostic-to-control analysis alongside them. **No scientific training
comparison from the expanded study has completed.** Passing implementation and
mechanics checks does not establish a biological advantage.

## Implemented and checked

- Seven stateless adapter families, including the separate anatomical oracle;
  exact interface and total parameter accounting.
- Five fixed-adjacency plasticity regimes; real, degree/community/direction
  nulls and matched sparse random substrates; learned RNN/GRU and no-brain controls.
- Ten biological tasks across fly, worm and the documented simZFish-derived 3D body.
- All twenty requested drone tasks in one SI-unit six-rotor rigid-body environment.
- All 24 temporal diagnostic families, represented by 25 variants with NARMA-20
  and NARMA-30 separated. Immutable train/validation/test datasets are generated
  and hashed for every variant.
- Common recurrent PPO for embodied control; supervised diagnostic TBPTT through
  the same controller, with a linear readout, scored-target masks, independent
  splits, training-only scaling and causal input semantics.
- Atomic checkpoint/resume, exact restored trajectories and weights, paired
  zero-shot severity and acute state/edge interventions, censored frontiers,
  seed-paired control effects, missing-aware reports and capability heatmaps.
- Nested source-substrate-held-out prediction and predefined capability-specific
  task contrasts. Nulls, capacities and seeds stay with their source fold;
  insufficient independent inputs are reported as unidentifiable.

The [implementation map](PAPER_EXPERIMENTS.md), [measurement protocol](PAPER_PROTOCOL.md)
and [drone/temporal runbook](PAPER_BATTERY.md) map the requirements to runnable code.

## Current validation

**300 tests passed, zero failures, zero errors and zero skips**, in approximately
63 seconds. Ruff passes. The single warning is PyTorch's sparse-CSR beta notice.
The full suite includes 64 new drone/temporal/planner/analysis checks. It verifies
exact resume for the physical drone and diagnostic adapter-only, RNN, GRU and
plastic sparse-connectome controllers; every new task generator and drone
rollout; masked temporal targets; immutable data; split isolation; no clean-state
sensor bypass; control matching; and source-level prediction folds.

- [JUnit results](validation/paper-battery-20260916/tests.xml)
- [Machine-readable validation record](validation/paper-battery-20260916/validation.json)
- [Dataset identities](validation/paper-battery-20260916/datasets.json)
- [Drone mechanical qualification](validation/paper-battery-20260916/drone-qualification.json)

Runtime source fingerprint:
`92fa0f5815332f9d0b54caac27865e4f4b34abf56d247a8c46c49e24aeb6ea02`.
GOAL fingerprint:
`ef98f630428fb4781b2d7507e8aa9089fff191a6d9b9a0a9a2039c53ded58851`.

The ideal drone's trimmed hover drift is about 6.2e-14 m over two seconds.
Static single-motor-failure allocations satisfy the declared force/torque target
within 5.2e-6 in the allocation residual. This is a boundary trim, not a guarantee
of disturbance rejection after failure. The privileged analytic reference
survived hover, randomized attitude and gust probes; final position errors were
0.0072 m, 0.0349 m and 0.0039 m respectively. The reference is outside the learned
policy path. Contact rules and power proxies remain simplified model assumptions.

The worm and fish mechanics checks were refreshed after adding drone dispatch to
the shared body interface. Both pass with current code:
[worm v2](data/paper-bodies/worm/qualification-v2.json),
[fish v2](data/paper-bodies/fish/qualification-v2.json).
Earlier records are preserved. Qualification establishes finite modeled
mechanics/control authority, not biological fidelity or learned performance.

## Frozen catalogs and actual execution

| Catalog | Comparison cells | Distinct training runs | Ready cells | Blocked cells |
| --- | ---: | ---: | ---: | ---: |
| Full drone/temporal battery | 27,720 | 21,285 | 23,760 | 3,960 |
| Core capability battery | 10,584 | 8,127 | 9,072 | 1,512 |

These are configurable catalogs, **not queued or launched training jobs**.
Identical controls share an execution ID across comparison anchors, preventing
unnecessary duplicate training. Each worker invocation needs an explicit run
count and time allowance. Current reports contain zero measured scientific
conditions and do not render invented performance plots:
[full report](reports/battery-full-v1/REPORT.md),
[core report](reports/battery-core-v1/REPORT.md).

The original ten-experiment biological plan is refreshed separately using the
current native-body qualifications. Older v1/v2 plans remain historical snapshots;
v3 captured the temporary stale-qualification state. Current native commands use
`runs/paper-plan-v4`. Dataset preparation and catalog compilation launched no
training or cloud resources. GPU throughput, compute-matched widths and substantive
learning comparisons still require execution on the chosen hardware. The Docker
recipe has not been built or CUDA-tested.

## Biological input boundary

BANC, MaleCNS and Cook 2019 chemical C. elegans are prepared with pinned graph
identities, anatomy and task selections. Fish1 access and its entire v700 export
are complete: 29,474,316 physical synapses plus the separate labels. The imported
single-soma-root graph has 178,976 roots and 157,014 directed edges. Only 0.7994%
of raw synapses remain between selected roots, and 60.4053% of roots are isolated.
Independent endpoint checks and live count comparisons agree with the import.
Consequently its primary whole-connectome qualification is **false**, and both
planners retain that block. This is a reconstruction-coverage issue, not a login
or download problem.

A separate curated HMI circuit has 197 cells and 293 binary edges, with explicit
secondary-circuit scope. It is not silently substituted for a whole fish CNS.
[PAPER_DATA.md](PAPER_DATA.md) records the evidence and limitations. With only
three qualified biological source reconstructions, the four-source minimum for
nested temporal-to-drone prediction is not yet met; more rewires or training seeds
do not resolve that limitation.

## Preserved artifacts

Original source archives, validation snapshots, old BANC results, local checkpoints
and the downloaded update-008475 GPU checkpoint remain intact. The earlier
BANC-only FlyBody campaign is not presented as completion of the expanded study.
Credentials remain excluded from Git and source archives.
