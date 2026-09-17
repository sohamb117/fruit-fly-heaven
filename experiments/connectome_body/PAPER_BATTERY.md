# Drone and temporal benchmark expansion

The authoritative scope is [GOAL.md](../GOAL.md), additional benchmarks D, T and
analysis M. This extends the ten biological experiments; their fly/worm/fish
native-pair estimand remains unchanged.

The implementation includes all **20 drone tasks** and **25 temporal variants**
(the 24 requested families, with NARMA-20 and NARMA-30 separated). The core selection
contains seven drone tasks representing the six requested capability groups
(delay and dropout are separate), and eight diagnostic tasks. Severity grids
expand the number of conditions without adding new task families.

## Paper targets

| Evidence | Figure / estimand |
| --- | --- |
| Engineered control repertoire | Substrate × drone task held-out scores, success, physical errors, capacity and experience frontiers |
| Temporal repertoire | Substrate × diagnostic scores, memory-capacity curve, nonlinear memory, timing and forecasting error |
| Robustness | Delay/dropout/noise/actuator-latency/degradation curves; separately trained and fixed-policy zero-shot curves explicitly distinguished |
| Topology utility | Seed-paired real-minus-degree/community/random/no-brain/RNN/GRU effects with seed-cluster intervals |
| Mechanistic prediction | Nested source-substrate-held-out prediction of task-relative drone performance, versus size/capacity/task baseline |
| Specificity | Eight predeclared capability-to-task contrasts, whole-source permutations and Holm correction |
| Causal use | State resets and recurrent-edge removal on selected diagnostic and drone policies |

A missing or unqualified input is not a poor score. Predictions are reported as
unidentifiable when fewer than four eligible independent source groups have the
required complete measurements. BANC/MaleCNS are independent reconstructions of
one species, not two independent species. All rewires, null realizations,
capacities and training seeds of a source stay together in prediction folds.
Conventional controls are descriptive comparisons, not extra biological samples.

## Model and task semantics

`compatibility/drone.py` implements a six-rotor planar rigid body in SI units:
1 kg nominal mass, 0.22 m arms, alternating rotor spin, diagonal inertia
(0.018, 0.018, 0.032) kg m², up to 5 N per rotor, first-order motor lag 0.04 s,
linear drag, gravity and body-frame angular dynamics. Control interval is 20 ms
with four 5 ms semi-implicit integration substeps and rotation-exponential
orientation updates. Every controller commands the same six rotor thrusts in
[-1, 1]. No hidden flight controller or state estimator is in the policy path.

The 46-dimensional observation includes measured pose/velocity/body rates,
exogenous current position/velocity/heading commands, eight planar range rays,
explicit colored-target locations/cue, mission phase and a sensor-valid bit.
There is no frame stack, previous-action input or target-derived clean-state
error that bypasses sensor delay/dropout. Color navigation receives the two
candidate positions and cue, **not** the selected target position. Range and
symbolic color observations do not demonstrate image perception.

Circle, figure-eight, ascending/descending spiral and natural cubic-spline paths
have deterministic split-separated scenarios. Gust impulses, correlated
Ornstein–Uhlenbeck wind, delayed sensors/commands, dropout, sensor noise, one-rotor
weakness/failure and mass/inertia/COM changes are distinct task dynamics.
The failure task requests controlled descent/landing, not impossible guaranteed
full-attitude hover. Static hover after a rotor failure is feasible at a boundary
allocation with another rotor near zero: rank alone is not a robustness margin.

Ground and platform contact use a declared simple landing rule: radius 0.25 m,
touchdown speed below 0.6 m/s and tilt below 0.3 rad. Obstacle collision uses a
0.1 m drone radius; ground center height is 0.12 m. Contacts are task events,
not a high-fidelity contact solver. Propeller inflow, battery voltage, ground
effect and flexible airframes are omitted. The power metric is integral of
thrust^1.5, **not joules or measured electrical power**.

`qualify-drone` checks force balance, free hover drift and static single-failure
allocation, plus privileged analytic PD recovery for hover, attitude and gusts.
That reference reads clean simulated state only for qualification. These checks
establish modeled mechanical consistency; they neither validate hardware fidelity
nor prove learned task success.

Default position/attitude success requires stability during at least 80% of the
full episode without crashing. Navigation, color, landing and mission tasks use
the explicit objective-completion predicates in `DroneBody.metrics`. Errors,
recovery censoring, settling, overshoot, survival, trajectory lag, action variance,
path length, power proxy and touchdown speed are retained per episode.

## Diagnostic measurement

`temporal.py` generates independent full train/validation/test trajectories from
named seeds. All substrates use identical data files. The encoder sees current
low-dimensional signals and explicit task cues only; the substrate is the sole
learned persistent state. Primary encoder/readout family is `mlp_linear`, shared
with the drone battery; the readout is linear and parameter accounting includes
structural ports. Width is bounded by 5k/20k/80k interface ceilings.

Targets use training-only mean/std for regression. Classification uses binary
cross-entropy logits and reports balanced accuracy, precision/recall and false
alarms. Regression reports per-output MSE, NMSE, R² and squared correlation;
linear memory capacity sums per-delay squared correlations. Forecasting and
denoising also report persistence/noisy-observation baselines. Target masks
exclude warmup and unqueried recall periods. Frequency has randomized phase and
amplitude; phase comparison has an explicit reference channel.

NARMA-10 uses y[t+1] = 0.3y[t] + 0.05y[t]Σy[t−i] +
1.5u[t−9]u[t] + 0.1, with u uniform on [0, 0.5]. For orders 20 and 30, beta is
explicitly scaled to 0.5/order: these are **declared scaled-feedback variants**,
not claims to reproduce every published NARMA-20/30 convention. Divergent data
fail instead of being clipped. The NARMA-10 reference is
[Jaeger et al., multiple-timescale ESNs](https://eprints.whiterose.ac.uk/id/eprint/170213/7/fams-06-616658.pdf).

Mackey–Glass uses beta=0.2, gamma=0.1, exponent=10 and delay=17, Euler integration
at 0.1 with output interval 1 after burn-in. Lorenz uses sigma=10, rho=28 and
beta=8/3 with RK4 at 0.01, sampled every 0.05, after burn-in. Independent initial
histories/states are generated per split; future targets are never fed back to
the controller. See the [Mackey–Glass equation](https://doi.org/10.4249/scholarpedia.6908).

`temporal_training.py` trains through the same `Controller` used by recurrent
PPO. It carries neural state and replays one preceding TBPTT chunk with gradients,
then scores only the current chunk. This lets the first rhythm-continuation loss
reach the preceding input cue. Record **unique input timesteps**, optimization
exposures, extra replay timesteps and optimizer steps separately. These units are
not drone environment interactions. Checkpoints commit after complete sequence
batches, including model, optimizer, RNG, cursor, selected weights and curves.
Code/data/controller identity must match for exact resume.

## Runbook

Run from `experiments/connectome_body` with the existing environment; no sync or
new cloud provisioning is needed. The example cache path is writable locally.

```bash
# Freeze the complete catalog; performs no training.
uv --cache-dir /private/tmp/connectome-body-uv run --no-sync python -m connectome_body.compatibility.battery plan --config configs/paper/battery.json --output runs/battery-full-v1

# Or compile the core task selection with the same controls and severity grids.
uv --cache-dir /private/tmp/connectome-body-uv run --no-sync python -m connectome_body.compatibility.battery plan --config configs/paper/battery.json --output runs/battery-core-v1 --core

# Generate and checksum all shared diagnostic trajectories.
uv --cache-dir /private/tmp/connectome-body-uv run --no-sync python -m connectome_body.compatibility.battery datasets --plan runs/battery-full-v1/plan.json

# Inspect condition IDs, readiness, parameter counts and shared execution IDs in plan.json.
# Run exactly one selected condition, with a 30-minute allowance on this machine.
uv --cache-dir /private/tmp/connectome-body-uv run --no-sync python -m connectome_body.compatibility.battery worker --plan runs/battery-core-v1/plan.json --condition CONDITION_ID --max-runs 1 --max-seconds 1800

# Repeating the worker command resumes latest.pt or skips a finished run.
# The time cap is checked at safe rollout/sequence-batch boundaries, not a hard kill.

# Fixed selected drone policy: paired zero-shot severity plus acute reset/edge tests.
uv --cache-dir /private/tmp/connectome-body-uv run --no-sync python -m connectome_body.compatibility.battery evaluate --plan runs/battery-core-v1/plan.json --condition CONDITION_ID --episodes 20 --output reports/battery-core-v1/zero-shot.json

# Missing-aware results, frontiers, paired effects, heatmaps and predictive analysis.
uv --cache-dir /private/tmp/connectome-body-uv run --no-sync python -m connectome_body.compatibility.battery report --plan runs/battery-core-v1/plan.json --output reports/battery-core-v1
```

The supplied config requests CUDA for scientific runs and never silently falls
back to another device. For a deliberate local CPU study, copy the JSON and set
`device` to `cpu` **before** compiling a new plan. No tool provisions or starts a
pod. Do not launch the entire Cartesian catalog as an unreviewed budget request.

Default battery plasticities are frozen substrate and joint learning. The same
planner supports encoder-only, decoder-only and substrate-only by changing the
`plasticity` array; `battery_plasticity.json` supplies all five at 20k. Other
stateless families are configurable; primary diagnostics explicitly reject a
nonlinear readout unless the labeled sensitivity flag is changed.

Each graph has real, degree/community-rewired and matched sparse random variants.
RNN/GRU match the real graph condition's total active parameter ceiling, with
integer-width matching error recorded; no-brain matches the interface ceiling.
Identical controls share execution IDs across biological comparison anchors, so
one completed training run supplies all its paired references without being
counted as independent replication.

Primary BANC, MaleCNS and Cook chemical graphs are ready. Fish1's v700 soma-root
graph remains cataloged but primary runs are blocked by its independently audited
fragmentation; see [PAPER_DATA.md](PAPER_DATA.md). Its separate HMI circuit must be
explicitly declared as secondary, not silently substituted into the Fish1 row.
