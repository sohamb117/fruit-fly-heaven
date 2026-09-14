This plan was frozen from one read-only SQLite snapshot after exactly 32 accepted training jobs, with generation 4 created but unevaluated and no active lease. It does not execute or upload evaluations.

The original declaration and validation protocol remain unchanged. [The pre-held-out amendment](../validation-protocol-amendment-001.json) retains generation 0 versus generation 4 as the primary comparison and adds one secondary head selected by maximum score among those 32 training jobs, with lexical job ID breaking ties. The selected job is `g3-p0-neg`; it is a single perturbed training candidate, not the generation-4 center.

[plan.json](plan.json) alternates `g0`, `g4`, and `training-best` for each predeclared seed, in order: 490888, 590888, 690888. All nine jobs request the full eight-second landing stage, with the same pinned Dawn Metal backend and criteria-3 configuration. Each parameters array is inserted verbatim from stored SQLite JSON. The [database snapshot](database-snapshot.json) preserves the exact strings and selection inventory; it excludes lease tokens and contributor identities.

Once the existing evaluator has finished all nine jobs, generate the offline report:

```sh
python3 scripts/validate-flight-checkpoints.py report reports/flight-development-v6/held-out-plan/plan.json reports/flight-development-v6/held-out-evaluation reports/flight-development-v6/held-out-comparison
```

Use the actual evaluator output directory if it differs from the example. The reporter refuses missing, duplicate, cancelled or mismatched results and refuses to overwrite a prior report. It validates the full horizon or an allowed physical failure; legitimate early crashes are retained. Configuration, source pins, backend, assignment, initial state and objective criteria must match.

Eight synthetic guard tests pass:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reports/flight-development-v6/validation-helper-tests.py
```

These tests validate report behavior only. They run no brain, muscle or body simulation. Held-out airtime, takeoff, landing, termination and score are reported separately for the primary center and secondary selected-trial comparisons. Parameter movement alone cannot establish learning; a later crash alone cannot establish controlled flight.
