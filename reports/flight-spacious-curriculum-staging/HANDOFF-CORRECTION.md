# Plan preflight corrections

Root's first final-plan launch was rejected before physics because the inherited configuration retained several legacy stage descriptors. The spacious environment intentionally requires `stage: maintained_flight` and a single maintained-flight descriptor. The pure source tests and manifest audit did not validate this inherited stage-list combination.

Root owns the corrected config-only `plans-v2-single-stage` copies. They preserve every frozen executable source and the existing failed zero-row result. Use root's corrected plan, not `plans-v1-final`, for execution. The older `plans-v1` and `plans-v1-resolved` also remain preserved, superseded preparation artifacts as described in README.

`validate-native-geometry.mjs` is a new independent validator, not part of any running plan. It checks the single-stage requirement and every source/asset digest before allocating a model. It creates the original and spacious scenes sequentially, destroys each native pair before allocating the next, runs exactly one `mj_forward` per pair, and never calls `mj_step` or creates a neural engine/muscle instance. It compares body dimensions/mass/inertia/joints/actuators/geometry and reference transforms, checks native wall/apron/ceiling arrays, and measures edge-height mismatch from native heightfield samples. It has only been syntax checked by this agent; root must run it while the native rig is free.

```sh
node reports/flight-spacious-curriculum-staging/validate-native-geometry.mjs CORRECTED_PLAN_JSON NEW_REPORT_DIRECTORY
```

The static pose uses metadata neutral joints and the configured airborne root pose. It does not reproduce the live 500ms warmup state, and its sampled geometry checks do not establish dynamic seam-contact behavior or stable flight.
