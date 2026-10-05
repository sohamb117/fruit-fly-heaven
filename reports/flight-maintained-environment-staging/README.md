# Maintained-flight environment staging

`staged/web/training/environment.js` adds one diagnostic `maintained_flight` evaluator. No live environment, optimizer, coordinator, model or native code was edited. The original grounded evaluator remains verbatim, and its returned dispatch enters the new path only for `job.stage === 'maintained_flight'`.

The new path runs **0.5 seconds of live BANC feedback and native body motion while the explicit reset helper restrains only the root**, then **five scored seconds after release**. It creates the event reader once at neural time zero, accepts packet zero before airborne placement, and uses the same neural/body feedback-block routine for both phases. No brain, event, muscle, body-time or wing-phase reset occurs at release. Full-body nonwing dynamics stay enabled.

The maintained scorer and clock are the separate pure contracts in `../flight-maintained-staging/`. Root placement, warmup restraint and the continuing root-write audit belong to `../flight-airborne-reset-staging/reset.mjs`. This environment does not implement another controller or root reset.

## Inputs and loading

Declare a configured stage `{id:'maintained_flight', durationSeconds:5}`, enable the existing explicit `wingEventExcitation`, and provide:

```js
config.initialCondition = {
  schemaVersion: 1,
  profile: 'airborne-live-warmup-v1',
  warmupSeconds: 0.5,
  bodyVariant: 'full-native',
  bodyVariantHash: config.assets['/body-model/flybody-mujoco.json'],
  rootQpos: [0, 0, 3.5, 1, 0, 0, 0]
};
```

The body metadata digest must match the actually verified asset entry. The new path requires these three additional pinned asset URLs:

- `/diagnostic/maintained-flight-objective.mjs`
- `/diagnostic/airborne-reset-contract.mjs`
- `/diagnostic/airborne-reset.mjs`

The diagnostic runner must resolve them to the actual report-module file URLs. Serving the source verbatim at `/diagnostic/` without an import mapping is insufficient: the staged scorer's relative base-scorer import is relative to its report file location. No existing evaluator or server was modified here.

`ready()` is deliberately unchanged and remains an unadvanced initialization preview with the original criteria metadata. This staged diagnostic runner should call `evaluate()` directly, or treat `ready()` solely as initialization. The maintained evaluation result carries `MAINTAINED_FLIGHT_CRITERIA`; its release observation is the actual scored initial condition.

## Clock and observer contract

`body.time`, neural time, packet timestamps, `frame.time`, `frame.neuralMs`, and `onPhysicsStep.timeBefore` remain **absolute**. Release occurs at native time approximately 0.5 seconds / neural time exactly 500 ms. Returned `simSeconds`, `steps`, frame `simSeconds`, and `episodeTimeSeconds` describe **only scored time**, starting at zero. Warmup counts, total blocks, absolute native/neural clocks and `releaseNativeTime` are separate explicit fields.

The helper's `finish({neuralTimeMs})` runs at the fully consumed warmup boundary. The clock then checks the actual adapter's pending state once, event/body/neural alignment, zero body remainder, removed root restraint, no applied force and the independent root-write count. Each scored block checks the same clock/force/write invariants. Pending coverage thereafter uses the adapter's source invariant that `elapsedMs === observedMs` means its interval has been consumed; no repeated adapter-history snapshot is allocated.

`onInitialState` runs **once at scored release**, after `finish()`, clock validation and fresh score construction. It includes `phase:'scored-release'`, release clocks, episode time zero and warmup metadata. `onPhysicsStep` runs during both phases, with:

- `index`: total block index, including warmup;
- `episodeIndex`: null during warmup, then zero-based scored block index;
- `phase`: `warmup` or `scored`;
- absolute `timeBefore` and `neuralMs`;
- episode-relative time and release time.

Thus warmup physics callbacks precede the release initial-state callback. **The existing time-zero motor-replay recorder is incompatible with this contract.** `job.captureMotorReplay:true` is explicitly rejected. Custom observers must understand maintained-flight release; a diagnostic observer exception is an error, never a completed flight evaluation.

All exits restore the helper's native wrapper before disposing the body/brain. Errors after a completed native block report actual final clocks rather than the last successful observation. Warmup never enters scoring windows or earns takeoff/landing credit.

## Verification

```sh
node reports/flight-maintained-environment-staging/build.mjs
node --check reports/flight-maintained-environment-staging/staged/web/training/environment.js
node --test reports/flight-maintained-environment-staging/environment.test.mjs
```

Nine tests pass. They use the actual staged environment, event reader, maintained scorer and reset clock with mocked brain/body/reset operations. Coverage includes original grounded trace/source parity; all 250 warmup plus 2,500 scored blocks; exactly one packet zero; continuous absolute clocks and native-state accumulation; sequence/event ordering; physical early termination; warmup cancellation; cleanup and true clocks after observer errors; pending/root-write guards; invalid metadata/event/warmup declarations; and incompatible replay capture.

These tests establish scheduling and lifecycle behavior. **No MuJoCo, full BANC, GPU or body simulation was run by this agent.** Actual native held-root warmup and release still require the parent-owned diagnostic execution. `source-pins.json` and `validation.json` record the staged evidence.
