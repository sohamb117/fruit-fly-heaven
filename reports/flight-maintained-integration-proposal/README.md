# Full-body maintained-flight integration proposal

Prepared and reviewed; **not applied**. [canonical-maintained.patch](canonical-maintained.patch) changes only four training modules. It packages the tested full-body 0.5 s live-BANC warm-up and 5 scored seconds under canonical `/training/*.js` paths. It includes no reduced-body adapters, changed mouth-frame logic, power parameterization, configuration, UI or deployment changes.

The original grounded evaluator remains verbatim. All scoring, native reset, event timing and cleanup behavior matches the frozen diagnostic implementation. The only changes to those four diagnostic sources are import/asset paths: the environment verifies `/training/maintained-flight-objective.js`, `/training/airborne-reset-contract.js` and `/training/airborne-reset.js`, while loading them through module-relative imports. The scorer imports `./flight-objective.js`; the clock contract imports `./maintained-flight-objective.js`. This permits normal browser and Node module resolution without diagnostic loader rules.

The patch passes `git apply --check`. All **28 adapted original tests pass**: 9 environment scheduling/resource fixtures, 13 scorer/reset-clock tests and 6 native-reset mock tests. Four syntax checks pass. Fixture changes only point the existing tests at the proposed modules; the scorer's unchanged base dependency is resolved by the test-only loader. No native, muscle-WASM or neural simulation was performed. An independent review verified every provenance input/output hash and the exact path-only source transformations.

```sh
node --import ./reports/flight-maintained-integration-proposal/test-loader.mjs --test reports/flight-maintained-integration-proposal/environment.test.mjs reports/flight-maintained-integration-proposal/reset.test.mjs reports/flight-maintained-integration-proposal/maintained-flight.test.mjs
```

`provenance.json` records source/fixture/patch hashes and the precise substitutions; `validation.json` records the completed checks. `prepare.mjs` reproduces the proposal while requiring the frozen input hashes and unchanged live environment. It does not apply the patch.

## Remaining integration boundaries

- The canonical files must exist in the executing checkout before using the standard Node contributor. It intentionally verifies local files against the served manifest and has no executable development-bundle override. Staging by itself does not activate this proposal.
- After a deliberate source integration, regenerate the training manifest and create a **new local config/run namespace**. `scripts/prepare-training-manifest.mjs` already discovers these literal relative imports; no discovery-code change is needed. The config must declare the `maintained_flight` stage, duration 5, the existing wing-event interface and the explicit full-native initial-condition profile with 0.5 s warm-up and exact metadata hash. No config has been changed here.
- Contributor, worker, client and coordinator use scored `simSeconds` and `steps`; the setup half-second stays outside those values. Absolute native/neural/event clocks and release times remain separately recorded. Full-horizon `stage_success`/`time_limit` and the existing physical-failure reasons match coordinator acceptance. Heartbeats and cancellation continue during warm-up. Read-only integration review found no code blocker in these paths.
- `ready().criteria` remains the legacy criteria object, preserving the tested source; actual maintained evaluation results contain the maintained criteria. Current contributor/client acceptance does not use the ready criteria. Initial ready preview also remains the existing grounded setup preview. These are informational limitations, not evidence of maintained-flight success.
- Schema-2 guarded jobs still require the pinned native Dawn execution class. Ordinary browser workers can evaluate the maintained stage, but their results do not satisfy that existing guarded provenance policy. Broader public heterogeneous training requires a separate coordinator decision; this proposal does not relax it.
- The staged patch and mocked checks do not replace an actual run through the standard contributor after source/config integration. The parent controls that validation and any later promotion. All checkpoints remain unverified unless the existing independent behavioral process establishes otherwise.

The maintained stage measures flight after explicit airborne setup; it does not give takeoff or landing credit. Success requires completing the full five-second horizon while currently satisfying the existing continuous-flight criterion, not five entirely qualified seconds. Existing pure tests preserve that distinction.
