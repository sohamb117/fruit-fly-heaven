# Local spacious maintained-flight training recipe

Prepared only: no canonical edits, database/server, backend installation or simulation. The proposed fresh namespace is `reports/flight-maintained-amplitude-local-v2`, endpoint `http://127.0.0.1:7862/`.

The template now uses the independently reviewed combined patch `27260780d4b7f99e84b1a3ffd667bcdd8c1e6d47c852ae6fd021df1fb60a47ce`, all 11 exact changed module hashes, and spacious full-body metadata `a842121877df5cfaa050a20298d98603f011ca7a576c93635eecf7dc601daf7a`. There are 59 pinned assets, including the preview and scene helper. The config contains **only one stage**, `maintained_flight`; extra grounded stage entries violate this scene's explicit contract.

**The initial amplitude is provisionally 0.80, awaiting the parent's 0.80/0.81 native comparison.** Frequency multiplier is exactly 1 (coordinate 2 is zero); the other 25 initial coordinates match the migration artifact. If 0.81 is chosen, update only `config.parameters[0].initial` to `Math.log(.81)` and the source contract's template SHA. Mark the contract `reviewed-ready` when that decision is final. No code/model/source hashes need changing. The finalizer currently refuses the provisional status. Backend hashes and the config/model fingerprint deliberately remain unresolved until installed-source verification.

The model keeps the full articulated native body and the previously tested diagnostic XML with zero phenomenological claw adhesion, ordinary collisions and friction. This is not the structurally reduced body. It uses live BANC, exact wing motor events, the ionic DLM profiles and existing tegula feedback; mechanical haltere feedback is absent. No neural, muscle or sensory parameter changes are introduced by this recipe.

The root warms at `[0,0,3.5,1,0,0,0]` for 0.5 seconds while neural/muscle/native histories continue. It is then fully released for five scored seconds without root correction. The spacious scene has radius 50 cm, ceiling 50 cm and floor cap radius 6.5 cm, retaining central fruit/odor. Version 2 scores vertical speed from a fresh contact-free 50 ms COM height window, keeping the −1 cm/s threshold with 1e−9 cm/s numerical tolerance. Other powered-flight, attitude, support and contact gates remain unchanged. Full-horizon success requires a qualifying final bout of at least one second; it does not claim five entirely qualified seconds, takeoff or landing.

## Search and acceptance

The new first-coordinate meaning is amplitude after fixed activation normalization `exp(0.693147)=1.999999638880142`, bounded above at 1. This is a numerical model choice, not a physiology measurement. Historical coordinates, optimizer states, scores and checkpoint identities are not imported; the explicit migrated initial vector starts a fresh run.

Use global sigma 0.10 and search scales `[0.10,0.25,0.05,1,…,1]` for amplitude, deployment, frequency and 24 steering gains. With provisional amplitude 0.80, a perturbation is `0.8*exp(0.01*epsilon)` before bounds:

| Coordinate | One-sigma interval | Approximately 95% interval |
| --- | --- | --- |
| Amplitude | 0.79204–0.80804 | 0.78447–0.81583 |
| Deployment multiplier | 0.67630–0.71097 | 0.6603–0.7282 |
| Frequency multiplier | 0.99501–1.00501 | 0.99025–1.00985 |
| Steering gain relative to its center | 0.90484–1.10517× | 0.82201–1.21653× |

These are marginal Gaussian intervals, not hard bounds or joint 27-dimensional coverage. Parameter limits still clip candidates; several steering gains start at their lower bound. Three-sigma amplitude is 0.77636–0.82436. The exact calculations are in `search-ranges.json`; if 0.81 is chosen, multiply its amplitude intervals by 0.81/0.80.

Keep the schema 2 guard: eight antithetic search jobs nominate one best search job, then three fresh matched training seeds compare that candidate against the incumbent in six jobs. Replace the saved center only for strictly positive paired mean return. An identical nominee skips comparisons. Rejection preserves every stored coordinate; all checkpoints remain unverified. `learningRate` and `maximumUpdate` do not limit best-job nomination; sigma/scales and bounds do.

Stop after two generations: at most 28 accepted evaluations, or 16/22 if identical nominations skip comparisons. The backend plan sets `stopAfterGeneration:2` in addition to maxJobs 28, preventing a no-op from spilling into a third generation. A full evaluation uses at most 5.5 native seconds including warm-up: at most 154 for training plus 33 for the six final evaluations. Measure wall time from actual runs rather than extrapolating short crash episodes.

Optimizer seed 1888888 and reserved final seeds 2490888/2590888/2690888 remain fixed. Historical diagnostic/validation seeds are excluded; the reserved final seeds must not enter either search or acceptance jobs. Guard seeds are coordinator-generated training seeds, not held-out validation. The final comparison is exactly initial versus final incumbent on three paired reserved seeds, without selecting another best trial.

## Integration and source freeze

The following commands are prepared for later execution, after the parent finishes its native comparisons and chooses the initial amplitude. This report has not executed them:

```sh
git apply --check reports/flight-maintained-fullbody-integration/canonical-fullbody.patch
git apply reports/flight-maintained-fullbody-integration/canonical-fullbody.patch
node scripts/prepare-training-manifest.mjs
node reports/flight-maintained-training-plan/prepare-run.mjs reports/flight-maintained-training-plan/source-contract.json reports/flight-maintained-amplitude-local-v2
```

If the exact combined patch is already installed, skip applying it again. Do not also apply the earlier maintained-only or optional-reduced patch. The manifest command updates canonical asset pins; this local bundle, not that canonical config, selects the experimental stage/model.

The finalizer verifies the 59 reviewed asset hashes and their literal import closure, source model/metadata, prepared graph binaries, exact initial coordinates and scene agreement. It recomputes the local asset fingerprint/config hash and reads current native backend module/package-lock hashes only after source installation. It does not allocate Dawn, a brain or a body. It refuses an existing run directory. Actual backend execution separately verifies pinned native addon/package bytes and rejects software fallback.

Small runtime/tooling sources and exact config/model overrides are archived in the fresh run. Large graph arrays and native binaries are hash-verified in place to avoid duplicating hundreds of MB; retain those bytes until archived elsewhere. A hash is not a backup. The earlier original-bowl/v1/0.90 draft is preserved under `previous-original-bowl-draft/` and is not this experiment.

## Start and monitor the actual standard path

Port7862 had no listener when this recipe was prepared; check again immediately before starting. Run only one full-fly contributor/evaluator at once.

```sh
lsof -nP -iTCP:7862 -sTCP:LISTEN
.venv/bin/python -B scripts/serve-training-dev.py --port 7862 --database reports/flight-maintained-amplitude-local-v2/coordinator.sqlite3 --bundle reports/flight-maintained-amplitude-local-v2/bundle.json
```

In a separate terminal, obtain initial status and the generation0 checkpoint before contributing:

```sh
curl --fail --silent --show-error http://127.0.0.1:7862/api/training/status -o reports/flight-maintained-amplitude-local-v2/status-initial.json
curl --fail --silent --show-error http://127.0.0.1:7862/api/training/checkpoint -o reports/flight-maintained-amplitude-local-v2/checkpoint-initial.json
node scripts/contribute-training-native.mjs http://127.0.0.1:7862/ reports/flight-maintained-amplitude-local-v2/contributions 28 reports/flight-maintained-amplitude-local-v2/backend-plan.json
```

The standard coordinator owns all jobs and saves every accepted result and incumbent transition in the fresh SQLite database. The standard native contributor executes the same environment with Dawn/Metal, one fly at a time; it is not a local optimizer or a browser preview. Ordinary browser workers cannot satisfy this intentionally pinned native provenance guard. Its per-evaluation files retain actual backend, parameters, return, sparse frames and metrics; `parameter-updates.jsonl` records all27 coordinate changes when a generation finishes. A line saying `accepted` acknowledges a result, not necessarily a new checkpoint.

Use the existing read-only reporter during search/comparison and after each completed generation:

```sh
.venv/bin/python scripts/report-flight-training.py reports/flight-maintained-amplitude-local-v2/coordinator.sqlite3 --last-generations 2 --output reports/flight-maintained-amplitude-local-v2/training-progress.json
```

Inspect `completedByPurpose`, `acceptance.decision`, paired differences/mean, and `parameterUpdate.deltaByParameter`, changed count and L2. Compare airtime/success/failure metrics as well as return. A rejected proposal must leave all27 saved coordinates exactly unchanged. No parameter movement is expected while six comparison jobs are incomplete. A finite valid physical failure is data; cancellation/infrastructure errors must not be folded into the guard's means. For a clean pause, Ctrl-C the contributor or create its output `STOP` file. Do not delete `.pending.json` results to force restart; reconcile them with the coordinator first. Keep source/config bytes fixed while the run is active.

## Predeclared final comparison

After generation2, with contribution stopped:

```sh
.venv/bin/python scripts/report-flight-training.py reports/flight-maintained-amplitude-local-v2/coordinator.sqlite3 --output reports/flight-maintained-amplitude-local-v2/training-final.json
curl --fail --silent --show-error http://127.0.0.1:7862/api/training/checkpoint -o reports/flight-maintained-amplitude-local-v2/checkpoint-final.json
node reports/flight-maintained-training-plan/prepare-heldout.mjs reports/flight-maintained-amplitude-local-v2
node scripts/evaluate-training-native.mjs http://127.0.0.1:7862/ reports/flight-maintained-amplitude-local-v2/heldout-plan.json reports/flight-maintained-amplitude-local-v2/heldout
```

The plan builder checks both guard decisions are final, no third-generation results exist, and reserved seeds did not enter training. It compares the exact initial vector against the final incumbent, paired on three fresh seeds; no best-trial cherry-picking. The evaluator uses read-only coordinator requests and uploads no results. Report paired return, qualified airtime, full5s success and failure reasons; three seeds give limited evidence and do not establish the full food→approach→landing→feeding→takeoff task. If the guard retained the same center, report that explicitly rather than inventing learned improvement.

Stop the coordinator before copying its database for a final archive, or use SQLite's backup API while it is running; copying a WAL database file alone is not a consistent backup. Preserve the exact config, bundle, initial/final checkpoints, backend plan, declaration/source archive, accepted jobs and read-only evaluation artifacts together.
