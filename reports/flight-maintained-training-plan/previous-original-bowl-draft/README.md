# Local maintained-flight training recipe

Prepared only. No canonical source, configuration, database, server or simulation was changed by this work. The exact 27-coordinate proposal is `config.template.json`; its deliberately null backend/fingerprint fields make it non-runnable until `prepare-run.mjs` verifies the installed reviewed sources. This is a fresh local experiment, not a production deployment.

**Pending decisions:** the parent is staging a50ms COM-height-derived vertical-speed gate and a spacious maintained-flight scene, and may choose amplitude0.80/frequency1 instead of0.90/the historical frequency. The current template/source contract intentionally preserves the earlier0.90/v1/original-bowl proposal as a draft. The finalizer refuses it until a reviewed replacement contract is marked `reviewed-ready`. Update the exact initial27-vector, scorer/scene/template hashes, and model metadata together after those diagnostics; do not run this draft merely because its commands are available. The same log-space search scales at0.80 would give one-sigma amplitude0.79204–0.80804 and approximately95%0.78447–0.81583.

The primary model is the full articulated native body, using the exact XML/metadata in `reports/flight-power-parameterization-staging/plans-v1/`. It retains that diagnostic model's **zero phenomenological claw adhesion** and ordinary contacts/friction; it is not the structurally reduced six-wing-joint model. The task is `maintained_flight`, with live BANC/muscle/native histories warming for0.5s at root `[0,0,3.5,1,0,0,0]`, followed by5 scored seconds with no root restraint or reset. Existing powered-flight/attitude/contact/vertical-speed criteria remain in force. Success requires reaching the full horizon with the scorer's qualifying final bout; the task awards no takeoff/landing credit.

`activation-amplitude-v1` changes the first coordinate's meaning. Its fixed activation normalization is `exp(0.693147)=1.999999638880142`; the learned first multiplier is amplitude after that saturation, initialized at0.90 and bounded above at1. All26 remaining initial coordinates and bounds are copied exactly from the explicit migration artifact. No old database, optimizer state, objective values or checkpoint identity is imported. This normalization is a numerical model choice, not a physiology measurement.

## Bounded proposal

Use sigma0.10 and search scales `[0.10,0.25,0.05,1,…,1]` for amplitude, deployment, frequency, then24 steering gains. The initial amplitude perturbation is `0.9*exp(0.01*epsilon)` before bounds:

| Coordinate | One-sigma range from initial value | Approximately95% range |
| --- | --- | --- |
| Amplitude | 0.8910–0.9090 | 0.8825–0.9178 |
| Deployment multiplier | 0.6763–0.7110 | 0.6603–0.7282 |
| Frequency multiplier | 0.95869–0.96832 | 0.95410–0.97298 |
| Each steering gain, relative to its center | 0.90484–1.10517× | 0.82201–1.21653× |

These are marginal Gaussian intervals, not hard bounds or joint27-dimensional coverage. Three-sigma amplitude is0.8734–0.9274; six-sigma is0.8476–0.9557. Parameter limits still clip candidates. Several steering gains begin at their lower bound, so their antithetic changes are asymmetric. `search-ranges.json` contains the exact calculations.

Keep the existing schema2 guard: four antithetic pairs/eight search evaluations nominate the best search job; candidate and incumbent then receive the same three fresh training seeds/six evaluations. Replace the saved center only for strictly positive mean paired return. The same parameters skip comparison jobs; negative/equal means retain the incumbent exactly. Checkpoints remain `unverified`. `learningRate` and `maximumUpdate` are retained for schema compatibility but do not limit this best-job nomination protocol.

Stop after two generations: at most28 accepted evaluations, or16/22 if a generation's nominee is identical. `stopAfterGeneration:2` is essential; maxJobs28 alone can spill into a third generation after a no-op. A full successful evaluation uses5.5 native seconds including warm-up;28 evaluations have at most154 native seconds, plus at most33 for six held-out evaluations. Wall time must be measured from actual runs rather than extrapolated from short crash episodes.

Optimizer seed1888888 is new. Reserved final comparison seeds2490888/2590888/2690888 were absent from inspected saved plan/config seed fields at preparation. Historical validation/diagnostic seeds are included in the exclusion set. The actual search and guard seeds are hash-derived by the coordinator and exclude both configured validation lists. Do not use the reserved seeds for tuning or diagnostics before the final comparison.

## Source integration and freeze

Run these later, after the parent has finished reviewing/integrating the maintained-stage proposal and power mapping. They are **not executed by this report**:

```sh
git apply --check reports/flight-maintained-integration-proposal/canonical-maintained.patch
git apply reports/flight-maintained-integration-proposal/canonical-maintained.patch
cp reports/flight-power-parameterization-staging/staged/web/flybody-wings.js web/flybody-wings.js
node scripts/prepare-training-manifest.mjs
node reports/flight-maintained-training-plan/prepare-run.mjs reports/flight-maintained-training-plan/source-contract.json reports/flight-maintained-amplitude-local-v1
```

If those files are already integrated, skip the apply/copy commands after checking their exact hashes. The manifest command updates canonical asset pins; it does not set the canonical task to this experiment. The finalizer verifies the contract’s explicitly reviewed assets, their import closure, full-body XML/metadata, prepared graph binary hashes, the initial27-vector, and the exact staged source identities. It recomputes the local run's asset fingerprint/config hash and **reads current native backend module/package-lock hashes at that moment**. It does not import/install Dawn or MuJoCo. It refuses an existing output directory. The source contract intentionally rejects unrelated unreviewed runtime changes.

The optional spacious curriculum is being prepared separately. It needs a **separate reviewed source contract and template**: exact `config.maintainedScene` and `metadata.maintained_scene` profile, updated metadata/bodyVariantHash, new `/flight-scene-profile.js`, and all reviewed world/collision/environment/scorer/preview hashes. Its planned fixed dimensions are radius50cm, ceiling50cm, floor cap radius6.5cm. Keep full native body, initial3.5cm height and flight criteria. The same finalizer can consume that explicit contract without a permissive hash bypass. Do not run the original-bowl contract and silently change the arena mid-run. The original bowl tests obstacle interactions after roughly2s; that differs from a spacious maintained-flight curriculum.

The fresh output archives config bytes, model override bytes, small runtime/tooling sources and their hashes, plus backend plan and run declaration. Large prepared arrays/native binaries are streamed and hash-verified, then pinned in place to avoid another200+MB duplicate. Retain those exact files until the run is archived elsewhere; a hash is not a backup. The backend itself additionally verifies its pinned native addon/package bytes and rejects software fallback at execution.

## Start and monitor the actual standard path

Port7862 had no listener when this recipe was prepared; check again immediately before starting. Run only one full-fly contributor/evaluator at once.

```sh
lsof -nP -iTCP:7862 -sTCP:LISTEN
.venv/bin/python -B scripts/serve-training-dev.py --port 7862 --database reports/flight-maintained-amplitude-local-v1/coordinator.sqlite3 --bundle reports/flight-maintained-amplitude-local-v1/bundle.json
```

In a separate terminal, obtain initial status and the generation0 checkpoint before contributing:

```sh
curl --fail --silent --show-error http://127.0.0.1:7862/api/training/status -o reports/flight-maintained-amplitude-local-v1/status-initial.json
curl --fail --silent --show-error http://127.0.0.1:7862/api/training/checkpoint -o reports/flight-maintained-amplitude-local-v1/checkpoint-initial.json
node scripts/contribute-training-native.mjs http://127.0.0.1:7862/ reports/flight-maintained-amplitude-local-v1/contributions 28 reports/flight-maintained-amplitude-local-v1/backend-plan.json
```

The standard coordinator owns all jobs and saves every accepted result and incumbent transition in the fresh SQLite database. The standard native contributor executes the same environment with Dawn/Metal, one fly at a time; it is not a local optimizer or a browser preview. Ordinary browser workers cannot satisfy this intentionally pinned native provenance guard. Its per-evaluation files retain actual backend, parameters, return, sparse frames and metrics; `parameter-updates.jsonl` records all27 coordinate changes when a generation finishes. A line saying `accepted` acknowledges a result, not necessarily a new checkpoint.

Use the existing read-only reporter during search/comparison and after each completed generation:

```sh
.venv/bin/python scripts/report-flight-training.py reports/flight-maintained-amplitude-local-v1/coordinator.sqlite3 --last-generations 2 --output reports/flight-maintained-amplitude-local-v1/training-progress.json
```

Inspect `completedByPurpose`, `acceptance.decision`, paired differences/mean, and `parameterUpdate.deltaByParameter`, changed count and L2. Compare airtime/success/failure metrics as well as return. A rejected proposal must leave all27 saved coordinates exactly unchanged. No parameter movement is expected while six comparison jobs are incomplete. A finite valid physical failure is data; cancellation/infrastructure errors must not be folded into the guard's means. For a clean pause, Ctrl-C the contributor or create its output `STOP` file. Do not delete `.pending.json` results to force restart; reconcile them with the coordinator first. Keep source/config bytes fixed while the run is active.

## Predeclared final comparison

After generation2, with contribution stopped:

```sh
.venv/bin/python scripts/report-flight-training.py reports/flight-maintained-amplitude-local-v1/coordinator.sqlite3 --output reports/flight-maintained-amplitude-local-v1/training-final.json
curl --fail --silent --show-error http://127.0.0.1:7862/api/training/checkpoint -o reports/flight-maintained-amplitude-local-v1/checkpoint-final.json
node reports/flight-maintained-training-plan/prepare-heldout.mjs reports/flight-maintained-amplitude-local-v1
node scripts/evaluate-training-native.mjs http://127.0.0.1:7862/ reports/flight-maintained-amplitude-local-v1/heldout-plan.json reports/flight-maintained-amplitude-local-v1/heldout
```

The plan builder checks both guard decisions are final, no third-generation results exist, and reserved seeds did not enter training. It compares the exact initial vector against the final incumbent, paired on three fresh seeds; no best-trial cherry-picking. The evaluator uses read-only coordinator requests and uploads no results. Report paired return, qualified airtime, full5s success and failure reasons; three seeds give limited evidence and do not establish the full food→approach→landing→feeding→takeoff task. If the guard retained the same center, report that explicitly rather than inventing learned improvement.

Stop the coordinator before copying its database for a final archive, or use SQLite's backup API while it is running; copying a WAL database file alone is not a consistent backup. Preserve the exact config, bundle, initial/final checkpoints, backend plan, declaration/source archive, accepted jobs and read-only evaluation artifacts together.
