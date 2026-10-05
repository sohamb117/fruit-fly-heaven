# Sequential sensorimotor training

This experiment uses **simulation-based engineering calibration**. It does not fit measured electrophysiology or claim to identify biological receptor parameters. The existing BANC graph, neuron identities, signs, muscle assignments and native mechanics remain fixed.

The browser executes one assigned BANC/MuJoCo WASM trial at a time. A durable coordinator assigns every candidate, demonstration and validation trial. The existing training page displays the current phase, fly, brain and current trial values. There is no disconnected training fallback.

## Order and trainable values

| Phase | Active coordinates | Method / task |
| --- | ---: | --- |
| Leg feedback | 5 | Bounded search during maintained flight |
| Antenna feedback | 6 | Bounded search during maintained flight |
| Haltere feedback | 7 | Bounded search during maintained flight |
| Wing strain feedback | 4 | Bounded search during maintained flight |
| Visual feedback | 2 | Bounded search during maintained flight |
| Fit movement | 672 | Demonstration collection and bounded ridge command imitation during recovery |
| Recover | 672 | Autonomous search after a seeded release-velocity disturbance |
| Take off | 672 | Autonomous search from a grounded start |
| Land | 672 | Autonomous takeoff, flight and landing |

The vector has 696 entries. The first 24 are log multipliers around the structural sensor profiles; zero preserves the original configuration exactly. They change sensitivity, filtering or declared mechanical priors, never receptor identity, polarity or receptive direction. The last 672 are the existing absolute motor-decoder coefficients: 24 power weights and 648 steering/history/phase coefficients. A phase changes only its declared coordinates. Sensory state is rebuilt per candidate; the expensive graph, body factory and current calibration are reused.

This is behavioral optimization of sensory priors, not stimulus-response identification. A sensor that is barely excited in maintained flight may receive little useful training signal. The sequence does not manufacture a fit for such a stage. It stops for review if the declared budget cannot pass the tests. Grounded stimulus protocols and measured recordings can replace these engineering objectives later.

## Acceptance and retention

The initial configuration specifies two positive/negative candidate pairs. The coordinator's separately recorded scheduling policy can increase the search batch without changing existing candidate identities, seeds, neural dynamics or mechanics. The flytrain scheduling update selects eight pairs, or **16 search trials**, with a common seed within each pair. The best candidate is compared with the incumbent on three new matched seeds. A mean improvement of at least 0.01 leads to another comparison on three fresh matched seeds. The candidate must improve by at least 0.01 there too, and retain two successes out of three fresh episodes on every previously passed physical task.

A candidate that passes those checks becomes the new incumbent even when it has not yet mastered the current task. The next round searches around those improved values. Completing the phase additionally requires two successes out of three on the current task's fresh candidate episodes. This separates incremental learning from task completion. Rejected candidates leave the incumbent intact; exhausting six rounds stops the phase for review while preserving any improvements already accepted. The page stops requesting work at `needs-review`.

These thresholds are explicit engineering choices, not statistical significance tests. With only three episodes per comparison, passing is provisional evidence. Retention means retaining task success, not matching the previous mean score. Anonymous browser submissions are checked for assignment, source, backend, parameters and complete clocks, but remain unverified execution claims. Checkpoints retain `status: unverified` and `biologicalSuccessValidated: false`.

## Scheduling and overdue trials

Batch size is an operational policy stored with the run and exposed in its status/checkpoint. The immutable configuration still records the original defaults; reproducing the training search also requires the recorded scheduling policy. Increasing the active search batch appends deterministic pairs and retains all existing results and assignments. If the run has already entered comparison or validation, the larger batch begins at the next search round. Comparison and validation seed counts and acceptance thresholds do not change.

The 180-second renewable lease detects disconnected contributors. Separately, each assignment attempt has an absolute wall-clock deadline that heartbeats cannot extend. The flytrain policy allows 1,200 wall seconds per five assigned simulation seconds, with a 600-second minimum. That means 20 minutes for maintained flight/recovery, 12 minutes for takeoff, 32 minutes for landing, and 60 minutes for the three-episode decoder fit. These limits include rest and manual pauses. An already leased job at migration receives one explicitly recorded grace window because the old schema did not store its original start time.

When the deadline passes, another browser can lease the same candidate and seed with a new token and deadline. The previous owner has a 60-second cooldown for that specific trial and can work on other candidates. Late results from the obsolete assignment are rejected; accepted-result retries remain idempotent. Updated clients discard expired attempts and request work automatically while respecting Pause and Stop. Pages open before the update need one reload to receive that client behavior; the server deadline applies independently.

The scheduler still has a batch barrier: comparison begins after every search result arrives. A larger pool and bounded reassignment reduce idle time; they do not make the optimizer asynchronous or establish improved behavior.

Recovery applies one deterministic seeded initial velocity at the end of the existing 0.5-second root-held warmup, then integrates freely for five scored seconds. There are no continuing root forces or corrections. Angular disturbance is 24–36 rad/s, linear disturbance 2–6 cm/s. Recovery requires settled powered flight within two seconds, a final settled second and no environment contacts. Takeoff and landing retain their grounded starts and three/eight-second horizons. Every task uses the same pinned native body; previously passed tasks are retested after later changes.

## Decoder fitting

The coordinator assigns two training trajectories and one held-out trajectory. A bounded state-feedback teacher uses an independently measured native wing-control Jacobian to produce targets through the existing wing boundary. Privileged pose/velocity inputs belong only to this demonstration teacher. They never enter the deployed decoder.

The collector pairs the actual individual-MN history features with commands at every applied 0.2 ms wing sample, retaining compact sufficient statistics rather than full trajectories. The existing 1 ms MN excitation/history update remains unchanged. Bounded ridge regression uses only training trajectories, regularizes toward the incumbent and preserves unexcited coefficients. Held-out trajectories never enter the fit.

All three teacher trajectories must succeed for their full horizon; the eight fitted output blocks must converge and improve held-out command error. A candidate still receives **no autonomous flight credit** until it passes separate coordinator-assigned comparisons and fresh gates. A failed teacher produces a failed fit attempt, never fitted weights or a fake flight score. Whole-fit cancellation releases the assignment. Progress accumulates real neural clocks across the three episodes so clock resets cannot trigger a false stall.

The teacher calibration source is `reports/flight-decoder-feasibility-20260914/calibration.json`. The builder verifies its source configuration and native XML, metadata, binary and wing code. The only permitted physics-wrapper difference is the reviewed read-only local-frame force observer. Any further mechanics change requires new calibration/review. Teacher gains use the previously exercised 12 rad/s natural frequency and 0.9 damping ratio; new sensory conditions still require new successful demonstrations.

## Build and run

From this checkout, with the existing prepared BANC data and native runtimes available:

```sh
node scripts/prepare-sequential-training.mjs --output=reports/my-sequence
python3 scripts/serve-training-dev.py --port 7886 \
  --bundle reports/my-sequence/sequence.bundle.json \
  --database reports/my-sequence/coordinator.sqlite3
```

Open `http://127.0.0.1:7886/train.html` in Safari and press Start. The builder refuses to overwrite its bundle. It snapshots small source/model assets and pins the existing large WASM/graph files without duplicating them. It also creates the matching brain-display sample. A different build requires a new output directory and database. Restarting with the same bundle/database resumes the exact accepted vector and outstanding jobs.

The local coordinator uses SQLite WAL. `sequence_state` contains the accepted vector, phase and history; `sequence_jobs` contains assignments and accepted evidence; `sequence_checkpoints` retains every accepted update. **Download checkpoint** fetches the coordinator's current accepted vector. `/api/training/status?compact=1` gives progress, including the last accepted update's changed-coordinate count and magnitude; `/api/training/checkpoint` gives the full current checkpoint. Candidate values on screen may differ from the saved incumbent while an evaluation is running.

## Google-managed hosting

`FirestoreSequentialCoordinator` implements the same sequence through atomic Firestore transactions. Firestore owns the durable state, leases, results, contributor counters and checkpoints. Each request uses an in-memory SQLite instance only to execute the shared transition logic, then discards it; Cloud Run instance files never own training progress. Status reads use the current batch and a compact recent-trial summary rather than scanning completed history.

The storage path is `training_sequences/{run_id}`, with `jobs`, `checkpoints` and `contributors` subcollections. A completed generation remains addressable for idempotent result retries. Stored JSON preserves matrices without Firestore nested-array issues, and every document/transaction is size-checked. Old `training_runs` namespaces are unaffected.

Validate a prepared configuration without creating a cloud client:

```sh
.venv/bin/python scripts/firestore_sequential.py \
  --config reports/my-sequence/config.json \
  --run-id banc888-sensorimotor-20260915 --project flyheaven --dry-run
```

When deliberately provisioning this new experiment, replace `--dry-run` with `--initialize`. Initialization is explicit and idempotent; serving never silently creates or replaces missing state. Cloud Run's entry point selects the sequential adapter when the packaged configuration declares `trainingSequence`. Its packager includes both new Python modules, and deployment preflight validates the current phase of a resumed sequence rather than assuming it is still on its initial task. Use the existing [managed deployment workflow](../deploy/cloudrun/README.md) with a freshly packaged matching browser bundle and the new run ID.

The sequence is now deployed at [flytrain.morisoba.moe/train.html](https://flytrain.morisoba.moe/train.html), using `training_sequences/banc888-sensorimotor-wasm-20260915` in project `flyheaven`. The [release record](../reports/flyheaven-sequence-eyes-20260915/README.md) identifies the exact bundle and Cloud Run revision. The previous 672-parameter run remains in its original namespace; no scores were merged across the two configurations.

## Validation scope

Protocol tests exercise masked exploration, compare/gate promotion, cross-task retention, failed-fit handling, exact retry semantics, concurrent results, expiry, releases and restart identity. Runtime tests cover candidate-specific sensory reconstruction, recovery initialization, causal demonstration samples, bounded fitting, browser lifecycle and progression across resetting episode clocks. Rendering tests cover phase labels and current-value identity.

These tests establish implementation behavior. A completed browser trial verifies the full local execution/upload path; it does not establish that all phases will train successfully or that the resulting fly will achieve the complete food-localization-to-feeding sequence.
