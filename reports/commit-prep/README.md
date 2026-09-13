# Full implementation commit preparation

Verified on 2026-09-13. The staged implementation includes the BANC v888 physiology/runtime, native FlyBody integration, original observation console, training UI/coordinator, managed Cloud Run and Firestore deployment, migration and backup tools, source locks, model assets, licenses, tests, and selected experiment summaries. No commit or push was performed during preparation.

Datasets, installed dependencies, root distribution archives, runtime logs, training databases, exported shared checkpoints, and large raw experiment captures remain outside Git. The small BANC WASM artifacts and reference controller assets are intentionally included. The [reproducibility guide](../../docs/reproducibility.md) explains preparation and the archived inputs needed by historical analysis tools.

## Validation

| Check | Result |
|---|---|
| BANC runtime and transplant metric tests | 43 passed |
| Complete web unit/integration suite | 124 passed, no skips |
| Python preparation, coordinator, and legacy brain tests | 41 passed |
| GCP packaging/verification tests | 3 passed |
| Cloud Run serving/packaging tests | 13 passed, including segmented oversized requests and bounded draining |
| Firestore emulator integration tests | 11 passed, including 15 simultaneous clients and complete SQLite migration |
| Reference trajectory check | Passed, 18,000,000 trajectory steps; physical state unchanged |
| Staged whitespace check | Passed; exact upstream license whitespace retained through narrowly scoped attributes |

The web suite uses the documented `--test-force-exit --test-concurrency=1` flags. Without forced process exit, the existing legacy WASM diagnostic test keeps its process alive after its assertions complete and prevents later files from running. The complete flagged run executes all 124 tests. Python validation on this macOS host used the locally compiled, ignored `build/libflybrain.dylib`.

The component suites total **235 passing tests**. The twenty-seven SQLite coordinator cases were rerun after extracting the storage-independent rules; the unrelated runtime/web/preparation suites retain the earlier passing evidence above.

The simplified UI passed seven-origin browser checks, including mandatory contribution routing, plain visible copy, fresh checkpoint downloads, failed connections, and desktop/mobile layouts. A real browser trial on Cloud Run loaded the pinned WebGPU network and native body, rendered 80 progressing preview frames, completed its one-second posture criterion, and uploaded exactly one result. Firestore retained all fourteen previous results and now holds fifteen. A fresh Cloud Run revision retained the same count and checkpoint. These checks establish working infrastructure and one seeded posture outcome, not learned full-sequence behavior. See the [managed deployment record](../flyheaven-deployment/README.md) and [Firestore test record](../cloudrun-firestore/README.md). Earlier VM evidence remains in the [historical rollout record](../gcp-always-shared/README.md).

Local test logs and the staged-file manifest are generated evidence in this directory and remain ignored. The proposed commit message is stored in `.git/PREPARED_COMMIT_MESSAGE` for review.
