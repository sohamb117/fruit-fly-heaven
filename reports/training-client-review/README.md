# Training client review

Read-only review of the initial client integration. The [reproduction](findings.json) uses the actual client with explicitly synthetic coordinator/worker responses; it performs no neural or native compute. Findings were sent to the parent for repair before final acceptance.

**Repair status:** the parent repaired these findings. Nine regression tests in `web/test/training-shared-state.test.mjs` now pass, alongside all nine existing client tests. They cover local/shared checkpoint isolation, delayed status/lease/connection handling, exported independent validation, stale curriculum evidence, and a new run's heartbeat surviving an old request's cleanup. A retired worker error also cannot reject a newer worker's request. These remain orchestration tests, separate from the actual BANC/native smoke.

1. Shared work rewrites displayed/local stage and generation while retaining a previous local parameter vector, optimizer round and validation. Switching back to local can reuse a posture round under a feeding stage; persistence combines local parameters with shared stage/generation. Reproduced.
2. A delayed coordinator status response mutates generation, checkpoint and message after Stop because it lacks a run-token check after the await. Reproduced. Stop also needs to avoid overwriting a newer start after an awaited lease release.
3. Independent evaluation of a shared checkpoint sets a locally-tested UI label but export still selects the stripped, unverified shared copy and omits the local test evidence. Reproduced.
4. Curriculum badges survive changes to the parameter vector; import clears the internal completed-stage set without rebuilding the displayed curriculum. Later-stage manual starts can reach the final stage's all-stages-passed message without a complete validated-stage set. Stage changes should also invalidate test evidence for a different stage.

Suggested boundary: retain distinct local/shared progress; bind validation evidence to exact parameters and stage; guard each asynchronous state commit by the originating run token. Contribution counters otherwise distinguish completed episodes from server-accepted submissions, and result checks compare the parameter vector, seed, stage, backend, model/config hashes and neural/body clock before counting.
