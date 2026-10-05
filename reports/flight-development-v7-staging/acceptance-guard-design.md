# Opt-in incumbent acceptance guard — design only

No implementation, database changes, or simulations accompany this proposal. It is conditional on the v7 results. [Source pins](acceptance-guard-source-pins.json) identify the inspected coordinator/client/test code.

## Recommendation

After the eight existing search evaluations, freeze one proposed center and compare it with the incumbent on **three fresh matched training seeds**. Keep serving the incumbent checkpoint until all six comparison jobs finish. Accept only if the mean paired return difference is strictly positive; ties and lower means retain the incumbent. Then advance the generation number and start the next search around the selected center.

This rejects regressions *measured by that comparison*. Three seeds do not guarantee improvement on unseen seeds, and repeated acceptance decisions can still admit regressions. Comparison seeds are training data, not independent held-out validation. Checkpoints remain `status: unverified`, `heldOutValidated: false`, with no curriculum promotion or claim of validated flight.

| Proposal(s) checked | Two seeds | Three seeds |
| --- | ---: | ---: |
| One proposal plus incumbent | 8 + 4 = 12 jobs/generation | 8 + 6 = 14 |
| ES proposal, best search job, incumbent | 8 + 6 = 14 | 8 + 9 = 17 |

For four generations, the recommended one-proposal/three-seed guard is 56 evaluations versus 32 today: 75% more evaluations, not necessarily exactly 75% more elapsed time because episode lifetimes differ. Two seeds cost 48 evaluations over four generations and provide a noisier retention test.

## Proposal choice

Predeclare one mode in the immutable configuration:

- `es`: use the existing `_next_center` on **search jobs only**, preserving the current scaled, clipped ES proposal. This is the smallest extension.
- `best-search-job`: use the exact parameter vector of the highest-return completed search job; break ties by a fixed job-ID order. Do not copy its original return into the acceptance comparison. This can preserve useful sampled trials even when the gradient proposal is poor. Original search jobs used different seeds, so this choice is a noisy nomination; the fresh comparison is essential.

Both modes cost the same number of comparison jobs. The second changes the center-proposal policy from an ES gradient step to selecting a sampled candidate; record that policy explicitly rather than describing the whole update as unchanged ES. Testing both nominations against the incumbent adds another two or three jobs and additional selection bias. Defer that larger variant. Do not switch proposal mode based on the current generation's comparison outcomes.

If a proposal is byte-identical to the incumbent, a predeclared no-op shortcut may advance without comparison jobs and record `decision: identical_parameters`; there is no center change to accept. Otherwise never reuse the incumbent's old search/acceptance return on new seeds.

## Minimum durable state

Retain the existing generation `center` as the immutable incumbent. Add one nullable `acceptance` JSON field/column to a generation, containing:

```text
profile, proposalMode, proposalJobId (if applicable)
proposedParameters, proposedParametersHash
comparisonSeeds, comparisonJobIds, executionClassPin
decision: pending | accepted | rejected | identical_parameters
candidateMean, incumbentMean, meanPairedDifference
pairedDifferences, decisionReason
```

Only fill statistics when all comparison returns are valid. The existing per-job result rows already retain the actual returns, failure reasons, metrics, contributor provenance, and timestamps; reference them rather than duplicating full payloads in generation metadata.

Reuse generation status as `evaluating` during search, `checking` during comparison, then the existing `unverified` terminal state. Keep `finished` unset until the decision. A rejection still completes a generation and creates generation `g+1` with the unchanged incumbent vector, so generation accounting and deterministic exploration continue normally.

Jobs need no new mandatory storage columns. Reserve exact IDs `g{g}-a{seedIndex}-pos/neg`, with `sign=+1` meaning proposal and `-1` incumbent. Store a zero noise vector solely to satisfy the existing row shape; these rows are comparisons, **not antithetic Gaussian samples**. A shared phase-aware validator must classify the exact reserved job sets and expose an optional assignment `purpose` for logs. It must never pass comparison rows to `_next_center`. An explicit nullable job-kind column is cleaner but is not required for this bounded two-role design.

Generate comparison seeds from a separate deterministic acceptance domain and record them before any comparison runs. Exclude current search seeds, previously used training comparison seeds, and reserved `validation.seeds`/`validation.testSeeds`; resolve collisions deterministically. Keep proposal selection independent of comparison outcomes. Current fixed test seeds must not become acceptance data.

## State transitions and backend changes

1. **Last search result:** within the existing transaction, store that result, freeze the proposal and seed list, mark the generation `checking`, and create four/six comparison jobs. Do not create `g+1` or expose the proposed center through `/checkpoint`.
2. **Intermediate comparison result:** use existing lease, expiration, heartbeat, release, idempotency, and provenance handling. The checkpoint still contains the incumbent.
3. **Last comparison result:** compute `mean(candidate_i - incumbent_i)` from matched pairs, record the means and rejection/acceptance reason, finish the old generation, create the next search jobs, and atomically advance the current-generation pointer.

SQLite can add the nullable generation column in the **new guarded database only**, with explicit column lists replacing positional generation inserts. No migration of the running/default database is required. Keep legacy initialization and transition paths unchanged when the guard is absent.

Firestore requires these concrete changes:

- `_batch()` currently insists on exactly `2 * populationPairs` jobs. Validate phase-specific exact IDs, roles, seeds, and parameter hashes; do not merely permit a larger count.
- `result()` currently advances after every current row completes. Branch on search versus comparison phase. Read all required documents before writing; deterministic proposal/seed creation must survive transaction retries.
- Keep completed-result duplicate handling before current-phase/current-generation rejection. Retried search results after the phase change and retried comparison results after advancement remain no-ops.
- Preserve result/contributor counters: one accepted submission is one result, regardless of whether its center is retained. Acceptance of a report is distinct from acceptance of a new center.
- Validate the merged generation document with `checked_document`. At four search pairs, the last-search transaction writes 8 or 10 documents; the last-comparison transaction writes 13. Keep these atomic instead of chunking them.

`read_sqlite_history()` currently requires every generation to contain only search jobs and every successor center to equal raw `_next_center`. Update it to replay the recorded proposal and acceptance decision, including unfinished comparison phases; otherwise guarded SQLite history cannot safely import to Firestore. Retain the existing legacy import branch.

## Version boundary and execution comparability

An unknown optional config flag is insufficient protection against an old server: old code ignores it and can advance after eight search results. Use **config schema version 2 only for guarded runs**, retaining vanilla version 1 exactly. Existing Python and JS config readers then reject guarded configurations until explicitly updated. Keep parameter/checkpoint representations unchanged. Use guarded Firestore metadata schema 2 as well. Metadata-only versioning would not protect old SQLite code, whose initializer currently does not enforce the stored metadata schema value.

For the contemplated local run, require its fixed `dawn-metal` execution class and the exact native-tooling provenance already used by the contributor backend plan. Both members of every pair must match that pinned class. Current coordinator provenance permits either `webgpu` or `wasm`; equal seeds alone do not make those executions equivalent. A generic `webgpu` tag is also weaker than the existing exact native plan. Broader heterogeneous public execution needs a separate design and is outside this proposal.

An infrastructure error, cancellation, incompatible backend, or incomplete evaluation is not a low scientific return. Leave its comparison unfinished/retryable, retain the incumbent, and preserve the existing local error artifact. Do not manufacture a mean from missing jobs. Legitimate completed physical failures retain their configured finite returns and failure metrics. Record a rejected candidate's comparison failure as `decisionReason: no_positive_mean_improvement`, with both means and all paired differences; avoid labeling that decision successful flight.

## API/client and focused tests

Both current contributors already evaluate assigned parameters, seed, stage, and full horizon; they do not require jobs to be Gaussian perturbations. No new endpoint or evaluation controller is needed. Extend config validation/version handling, and optionally add machine-readable phase/proposal metadata to status. Keep existing checkpoint fields and `unverified` labels. Comparison work counts toward completed/total job counts. No product UI copy is required.

Reuse the existing SQLite, Firestore emulator, import, and browser-client fixtures to test:

- Byte-compatible vanilla jobs, updates, checkpoint/status contracts, and legacy history import.
- Strict better/tie/worse paired means, noisy individual pairs, both proposal modes, deterministic ties, and exact no-op proposals.
- Incumbent checkpoint throughout checking, restart during checking, and exact rejection/acceptance artifacts.
- Concurrent final search submissions creating comparison jobs once; concurrent final comparison submissions advancing once.
- Duplicate/conflicting results across both phase boundaries, expired leases, missing or malformed job sets, and provenance mismatches.
- Guarded import reproducing selection decisions; tampered proposal/seeds/means/next center rejected.
- Old schema/config rejection, including the SQLite path; comparison rows never entering ES aggregation.

Do not implement or launch until the parent selects a proposal mode, seed count, and guarded namespace after the v7 held-out assessment.
