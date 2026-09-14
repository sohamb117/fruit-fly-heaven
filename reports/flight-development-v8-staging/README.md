# Guarded checkpoint selection — staged v8

This directory stages a coordinator-only retention guard. It does not change the 27 physical parameters, body, neural model, sensory mapping, reward, or episode horizon. No fly simulation or active training database was touched by this implementation task.

The [Rules/SQLite patch and durable tests](coordinator-and-tests.patch) passed a clean apply check. [Core validation](coordinator-validation.json) records 17 focused guard tests and 26 existing coordinator/optimizer tests against staged source. Original source snapshots remain under `original/`; root-owned JS/contributor/reporter changes are separately staged under `root/`.

## Exact configuration

```json
{
  "schemaVersion": 2,
  "optimizer": {
    "acceptance": {
      "profile": 1,
      "proposal": "best-search-job",
      "seedCount": 3,
      "nativeExecution": {
        "backend": "dawn-metal",
        "moduleSha256": "<exact lowercase SHA256>",
        "packageLockSha256": "<exact lowercase SHA256>"
      }
    }
  }
}
```

The other optimizer settings and model configuration remain required. Acceptance and nativeExecution objects reject unknown keys. Schema 1 rejects an acceptance property; ordinary schema-1 jobs/checkpoint behavior is preserved. Schema 2 prevents old coordinators from silently applying their immediate-advance behavior. Storage metadata uses schema 2 for guarded runs; checkpoint/API envelopes retain version 1 and `unverified` status.

The backend pins match the existing native contributor plan. Do not pin the backend-plan file hash inside config: that file contains configHash and would introduce a circular hash dependency. The server requires `backend=webgpu`, `neuralEngine=dawn-metal`, matching native loader/package-lock hashes, native backend `dawn-metal`, and a non-fallback adapter declaration. This remains reported provenance, not remote execution attestation or a guarantee of equivalence across different machines.

## Selection and persistence

After eight search jobs, nominate the exact highest-return job vector; break ties by job ID. This is explicitly the `best-search-job` proposal policy, not the existing ES gradient proposal. Never use comparison jobs in `_next_center`.

The generation's `center` remains the incumbent while its status changes from `evaluating` to `checking`. Six ordinary leased jobs compare the nominee and incumbent on three fresh common seeds. IDs are `gN-a0/1/2-pos/neg`; positive is the candidate, negative the incumbent. Comparison rows contain zero noise vectors to preserve the row format; those vectors are not Gaussian exploration.

Only after all six valid returns does the same transaction record the strict mean paired difference, finish the generation, create its successor, and select the nominee if that mean is positive. Ties and lower means retain the incumbent. The next generation still advances, allowing a fresh search. `/checkpoint` never exposes an unaccepted candidate.

If the nominated numerical parameter vector equals the incumbent, skip the six redundant comparisons and record `identical_parameters`. Signed zeros count as the same physical value, and the incumbent representation is preserved. Thus two guarded generations require at most 28 evaluations, with 16 or 22 possible when no-op generations occur. The root contributor's staged generation limit prevents a 28-job budget from spilling into another generation after no-ops.

Each generation stores nullable `acceptance` JSON with these keys:

```text
profile, proposal, proposalJobId
proposedParameters, proposedParametersHash
comparisonSeeds, comparisonJobIds
decision: pending | accepted | rejected | identical_parameters
candidateMean, incumbentMean, meanPairedDifference, pairedDifferences
```

Means/differences remain null until a completed comparison; no-op means remain null. Rejection evidence is retained alongside all original job returns, metrics, failure reasons, and provenance. Selection replays regenerate exact assignments and recompute the decision rather than trusting stored metadata alone.

## Seed and outcome boundaries

Guarded search seeds occupy the lower half of uint32, with deterministic probing to avoid reserved validation/test seeds and duplicate pair seeds within a generation. Comparison seeds occupy the upper half, using disjoint 512-slot affine-permutation blocks by generation. Both exclude `validation.seeds` and `validation.testSeeds` (at most 128 entries each). Comparison seeds never overlap guarded search seeds or earlier comparison seeds. The finite seed domain explicitly ends after `2^22` guarded generations. None of this changes schema-1 random streams.

All guarded results require `cancelled=false`, boolean success/terminated, positive finite elapsed time and integer steps matching the body block duration. Accepted outcomes are:

- Full assigned horizon with `time_limit`, success false, terminated false.
- Full assigned horizon with `stage_success`, success true, terminated true.
- A legitimate physical failure (`outside_habitat`, `excessive_rotation`, `overturned`) with success false and terminated true, within the assigned horizon.

Missing/unknown outcomes, inconsistent flags, cancellations, simulation errors, invalid observations, and unexpected external forces do not enter selection means. They leave their assignment retryable or pending rather than fabricating a low training return. The root's staged contributor writes the additional flags only for schema 2.

## Integration and tests

SQLite uses one nullable generation column in a new guarded database. Existing databases with another config/schema are rejected before any guard-specific schema mutation. Firestore stages the same transitions atomically and makes batch/history validation phase-aware; its test evidence is provided separately with that implementation. A guarded namespace must use the matching new source and config together.

Run the durable core suite from staging:

```sh
.venv/bin/python -B reports/flight-development-v8-staging/staged/tests/test_training_acceptance.py
```

Its cases cover both transaction barriers, concurrent/duplicate submissions, restart and lease recovery, nomination and mean decisions, no-op handling, strict provenance/outcomes, seed exclusion, metadata tampering, no guard mutation of an incompatible vanilla database, and a fixed pre-guard generation/update signature. The test file can be copied directly to `tests/` after integration; it has no historical-report import dependency.

The root-owned schema, assignment, local-optimizer rejection, contributor generation limit, and reporting changes were independently read and had no remaining review blocker at core handoff. No new validation is claimed for the body or neural behavior. A three-seed training acceptance decision can still overfit or admit an unseen-seed regression; the independent held-out evaluation remains necessary and checkpoints remain unverified.
