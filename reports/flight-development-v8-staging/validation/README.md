The staged helper exports exact g0/g2 incumbents after two completed guarded generations, then reports their six paired held-out evaluations. It does not run simulations, instantiate the coordinator, issue leases, or update the database. Existing v6/v7 helper and result files are unchanged.

Fill and freeze [protocol.template.json](protocol.template.json) after the v8 declaration, bundle, backend plan and final coordinator source exist. The declaration requires `maximumInitialJobs:28`, `validation.afterGeneration:2`, the exact configured initial vector, and the same three held-out seeds/stage/horizon. The config uses schema 2 and reserves all three seeds in `validation.testSeeds`. Additional backend scheduling fields such as `stopAfterGeneration:2` are permitted.

After the contributor stops at generation 2:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reports/flight-development-v8-staging/validation/validate-guarded-flight.py prepare reports/flight-development-v8/validation-protocol.json reports/flight-development-v8/held-out-plan
```

The helper reads one SQLite transaction with `mode=ro` and `query_only`. It calls only source-pinned `CoordinatorRules` methods to reconstruct assignments and audit nominations, matched guard seeds, score comparisons and selected successor centers. It checks accepted result hashes/provenance in memory and omits contributor identities, lease tokens and credential-bearing payloads from the exported snapshot. Pending or leased work, unexpected generations, undeclared candidate changes, missing comparison results and held-out overlap reject.

Two ordinary generations contain 16 search and 12 guard evaluations. An explicitly audited identical-vector nomination skips its six guard jobs, so 16 or 22 accepted evaluations can also be a complete two-generation run. No other shortened count is accepted. The exporter keeps the stored SQLite parameter JSON verbatim and refuses to overwrite output.

After all six evaluator records are saved:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reports/flight-development-v8-staging/validation/validate-guarded-flight.py report reports/flight-development-v8/held-out-plan/plan.json reports/flight-development-v8/held-out-evaluation reports/flight-development-v8/held-out-comparison
```

The report re-audits the frozen guard history. It preserves the existing finite result, source/config/backend, assignment, body-step, full-horizon-or-physical-failure, criteria-3 and native digest guards. [evaluation-checks.py](evaluation-checks.py) is extracted from the tested v6/v7 helper; only the returned head-label index changes from modulo three to modulo two. [evaluation-checks-origin.json](evaluation-checks-origin.json) records that exact adaptation and source hashes.

Eight pure history tests pass:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reports/flight-development-v8-staging/validation/test_guarded_validation.py
```

They cover accepted/rejected successor reconstruction, legitimate no-op skips, corrupted decisions/nominations, missing guard results, seed overlap, started final generations, forged successor centers and SQLite JSON-field decoding. They use no database writes, neural/body simulations or GPU imports. Guard decisions are training evidence; even a mean-positive guard does not guarantee improvement on every guard seed or independent held-out behavior.
