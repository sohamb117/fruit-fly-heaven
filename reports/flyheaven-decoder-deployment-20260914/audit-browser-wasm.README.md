# Browser WASM run audit

Run from the repository root with existing ADC credentials:

```sh
.venv/bin/python -B reports/flyheaven-decoder-deployment-20260914/audit-browser-wasm.py --project flyheaven --run-id banc888-motor-decoder-wasm-20260914 --config reports/motor-decoder-v1/browser-wasm-001/config.json --output reports/flyheaven-decoder-deployment-20260914/browser-wasm-audit-001.json
```

Use a new output filename for later snapshots; the script refuses to overwrite one. Omit `--output` to print sanitized JSON. It uses a Firestore read-only transaction and never leases, submits, modifies records, starts a worker, or simulates a fly.

The local config bytes must match the stored config hash. The audit reuses the current storage-independent coordinator rules to check assignment/noise reconstruction, acceptance decisions, completion/provenance, and retained centers. It also checks the stored payload hash and explicitly verifies the worker-reported `provenance.parameters` against the assignment and its hash. Applied JSON numbers are normalized to Float64 before hashing, because browser JSON writes `1.0` as `1`.

Output separates candidate coefficient differences from selected checkpoint differences, with counts for the 24 power and 648 steering coordinates. It includes only scalar metrics, execution timing, checks, and generation/pair/sign labels. Lease and contributor identifiers, full vectors, tokens, and complete result payloads are never emitted. Stored result payloads necessarily enter memory to validate their hashes, but are not saved or logged. Error output includes only the exception class.

`allChecksPassed` with zero completed results validates empty history only. Accepted results and retained checkpoints remain **unverified**: matching stored provenance is not independent proof of remote execution or successful flight. The client does not currently upload total qualified airtime; the audit does not infer it from best/current bout lengths.

Offline plumbing check (no SDK client, network, or simulation):

```sh
.venv/bin/python -B reports/flyheaven-decoder-deployment-20260914/audit-browser-wasm.py --self-test
```

Five fixture cases pass: valid candidate with unchanged checkpoint, and rejection of applied-vector, WASM-pin, result-hash, and initial-center corruption. This is not a live Firestore result.
