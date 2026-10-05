This auditor reads this experiment's SQLite database using `mode=ro` and `query_only` in one snapshot. It imports no simulator or native backend and makes no network requests. It reads archived source pins, contributions, checkpoint files and an explicitly selected held-out run manifest. It does not modify any of them.

During the run:

```sh
python3 -B reports/flight-maintained-amplitude-local-v2/outcome-audit.py
```

After both guards and the six reserved evaluations finish, supply the actual evaluator manifest and a fresh output name:

```sh
python3 -B reports/flight-maintained-amplitude-local-v2/outcome-audit.py \
  --heldout-run reports/flight-maintained-amplitude-local-v2/heldout/run-UUID.json \
  --output reports/flight-maintained-amplitude-local-v2/outcome-final-audit.json \
  --require-complete
```

The script never replaces an existing output. Exit 2 means final evidence is still missing; a failed assertion means inconsistent evidence. It requires `checkpoint-final.json`, the contributor's checkpoint and both `parameter-updates.jsonl` lines, including zero changes after rejection. It verifies all 27 stored deltas independently of candidate movement. Each three-seed guard is recalculated using its actual rule: a strictly positive mean paired return difference. A submitted result being accepted by the database does not mean its vector became the incumbent.

Held-out jobs must exactly match the declared three reserved seeds, paired initial and final incumbent vectors, source pins and five-second horizon. The auditor checks terminal success using the final continuous qualified bout and all final flight gates. A longer historical bout alone is insufficient. It reports paired returns, best/current/total qualified airtime and physical failure reasons; it does not automatically label the checkpoint or biology validated.

Sparse preview data cannot independently reproduce all 2 ms scoring updates. This is an audit of saved terminal algebra, recorded native root/reset evidence and source identity, not another physical evaluation. Source-pinned input files remain unchanged. Repeated held-out runs require explicit selection; the auditor never picks the most favorable retry.

Data-only tests:

```sh
python3 -B reports/flight-maintained-amplitude-local-v2/outcome-audit.test.py
```
