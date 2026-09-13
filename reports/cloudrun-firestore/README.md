# Firestore coordinator validation

Validated locally on 2026-09-13 using **Cloud Firestore Emulator 1.20.4**, Java 21.0.4, Python 3.11.16, and **google-cloud-firestore 2.30.0**. The SDK and its complete dependency graph are pinned in [requirements-cloudrun.txt](../../requirements-cloudrun.txt). All Firestore test connections use the loopback emulator and the synthetic project `fly-training-emulator`; these checks did not write to production Firestore.

## Result

**11 emulator test cases and all 27 existing SQLite coordinator tests passed.** The emulator cases ran as an initial nine-case suite and two additional contention tests. An earlier contention failure was fixed before these passing runs: the SDK's immediate commit retries could repeatedly collide during simultaneous lease allocation.

The checks cover matching deterministic jobs and checkpoints across SQLite/Firestore; configuration isolation; refusal to initialize over orphaned or incomplete history; simultaneous lease allocation across separate clients; duplicate result submissions; one-time generation advancement; heartbeat, release and expiry; provenance rejection; new-instance persistence; the existing HTTP API and checkpoint download; exact history migration; interrupted-import recovery; and concurrent initial requests from one contributor.

With **15 visitors released simultaneously**, the service returned **eight distinct leases and seven waiting responses**. The slowest response was **8.783 seconds**, below the browser's 15-second API timeout in this local measurement. Waiting clients receive a 10-second retry delay. Four simultaneous requests from the same contributor returned one identical assignment and capability token.

This is an emulator measurement, not a production latency guarantee or a sustained-load benchmark. Application retries stop after at most ten attempts, with randomized delay after transaction aborts and a 12-second retry cutoff. The cutoff is checked after a failed attempt; it does not cancel an in-flight Firestore RPC. Exhausted contention returns a retryable HTTP 503. Only aborted transactions are retried; validation errors are preserved.

## Migration evidence

The migration fixture contained two generations, sixteen jobs, nine completed results, and an active lease. After import, the full status/checkpoint and every job field matched SQLite, including historical parameter arrays, result hashes, contributor identity, and lease identity. The source SQLite file's SHA-256 remained unchanged. Retrying an old completed result did not increase the accepted count. Completing the preserved active lease increased the count to ten, and rerunning the original import preserved that new progress instead of overwriting it.

An injected interruption after document writes left the run unavailable. The serving constructor refused it until the matching import resumed, verified all records, and atomically marked the run ready. A different or already-serving destination is rejected. Metadata, generations, jobs, and contributor counters are separate documents; the complete history is never stored in one Firestore document.

The production cutover snapshot and any deliberate release of migrated in-flight leases are separate operator actions. These emulator results do not claim that the production import has happened.

## Reproduction

Install the pinned Python dependencies into an isolated environment, then start the official emulator in a separate terminal:

```sh
uv venv --python 3.11 /tmp/fly-firestore-venv
uv pip install --python /tmp/fly-firestore-venv/bin/python -r requirements-cloudrun.txt
gcloud components install cloud-firestore-emulator --quiet
gcloud emulators firestore start --host-port=127.0.0.1:8787 --project=fly-training-emulator --quiet
```

Run the complete coordinator checks:

```sh
FIRESTORE_EMULATOR_HOST=127.0.0.1:8787 \
  uv run --offline --python /tmp/fly-firestore-venv/bin/python python \
  -W error::ResourceWarning -m unittest discover -s tests -p test_firestore_coordinator.py -v

uv run --offline python -W error::ResourceWarning \
  -m unittest discover -s tests -p test_training_coordinator.py
```

The exact additional stress-test invocation used for the reported 8.783-second measurement was:

```sh
FIRESTORE_EMULATOR_HOST=127.0.0.1:8787 \
  uv run --offline --python /tmp/fly-firestore-venv/bin/python python \
  -W error::ResourceWarning -m unittest discover -s tests \
  -p test_firestore_coordinator.py -k fifteen -k same_contributor -v
```

The test module refuses non-loopback emulator endpoints and skips when the emulator or SDK is unavailable. Its objectives are synthetic algebra fixtures. These results validate persistence, scheduling, migration and API behavior; they do not establish learned fly behavior or successful completion of any biological task.
