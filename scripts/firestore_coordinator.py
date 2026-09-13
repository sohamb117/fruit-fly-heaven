"""Durable, transactional storage for the existing training coordinator API.

Each run has a small metadata document and separate generation, job, and
contributor documents. No instance-local state determines lease ownership.
The SDK retries transaction bodies; all side effects stay inside the transaction.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.api_core.exceptions import Aborted

from training_coordinator import (
    APIError, CoordinatorRules, canonical, finite_number, fingerprint, identifier,
    read_config,
)


COLLECTION = "training_runs"
SCHEMA = 1
MAX_DOCUMENT_JSON_BYTES = 800_000


def checked_document(value):
    # Leave room for Firestore field names/type encoding below its 1 MiB limit.
    if len(canonical(value).encode()) > MAX_DOCUMENT_JSON_BYTES:
        raise APIError(413, "body_too_large", "Stored training record exceeds the document limit")
    return value


class FirestoreCoordinator(CoordinatorRules):
    """Same public methods as TrainingCoordinator, safe across service instances."""

    def __init__(self, config, config_hash=None, *, project=None, database="(default)",
                 run_id, client=None, initialize=False, clock=time.time):
        self._configure(config, config_hash, clock)
        self.run_id = identifier(run_id, "run_id")
        self.client = client or firestore.Client(project=project, database=database)
        self._owns_client = client is None
        self.run = self.client.collection(COLLECTION).document(self.run_id)
        self.generations = self.run.collection("generations")
        self.jobs = self.run.collection("jobs")
        self.contributors = self.run.collection("contributors")
        try:
            if initialize:
                self._transaction(self._initialize)
            self._transaction(lambda tx: self._read_current(tx), read_only=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._owns_client:
            self.client.close()

    def _transaction(self, operation, *, read_only=False):
        # The SDK retries commits immediately; competing read/write batches can
        # repeatedly collide, especially in the emulator. Roll back each failed
        # attempt before jittering, and also retry Aborted errors during reads.
        deadline = time.monotonic()+12
        for attempt in range(10):
            transaction = self.client.transaction(max_attempts=1, read_only=read_only)
            try:
                return firestore.transactional(operation)(transaction)
            except (Aborted, ValueError) as error:
                if not isinstance(error, Aborted) and not isinstance(error.__cause__, Aborted):
                    raise
                if attempt == 9 or time.monotonic() >= deadline:
                    raise APIError(503, "temporarily_busy", "Training service is busy; retry the same request") from error
                ceiling = min(1., .05*2**attempt)
                time.sleep(.025 + secrets.randbelow(1000)/1000*ceiling)

    def _metadata(self, *, ready=True):
        return checked_document({"schemaVersion": SCHEMA, "ready": ready,
            "configHash": self.config_hash, "modelFingerprint": self.model_fingerprint,
            "config": self.config, "currentGeneration": 0, "acceptedResults": 0,
            "contributorsCount": 0})

    def _validate_metadata(self, value, *, require_ready=True):
        if (value.get("schemaVersion") != SCHEMA or value.get("configHash") != self.config_hash
                or value.get("modelFingerprint") != self.model_fingerprint
                or canonical(value.get("config")) != canonical(self.config)):
            raise ValueError("Firestore run uses a different training configuration or schema")
        if require_ready and value.get("ready") is not True:
            raise APIError(503, "run_not_ready", "Training history import is incomplete")
        return value

    def _read_metadata(self, tx, *, require_ready=True):
        snapshot = self.run.get(transaction=tx)
        if not snapshot.exists:
            raise APIError(503, "run_not_ready", "Training run has not been initialized")
        return self._validate_metadata(snapshot.to_dict(), require_ready=require_ready)

    def _empty_children(self, tx):
        return not any(list(collection.limit(1).stream(transaction=tx))
                       for collection in (self.generations, self.jobs, self.contributors))

    def _initialize(self, tx):
        snapshot = self.run.get(transaction=tx)
        if snapshot.exists:
            self._validate_metadata(snapshot.to_dict())
            return
        if not self._empty_children(tx):
            raise ValueError("Refusing to initialize a run containing orphaned history")
        generation, jobs = self._generation_records(0, self.initial)
        tx.create(self.run, self._metadata())
        self._write_generation(tx, generation, jobs)

    def _read_current(self, tx):
        metadata = self._read_metadata(tx)
        snapshot = self.generations.document(str(metadata["currentGeneration"])).get(transaction=tx)
        if not snapshot.exists:
            raise APIError(503, "invalid_history", "Current training generation is missing")
        return metadata, snapshot.to_dict()

    def _batch(self, tx, generation):
        rows = [snapshot.to_dict() for snapshot in self.jobs.where(
            filter=FieldFilter("generation", "==", generation)).stream(transaction=tx)]
        if len(rows) != 2*self.pairs:
            raise APIError(503, "invalid_history", "Current training job batch is incomplete")
        return sorted(rows, key=lambda row: (row["pair_id"], -row["sign"]))

    def _write_generation(self, tx, generation, jobs):
        tx.create(self.generations.document(str(generation["generation"])), checked_document(generation))
        for job in jobs:
            tx.create(self.jobs.document(job["job_id"]), checked_document(job))

    def _read_job(self, tx, payload, contributor):
        job_id, token = self._lease_credentials(payload)
        snapshot = self.jobs.document(job_id).get(transaction=tx)
        return self._verify_assignment(snapshot.to_dict() if snapshot.exists else None, contributor, token)

    def lease(self, payload):
        contributor = self._lease_contributor(payload)

        def assign(tx):
            _, current = self._read_current(tx)
            rows = self._batch(tx, current["generation"])
            now = self.clock()
            # Completed generations cannot contain an active lease. Reading the
            # complete bounded current batch enforces one lease per contributor.
            active = sorted((row for row in rows if row["state"] == "leased"
                             and row["contributor"] == contributor and row["expires"] > now),
                            key=lambda row: row["job_id"])
            if active:
                return {"job": self._job_json(active[0]), "retry": True}
            row = next((row for row in rows if row["state"] == "pending"
                        or (row["state"] == "leased" and row["expires"] <= now)), None)
            if row is None:
                return {"job": None, "waitMs": 10000, "retryAfterSeconds": 10, "status": "waiting_for_results"}
            row = {**row, "state": "leased", "contributor": contributor,
                   "token": secrets.token_urlsafe(32), "expires": now+self.lease_seconds}
            tx.set(self.jobs.document(row["job_id"]), checked_document(row))
            return {"job": self._job_json(row), "retry": False}

        return self._transaction(assign)

    def result(self, payload):
        contributor, score, result_hash = self._result_inputs(payload)

        def submit(tx):
            metadata = self._read_metadata(tx)
            row = self._read_job(tx, payload, contributor)
            self._validate_result_provenance(row, payload)
            if row["state"] == "completed":
                if row["result_hash"] != result_hash:
                    raise APIError(409, "result_conflict", "A different result was already accepted for this job")
                return {"accepted": True, "duplicate": True, "status": "unverified", "generation": row["generation"]}
            now = self.clock()
            if row["state"] != "leased" or row["expires"] <= now:
                raise APIError(410, "lease_expired", "Lease expired or was released; request a new assignment")
            if row["generation"] != metadata["currentGeneration"]:
                raise APIError(503, "invalid_history", "Lease is outside the current training generation")
            generation_ref = self.generations.document(str(row["generation"]))
            generation = generation_ref.get(transaction=tx).to_dict()
            rows = self._batch(tx, row["generation"])
            contributor_ref = self.contributors.document(hashlib.sha256(contributor.encode()).hexdigest())
            contributor_snapshot = contributor_ref.get(transaction=tx)
            contributor_record = contributor_snapshot.to_dict() if contributor_snapshot.exists else {
                "contributorId": contributor, "acceptedResults": 0}
            if not generation or generation["status"] != "evaluating":
                raise APIError(503, "invalid_history", "Training generation cannot accept results")
            accepted = {**row, "state": "completed", "score": score, "completed": now,
                        "result_hash": result_hash, "result": canonical(payload)}
            rows = [accepted if item["job_id"] == row["job_id"] else item for item in rows]
            advanced = all(item["state"] == "completed" for item in rows)
            next_generation = None
            if advanced:
                next_generation = self._generation_records(row["generation"]+1,
                                                           self._next_center(generation, rows), created=now)
            # All reads are finished before any writes. The metadata update
            # serializes accepted-result counters and generation advancement.
            tx.set(self.jobs.document(row["job_id"]), checked_document(accepted))
            tx.set(contributor_ref, {**contributor_record,
                "acceptedResults": contributor_record["acceptedResults"]+1, "lastAcceptedAt": now})
            updates = {"acceptedResults": metadata["acceptedResults"]+1,
                       "contributorsCount": metadata["contributorsCount"]+(not contributor_snapshot.exists)}
            if advanced:
                tx.update(generation_ref, {"status": "unverified", "finished": now})
                self._write_generation(tx, *next_generation)
                updates["currentGeneration"] = row["generation"]+1
            tx.update(self.run, updates)
            return {"accepted": True, "duplicate": False, "status": "unverified",
                    "generation": row["generation"], "nextGenerationCreated": advanced}

        return self._transaction(submit)

    def release(self, payload):
        contributor = self._identity(payload)

        def release(tx):
            self._read_metadata(tx)
            row = self._read_job(tx, payload, contributor)
            if row["state"] == "completed":
                raise APIError(409, "already_completed", "Completed results cannot be released")
            if row["state"] == "pending":
                return {"released": True, "duplicate": True}
            if row["expires"] <= self.clock():
                raise APIError(410, "lease_expired", "Lease already expired")
            tx.update(self.jobs.document(row["job_id"]), {"state": "pending", "expires": None})
            return {"released": True, "duplicate": False}

        return self._transaction(release)

    def heartbeat(self, payload):
        contributor = self._identity(payload)

        def renew(tx):
            self._read_metadata(tx)
            row = self._read_job(tx, payload, contributor)
            if row["state"] != "leased":
                raise APIError(409, "not_leased", "Only an active lease can be renewed")
            if row["expires"] <= self.clock():
                raise APIError(410, "lease_expired", "Expired leases cannot be renewed")
            expires = self.clock()+self.lease_seconds
            tx.update(self.jobs.document(row["job_id"]), {"expires": expires})
            return {"renewed": True, "jobId": row["job_id"], "leaseExpiresAt": expires}

        return self._transaction(renew)

    def checkpoint(self):
        return self._transaction(lambda tx: self._checkpoint_from_row(self._read_current(tx)[1]), read_only=True)

    def status(self):
        def status(tx):
            metadata, current = self._read_current(tx)
            return self._status_from_rows(current, self._batch(tx, current["generation"]),
                                          metadata["acceptedResults"], metadata["contributorsCount"])
        return self._transaction(status, read_only=True)


def read_sqlite_history(source, config, config_hash=None):
    """Read one consistent snapshot without changing the source SQLite database."""
    rules = CoordinatorRules()
    rules._configure(config, config_hash)
    path = Path(source).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite source failed its integrity check")
        meta = dict(connection.execute("SELECT key,value FROM meta"))
        if (meta.get("configHash") != rules.config_hash or meta.get("modelFingerprint") != rules.model_fingerprint
                or meta.get("schemaVersion") != "1" or meta.get("config") != canonical(config)):
            raise ValueError("SQLite source uses a different configuration or schema")
        generations = [dict(row) for row in connection.execute("SELECT * FROM generations ORDER BY generation")]
        jobs = [dict(row) for row in connection.execute("SELECT * FROM jobs ORDER BY generation,pair_id,sign DESC")]
    finally:
        connection.close()
    if not generations or [row["generation"] for row in generations] != list(range(len(generations))):
        raise ValueError("SQLite generations must be a contiguous nonempty history")
    grouped, contributors = defaultdict(list), {}
    for generation in generations:
        generation["center"] = json.loads(generation["center"])
        if len(generation["center"]) != len(rules.bounds):
            raise ValueError("Invalid historical parameter vector")
        for value, (low, high) in zip(generation["center"], rules.bounds):
            finite_number(value, low, high, "historical parameter")
    for job in jobs:
        job["noise"], job["parameters"] = json.loads(job["noise"]), json.loads(job["parameters"])
        if job["generation"] not in range(len(generations)) or job["state"] not in ("pending", "leased", "completed"):
            raise ValueError("Invalid historical job state")
        if len(job["noise"]) != len(rules.bounds) or len(job["parameters"]) != len(rules.bounds):
            raise ValueError("Invalid historical job dimensions")
        for value, (low, high) in zip(job["parameters"], rules.bounds):
            finite_number(value, low, high, "historical parameter")
        for value in job["noise"]:
            finite_number(value, -1e15, 1e15, "historical noise")
        if fingerprint(job["parameters"]) != job["parameters_hash"]:
            raise ValueError("Historical parameter checksum mismatch")
        if job["state"] == "completed":
            payload = json.loads(job["result"])
            contributor, score, result_hash = rules._result_inputs(payload)
            _, token = rules._lease_credentials(payload)
            rules._verify_assignment(job, contributor, token)
            rules._validate_result_provenance(job, payload)
            if score != job["score"] or result_hash != job["result_hash"]:
                raise ValueError("Historical result checksum or score mismatch")
            record = contributors.setdefault(contributor, {"contributorId": contributor, "acceptedResults": 0, "lastAcceptedAt": 0.})
            record["acceptedResults"] += 1
            record["lastAcceptedAt"] = max(record["lastAcceptedAt"], job["completed"])
        elif any(job[key] is not None for key in ("result_hash", "result", "score", "completed")):
            raise ValueError("Uncompleted job contains a result")
        if job["state"] == "leased" and (not job["token"] or not job["contributor"] or job["expires"] is None):
            raise ValueError("Historical lease is incomplete")
        grouped[job["generation"]].append(checked_document(job))
    for generation in generations:
        number = generation["generation"]
        batch = grouped[number]
        expected_ids = {f"g{number}-p{pair}-{sign}" for pair in range(rules.pairs) for sign in ("pos", "neg")}
        if len(batch) != 2*rules.pairs or {job["job_id"] for job in batch} != expected_ids:
            raise ValueError("Historical generation has an incomplete job batch")
        for pair in range(rules.pairs):
            members = [job for job in batch if job["pair_id"] == f"g{number}-p{pair}"]
            if len(members) != 2 or {job["sign"] for job in members} != {-1, 1} or members[0]["noise"] != members[1]["noise"] or members[0]["seed"] != members[1]["seed"]:
                raise ValueError("Historical antithetic pair is inconsistent")
        complete = all(job["state"] == "completed" for job in batch)
        if number < len(generations)-1:
            if not complete or generation["status"] != "unverified":
                raise ValueError("An older generation has unfinished jobs")
            if rules._next_center(generation, batch) != generations[number+1]["center"]:
                raise ValueError("Historical aggregate does not match its next generation")
        elif complete or generation["status"] != "evaluating":
            raise ValueError("Current generation is inconsistent")
    history = {"generations": generations, "jobs": jobs, "contributors": list(contributors.values()),
               "currentGeneration": generations[-1]["generation"],
               "acceptedResults": sum(record["acceptedResults"] for record in contributors.values()),
               "contributorsCount": len(contributors)}
    history["sourceDigest"] = fingerprint({"meta": meta, "generations": generations, "jobs": jobs})
    return history


def import_sqlite(source, config, config_hash=None, *, project=None, database="(default)", run_id, client=None):
    """Resumable import into an empty namespace; never overwrite a serving run."""
    history = read_sqlite_history(source, config, config_hash)
    # Construct only the storage handles: the regular constructor deliberately
    # refuses a missing/incomplete run until the import has finished.
    target = FirestoreCoordinator.__new__(FirestoreCoordinator)
    target._configure(config, config_hash)
    target.run_id = identifier(run_id, "run_id")
    target.client = client or firestore.Client(project=project, database=database)
    target._owns_client = client is None
    target.run = target.client.collection(COLLECTION).document(target.run_id)
    target.generations, target.jobs, target.contributors = (target.run.collection(name) for name in ("generations", "jobs", "contributors"))

    def import_metadata(tx):
        snapshot = target.run.get(transaction=tx)
        if snapshot.exists:
            value = target._validate_metadata(snapshot.to_dict(), require_ready=False)
            if value.get("migrationSourceDigest") != history["sourceDigest"]:
                raise ValueError("Refusing to overwrite a different or already active Firestore run")
            return value
        if not target._empty_children(tx):
            raise ValueError("Refusing to import over orphaned Firestore history")
        value = {**target._metadata(ready=False), "migrationSourceDigest": history["sourceDigest"],
                 "migrationState": "importing", "currentGeneration": history["currentGeneration"],
                 "acceptedResults": history["acceptedResults"], "contributorsCount": history["contributorsCount"]}
        tx.create(target.run, value)
        return value

    def summary(already=False):
        return {"runId": run_id, "sourceDigest": history["sourceDigest"], "alreadyImported": already,
                "generations": len(history["generations"]), "jobs": len(history["jobs"]),
                "acceptedResults": history["acceptedResults"], "contributors": history["contributorsCount"],
                "currentGeneration": history["currentGeneration"]}

    try:
        if target._transaction(import_metadata).get("ready"):
            return summary(already=True)
        documents = ([(target.generations.document(str(row["generation"])), row) for row in history["generations"]]
                     + [(target.jobs.document(row["job_id"]), row) for row in history["jobs"]]
                     + [(target.contributors.document(hashlib.sha256(row["contributorId"].encode()).hexdigest()), row) for row in history["contributors"]])
        chunks, chunk, size = [], [], 0
        for ref, value in documents:
            checked_document(value)
            document_bytes = len(canonical(value).encode())
            if chunk and (len(chunk) >= 200 or size+document_bytes > 2_000_000):
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append((ref, value))
            size += document_bytes
        if chunk:
            chunks.append(chunk)
        for chunk in chunks:
            def write_chunk(tx):
                metadata = target._read_metadata(tx, require_ready=False)
                if metadata.get("migrationSourceDigest") != history["sourceDigest"]:
                    raise ValueError("Import ownership changed")
                if metadata.get("ready"):
                    return False
                for ref, value in chunk:
                    tx.set(ref, value)
                return True
            if not target._transaction(write_chunk):
                return summary(already=True)
        # Verify every field, including completed result identities and hashes,
        # before exposing this namespace to the serving constructor.
        for collection, expected in ((target.generations, history["generations"]), (target.jobs, history["jobs"]), (target.contributors, history["contributors"])):
            actual = [snapshot.to_dict() for snapshot in collection.stream()]
            if sorted(map(canonical, actual)) != sorted(map(canonical, expected)):
                current = target.run.get().to_dict()
                if current.get("ready") and current.get("migrationSourceDigest") == history["sourceDigest"]:
                    return summary(already=True)
                raise ValueError("Firestore import verification failed; run remains unavailable")

        def finish(tx):
            value = target._read_metadata(tx, require_ready=False)
            if value.get("migrationSourceDigest") != history["sourceDigest"]:
                raise ValueError("Import ownership changed")
            if value.get("ready"):
                return False
            tx.update(target.run, {"ready": True, "migrationState": "complete",
                                   "importedGenerations": len(history["generations"]),
                                   "importedJobs": len(history["jobs"])})
            return True
        return summary(already=not target._transaction(finish))
    finally:
        target.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Consistent SQLite snapshot to import")
    parser.add_argument("--config", default="web/training/config.json")
    parser.add_argument("--project", required=True)
    parser.add_argument("--database", default="(default)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate the SQLite snapshot without contacting Firestore")
    args = parser.parse_args()
    config, config_hash = read_config(args.config)
    if args.dry_run:
        history = read_sqlite_history(args.source, config, config_hash)
        result = {"sourceDigest": history["sourceDigest"], "generations": len(history["generations"]),
                  "jobs": len(history["jobs"]), "acceptedResults": history["acceptedResults"],
                  "contributors": history["contributorsCount"], "currentGeneration": history["currentGeneration"]}
    else:
        result = import_sqlite(args.source, config, config_hash, project=args.project,
                               database=args.database, run_id=args.run_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
