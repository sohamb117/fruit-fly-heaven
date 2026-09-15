"""Firestore authority for the sequential training protocol.

SQLite is a disposable transition interpreter, never a second durable store.
Every attempt reads one metadata document and the exact current job batch, plus
an explicitly requested historical job. The resulting jobs, state, counters and
phase checkpoint commit together. No transaction reads completed-job history.

Run ``--dry-run`` for local validation. Creating a run requires --initialize;
the serving process never initializes or repairs a run implicitly.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import time

from google.cloud import firestore

from firestore_coordinator import FirestoreCoordinator, checked_document
from sequential_training import SequentialTrainingCoordinator, scheduling_policy, validate_scheduling
from training_coordinator import APIError, canonical, finite_number, identifier, read_config


COLLECTION = "training_sequences"
SCHEMA = 1
STORAGE_KIND = "banc-sequential-firestore-v1"
RECENT_LIMIT = 120
MAX_TRANSACTION_WRITES = 450
MAX_TRANSACTION_JSON_BYTES = 8_000_000
JOB_COLUMNS = ("job_id", "generation", "role", "payload", "state", "contributor", "token", "expires",
               "result_hash", "result", "score", "completed")


def _invalid(message):
    return APIError(503, "invalid_history", message)


def _require(condition):
    if not condition:
        raise ValueError("Stored sequence invariant failed")


def _rows(engine):
    return {row["job_id"]: dict(row) for row in engine.db.execute("SELECT * FROM sequence_jobs")}


def _batch_ids(engine, state):
    return [row[0] for row in engine.db.execute(
        "SELECT job_id FROM sequence_jobs WHERE generation=? ORDER BY job_id", (state["generation"],))]


def _trial(row):
    job, result = json.loads(row["payload"]), json.loads(row["result"])
    return dict(jobId=row["job_id"], generation=job["generation"], stage=job["stage"], phaseId=job["phaseId"],
                mode=job["mode"], reason=result.get("metrics", {}).get("reason"),
                **{"return": row["score"]}, success=result.get("metrics", {}).get("success", False),
                completedAt=row["completed"])


def _budget(documents):
    if len(documents) > MAX_TRANSACTION_WRITES:
        raise APIError(413, "body_too_large", "Sequential transaction has too many documents")
    # Include conservative per-document encoding overhead; JSON strings remain
    # strings in Firestore, so matrices never become forbidden nested arrays.
    total = sum(len(canonical(checked_document(doc)).encode()) + 1024 for doc in documents)
    if total > MAX_TRANSACTION_JSON_BYTES:
        raise APIError(413, "body_too_large", "Sequential transaction exceeds its storage budget")
    return total


def storage_plan(config, config_hash=None, *, scheduling=None):
    """Validate the protocol and estimate bounded fanout without creating a client."""
    engine = SequentialTrainingCoordinator(":memory:", config, config_hash)
    try:
        if scheduling is not None:
            validate_scheduling(scheduling)
            engine.set_scheduling(search_pairs=scheduling["searchPairs"],
                                  trial_timeout_seconds=scheduling["trialTimeoutSeconds"])
        pairs = engine.pairs if scheduling is None else scheduling["searchPairs"]
        stages, maximum = set(), 0
        for phase in engine.sequence["phases"]:
            stages.add(phase["stage"])
            maximum = max(maximum, 2*pairs, 2*phase["validationCount"],
                          (len(stages)+1)*phase["validationCount"])
        if maximum + 4 > MAX_TRANSACTION_WRITES:
            raise ValueError("Sequence fanout exceeds the Firestore atomic-write budget")
        rows = _rows(engine)
        state = engine._state()
        config_json = canonical(config)
        metadata = dict(storageKind=STORAGE_KIND, storageSchema=SCHEMA, ready=True,
                        configHash=engine.config_hash, modelFingerprint=engine.model_fingerprint,
                        config_json=config_json, state_json=canonical(state),
                        currentJobIds=sorted(rows), contributorsCount=0, recent_trials_json="[]")
        initial_bytes = _budget([metadata, *rows.values()])
        # Parameter magnitudes can change; enforce exact sizes again on every
        # commit. This is an operator estimate, not permission to exceed limits.
        largest = max(len(canonical(row).encode()) + 1024 for row in rows.values())
        return dict(collection=COLLECTION, storageKind=STORAGE_KIND, storageSchema=SCHEMA,
                    configHash=engine.config_hash, modelFingerprint=engine.model_fingerprint,
                    initialJobs=len(rows), maximumBatchJobs=maximum,
                    initialTransactionBytes=initial_bytes,
                    estimatedMaximumPendingBatchBytes=len(canonical(metadata).encode()) + 1024 + maximum*largest,
                    documentLimitBytes=800_000, transactionBudgetBytes=MAX_TRANSACTION_JSON_BYTES,
                    **({"scheduling":scheduling} if scheduling is not None else {}))
    finally:
        engine.close()


class FirestoreSequentialCoordinator:
    """The SQLite sequence API, safe across independent Cloud Run instances."""

    def __init__(self, config, config_hash=None, *, project=None, database="(default)", run_id,
                 client=None, initialize=False, clock=time.time, lease_timeout_seconds=None):
        self.plan = storage_plan(config, config_hash)
        self.config = copy.deepcopy(config)
        self.config_json = canonical(config)
        self.config_hash = self.plan["configHash"]
        self.model_fingerprint = self.plan["modelFingerprint"]
        self.clock = clock
        self.run_id = identifier(run_id, "run_id")
        with self._engine() as engine:
            self.lease_seconds = engine.lease_seconds
            self.acceptance_config = copy.deepcopy(engine.acceptance_config)
            self.pairs = engine.pairs
        self.lease_timeout_seconds = self.lease_seconds if lease_timeout_seconds is None else min(
            self.lease_seconds, finite_number(lease_timeout_seconds, 1, 86400, "lease timeout"))
        self.client = client or firestore.Client(project=project, database=database)
        self._owns_client = client is None
        self.run = self.client.collection(COLLECTION).document(self.run_id)
        self.jobs = self.run.collection("jobs")
        self.checkpoints = self.run.collection("checkpoints")
        self.contributors = self.run.collection("contributors")
        try:
            if initialize:
                self._transaction(self._initialize)
            self._transaction(self._read_snapshot, read_only=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._owns_client:
            self.client.close()

    def _transaction(self, operation, *, read_only=False):
        # Reuse the established SDK abort/retry policy, not its legacy schema.
        return FirestoreCoordinator._transaction(self, operation, read_only=read_only)

    @contextmanager
    def _engine(self, metadata=None, rows=()):
        # Freeze the clock for one transaction attempt. A retry gets a fresh
        # clock and a fresh interpreter, so no uncommitted state can escape.
        now = self.clock()
        engine = SequentialTrainingCoordinator(":memory:", self.config, self.config_hash, clock=lambda: now)
        try:
            if hasattr(self, "lease_timeout_seconds"):
                engine.lease_seconds = self.lease_timeout_seconds
            if metadata is not None:
                engine.db.execute("DELETE FROM sequence_jobs")
                engine.db.execute("DELETE FROM sequence_checkpoints")
                engine.db.execute("UPDATE sequence_state SET value=? WHERE id=1", (metadata["state_json"],))
                engine.db.executemany("INSERT INTO sequence_jobs VALUES(" + ",".join("?" for _ in JOB_COLUMNS) + ")",
                                      [tuple(row[name] for name in JOB_COLUMNS) for row in rows])
                engine._state()
            yield engine
        finally:
            engine.close()

    def _validate_metadata(self, metadata):
        if (metadata.get("storageKind") != STORAGE_KIND or metadata.get("storageSchema") != SCHEMA
                or metadata.get("configHash") != self.config_hash
                or metadata.get("modelFingerprint") != self.model_fingerprint
                or metadata.get("config_json") != self.config_json):
            raise ValueError("Firestore sequence uses a different configuration or storage schema")
        if metadata.get("ready") is not True:
            raise APIError(503, "run_not_ready", "Sequential run is not ready")
        try:
            state = json.loads(metadata["state_json"])
            phase_index, generation = state["phaseIndex"], state["generation"]
            phases = self.config["trainingSequence"]["phases"]
            _require(type(phase_index) is int and 0 <= phase_index <= len(phases))
            _require(type(generation) is int and generation >= 0)
            if "scheduling" in state:
                validate_scheduling(state["scheduling"])
                _require(type(state.get("schedulingRevision")) is int and state["schedulingRevision"] >= 1)
                history = state.get("schedulingHistory")
                _require(isinstance(history, list) and 1 <= len(history) <= 120)
                _require(history[-1]["revision"] == state["schedulingRevision"]
                         and history[-1]["policy"] == state["scheduling"])
                current_pairs = state.get("currentSearchPairs", self.pairs)
                _require(type(current_pairs) is int and 1 <= current_pairs <= 128)
            else:
                _require("currentSearchPairs" not in state)
                current_pairs = self.pairs
            _require(type(metadata["contributorsCount"]) is int and metadata["contributorsCount"] >= 0)
            recent = json.loads(metadata["recent_trials_json"])
            _require(isinstance(recent, list) and len(recent) <= RECENT_LIMIT)
            ids = metadata["currentJobIds"]
            _require(isinstance(ids, list) and ids == sorted(set(ids)))
            for job_id in ids:
                identifier(job_id, "stored jobId")
            if state["status"] in ("complete", "needs-review"):
                expected = 0
            else:
                _require(state["status"] == "running" and phase_index < len(phases))
                phase = phases[phase_index]
                completed_ids = {entry["phaseId"] for entry in state["completedPhases"]}
                gate_stages = {phase["stage"]} | {p["stage"] for p in phases if p["id"] in completed_ids}
                if state["role"] == "search" and "scheduling" in state:
                    _require("currentSearchPairs" in state)
                expected = dict(search=2*current_pairs, fit=1,
                                compare=2*phase["validationCount"],
                                gate=(len(gate_stages)+1)*phase["validationCount"])[state["role"]]
            _require(len(ids) == expected)
        except (KeyError, TypeError, ValueError, APIError) as error:
            raise _invalid("Sequential metadata is malformed or its active batch is incomplete") from error
        return metadata

    def _read_snapshot(self, tx, requested_job_id=None, *, with_jobs=True):
        snapshot = self.run.get(transaction=tx)
        if not snapshot.exists:
            raise APIError(503, "run_not_ready", "Sequential run has not been explicitly initialized")
        metadata = self._validate_metadata(snapshot.to_dict())
        ids = list(metadata["currentJobIds"]) if with_jobs else []
        if requested_job_id and requested_job_id not in ids:
            ids.append(requested_job_id)
        rows = {}
        if ids:
            for snapshot in self.client.get_all([self.jobs.document(job_id) for job_id in ids], transaction=tx):
                if not snapshot.exists:
                    if snapshot.id in metadata["currentJobIds"]:
                        raise _invalid("Current sequential job is missing")
                    continue  # The interpreter returns the normal unknown_job error.
                row = snapshot.to_dict()
                try:
                    _require(set(row) == set(JOB_COLUMNS) and row["job_id"] == snapshot.id)
                    job = json.loads(row["payload"])
                    _require(job["jobId"] == row["job_id"] and job["generation"] == row["generation"])
                    _require(job["configHash"] == self.config_hash and job["modelFingerprint"] == self.model_fingerprint)
                    _require(row["state"] in ("pending", "leased", "completed"))
                    attempt_keys = {"assignmentStartedAt", "assignmentDeadlineAt", "assignmentAttempt",
                                    "assignmentTimeoutSeconds", "assignmentStartSource"}
                    if attempt_keys.intersection(job):
                        _require(attempt_keys <= job.keys())
                        started = finite_number(job["assignmentStartedAt"], 0, 1e12, "assignment start")
                        deadline = finite_number(job["assignmentDeadlineAt"], started, 1e12, "assignment deadline")
                        timeout = finite_number(job["assignmentTimeoutSeconds"], 600, 1e9, "assignment timeout")
                        _require(abs((deadline-started)-timeout) < 1e-6)
                        _require(type(job["assignmentAttempt"]) is int and job["assignmentAttempt"] >= 1)
                        _require(job["assignmentStartSource"] in ("assigned", "migration-grace"))
                    if snapshot.id in metadata["currentJobIds"]:
                        _require(row["generation"] == json.loads(metadata["state_json"])["generation"])
                    else:
                        _require(row["state"] == "completed")
                except (KeyError, TypeError, ValueError) as error:
                    raise _invalid("Stored sequential assignment is malformed") from error
                rows[snapshot.id] = row
        return metadata, rows

    def _initialize(self, tx):
        snapshot = self.run.get(transaction=tx)
        if snapshot.exists:
            self._validate_metadata(snapshot.to_dict())
            return {"initialized": False, "alreadyExists": True}
        # All reads precede all writes, including checks for abandoned imports.
        if any(list(collection.limit(1).stream(transaction=tx))
               for collection in (self.jobs, self.checkpoints, self.contributors)):
            raise ValueError("Refusing to initialize a sequence containing orphaned documents")
        with self._engine() as engine:
            state, rows = engine._state(), _rows(engine)
            metadata = dict(storageKind=STORAGE_KIND, storageSchema=SCHEMA, ready=True,
                            configHash=self.config_hash, modelFingerprint=self.model_fingerprint,
                            config_json=self.config_json, state_json=canonical(state),
                            currentJobIds=sorted(rows), contributorsCount=0, recent_trials_json="[]")
            _budget([metadata, *rows.values()])
            tx.create(self.run, metadata)
            for job_id, row in rows.items():
                tx.create(self.jobs.document(job_id), row)
        return {"initialized": True, "alreadyExists": False}

    def _mutate(self, method, payload):
        payload = copy.deepcopy(payload)
        # Reject invalid identities/credentials before any Firestore access.
        with self._engine() as engine:
            contributor = engine._identity(payload)
            job_id = engine._lease_credentials(payload)[0] if method != "lease" else None
            if method == "result":
                engine._result_inputs(payload)
        contributor_id = hashlib.sha256(contributor.encode()).hexdigest()

        def perform(tx):
            metadata, before = self._read_snapshot(tx, job_id)
            contributor_ref, contributor_doc = None, None
            if method == "result" and before.get(job_id, {}).get("state") != "completed":
                contributor_ref = self.contributors.document(contributor_id)
                contributor_doc = contributor_ref.get(transaction=tx)
            with self._engine(metadata, before.values()) as engine:
                result = getattr(engine, method)(payload)
                after, state = _rows(engine), engine._state()
                updated = {key: row for key, row in after.items() if row != before.get(key)}
                checkpoints = [dict(generation=row["generation"], checkpoint_json=row["value"],
                                    configHash=self.config_hash, modelFingerprint=self.model_fingerprint)
                               for row in engine.db.execute("SELECT * FROM sequence_checkpoints")]
                new_metadata = {**metadata, "state_json": canonical(state), "currentJobIds": _batch_ids(engine, state)}
                contributor_value = None
                if method == "result" and result.get("accepted") and not result.get("duplicate"):
                    trial = _trial(after[job_id])
                    recent = json.loads(metadata["recent_trials_json"])
                    recent = sorted([*recent, trial], key=lambda r: (r["completedAt"], r["jobId"]))[-RECENT_LIMIT:]
                    new_metadata["recent_trials_json"] = canonical(recent)
                    is_new = not contributor_doc.exists
                    new_metadata["contributorsCount"] += int(is_new)
                    old = {} if is_new else contributor_doc.to_dict()
                    contributor_value = dict(contributorId=contributor,
                                             acceptedResults=old.get("acceptedResults", 0) + 1,
                                             lastCompletedAt=trial["completedAt"])
                documents = [*updated.values(), *checkpoints]
                if new_metadata != metadata:
                    documents.append(new_metadata)
                if contributor_value is not None:
                    documents.append(contributor_value)
                _budget(documents)
                # Also bound the next read snapshot. Reject an oversized new
                # auxiliary result atomically, without stranding a huge batch.
                _budget([new_metadata, *[after[key] for key in new_metadata["currentJobIds"]]])
                for key, row in updated.items():
                    if key in before:
                        tx.set(self.jobs.document(key), row)
                    else:
                        tx.create(self.jobs.document(key), row)
                for checkpoint in checkpoints:
                    tx.create(self.checkpoints.document(str(checkpoint["generation"])), checkpoint)
                if new_metadata != metadata:
                    tx.set(self.run, new_metadata)
                if contributor_value is not None:
                    tx.set(contributor_ref, contributor_value)
                return result
        return self._transaction(perform)

    def lease(self, payload):
        return self._mutate("lease", payload)

    def set_scheduling(self, *, search_pairs, trial_timeout_seconds):
        """Explicit atomic administration; not exposed by the HTTP server."""
        policy = scheduling_policy(search_pairs, trial_timeout_seconds)
        storage_plan(self.config, self.config_hash, scheduling=policy)

        def migrate(tx):
            metadata, before = self._read_snapshot(tx)
            with self._engine(metadata, before.values()) as engine:
                result = engine.set_scheduling(search_pairs=search_pairs,
                                              trial_timeout_seconds=trial_timeout_seconds)
                state, after = engine._state(), _rows(engine)
                updated = {key: row for key, row in after.items() if row != before.get(key)}
                new_metadata = {**metadata, "state_json":canonical(state),
                                "currentJobIds":_batch_ids(engine, state)}
                self._validate_metadata(new_metadata)
                _budget([new_metadata, *after.values()])
                for key, row in updated.items():
                    if key in before:tx.set(self.jobs.document(key), row)
                    else:tx.create(self.jobs.document(key), row)
                if new_metadata != metadata:tx.set(self.run, new_metadata)
                return result
        return self._transaction(migrate)

    def result(self, payload):
        return self._mutate("result", payload)

    def heartbeat(self, payload):
        return self._mutate("heartbeat", payload)

    def release(self, payload):
        return self._mutate("release", payload)

    def status(self):
        def read(tx):
            metadata, rows = self._read_snapshot(tx)
            with self._engine(metadata, rows.values()) as engine:
                result = engine.status()
            result["contributors"] = metadata["contributorsCount"]
            result["recentTrials"] = json.loads(metadata["recent_trials_json"])
            return result
        return self._transaction(read, read_only=True)

    def checkpoint(self):
        def read(tx):
            metadata, _ = self._read_snapshot(tx, with_jobs=False)
            with self._engine(metadata) as engine:
                return engine.checkpoint()
        return self._transaction(read, read_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project")
    parser.add_argument("--database", default="(default)")
    parser.add_argument("--lease-timeout-seconds", type=float)
    parser.add_argument("--search-pairs", type=int)
    parser.add_argument("--trial-timeout-seconds", type=float)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate locally; never create a Firestore client")
    mode.add_argument("--initialize", action="store_true", help="Explicitly create a new Firestore sequence namespace")
    mode.add_argument("--set-scheduling", action="store_true", help="Explicitly update scheduling within an existing sequence")
    args = parser.parse_args(argv)
    config, config_hash = read_config(args.config)
    identifier(args.run_id, "run_id")
    scheduling = None
    if args.search_pairs is not None or args.trial_timeout_seconds is not None or args.set_scheduling:
        if args.search_pairs is None or args.trial_timeout_seconds is None:
            parser.error("Scheduling requires --search-pairs and --trial-timeout-seconds")
        if args.initialize:
            parser.error("Initialize first, then explicitly --set-scheduling")
        scheduling = scheduling_policy(args.search_pairs, args.trial_timeout_seconds)
    plan = storage_plan(config, config_hash, scheduling=scheduling)
    if args.lease_timeout_seconds is not None:
        finite_number(args.lease_timeout_seconds, 1, 86400, "lease timeout")
    if args.dry_run:
        print(canonical({**plan, "runId": args.run_id, "dryRun": True, "cloudWrites": False}))
        return
    if not args.project:
        parser.error("Cloud administration requires an explicit --project")
    coordinator = FirestoreSequentialCoordinator(config, config_hash, project=args.project, database=args.database,
                    run_id=args.run_id, initialize=args.initialize, lease_timeout_seconds=args.lease_timeout_seconds)
    try:
        if args.set_scheduling:
            print(canonical({"runId":args.run_id, "cloudWrites":True,
                             **coordinator.set_scheduling(search_pairs=args.search_pairs,
                                                        trial_timeout_seconds=args.trial_timeout_seconds)}))
            return
        print(canonical({**plan, "runId": args.run_id, "ready": True,
                         "sequenceStatus": coordinator.status()["sequence"]["status"]}))
    finally:
        coordinator.close()


if __name__ == "__main__":
    main()
