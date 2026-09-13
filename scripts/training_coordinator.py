#!/usr/bin/env python3
"""Local, persistent coordinator for unverified paired-ES evaluations.

This service schedules reported evaluations. It cannot verify anonymous remote
execution and never promotes a curriculum or labels a checkpoint validated.
Only Python's standard library is required.
"""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import re
import secrets
import sqlite3
import threading
import time
from urllib.parse import urlsplit


DEFAULT_ORIGINS = tuple(f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (7842, 7843))
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MAX_BODY_BYTES = 65536


class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def finite_number(value, low, high, name):
    valid = not isinstance(value, bool) and isinstance(value, (int, float))
    try:
        number = float(value) if valid else math.nan
    except (OverflowError, ValueError):
        number = math.nan
    if not math.isfinite(number) or not low <= number <= high:
        raise APIError(400, "invalid_value", f"{name} must be finite and between {low} and {high}")
    return number


def identifier(value, name):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise APIError(400, "invalid_identifier", f"Invalid {name}")
    return value


def validate_json(value, depth=0):
    """Bound all auxiliary metrics/provenance, including unrecognized leaves."""
    if depth > 8:
        raise APIError(400, "invalid_value", "JSON nesting exceeds eight levels")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        finite_number(value, -1e15, 1e15, "JSON number")
    elif isinstance(value, str):
        if len(value) > 4096:
            raise APIError(400, "invalid_value", "String is too long")
    elif isinstance(value, list):
        if len(value) > 1024:
            raise APIError(400, "invalid_value", "Array is too long")
        for item in value:
            validate_json(item, depth + 1)
    elif isinstance(value, dict):
        if len(value) > 128 or any(not isinstance(key, str) or len(key) > 128 for key in value):
            raise APIError(400, "invalid_value", "Object has too many or invalid keys")
        for item in value.values():
            validate_json(item, depth + 1)
    else:
        raise APIError(400, "invalid_value", "Unsupported JSON value")


def read_config(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw, parse_constant=reject_constant)
    return value, hashlib.sha256(raw).hexdigest()


class CoordinatorRules:
    """Storage-independent validation, deterministic jobs, and ES mathematics."""

    def _configure(self, config, config_hash=None, clock=time.time):
        self.config = config
        self.config_hash = config_hash or fingerprint(config)
        self.clock = clock
        self._parse_config()

    @staticmethod
    def _array(value):
        return json.loads(value) if isinstance(value, str) else value

    def _parse_config(self):
        if not isinstance(self.config, dict):
            raise ValueError("Config must be an object")
        if type(self.config.get("schemaVersion")) is not int or self.config["schemaVersion"] != 1:
            raise ValueError("Unsupported training config schemaVersion")
        self.model_fingerprint = identifier(self.config.get("modelFingerprint"), "modelFingerprint")
        if self.model_fingerprint.lower().startswith("pending"):
            raise ValueError("Build manifest is pending; coordinator requires a finalized modelFingerprint")
        assets = self.config.get("assets")
        if not isinstance(assets, dict) or not 1 <= len(assets) <= 512:
            raise ValueError("Config requires a nonempty pinned asset manifest")
        if any(not isinstance(url, str) or not url or len(url) > 512 or "\n" in url or
               not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
               for url, digest in assets.items()):
            raise ValueError("Invalid pinned asset manifest")
        # prepare-training-manifest.mjs writes the ordered asset map. Preserve
        # file order rather than imposing a different language's locale sort.
        manifest_hash = hashlib.sha256("".join(f"{url}:{digest}\n" for url, digest in assets.items()).encode()).hexdigest()
        if self.model_fingerprint != manifest_hash:
            raise ValueError("modelFingerprint does not match the pinned asset manifest")
        self.environment_version = identifier(self.config.get("environmentVersion"), "environmentVersion")
        self.algorithm = self.config.get("algorithm")
        if self.algorithm != "antithetic-evolution-strategies":
            raise ValueError("Unsupported training algorithm")
        self.dt_ms = finite_number(self.config.get("dtMs"), .001, 100, "dtMs")
        self.body_block_ms = finite_number(self.config.get("bodyBlockMs"), .001, 1000, "bodyBlockMs")
        parameters = self.config.get("parameters")
        if not isinstance(parameters, list) or not 1 <= len(parameters) <= 64:
            raise ValueError("Config parameters must contain 1–64 bounded entries")
        self.names, self.bounds, self.initial = [], [], []
        for item in parameters:
            name = identifier(item.get("name"), "parameter name")
            lower = finite_number(item.get("min"), -1e9, 1e9, name + " min")
            upper = finite_number(item.get("max"), -1e9, 1e9, name + " max")
            if lower >= upper or name in self.names:
                raise ValueError("Parameter bounds must increase and names must be unique")
            initial = finite_number(item.get("initial"), lower, upper, name + " initial")
            self.names.append(name)
            self.bounds.append((lower, upper))
            self.initial.append(initial)
        es = self.config.get("optimizer", {})
        self.pairs = int(finite_number(es.get("populationPairs", 4), 1, 128, "populationPairs"))
        if es.get("populationPairs", 4) != self.pairs:
            raise ValueError("populationPairs must be an integer")
        self.sigma = finite_number(es.get("sigma", .05), 1e-6, 1, "sigma")
        self.learning_rate = finite_number(es.get("learningRate", .05), 1e-9, 1, "learningRate")
        self.maximum_update = finite_number(es.get("maximumUpdate", .15), 1e-9, 100, "maximumUpdate")
        self.seed = int(finite_number(es.get("seed", 1), 0, 2**32 - 1, "seed"))
        if es.get("seed", 1) != self.seed:
            raise ValueError("seed must be an integer")
        self.stage = identifier(self.config.get("stage", "ground_support"), "stage")
        self.duration = finite_number(self.config.get("durationSeconds", .2), .001, 3600, "durationSeconds")
        objective = self.config.get("objective", {})
        if objective.get("direction") != "maximize":
            raise ValueError("Only maximizing the configured objective is supported")
        limits = [objective.get("min"), objective.get("max")]
        self.score_bounds = [finite_number(x, -1e9, 1e9, "score bound") for x in limits]
        if self.score_bounds[0] >= self.score_bounds[1]:
            raise ValueError("scoreBounds must increase")
        self.lease_seconds = finite_number(self.config.get("contribution", {}).get("leaseSeconds", 1800), 1, 86400, "leaseSeconds")
        self.max_request_bytes = int(finite_number(self.config.get("contribution", {}).get("maxRequestBytes", MAX_BODY_BYTES), 1024, 1024*1024, "maxRequestBytes"))


    def _identity(self, payload):
        if not isinstance(payload, dict):
            raise APIError(400, "invalid_request", "Request body must be an object")
        validate_json(payload)
        contributor = identifier(payload.get("contributorId"), "contributorId")
        if payload.get("modelFingerprint") != self.model_fingerprint:
            raise APIError(409, "incompatible_model", "modelFingerprint does not match this coordinator")
        if payload.get("configHash") != self.config_hash:
            raise APIError(409, "incompatible_config", "configHash does not match this coordinator")
        return contributor


    def _job_json(self, row):
        return {"jobId": row["job_id"], "leaseToken": row["token"], "leaseExpiresAt": row["expires"],
                "modelFingerprint": self.model_fingerprint, "configHash": self.config_hash,
                "parameters": self._array(row["parameters"]), "parameterNames": self.names,
                "parametersHash": row["parameters_hash"], "seed": row["seed"], "stage": self.stage,
                "durationSeconds": self.duration, "sign": row["sign"], "pairId": row["pair_id"], "generation": row["generation"]}


    def _lease_contributor(self, payload):
        contributor = self._identity(payload)
        for key, value in (("stage", self.stage), ("durationSeconds", self.duration)):
            if key in payload and (payload[key] != value or isinstance(payload[key], bool)):
                raise APIError(409, "incompatible_stage", "Shared evaluation stage and duration are fixed by the coordinator")
        if "parameters" in payload:
            raise APIError(400, "invalid_request", "The coordinator assigns shared parameter candidates")
        return contributor

    def _lease_credentials(self, payload):
        job_id = identifier(payload.get("jobId"), "jobId")
        token = payload.get("leaseToken")
        if not isinstance(token, str) or not 16 <= len(token) <= 128:
            raise APIError(400, "invalid_token", "Invalid leaseToken")
        return job_id, token

    def _verify_assignment(self, row, contributor, token):
        if row is None:
            raise APIError(404, "unknown_job", "Unknown jobId")
        if row["contributor"] != contributor or row["token"] is None or not secrets.compare_digest(row["token"], token):
            raise APIError(409, "stale_lease", "Lease token or contributor does not match the current assignment")
        return row


    def _result_inputs(self, payload):
        contributor = self._identity(payload)
        score = finite_number(payload.get("objective"), *self.score_bounds, "objective")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise APIError(400, "missing_provenance", "Result requires provenance")
        if not isinstance(payload.get("metrics", {}), dict):
            raise APIError(400, "invalid_metrics", "metrics must be an object")
        result_hash = fingerprint(payload)
        return contributor, score, result_hash

    def _validate_result_provenance(self, row, payload):
        provenance = payload["provenance"]
        expected = self._job_json(row)
        expected.update(environmentVersion=self.environment_version, dtMs=self.dt_ms,
                        bodyBlockMs=self.body_block_ms, bodyBackend="mujoco-wasm")
        for key in ("modelFingerprint", "configHash", "dtMs", "bodyBlockMs", "bodyBackend", "seed", "stage"):
            if provenance.get(key) != expected[key] or isinstance(provenance.get(key), bool):
                raise APIError(409, "provenance_mismatch", f"Result provenance {key} does not match lease")
        version_keys = [key for key in ("envVersion", "environmentVersion") if key in provenance]
        if not version_keys or any(provenance[key] != self.environment_version for key in version_keys):
            raise APIError(409, "provenance_mismatch", "Result environment version does not match lease")
        if provenance.get("backend") not in ("webgpu", "wasm"):
            raise APIError(409, "provenance_mismatch", "Unsupported reported neural backend")
        for key in ("parametersHash", "durationSeconds", "sign", "pairId", "generation"):
            if key in provenance and (provenance[key] != expected[key] or isinstance(provenance[key], bool)):
                raise APIError(409, "provenance_mismatch", f"Result provenance {key} does not match lease")
            if key in payload and (payload[key] != expected[key] or isinstance(payload[key], bool)):
                raise APIError(409, "provenance_mismatch", f"Result {key} does not match lease")
        if "parameters" in payload and (not isinstance(payload["parameters"], list) or payload["parameters"] != expected["parameters"]
                                        or any(isinstance(x, bool) for x in payload["parameters"])):
            raise APIError(409, "provenance_mismatch", "Reported parameters do not match lease")

    def _next_center(self, old, rows):
        center = self._array(old["center"])
        gradient = [0.] * len(center)
        for index in range(0, len(rows), 2):
            positive, negative = rows[index:index+2]
            assert positive["pair_id"] == negative["pair_id"] and positive["sign"] == 1 and negative["sign"] == -1
            difference = positive["score"] - negative["score"]
            for axis, epsilon in enumerate(self._array(positive["noise"])):
                gradient[axis] += difference*epsilon
        updates = [max(-self.maximum_update, min(self.maximum_update,
                   self.learning_rate*grad/(2*self.pairs*self.sigma))) for grad in gradient]
        updated = [max(low, min(high, value + update)) for value, update, (low, high) in zip(center, updates, self.bounds)]
        if not all(math.isfinite(value) for value in updated):
            raise APIError(422, "invalid_aggregate", "Aggregate is not finite")
        return updated

    def _generation_records(self, generation, center, created=None):
        record = {"generation": generation, "center": list(center), "status": "evaluating",
                  "created": self.clock() if created is None else created, "finished": None}
        jobs = []
        for pair in range(self.pairs):
            digest = hashlib.sha256(f"{self.config_hash}:{self.seed}:{generation}:{pair}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            noise = [rng.gauss(0, 1) for _ in center]
            seed = int.from_bytes(digest[8:12], "big")
            pair_id = f"g{generation}-p{pair}"
            for sign in (1, -1):
                parameters = [max(low, min(high, value + sign*self.sigma*epsilon))
                              for value, epsilon, (low, high) in zip(center, noise, self.bounds)]
                jobs.append({"job_id": pair_id + ("-pos" if sign == 1 else "-neg"),
                             "generation": generation, "pair_id": pair_id, "sign": sign, "seed": seed,
                             "noise": list(noise), "parameters": parameters,
                             "parameters_hash": fingerprint(parameters), "state": "pending",
                             "contributor": None, "token": None, "expires": None,
                             "result_hash": None, "result": None, "score": None, "completed": None})
        return record, jobs

    def _checkpoint_from_row(self, row):
        center = self._array(row["center"])
        return {"schemaVersion": 1, "status": "unverified", "heldOutValidated": False,
                "biologicalSuccessValidated": False, "generation": row["generation"], "stage": self.stage,
                "modelFingerprint": self.model_fingerprint, "configHash": self.config_hash,
                "parameterNames": self.names, "arraynames": self.names, "parameters": center, "algorithm": self.algorithm,
                "completedGenerations": row["generation"], "createdAt": row["created"],
                "provenance": "Aggregated anonymous contributor reports; execution and outcomes are unverified."}


    def _status_from_rows(self, current, rows, accepted, contributors):
        now = self.clock()
        counts = {"pending": 0, "leased": 0, "completed": 0}
        for row in rows:
            state = "pending" if row["state"] == "leased" and row["expires"] <= now else row["state"]
            counts[state] += 1
        return {"schemaVersion": 1, "status": "unverified", "generation": current["generation"],
                "stage": self.stage, "modelFingerprint": self.model_fingerprint, "configHash": self.config_hash,
                "config": self.config, "jobs": counts, "completedJobs": counts["completed"], "totalJobs": len(rows),
                "checkpointStatus": "unverified", "acceptedResults": accepted, "contributors": contributors,
                "leaseSeconds": self.lease_seconds, "automaticCurriculumPromotion": False,
                "checkpoint": self._checkpoint_from_row(current)}



class TrainingCoordinator(CoordinatorRules):
    """Thread-safe SQLite state machine; all lease/result transitions atomic."""

    def __init__(self, database, config, config_hash=None, clock=time.time):
        self._configure(config, config_hash, clock)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(database), timeout=10, check_same_thread=False, isolation_level=None)
        try:
            self._initialize_database()
        except BaseException:
            self.db.close()
            raise


    def _initialize_database(self):
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS generations(
                generation INTEGER PRIMARY KEY, center TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, finished REAL);
            CREATE TABLE IF NOT EXISTS jobs(
                job_id TEXT PRIMARY KEY, generation INTEGER NOT NULL REFERENCES generations(generation),
                pair_id TEXT NOT NULL, sign INTEGER NOT NULL CHECK(sign IN (-1,1)),
                seed INTEGER NOT NULL, noise TEXT NOT NULL, parameters TEXT NOT NULL,
                parameters_hash TEXT NOT NULL, state TEXT NOT NULL,
                contributor TEXT, token TEXT, expires REAL,
                result_hash TEXT, result TEXT, score REAL, completed REAL);
            CREATE INDEX IF NOT EXISTS jobs_pending ON jobs(generation,state,expires);
        """)
        with self.transaction():
            stored = self.db.execute("SELECT value FROM meta WHERE key='configHash'").fetchone()
            if stored and stored[0] != self.config_hash:
                raise ValueError("Existing database uses a different configHash; use its original config or a new database")
            stored_model = self.db.execute("SELECT value FROM meta WHERE key='modelFingerprint'").fetchone()
            if stored_model and stored_model[0] != self.model_fingerprint:
                raise ValueError("Existing database uses a different modelFingerprint")
            stored_config = self.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
            if stored_config and stored_config[0] != canonical(self.config):
                raise ValueError("Existing database config content differs, despite supplied configHash")
            if not stored:
                self.db.executemany("INSERT INTO meta(key,value) VALUES(?,?)", [
                    ("configHash", self.config_hash), ("modelFingerprint", self.model_fingerprint),
                    ("config", canonical(self.config)), ("schemaVersion", "1")])
                self._create_generation(0, self.initial)


    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise


    def close(self):
        with self.lock:
            self.db.close()


    def _create_generation(self, generation, center):
        record, jobs = self._generation_records(generation, center)
        self.db.execute("INSERT INTO generations VALUES(?,?,?,?,?)",
                        (generation, canonical(record["center"]), record["status"], record["created"], record["finished"]))
        for job in jobs:
            self.db.execute("""INSERT INTO jobs(job_id,generation,pair_id,sign,seed,noise,parameters,parameters_hash,state)
                VALUES(?,?,?,?,?,?,?,?,?)""", (job["job_id"], generation, job["pair_id"], job["sign"], job["seed"],
                    canonical(job["noise"]), canonical(job["parameters"]), job["parameters_hash"], job["state"]))

    def _current(self):
        return self.db.execute("SELECT * FROM generations ORDER BY generation DESC LIMIT 1").fetchone()


    def lease(self, payload):
        contributor = self._lease_contributor(payload)
        with self.transaction():
            now, generation = self.clock(), self._current()["generation"]
            # Retrying a lost lease response returns the same live assignment.
            row = self.db.execute("SELECT * FROM jobs WHERE state='leased' AND contributor=? AND expires>? ORDER BY job_id LIMIT 1", (contributor, now)).fetchone()
            if row:
                return {"job": self._job_json(row), "retry": True}
            row = self.db.execute("""SELECT * FROM jobs WHERE generation=? AND
                (state='pending' OR (state='leased' AND expires<=?)) ORDER BY pair_id,sign DESC LIMIT 1""", (generation, now)).fetchone()
            if row is None:
                return {"job": None, "waitMs": 2000, "retryAfterSeconds": 2, "status": "waiting_for_results"}
            token = secrets.token_urlsafe(32)
            self.db.execute("UPDATE jobs SET state='leased',contributor=?,token=?,expires=? WHERE job_id=?", (contributor, token, now+self.lease_seconds, row["job_id"]))
            return {"job": self._job_json(self.db.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()), "retry": False}


    def _leased_job(self, payload, contributor):
        job_id, token = self._lease_credentials(payload)
        row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._verify_assignment(row, contributor, token)

    def result(self, payload):
        contributor, score, result_hash = self._result_inputs(payload)
        with self.transaction():
            row = self._leased_job(payload, contributor)
            self._validate_result_provenance(row, payload)
            if row["state"] == "completed":
                if result_hash != row["result_hash"]:
                    raise APIError(409, "result_conflict", "A different result was already accepted for this job")
                return {"accepted": True, "duplicate": True, "status": "unverified", "generation": row["generation"]}
            if row["state"] != "leased" or row["expires"] <= self.clock():
                raise APIError(410, "lease_expired", "Lease expired or was released; request a new assignment")
            self.db.execute("UPDATE jobs SET state='completed',result_hash=?,result=?,score=?,completed=? WHERE job_id=?", (result_hash, canonical(payload), score, self.clock(), row["job_id"]))
            advanced = self._aggregate(row["generation"])
            return {"accepted": True, "duplicate": False, "status": "unverified", "generation": row["generation"], "nextGenerationCreated": advanced}


    def _aggregate(self, generation):
        rows = self.db.execute("SELECT * FROM jobs WHERE generation=? ORDER BY pair_id,sign DESC", (generation,)).fetchall()
        if any(row["state"] != "completed" for row in rows):
            return False
        old = self.db.execute("SELECT * FROM generations WHERE generation=?", (generation,)).fetchone()
        if old["status"] != "evaluating":
            return False
        updated = self._next_center(old, rows)
        self.db.execute("UPDATE generations SET status='unverified',finished=? WHERE generation=?", (self.clock(), generation))
        self._create_generation(generation+1, updated)
        return True


    def release(self, payload):
        contributor = self._identity(payload)
        with self.transaction():
            row = self._leased_job(payload, contributor)
            if row["state"] == "completed":
                raise APIError(409, "already_completed", "Completed results cannot be released")
            if row["state"] == "pending":
                return {"released": True, "duplicate": True}
            if row["expires"] <= self.clock():
                raise APIError(410, "lease_expired", "Lease already expired")
            # Retain token for retry idempotency until the next lease replaces it.
            self.db.execute("UPDATE jobs SET state='pending',expires=NULL WHERE job_id=?", (row["job_id"],))
            return {"released": True, "duplicate": False}


    def heartbeat(self, payload):
        contributor = self._identity(payload)
        with self.transaction():
            row = self._leased_job(payload, contributor)
            if row["state"] != "leased":
                raise APIError(409, "not_leased", "Only an active lease can be renewed")
            if row["expires"] <= self.clock():
                raise APIError(410, "lease_expired", "Expired leases cannot be renewed")
            expires = self.clock() + self.lease_seconds
            self.db.execute("UPDATE jobs SET expires=? WHERE job_id=?", (expires, row["job_id"]))
            return {"renewed": True, "jobId": row["job_id"], "leaseExpiresAt": expires}


    def checkpoint(self):
        with self.lock:
            return self._checkpoint_from_row(self._current())

    def status(self):
        with self.lock:
            current = self._current()
            rows = self.db.execute("SELECT state,expires FROM jobs WHERE generation=?", (current["generation"],)).fetchall()
            contributors = self.db.execute("SELECT COUNT(DISTINCT contributor) FROM jobs WHERE state='completed'").fetchone()[0]
            accepted = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state='completed'").fetchone()[0]
            return self._status_from_rows(current, rows, accepted, contributors)


def validate_origin(origin):
    value = urlsplit(origin)
    if value.scheme not in ("http", "https") or not value.hostname or value.username or value.password or value.path or value.query or value.fragment:
        raise ValueError(f"Invalid allowed origin: {origin}")
    # Validate malformed ports as well.
    value.port
    return origin


def make_server(coordinator, host="127.0.0.1", port=7850, allowed_origins=DEFAULT_ORIGINS, max_body_bytes=MAX_BODY_BYTES):
    origins = frozenset(validate_origin(value) for value in allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format_string, *args):
            # Deliberately do not log request bodies or capability lease tokens.
            pass

        def origin(self):
            origin = self.headers.get("Origin")
            if origin is not None and origin not in origins:
                raise APIError(403, "origin_denied", "Origin is not allowed by this coordinator")
            return origin

        def respond(self, status, value, origin=None, download_filename=None):
            data = canonical(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Vary", "Origin")
            if download_filename is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{download_filename}"')
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.end_headers()
            self.wfile.write(data)

        def request_body(self):
            if self.headers.get("Transfer-Encoding"):
                raise APIError(400, "unsupported_transfer", "Chunked bodies are not supported")
            length = self.headers.get("Content-Length")
            if length is None:
                raise APIError(411, "length_required", "Content-Length is required")
            if len(self.headers.get_all("Content-Length", [])) != 1:
                raise APIError(400, "invalid_length", "Repeated Content-Length is not supported")
            try:
                length = int(length)
            except ValueError:
                raise APIError(400, "invalid_length", "Invalid Content-Length")
            if length < 0:
                raise APIError(400, "invalid_length", "Invalid Content-Length")
            if length > max_body_bytes:
                raise APIError(413, "body_too_large", "Request body exceeds configured limit")
            if self.headers.get_content_type() != "application/json":
                raise APIError(415, "content_type", "Content-Type must be application/json")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise APIError(400, "incomplete_body", "Incomplete request body")
            try:
                return json.loads(raw, parse_constant=reject_constant)
            except (ValueError, UnicodeError, RecursionError):
                raise APIError(400, "invalid_json", "Body must contain valid finite JSON")

        def dispatch(self, method):
            origin = None
            try:
                origin = self.origin()
                path = urlsplit(self.path).path
                download_filename = None
                if method == "GET" and path == "/api/training/status":
                    value = coordinator.status()
                elif method == "GET" and path == "/api/training/checkpoint":
                    value = coordinator.checkpoint()
                    download_filename = f'heaven-checkpoint-generation-{value["generation"]}.json'
                elif method == "POST" and path in ("/api/training/lease", "/api/training/result", "/api/training/release", "/api/training/heartbeat"):
                    value = getattr(coordinator, path.rsplit("/", 1)[1])(self.request_body())
                else:
                    raise APIError(404, "not_found", "Unknown API endpoint")
                self.respond(200, value, origin, download_filename=download_filename)
            except APIError as error:
                self.close_connection = True
                self.respond(error.status, {"error": error.code, "message": error.message}, origin)
            except (TimeoutError, ConnectionError):
                self.close_connection = True
            except Exception:
                self.close_connection = True
                self.respond(500, {"error": "internal_error", "message": "Coordinator could not complete request"}, origin)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

        def do_OPTIONS(self):
            origin = None
            try:
                origin = self.origin()
                if urlsplit(self.path).path not in {"/api/training/" + name for name in ("status", "lease", "result", "release", "heartbeat", "checkpoint")}:
                    raise APIError(404, "not_found", "Unknown API endpoint")
                method = self.headers.get("Access-Control-Request-Method", "GET")
                headers = {x.strip().lower() for x in self.headers.get("Access-Control-Request-Headers", "").split(",") if x.strip()}
                if method not in ("GET", "POST") or not headers <= {"content-type"}:
                    raise APIError(403, "preflight_denied", "Requested method or headers are not allowed")
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.send_header("Vary", "Origin")
                if origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.end_headers()
            except APIError as error:
                self.respond(error.status, {"error": error.code, "message": error.message}, origin)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="web/training/config.json")
    parser.add_argument("--database", default="data/training/coordinator.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7850)
    parser.add_argument("--allow-origin", action="append", help="Exact allowed browser origin; repeat as needed. Overrides localhost defaults.")
    parser.add_argument("--max-request-bytes", type=int, help="Override contribution.maxRequestBytes (1024–1048576)")
    args = parser.parse_args()
    if args.max_request_bytes is not None and not 1024 <= args.max_request_bytes <= 1024*1024:
        parser.error("--max-request-bytes must be between 1024 and 1048576")
    config, config_hash = read_config(args.config)
    database = Path(args.database)
    database.parent.mkdir(parents=True, exist_ok=True)
    coordinator = TrainingCoordinator(database, config, config_hash)
    server = make_server(coordinator, args.host, args.port, args.allow_origin or DEFAULT_ORIGINS,
                         args.max_request_bytes if args.max_request_bytes is not None else coordinator.max_request_bytes)
    print(f"Unverified training coordinator: http://{args.host}:{server.server_address[1]}", flush=True)
    print(f"Config SHA256: {config_hash}; SQLite: {database}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        coordinator.close()


if __name__ == "__main__":
    main()
