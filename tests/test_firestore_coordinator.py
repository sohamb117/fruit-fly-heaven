"""Real local Firestore emulator checks; objectives are algebra fixtures only.

Run with FIRESTORE_EMULATOR_HOST=127.0.0.1:8787 and requirements-cloudrun.txt.
No test is allowed to connect to a non-loopback endpoint or a production project.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import training_coordinator as sqlite_module

EMULATOR = os.environ.get("FIRESTORE_EMULATOR_HOST", "")
try:
    from google.cloud import firestore
    import firestore_coordinator as module
except ImportError:
    firestore = module = None


def fixture():
    return {
        "schemaVersion": 1, "environmentVersion": "banc-flybody-flight-objective-v3",
        "modelFingerprint": hashlib.sha256(("fixture.wasm:" + "b" * 64 + "\n").encode()).hexdigest(),
        "assets": {"fixture.wasm": "b" * 64}, "algorithm": "antithetic-evolution-strategies",
        "dtMs": .5, "bodyBlockMs": 2,
        "parameters": [{"name": "gain_log", "min": -2, "max": 2, "initial": 0}],
        "optimizer": {"populationPairs": 4, "sigma": .25, "learningRate": .035, "maximumUpdate": .15, "seed": 888},
        "objective": {"min": -10, "max": 10, "direction": "maximize"},
        "stage": "landing", "durationSeconds": 8,
        "stages": [{"id": "takeoff", "durationSeconds": 3},
                   {"id": "flight", "durationSeconds": 5},
                   {"id": "landing", "durationSeconds": 8}],
        "contribution": {"leaseSeconds": 10, "maxRequestBytes": 262144},
    }


@unittest.skipUnless(module and EMULATOR, "Requires the local Firestore emulator and requirements-cloudrun.txt")
class FirestoreTests(unittest.TestCase):
    def setUp(self):
        if EMULATOR.rsplit(":", 1)[0] not in ("127.0.0.1", "localhost", "[::1]"):
            self.fail("Firestore tests require an explicitly loopback emulator")
        self.config = fixture()
        self.now = 1000.
        self.temp = tempfile.TemporaryDirectory()
        self.clients = []
        self.client = self.new_client()
        self.run_id = "test-" + uuid.uuid4().hex
        self.coordinator = self.new_coordinator(initialize=True)

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.temp.cleanup()

    def new_client(self):
        client = firestore.Client(project="fly-training-emulator")
        self.clients.append(client)
        return client

    def new_coordinator(self, **kwargs):
        return module.FirestoreCoordinator(self.config, run_id=self.run_id,
            client=kwargs.pop("client", self.client), clock=lambda: self.now, **kwargs)

    def identity(self, contributor="alice"):
        return {"contributorId": contributor, "modelFingerprint": self.coordinator.model_fingerprint,
                "configHash": self.coordinator.config_hash}

    def result_body(self, job, contributor="alice", objective=0):
        provenance = {key: job[key] for key in ("modelFingerprint", "configHash", "parametersHash", "seed", "stage", "durationSeconds", "sign", "pairId", "generation")}
        provenance.update(environmentVersion=self.config["environmentVersion"], backend="wasm", bodyBackend="mujoco-wasm", dtMs=.5, bodyBlockMs=2)
        return {**self.identity(contributor), "jobId": job["jobId"], "leaseToken": job["leaseToken"],
                "objective": objective, "parameters": job["parameters"], "metrics": {"fixture": True}, "provenance": provenance}

    def assert_api_error(self, code, call):
        with self.assertRaises(sqlite_module.APIError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_default_constructor_does_not_create_or_serve_incomplete_runs(self):
        unknown = self.run_id + "-unknown"
        self.assert_api_error("run_not_ready", lambda: module.FirestoreCoordinator(self.config, run_id=unknown, client=self.client))
        self.assertFalse(self.client.collection(module.COLLECTION).document(unknown).get().exists)
        self.coordinator.run.update({"ready": False})
        self.assert_api_error("run_not_ready", lambda: self.new_coordinator())
        self.assert_api_error("run_not_ready", lambda: self.new_coordinator(initialize=True))
        self.assert_api_error("run_not_ready", self.coordinator.status)
        orphan = self.run_id + "-orphan"
        self.client.collection(module.COLLECTION).document(orphan).collection("jobs").document("orphan").set({"generation": 0})
        with self.assertRaisesRegex(ValueError, "orphaned"):
            module.FirestoreCoordinator(self.config, run_id=orphan, client=self.client, initialize=True)

    def test_sqlite_and_firestore_share_seeded_jobs_and_checkpoint_contract(self):
        sqlite = sqlite_module.TrainingCoordinator(Path(self.temp.name) / "parity.sqlite3", self.config, clock=lambda: self.now)
        try:
            self.assertEqual(self.coordinator.checkpoint(), sqlite.checkpoint())
            for i in range(8):
                owner = f"owner-{i}"
                cloud = self.coordinator.lease(self.identity(owner))["job"]
                local = sqlite.lease(self.identity(owner))["job"]
                for key in cloud:
                    if key != "leaseToken":
                        self.assertEqual(cloud[key], local[key], key)
                score = cloud["parameters"][0]
                self.coordinator.result(self.result_body(cloud, owner, score))
                sqlite.result(self.result_body(local, owner, score))
            self.assertEqual(self.coordinator.checkpoint(), sqlite.checkpoint())
            self.assertEqual(self.coordinator.status(), sqlite.status())
        finally:
            sqlite.close()

    def test_multi_instance_concurrent_leases_are_unique_and_retries_idempotent(self):
        peers = [self.new_coordinator(client=self.new_client()) for _ in range(8)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda pair: pair[1].lease(self.identity(f"owner-{pair[0]}"))["job"], enumerate(peers)))
        self.assertEqual(len({job["jobId"] for job in jobs}), 8)
        self.assertEqual(self.coordinator.status()["jobs"], {"pending": 0, "leased": 8, "completed": 0})
        waiting = self.coordinator.lease(self.identity("waiting"))
        self.assertIsNone(waiting["job"])
        self.assertEqual(waiting["waitMs"], 10000)
        with ThreadPoolExecutor(max_workers=4) as pool:
            retries = list(pool.map(lambda peer: peer.lease(self.identity("owner-0")), peers[:4]))
        self.assertTrue(all(value["retry"] and value["job"] == jobs[0] for value in retries))

    def test_fifteen_clients_finish_below_browser_timeout_with_eight_assignments(self):
        peers = [self.new_coordinator(client=self.new_client()) for _ in range(15)]
        barrier = threading.Barrier(15)

        def assign(pair):
            index, peer = pair
            barrier.wait(timeout=5)
            start = time.monotonic()
            value = peer.lease(self.identity(f"visitor-{index}"))
            return value, time.monotonic()-start

        with ThreadPoolExecutor(max_workers=15) as pool:
            replies = list(pool.map(assign, enumerate(peers)))
        jobs = [value["job"] for value, _ in replies if value["job"]]
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({job["jobId"] for job in jobs}), 8)
        self.assertEqual(sum(value["job"] is None and value["waitMs"] == 10000 for value, _ in replies), 7)
        longest = max(duration for _, duration in replies)
        print(f"\n15 concurrent visitors: 8 unique leases, 7 waiting; max response {longest:.3f}s", flush=True)
        self.assertLess(longest, 14, "Lease replies should fit the browser's 15-second timeout")

    def test_same_contributor_initial_requests_get_one_assignment(self):
        peers = [self.new_coordinator(client=self.new_client()) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda peer: peer.lease(self.identity("same-owner")), peers))
        self.assertEqual(len({value["job"]["jobId"] for value in replies}), 1)
        self.assertEqual(len({value["job"]["leaseToken"] for value in replies}), 1)
        self.assertEqual(sum(not value["retry"] for value in replies), 1)
        self.assertEqual(self.coordinator.status()["jobs"], {"pending": 7, "leased": 1, "completed": 0})

    def test_concurrent_submissions_advance_once_and_retries_do_not_count_twice(self):
        peers = [self.new_coordinator(client=self.new_client()) for _ in range(8)]
        jobs = [self.coordinator.lease(self.identity(f"owner-{i}"))["job"] for i in range(8)]
        bodies = [self.result_body(job, f"owner-{i}", objective=job["parameters"][0]) for i, job in enumerate(jobs)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            duplicates = list(pool.map(lambda peer: peer.result(bodies[0]), peers[:4]))
        self.assertEqual(sum(not value["duplicate"] for value in duplicates), 1)
        with ThreadPoolExecutor(max_workers=7) as pool:
            remaining = list(pool.map(lambda pair: pair[1].result(bodies[pair[0]]), enumerate(peers[1:], 1)))
        self.assertEqual(sum(value["nextGenerationCreated"] for value in remaining), 1)
        state = self.coordinator.status()
        self.assertEqual((state["generation"], state["acceptedResults"], state["contributors"]), (1, 8, 8))
        self.assertEqual(state["jobs"], {"pending": 8, "leased": 0, "completed": 0})
        self.assertEqual(len(list(self.coordinator.generations.stream())), 2)
        self.assertEqual(len(list(self.coordinator.jobs.stream())), 16)
        self.assertTrue(self.coordinator.result(bodies[0])["duplicate"])
        changed = copy.deepcopy(bodies[0]); changed["objective"] += .1
        self.assert_api_error("result_conflict", lambda: self.coordinator.result(changed))
        self.assertEqual(self.coordinator.status()["acceptedResults"], 8)

    def test_expiry_release_heartbeat_and_invalid_provenance(self):
        job = self.coordinator.lease(self.identity())["job"]
        token = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        body = self.result_body(job)
        body["provenance"]["dtMs"] = 1
        self.assert_api_error("provenance_mismatch", lambda: self.coordinator.result(body))
        self.now += 9
        self.assertEqual(self.coordinator.heartbeat(token)["leaseExpiresAt"], 1019)
        self.assertTrue(self.coordinator.release(token)["released"])
        self.assertTrue(self.coordinator.release(token)["duplicate"])
        self.assert_api_error("lease_expired", lambda: self.coordinator.result(self.result_body(job)))
        next_job = self.coordinator.lease(self.identity("bob"))["job"]
        self.assertEqual(next_job["jobId"], job["jobId"])
        self.assert_api_error("stale_lease", lambda: self.coordinator.heartbeat(token))
        self.now += 11
        self.assert_api_error("lease_expired", lambda: self.coordinator.result(self.result_body(next_job, "bob")))
        third = self.coordinator.lease(self.identity("charlie"))["job"]
        self.assertEqual(third["jobId"], job["jobId"])
        self.assertNotEqual(third["leaseToken"], next_job["leaseToken"])

    def long_lease_run(self):
        self.config = fixture()
        self.config["contribution"]["leaseSeconds"] = 1800
        self.run_id = "test-lease-" + uuid.uuid4().hex
        self.coordinator = self.new_coordinator(initialize=True)
        return self.coordinator

    def legacy_job(self, contributor="alice"):
        job = self.coordinator.lease(self.identity(contributor))["job"]
        # The deployed predecessor has no explicit renewal timestamp.
        self.coordinator.jobs.document(job["jobId"]).update({"lease_renewed_at": firestore.DELETE_FIELD})
        return job

    def test_operational_timeout_keeps_recent_legacy_owner_and_scientific_identity(self):
        original = self.long_lease_run()
        job = self.legacy_job()
        metadata = original.run.get().to_dict()
        checkpoint = original.checkpoint()
        self.now += 120
        capped = self.new_coordinator(lease_timeout_seconds=180)
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        retry = capped.lease(self.identity())
        self.assertTrue(retry["retry"])
        self.assertEqual(retry["job"], {**job, "leaseExpiresAt": 1180})
        status = capped.status()
        self.assertEqual(status["leaseSeconds"], 180)
        self.assertEqual(status["config"]["contribution"]["leaseSeconds"], 1800)
        self.assertEqual(status["jobs"], {"pending": 7, "leased": 1, "completed": 0})
        self.assertEqual(capped.checkpoint(), checkpoint)
        self.assertEqual(capped.run.get().to_dict(), metadata)
        self.assertEqual(capped.heartbeat(identity)["leaseExpiresAt"], 1300)
        saved = original.jobs.document(job["jobId"]).get().to_dict()
        self.assertEqual((saved["expires"], saved["lease_renewed_at"]), (1300, 1120))
        self.assertNotIn("lease_renewed_at", json.dumps(capped.status()))
        self.assertNotIn("lease_renewed_at", capped.lease(self.identity())["job"])
        self.now = 1299
        self.assertTrue(capped.result(self.result_body(job))["accepted"])
        self.now = 5000
        self.assertTrue(capped.result(self.result_body(job))["duplicate"])
        self.assertEqual(capped.status()["acceptedResults"], 1)

    def test_abandoned_legacy_last_job_is_reassigned_once_at_exact_timeout(self):
        original = self.long_lease_run()
        job = self.legacy_job()
        for index in range(7):
            owner = f"finisher-{index}"
            finished = original.lease(self.identity(owner))["job"]
            original.result(self.result_body(finished, owner))
        capped = self.new_coordinator(lease_timeout_seconds=180)
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        checkpoint = capped.checkpoint()
        self.now = 1179.999
        self.assertIsNone(capped.lease(self.identity("waiting"))["job"])
        self.now = 1180
        self.assertEqual(capped.status()["jobs"], {"pending": 1, "leased": 0, "completed": 7})
        for action in (lambda: capped.result(self.result_body(job)), lambda: capped.heartbeat(identity),
                       lambda: capped.release(identity)):
            self.assert_api_error("lease_expired", action)
        peers = [self.new_coordinator(client=self.new_client(), lease_timeout_seconds=180) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda pair: (pair[0], pair[1].lease(self.identity(f"recovery-{pair[0]}"))), enumerate(peers)))
        assigned = [(index, value["job"]) for index, value in replies if value["job"]]
        self.assertEqual(len(assigned), 1)
        index, replacement = assigned[0]
        self.assertEqual(replacement["jobId"], job["jobId"])
        for field in ("parameters", "parametersHash", "seed", "generation", "pairId", "sign"):
            self.assertEqual(replacement[field], job[field])
        self.assertNotEqual(replacement["leaseToken"], job["leaseToken"])
        for action in (lambda: capped.result(self.result_body(job)), lambda: capped.heartbeat(identity),
                       lambda: capped.release(identity)):
            self.assert_api_error("stale_lease", action)
        self.assertEqual(capped.checkpoint(), checkpoint)
        self.assertEqual(capped.status()["acceptedResults"], 7)
        accepted = capped.result(self.result_body(replacement, f"recovery-{index}"))
        self.assertTrue(accepted["nextGenerationCreated"])
        self.assertEqual(capped.status()["acceptedResults"], 8)
        self.assertTrue(capped.result(self.result_body(replacement, f"recovery-{index}"))["duplicate"])

    def test_short_lease_renews_without_trial_lifetime_limit_or_resurrection(self):
        self.long_lease_run()
        capped = self.new_coordinator(lease_timeout_seconds=180)
        job = capped.lease(self.identity())["job"]
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        self.assertEqual(job["leaseExpiresAt"], 1180)
        for _ in range(31):
            self.now += 60
            renewed = capped.heartbeat(identity)
            self.assertEqual(renewed["leaseExpiresAt"], self.now + 180)
            self.assertEqual(capped.lease(self.identity())["job"]["leaseToken"], job["leaseToken"])
        # A more permissive later policy cannot extend an already issued lease.
        relaxed = self.new_coordinator(lease_timeout_seconds=3600)
        self.assertEqual(relaxed.status()["leaseSeconds"], 1800)
        self.assertEqual(relaxed.lease(self.identity())["job"]["leaseExpiresAt"], renewed["leaseExpiresAt"])
        self.now = renewed["leaseExpiresAt"]
        self.assert_api_error("lease_expired", lambda: relaxed.heartbeat(identity))
        self.assert_api_error("lease_expired", lambda: relaxed.result(self.result_body(job)))

    def test_observer_status_and_preview_persist_across_instances_without_history_changes(self):
        from test_training_coordinator import preview_frame
        sqlite = sqlite_module.TrainingCoordinator(Path(self.temp.name)/"observer.sqlite3", self.config, clock=lambda: self.now)
        try:
            native = self.coordinator.lease(self.identity())["job"]
            local = sqlite.lease(self.identity())["job"]
            initial = self.coordinator.checkpoint()
            for coordinator, job in ((self.coordinator, native), (sqlite, local)):
                coordinator.heartbeat({**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"], "previewFrame": preview_frame()})
            peer = self.new_coordinator(client=self.new_client())
            state = peer.status()
            self.assertEqual(state, sqlite.status())
            self.assertFalse(state["latestFrame"]["stale"])
            self.assertEqual(state["checkpoint"], initial)
            self.assertEqual(state["acceptedResults"], 0)
            saved = self.coordinator.telemetry.document("latest").get().to_dict()
            self.assertNotIn("contributor", saved)
            self.assertNotIn("token", saved)
            self.assertIsInstance(saved["frame_json"], str)
            for coordinator, job in ((self.coordinator, native), (sqlite, local)):
                body = self.result_body(job)
                body["metrics"].update(simSeconds=.1, wallSeconds=2, success=False)
                coordinator.result(body)
            state = peer.status()
            self.assertEqual(state, sqlite.status())
            self.assertTrue(state["latestFrame"]["stale"])
            self.assertEqual(state["recentWallSeconds"], 2)
            self.assertEqual(state["recentTrials"], [{"episode": 1, "generation": 0, "stage": "landing",
                              "return": 0., "success": False, "simSeconds": .1, "wallSeconds": 2, "completedAt": self.now}])
            self.assertNotIn("leaseToken", json.dumps(state["recentTrials"]))
        finally:
            sqlite.close()

    def test_preview_requires_valid_live_lease_and_rejects_malformed_geometry(self):
        from test_training_coordinator import preview_frame
        job = self.coordinator.lease(self.identity())["job"]
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        frame = preview_frame(); frame["position"] = [1e7, 0, 0]
        self.assert_api_error("invalid_preview", lambda: self.coordinator.heartbeat({**identity, "previewFrame": frame}))
        self.assert_api_error("stale_lease", lambda: self.coordinator.heartbeat({**identity, "leaseToken": "x"*32, "previewFrame": preview_frame()}))
        self.now = job["leaseExpiresAt"]+1
        self.assert_api_error("lease_expired", lambda: self.coordinator.heartbeat({**identity, "previewFrame": preview_frame()}))
        self.assertFalse(self.coordinator.telemetry.document("latest").get().exists)
        self.assertIsNone(self.coordinator.status()["latestFrame"])

    def test_new_instance_reads_persisted_state_and_rejects_config_mismatch(self):
        job = self.coordinator.lease(self.identity())["job"]
        self.coordinator.result(self.result_body(job))
        peer = self.new_coordinator(client=self.new_client())
        self.assertEqual(peer.status(), self.coordinator.status())
        for key, value in (("durationSeconds", 2), ("environmentVersion", "banc-flybody-flight-interface-v2")):
            changed = copy.deepcopy(self.config); changed[key] = value
            with self.assertRaisesRegex(ValueError, "different training configuration"):
                module.FirestoreCoordinator(changed, run_id=self.run_id, client=self.client, initialize=True)
        self.assertEqual(peer.status(), self.coordinator.status())

    def make_history(self):
        path = Path(self.temp.name) / "history.sqlite3"
        sqlite = sqlite_module.TrainingCoordinator(path, self.config, clock=lambda: self.now)
        for i in range(9):
            job = sqlite.lease(self.identity("old-owner"))["job"]
            sqlite.result(self.result_body(job, "old-owner", objective=job["parameters"][0]))
        active = sqlite.lease(self.identity("in-flight"))["job"]
        expected = sqlite.status()
        sqlite.close()
        return path, active, expected

    def test_sqlite_import_preserves_all_history_lease_and_checkpoint(self):
        path, active, expected = self.make_history()
        digest_before = hashlib.sha256(path.read_bytes()).hexdigest()
        target = self.run_id + "-import"
        summary = module.import_sqlite(path, self.config, run_id=target, client=self.client)
        imported = module.FirestoreCoordinator(self.config, run_id=target, client=self.client, clock=lambda: self.now)
        self.assertEqual(imported.status(), expected)
        self.assertEqual((summary["generations"], summary["jobs"], summary["acceptedResults"]), (2, 16, 9))
        self.assertEqual(imported.lease(self.identity("in-flight"))["job"], active)
        before_rows = module.read_sqlite_history(path, self.config)["jobs"]
        after_rows = [snapshot.to_dict() for snapshot in imported.jobs.stream()]
        self.assertEqual(sorted(map(sqlite_module.canonical, before_rows)), sorted(map(sqlite_module.canonical, after_rows)))
        old_result = json.loads(next(job["result"] for job in before_rows if job["state"] == "completed"))
        self.assertTrue(imported.result(old_result)["duplicate"])
        imported.result(self.result_body(active, "in-flight"))
        self.assertTrue(module.import_sqlite(path, self.config, run_id=target, client=self.client)["alreadyImported"])
        self.assertEqual(imported.status()["acceptedResults"], 10)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest_before)
        with self.assertRaisesRegex(ValueError, "already active"):
            module.import_sqlite(path, self.config, run_id=self.run_id, client=self.client)

    def test_interrupted_import_fails_closed_and_resumes(self):
        path, _, expected = self.make_history()
        target = self.run_id + "-interrupted"
        original = module.FirestoreCoordinator._transaction
        calls = 0

        def interrupted(coordinator, operation, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("fixture interruption before ready marker")
            return original(coordinator, operation, **kwargs)

        with patch.object(module.FirestoreCoordinator, "_transaction", interrupted):
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                module.import_sqlite(path, self.config, run_id=target, client=self.client)
        self.assert_api_error("run_not_ready", lambda: module.FirestoreCoordinator(self.config, run_id=target, client=self.client))
        module.import_sqlite(path, self.config, run_id=target, client=self.client)
        imported = module.FirestoreCoordinator(self.config, run_id=target, client=self.client, clock=lambda: self.now)
        self.assertEqual(imported.status(), expected)

    def test_existing_http_handler_roundtrip_uses_firestore_and_plain_download(self):
        server = sqlite_module.make_server(self.coordinator, port=0)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()

        def request(method, endpoint, body=None):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
            try:
                connection.request(method, "/api/training/"+endpoint,
                    None if body is None else json.dumps(body), {"Content-Type": "application/json"})
                response = connection.getresponse()
                return response.status, dict(response.headers), json.loads(response.read())
            finally:
                connection.close()

        try:
            status, _, leased = request("POST", "lease", self.identity())
            self.assertEqual(status, 200)
            self.assertTrue(request("POST", "result", self.result_body(leased["job"]))[2]["accepted"])
            status, headers, checkpoint = request("GET", "checkpoint")
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Disposition"], 'attachment; filename="heaven-checkpoint-generation-0.json"')
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertEqual(checkpoint, self.coordinator.checkpoint())
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


@unittest.skipUnless(module and EMULATOR, "Requires the local Firestore emulator and requirements-cloudrun.txt")
class GuardedFirestoreTests(unittest.TestCase):
    """Real emulator transactions with deterministic algebraic episode reports."""

    def setUp(self):
        if EMULATOR.rsplit(":", 1)[0] not in ("127.0.0.1", "localhost", "[::1]"):
            self.fail("Guard tests require the loopback emulator")
        self.config = fixture()
        self.config["schemaVersion"] = 2
        self.config["optimizer"]["acceptance"] = {"profile": 1, "proposal": "best-search-job", "seedCount": 3,
            "nativeExecution": {"backend": "dawn-metal", "moduleSha256": "c"*64, "packageLockSha256": "d"*64}}
        self.config["validation"] = {"seeds": [190888, 290888], "testSeeds": [1190888, 1290888, 1390888]}
        self.now = 1000.
        self.temp = tempfile.TemporaryDirectory()
        self.clients = []
        self.client = self.new_client()
        self.run_id = "guard-"+uuid.uuid4().hex
        self.coordinator = self.new_coordinator(initialize=True)

    tearDown = FirestoreTests.tearDown
    new_coordinator = FirestoreTests.new_coordinator
    identity = FirestoreTests.identity
    assert_api_error = FirestoreTests.assert_api_error

    def new_client(self):
        client = firestore.Client(project="demo-flyheaven-guard")
        self.clients.append(client)
        return client

    def result_body(self, job, contributor="alice", objective=0):
        body = FirestoreTests.result_body(self, job, contributor, objective)
        body["metrics"] = {"success": False, "terminated": False, "cancelled": False,
                           "simSeconds": 8., "steps": 4000, "reason": "time_limit"}
        body["provenance"].update(backend="webgpu", neuralEngine="dawn-metal",
            nativeWebGPU={"backend": "dawn-metal", "moduleSha256": "c"*64, "packageLockSha256": "d"*64,
                          "adapter": {"isFallbackAdapter": False}})
        return body

    def lease(self, owner):
        return self.coordinator.lease(self.identity(owner))["job"]

    def test_guarded_sqlite_firestore_two_generations_and_real_import_agree(self):
        path = Path(self.temp.name)/"guarded.sqlite3"
        sqlite = sqlite_module.TrainingCoordinator(path, self.config, clock=lambda: self.now)
        try:
            for generation in range(2):
                incumbent = self.coordinator.checkpoint()["parameters"]
                for index in range(8):
                    owner = f"g{generation}-s{index}"
                    cloud, local = self.lease(owner), sqlite.lease(self.identity(owner))["job"]
                    self.assertEqual({k: v for k, v in cloud.items() if k != "leaseToken"},
                                     {k: v for k, v in local.items() if k != "leaseToken"})
                    score = cloud["parameters"][0]
                    self.assertFalse(self.coordinator.result(self.result_body(cloud, owner, score))["nextGenerationCreated"])
                    sqlite.result(self.result_body(local, owner, score))
                self.assertEqual(self.coordinator.checkpoint()["parameters"], incumbent)
                self.assertEqual(self.coordinator.status()["generationPhase"], "comparison")
                self.assertEqual(self.coordinator.status(), sqlite.status())
                snapshot = module.read_sqlite_history(path, self.config)
                self.assertEqual(snapshot["generations"][-1]["status"], "checking")
                restarted = self.new_coordinator(client=self.new_client())
                self.assertEqual(restarted.checkpoint(), sqlite.checkpoint())
                for index in range(6):
                    owner = f"g{generation}-a{index}"
                    cloud, local = self.lease(owner), sqlite.lease(self.identity(owner))["job"]
                    self.assertEqual((cloud["seed"], cloud["parameters"]), (local["seed"], local["parameters"]))
                    score = (.2 if generation == 0 else -.2) if cloud["sign"] == 1 else 0.
                    reply = self.coordinator.result(self.result_body(cloud, owner, score))
                    sqlite.result(self.result_body(local, owner, score))
                    self.assertEqual(reply["nextGenerationCreated"], index == 5)
                self.assertEqual(self.coordinator.checkpoint(), sqlite.checkpoint())
                self.assertEqual(self.coordinator.status(), sqlite.status())
                decision = self.coordinator.generations.document(str(generation)).get().to_dict()["acceptance"]
                self.assertEqual(decision["decision"], "accepted" if generation == 0 else "rejected")
            before = list(sqlite.db.iterdump())
            target = self.run_id+"-import"
            summary = module.import_sqlite(path, self.config, run_id=target, client=self.client)
            imported = module.FirestoreCoordinator(self.config, run_id=target, client=self.client, clock=lambda: self.now)
            self.assertEqual(imported.status(), sqlite.status())
            self.assertEqual((summary["generations"], summary["jobs"], summary["acceptedResults"]), (3, 36, 28))
            self.assertEqual(list(sqlite.db.iterdump()), before)
            self.assertEqual(imported.run.get().to_dict()["schemaVersion"], 2)
            self.assertEqual(imported.checkpoint()["status"], "unverified")
            self.assertFalse(imported.checkpoint()["heldOutValidated"])
        finally:
            sqlite.close()

    def test_concurrent_guard_phase_boundaries_and_old_result_retries(self):
        search = [(f"s{i}", self.lease(f"s{i}")) for i in range(8)]
        payloads = [self.result_body(job, owner, job["parameters"][0]) for owner, job in search]
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(self.coordinator.result, payloads+payloads[-1:]*2))
        self.assertEqual(sum(not reply["duplicate"] for reply in replies), 8)
        self.assertFalse(any(reply.get("nextGenerationCreated", False) for reply in replies))
        self.assertEqual(self.coordinator.status()["totalJobs"], 14)
        comparison = [(f"a{i}", self.lease(f"a{i}")) for i in range(6)]
        bodies = [self.result_body(job, owner, .2 if job["sign"] == 1 else 0.) for owner, job in comparison]
        with ThreadPoolExecutor(max_workers=3) as pool:
            replies = list(pool.map(self.coordinator.result, bodies+bodies[-1:]*2))
        self.assertEqual(sum(reply.get("nextGenerationCreated", False) for reply in replies), 1)
        self.assertEqual(self.coordinator.status()["acceptedResults"], 14)
        self.assertEqual(len(list(self.coordinator.jobs.stream())), 22)
        for body in (payloads[-1], bodies[-1]):
            self.assertTrue(self.coordinator.result(body)["duplicate"])
            changed = copy.deepcopy(body); changed["objective"] += 1
            self.assert_api_error("result_conflict", lambda: self.coordinator.result(changed))
        self.assertEqual(self.coordinator.status()["acceptedResults"], 14)

    def test_guarded_import_rejects_self_consistent_wrong_initial_center(self):
        path = Path(self.temp.name)/"wrong-initial.sqlite3"
        sqlite = sqlite_module.TrainingCoordinator(path, self.config, clock=lambda: self.now)
        try:
            self.assertEqual(module.read_sqlite_history(path, self.config)["currentGeneration"], 0)
            record, jobs = sqlite._generation_records(0, [.25], created=self.now)
            with sqlite.transaction():
                sqlite.db.execute("DELETE FROM jobs")
                sqlite.db.execute("UPDATE generations SET center=? WHERE generation=0", (sqlite_module.canonical(record["center"]),))
                sqlite._insert_jobs(jobs)
            # Job hashes, noise and coordinates all agree with the tampered
            # center; the configured initial condition is the remaining gate.
            sqlite._validate_guard_batch(dict(sqlite.db.execute("SELECT * FROM generations").fetchone()),
                                        list(sqlite.db.execute("SELECT * FROM jobs")))
            with self.assertRaisesRegex(ValueError, "initial center"):
                module.read_sqlite_history(path, self.config)
            with self.assertRaisesRegex(ValueError, "initial center"):
                module.import_sqlite(path, self.config, run_id=self.run_id+"-bad", client=self.client)
        finally:
            sqlite.close()

    def test_identical_nominee_advances_once_without_comparison_documents(self):
        self.config["parameters"][0].update(initial=1., searchScale=5e-324)
        self.run_id += "-identical"
        self.coordinator = self.new_coordinator(initialize=True)
        for index in range(8):
            owner = str(index); job = self.lease(owner)
            reply = self.coordinator.result(self.result_body(job, owner, 1.))
            self.assertEqual(reply["nextGenerationCreated"], index == 7)
        old = self.coordinator.generations.document("0").get().to_dict()
        self.assertEqual(old["acceptance"]["decision"], "identical_parameters")
        self.assertEqual(old["acceptance"]["comparisonJobIds"], [])
        self.assertEqual(self.coordinator.checkpoint()["parameters"], [1.])
        self.assertEqual(self.coordinator.status()["acceptedResults"], 8)
        self.assertEqual(len(list(self.coordinator.jobs.stream())), 16)


@unittest.skipUnless(module and EMULATOR, "Requires the local Firestore emulator and requirements-cloudrun.txt")
class WasmGuardedFirestoreTests(unittest.TestCase):
    def setUp(self):
        from test_training_acceptance import wasm_fixture
        if EMULATOR.rsplit(":", 1)[0] not in ("127.0.0.1", "localhost", "[::1]"):
            self.fail("WASM guard tests require the loopback emulator")
        self.config = wasm_fixture()
        self.now = 1000.
        self.temp = tempfile.TemporaryDirectory()
        self.clients = []
        self.client = self.new_client()
        self.run_id = "wasm-guard-"+uuid.uuid4().hex
        self.coordinator = self.new_coordinator(initialize=True)

    tearDown = FirestoreTests.tearDown
    new_client = FirestoreTests.new_client
    new_coordinator = FirestoreTests.new_coordinator
    identity = FirestoreTests.identity
    assert_api_error = FirestoreTests.assert_api_error

    def test_wasm_cohort_accept_reject_and_migration_match_sqlite_without_native_reports(self):
        from test_training_acceptance import wasm_result
        path = Path(self.temp.name)/"wasm.sqlite3"
        sqlite = sqlite_module.TrainingCoordinator(path, self.config, clock=lambda: self.now)
        try:
            self.assertEqual(self.coordinator.status(), sqlite.status())
            for generation, gain in ((0, .1), (1, -.1)):
                incumbent = self.coordinator.checkpoint()["parameters"]
                for phase, count in (("search", 4), ("comparison", 6)):
                    for index in range(count):
                        owner = f"{generation}-{phase}-{index}"
                        cloud = self.coordinator.lease(self.identity(owner))["job"]
                        local = sqlite.lease(self.identity(owner))["job"]
                        self.assertEqual(cloud["parameters"], local["parameters"])
                        score = cloud["parameters"][0] if phase == "search" else gain if cloud["sign"] == 1 else 0.
                        payload = wasm_result(self.coordinator, cloud, owner, score)
                        if generation == 0 and phase == "search" and index == 0:
                            wrong = copy.deepcopy(payload); wrong["provenance"]["nativeWebGPU"] = {"backend": "dawn-metal"}
                            self.assert_api_error("provenance_mismatch", lambda: self.coordinator.result(wrong))
                            self.assertEqual(self.coordinator.status()["acceptedResults"], 0)
                        self.coordinator.result(payload)
                        sqlite.result(wasm_result(sqlite, local, owner, score))
                    self.assertEqual(self.coordinator.status(), sqlite.status())
                if gain > 0:
                    self.assertNotEqual(self.coordinator.checkpoint()["parameters"], incumbent)
                else:
                    self.assertEqual(self.coordinator.checkpoint()["parameters"], incumbent)
            imported_id = self.run_id+"-imported"
            module.import_sqlite(path, self.config, run_id=imported_id, client=self.client)
            imported = module.FirestoreCoordinator(self.config, run_id=imported_id, client=self.client, clock=lambda: self.now)
            self.assertEqual(imported.status(), sqlite.status())
            self.assertEqual(imported.status()["acceptedResults"], 20)
            self.assertFalse(imported.checkpoint()["heldOutValidated"])
        finally:
            sqlite.close()


if __name__ == "__main__":
    unittest.main()
