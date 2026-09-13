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
        "schemaVersion": 1, "environmentVersion": "banc-flybody-rl-v1",
        "modelFingerprint": hashlib.sha256(("fixture.wasm:" + "b" * 64 + "\n").encode()).hexdigest(),
        "assets": {"fixture.wasm": "b" * 64}, "algorithm": "antithetic-evolution-strategies",
        "dtMs": .5, "bodyBlockMs": 2,
        "parameters": [{"name": "gain_log", "min": -2, "max": 2, "initial": 0}],
        "optimizer": {"populationPairs": 4, "sigma": .25, "learningRate": .035, "maximumUpdate": .15, "seed": 888},
        "objective": {"min": -10, "max": 10, "direction": "maximize"},
        "stage": "posture", "durationSeconds": 1, "stages": [{"id": "posture", "durationSeconds": 1}],
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

    def test_new_instance_reads_persisted_state_and_rejects_config_mismatch(self):
        job = self.coordinator.lease(self.identity())["job"]
        self.coordinator.result(self.result_body(job))
        peer = self.new_coordinator(client=self.new_client())
        self.assertEqual(peer.status(), self.coordinator.status())
        changed = copy.deepcopy(self.config); changed["durationSeconds"] = 2
        with self.assertRaisesRegex(ValueError, "different training configuration"):
            module.FirestoreCoordinator(changed, run_id=self.run_id, client=self.client, initialize=True)

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


if __name__ == "__main__":
    unittest.main()
