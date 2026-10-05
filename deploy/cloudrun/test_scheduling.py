"""Loopback HTTP scheduling checks with synthetic results and a controlled clock.

This exercises the deployed server's error contract; it is not a physical fly run.
"""
import copy
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from sequential_training import SequentialTrainingCoordinator
from test_sequential_training import SequenceFixture, fixture

SPEC = importlib.util.spec_from_file_location("scheduling_http_server", Path(__file__).with_name("server.py"))
server_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server_module)


class SchedulingHTTPTests(unittest.TestCase):
    identity = SequenceFixture.identity
    credentials = SequenceFixture.credentials
    result = SequenceFixture.result

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.now = 1000.
        self.config = fixture()
        self.config["contribution"]["leaseSeconds"] = 180
        self.coordinator = SequentialTrainingCoordinator(":memory:", self.config, clock=lambda: self.now)
        self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        root = Path(self.temporary.name)
        public = root / "public"
        public.mkdir()
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(dict(schemaVersion=1, configHash=self.coordinator.config_hash,
                            modelFingerprint=self.coordinator.model_fingerprint, files={})))
        self.server = server_module.make_server(self.coordinator, public, manifest, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.coordinator.close()
        self.temporary.cleanup()

    def post(self, endpoint, payload):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request("POST", "/api/training/" + endpoint, body=json.dumps(payload),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_sixteen_jobs_share_the_existing_scientific_identity(self):
        jobs = []
        for i in range(16):
            status, response = self.post("lease", self.identity(f"worker-{i}"))
            self.assertEqual(status, 200)
            jobs.append(response["job"])
        self.assertEqual(len({job["jobId"] for job in jobs}), 16)
        self.assertEqual({job["pairId"] for job in jobs}, set(range(8)))
        self.assertEqual({job["configHash"] for job in jobs}, {self.coordinator.config_hash})
        self.assertEqual(self.coordinator.config, self.config)
        self.assertTrue(all(job["assignmentDeadlineAt"] == 2200 for job in jobs))
        status, response = self.post("lease", self.identity("idle-worker"))
        self.assertEqual(status, 200)
        self.assertIsNone(response["job"])

    def test_renewing_worker_times_out_and_another_browser_can_finish_its_trial(self):
        status, response = self.post("lease", self.identity("slow-browser"))
        self.assertEqual(status, 200)
        original = response["job"]
        original_parameters = copy.deepcopy(original["parameters"])
        # All normal heartbeat windows remain valid, including the final one.
        for now in (*range(1120, 2200, 120), 2199):
            self.now = float(now)
            status, response = self.post("heartbeat", self.credentials(original, "slow-browser"))
            self.assertEqual(status, 200)
            self.assertLessEqual(response["leaseExpiresAt"], original["assignmentDeadlineAt"])
        self.now = original["assignmentDeadlineAt"]
        status, response = self.post("heartbeat", self.credentials(original, "slow-browser"))
        self.assertEqual((status, response["error"]), (410, "assignment_timeout"))
        status, response = self.post("result", self.result(original, "slow-browser"))
        self.assertEqual((status, response["error"]), (410, "assignment_timeout"))
        status, response = self.post("lease", self.identity("slow-browser"))
        self.assertEqual(status, 200)
        self.assertNotEqual(response["job"]["jobId"], original["jobId"])
        status, response = self.post("lease", self.identity("fast-browser"))
        self.assertEqual(status, 200)
        replacement = response["job"]
        self.assertEqual(replacement["jobId"], original["jobId"])
        self.assertEqual(replacement["parameters"], original_parameters)
        self.assertEqual(replacement["seed"], original["seed"])
        self.assertNotEqual(replacement["leaseToken"], original["leaseToken"])
        self.assertEqual(replacement["assignmentDeadlineAt"], 3400)
        status, response = self.post("result", self.result(original, "slow-browser"))
        self.assertEqual((status, response["error"]), (409, "stale_lease"))
        payload = self.result(replacement, "fast-browser", score=1.)
        status, response = self.post("result", payload)
        self.assertEqual(status, 200)
        self.assertTrue(response["accepted"])
        status, response = self.post("result", payload)
        self.assertEqual(status, 200)
        self.assertTrue(response["duplicate"])
        self.assertEqual(self.coordinator.status()["acceptedResults"], 1)


if __name__ == "__main__":
    unittest.main()
