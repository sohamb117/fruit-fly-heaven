"""Real coordinator transitions with algebra fixtures, not biological evidence."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import training_coordinator

SPEC = importlib.util.spec_from_file_location("flight_report", ROOT / "scripts/report-flight-training.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def fixture(count=27, initial=0):
    return {"schemaVersion": 1, "environmentVersion": "flight-reporter-algebra-fixture",
            "modelFingerprint": hashlib.sha256(("fixture.wasm:" + "b" * 64 + "\n").encode()).hexdigest(),
            "assets": {"fixture.wasm": "b" * 64}, "algorithm": "antithetic-evolution-strategies",
            "dtMs": .5, "bodyBlockMs": 2,
            "parameters": [{"name": f"muscle_{index}_gain", "min": -2, "max": 2, "initial": initial} for index in range(count)],
            "optimizer": {"populationPairs": 2, "sigma": .25, "learningRate": .035, "maximumUpdate": .15, "seed": 888},
            "objective": {"min": -10, "max": 10, "direction": "maximize"},
            "stage": "landing", "durationSeconds": 8,
            "stages": [{"id": "landing", "durationSeconds": 8}]}


class FlightReporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "fixture.sqlite3"
        self.config = fixture()
        self.coordinator = training_coordinator.TrainingCoordinator(self.path, self.config)

    def tearDown(self):
        self.coordinator.close()
        self.temporary.cleanup()

    def accept(self, objective, metrics=None):
        identity = {"contributorId": "reporter-fixture", "modelFingerprint": self.coordinator.model_fingerprint,
                    "configHash": self.coordinator.config_hash}
        job = self.coordinator.lease(identity)["job"]
        provenance = {key: job[key] for key in ("modelFingerprint", "configHash", "parametersHash", "seed", "stage", "durationSeconds", "sign", "pairId", "generation")}
        provenance.update(environmentVersion=self.config["environmentVersion"], backend="wasm", bodyBackend="mujoco-wasm", dtMs=.5, bodyBlockMs=2)
        self.coordinator.result({**identity, "jobId": job["jobId"], "leaseToken": job["leaseToken"],
                                 "objective": objective(job), "parameters": job["parameters"],
                                 "metrics": {"fixture": True, **(metrics or {})}, "provenance": provenance})

    def test_actual_update_and_equal_pair_scores_are_distinct_from_incomplete_generation(self):
        first = reporter.summarize(self.path)
        self.assertEqual(first["parameterCount"], 27)
        self.assertEqual(first["generations"][0]["parameterUpdate"]["status"], "incomplete")
        self.assertIsNone(first["generations"][0]["parameterUpdate"]["changedParameterCount"])
        for _ in range(4):
            self.accept(lambda job: job["parameters"][0])
        updated = reporter.summarize(self.path)
        transition = updated["generations"][0]["parameterUpdate"]
        next_center = self.coordinator.checkpoint()["parameters"]
        self.assertEqual(transition["status"], "updated")
        self.assertEqual(transition["changedParameterCount"], 27)
        self.assertEqual(transition["deltaByParameter"], dict(zip(updated["parameterNames"], next_center)))
        self.assertEqual(transition["l2"], math.hypot(*next_center))
        self.assertEqual(transition["maxAbs"], max(map(abs, next_center)))
        self.assertEqual(updated["behavior"]["takeoffs"], 0)
        self.assertEqual(updated["behavior"]["bestFlightSeconds"], 0)
        # Equal within each pair, but not across pairs: both advantages vanish.
        for _ in range(4):
            self.accept(lambda job: 1 if job["pairId"].endswith("p0") else 2)
        self.accept(lambda job: .3, {"hasTakenOff": True, "bestFlightSeconds": .4, "success": False,
                                    "diagnostics": {"poweredAirborneSeconds": .6}})
        final = reporter.summarize(self.path)
        unchanged = final["generations"][1]["parameterUpdate"]
        self.assertEqual(unchanged["status"], "completed_equal_scores_no_update")
        self.assertTrue(unchanged["equalScoresWithinEveryPair"])
        self.assertEqual(unchanged["changedParameterCount"], 0)
        self.assertEqual(unchanged["l2"], 0)
        self.assertEqual(set(unchanged["deltaByParameter"].values()), {0})
        pending = final["generations"][2]["parameterUpdate"]
        self.assertEqual(pending["status"], "incomplete")
        self.assertEqual(pending["completedJobs"], 1)
        self.assertIsNone(pending["deltaByParameter"])
        self.assertEqual(final["behavior"], {"takeoffs": 1, "successes": 0, "bestFlightSeconds": .4, "maximumPoweredAirborneSeconds": .6})

    def test_clipped_gradient_can_leave_center_unchanged_despite_unequal_pair_scores(self):
        self.coordinator.close()
        self.path = Path(self.temporary.name) / "bounded.sqlite3"
        self.config = fixture(count=1, initial=2)
        self.coordinator = training_coordinator.TrainingCoordinator(self.path, self.config)
        for _ in range(4):
            self.accept(lambda job: job["parameters"][0])
        transition = reporter.summarize(self.path)["generations"][0]["parameterUpdate"]
        self.assertEqual(transition["status"], "completed_no_update")
        self.assertFalse(transition["equalScoresWithinEveryPair"])
        self.assertEqual(transition["changedParameterCount"], 0)

    def test_report_is_read_only_and_does_not_create_missing_database(self):
        self.accept(lambda job: .1)
        before = list(self.coordinator.db.iterdump())
        reporter.summarize(self.path)
        self.assertEqual(list(self.coordinator.db.iterdump()), before)
        missing = Path(self.temporary.name) / "missing.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            reporter.summarize(missing)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
