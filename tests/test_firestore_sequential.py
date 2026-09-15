"""Sequential storage/protocol fixtures, not neural or behavioral evidence.

Emulator tests only run against an explicitly loopback endpoint and the fixed
fly-training-emulator project. Pure tests and the dry-run never create clients.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from test_sequential_training import fixture, phase, SequenceFixture, SchedulingChecks
from sequential_training import SequentialTrainingCoordinator
from training_coordinator import APIError, canonical

try:
    from google.cloud import firestore
    import firestore_sequential as module
except ImportError:
    firestore = module = None

EMULATOR = os.environ.get("FIRESTORE_EMULATOR_HOST", "")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@unittest.skipUnless(module, "Requires requirements-cloudrun.txt")
class StoragePlanTests(unittest.TestCase):
    def test_dry_run_cannot_create_a_client(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(fixture()))
            output = io.StringIO()
            with patch.object(module.firestore, "Client", side_effect=AssertionError("No cloud client")), \
                    patch("sys.stdout", output):
                module.main(["--config", str(config), "--run-id", "fixture", "--dry-run"])
            result = json.loads(output.getvalue())
            self.assertTrue(result["dryRun"])
            self.assertFalse(result["cloudWrites"])
            self.assertEqual(result["collection"], "training_sequences")
            self.assertEqual(result["initialJobs"], 4)
            self.assertLess(result["initialTransactionBytes"], result["transactionBudgetBytes"])

    def test_storage_budget_counts_strings_not_nested_firestore_arrays(self):
        document = {"matrix_json": canonical([[1, 2], [3, 4]])}
        self.assertGreater(module._budget([document]), 1024)
        for documents in ([{"large": "x"*800_000}], [{}]*(module.MAX_TRANSACTION_WRITES+1),
                          [{"large": "x"*700_000} for _ in range(12)]):
            with self.subTest(count=len(documents)), self.assertRaises(APIError) as caught:
                module._budget(documents)
            self.assertEqual(caught.exception.code, "body_too_large")

    def test_interpreter_is_fresh_and_discarded(self):
        adapter = module.FirestoreSequentialCoordinator.__new__(module.FirestoreSequentialCoordinator)
        adapter.config = fixture()
        adapter.config_hash = module.storage_plan(adapter.config)["configHash"]
        adapter.clock = lambda: 1000.
        with adapter._engine() as first:
            first_state = first._state()
            first_state["center"][0] = .123
            first._save(first_state)
            connection = first.db
        with self.assertRaises(Exception):
            connection.execute("SELECT 1")
        with adapter._engine() as second:
            self.assertEqual(second._state()["center"][0], 0.)
        self.assertNotIn("db", vars(adapter))

    def test_server_selects_sequence_store_without_implicit_initialization(self):
        server = load_file("sequence_cloudrun_server", ROOT / "deploy/cloudrun/server.py")
        for config, expected_module, expected_class in ((fixture(), "firestore_sequential", "FirestoreSequentialCoordinator"),
                    ({"legacy": True}, "firestore_coordinator", "FirestoreCoordinator")):
            constructor = Mock()
            fake_module = Mock(**{expected_class: constructor})
            with self.subTest(module=expected_module), \
                    patch.dict(server.os.environ, {"TRAINING_RUN_ID": "fixture-run", "GOOGLE_CLOUD_PROJECT": "fly-training-emulator"}, clear=True), \
                    patch.dict(sys.modules, {expected_module: fake_module}), \
                    patch.object(server.training_coordinator, "read_config", return_value=(config, "c"*64)), \
                    patch.object(server, "make_server", return_value=Mock(server_port=8080)), \
                    patch.object(server.signal, "signal"), patch("builtins.print"):
                server.main()
            constructor.assert_called_once_with(config, "c"*64, project="fly-training-emulator", database="(default)",
                                                run_id="fixture-run", initialize=False, lease_timeout_seconds=180)
            constructor.return_value.close.assert_called_once()

    def test_deployment_package_contains_transition_and_storage_modules(self):
        package = load_file("sequence_cloudrun_package", ROOT / "deploy/cloudrun/package.py")
        self.assertTrue({"sequential_training.py", "firestore_sequential.py"} <= set(package.SERVER_FILES))

    def test_scheduling_dry_run_has_sixteen_jobs_without_cloud_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"; path.write_text(json.dumps(fixture()))
            output = io.StringIO()
            with patch.object(module.firestore, "Client", side_effect=AssertionError("No cloud client")), \
                    patch("sys.stdout", output):
                module.main(["--config", str(path), "--run-id", "fixture", "--dry-run",
                             "--search-pairs", "8", "--trial-timeout-seconds", "1200"])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["initialJobs"], 16)
            self.assertEqual(plan["maximumBatchJobs"], 16)
            self.assertLess(plan["initialTransactionBytes"], plan["transactionBudgetBytes"])


@unittest.skipUnless(module and EMULATOR, "Requires the loopback Firestore emulator and requirements-cloudrun.txt")
class FirestoreSequentialTests(SchedulingChecks, unittest.TestCase):
    identity = SequenceFixture.identity
    credentials = SequenceFixture.credentials
    lease = SequenceFixture.lease
    result = SequenceFixture.result
    finish_generation = SequenceFixture.finish_generation
    error = SequenceFixture.error

    def setUp(self):
        if EMULATOR.rsplit(":", 1)[0] not in ("127.0.0.1", "localhost", "[::1]"):
            self.fail("Tests refuse a non-loopback Firestore endpoint")
        self.config = fixture()
        self.now = 1000.
        self.clients = []
        self.client = self.new_client()
        self.run_id = "seq-test-" + uuid.uuid4().hex
        self.coordinator = self.new_coordinator(initialize=True)

    def tearDown(self):
        for client in self.clients:
            client.close()

    def new_client(self):
        client = firestore.Client(project="fly-training-emulator")
        self.clients.append(client)
        return client

    def new_coordinator(self, **kwargs):
        return module.FirestoreSequentialCoordinator(self.config, run_id=self.run_id,
            client=kwargs.pop("client", self.client), clock=lambda: self.now, **kwargs)

    def reset(self, config):
        self.config = config
        self.run_id = "seq-test-" + uuid.uuid4().hex
        self.coordinator = self.new_coordinator(initialize=True)

    def test_explicit_initialization_and_metadata_identity_are_mandatory(self):
        unknown = "seq-unknown-" + uuid.uuid4().hex
        self.error("run_not_ready", lambda: module.FirestoreSequentialCoordinator(
            self.config, run_id=unknown, client=self.client))
        reference = self.client.collection(module.COLLECTION).document(unknown)
        self.assertFalse(reference.get().exists)
        reference.collection("jobs").document("orphan").set({"state": "completed"})
        with self.assertRaisesRegex(ValueError, "orphaned"):
            module.FirestoreSequentialCoordinator(self.config, run_id=unknown, client=self.client, initialize=True)
        original = self.coordinator.run.get().to_dict()
        self.new_coordinator(initialize=True)
        self.assertEqual(original, self.coordinator.run.get().to_dict())
        for update in ({"storageSchema": 99}, {"configHash": "other"}, {"config_json": "{}"}):
            with self.subTest(update=update):
                self.coordinator.run.update(update)
                with self.assertRaises(ValueError):
                    self.new_coordinator(initialize=True)
                self.coordinator.run.set(original)
        self.coordinator.run.update({"ready": False})
        self.error("run_not_ready", lambda: self.new_coordinator(initialize=True))

    def test_scheduling_migration_preserves_documents_and_restart_is_read_only(self):
        completed = self.lease(); self.coordinator.result(self.result(completed))
        active = self.lease("slow")
        before = {s.id:s.to_dict() for s in self.coordinator.jobs.stream()}
        identity = self.coordinator.run.get().to_dict()
        self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        after = {s.id:s.to_dict() for s in self.coordinator.jobs.stream()}
        self.assertEqual(before[completed["jobId"]], after[completed["jobId"]])
        for key, row in before.items():
            for field in module.JOB_COLUMNS:
                if field != "payload":self.assertEqual(row[field], after[key][field])
            original = json.loads(row["payload"]); migrated = json.loads(after[key]["payload"])
            for field, value in original.items():self.assertEqual(migrated[field], value)
        metadata = self.coordinator.run.get().to_dict()
        for key in ("configHash", "modelFingerprint", "config_json", "recent_trials_json", "contributorsCount"):
            self.assertEqual(metadata[key], identity[key])
        restarted = self.new_coordinator(client=self.new_client())
        self.assertEqual(restarted.status()["totalJobs"], 16)
        self.assertEqual(metadata, self.coordinator.run.get().to_dict())
        self.assertEqual(self.lease("slow")["leaseToken"], active["leaseToken"])

    def test_concurrent_migration_is_idempotent_and_cannot_extend_grace(self):
        active = self.lease("slow")
        other = self.new_coordinator(client=self.new_client())
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda c:c.set_scheduling(search_pairs=8, trial_timeout_seconds=1200),
                                        (self.coordinator, other)))
        self.assertEqual(sum(r["addedJobs"] for r in results), 12)
        self.assertEqual(sum(r["migratedLeases"] for r in results), 1)
        self.assertEqual(len(list(self.coordinator.jobs.stream())), 16)
        job = self.lease("slow")
        self.assertEqual(job["assignmentDeadlineAt"], self.now+1200)
        self.assertEqual(job["leaseToken"], active["leaseToken"])
        self.now += 3
        other.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        self.assertEqual(self.lease("slow")["assignmentDeadlineAt"], job["assignmentDeadlineAt"])
        self.assertEqual(self.coordinator.status()["schedulingRevision"], 1)

    def test_migration_racing_last_result_preserves_either_serial_order(self):
        for _ in range(3):
            job = self.lease(); self.coordinator.result(self.result(job))
        last = self.lease(); payload = self.result(last)
        other = self.new_coordinator(client=self.new_client())
        with ThreadPoolExecutor(max_workers=2) as executor:
            migration = executor.submit(other.set_scheduling, search_pairs=8, trial_timeout_seconds=1200)
            completion = executor.submit(self.coordinator.result, payload)
            migration.result(); self.assertTrue(completion.result()["accepted"])
        status = self.coordinator.status()
        self.assertEqual(status["acceptedResults"], 4)
        self.assertEqual(status["checkpoint"]["parameters"], [p["initial"] for p in self.config["parameters"]])
        self.assertEqual(status["scheduling"]["searchPairs"], 8)
        if status["sequence"]["role"] == "search":
            self.assertEqual(status["jobs"], dict(pending=12, leased=0, completed=4))
        else:
            self.assertEqual(status["sequence"]["role"], "compare")
            self.assertEqual(status["jobs"], dict(pending=4, leased=0, completed=0))
        self.assertTrue(other.result(payload)["duplicate"])

    def test_corrupt_operational_batch_count_is_rejected(self):
        self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        metadata = self.coordinator.run.get().to_dict()
        for value in (True, "8", 7, None):
            corrupt = copy.deepcopy(metadata); state = json.loads(corrupt["state_json"])
            if value is None:state.pop("currentSearchPairs")
            else:state["currentSearchPairs"] = value
            corrupt["state_json"] = canonical(state)
            self.coordinator.run.set(corrupt)
            self.error("invalid_history", self.coordinator.status)
        self.coordinator.run.set(metadata)

    def test_sqlite_parity_search_comparison_gate_and_historical_retry(self):
        local = SequentialTrainingCoordinator(":memory:", self.config, clock=lambda: self.now)
        self.addCleanup(local.close)
        historical = None
        self.assertEqual(self.coordinator.status(), local.status())
        while local.status()["sequence"]["status"] == "running":
            phase_status = local.status()
            for index in range(phase_status["jobs"]["pending"]):
                self.now += .1
                job = self.lease()
                local_job = local.lease(self.identity())["job"]
                self.assertEqual({k: v for k, v in job.items() if k != "leaseToken"},
                                 {k: v for k, v in local_job.items() if k != "leaseToken"})
                score = 0. if "-baseline-" in job["jobId"] else 2.
                payload = self.result(job, score=score, success=True)
                local_payload = {**payload, "leaseToken": local_job["leaseToken"]}
                self.assertEqual(self.coordinator.result(payload), local.result(local_payload))
                historical = historical or payload
                self.assertEqual(self.coordinator.status(), local.status())
        status = self.coordinator.status()
        self.assertEqual(status["sequence"]["status"], "complete")
        self.assertGreater(status["checkpoint"]["changedParameters"], 0)
        self.assertEqual(len(list(self.coordinator.checkpoints.stream())), 1)
        checkpoint = next(self.coordinator.checkpoints.stream()).to_dict()
        self.assertEqual(json.loads(checkpoint["checkpoint_json"])["parameters"], self.coordinator.checkpoint()["parameters"])
        self.assertEqual(len(list(self.coordinator.jobs.stream())), status["acceptedResults"])
        before = self.coordinator.run.get().to_dict()
        restarted = self.new_coordinator(client=self.new_client())
        self.assertEqual(restarted.status(), status)
        self.assertTrue(restarted.result(historical)["duplicate"])
        self.assertEqual(before, self.coordinator.run.get().to_dict())
        self.error("result_conflict", lambda: restarted.result({**historical, "objective": 1.}))

    def test_all_prior_stages_are_kept_in_gate_and_checkpoint_history(self):
        self.reset(fixture([phase("legs"), phase("recovery-sensors", [5], stage="recovery")], pairs=1))
        for _ in range(2):
            self.finish_generation("search")
            self.finish_generation("compare")
            if self.coordinator.status()["sequence"]["phaseIndex"] == 1:
                rows = list(self.coordinator.jobs.where(filter=module.firestore.FieldFilter("generation", "==", 5)).stream())
                self.assertEqual({json.loads(row.to_dict()["payload"])["stage"] for row in rows}, {"flight", "recovery"})
            self.finish_generation("gate")
        self.assertEqual(self.coordinator.status()["sequence"]["status"], "complete")
        self.assertEqual(len(list(self.coordinator.checkpoints.stream())), 2)

    def test_improvement_before_task_success_updates_center_and_survives_restart(self):
        self.reset(fixture([phase("legs", rounds=2)], pairs=1))
        initial = self.coordinator.checkpoint()["parameters"]
        self.finish_generation("search", success=False)
        self.finish_generation("compare", success=False)
        self.finish_generation("gate", success=False)
        status = self.coordinator.status()
        checkpoint = status["checkpoint"]
        self.assertEqual(status["sequence"]["status"], "running")
        self.assertEqual((status["sequence"]["phaseIndex"], status["sequence"]["round"]), (0, 1))
        self.assertEqual(checkpoint["completedPhases"], [])
        self.assertNotEqual(checkpoint["parameters"][:5], initial[:5])
        self.assertEqual(checkpoint["parameters"][5:], initial[5:])
        self.assertEqual(len(list(self.coordinator.checkpoints.stream())), 1)
        restarted = self.new_coordinator(client=self.new_client())
        self.assertEqual(restarted.checkpoint(), checkpoint)
        self.assertEqual(restarted.status(), status)
        state = json.loads(restarted.run.get().to_dict()["state_json"])
        self.assertEqual(state["center"], checkpoint["parameters"])
        self.assertEqual(len(restarted.run.get().to_dict()["currentJobIds"]), 2)

    def test_simultaneous_instances_have_unique_leases_and_exact_result_counts(self):
        instances = [self.new_coordinator(client=self.new_client()) for _ in range(4)]
        def lease_one(i):
            return instances[i].lease(self.identity("owner-" + str(i)))["job"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = list(pool.map(lease_one, range(4)))
        self.assertEqual(len({job["jobId"] for job in jobs}), 4)
        def result_one(i):
            return instances[i].result(self.result(jobs[i], "owner-"+str(i), score=float(i)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            accepted = list(pool.map(result_one, range(4)))
        self.assertTrue(all(result["accepted"] for result in accepted))
        status = self.coordinator.status()
        self.assertEqual((status["acceptedResults"], status["contributors"], status["generation"]), (4, 4, 1))
        self.assertEqual(len(status["recentTrials"]), 4)
        self.assertEqual(status["totalJobs"], 4)
        self.assertEqual(len(list(self.coordinator.jobs.stream())), 8)

    def test_same_contributor_retries_and_concurrent_duplicate_results_are_idempotent(self):
        instances = [self.new_coordinator(client=self.new_client()) for _ in range(3)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = list(pool.map(lambda instance: instance.lease(self.identity())["job"], instances))
        self.assertEqual(len({job["jobId"] for job in jobs}), 1)
        self.assertEqual(len({job["leaseToken"] for job in jobs}), 1)
        payload = self.result(jobs[0])
        with ThreadPoolExecutor(max_workers=3) as pool:
            accepted = list(pool.map(lambda instance: instance.result(payload), instances))
        self.assertEqual(sum(not result["duplicate"] for result in accepted), 1)
        self.assertEqual(self.coordinator.status()["acceptedResults"], 1)
        self.assertEqual(self.coordinator.status()["contributors"], 1)

    def test_operational_expiry_heartbeat_release_and_reassignment(self):
        self.coordinator = self.new_coordinator(lease_timeout_seconds=3)
        original = copy.deepcopy(self.config)
        job = self.lease()
        self.assertEqual(job["leaseExpiresAt"], 1003.)
        self.now = 1002.
        self.assertEqual(self.coordinator.heartbeat(self.credentials(job))["leaseExpiresAt"], 1005.)
        self.assertTrue(self.coordinator.release(self.credentials(job))["released"])
        self.assertTrue(self.coordinator.release(self.credentials(job))["duplicate"])
        replacement = self.lease("bob")
        self.assertEqual(job["jobId"], replacement["jobId"])
        self.error("stale_lease", lambda: self.coordinator.result(self.result(job)))
        self.now = 1006.
        self.error("lease_expired", lambda: self.coordinator.heartbeat(self.credentials(replacement, "bob")))
        renewed = self.lease()
        self.assertEqual(renewed["jobId"], job["jobId"])
        self.assertNotEqual(renewed["leaseToken"], replacement["leaseToken"])
        self.assertEqual(self.coordinator.status()["leaseSeconds"], 3.)
        self.assertEqual(self.config, original)

    def test_failed_result_and_storage_budget_roll_back_all_durable_state(self):
        job = self.lease()
        before = self.coordinator.run.get().to_dict()
        before_job = self.coordinator.jobs.document(job["jobId"]).get().to_dict()
        payload = self.result(job)
        bad = copy.deepcopy(payload)
        bad["provenance"]["seed"] += 1
        self.error("provenance_mismatch", lambda: self.coordinator.result(bad))
        with patch.object(module, "MAX_TRANSACTION_JSON_BYTES", 1000):
            self.error("body_too_large", lambda: self.coordinator.result(payload))
        self.assertEqual(before, self.coordinator.run.get().to_dict())
        self.assertEqual(before_job, self.coordinator.jobs.document(job["jobId"]).get().to_dict())
        self.assertEqual(list(self.coordinator.contributors.stream()), [])
        self.assertFalse(self.coordinator.result(payload)["duplicate"])

    def test_fit_candidates_are_only_proposals_until_comparison_and_gate_pass(self):
        self.reset(fixture([phase("fit", list(range(24, 696)), kind="decoder-fit")]))
        job = self.lease()
        self.assertEqual(job["mode"], "decoder-fit")
        self.assertEqual([trial["split"] for trial in job["calibrationTrials"]], ["train", "train", "validation"])
        payload = self.result(job)
        payload["calibration"] = dict(passed=True, trainingTrials=2, validationTrials=1,
            teacherUsedForEvaluation=False, teacherOnlyDuringDemonstrations=True, autonomousEvaluationPerformed=False)
        payload["candidateParameters"] = list(job["parameters"])
        payload["candidateParameters"][24] = .1
        bad = copy.deepcopy(payload)
        bad["candidateParameters"][0] = .1
        self.error("invalid_calibration", lambda: self.coordinator.result(bad))
        self.coordinator.result(payload)
        self.assertEqual(self.coordinator.status()["sequence"]["role"], "compare")
        self.assertEqual(self.coordinator.checkpoint()["parameters"][24], 0.)
        self.finish_generation("compare")
        self.assertEqual(self.coordinator.checkpoint()["parameters"][24], 0.)
        self.finish_generation("gate")
        self.assertEqual(self.coordinator.checkpoint()["parameters"][24], .1)

    def test_failed_fit_attempt_count_and_round_limit_persist_on_restart(self):
        self.reset(fixture([phase("fit", list(range(24, 696)), kind="decoder-fit", rounds=1)]))
        job = self.lease()
        payload = self.result(job)
        payload["calibration"] = dict(passed=False, trainingTrials=1, validationTrials=0,
            teacherUsedForEvaluation=False, teacherOnlyDuringDemonstrations=True, autonomousEvaluationPerformed=False)
        self.coordinator.result(payload)
        restarted = self.new_coordinator()
        self.assertEqual(restarted.status()["sequence"]["status"], "needs-review")
        self.assertIsNone(restarted.lease(self.identity())["job"])
        self.assertTrue(restarted.result(payload)["duplicate"])
        self.assertEqual(list(self.coordinator.checkpoints.stream()), [])

    def test_status_reads_only_current_batch_and_metadata_cached_recent_trials(self):
        self.finish_generation("search")
        previous_recent = self.coordinator.status()["recentTrials"]
        read_ids = []
        original = self.client.get_all
        def counted(references, **kwargs):
            read_ids.extend(reference.id for reference in references)
            return original(references, **kwargs)
        # No history collection queries are permitted on this API path.
        with patch.object(self.client, "get_all", side_effect=counted), \
                patch.object(type(self.coordinator.jobs), "stream", side_effect=AssertionError("History scan")):
            status = self.coordinator.status()
            checkpoint = self.coordinator.checkpoint()
        self.assertEqual(set(read_ids), set(self.coordinator.run.get().to_dict()["currentJobIds"]))
        self.assertEqual(len(read_ids), status["totalJobs"])
        self.assertEqual(status["recentTrials"], previous_recent)
        self.assertEqual(status["checkpoint"], checkpoint)

    def test_missing_active_job_or_incomplete_batch_is_rejected(self):
        metadata = self.coordinator.run.get().to_dict()
        self.coordinator.run.update({"currentJobIds": metadata["currentJobIds"][:-1]})
        self.error("invalid_history", self.coordinator.status)
        self.coordinator.run.set(metadata)
        self.coordinator.jobs.document(metadata["currentJobIds"][0]).delete()
        self.error("invalid_history", lambda: self.new_coordinator())

    def test_json_matrices_are_preserved_as_strings_with_legacy_namespace_isolated(self):
        legacy = self.client.collection("training_runs").document(self.run_id)
        legacy.set({"legacy_marker": True})
        job = self.lease()
        payload = self.result(job)
        payload["metrics"]["matrix"] = [[1, 2], [3, 4]]
        self.coordinator.result(payload)
        row = self.coordinator.jobs.document(job["jobId"]).get().to_dict()
        self.assertIsInstance(row["result"], str)
        self.assertEqual(json.loads(row["result"])["metrics"]["matrix"], [[1, 2], [3, 4]])
        self.assertIsInstance(self.coordinator.run.get().to_dict()["config_json"], str)
        self.assertEqual(legacy.get().to_dict(), {"legacy_marker": True})


if __name__ == "__main__":
    unittest.main()
