"""Real local HTTP + SQLite tests; no simulated biological success claim."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("training_coordinator", ROOT / "scripts/training_coordinator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(pairs=2):
    return {
        "schemaVersion": 1, "environmentVersion": "banc-flybody-flight-objective-v3",
        "modelFingerprint": hashlib.sha256(("fixture.wasm:" + "b" * 64 + "\n").encode()).hexdigest(),
        "assets": {"fixture.wasm": "b" * 64},
        "algorithm": "antithetic-evolution-strategies", "dtMs": .5, "bodyBlockMs": 2,
        "parameters": [{"name": "gain_log", "min": -2, "max": 2, "initial": 0}],
        "optimizer": {"populationPairs": pairs, "sigma": .25, "learningRate": .035, "maximumUpdate": .15, "seed": 888},
        "objective": {"min": -10, "max": 10, "direction": "maximize"},
        "stage": "landing", "durationSeconds": 8,
        "stages": [{"id": "takeoff", "durationSeconds": 3},
                   {"id": "flight", "durationSeconds": 5},
                   {"id": "landing", "durationSeconds": 8}],
        "contribution": {"leaseSeconds": 10, "maxRequestBytes": 1024},
    }


class CoordinatorFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "state.sqlite3"
        self.now = 1000.
        self.config = fixture()
        self.coordinator = module.TrainingCoordinator(self.database, self.config, clock=lambda: self.now)

    def tearDown(self):
        self.coordinator.close()
        self.temp.cleanup()

    def identity(self, contributor="alice"):
        return {"contributorId": contributor, "modelFingerprint": self.coordinator.model_fingerprint,
                "configHash": self.coordinator.config_hash}

    def lease(self, contributor="alice"):
        return self.coordinator.lease(self.identity(contributor))["job"]

    def result_body(self, job, contributor="alice", objective=0):
        provenance = {key: job[key] for key in ("modelFingerprint", "configHash", "parametersHash", "seed", "stage", "durationSeconds", "sign", "pairId", "generation")}
        provenance.update(environmentVersion=self.config["environmentVersion"], backend="wasm", bodyBackend="mujoco-wasm", dtMs=.5, bodyBlockMs=2)
        return {**self.identity(contributor), "jobId": job["jobId"], "leaseToken": job["leaseToken"],
                "objective": objective, "parameters": job["parameters"], "metrics": {"uprightFraction": .25}, "provenance": provenance}

    def assert_api_error(self, code, call):
        with self.assertRaises(module.APIError) as context:
            call()
        self.assertEqual(context.exception.code, code)


class CoordinatorTests(CoordinatorFixture):

    def test_initial_checkpoint_preserves_log_coordinates_and_is_unverified(self):
        value = self.coordinator.checkpoint()
        self.assertEqual(value["parameters"], [0.])
        self.assertEqual(value["parameterNames"], ["gain_log"])
        self.assertEqual(value["algorithm"], "antithetic-evolution-strategies")
        self.assertEqual(value["status"], "unverified")
        self.assertFalse(value["heldOutValidated"])
        self.assertFalse(value["biologicalSuccessValidated"])

    def test_paired_jobs_share_seed_and_antithetic_parameters(self):
        positive, negative = self.lease("alice"), self.lease("bob")
        self.assertEqual(positive["pairId"], negative["pairId"])
        self.assertEqual(positive["seed"], negative["seed"])
        self.assertEqual((positive["sign"], negative["sign"]), (1, -1))
        self.assertAlmostEqual(positive["parameters"][0], -negative["parameters"][0])
        self.assertEqual(positive["stage"], self.config["stage"])
        self.assertEqual(positive["durationSeconds"], self.config["durationSeconds"])

    def test_job_generation_is_deterministic_across_databases(self):
        other = module.TrainingCoordinator(Path(self.temp.name) / "other.sqlite3", self.config, clock=lambda: self.now)
        try:
            first, second = self.lease(), other.lease(self.identity())["job"]
            for key in ("jobId", "pairId", "sign", "seed", "parameters", "parametersHash"):
                self.assertEqual(first[key], second[key])
            self.assertNotEqual(first["leaseToken"], second["leaseToken"])
        finally:
            other.close()

    def test_lease_retry_is_idempotent_and_parallel_contributors_get_unique_jobs(self):
        first = self.lease()
        self.assertEqual(first, self.lease())
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(self.lease, [f"worker-{i}" for i in range(8)]))
        assigned = [job for job in jobs if job]
        self.assertEqual(len(assigned), 3)
        self.assertEqual(len({job["jobId"] for job in assigned}), 3)
        self.assertEqual(self.coordinator.status()["jobs"], {"pending": 0, "leased": 4, "completed": 0})

    def test_expiry_reassigns_new_token_and_rejects_old_contributor(self):
        old = self.lease()
        self.now += 10
        self.assert_api_error("lease_expired", lambda: self.coordinator.result(self.result_body(old)))
        new = self.lease("bob")
        self.assertEqual(old["jobId"], new["jobId"])
        self.assertNotEqual(old["leaseToken"], new["leaseToken"])
        self.assert_api_error("stale_lease", lambda: self.coordinator.result(self.result_body(old)))
        self.assertTrue(self.coordinator.result(self.result_body(new, "bob"))["accepted"])

    def test_heartbeat_extends_only_live_lease_and_cannot_resurrect(self):
        job = self.lease()
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        self.now += 9
        renewed = self.coordinator.heartbeat(identity)
        self.assertEqual(renewed["leaseExpiresAt"], 1019)
        self.now = 1019
        self.assert_api_error("lease_expired", lambda: self.coordinator.heartbeat(identity))

    def test_forged_token_and_contributor_and_unknown_job_are_rejected(self):
        job = self.lease()
        result = self.result_body(job)
        self.assert_api_error("stale_lease", lambda: self.coordinator.result({**result, "leaseToken": "z" * 43}))
        self.assert_api_error("stale_lease", lambda: self.coordinator.result({**result, "contributorId": "intruder"}))
        self.assert_api_error("unknown_job", lambda: self.coordinator.result({**result, "jobId": "missing"}))
        self.assertEqual(self.coordinator.status()["acceptedResults"], 0)

    def test_model_config_and_shared_stage_are_fixed(self):
        identity = self.identity()
        self.assert_api_error("incompatible_model", lambda: self.coordinator.lease({**identity, "modelFingerprint": "c" * 64}))
        self.assert_api_error("incompatible_config", lambda: self.coordinator.lease({**identity, "configHash": "d" * 64}))
        self.assert_api_error("incompatible_stage", lambda: self.coordinator.lease({**identity, "stage": "flight"}))
        self.assert_api_error("incompatible_stage", lambda: self.coordinator.lease({**identity, "durationSeconds": -1}))
        self.assert_api_error("invalid_request", lambda: self.coordinator.lease({**identity, "parameters": [1]}))

    def test_mismatched_provenance_and_parameter_vector_are_rejected(self):
        job = self.lease()
        result = self.result_body(job)
        for key, value in (("seed", job["seed"]+1), ("dtMs", 2), ("bodyBlockMs", 10), ("stage", "flight"),
                           ("modelFingerprint", "wrong"), ("configHash", "wrong"), ("parametersHash", "wrong"),
                           ("generation", 2), ("sign", -job["sign"]), ("pairId", "wrong"),
                           ("bodyBackend", "fake"), ("backend", "fake"), ("environmentVersion", "fake")):
            with self.subTest(key=key):
                bad = copy.deepcopy(result)
                bad["provenance"][key] = value
                self.assert_api_error("provenance_mismatch", lambda: self.coordinator.result(bad))
        self.assert_api_error("provenance_mismatch", lambda: self.coordinator.result({**result, "parameters": [999]}))
        self.assertEqual(self.coordinator.status()["acceptedResults"], 0)

    def test_bounded_negative_objectives_accepted_but_nonfinite_or_bool_rejected(self):
        job = self.lease()
        for value in (float("nan"), float("inf"), -float("inf"), -11, 11, True, "1", 10**1000):
            with self.subTest(value=str(value)[:30]):
                self.assert_api_error("invalid_value", lambda: self.coordinator.result(self.result_body(job, objective=value)))
        self.assertTrue(self.coordinator.result(self.result_body(job, objective=-3))["accepted"])

    def test_nonfinite_nested_metrics_rejected_without_state_change(self):
        result = self.result_body(self.lease())
        result["metrics"] = {"nested": [1, float("nan")]}
        self.assert_api_error("invalid_value", lambda: self.coordinator.result(result))
        self.assertEqual(self.coordinator.status()["acceptedResults"], 0)

    def test_client_environment_version_alias_and_conflict(self):
        result = self.result_body(self.lease())
        result["provenance"]["envVersion"] = result["provenance"].pop("environmentVersion")
        bad = copy.deepcopy(result)
        bad["provenance"]["environmentVersion"] = "different"
        self.assert_api_error("provenance_mismatch", lambda: self.coordinator.result(bad))
        self.assertTrue(self.coordinator.result(result)["accepted"])

    def test_result_retry_exactly_once_even_after_expiry(self):
        result = self.result_body(self.lease(), objective=-1)
        self.assertFalse(self.coordinator.result(result)["duplicate"])
        self.now += 100
        self.assertTrue(self.coordinator.result(copy.deepcopy(result))["duplicate"])
        self.assert_api_error("result_conflict", lambda: self.coordinator.result({**result, "objective": 2}))
        self.assertEqual(self.coordinator.status()["acceptedResults"], 1)

    def test_release_is_idempotent_then_old_token_becomes_stale(self):
        job = self.lease()
        payload = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        self.assertFalse(self.coordinator.release(payload)["duplicate"])
        self.assertTrue(self.coordinator.release(payload)["duplicate"])
        new = self.lease("bob")
        self.assertEqual(job["jobId"], new["jobId"])
        self.assert_api_error("stale_lease", lambda: self.coordinator.release(payload))

    def test_complete_generation_updates_shared_log_vector_without_promotion(self):
        initial = self.coordinator.checkpoint()["parameters"]
        bodies = []
        for index in range(4):
            user = f"worker-{index}"
            job = self.lease(user)
            # A synthetic algebraic objective tests the optimizer only.
            body = self.result_body(job, user, objective=job["parameters"][0])
            body["metrics"]["claimedBiologicalSuccess"] = True
            bodies.append(body)
            self.coordinator.result(body)
            if index < 3:
                self.assertEqual(self.coordinator.checkpoint()["generation"], 0)
        checkpoint = self.coordinator.checkpoint()
        self.assertEqual(checkpoint["generation"], 1)
        self.assertGreater(checkpoint["parameters"][0], initial[0])
        self.assertLessEqual(checkpoint["parameters"][0], .15)
        self.assertEqual(checkpoint["stage"], self.config["stage"])
        self.assertEqual(checkpoint["status"], "unverified")
        self.assertFalse(checkpoint["heldOutValidated"])
        self.assertFalse(self.coordinator.status()["automaticCurriculumPromotion"])
        self.assertEqual(self.coordinator.status()["contributors"], 4)
        self.assertTrue(self.coordinator.result(bodies[-1])["duplicate"])
        self.assertEqual(self.coordinator.checkpoint()["generation"], 1)

    def test_generation_update_matches_raw_logspace_formula_and_maximum_update(self):
        jobs = []
        for index in range(4):
            user = f"worker-{index}"
            job = self.lease(user)
            jobs.append(job)
            self.coordinator.result(self.result_body(job, user, objective=10*job["sign"]))
        noise_sum = sum((jobs[index]["parameters"][0]-jobs[index+1]["parameters"][0])/(2*.25) for index in (0, 2))
        expected = max(-.15, min(.15, .035/(2*2*.25)*20*noise_sum))
        self.assertAlmostEqual(self.coordinator.checkpoint()["parameters"][0], expected)

    def test_restart_preserves_lease_result_idempotency_and_candidate(self):
        job = self.lease()
        pending = self.lease("bob")
        result = self.result_body(job, objective=.3)
        self.coordinator.result(result)
        self.coordinator.close()
        self.coordinator = module.TrainingCoordinator(self.database, self.config, clock=lambda: self.now)
        self.assertEqual(self.lease("bob"), pending)
        self.assertTrue(self.coordinator.result(result)["duplicate"])
        for user in ("bob", "carol", "dave"):
            active = self.lease(user)
            self.coordinator.result(self.result_body(active, user, objective=0))
        checkpoint = self.coordinator.checkpoint()
        self.coordinator.close()
        self.coordinator = module.TrainingCoordinator(self.database, self.config, clock=lambda: self.now)
        self.assertEqual(self.coordinator.checkpoint(), checkpoint)

    def test_changed_config_cannot_resume_old_database_and_pending_manifest_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["durationSeconds"] = 2
        with self.assertRaisesRegex(ValueError, "different configHash"):
            module.TrainingCoordinator(self.database, changed)
        pending = copy.deepcopy(self.config)
        pending["modelFingerprint"] = "pending-build-manifest"
        with self.assertRaisesRegex(ValueError, "pending"):
            module.TrainingCoordinator(Path(self.temp.name) / "pending.sqlite3", pending)

    def test_changed_reward_contract_does_not_mix_existing_scores(self):
        job = self.lease()
        self.coordinator.result(self.result_body(job, objective=.3))
        before = self.coordinator.status()
        changed = copy.deepcopy(self.config)
        changed["environmentVersion"] = "banc-flybody-flight-interface-v2"
        with self.assertRaisesRegex(ValueError, "different configHash"):
            module.TrainingCoordinator(self.database, changed)
        self.assertEqual(self.coordinator.status(), before)

    def test_repository_default_assigns_the_entire_flight_cycle(self):
        config, config_hash = module.read_config(ROOT / "web/training/config.json")
        self.assertEqual(config["environmentVersion"], "banc-flybody-flight-interpreter27-v6")
        self.assertEqual(len(config["parameters"]), 27)
        self.assertTrue(all(p["name"].startswith("flight_") for p in config["parameters"]))
        self.assertEqual(config["stage"], "landing")
        self.assertEqual(config["durationSeconds"], 8)
        coordinator = module.TrainingCoordinator(Path(self.temp.name) / "current.sqlite3", config, config_hash)
        try:
            identity = {"contributorId": "current-fixture", "configHash": config_hash,
                        "modelFingerprint": config["modelFingerprint"]}
            job = coordinator.lease(identity)["job"]
            self.assertEqual(job["stage"], "landing")
            self.assertEqual(job["durationSeconds"], 8)
            self.assertFalse(coordinator.status()["automaticCurriculumPromotion"])
        finally:
            coordinator.close()

    def test_config_hash_is_exact_file_bytes(self):
        path = Path(self.temp.name) / "config.json"
        raw = json.dumps(self.config, indent=2).encode() + b"\n"
        path.write_bytes(raw)
        value, digest = module.read_config(path)
        self.assertEqual(value, self.config)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_final_fingerprint_must_match_pinned_asset_manifest(self):
        bad = copy.deepcopy(self.config)
        bad["modelFingerprint"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "pinned asset manifest"):
            module.TrainingCoordinator(Path(self.temp.name) / "bad.sqlite3", bad)

    @unittest.skipUnless(shutil.which("node"), "Node is needed only for the existing browser-optimizer contract check")
    def test_actual_js_optimizer_and_checkpoint_contract_agree(self):
        for index in range(4):
            user = f"worker-{index}"
            job = self.lease(user)
            self.coordinator.result(self.result_body(job, user, objective=job["parameters"][0]))
        records = self.coordinator.db.execute("SELECT * FROM jobs WHERE generation=0 ORDER BY pair_id,sign DESC").fetchall()
        pairs = []
        for index in (0, 2):
            plus, minus = records[index:index+2]
            pairs.append({"noise": json.loads(plus["noise"]), "jobs": [
                {"sign": 1, "parameters": json.loads(plus["parameters"])},
                {"sign": -1, "parameters": json.loads(minus["parameters"])}], "results": {
                "1": {"return": plus["score"], "success": False, "steps": 0, "simSeconds": 0},
                "-1": {"return": minus["score"], "success": False, "steps": 0, "simSeconds": 0}}})
        payload = {"config": self.config, "configHash": self.coordinator.config_hash,
                   "checkpoint": self.coordinator.checkpoint(), "round": {"baseline": [0.], "pairs": pairs}}
        code = """import {updateGeneration,readCheckpoint} from './web/training/optimizer.js';
let raw='';for await(const chunk of process.stdin)raw+=chunk;const x=JSON.parse(raw);
console.log(JSON.stringify({parameters:updateGeneration(x.round,x.config),
checkpoint:readCheckpoint(x.checkpoint,x.config,x.configHash)}));"""
        completed = subprocess.run(["node", "--input-type=module", "-e", code], cwd=ROOT,
                                   input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=10)
        actual = json.loads(completed.stdout)
        self.assertEqual(actual["parameters"], payload["checkpoint"]["parameters"])
        self.assertEqual(actual["checkpoint"]["parameters"], actual["parameters"])
        self.assertEqual(actual["checkpoint"]["status"], "unverified")


def preview_frame():
    return {"time": .6, "simSeconds": .1, "stage": "landing", "position": [0, 0, 3],
            "quaternion": [1, 0, 0, 0], "feet": [[0, 0, 1]]*6,
            "legs": [[[0, 0, 2], [0, 0, 1], [0, 0, 0]]]*6,
            "bowl": {"radiusCm": 50, "floor": {"baseCm": .15, "radialCoefficientPerCm": .037, "capRadiusCm": 6.5}},
            "phase": "warmup", "vision": False, "neuralMs": 600, "neuralSpikes": 1500}


class ObserverStatusTests(CoordinatorFixture):
    def test_recent_trials_are_bounded_ordered_and_contain_only_public_metrics(self):
        self.assertIsNone(self.coordinator.status()["lastParameterUpdate"])
        for i in range(124):
            self.now += 1
            job = self.lease()
            body = self.result_body(job, objective=0)
            body["metrics"].update(success=False, simSeconds=.1, wallSeconds=2,
                                   privateLabel="do not expose", nested={"token": "private"})
            self.coordinator.result(body)
        state = self.coordinator.status()
        self.assertEqual(len(state["recentTrials"]), 120)
        self.assertEqual([row["episode"] for row in state["recentTrials"]], list(range(5, 125)))
        self.assertEqual(state["recentWallSeconds"], 240)
        self.assertEqual(state["lastParameterUpdate"], {"generation": 31, "changedCount": 0,
                         "parameterCount": 1, "completedAt": self.now})
        self.assertEqual(set(state["recentTrials"][0]), {"episode", "generation", "stage", "return", "success", "simSeconds", "wallSeconds", "completedAt"})
        text = json.dumps(state["recentTrials"])
        for private in ("leaseToken", "contributor", "parameters", "private", "token"):
            self.assertNotIn(private, text)

    def test_last_update_counts_actual_center_change_and_ignores_candidate_noise(self):
        for i in range(4):
            job = self.lease()
            self.coordinator.result(self.result_body(job, objective=job["parameters"][0]))
        state = self.coordinator.status()
        self.assertNotEqual(state["checkpoint"]["parameters"], [0.])
        self.assertEqual(state["lastParameterUpdate"]["changedCount"], 1)
        self.lease()
        self.assertEqual(self.coordinator.status()["lastParameterUpdate"], state["lastParameterUpdate"])

    def test_preview_persists_sanitized_separately_and_becomes_stale_on_completion(self):
        job = self.lease()
        frame = preview_frame()
        frame["ignored"] = {"contributorId": "not public", "html": "<script>"}
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        checkpoint = self.coordinator.checkpoint()
        self.coordinator.heartbeat({**identity, "previewFrame": frame})
        state = self.coordinator.status()
        self.assertEqual(state["checkpoint"], checkpoint)
        self.assertEqual(state["acceptedResults"], 0)
        self.assertFalse(state["latestFrame"]["stale"])
        self.assertEqual(state["latestFrame"]["frame"], preview_frame())
        self.assertEqual(state["latestFrame"]["serverReceivedAt"], self.now)
        self.coordinator.close()
        self.coordinator = module.TrainingCoordinator(self.database, self.config, clock=lambda: self.now)
        self.assertEqual(self.coordinator.status()["latestFrame"], state["latestFrame"])
        self.coordinator.result(self.result_body(job))
        self.assertTrue(self.coordinator.status()["latestFrame"]["stale"])
        for i in range(3):
            self.coordinator.result(self.result_body(self.lease()))
        self.assertIsNone(self.coordinator.status()["latestFrame"])

    def test_invalid_preview_or_invalid_lease_cannot_write_or_extend_telemetry(self):
        job = self.lease()
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        variants = []
        for key, value in (("position", [1e7, 0, 0]), ("quaternion", [0, 0, 0, 0]),
                           ("phase", "<script>"), ("feet", [[0, 0, 0]]*7),
                           ("stage", "different"), ("wings", [None]),
                           ("ignored", ["x"*1000]*40)):
            frame = preview_frame(); frame[key] = value; variants.append(frame)
        self.now += 1
        for frame in variants:
            self.assert_api_error("invalid_preview", lambda: self.coordinator.heartbeat({**identity, "previewFrame": frame}))
        self.assertEqual(self.coordinator.db.execute("SELECT expires FROM jobs WHERE job_id=?", (job["jobId"],)).fetchone()[0], job["leaseExpiresAt"])
        self.assertIsNone(self.coordinator.status()["latestFrame"])
        self.assert_api_error("stale_lease", lambda: self.coordinator.heartbeat({**identity, "leaseToken": "x"*32, "previewFrame": preview_frame()}))
        self.now = job["leaseExpiresAt"]+1
        self.assert_api_error("lease_expired", lambda: self.coordinator.heartbeat({**identity, "previewFrame": preview_frame()}))
        self.assertIsNone(self.coordinator.status()["latestFrame"])


class BoundedOptimizerTests(unittest.TestCase):
    @staticmethod
    def bounded_rules():
        config = fixture()
        config["parameters"] = [
            {"name": "near_upper", "min": -1, "max": 1, "initial": .99},
            {"name": "near_lower", "min": -1, "max": 1, "initial": -.99},
            {"name": "frequency", "min": math.log(.8), "max": math.log(1.1), "initial": .09},
            {"name": "narrow", "min": -.001, "max": .001, "initial": 0},
        ]
        rules = module.CoordinatorRules()
        rules._configure(config)
        return rules

    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser/server optimizer parity")
    def test_bound_clipped_assigned_jobs_match_js_for_native_and_json_storage(self):
        rules = self.bounded_rules()
        old, jobs = rules._generation_records(0, rules.initial)
        # These are the actual deterministic assigned vectors. Establish that
        # both near-bound and narrow-range coordinates were really clipped.
        clipped = set()
        for job in jobs:
            job["score"] = -job["parameters"][0] + job["parameters"][1] - job["parameters"][2]
            for axis, parameter in enumerate(job["parameters"]):
                requested = rules.initial[axis] + job["sign"]*rules.sigma*job["noise"][axis]
                if parameter != requested:
                    clipped.add(axis)
        self.assertEqual(clipped, {0, 1, 2, 3})
        pairs = []
        for index in range(0, len(jobs), 2):
            plus, minus = jobs[index:index+2]
            pairs.append({"noise": plus["noise"], "jobs": [
                {"sign": job["sign"], "parameters": job["parameters"]} for job in (plus, minus)],
                "results": {str(job["sign"]): {"return": job["score"], "success": False,
                            "steps": 0, "simSeconds": 0} for job in (plus, minus)}})
        payload = {"config": rules.config, "round": {"baseline": rules.initial, "pairs": pairs}}
        code = """import {updateGeneration} from './web/training/optimizer.js';
let raw='';for await(const chunk of process.stdin)raw+=chunk;const x=JSON.parse(raw);
console.log(JSON.stringify(updateGeneration(x.round,x.config)));"""
        completed = subprocess.run(["node", "--input-type=module", "-e", code], cwd=ROOT,
                                   input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=10)
        expected = json.loads(completed.stdout)
        self.assertNotEqual(expected, rules.initial)
        self.assertEqual(rules._next_center(old, jobs), expected)
        # Firestore uses arrays; SQLite stores these same assigned vectors as JSON.
        json_jobs = [{**job, "parameters": json.dumps(job["parameters"]), "noise": json.dumps(job["noise"])}
                     for job in jobs]
        self.assertEqual(rules._next_center({**old, "center": json.dumps(old["center"])}, json_jobs), expected)

        # Ensure the fixture exposes the previous raw-noise bug rather than
        # accidentally matching because a final bound hides every difference.
        gradient = [0.] * len(rules.initial)
        for index in range(0, len(jobs), 2):
            plus, minus = jobs[index:index+2]
            for axis, noise in enumerate(plus["noise"]):
                gradient[axis] += (plus["score"]-minus["score"])*noise
        scale = rules.learning_rate/(2*rules.pairs*rules.sigma)
        old_result = [max(low, min(high, value+max(-rules.maximum_update, min(rules.maximum_update, scale*grad))))
                      for value, grad, (low, high) in zip(rules.initial, gradient, rules.bounds)]
        self.assertNotEqual(expected, old_result)

    def test_equal_rewards_cannot_move_a_bound_clipped_center(self):
        rules = self.bounded_rules()
        old, jobs = rules._generation_records(0, rules.initial)
        for job in jobs:
            job["score"] = -3.
        self.assertEqual(rules._next_center(old, jobs), rules.initial)

    def test_zero_realized_separation_cannot_move_even_with_different_rewards(self):
        rules = self.bounded_rules()
        old, jobs = rules._generation_records(0, rules.initial)
        for job in jobs:
            job["parameters"] = list(rules.initial)
            job["score"] = 10*job["sign"]
        self.assertEqual(rules._next_center(old, jobs), rules.initial)


class HTTPTests(CoordinatorFixture):
    def setUp(self):
        super().setUp()
        self.server = module.make_server(self.coordinator, port=0, allowed_origins=["http://localhost:7842"], max_body_bytes=4096)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        super().tearDown()

    def request(self, method, endpoint, value=None, origin="http://localhost:7842", raw=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        sent_headers = {"Content-Type": "application/json"}
        if origin is not None:
            sent_headers["Origin"] = origin
        if headers:
            sent_headers.update(headers)
        body = raw if raw is not None else json.dumps(value).encode() if value is not None else None
        try:
            connection.request(method, "/api/training/" + endpoint, body, sent_headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, dict(response.headers), json.loads(data) if data else None
        finally:
            connection.close()

    def test_http_status_lease_result_retry_checkpoint_roundtrip(self):
        status, headers, state = self.request("GET", "status")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:7842")
        self.assertEqual((state["completedJobs"], state["totalJobs"], state["checkpointStatus"]), (0, 4, "unverified"))
        status, _, leased = self.request("POST", "lease", self.identity())
        self.assertEqual(status, 200)
        result = self.result_body(leased["job"], objective=-2)
        self.assertEqual(self.request("POST", "result", result)[2]["duplicate"], False)
        self.assertEqual(self.request("POST", "result", result)[2]["duplicate"], True)
        self.assertEqual(self.request("GET", "checkpoint")[2]["status"], "unverified")

    def test_compact_status_drops_only_config(self):
        full = self.request("GET", "status")[2]
        compact = self.request("GET", "status?compact=1")[2]
        self.assertIn("config", full)
        self.assertEqual(compact, {key: value for key, value in full.items() if key != "config"})
        self.assertEqual(self.request("GET", "status?compact=0")[2], full)

    def test_http_checkpoint_download_has_safe_filename_and_no_cache(self):
        status, headers, checkpoint = self.request("GET", "checkpoint?filename=arbitrary.txt", origin=None)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="heaven-checkpoint-generation-0.json"')
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(checkpoint, self.coordinator.checkpoint())
        self.assertNotIn("Content-Disposition", self.request("GET", "status")[1])
        self.assertNotIn("Content-Disposition", self.request("GET", "checkpoint", origin="https://untrusted.example")[1])

    def test_http_checkpoint_download_refreshes_after_generation_completes(self):
        initial = self.request("GET", "checkpoint")[2]
        for _ in range(self.config["optimizer"]["populationPairs"] * 2):
            job = self.request("POST", "lease", self.identity())[2]["job"]
            body = self.result_body(job, objective=job["parameters"][0])
            self.assertTrue(self.request("POST", "result", body)[2]["accepted"])
        status, headers, checkpoint = self.request("GET", "checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="heaven-checkpoint-generation-1.json"')
        self.assertEqual(checkpoint["generation"], initial["generation"] + 1)
        self.assertGreater(checkpoint["parameters"][0], initial["parameters"][0])
        self.assertEqual(checkpoint, self.coordinator.checkpoint())
        self.assertEqual(checkpoint["status"], "unverified")
        self.assertFalse(checkpoint["heldOutValidated"])

    def test_http_origin_preflight_and_private_network_headers(self):
        status, headers, _ = self.request("OPTIONS", "lease", headers={"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Content-Type", "Access-Control-Request-Private-Network": "true"})
        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Private-Network"], "true")
        status, headers, _ = self.request("POST", "lease", self.identity(), origin="https://untrusted.example")
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request("GET", "status", origin="null")[0], 403)
        self.assertEqual(self.request("OPTIONS", "lease", headers={"Access-Control-Request-Method": "DELETE"})[0], 403)
        self.assertEqual(self.request("GET", "status", origin=None)[0], 200)

    def test_http_invalid_json_nonfinite_oversize_and_content_type(self):
        for raw in (b"{", b'{"objective":NaN}', b'{"objective":Infinity}'):
            self.assertEqual(self.request("POST", "result", raw=raw)[0], 400)
        self.assertEqual(self.request("POST", "result", raw=b"x"*4097)[0], 413)
        self.assertEqual(self.request("POST", "lease", self.identity(), headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "lease", raw=b"{}", headers={"Content-Length": "-1"})[0], 400)

    def test_http_heartbeat_release_and_stale_submission(self):
        job = self.request("POST", "lease", self.identity())[2]["job"]
        identity = {**self.identity(), "jobId": job["jobId"], "leaseToken": job["leaseToken"]}
        self.now += 5
        self.assertEqual(self.request("POST", "heartbeat", identity)[2]["leaseExpiresAt"], 1015)
        self.assertTrue(self.request("POST", "release", identity)[2]["released"])
        status, _, result = self.request("POST", "result", self.result_body(job))
        self.assertEqual((status, result["error"]), (410, "lease_expired"))


if __name__ == "__main__":
    unittest.main()
