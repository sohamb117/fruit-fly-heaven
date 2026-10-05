"""Synthetic protocol/SQLite fixtures only: no neural, physical or training evidence."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sequential_training import SequentialTrainingCoordinator, parse_sequence, CONTRACT, PROGRESS_ACCEPTANCE
from training_coordinator import APIError, canonical


def phase(name="legs", indices=None, kind="search", rounds=2, stage="flight"):
    return dict(id=name, kind=kind, stage=stage, parameterIndices=indices or list(range(5)),
                maxRounds=rounds, validationCount=2, minimumSuccesses=2,
                minimumMeanImprovement=.1)


def fixture(phases=None, pairs=2):
    assets = {"/banc-engine/dist/core.wasm": "e" * 64}
    sensory = [dict(name=f"sensory_{group}_{k}", min=-.4, max=.4, initial=0., searchScale=.5,
                    group=group) for group, count in (("legs", 5), ("antenna", 6),
                    ("haltere", 7), ("tegula", 4), ("vision", 2)) for k in range(count)]
    motor = [dict(name=f"decoder_{i}", min=-.25, max=.25, initial=0., searchScale=.1) for i in range(672)]
    return dict(schemaVersion=2, environmentVersion="sequence-synthetic-protocol-v1",
                modelFingerprint=hashlib.sha256("".join(f"{u}:{h}\n" for u, h in assets.items()).encode()).hexdigest(),
                assets=assets, algorithm="antithetic-evolution-strategies", dtMs=.5, bodyBlockMs=2,
                parameterContract=CONTRACT, freezeNeuralParameters=True, wingEventExcitation={},
                sensorimotorSequence=dict(schema=1, sensoryParameterNames=[p["name"] for p in sensory]),
                motorDecoderContract=dict(parameters=copy.deepcopy(motor)), parameters=sensory + motor,
                optimizer=dict(populationPairs=pairs, sigma=.2, learningRate=.03, maximumUpdate=.1, seed=888,
                               acceptance=dict(profile=1, proposal="best-search-job", seedCount=3,
                                               nativeExecution=dict(backend="wasm", moduleSha256="e" * 64))),
                objective=dict(min=-10, max=10, direction="maximize"), stage="flight", durationSeconds=5,
                stages=[dict(id="flight", durationSeconds=5), dict(id="recovery", durationSeconds=4),
                        dict(id="takeoff", durationSeconds=3), dict(id="landing", durationSeconds=8)],
                validation=dict(seeds=[190888], testSeeds=[1190888]),
                contribution=dict(leaseSeconds=10, maxRequestBytes=1024 * 1024),
                trainingSequence=dict(schema=1, calibrationMode="simulation-engineering", progressAcceptance=PROGRESS_ACCEPTANCE,
                                      phases=phases or [phase()]))


class SequenceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "sequence.sqlite3"
        self.now = 1000.
        self.config = fixture()
        self.coordinator = SequentialTrainingCoordinator(self.database, self.config, clock=lambda: self.now)

    def tearDown(self):
        self.coordinator.close()
        self.temporary.cleanup()

    def reset(self, config):
        self.coordinator.close()
        self.database = Path(self.temporary.name) / ("reset-" + str(len(list(Path(self.temporary.name).glob("*.sqlite3")))) + ".sqlite3")
        self.config = config
        self.coordinator = SequentialTrainingCoordinator(self.database, config, clock=lambda: self.now)

    def identity(self, owner="alice"):
        return dict(contributorId=owner, configHash=self.coordinator.config_hash,
                    modelFingerprint=self.coordinator.model_fingerprint)

    def lease(self, owner="alice"):
        return self.coordinator.lease(self.identity(owner))["job"]

    def credentials(self, job, owner="alice"):
        return dict(**self.identity(owner), jobId=job["jobId"], leaseToken=job["leaseToken"])

    def result(self, job, owner="alice", score=0., success=False):
        provenance = {key: copy.deepcopy(job[key]) for key in ("configHash", "modelFingerprint", "parametersHash",
                      "seed", "stage", "durationSeconds", "sign", "pairId", "generation", "parameters")}
        provenance.update(environmentVersion=self.config["environmentVersion"], dtMs=.5, bodyBlockMs=2,
                          bodyBackend="mujoco-wasm", backend="wasm", neuralEngine="wasm",
                          wasmExecution=copy.deepcopy(self.coordinator.acceptance_config["nativeExecution"]))
        return dict(**self.credentials(job, owner), objective=score, provenance=provenance,
                    metrics=dict(success=success, terminated=success, cancelled=False,
                                 simSeconds=job["durationSeconds"], steps=int(job["durationSeconds"] / .002),
                                 reason=("recovery_success" if job["stage"] == "recovery" else "stage_success") if success else "time_limit"))

    def finish_generation(self, role=None, candidate_score=2., baseline_score=0., success=True):
        status = self.coordinator.status()
        if role is not None:
            self.assertEqual(status["sequence"]["role"], role)
        jobs = []
        for _ in range(status["jobs"]["pending"]):
            job = self.lease()
            jobs.append(job)
            score = baseline_score if "-baseline-" in job["jobId"] else candidate_score
            payload = self.result(job, score=score, success=success)
            self.coordinator.result(payload)
        return jobs

    def error(self, code, action):
        with self.assertRaises(APIError) as context:
            action()
        self.assertEqual(context.exception.code, code)


class SchedulingChecks:
    """Run the same operational contract against SQLite and real Firestore."""
    def test_scheduling_expands_search_preserving_results_and_grants_grace_once(self):
        completed = self.lease()
        payload = self.result(completed, score=1.)
        self.coordinator.result(payload)
        active = self.lease("slow")
        before = self.coordinator.status()
        self.now += 2
        change = self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        self.assertEqual((change["addedJobs"], change["migratedLeases"]), (12, 1))
        status = self.coordinator.status()
        self.assertEqual(status["jobs"], dict(pending=14, leased=1, completed=1))
        self.assertEqual(status["acceptedResults"], before["acceptedResults"])
        self.assertEqual(status["checkpoint"]["parameters"], before["checkpoint"]["parameters"])
        self.assertEqual(status["configHash"], before["configHash"])
        self.assertEqual(status["config"]["optimizer"]["populationPairs"], 2)
        upgraded = self.lease("slow")
        for key, value in active.items():self.assertEqual(upgraded[key], value)
        self.assertEqual(upgraded["assignmentDeadlineAt"], self.now+1200)
        self.assertEqual(upgraded["assignmentStartSource"], "migration-grace")
        self.assertTrue(self.coordinator.result(payload)["duplicate"])
        checkpoint = self.coordinator.checkpoint()
        self.now += 1
        retry = self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        self.assertEqual((retry["addedJobs"], retry["migratedLeases"]), (0, 0))
        self.assertEqual(self.coordinator.checkpoint(), checkpoint)
        self.assertEqual(self.lease("slow")["assignmentDeadlineAt"], upgraded["assignmentDeadlineAt"])
        self.assertEqual(checkpoint["schedulingRevision"], 1)
        self.assertEqual(checkpoint["schedulingHistory"][0]["effectiveGeneration"], 0)
        self.assertEqual(checkpoint["currentSearchPairs"], 8)

    def test_deadline_cannot_be_extended_and_another_worker_can_finish_once(self):
        config = fixture(); config["contribution"]["leaseSeconds"] = 180
        self.reset(config)
        self.coordinator.set_scheduling(search_pairs=2, trial_timeout_seconds=1200)
        job = self.lease("slow"); deadline = job["assignmentDeadlineAt"]
        payload = self.result(job, owner="slow")
        for elapsed in (100, 250, 400, 550, 700, 850, 1000, 1150):
            self.now = job["assignmentStartedAt"]+elapsed
            response = self.coordinator.heartbeat(self.credentials(job, "slow"))
            self.assertEqual(response["assignmentDeadlineAt"], deadline)
            self.assertLessEqual(response["leaseExpiresAt"], deadline)
        self.assertEqual(response["leaseExpiresAt"], deadline)
        self.now = deadline
        self.error("assignment_timeout", lambda:self.coordinator.heartbeat(self.credentials(job, "slow")))
        self.error("assignment_timeout", lambda:self.coordinator.result(payload))
        replacement = self.lease("fast")
        self.assertEqual(replacement["jobId"], job["jobId"])
        self.assertEqual(replacement["parameters"], job["parameters"])
        self.assertEqual(replacement["seed"], job["seed"])
        self.assertNotEqual(replacement["leaseToken"], job["leaseToken"])
        self.assertEqual(replacement["assignmentAttempt"], 2)
        self.error("stale_lease", lambda:self.coordinator.result(payload))
        result = self.result(replacement, owner="fast")
        self.assertFalse(self.coordinator.result(result)["duplicate"])
        self.assertTrue(self.coordinator.result(result)["duplicate"])
        self.assertEqual(self.coordinator.status()["acceptedResults"], 1)

    def test_timeout_owner_cooldown_is_temporary_and_does_not_block_other_work(self):
        self.coordinator.set_scheduling(search_pairs=2, trial_timeout_seconds=1200)
        overdue = self.lease("slow")
        self.now = overdue["assignmentDeadlineAt"]
        different = self.lease("slow")
        self.assertNotEqual(different["jobId"], overdue["jobId"])
        self.coordinator.result(self.result(different, owner="slow"))
        for _ in range(2):
            other = self.lease("fast")
            # The fast worker can immediately recover the overdue assignment.
            self.coordinator.result(self.result(other, owner="fast"))
        remaining = self.lease("slow")
        self.now = remaining["assignmentDeadlineAt"]
        self.assertIsNone(self.lease("slow"))
        self.now += 59.999
        self.assertIsNone(self.lease("slow"))
        self.now += .001
        retry = self.lease("slow")
        self.assertEqual(retry["jobId"], remaining["jobId"])
        self.assertEqual(retry["assignmentAttempt"], 2)

    def test_policy_changed_during_comparison_applies_to_next_search_only(self):
        self.finish_generation("search")
        before = self.coordinator.status()
        change = self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        self.assertEqual(change["addedJobs"], 0)
        self.assertEqual(self.coordinator.status()["totalJobs"], 4)
        self.assertEqual(self.coordinator.status()["generation"], before["generation"])
        self.finish_generation("compare", candidate_score=-1, baseline_score=1, success=False)
        after = self.coordinator.status()
        self.assertEqual(after["sequence"]["role"], "search")
        self.assertEqual(after["jobs"], dict(pending=16, leased=0, completed=0))
        self.assertEqual(after["currentSearchPairs"], 8)

    def test_runtime_scales_by_assigned_duration_and_fit_total(self):
        for stage, duration, expected in (("takeoff", 3, 720), ("landing", 8, 1920), ("flight", 1, 600)):
            config = fixture([phase(stage=stage)])
            next(s for s in config["stages"] if s["id"] == stage)["durationSeconds"] = duration
            self.reset(config)
            self.coordinator.set_scheduling(search_pairs=2, trial_timeout_seconds=1200)
            self.assertEqual(self.lease()["assignmentTimeoutSeconds"], expected)
        config = fixture([phase("decoder", list(range(24, 696)), kind="decoder-fit")])
        self.reset(config)
        self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        job = self.lease()
        self.assertEqual(job["mode"], "decoder-fit")
        self.assertEqual(job["assignmentTimeoutSeconds"], 3600)
        self.assertEqual(self.coordinator.status()["totalJobs"], 1)

    def test_invalid_scheduling_does_not_modify_run(self):
        before = self.coordinator.status()
        for pairs, timeout in ((True,1200), (0,1200), (129,1200), (8,599), (8,float("inf"))):
            with self.subTest(pairs=pairs, timeout=timeout), self.assertRaises((ValueError, APIError)):
                self.coordinator.set_scheduling(search_pairs=pairs, trial_timeout_seconds=timeout)
            self.assertEqual(self.coordinator.status(), before)


class SequentialSchedulingTests(SchedulingChecks, SequenceFixture):
    def test_expansion_preserves_existing_complete_rows_and_scientific_payloads(self):
        complete = self.lease(); self.coordinator.result(self.result(complete))
        active = self.lease("slow")
        before = {r["job_id"]:dict(r) for r in self.coordinator.db.execute("SELECT * FROM sequence_jobs")}
        self.coordinator.set_scheduling(search_pairs=8, trial_timeout_seconds=1200)
        after = {r["job_id"]:dict(r) for r in self.coordinator.db.execute("SELECT * FROM sequence_jobs")}
        self.assertEqual(before[complete["jobId"]], after[complete["jobId"]])
        for key, row in before.items():
            old = json.loads(row["payload"]); new = json.loads(after[key]["payload"])
            for field, value in old.items():self.assertEqual(new[field], value)
            for field in ("contributor", "token", "expires", "state", "result", "score"):
                self.assertEqual(row[field], after[key][field])
        self.coordinator.close()
        self.coordinator = SequentialTrainingCoordinator(self.database, self.config, clock=lambda:self.now)
        self.assertEqual(self.coordinator.status()["totalJobs"], 16)
        self.assertEqual(self.lease("slow")["leaseToken"], active["leaseToken"])


class SequentialTrainingTests(SequenceFixture):
    def test_search_mask_freezes_every_other_family_and_jobs_are_paired(self):
        jobs = [self.lease(f"worker-{i}") for i in range(4)]
        self.assertEqual(len({j["jobId"] for j in jobs}), 4)
        self.assertTrue(all(len(j["parameters"]) == 696 for j in jobs))
        for job in jobs:
            self.assertEqual(job["parameters"][5:], [0.] * 691)
            self.assertTrue(any(job["parameters"][:5]))
            self.assertEqual(job["mode"], "evaluation")
        for pair in range(2):
            pair_jobs = [j for j in jobs if j["pairId"] == pair]
            self.assertEqual(len(pair_jobs), 2)
            self.assertEqual(pair_jobs[0]["seed"], pair_jobs[1]["seed"])
            self.assertEqual({j["sign"] for j in pair_jobs}, {-1, 1})
            for a, b in zip(pair_jobs[0]["parameters"], pair_jobs[1]["parameters"]):
                self.assertAlmostEqual(a, -b)

    def test_search_comparison_fresh_gate_then_next_phase_and_checkpoint(self):
        self.reset(fixture([phase(), phase("antenna", list(range(5, 11)), stage="takeoff")]))
        initial = self.coordinator.checkpoint()["parameters"]
        search = self.finish_generation("search")
        self.assertEqual(self.coordinator.checkpoint()["parameters"], initial)
        compare = self.finish_generation("compare")
        self.assertEqual(self.coordinator.checkpoint()["parameters"], initial)
        gates = self.finish_generation("gate")
        self.assertTrue(set(j["seed"] for j in gates).isdisjoint(j["seed"] for j in search + compare))
        for seed in set(j["seed"] for j in compare):
            self.assertEqual(len([j for j in compare if j["seed"] == seed]), 2)
        checkpoint = self.coordinator.checkpoint()
        self.assertNotEqual(checkpoint["parameters"][:5], initial[:5])
        self.assertEqual(checkpoint["parameters"][5:], initial[5:])
        self.assertEqual(len(checkpoint["completedPhases"]), 1)
        self.assertEqual(checkpoint["status"], "unverified")
        self.assertFalse(checkpoint["biologicalSuccessValidated"])
        self.assertEqual(self.coordinator.status()["sequence"]["phaseId"], "antenna")
        job = self.lease()
        self.assertEqual((job["stage"], job["durationSeconds"]), ("takeoff", 3))
        self.assertEqual(job["parameters"][:5], checkpoint["parameters"][:5])
        self.assertEqual(job["parameters"][11:], checkpoint["parameters"][11:])
        self.assertEqual(self.coordinator.db.execute("SELECT COUNT(*) FROM sequence_checkpoints").fetchone()[0], 1)

    def test_failed_fresh_gate_never_promotes_and_exhaustion_stops_work(self):
        self.reset(fixture([phase(rounds=1)]))
        before = self.coordinator.checkpoint()["parameters"]
        self.finish_generation("search"); self.finish_generation("compare")
        self.finish_generation("gate", candidate_score=-1., success=False)
        status = self.coordinator.status()
        self.assertEqual(status["sequence"]["status"], "needs-review")
        self.assertEqual(status["sequence"]["reason"], "candidate-did-not-pass-fresh-paired-comparison")
        self.assertEqual(status["checkpoint"]["parameters"], before)
        self.assertEqual(status["checkpoint"]["completedPhases"], [])
        self.assertIsNone(self.lease())
        self.assertEqual(self.coordinator.db.execute("SELECT COUNT(*) FROM sequence_checkpoints").fetchone()[0], 0)

    def test_failed_comparison_retries_with_fresh_jobs_without_changing_center(self):
        before = self.coordinator.checkpoint()["parameters"]
        search = self.finish_generation("search")
        compare = self.finish_generation("compare", candidate_score=-1., baseline_score=1., success=True)
        status = self.coordinator.status()
        self.assertEqual(status["sequence"]["role"], "search")
        self.assertEqual(status["sequence"]["round"], 1)
        self.assertEqual(status["checkpoint"]["parameters"], before)
        self.assertNotIn(self.lease()["seed"], [j["seed"] for j in search + compare])

    def test_unsuccessful_but_improving_candidate_updates_center_then_later_success_advances(self):
        self.reset(fixture([phase(rounds=3), phase("takeoff", list(range(24, 696)), stage="takeoff")]))
        original=self.coordinator.checkpoint()["parameters"]
        self.finish_generation("search",candidate_score=-2,success=False)
        self.finish_generation("compare",candidate_score=-2,baseline_score=-3,success=False)
        gates=self.finish_generation("gate",candidate_score=-1.8,baseline_score=-2.8,success=False)
        self.assertEqual(len(gates),4)
        self.assertEqual(len({job["seed"] for job in gates}),2)
        for seed in {job["seed"] for job in gates}:
            matched=[job for job in gates if job["seed"]==seed]
            self.assertEqual({job["sign"] for job in matched},{-1,1})
            self.assertEqual(len(matched),2)
        accepted=self.coordinator.checkpoint()
        status=self.coordinator.status()
        self.assertNotEqual(accepted["parameters"],original)
        self.assertEqual(accepted["parameters"][5:],original[5:])
        self.assertEqual(accepted["completedPhases"],[])
        self.assertEqual((status["sequence"]["phaseId"],status["sequence"]["round"],status["sequence"]["role"]),("legs",1,"search"))
        self.assertTrue(status["sequence"]["history"][-1]["accepted"])
        self.assertFalse(status["sequence"]["history"][-1]["phasePassed"])
        update=accepted["lastParameterUpdate"]
        self.assertEqual(update,status["lastParameterUpdate"])
        self.assertEqual(update["changedCount"],sum(a!=b for a,b in zip(original,accepted["parameters"])))
        self.assertEqual(update["parameterCount"],696)
        self.assertFalse(update["phasePassed"])
        self.assertAlmostEqual(update["updateL2"],math.sqrt(sum((a-b)**2 for a,b in zip(original,accepted["parameters"]))))
        self.finish_generation("search")
        comparison=self.finish_generation("compare")
        self.assertTrue(all(job["parameters"]==accepted["parameters"] for job in comparison if "-baseline-" in job["jobId"]))
        self.finish_generation("gate")
        status=self.coordinator.status()
        self.assertEqual(status["sequence"]["phaseId"],"takeoff")
        self.assertEqual(len(status["checkpoint"]["completedPhases"]),1)
        self.assertTrue(status["checkpoint"]["completedPhases"][0]["phasePassed"])
        self.assertEqual(self.coordinator.db.execute("SELECT COUNT(*) FROM sequence_checkpoints").fetchone()[0],2)

    def test_budget_stop_keeps_an_improved_incomplete_checkpoint_across_restart(self):
        self.reset(fixture([phase(rounds=1)]))
        original=self.coordinator.checkpoint()["parameters"]
        self.finish_generation("search",success=False)
        self.finish_generation("compare",candidate_score=-2,baseline_score=-3,success=False)
        self.finish_generation("gate",candidate_score=-1,baseline_score=-2,success=False)
        accepted=self.coordinator.checkpoint()
        self.assertNotEqual(accepted["parameters"],original)
        self.assertEqual(accepted["sequenceStatus"],"needs-review")
        self.assertEqual(accepted["completedPhases"],[])
        self.assertFalse(accepted["lastParameterUpdate"]["phasePassed"])
        self.assertIsNone(self.lease())
        stored=json.loads(self.coordinator.db.execute("SELECT value FROM sequence_checkpoints").fetchone()[0])
        self.assertEqual(stored["parameters"],accepted["parameters"])
        self.assertEqual(stored["sequenceStatus"],"needs-review")
        self.coordinator.close()
        self.coordinator=SequentialTrainingCoordinator(self.database,self.config,clock=lambda:self.now)
        self.assertEqual(self.coordinator.checkpoint(),accepted)

    def test_later_nonimprovement_keeps_the_previous_incremental_progress(self):
        self.reset(fixture([phase(rounds=2)]))
        self.finish_generation("search",success=False)
        self.finish_generation("compare",success=False)
        self.finish_generation("gate",success=False)
        accepted=self.coordinator.checkpoint()
        self.finish_generation("search",success=False)
        self.finish_generation("compare",candidate_score=-1,baseline_score=0,success=False)
        stopped=self.coordinator.checkpoint()
        self.assertEqual(stopped["parameters"],accepted["parameters"])
        self.assertEqual(stopped["lastParameterUpdate"],accepted["lastParameterUpdate"])
        self.assertEqual(stopped["sequenceStatus"],"needs-review")

    def test_previous_passed_current_task_cannot_regress_in_a_later_sensory_phase(self):
        self.reset(fixture([phase(),phase("antenna",list(range(5,11)),rounds=1)]))
        self.finish_generation("search");self.finish_generation("compare");self.finish_generation("gate")
        accepted=self.coordinator.checkpoint()
        self.finish_generation("search",success=False)
        self.finish_generation("compare",success=False)
        self.finish_generation("gate",success=False)
        stopped=self.coordinator.checkpoint()
        self.assertEqual(stopped["parameters"],accepted["parameters"])
        self.assertEqual(stopped["completedPhases"],accepted["completedPhases"])
        self.assertEqual(self.coordinator.status()["sequence"]["reason"],"candidate-did-not-retain-completed-stage")

    def test_completed_sequence_has_no_more_jobs_and_retains_unverified_checkpoint(self):
        self.finish_generation("search"); self.finish_generation("compare"); self.finish_generation("gate")
        status = self.coordinator.status()
        self.assertEqual(status["sequence"]["status"], "complete")
        self.assertEqual(status["checkpoint"]["sequenceStatus"], "complete")
        self.assertIsNone(self.lease())
        self.assertFalse(status["biologicalSuccessValidated"])

    def fit_payload(self, passed=True):
        self.reset(fixture([phase("decoder-fit", list(range(24, 696)), kind="decoder-fit", rounds=1)]))
        job = self.lease(); payload = self.result(job)
        payload["metrics"]["reason"] = "fit_candidate" if passed else "fit_failed"
        payload["calibration"] = dict(passed=passed, trainingTrials=2, validationTrials=1, teacherUsedForEvaluation=False,
                                      teacherOnlyDuringDemonstrations=True, autonomousEvaluationPerformed=False)
        if passed:
            payload["candidateParameters"] = job["parameters"].copy()
            payload["candidateParameters"][24] = .1
        return job, payload

    def test_decoder_fit_is_only_a_proposal_and_requires_autonomous_comparison_and_gate(self):
        job, payload = self.fit_payload()
        self.assertEqual(job["mode"], "decoder-fit")
        self.assertEqual([t["split"] for t in job["calibrationTrials"]], ["train", "train", "validation"])
        self.assertEqual(len({t["seed"] for t in job["calibrationTrials"]}), 3)
        self.coordinator.result(payload)
        self.assertEqual(self.coordinator.checkpoint()["parameters"], job["parameters"])
        comparison = self.finish_generation("compare")
        self.assertTrue(all(j["mode"] == "evaluation" for j in comparison))
        self.assertEqual(self.coordinator.checkpoint()["parameters"], job["parameters"])
        self.finish_generation("gate")
        self.assertEqual(self.coordinator.checkpoint()["parameters"], payload["candidateParameters"])
        fit_trial=next(trial for trial in self.coordinator.status()["recentTrials"] if trial["mode"]=="decoder-fit")
        self.assertEqual(fit_trial["reason"], "fit_candidate")

    def test_failed_fit_or_autonomous_comparison_cannot_promote_claimed_weights(self):
        job, payload = self.fit_payload(False)
        self.coordinator.result(payload)
        self.assertEqual(self.coordinator.status()["sequence"]["status"], "needs-review")
        self.assertEqual(self.coordinator.checkpoint()["parameters"], job["parameters"])
        job, payload = self.fit_payload(True)
        self.coordinator.result(payload)
        self.finish_generation("compare", candidate_score=-1., baseline_score=1.)
        self.assertEqual(self.coordinator.status()["sequence"]["status"], "needs-review")
        self.assertEqual(self.coordinator.checkpoint()["parameters"], job["parameters"])

    def test_fit_cannot_change_sensory_parameters_or_submit_malformed_candidates(self):
        job, payload = self.fit_payload()
        for mutate in [lambda p: p["candidateParameters"].__setitem__(0, .1),
                       lambda p: p["candidateParameters"].__setitem__(24, .3),
                       lambda p: p["candidateParameters"].pop(),
                       lambda p: p["calibration"].__setitem__("teacherUsedForEvaluation", True),
                       lambda p: p["calibration"].__setitem__("validationTrials", 0)]:
            bad = copy.deepcopy(payload); mutate(bad)
            with self.assertRaises(APIError): self.coordinator.result(bad)
            self.assertEqual(self.coordinator.status()["acceptedResults"], 0)
        self.coordinator.result(payload)

    def test_fit_provenance_is_pinned_even_for_failed_fits_and_malformed_counts_are_client_errors(self):
        for passed in (True, False):
            job, payload = self.fit_payload(passed)
            for key, value in [("environmentVersion", "wrong"), ("dtMs", 1), ("bodyBlockMs", 1),
                               ("bodyBackend", "other"), ("backend", "webgpu"), ("neuralEngine", "webgpu"),
                               ("wasmExecution", {}), ("generation", False), ("sign", False),
                               ("pairId", False), ("durationSeconds", 1), ("parameters", [False] * 696)]:
                with self.subTest(passed=passed, field=key):
                    bad = copy.deepcopy(payload); bad["provenance"][key] = value
                    self.error("provenance_mismatch", lambda: self.coordinator.result(bad))
            for key, value in [("trainingTrials", "2"), ("validationTrials", True), ("trainingTrials", -1),
                               ("validationTrials", 2), ("teacherOnlyDuringDemonstrations", False),
                               ("autonomousEvaluationPerformed", True)]:
                with self.subTest(passed=passed, field=key):
                    bad = copy.deepcopy(payload); bad["calibration"][key] = value
                    self.error("invalid_calibration", lambda: self.coordinator.result(bad))
            self.assertEqual(self.coordinator.status()["acceptedResults"], 0)
            self.coordinator.result(payload)

    def test_failed_fit_can_report_only_attempted_demonstrations(self):
        job,payload=self.fit_payload(False)
        payload["calibration"].update(trainingTrials=1,validationTrials=0)
        payload["metrics"]["reason"]="fit_failed"
        self.coordinator.result(payload)
        status=self.coordinator.status()
        self.assertEqual(status["sequence"]["status"],"needs-review")
        self.assertEqual(status["recentTrials"][0]["mode"],"decoder-fit")
        self.assertEqual(status["recentTrials"][0]["reason"],"fit_failed")
        self.assertEqual(status["checkpoint"]["parameters"],job["parameters"])

    def test_seed_domains_separate_whole_episode_splits_and_deterministically_skip_reserved_seeds(self):
        state={"phaseIndex":0,"round":0}
        roles=["noise","search","fit-job","fit-train","fit-validation","compare","gate-flight"]
        seeds=[self.coordinator._seed(state,role,0) for role in roles]
        self.assertEqual([seed>>29 for seed in seeds],list(range(7)))
        self.assertEqual(len(set(seeds)),len(seeds))
        self.assertTrue(set(seeds).isdisjoint(self.coordinator.guard_reserved_seeds))
        reserved=self.coordinator._seed(state,"fit-validation",0)
        self.coordinator.guard_reserved_seeds.add(reserved)
        retried=self.coordinator._seed(state,"fit-validation",0)
        self.assertNotEqual(retried,reserved)
        self.assertEqual(retried,self.coordinator._seed(state,"fit-validation",0))
        self.assertEqual(retried>>29,4)

    def test_regression_gate_retests_previous_physical_stages_and_blocks_forgetting(self):
        self.reset(fixture([phase(), phase("recover", list(range(5, 11)), rounds=1, stage="recovery")]))
        self.finish_generation("search"); self.finish_generation("compare"); first_gate = self.finish_generation("gate")
        before = self.coordinator.checkpoint()
        self.finish_generation("search"); comparison = self.finish_generation("compare")
        jobs = [self.lease(f"gate-{i}") for i in range(6)]
        self.assertEqual({j["stage"] for j in jobs}, {"recovery", "flight"})
        self.assertEqual(len({j["seed"] for j in jobs}), 4)
        self.assertTrue({j["seed"] for j in jobs}.isdisjoint(j["seed"] for j in first_gate + comparison))
        for i, job in enumerate(jobs):
            self.assertEqual(job["phaseId"], "recover")
            self.assertEqual(job["durationSeconds"], 4 if job["stage"] == "recovery" else 5)
            # Recovery passes, but previously learned flight fails. Neither
            # good scores nor aggregate successes may hide this regression.
            self.coordinator.result(self.result(job, f"gate-{i}", score=0 if "-baseline-" in job["jobId"] else 4,
                                                success=job["stage"] == "recovery"))
        after = self.coordinator.checkpoint()
        self.assertEqual(after["parameters"], before["parameters"])
        self.assertEqual(after["completedPhases"], before["completedPhases"])
        self.assertEqual(after["sequenceStatus"], "needs-review")
        self.assertEqual(self.coordinator.status()["sequence"]["reason"],"candidate-did-not-retain-completed-stage")

    def test_regression_gate_deduplicates_physical_tasks_and_records_success_per_stage(self):
        self.reset(fixture([phase(), phase("antenna", list(range(5, 11))),
                            phase("recover", list(range(11, 18)), stage="recovery")]))
        for _ in range(2):
            self.finish_generation("search"); self.finish_generation("compare")
            self.assertEqual(self.coordinator.status()["jobs"]["pending"], 4)
            self.finish_generation("gate")
        self.finish_generation("search"); self.finish_generation("compare")
        self.assertEqual(self.coordinator.status()["jobs"]["pending"], 6)
        self.finish_generation("gate")
        checkpoint = self.coordinator.checkpoint()
        self.assertEqual(checkpoint["sequenceStatus"], "complete")
        self.assertEqual(checkpoint["completedPhases"][-1]["stageResults"],
                         [dict(stage="recovery", episodes=2, successes=2, meanReturn=2), dict(stage="flight", episodes=2, successes=2, meanReturn=2)])
        self.coordinator.close()
        self.coordinator = SequentialTrainingCoordinator(self.database, self.config, clock=lambda: self.now)
        self.assertEqual(self.coordinator.checkpoint(), checkpoint)

    def test_result_retry_release_expiry_and_restart_keep_assignment_identity(self):
        job = self.lease()
        self.assertEqual(job, self.lease())
        payload = self.result(job)
        self.assertFalse(self.coordinator.result(payload)["duplicate"])
        self.assertTrue(self.coordinator.result(payload)["duplicate"])
        conflict = copy.deepcopy(payload); conflict["objective"] = 1
        self.error("result_conflict", lambda: self.coordinator.result(conflict))
        released = self.lease()
        self.assertFalse(self.coordinator.release(self.credentials(released))["duplicate"])
        self.assertTrue(self.coordinator.release(self.credentials(released))["duplicate"])
        renewed = self.lease("bob")
        self.assertEqual(renewed["jobId"], released["jobId"])
        self.error("stale_lease", lambda: self.coordinator.result(self.result(released)))
        self.now += 10
        self.error("lease_expired", lambda: self.coordinator.result(self.result(renewed, "bob")))
        after_expiry = self.lease("carol")
        self.assertEqual(after_expiry["jobId"], renewed["jobId"])
        self.assertNotEqual(after_expiry["leaseToken"], renewed["leaseToken"])
        before = self.coordinator.checkpoint()
        self.coordinator.close()
        self.coordinator = SequentialTrainingCoordinator(self.database, self.config, clock=lambda: self.now)
        self.assertEqual(self.coordinator.checkpoint(), before)
        self.assertEqual(self.lease("carol"), after_expiry)
        self.assertTrue(self.coordinator.result(payload)["duplicate"])

    def test_concurrent_contributors_and_duplicate_final_submissions_advance_once(self):
        owners = [f"worker-{i}" for i in range(8)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            assignments = list(executor.map(self.lease, owners))
        leased = [(o, j) for o, j in zip(owners, assignments) if j]
        self.assertEqual(len(leased), 4)
        self.assertEqual(len({j["jobId"] for _, j in leased}), 4)
        for owner, job in leased[:-1]: self.coordinator.result(self.result(job, owner))
        owner, job = leased[-1]; payload = self.result(job, owner)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.coordinator.result(payload), range(4)))
        self.assertEqual(sum(not r["duplicate"] for r in results), 1)
        status = self.coordinator.status()
        self.assertEqual(status["acceptedResults"], 4)
        self.assertEqual(status["generation"], 1)
        self.assertEqual(status["totalJobs"], 4)

    def test_client_and_database_identity_mismatches_fail_closed(self):
        bad = self.identity(); bad["configHash"] = "0" * 64
        self.error("incompatible_config", lambda: self.coordinator.lease(bad))
        bad = self.identity(); bad["modelFingerprint"] = "0" * 64
        self.error("incompatible_model", lambda: self.coordinator.lease(bad))
        job = self.lease(); payload = self.result(job); payload["provenance"]["parameters"][0] += .01
        self.error("provenance_mismatch", lambda: self.coordinator.result(payload))
        changed = copy.deepcopy(self.config); changed["environmentVersion"] += "-different"
        with self.assertRaisesRegex(ValueError, "different configuration"):
            SequentialTrainingCoordinator(self.database, changed, clock=lambda: self.now)

    def test_constructor_closes_connection_after_database_identity_failure(self):
        import sqlite3
        connection = sqlite3.connect(str(self.database), check_same_thread=False, isolation_level=None)
        changed = copy.deepcopy(self.config); changed["environmentVersion"] += "-different"
        with patch("sequential_training.sqlite3.connect", return_value=connection):
            with self.assertRaisesRegex(ValueError, "different configuration"):
                SequentialTrainingCoordinator(self.database, changed, clock=lambda: self.now)
        with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
            connection.execute("SELECT 1")

    def test_multiple_coordinator_instances_serialize_duplicate_final_result(self):
        jobs = [self.lease(f"worker-{i}") for i in range(4)]
        for i, job in enumerate(jobs[:-1]): self.coordinator.result(self.result(job, f"worker-{i}"))
        other = SequentialTrainingCoordinator(self.database, self.config, clock=lambda: self.now)
        payload = self.result(jobs[-1], "worker-3")
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda coordinator: coordinator.result(payload), [self.coordinator, other]))
            self.assertEqual(sum(not result["duplicate"] for result in results), 1)
            self.assertEqual(self.coordinator.status()["generation"], 1)
            self.assertEqual(other.status()["acceptedResults"], 4)
        finally:
            other.close()

    def test_sequence_rejects_non_wasm_or_unguarded_execution(self):
        config = fixture(); config["schemaVersion"] = 1; del config["optimizer"]["acceptance"]
        with self.assertRaisesRegex(ValueError, "schema 2 and pinned WASM"):
            SequentialTrainingCoordinator(":memory:", config)
        config = fixture(); config["optimizer"]["acceptance"]["nativeExecution"] = dict(
            backend="dawn-metal", moduleSha256="a" * 64, packageLockSha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "schema 2 and pinned WASM"):
            SequentialTrainingCoordinator(":memory:", config)

    def test_invalid_phase_masks_and_fit_masks_are_rejected(self):
        for mutate in [lambda c: c["trainingSequence"]["phases"][0].__setitem__("parameterIndices", [0, 0]),
                       lambda c: c["trainingSequence"]["phases"][0].__setitem__("parameterIndices", [696]),
                       lambda c: c["trainingSequence"]["phases"][0].__setitem__("kind", "decoder-fit"),
                       lambda c: c["trainingSequence"]["phases"][0].__setitem__("minimumSuccesses", 3),
                       lambda c: c["motorDecoderContract"]["parameters"][0].__setitem__("max", 1),
                       lambda c: c["trainingSequence"].pop("progressAcceptance"),
                       lambda c: c["trainingSequence"].__setitem__("progressAcceptance","previous-profile")]:
            bad = copy.deepcopy(self.config); mutate(bad)
            with self.assertRaises((ValueError, APIError)): parse_sequence(bad)


if __name__ == "__main__":
    unittest.main()
