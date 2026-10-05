"""Durable coordinator for a fixed-identity, browser-executed training sequence.

Every search, demonstration fit, comparison and gate is assigned. Stage progress
means the declared engineering tests passed; it never identifies physiology.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import random
import secrets
import sqlite3
import threading
import time

from training_coordinator import APIError, CoordinatorRules, canonical, finite_number, fingerprint, identifier

CONTRACT = "banc-sensorimotor-sequence-v1"
PROGRESS_ACCEPTANCE = "paired-improvement-before-stage-completion-v1"
SCHEDULING_KIND = "bounded-search-runtime-v1"
REASSIGNMENT_COOLDOWN_SECONDS = 60


def scheduling_policy(search_pairs, trial_timeout_seconds):
    """Operational scheduling is separate from the immutable simulation config."""
    if type(search_pairs) is not int or not 1 <= search_pairs <= 128:
        raise ValueError("search_pairs must be an integer from 1 to 128")
    timeout = finite_number(trial_timeout_seconds, 600, 86400, "trial timeout")
    return dict(schema=1, kind=SCHEDULING_KIND, searchPairs=search_pairs,
                trialTimeoutSeconds=timeout, referenceDurationSeconds=5,
                minimumTimeoutSeconds=600, reassignmentCooldownSeconds=REASSIGNMENT_COOLDOWN_SECONDS)


def validate_scheduling(policy):
    if not isinstance(policy, dict) or policy != scheduling_policy(
            policy.get("searchPairs"), policy.get("trialTimeoutSeconds")):
        raise ValueError("Invalid sequential scheduling policy")
    return policy


def parse_sequence(config):
    marker = config.get("sensorimotorSequence", {})
    parameters = config.get("parameters", [])
    names = marker.get("sensoryParameterNames", [])
    if (config.get("parameterContract") != CONTRACT or marker.get("schema") != 1
            or not isinstance(names, list) or not 1 <= len(names) <= 96
            or len(parameters) != len(names) + 672
            or [p.get("name") for p in parameters[:len(names)]] != names
            or len(set(names)) != len(names) or any(not n.startswith("sensory_") for n in names)):
        raise ValueError("Invalid sensorimotor parameter contract")
    motor = config.get("motorDecoderContract", {}).get("parameters", [])
    if len(motor) != 672 or any(any(p.get(k) != m.get(k) for k in ("name", "min", "max"))
                                  for p, m in zip(parameters[len(names):], motor)):
        raise ValueError("Sequence motor suffix differs from anatomical decoder contract")
    sequence = config.get("trainingSequence")
    if (not isinstance(sequence, dict) or sequence.get("schema") != 1
            or sequence.get("calibrationMode") != "simulation-engineering"
            or sequence.get("progressAcceptance") != PROGRESS_ACCEPTANCE):
        raise ValueError("Explicit simulation-engineering sequence required")
    phases = sequence.get("phases")
    if not isinstance(phases, list) or not 1 <= len(phases) <= 24:
        raise ValueError("Sequence requires ordered phases")
    available = {s["id"]: s["durationSeconds"] for s in config["stages"]}
    ids = set()
    for phase in phases:
        phase_id = identifier(phase.get("id"), "phase id")
        indices = phase.get("parameterIndices")
        if phase_id in ids or phase.get("kind") not in ("search", "decoder-fit") or phase.get("stage") not in available:
            raise ValueError("Invalid or duplicate training phase")
        ids.add(phase_id)
        if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(parameters) for i in indices)):
            raise ValueError("Phase mask requires unique parameter indices")
        if phase["kind"] == "decoder-fit" and indices != list(range(len(names), len(parameters))):
            raise ValueError("Decoder fitting may change only the complete motor suffix")
        for key, lo, hi in (("maxRounds", 1, 100), ("validationCount", 2, 16), ("minimumSuccesses", 0, 16)):
            if type(phase.get(key)) is not int or not lo <= phase[key] <= hi:
                raise ValueError("Invalid phase " + key)
        if phase["minimumSuccesses"] > phase["validationCount"]:
            raise ValueError("Impossible success gate")
        finite_number(phase.get("minimumMeanImprovement"), 0, 20, "minimumMeanImprovement")
    return copy.deepcopy(sequence), len(names)


class SequentialTrainingCoordinator(CoordinatorRules):
    def __init__(self, database, config, config_hash=None, clock=time.time):
        self.sequence, self.sensory_count = parse_sequence(config)
        # Reuse the existing execution/asset/guard validation for the unchanged
        # motor suffix; the complete sequence vector is checked independently.
        shadow = copy.deepcopy(config)
        shadow["parameterContract"] = "banc-masked-motor-decoder-v1"
        shadow["parameters"] = shadow["parameters"][self.sensory_count:]
        self._configure(shadow, config_hash or fingerprint(config), clock)
        if not self.guarded or self.acceptance_config["nativeExecution"]["backend"] != "wasm":
            raise ValueError("Sequential training requires schema 2 and pinned WASM execution")
        self.config = copy.deepcopy(config)
        self.names, self.bounds, self.initial, self.search_scales = [], [], [], []
        for p in config["parameters"]:
            name = identifier(p.get("name"), "parameter name")
            lo = finite_number(p.get("min"), -1e9, 1e9, "min")
            hi = finite_number(p.get("max"), -1e9, 1e9, "max")
            if lo >= hi or name in self.names:
                raise ValueError("Invalid sequence parameter bounds/names")
            self.names.append(name); self.bounds.append((lo, hi))
            self.initial.append(finite_number(p.get("initial"), lo, hi, "initial"))
            self.search_scales.append(finite_number(p.get("searchScale", 1), 1e-9, 1, "searchScale"))
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(database), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sequence_identity(id INTEGER PRIMARY KEY CHECK(id=1), config_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sequence_state(id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sequence_jobs(job_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, role TEXT NOT NULL,
                payload TEXT NOT NULL, state TEXT NOT NULL, contributor TEXT, token TEXT, expires REAL,
                result_hash TEXT, result TEXT, score REAL, completed REAL);
            CREATE INDEX IF NOT EXISTS sequence_work ON sequence_jobs(generation,state,expires);
            CREATE TABLE IF NOT EXISTS sequence_checkpoints(generation INTEGER PRIMARY KEY, value TEXT NOT NULL);
            """)
            with self.transaction():
                row = self.db.execute("SELECT config_hash FROM sequence_identity WHERE id=1").fetchone()
                if row and row[0] != self.config_hash:
                    raise ValueError("Sequence database belongs to a different configuration")
                if not row:
                    self.db.execute("INSERT INTO sequence_identity VALUES(1,?)", (self.config_hash,))
                    state = dict(phaseIndex=0, round=0, generation=0, center=self.initial, status="running", role="search",
                                 acceptedResults=0, completedPhases=[], history=[], changedParameters=0, proposal=None)
                    self._start_round(state); self._save(state)
                else:
                    self._state()
        except BaseException:
            self.db.close()
            raise

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

    def _state(self):
        state = json.loads(self.db.execute("SELECT value FROM sequence_state WHERE id=1").fetchone()[0])
        if "scheduling" in state:
            validate_scheduling(state["scheduling"])
            if state["status"] == "running" and state["role"] == "search":
                pairs = state.get("currentSearchPairs")
                if type(pairs) is not int or not 1 <= pairs <= 128:
                    raise APIError(503, "invalid_history", "Invalid active search batch size")
        elif "currentSearchPairs" in state:
            raise APIError(503, "invalid_history", "Active search override has no scheduling policy")
        phase = self.sequence["phases"][min(state["phaseIndex"], len(self.sequence["phases"])-1)]
        self.stage = phase["stage"]
        self.duration = next(s["durationSeconds"] for s in self.config["stages"] if s["id"] == self.stage)
        return state

    def _save(self, state):
        self.db.execute("INSERT OR REPLACE INTO sequence_state VALUES(1,?)", (canonical(state),))

    def _phase(self, state):
        return self.sequence["phases"][state["phaseIndex"]]

    def _seed(self, state, role, index):
        # Disjoint uint32 domains keep fitting, comparison and final-gate
        # episodes separate even after truncating the digest. Whole episodes
        # are the split unit; configured held-out seeds are never assigned.
        domains={"noise":0,"search":1,"fit-job":2,"fit-train":3,"fit-validation":4,"compare":5}
        domain=6 if role.startswith("gate-") else domains[role]
        label = f'{self.config_hash}:{state["phaseIndex"]}:{state["round"]}:{role}:{index}'
        attempt=0
        while True:
            material=label if attempt==0 else f"{label}:retry:{attempt}"
            seed=(int.from_bytes(hashlib.sha256(material.encode()).digest()[:4],"little") & 0x1fffffff) | (domain<<29)
            if seed not in self.guard_reserved_seeds:
                return seed
            attempt+=1

    def _job(self, state, role, vector, seed, pair=0, sign=0, stage=None, **extra):
        phase = self._phase(state)
        stage = stage or phase["stage"]
        job_id = f'p{state["phaseIndex"]}-r{state["round"]}-g{state["generation"]}-{role}-{pair}-{sign}'
        job = dict(jobId=job_id, modelFingerprint=self.model_fingerprint, configHash=self.config_hash,
                   environmentVersion=self.environment_version, parameters=list(vector), parameterNames=self.names,
                   parametersHash=fingerprint(vector), seed=seed, stage=stage,
                   durationSeconds=next(s["durationSeconds"] for s in self.config["stages"] if s["id"] == stage),
                   sign=sign, pairId=pair, generation=state["generation"], phaseId=phase["id"],
                   mode="evaluation", **extra)
        self.db.execute("INSERT INTO sequence_jobs(job_id,generation,role,payload,state) VALUES(?,?,?,?, 'pending')",
                        (job_id, state["generation"], role, canonical(job)))

    def _start_round(self, state):
        phase = self._phase(state)
        state["role"] = "fit" if phase["kind"] == "decoder-fit" else "search"
        state["proposal"] = None
        if phase["kind"] == "decoder-fit":
            trials = [dict(seed=self._seed(state, "fit-"+split, i), split=split, stage=phase["stage"],
                           durationSeconds=next(s["durationSeconds"] for s in self.config["stages"] if s["id"] == phase["stage"]))
                      for i, split in enumerate(("train", "train", "validation"))]
            self._job(state, "fit", state["center"], self._seed(state,"fit-job",0), calibrationTrials=trials)
            # mode is deliberately explicit in the stored assignment.
            row=self.db.execute("SELECT job_id,payload FROM sequence_jobs WHERE generation=?",(state["generation"],)).fetchone()
            payload=json.loads(row["payload"]);payload["mode"]="decoder-fit"
            self.db.execute("UPDATE sequence_jobs SET payload=? WHERE job_id=?",(canonical(payload),row["job_id"]))
            return
        pairs = self.pairs
        if state.get("scheduling") is not None:
            pairs = validate_scheduling(state["scheduling"])["searchPairs"]
            state["currentSearchPairs"] = pairs
        self._append_search_pairs(state, 0, pairs)

    def _append_search_pairs(self, state, first, stop):
        phase = self._phase(state)
        for pair in range(first, stop):
            rng = random.Random(self._seed(state, "noise", pair))
            noise = {i: rng.gauss(0, 1) for i in phase["parameterIndices"]}
            for sign in (1, -1):
                candidate = list(state["center"])
                for i, value in noise.items():
                    candidate[i] = max(self.bounds[i][0], min(self.bounds[i][1], candidate[i] + sign*self.sigma*self.search_scales[i]*value))
                self._job(state,"search",candidate,self._seed(state,"search",pair),pair,sign)

    def _assignment_timeout(self, job, policy):
        duration = (sum(trial["durationSeconds"] for trial in job["calibrationTrials"])
                    if job["mode"] == "decoder-fit" else job["durationSeconds"])
        return max(policy["minimumTimeoutSeconds"],
                   policy["trialTimeoutSeconds"] * duration / policy["referenceDurationSeconds"])

    def set_scheduling(self, *, search_pairs, trial_timeout_seconds):
        """Explicit operator migration, never invoked by startup or public HTTP.

        Leases predating this feature receive one grace window because their
        original assignment time was not recorded. Retrying cannot extend it.
        """
        policy = scheduling_policy(search_pairs, trial_timeout_seconds)
        with self.transaction():
            state = self._state(); now = self.clock(); added = 0; migrated = 0
            changed = state.get("scheduling") != policy
            if state["status"] == "running" and state["role"] == "search":
                rows = self.db.execute("SELECT * FROM sequence_jobs WHERE generation=?", (state["generation"],)).fetchall()
                current = state.get("currentSearchPairs", self.pairs)
                identities = {(json.loads(row["payload"])["pairId"], json.loads(row["payload"])["sign"]) for row in rows}
                if identities != {(pair, sign) for pair in range(current) for sign in (-1, 1)} or len(rows) != 2*current:
                    raise APIError(503, "invalid_history", "Cannot expand an incomplete search batch")
                target = max(current, search_pairs)
                self._append_search_pairs(state, current, target)
                state["currentSearchPairs"] = target
                added = 2*(target-current)
            state["scheduling"] = policy
            for row in self.db.execute("SELECT * FROM sequence_jobs WHERE state='leased'").fetchall():
                job = json.loads(row["payload"])
                if "assignmentDeadlineAt" in job:
                    continue
                timeout = self._assignment_timeout(job, policy)
                job.update(assignmentStartedAt=now, assignmentDeadlineAt=now+timeout,
                           assignmentAttempt=1, assignmentTimeoutSeconds=timeout,
                           assignmentStartSource="migration-grace")
                self.db.execute("UPDATE sequence_jobs SET payload=?,expires=? WHERE job_id=?",
                                (canonical(job), min(row["expires"], now+timeout), row["job_id"]))
                migrated += 1
            if changed:
                state["schedulingRevision"] = state.get("schedulingRevision", 0)+1
                entry = dict(revision=state["schedulingRevision"], effectiveAt=now,
                             effectiveGeneration=state["generation"], effectivePhaseIndex=state["phaseIndex"],
                             effectiveRole=state["role"], policy=policy, addedJobs=added, migratedLeases=migrated,
                             activeSearchPairs=state.get("currentSearchPairs") if state["role"] == "search" else None)
                state["schedulingHistory"] = [*state.get("schedulingHistory", []), entry][-120:]
            self._save(state)
            return dict(scheduling=policy, addedJobs=added, migratedLeases=migrated,
                        schedulingRevision=state["schedulingRevision"],
                        generation=state["generation"], role=state["role"], configHash=self.config_hash)

    def _comparison(self, state, proposal):
        state["proposal"] = proposal; state["role"] = "compare"; state["generation"] += 1
        for i in range(self._phase(state)["validationCount"]):
            seed=self._seed(state,"compare",i)
            self._job(state,"baseline",state["center"],seed,i,-1)
            self._job(state,"candidate",proposal,seed,i,1)

    def _reject(self,state,reason):
        state["history"].append(dict(phaseId=self._phase(state)["id"],round=state["round"],accepted=False,reason=reason))
        state["history"] = state["history"][-120:]
        state["proposal"]=None;state["round"]+=1;state["generation"]+=1
        if state["round"]>=self._phase(state)["maxRounds"]:
            state["status"]="needs-review";state["reason"]=reason
        else:
            self._start_round(state)

    def _gate_stages(self, state):
        """Retest every previously passed physical task on fresh whole episodes."""
        stages = [self._phase(state)["stage"]]
        completed = {entry["phaseId"] for entry in state["completedPhases"]}
        for phase in self.sequence["phases"]:
            if phase["id"] in completed and phase["stage"] not in stages:
                stages.append(phase["stage"])
        return stages

    def _advance(self,state):
        rows=self.db.execute("SELECT * FROM sequence_jobs WHERE generation=? ORDER BY job_id",(state["generation"],)).fetchall()
        if not rows or any(r["state"]!="completed" for r in rows): return
        phase=self._phase(state)
        if state["role"]=="search":
            best=max(rows,key=lambda r:(r["score"],r["job_id"]))
            self._comparison(state,json.loads(best["payload"])["parameters"])
        elif state["role"]=="fit":
            result=json.loads(rows[0]["result"])
            if result.get("calibration",{}).get("passed") is not True or result.get("candidateParameters") is None:
                self._reject(state,"decoder-fit-did-not-pass-held-out-fit-check")
            else:self._comparison(state,result["candidateParameters"])
        elif state["role"]=="compare":
            baseline=[r for r in rows if r["role"]=="baseline"];candidate=[r for r in rows if r["role"]=="candidate"]
            delta=sum(r["score"] for r in candidate)/len(candidate)-sum(r["score"] for r in baseline)/len(baseline)
            successes=sum(json.loads(r["result"])["metrics"]["success"] for r in candidate)
            if delta+1e-12<phase["minimumMeanImprovement"]:
                self._reject(state,"candidate-did-not-pass-paired-comparison")
            else:
                state["comparison"]={"meanImprovement":delta,"successes":successes,"episodes":len(candidate)}
                state["role"]="gate";state["generation"]+=1
                for stage_index, stage in enumerate(self._gate_stages(state)):
                    for i in range(phase["validationCount"]):
                        seed=self._seed(state,"gate-"+stage,i)
                        pair=stage_index*phase["validationCount"]+i
                        if stage==phase["stage"]:
                            self._job(state,"gate-baseline",state["center"],seed,pair,-1,stage=stage)
                        self._job(state,"gate-candidate",state["proposal"],seed,pair,1,stage=stage)
        elif state["role"]=="gate":
            stage_results=[]
            for stage in self._gate_stages(state):
                stage_rows=[r for r in rows if r["role"]=="gate-candidate" and json.loads(r["payload"])["stage"]==stage]
                if len(stage_rows)!=phase["validationCount"]:
                    raise APIError(503,"invalid_history","Fresh candidate batch is incomplete")
                stage_results.append(dict(stage=stage,episodes=len(stage_rows),
                    successes=sum(json.loads(r["result"])["metrics"]["success"] for r in stage_rows),
                    meanReturn=sum(r["score"] for r in stage_rows)/len(stage_rows)))
            baseline=[r for r in rows if r["role"]=="gate-baseline" and json.loads(r["payload"])["stage"]==phase["stage"]]
            if len(baseline)!=phase["validationCount"]:
                raise APIError(503,"invalid_history","Fresh baseline batch is incomplete")
            current=stage_results[0]
            baseline_mean=sum(r["score"] for r in baseline)/len(baseline)
            fresh_comparison=dict(meanImprovement=current["meanReturn"]-baseline_mean,
                                  baselineMeanReturn=baseline_mean,candidateMeanReturn=current["meanReturn"],
                                  episodes=current["episodes"])
            completed_ids={entry["phaseId"] for entry in state["completedPhases"]}
            previous_stages={p["stage"] for p in self.sequence["phases"] if p["id"] in completed_ids}
            if fresh_comparison["meanImprovement"]+1e-12<phase["minimumMeanImprovement"]:
                self._reject(state,"candidate-did-not-pass-fresh-paired-comparison")
            elif any(s["successes"]<phase["minimumSuccesses"] for s in stage_results if s["stage"] in previous_stages):
                self._reject(state,"candidate-did-not-retain-completed-stage")
            else:
                phase_passed=current["successes"]>=phase["minimumSuccesses"]
                changed=sum(a!=b for a,b in zip(state["center"],state["proposal"]))
                update_l2=math.sqrt(sum((a-b)**2 for a,b in zip(state["center"],state["proposal"])))
                state["center"]=state["proposal"];state["proposal"]=None;state["changedParameters"]+=changed
                state["lastParameterUpdate"]=dict(generation=state["generation"],changedCount=changed,
                    parameterCount=len(self.names),updateL2=update_l2,phaseId=phase["id"],phasePassed=phase_passed)
                evidence=dict(phaseId=phase["id"],round=state["round"],accepted=True,changedParameters=changed,
                              phasePassed=phase_passed,successes=current["successes"],episodes=current["episodes"],
                              evaluatedEpisodes=len(rows),stageResults=stage_results,comparison=state["comparison"],
                              freshComparison=fresh_comparison,
                              interpretation="Improved on fresh paired simulation episodes; task completion is separate and anonymous execution and physiology remain unverified")
                state["history"].append(evidence);state["history"]=state["history"][-120:]
                if phase_passed:
                    state["completedPhases"].append(evidence)
                    if state["phaseIndex"]+1==len(self.sequence["phases"]):
                        state["status"]="complete";state["role"]="complete"
                else:
                    state["round"]+=1
                    if state["round"]>=phase["maxRounds"]:
                        state["status"]="needs-review";state["reason"]="improved-candidate-has-not-completed-stage"
                self.db.execute("INSERT INTO sequence_checkpoints VALUES(?,?)",(state["generation"],canonical(self._checkpoint(state))))
                state["generation"]+=1
                if phase_passed:
                    state["phaseIndex"]+=1;state["round"]=0
                if state["status"]=="running":self._start_round(state)

    def _job_json(self,row):
        job=json.loads(row["payload"])
        return {**job,"leaseToken":row["token"],"leaseExpiresAt":row["expires"]}

    def _leased(self,payload,contributor):
        job_id,token=self._lease_credentials(payload)
        row=self.db.execute("SELECT * FROM sequence_jobs WHERE job_id=?",(job_id,)).fetchone()
        return self._verify_assignment(row,contributor,token)

    @staticmethod
    def _deadline(row):
        return json.loads(row["payload"]).get("assignmentDeadlineAt")

    def _expired(self, row, now):
        deadline = self._deadline(row)
        return row["expires"] <= now or (deadline is not None and deadline <= now)

    def _require_active(self, row):
        now = self.clock(); deadline = self._deadline(row)
        if row["state"] == "leased" and deadline is not None and deadline <= now:
            raise APIError(410, "assignment_timeout", "Assignment exceeded its absolute runtime; request new work")
        if row["state"] != "leased" or row["expires"] <= now:
            raise APIError(410, "lease_expired", "Assignment expired")

    def lease(self,payload):
        contributor=self._identity(payload)
        if "parameters" in payload:raise APIError(400,"invalid_request","Coordinator assigns parameters")
        with self.transaction():
            state=self._state();now=self.clock()
            for row in self.db.execute("SELECT * FROM sequence_jobs WHERE state='leased' AND contributor=?",(contributor,)).fetchall():
                if not self._expired(row, now):return {"job":self._job_json(row),"retry":True}
            if state["status"]!="running":return {"job":None,"status":state["status"],"waitMs":5000}
            row = None
            for candidate in self.db.execute("SELECT * FROM sequence_jobs WHERE generation=? AND state IN ('pending','leased') ORDER BY job_id",(state["generation"],)).fetchall():
                if candidate["state"] == "leased":
                    if not self._expired(candidate, now):continue
                    deadline = self._deadline(candidate)
                    if (deadline is not None and deadline <= now and candidate["contributor"] == contributor
                            and now < deadline+REASSIGNMENT_COOLDOWN_SECONDS):continue
                row = candidate; break
            if not row:return {"job":None,"status":"waiting_for_results","waitMs":2000}
            job = json.loads(row["payload"]); expires = now+self.lease_seconds
            if state.get("scheduling") is not None:
                timeout = self._assignment_timeout(job, validate_scheduling(state["scheduling"]))
                job.update(assignmentStartedAt=now, assignmentDeadlineAt=now+timeout,
                           assignmentAttempt=job.get("assignmentAttempt", 0)+1,
                           assignmentTimeoutSeconds=timeout, assignmentStartSource="assigned")
                expires = min(expires, job["assignmentDeadlineAt"])
            self.db.execute("UPDATE sequence_jobs SET state='leased',contributor=?,token=?,expires=?,payload=? WHERE job_id=?",(contributor,secrets.token_urlsafe(24),expires,canonical(job),row["job_id"]))
            return {"job":self._job_json(self.db.execute("SELECT * FROM sequence_jobs WHERE job_id=?",(row["job_id"],)).fetchone()),"retry":False}

    def _validate_fit(self,row,payload):
        job=self._job_json(row);provenance=payload["provenance"]
        expected={**job,"dtMs":self.dt_ms,"bodyBlockMs":self.body_block_ms,"bodyBackend":"mujoco-wasm"}
        for key in ("configHash","modelFingerprint","seed","stage","parametersHash","generation",
                    "sign","pairId","durationSeconds","dtMs","bodyBlockMs","bodyBackend"):
            if provenance.get(key)!=expected[key] or isinstance(provenance.get(key),bool):
                raise APIError(409,"provenance_mismatch","Calibration identity mismatch: "+key)
            if key in payload and (payload[key]!=expected[key] or isinstance(payload[key],bool)):
                raise APIError(409,"provenance_mismatch","Calibration result identity mismatch: "+key)
        version_keys=[key for key in ("envVersion","environmentVersion") if key in provenance]
        if not version_keys or any(provenance[key]!=self.environment_version for key in version_keys):
            raise APIError(409,"provenance_mismatch","Calibration environment version mismatch")
        for applied in [provenance.get("parameters")]+([payload["parameters"]] if "parameters" in payload else []):
            if (not isinstance(applied,list) or applied!=job["parameters"]
                    or any(isinstance(value,bool) for value in applied)):
                raise APIError(409,"provenance_mismatch","Calibration starting parameters differ")
        if (provenance.get("backend")!="wasm" or provenance.get("neuralEngine")!="wasm"
                or provenance.get("wasmExecution")!=self.acceptance_config["nativeExecution"]
                or "nativeWebGPU" in provenance):
            raise APIError(409,"provenance_mismatch","Fit requires pinned WASM execution")
        calibration=payload.get("calibration")
        if not isinstance(calibration,dict) or type(calibration.get("passed")) is not bool:
            raise APIError(422,"invalid_calibration","Calibration needs an explicit held-out fit result")
        if (calibration.get("teacherUsedForEvaluation") is not False
                or calibration.get("teacherOnlyDuringDemonstrations") is not True
                or calibration.get("autonomousEvaluationPerformed") is not False):
            raise APIError(422,"invalid_calibration","Fit is demonstration-based; autonomous evaluation must remain separate")
        for key, split in (("trainingTrials","train"),("validationTrials","validation")):
            assigned=sum(trial["split"]==split for trial in job["calibrationTrials"])
            count=calibration.get(key)
            if type(count) is not int or not 0<=count<=assigned or (calibration["passed"] and count!=assigned):
                raise APIError(422,"invalid_calibration","Fit trial counts differ from the assigned demonstrations")
        candidate=payload.get("candidateParameters")
        if not calibration["passed"]:
            if candidate is not None:raise APIError(422,"invalid_calibration","Failed fit cannot submit candidate weights")
            return
        if not isinstance(candidate,list) or len(candidate)!=len(self.names):raise APIError(422,"invalid_calibration","Incomplete candidate vector")
        for i,(value,(lo,hi)) in enumerate(zip(candidate,self.bounds)):finite_number(value,lo,hi,"candidate parameter")
        if candidate[:self.sensory_count]!=job["parameters"][:self.sensory_count]:raise APIError(422,"invalid_calibration","Decoder fit changed sensory parameters")

    def result(self,payload):
        contributor,score,result_hash=self._result_inputs(payload)
        with self.transaction():
            state=self._state();row=self._leased(payload,contributor)
            if row["state"]=="completed":
                if row["result_hash"]!=result_hash:raise APIError(409,"result_conflict","Different result already accepted")
                return {"accepted":True,"duplicate":True,"status":"unverified"}
            self._require_active(row)
            job=self._job_json(row)
            if job["mode"]=="decoder-fit":self._validate_fit(row,payload)
            else:
                old=self.duration;self.duration=job["durationSeconds"]
                try:self._validate_result_provenance(row,payload)
                finally:self.duration=old
            self.db.execute("UPDATE sequence_jobs SET state='completed',result_hash=?,result=?,score=?,completed=? WHERE job_id=?",(result_hash,canonical(payload),score,self.clock(),row["job_id"]))
            state["acceptedResults"]+=1;previous=state["generation"];self._advance(state);self._save(state)
            return {"accepted":True,"duplicate":False,"status":"unverified","generation":previous,"nextGenerationCreated":state["generation"]!=previous}

    def heartbeat(self,payload):
        contributor=self._identity(payload)
        with self.transaction():
            row=self._leased(payload,contributor)
            self._require_active(row)
            expires=self.clock()+self.lease_seconds
            deadline = self._deadline(row)
            if deadline is not None:expires = min(expires, deadline)
            self.db.execute("UPDATE sequence_jobs SET expires=? WHERE job_id=?",(expires,row["job_id"]))
            return {"renewed":True,"jobId":row["job_id"],"leaseExpiresAt":expires,
                    **({"assignmentDeadlineAt":deadline} if deadline is not None else {})}

    def release(self,payload):
        contributor=self._identity(payload)
        with self.transaction():
            row=self._leased(payload,contributor)
            if row["state"]=="completed":raise APIError(409,"already_completed","Completed job cannot be released")
            if row["state"]=="pending":return {"released":True,"duplicate":True}
            self._require_active(row)
            self.db.execute("UPDATE sequence_jobs SET state='pending',expires=NULL WHERE job_id=?",(row["job_id"],))
            return {"released":True,"duplicate":False}

    def _checkpoint(self,state):
        phase=self.sequence["phases"][min(state["phaseIndex"],len(self.sequence["phases"])-1)]
        return dict(schemaVersion=1,algorithm=self.algorithm,modelFingerprint=self.model_fingerprint,configHash=self.config_hash,
                    parameterNames=self.names,parameters=state["center"],generation=state["generation"],stage=phase["stage"],
                    status="unverified",biologicalSuccessValidated=False,sequenceStatus=state["status"],
                    completedPhases=state["completedPhases"],changedParameters=state["changedParameters"],
                    lastParameterUpdate=state.get("lastParameterUpdate"),
                    **(self._scheduling_summary(state)))

    @staticmethod
    def _scheduling_summary(state):
        if "scheduling" not in state:return {}
        return dict(scheduling=state["scheduling"], schedulingRevision=state["schedulingRevision"],
                    schedulingHistory=state["schedulingHistory"],
                    currentSearchPairs=state.get("currentSearchPairs") if state["role"] == "search" else None)

    def checkpoint(self):
        with self.transaction():return self._checkpoint(self._state())

    def status(self):
        with self.transaction():
            state=self._state();rows=self.db.execute("SELECT * FROM sequence_jobs WHERE generation=?",(state["generation"],)).fetchall();now=self.clock()
            counts={"pending":0,"leased":0,"completed":0}
            for row in rows:counts["pending" if row["state"]=="leased" and self._expired(row,now) else row["state"]]+=1
            recent=self.db.execute("SELECT * FROM sequence_jobs WHERE state='completed' ORDER BY completed DESC,job_id DESC LIMIT 120").fetchall()
            trials=[]
            for row in reversed(recent):
                job=json.loads(row["payload"]);result=json.loads(row["result"])
                trials.append(dict(jobId=row["job_id"],generation=job["generation"],stage=job["stage"],phaseId=job["phaseId"],
                                   mode=job["mode"],reason=result.get("metrics",{}).get("reason"),
                                   return_=row["score"],success=result.get("metrics",{}).get("success",False),completedAt=row["completed"]))
                trials[-1]["return"]=trials[-1].pop("return_")
            curriculum=[{**p,"status":"complete" if i<state["phaseIndex"] else state["status"] if i==state["phaseIndex"] else "pending"} for i,p in enumerate(self.sequence["phases"])]
            return dict(schemaVersion=2,stage=self.stage,durationSeconds=self.duration,generation=state["generation"],
                        modelFingerprint=self.model_fingerprint,configHash=self.config_hash,config=self.config,jobs=counts,
                        completedJobs=counts["completed"],totalJobs=len(rows),acceptedResults=state["acceptedResults"],
                        contributors=self.db.execute("SELECT COUNT(DISTINCT contributor) FROM sequence_jobs WHERE state='completed'").fetchone()[0],
                        checkpointStatus="unverified",checkpoint=self._checkpoint(state),recentTrials=trials,leaseSeconds=self.lease_seconds,
                        lastParameterUpdate=state.get("lastParameterUpdate"),
                        sequence={"status":state["status"],"phaseIndex":state["phaseIndex"],"phaseId":curriculum[min(state["phaseIndex"],len(curriculum)-1)]["id"],"role":state["role"],"round":state["round"],"reason":state.get("reason"),"history":state["history"]},
                        curriculum=curriculum,automaticCurriculumPromotion=True,biologicalSuccessValidated=False,
                        **self._scheduling_summary(state))

    def close(self):
        with self.lock:self.db.close()
