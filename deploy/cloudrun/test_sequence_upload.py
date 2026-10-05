"""Real loopback HTTP with synthetic fit telemetry; no neural/physical run.

The browser caps the complete fitted summary at 64 KiB. Exercise that boundary
with full precision 696-value vectors and the actual Cloud Run request limit,
including the durable coordinator's candidate/provenance validation.
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
from test_sequential_training import fixture, phase

SPEC = importlib.util.spec_from_file_location("sequence_upload_server", Path(__file__).with_name("server.py"))
server_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server_module)


def encoded(value):
    # JavaScript JSON.stringify produces compact UTF-8, not Python's default
    # whitespace-heavy representation or escaped Unicode.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def full_fit_payload(config, coordinator, job, *, contributor="fit-contributor"):
    candidate = list(job["parameters"])
    candidate[24:] = [-.12345678901234566] * 672
    output_names = ["power_left", "power_right", *[f"steer_{side}_{axis}" for side in ("left", "right") for axis in ("roll", "pitch", "yaw")]]
    diagnostics = {"outputs": [dict(name=name, rawRmse=.012345678901234567,
                      targetMean=.12345678901234566, targetStandardDeviation=.012345678901234567,
                      samples=55000) for name in output_names], "mixedUnitRawRmse": .012345678901234567}
    fit = dict(schemaVersion=1, kind="motor-decoder-calibration-candidate", status="awaiting-autonomous-validation",
        parameters=candidate[24:], metrics=dict(changedParameterCount=672, updateL2=.12345678901234566,
            before=dict(train=diagnostics, validation=diagnostics), after=dict(train=diagnostics, validation=diagnostics),
            fits=[dict(output=name, sweeps=2000, converged=True, maxCoordinateChange=1e-10,
                projectedGradient=1e-10, objectiveBefore=.12345678901234566,
                objectiveAfter=.012345678901234567, unexcitedParameterCount=0) for name in output_names]),
        trials=[{**trial, "success": True, "simSeconds": 5, "return": 0, "reason": "recovery_success",
                 "teacherIntervention": True, "collection": {"samples": 27500, "scoredSamples": 25000,
                    "warmupSamples": 2500, "lastSampleTimeSeconds": 5.4998}} for trial in job["calibrationTrials"]],
        provenance=dict(configHash=job["configHash"], modelFingerprint=job["modelFingerprint"],
            teacherCalibrationSha256="d"*64, priorParametersSha256="e"*64, statisticsSha256="f"*64,
            teacherOnlyDuringDemonstrations=True, containsPrivilegedDecoderInputs=False),
        transportBoundaryPadding=[])
    # Real sufficient-statistics arrays stay in the worker. Reserve every byte
    # permitted by its summary cap so future bounded diagnostics still upload.
    padding = fit["transportBoundaryPadding"]
    while 65535-len(encoded(fit)) > 4003:
        padding.append("x"*4000)
    remaining = 65535-len(encoded(fit))
    if remaining >= 2+bool(padding):
        padding.append("x"*(remaining-2-bool(padding)))
    elif remaining:
        padding[-1] += "x"*remaining
    if len(encoded(fit)) != 65535:
        raise AssertionError("Fit summary boundary fixture is invalid")
    provenance = {key: copy.deepcopy(job[key]) for key in ("configHash", "modelFingerprint", "parametersHash",
                    "seed", "stage", "durationSeconds", "sign", "pairId", "generation", "parameters")}
    provenance.update(environmentVersion=config["environmentVersion"], dtMs=.5, bodyBlockMs=2,
        bodyBackend="mujoco-wasm", backend="wasm", neuralEngine="wasm",
        parameterContract=config["parameterContract"], mode="decoder-fit",
        wasmExecution=copy.deepcopy(coordinator.acceptance_config["nativeExecution"]))
    return dict(contributorId=contributor, configHash=job["configHash"], modelFingerprint=job["modelFingerprint"],
        jobId=job["jobId"], leaseToken=job["leaseToken"], objective=0,
        metrics=dict(success=False, terminated=False, cancelled=False, simSeconds=15, steps=7500, reason="fit_candidate"),
        provenance=provenance, candidateParameters=candidate,
        calibration=dict(passed=True, trainingTrials=2, validationTrials=1,
            assignedTrainingTrials=2, assignedValidationTrials=1, teacherUsedForEvaluation=False,
            teacherOnlyDuringDemonstrations=True, teacherUsedForDemonstrations=True,
            autonomousEvaluationPerformed=False, teacherCalibrationSha256="d"*64,
            interpretation="Synthetic upload protocol fixture, not fitted flight evidence", fit=fit))


class SequentialUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = fixture([phase("fit", list(range(24, 696)), kind="decoder-fit", stage="recovery")])
        self.config["contribution"]["maxRequestBytes"] = 262144
        for parameter in self.config["parameters"]:
            parameter["initial"] = .12345678901234566
        self.coordinator = SequentialTrainingCoordinator(":memory:", self.config)
        public = self.root / "public"
        public.mkdir()
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(schemaVersion=1, configHash=self.coordinator.config_hash,
                            modelFingerprint=self.coordinator.model_fingerprint, files={})))
        self.server = server_module.make_server(self.coordinator, public, manifest, port=0, max_body_bytes=262144)
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
            connection.request("POST", "/api/training/"+endpoint, body=encoded(payload),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def assigned_fit(self):
        status, response = self.post("lease", dict(contributorId="fit-contributor",
            modelFingerprint=self.coordinator.model_fingerprint, configHash=self.coordinator.config_hash))
        self.assertEqual(status, 200)
        job = response["job"]
        self.assertEqual(job["mode"], "decoder-fit")
        return job

    def test_full_fit_summary_and_both_vectors_upload_and_retry_idempotently(self):
        job = self.assigned_fit()
        payload = full_fit_payload(self.config, self.coordinator, job)
        self.assertEqual(len(encoded(payload["calibration"]["fit"])), 65535)
        self.assertEqual(len(payload["provenance"]["parameters"]), 696)
        self.assertEqual(len(payload["candidateParameters"]), 696)
        self.assertLess(len(encoded(payload)), 110000)
        self.assertLess(len(encoded(payload)), self.config["contribution"]["maxRequestBytes"])
        status, accepted = self.post("result", payload)
        self.assertEqual(status, 200)
        self.assertTrue(accepted["accepted"])
        self.assertFalse(accepted["duplicate"])
        self.assertEqual(self.coordinator.status()["sequence"]["role"], "compare")
        self.assertEqual(self.coordinator.checkpoint()["parameters"], job["parameters"],
                         "Uploaded fit remains a proposal until autonomous gates pass")
        status, retried = self.post("result", payload)
        self.assertEqual(status, 200)
        self.assertTrue(retried["duplicate"])
        self.assertEqual(self.coordinator.status()["acceptedResults"], 1)

    def test_oversized_fit_returns_413_without_losing_the_assignment(self):
        job = self.assigned_fit()
        payload = full_fit_payload(self.config, self.coordinator, job)
        oversized = {**payload, "oversizeFixture": "x"*262144}
        status, error = self.post("result", oversized)
        self.assertEqual(status, 413)
        self.assertEqual(error["error"], "body_too_large")
        self.assertEqual(self.coordinator.status()["acceptedResults"], 0)
        status, accepted = self.post("result", payload)
        self.assertEqual(status, 200)
        self.assertTrue(accepted["accepted"])


if __name__ == "__main__":
    unittest.main()
