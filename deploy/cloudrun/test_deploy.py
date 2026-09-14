"""Mocked deployment orchestration only; no gcloud process or HTTP request."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cloudrun_deploy", Path(__file__).with_name("deploy.py"))
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
IMAGE = "us-central1-docker.pkg.dev/flyheaven/fly-training/web@sha256:" + "a"*64
PRIMARY = "https://fly-training-example-uc.a.run.app"
PUBLIC = "https://flytrain.morisoba.moe"
OLD = "fly-training-previous-good"


class CloudFixture:
    def __init__(self):
        self.calls, self.requests, self.revisions = [], [], []
        self.traffic = [{"revisionName": OLD, "percent": 100}]
        self.tags, self.environments = {}, []
        self.aliases = [PRIMARY]
        self.initial_aliases = [PRIMARY]
        self.wrong_tag = False
        self.promotion_ignored = False
        self.traffic_result_list = False
        self.fetch_error = None
        self.mutate_status = lambda value: value
        self.mutate_checkpoint = lambda value: value
        self.config = {"modelFingerprint": "b"*64, "algorithm": "antithetic-evolution-strategies", "stage": "landing",
                       "parameters": [{"name": "fixture", "min": -1, "max": 1, "initial": 0}]}
        self.raw_config = json.dumps(self.config).encode()
        self.config_hash = hashlib.sha256(self.raw_config).hexdigest()

    def record(self, aliases=None):
        return {"metadata": {"annotations": {"run.googleapis.com/urls": json.dumps(self.aliases if aliases is None else aliases)}},
                "status": {"url": PRIMARY, "latestCreatedRevisionName": "fly-training-concurrent-other",
                           "latestReadyRevisionName": "fly-training-concurrent-other",
                           "traffic": self.traffic + [{"tag": tag, "revisionName": "wrong-revision" if self.wrong_tag else revision,
                                                       "url": f"https://{tag}---fly-training-example-uc.a.run.app"}
                                                      for tag, revision in self.tags.items()]}}

    def command(self, args):
        self.calls.append(args)
        if args[:3] == ["run", "services", "list"]:
            return json.dumps([self.record(self.initial_aliases)])
        if args[:2] == ["run", "deploy"]:
            assert "--no-traffic" in args
            suffix = next(value.split("=", 1)[1] for value in args if value.startswith("--revision-suffix="))
            tag = next(value.split("=", 1)[1] for value in args if value.startswith("--tag="))
            self.revisions.append(args[2] + "-" + suffix)
            self.tags[tag] = self.revisions[-1]
            environment = next(value.split("=", 1)[1] for value in args if value.startswith("--env-vars-file="))
            self.environments.append(json.loads(Path(environment).read_text()))
            return ""
        if args[:3] == ["run", "services", "describe"]:
            return json.dumps(self.record())
        if args[:3] == ["run", "services", "update-traffic"]:
            for argument in args:
                if argument.startswith("--remove-tags="):
                    self.tags.pop(argument.split("=", 1)[1], None)
                if argument.startswith("--to-revisions=") and not self.promotion_ignored:
                    revision, percent = argument.split("=", 1)[1].split("=")
                    self.traffic = [{"revisionName": revision, "percent": int(percent)}]
            return json.dumps(self.traffic if self.traffic_result_list else self.record())
        raise AssertionError("Unexpected cloud command " + repr(args))

    def checkpoint(self):
        return {"schemaVersion": 1, "algorithm": self.config["algorithm"], "configHash": self.config_hash,
                "modelFingerprint": self.config["modelFingerprint"], "stage": "landing", "generation": 4,
                "parameterNames": ["fixture"], "parameters": [.25]}

    def fetch(self, url):
        self.requests.append(url)
        if self.fetch_error:
            raise self.fetch_error
        if url.endswith("/training/config.json"):
            return self.raw_config
        if url.endswith("/api/training/status?compact=1"):
            checkpoint = self.checkpoint()
            return json.dumps(self.mutate_status({**checkpoint, "checkpoint": checkpoint})).encode()
        if url.endswith("/api/training/checkpoint"):
            return json.dumps(self.mutate_checkpoint(self.checkpoint())).encode()
        raise AssertionError("Unexpected HTTP request " + url)

    def run(self):
        with patch.object(deploy, "command", self.command), patch.object(deploy, "fetch_bytes", self.fetch):
            return deploy.deploy(IMAGE, "fixture-run")

    def promotions(self):
        return [call for call in self.calls if any(arg.startswith("--to-revisions=") for arg in call)]


class DeploymentTests(unittest.TestCase):
    def test_guarded_update_does_not_create_a_service_without_protected_old_traffic(self):
        with patch.object(deploy, "command", return_value="[]") as command:
            with self.assertRaisesRegex(ValueError, "existing Cloud Run service"):
                deploy.deploy(IMAGE, "fixture-run")
        self.assertEqual(command.call_count, 1)
        self.assertEqual(command.call_args.args[0][:3], ["run", "services", "list"])

    def test_pinned_old_traffic_promotes_only_the_exact_preflighted_revision(self):
        cloud = CloudFixture()
        result = cloud.run()
        self.assertEqual(result["revision"], cloud.revisions[-1])
        self.assertEqual(result["trafficPercent"], 100)
        self.assertEqual(result["configHash"], cloud.config_hash)
        self.assertEqual(cloud.traffic, [{"revisionName": result["revision"], "percent": 100}])
        self.assertEqual(len(cloud.promotions()), 1)
        self.assertIn("--to-revisions="+result["revision"]+"=100", cloud.promotions()[0])
        self.assertFalse(any("LATEST" in argument or "--to-latest" in argument for call in cloud.calls for argument in call))
        self.assertEqual(cloud.tags, {})
        self.assertEqual(len(cloud.requests), 3)
        self.assertTrue(all(url.startswith("https://verify-") for url in cloud.requests))
        self.assertFalse(any("/healthz" in url for url in cloud.requests))
        self.assertEqual(cloud.environments[0]["TRAINING_RUN_ID"], "fixture-run")
        self.assertEqual(cloud.environments[0]["ALLOWED_ORIGINS"], PUBLIC+","+PRIMARY)

    def test_failed_public_preflight_preserves_old_traffic_and_removes_only_our_tag(self):
        cloud = CloudFixture()
        cloud.tags["existing-observer"] = OLD
        cloud.fetch_error = OSError("Fixture endpoint is unavailable")
        with self.assertRaisesRegex(OSError, "unavailable"):
            cloud.run()
        self.assertEqual(cloud.promotions(), [])
        self.assertEqual(cloud.traffic, [{"revisionName": OLD, "percent": 100}])
        self.assertEqual(cloud.tags, {"existing-observer": OLD})

    def test_list_mutation_output_is_followed_by_independent_service_traffic_verification(self):
        cloud = CloudFixture()
        cloud.traffic_result_list = True
        result = cloud.run()
        self.assertEqual(result["revision"], cloud.revisions[-1])
        self.assertEqual(result["trafficPercent"], 100)
        self.assertEqual(cloud.calls[-1][:4], ["run", "services", "describe", "fly-training"])
        self.assertIn("--project=flyheaven", cloud.calls[-1])
        self.assertIn("--region=us-central1", cloud.calls[-1])
        self.assertEqual(cloud.tags, {})

    def test_wrong_status_identity_or_checkpoint_never_promotes(self):
        for mutation in ("status", "checkpoint", "values", "names"):
            cloud = CloudFixture()
            if mutation == "status":
                cloud.mutate_status = lambda value: {**value, "configHash": "c"*64}
            elif mutation == "checkpoint":
                cloud.mutate_checkpoint = lambda value: {**value, "modelFingerprint": "c"*64}
            elif mutation == "values":
                cloud.mutate_checkpoint = lambda value: {**value, "parameters": [True]}
            else:
                cloud.mutate_checkpoint = lambda value: {**value, "parameterNames": ["wrong"]}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cloud.run()
            self.assertEqual(cloud.promotions(), [])
            self.assertEqual(cloud.traffic, [{"revisionName": OLD, "percent": 100}])

    def test_new_service_alias_is_baked_into_another_zero_traffic_revision(self):
        cloud = CloudFixture()
        cloud.aliases.append("https://fly-training-123.us-central1.run.app")
        result = cloud.run()
        self.assertEqual(len(cloud.revisions), 2)
        self.assertNotEqual(cloud.revisions[0], cloud.revisions[1])
        self.assertEqual(result["revision"], cloud.revisions[1])
        self.assertIn(cloud.aliases[-1], cloud.environments[-1]["ALLOWED_ORIGINS"].split(","))
        self.assertEqual(len(cloud.promotions()), 1)

    def test_tag_pointing_at_another_revision_cannot_preflight_or_promote(self):
        cloud = CloudFixture()
        cloud.wrong_tag = True
        with self.assertRaisesRegex(ValueError, "exact deployed revision"):
            cloud.run()
        self.assertEqual(cloud.requests, [])
        self.assertEqual(cloud.promotions(), [])
        self.assertEqual(cloud.traffic, [{"revisionName": OLD, "percent": 100}])

    def test_success_is_not_reported_if_traffic_is_still_pinned_elsewhere(self):
        cloud = CloudFixture()
        cloud.promotion_ignored = True
        cloud.traffic_result_list = True
        with self.assertRaisesRegex(RuntimeError, "100% traffic"):
            cloud.run()

    def test_a_concurrent_generation_update_is_allowed_but_same_generation_vector_change_is_not(self):
        cloud = CloudFixture()
        cloud.mutate_checkpoint = lambda value: {**value, "generation": 5, "parameters": [.5]}
        self.assertEqual(cloud.run()["generation"], 5)
        for values in ({"generation": 3}, {"parameters": [.5]}):
            cloud = CloudFixture()
            cloud.mutate_checkpoint = lambda value: {**value, **values}
            with self.assertRaises(ValueError):
                cloud.run()
            self.assertEqual(cloud.promotions(), [])


if __name__ == "__main__":
    unittest.main()
