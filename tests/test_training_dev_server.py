"""Temporary HTTP/SQLite infrastructure fixtures; no simulated behavior claims."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("training_dev_server", ROOT / "scripts/serve-training-dev.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class DevelopmentServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.config_bytes = (ROOT / "web/training/config.json").read_bytes()
        self.config = json.loads(self.config_bytes)
        self.config_hash = hashlib.sha256(self.config_bytes).hexdigest()
        self.coordinator = module.training_coordinator.TrainingCoordinator(
            Path(self.temporary.name) / "fixture.sqlite3", self.config, self.config_hash)
        self.server = module.make_server(self.coordinator, self.config_bytes, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.coordinator.close()
        self.temporary.cleanup()

    def request(self, path, *, method="GET", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_original_page_has_only_explicit_hidden_same_origin_marker(self):
        status, headers, body = self.request("/train.html")
        self.assertEqual(status, 200)
        original = (ROOT / "web/train.html").read_text()
        self.assertEqual(body.decode(), original.replace("</head>", module.DEVELOPMENT_META + "\n</head>"))
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["Cross-Origin-Embedder-Policy"], "require-corp")
        self.assertEqual(self.request("/training/config.json")[2], self.config_bytes)
        self.assertEqual(self.request("/training/worker.js")[0], 200)

    def test_original_coordinator_assigns_real_config_jobs_on_same_origin(self):
        status, _, body = self.request("/api/training/checkpoint")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["configHash"], self.config_hash)
        identity = {"contributorId": "server-infrastructure-fixture", "configHash": self.config_hash,
                    "modelFingerprint": self.config["modelFingerprint"]}
        status, _, body = self.request("/api/training/lease", method="POST", body=json.dumps(identity),
            headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{self.server.server_port}"})
        self.assertEqual(status, 200)
        job = json.loads(body)["job"]
        self.assertEqual(job["stage"], self.config["stage"])
        self.assertEqual(job["durationSeconds"], self.config["durationSeconds"])
        self.assertEqual(job["configHash"], self.config_hash)

    def test_remote_origin_and_private_paths_are_rejected(self):
        self.assertEqual(self.request("/api/training/status", headers={"Origin": "https://outside.example"})[0], 403)
        self.assertEqual(self.request("/train.html", headers={"Host": "outside.example"})[0], 403)
        for path in ("/%2e%2e/.git/config", "/body-model/../../.git/config", "/reports/training-flight-active/coordinator.sqlite3"):
            self.assertEqual(self.request(path)[0], 404)
        self.assertEqual(self.request("/training/")[0], 403)

    def test_default_model_assets_are_served_from_the_checkout(self):
        for name in ("flybody-mujoco.xml", "flybody-mujoco.json"):
            with self.subTest(name=name):
                status, _, body = self.request("/body-model/" + name)
                self.assertEqual(status, 200)
                self.assertEqual(body, (ROOT / "models" / name).read_bytes())


class DevelopmentBundleTests(unittest.TestCase):
    XML_URL = "/body-model/flybody-mujoco.xml"
    METADATA_URL = "/body-model/flybody-mujoco.json"
    request = DevelopmentServerTests.request

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="training-bundle-fixture-", dir=ROOT / "reports")
        self.addCleanup(self.temporary.cleanup)
        self.bundle_path = Path(self.temporary.name) / "bundle.json"
        self.config = json.loads((ROOT / "web/training/config.json").read_bytes())
        self.config["notes"] = [*self.config.get("notes", []), "Temporary HTTP fixture: α"]
        xml = '<mujoco model="temporary-bundle"><!-- UTF-8: α --><worldbody/></mujoco>\n'
        metadata = {"xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "fixture": True}
        self.bundle = {
            "schemaVersion": 1,
            "kind": "flight-development-bundle",
            "assets": {self.XML_URL: xml, self.METADATA_URL: json.dumps(metadata) + "\n"},
        }
        self.repin_assets()

    def serialize_config(self):
        # Whitespace and UTF-8 bytes matter to the coordinator identity.
        self.bundle["configText"] = json.dumps(self.config, indent=4, ensure_ascii=False) + "\n\n"
        self.bundle["configHash"] = hashlib.sha256(self.bundle["configText"].encode()).hexdigest()
        self.bundle["modelFingerprint"] = self.config["modelFingerprint"]

    def repin_assets(self):
        for url, content in self.bundle["assets"].items():
            self.config["assets"][url] = hashlib.sha256(content.encode()).hexdigest()
        manifest = "".join(f"{url}:{digest}\n" for url, digest in sorted(self.config["assets"].items()))
        self.config["modelFingerprint"] = hashlib.sha256(manifest.encode()).hexdigest()
        self.serialize_config()

    def write_bundle(self, bundle=None, path=None):
        target = path or self.bundle_path
        target.write_text(json.dumps(self.bundle if bundle is None else bundle, ensure_ascii=False), encoding="utf-8")
        return target

    def coordinator(self, config_bytes):
        coordinator = module.training_coordinator.TrainingCoordinator(
            Path(self.temporary.name) / "fixture.sqlite3", json.loads(config_bytes),
            hashlib.sha256(config_bytes).hexdigest())
        self.addCleanup(coordinator.close)
        return coordinator

    def test_loader_preserves_exact_config_and_asset_bytes(self):
        config_bytes, overrides = module.load_bundle(self.write_bundle())
        self.assertEqual(config_bytes, self.bundle["configText"].encode())
        self.assertEqual(overrides, {url: value.encode() for url, value in self.bundle["assets"].items()})
        self.assertTrue(all(isinstance(value, bytes) for value in overrides.values()))

    def test_optional_top_level_identity_fields_may_be_omitted(self):
        del self.bundle["configHash"]
        del self.bundle["modelFingerprint"]
        config_bytes, _ = module.load_bundle(self.write_bundle())
        self.assertEqual(config_bytes, self.bundle["configText"].encode())

    def test_rejects_wrong_schema_kind_and_override_shape(self):
        variants = {
            "schema": {**self.bundle, "schemaVersion": 2},
            "kind": {**self.bundle, "kind": "node-native-motor-capture"},
            "config_not_text": {**self.bundle, "configText": self.config},
            "extra_path": {**self.bundle, "assets": {**self.bundle["assets"], "/training/worker.js": "replaced"}},
            "missing_path": {**self.bundle, "assets": {self.XML_URL: self.bundle["assets"][self.XML_URL]}},
            "non_string_asset": {**self.bundle, "assets": {**self.bundle["assets"], self.XML_URL: {"xml": "invalid"}}},
        }
        for name, bundle in variants.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.load_bundle(self.write_bundle(bundle))

    def test_rejects_asset_tampering(self):
        for url in self.bundle["assets"]:
            tampered = {**self.bundle, "assets": {**self.bundle["assets"], url: self.bundle["assets"][url] + " "}}
            with self.subTest(url=url), self.assertRaises(ValueError):
                module.load_bundle(self.write_bundle(tampered))

    def test_rejects_metadata_xml_mismatch_even_with_recomputed_manifest(self):
        metadata = json.loads(self.bundle["assets"][self.METADATA_URL])
        metadata["xml_sha256"] = "0" * 64
        self.bundle["assets"][self.METADATA_URL] = json.dumps(metadata)
        self.repin_assets()
        with self.assertRaises(ValueError):
            module.load_bundle(self.write_bundle())

    def test_rejects_incorrect_model_fingerprint(self):
        self.config["modelFingerprint"] = "0" * 64
        self.serialize_config()
        with self.assertRaises(ValueError):
            module.load_bundle(self.write_bundle())

    def test_rejects_incorrect_top_level_identities(self):
        for field in ("configHash", "modelFingerprint"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.load_bundle(self.write_bundle({**self.bundle, field: "0" * 64}))

    def test_rejects_config_text_tampering_without_hash_update(self):
        self.bundle["configText"] += " "
        with self.assertRaises(ValueError):
            module.load_bundle(self.write_bundle())

    def test_rejects_noncanonical_manifest_order(self):
        self.config["assets"] = dict(reversed(list(self.config["assets"].items())))
        self.serialize_config()
        with self.assertRaises(ValueError):
            module.load_bundle(self.write_bundle())

    def test_rejects_bundle_outside_reports_and_symlink_escape(self):
        with tempfile.TemporaryDirectory(prefix="training-bundle-outside-") as directory:
            outside = self.write_bundle(path=Path(directory) / "bundle.json")
            escaped = Path(self.temporary.name) / "escaped.json"
            escaped.symlink_to(outside)
            for path in (outside, escaped):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    module.load_bundle(path)

    def test_served_bundle_and_coordinator_share_exact_identity(self):
        config_bytes, overrides = module.load_bundle(self.write_bundle())
        coordinator = self.coordinator(config_bytes)
        self.server = module.make_server(coordinator, config_bytes, port=0, asset_overrides=overrides)
        self.addCleanup(self.server.server_close)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        # Serving takes a snapshot; later caller mutations cannot change pins.
        overrides[self.XML_URL] = b"changed after server construction"

        status, headers, served_config = self.request("/training/config.json")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"].split(";")[0], "application/json")
        self.assertEqual(served_config, config_bytes)
        config = json.loads(served_config)
        for url, text in self.bundle["assets"].items():
            with self.subTest(url=url):
                status, headers, body = self.request(url)
                self.assertEqual(status, 200)
                self.assertEqual(body, text.encode())
                self.assertEqual(hashlib.sha256(body).hexdigest(), config["assets"][url])
                expected_types = {"application/xml", "text/xml"} if url.endswith(".xml") else {"application/json"}
                self.assertIn(headers["Content-Type"].split(";")[0], expected_types)
                self.assertEqual(int(headers["Content-Length"]), len(body))
                self.assertEqual(headers["Cross-Origin-Embedder-Policy"], "require-corp")
                self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
                head_status, head_headers, head_body = self.request(url, method="HEAD")
                self.assertEqual(head_status, 200)
                self.assertEqual(int(head_headers["Content-Length"]), len(body))
                self.assertEqual(head_body, b"")
                self.assertEqual(self.request(url, headers={"Origin": "https://outside.example"})[0], 403)
        metadata = json.loads(self.request(self.METADATA_URL)[2])
        self.assertEqual(metadata["xml_sha256"], hashlib.sha256(self.request(self.XML_URL)[2]).hexdigest())
        manifest = "".join(f"{url}:{digest}\n" for url, digest in sorted(config["assets"].items()))
        self.assertEqual(hashlib.sha256(manifest.encode()).hexdigest(), config["modelFingerprint"])
        self.assertEqual(self.request("/training/worker.js")[2], (ROOT / "web/training/worker.js").read_bytes())

        checkpoint = json.loads(self.request("/api/training/checkpoint")[2])
        self.assertEqual(checkpoint["configHash"], hashlib.sha256(served_config).hexdigest())
        self.assertEqual(checkpoint["modelFingerprint"], config["modelFingerprint"])
        identity = {"contributorId": "bundle-http-fixture", "configHash": checkpoint["configHash"],
                    "modelFingerprint": config["modelFingerprint"]}
        status, _, body = self.request("/api/training/lease", method="POST", body=json.dumps(identity),
            headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{self.server.server_port}"})
        self.assertEqual(status, 200)
        job = json.loads(body)["job"]
        self.assertEqual(job["configHash"], identity["configHash"])
        self.assertEqual(job["modelFingerprint"], identity["modelFingerprint"])

    def test_server_rejects_configuration_hash_mismatch(self):
        config_bytes, overrides = module.load_bundle(self.write_bundle())
        coordinator = self.coordinator(config_bytes)
        with self.assertRaises(ValueError):
            module.make_server(coordinator, config_bytes + b" ", port=0, asset_overrides=overrides)

    def test_server_revalidates_override_bytes(self):
        config_bytes, overrides = module.load_bundle(self.write_bundle())
        coordinator = self.coordinator(config_bytes)
        overrides[self.XML_URL] += b" "
        with self.assertRaises(ValueError):
            module.make_server(coordinator, config_bytes, port=0, asset_overrides=overrides)


if __name__ == "__main__":
    unittest.main()
