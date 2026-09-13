import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cloudrun_package", Path(__file__).with_name("package.py"))
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class PackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo, self.bundle = self.root / "repo", self.root / "bundle"
        self.bundle.mkdir()
        (self.repo / "web/training").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        for name in package.SERVER_FILES:
            (self.repo / "scripts" / name).write_text("# server\n")
        (self.repo / "requirements-cloudrun.txt").write_text("# dependencies\n")
        self.here = self.root / "deployment"
        self.here.mkdir()
        for name in ("server.py", "Dockerfile", "cloudbuild.yaml", ".gcloudignore"):
            (self.here / name).write_text("# " + name + "\n")
        self.files = []
        self.add("asset.bin", b"data" * 10000, "model")
        config = json.dumps({"modelFingerprint": "model", "assets": {"/asset.bin": self.files[0]["sha256"]}}).encode()
        (self.repo / "web/training/config.json").write_bytes(config)
        self.add("training/config.json", config, "client")
        self.add("index.html", b"<p>Training</p>", "entry")
        self.add("serve.py", b"# private local helper", "server")
        self.manifest = {"kind": "local-contributor-client", "configHash": hashlib.sha256(config).hexdigest(),
                         "modelFingerprint": "model", "files": self.files}
        self.save()

    def save(self):
        (self.bundle / "bundle-manifest.json").write_text(json.dumps(self.manifest))

    def add(self, name, data, kind):
        target = self.bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": kind})

    def build(self):
        with patch.object(package, "REPO", self.repo), patch.object(package, "HERE", self.here):
            return package.build(self.bundle, self.root / "output")

    def test_public_allowlist_and_compression(self):
        (self.bundle / "secrets.env").write_text("not in manifest")
        result = self.build()
        output = Path(result["directory"])
        manifest = json.loads((output / "public-manifest.json").read_text())
        self.assertEqual(set(manifest["files"]), {"/asset.bin", "/index.html", "/training/config.json"})
        self.assertIn("gzip", manifest["files"]["/asset.bin"])
        self.assertFalse((output / "public/serve.py").exists())
        self.assertFalse((output / "public/secrets.env").exists())
        self.assertTrue((output / "server/training_coordinator.py").is_file())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()

    def test_changed_payload_rejected(self):
        (self.bundle / "asset.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.build()
        self.assertFalse((self.root / "output").exists())

    def test_private_manifest_entry_rejected(self):
        self.add("backup.sqlite3", b"private", "client")
        self.save()
        with self.assertRaisesRegex(ValueError, "Private/server"):
            self.build()

    def test_traversal_and_symlink_rejected(self):
        for name in ("../private", "/private", ".env", "assets/../private", "assets\\private"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.safe_source(self.bundle, name)
        (self.bundle / "asset.bin").unlink()
        (self.bundle / "asset.bin").symlink_to(self.repo / "requirements-cloudrun.txt")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.build()


if __name__ == "__main__":
    unittest.main()
