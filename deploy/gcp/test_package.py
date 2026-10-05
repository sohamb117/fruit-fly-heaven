"""Packaging checks use a tiny fixture, not fabricated training results."""
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from backup import backup
import package
from verify import verify, verify_database


class DeploymentPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.repo / "web/training").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        (self.repo / "scripts/training_coordinator.py").write_text("# Fixture coordinator\n")
        self.asset = b"asset contents" * 500
        self.config = {"modelFingerprint": "a" * 64, "assets": {"/asset.bin": hashlib.sha256(self.asset).hexdigest()}}
        self.config_bytes = (json.dumps(self.config) + "\n").encode()
        self.config_hash = hashlib.sha256(self.config_bytes).hexdigest()
        (self.repo / "web/training/config.json").write_bytes(self.config_bytes)
        self.entries = []
        self.add_entry("training/config.json", self.config_bytes)
        self.add_entry("asset.bin", self.asset, "model")
        self.add_entry("index.html", b"Training", "entry")
        self.add_entry("serve.py", b"local server excluded", "server")
        self.add_entry("README.md", b"local instructions excluded", "instructions")
        # An unlisted secret must never be copied by directory recursion.
        (self.bundle / ".env").write_text("DO_NOT_PACKAGE")
        self.write_manifest()

    def add_entry(self, name, contents, kind="client"):
        destination = self.bundle / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        self.entries.append({"path": name, "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest(), "kind": kind})

    def write_manifest(self):
        (self.bundle / "bundle-manifest.json").write_text(json.dumps({"kind": "local-contributor-client", "modelFingerprint": self.config["modelFingerprint"], "configHash": self.config_hash, "files": self.entries}))

    def build(self, database=None):
        with patch.object(package, "REPO", self.repo):
            return package.build(self.bundle, self.root / "output", database)

    def test_allowlist_compression_and_integrity(self):
        result = self.build()
        with tarfile.open(result["archive"]) as archive:
            names = archive.getnames()
            self.assertNotIn("fly-training/public/serve.py", names)
            self.assertNotIn("fly-training/public/.env", names)
            self.assertNotIn("fly-training/public/README.md", names)
            self.assertEqual(gzip.decompress(archive.extractfile("fly-training/public/asset.bin.gz").read()), self.asset)
            extracted = self.root / "extracted"
            archive.extractall(extracted, filter="data")
        manifest = verify(extracted / "fly-training")
        self.assertEqual(manifest["configHash"], self.config_hash)
        (extracted / "fly-training/public/index.html").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify(extracted / "fly-training")

    def test_source_hash_and_symlink_rejected(self):
        (self.bundle / "asset.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bundle changed"):
            self.build()
        (self.bundle / "asset.bin").unlink()
        (self.bundle / "asset.bin").symlink_to(self.repo / "web/training/config.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.build()

    def test_online_snapshot_keeps_committed_wal_and_identity(self):
        database = self.root / "live.sqlite3"
        writer = sqlite3.connect(database)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")
        writer.executemany("INSERT INTO meta VALUES(?,?)", [("configHash", self.config_hash), ("modelFingerprint", self.config["modelFingerprint"])])
        writer.execute("CREATE TABLE observations(value INTEGER)")
        writer.execute("INSERT INTO observations VALUES(17)")
        writer.commit()
        snapshot = self.root / "snapshot.sqlite3"
        backup(database, snapshot)
        verify_database(snapshot, self.config_hash, self.config["modelFingerprint"])
        reader = sqlite3.connect(snapshot)
        try:
            self.assertEqual(reader.execute("SELECT value FROM observations").fetchone()[0], 17)
            self.assertEqual(reader.execute("PRAGMA journal_mode").fetchone()[0], "delete")
        finally:
            reader.close()
        with self.assertRaisesRegex(ValueError, "replace"):
            backup(database, snapshot)
        with self.assertRaisesRegex(ValueError, "another config"):
            verify_database(snapshot, "b" * 64, self.config["modelFingerprint"])
        result = self.build(database)
        with tarfile.open(result["archive"]) as archive:
            names = archive.getnames()
            self.assertIn("fly-training/state/bootstrap.sqlite3", names)
            self.assertFalse(any(name.startswith("fly-training/public/") and ".sqlite" in name for name in names))


if __name__ == "__main__":
    unittest.main()
