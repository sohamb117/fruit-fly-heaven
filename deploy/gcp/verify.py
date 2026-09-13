"""Verify a deployment allowlist, contents, and optional SQLite identity."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_path(root, name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or any(x in (".", "..") or x.startswith(".") for x in parts) or "\\" in name:
        raise ValueError("Invalid package path: " + name)
    target = root.joinpath(*parts)
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlink in package: " + name)
    if not target.is_file() or root not in target.resolve().parents:
        raise ValueError("Missing package file: " + name)
    return target


def verify_database(database, config_hash, model_fingerprint):
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Database consistency check failed")
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        if meta.get("configHash") != config_hash or meta.get("modelFingerprint") != model_fingerprint:
            raise ValueError("Database belongs to another config/model; it will not be replaced")
    finally:
        connection.close()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "deployment-manifest.json").read_text())
    if manifest.get("schemaVersion") != 1:
        raise ValueError("Unknown deployment manifest")
    if not re.fullmatch(r"[a-f0-9]{16}", manifest.get("deploymentId", "")):
        raise ValueError("Invalid deployment identity")
    if not all(re.fullmatch(r"[a-f0-9]{64}", manifest.get(key, "")) for key in ("configHash", "modelFingerprint")):
        raise ValueError("Invalid config/model identity")
    expected = {"deployment-manifest.json"}
    for record in manifest["files"]:
        name = record["path"]
        if name in expected:
            raise ValueError("Duplicate package entry: " + name)
        expected.add(name)
        target = safe_path(root, name)
        if target.stat().st_size != record["bytes"] or digest(target) != record["sha256"]:
            raise ValueError("Package checksum mismatch: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError("Unexpected files in deployment package: " + str(sorted(actual - expected)))
    if digest(root / "config/config.json") != manifest["configHash"]:
        raise ValueError("Config hash mismatch")
    if digest(root / "public/training/config.json") != manifest["configHash"]:
        raise ValueError("Public and coordinator configs differ")
    state = root / "state/bootstrap.sqlite3"
    if state.exists():
        verify_database(state, manifest["configHash"], manifest["modelFingerprint"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--database")
    args = parser.parse_args()
    manifest = verify(args.root)
    if args.database:
        verify_database(args.database, manifest["configHash"], manifest["modelFingerprint"])
    print(json.dumps({key: manifest[key] for key in ("deploymentId", "configHash", "modelFingerprint")}))
