"""Build a strict, verified GCP web/coordinator archive from a contributor bundle."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from backup import backup
from verify import digest, safe_path, verify, verify_database

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEPLOY_FILES = {
    "install.sh": "install.sh",
    "Caddyfile.template": "system/Caddyfile.template",
    "fly-training.service": "system/fly-training.service",
    "caddy.service": "system/caddy.service",
    "launcher.py": "server/launcher.py",
    "backup.py": "tools/backup.py",
    "verify.py": "tools/verify.py",
    "README.md": "DEPLOYMENT.md",
}


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def build(bundle, output, database=None):
    bundle = bundle.resolve(strict=True)
    canonical = REPO / "web/training/config.json"
    config = json.loads(canonical.read_bytes())
    config_hash = digest(canonical)
    source_manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    if source_manifest.get("kind") != "local-contributor-client" or source_manifest.get("configHash") != config_hash or source_manifest.get("modelFingerprint") != config["modelFingerprint"]:
        raise ValueError("Contributor bundle and canonical config must match")
    allowed_kinds = {"model", "graph", "client", "presentation", "license", "source", "instructions", "entry", "server"}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gcp-package-", dir=output) as temporary:
        stage = Path(temporary) / "fly-training"
        stage.mkdir()
        records = []

        def add(source, name, expected=None):
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise ValueError("Only regular files may enter a deployment: " + str(source))
            destination = stage / name
            if destination.exists() or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Invalid or duplicate destination: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            record = {"path": name, "bytes": destination.stat().st_size, "sha256": digest(destination)}
            if expected and (record["bytes"] != expected["bytes"] or record["sha256"] != expected["sha256"]):
                raise ValueError("Contributor file checksum mismatch: " + name)
            if digest(source) != record["sha256"]:
                raise ValueError("Source changed while copying: " + str(source))
            records.append(record)

        seen = set()
        for record in source_manifest["files"]:
            name = record["path"]
            if name in seen or record.get("kind") not in allowed_kinds:
                raise ValueError("Unexpected bundle entry: " + name)
            seen.add(name)
            source = safe_path(bundle, name)
            if digest(source) != record["sha256"] or source.stat().st_size != record["bytes"]:
                raise ValueError("Contributor bundle changed: " + name)
            if name in {"serve.py", "README.md"}:
                continue
            if record.get("kind") == "server":
                raise ValueError("A server file may not enter the public deployment: " + name)
            if name.endswith((".sqlite3", ".db", ".pem", ".key", ".env")):
                raise ValueError("Unexpected private data in public allowlist: " + name)
            add(source, "public/" + name, record)
        for url, sha in config["assets"].items():
            if url[1:] not in seen or digest(stage / "public" / url[1:]) != sha:
                raise ValueError("Missing canonical asset: " + url)
        # Precompression reduces client downloads without using the small VM's CPU.
        for record in list(records):
            if record["bytes"] < 1024:
                continue
            source = stage / record["path"]
            compressed = source.with_name(source.name + ".gz")
            with source.open("rb") as incoming, compressed.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            if compressed.stat().st_size >= record["bytes"] * .98:
                compressed.unlink()
                continue
            records.append({"path": record["path"] + ".gz", "bytes": compressed.stat().st_size, "sha256": digest(compressed)})

        add(REPO / "scripts/training_coordinator.py", "server/training_coordinator.py")
        add(canonical, "config/config.json")
        for source, target in DEPLOY_FILES.items():
            add(HERE / source, target)
        if database:
            snapshot = Path(temporary) / "bootstrap.sqlite3"
            backup(database, snapshot)
            verify_database(snapshot, config_hash, config["modelFingerprint"])
            add(snapshot, "state/bootstrap.sqlite3")
        records.sort(key=lambda record: record["path"])
        deployment_id = hashlib.sha256(json_bytes(records)).hexdigest()[:16]
        manifest = {"schemaVersion": 1, "deploymentId": deployment_id, "configHash": config_hash,
                    "modelFingerprint": config["modelFingerprint"], "sourceBundle": bundle.name,
                    "bootstrapDatabase": bool(database), "files": records}
        (stage / "deployment-manifest.json").write_bytes(json_bytes(manifest))
        verify(stage)
        target = output / f"fly-training-gcp-{deployment_id}.tar.gz"
        with target.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with tarfile.open(mode="w|", fileobj=zipped, format=tarfile.PAX_FORMAT) as archive:
                for name in sorted([record["path"] for record in records] + ["deployment-manifest.json"]):
                    source = stage / name
                    info = tarfile.TarInfo("fly-training/" + name)
                    info.size = source.stat().st_size
                    info.mode = 0o600 if name.startswith("state/") else 0o755 if name == "install.sh" else 0o644
                    info.mtime = 0
                    with source.open("rb") as incoming:
                        archive.addfile(info, incoming)
        checksum = digest(target)
        target.with_suffix(target.suffix + ".sha256").write_text(checksum + "  " + target.name + "\n")
        target.with_suffix(".manifest.json").write_bytes(json_bytes(manifest))
        return {"archive": str(target), "bytes": target.stat().st_size, "sha256": checksum,
                "deploymentId": deployment_id, "files": len(records), "configHash": config_hash,
                "bootstrapDatabase": bool(database)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "dist/gcp")
    parser.add_argument("--database", type=Path, help="Optional existing SQLite database; uses a consistent online backup")
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.output, args.database), indent=2))
