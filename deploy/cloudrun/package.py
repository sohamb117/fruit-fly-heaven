"""Create an allowlisted Cloud Build context from the verified contributor bundle."""
import argparse
import gzip
import hashlib
import json
import mimetypes
from pathlib import Path, PurePosixPath
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SERVER_FILES = ("training_coordinator.py", "firestore_coordinator.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_source(root, name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or "\\" in name or any(p.startswith(".") for p in parts):
        raise ValueError("Invalid bundle path: " + name)
    target = root.joinpath(*parts)
    if any(p.is_symlink() for p in [target, *target.parents] if p != root and root in p.parents):
        raise ValueError("Symlink in bundle: " + name)
    if not target.is_file() or root not in target.resolve().parents:
        raise ValueError("Missing bundle file: " + name)
    return target


def content_type(name):
    extension = Path(name).suffix
    return {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
            ".wasm": "application/wasm", ".wgsl": "text/plain; charset=utf-8",
            ".bin": "application/octet-stream", ".md": "text/plain; charset=utf-8"}.get(
                extension, mimetypes.guess_type(name)[0] or "application/octet-stream")


def build(bundle, output):
    bundle = Path(bundle).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a fresh build directory")
    canonical = REPO / "web/training/config.json"
    config = json.loads(canonical.read_bytes())
    source_manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    config_hash = digest(canonical)
    if (source_manifest.get("kind") != "local-contributor-client" or
            source_manifest.get("configHash") != config_hash or
            source_manifest.get("modelFingerprint") != config["modelFingerprint"]):
        raise ValueError("Bundle does not match the canonical model/config")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cloudrun-", dir=output.parent) as temporary:
        stage = Path(temporary) / "context"
        public = stage / "public"
        public.mkdir(parents=True)
        records, seen = {}, set()
        for record in source_manifest["files"]:
            name = record["path"]
            if name in seen:
                raise ValueError("Duplicate bundle path: " + name)
            seen.add(name)
            source = safe_source(bundle, name)
            if source.stat().st_size != record["bytes"] or digest(source) != record["sha256"]:
                raise ValueError("Bundle checksum mismatch: " + name)
            if name in {"serve.py", "README.md"}:
                continue
            if record.get("kind") not in {"model", "graph", "client", "presentation", "license", "source", "instructions", "entry"}:
                raise ValueError("Unexpected public file kind: " + name)
            if name.endswith((".sqlite", ".sqlite3", ".db", ".pem", ".key", ".env", ".py")):
                raise ValueError("Private/server file in public bundle: " + name)
            target = public / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if digest(target) != record["sha256"]:
                raise ValueError("Bundle changed during copy: " + name)
            entry = {"file": name, "bytes": record["bytes"], "sha256": record["sha256"], "contentType": content_type(name)}
            if record["bytes"] >= 1024:
                compressed = target.with_name(target.name + ".gz")
                with target.open("rb") as incoming, compressed.open("wb") as raw:
                    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
                        shutil.copyfileobj(incoming, zipped, 1024 * 1024)
                if compressed.stat().st_size < record["bytes"] * .98:
                    entry["gzip"] = {"file": name + ".gz", "bytes": compressed.stat().st_size, "sha256": digest(compressed)}
                else:
                    compressed.unlink()
            records["/" + name] = entry
        for url, expected in config["assets"].items():
            if records.get(url, {}).get("sha256") != expected:
                raise ValueError("Missing model asset: " + url)
        manifest = {"schemaVersion": 1, "configHash": config_hash,
                    "modelFingerprint": config["modelFingerprint"], "files": records}
        (stage / "public-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "server").mkdir()
        for name in SERVER_FILES:
            shutil.copyfile(REPO / "scripts" / name, stage / "server" / name)
        shutil.copyfile(HERE / "server.py", stage / "server/server.py")
        shutil.copyfile(canonical, stage / "config.json")
        shutil.copyfile(REPO / "requirements-cloudrun.txt", stage / "requirements.txt")
        for name in ("Dockerfile", "cloudbuild.yaml", ".gcloudignore"):
            shutil.copyfile(HERE / name, stage / name)
        inventory = [{"path": p.relative_to(stage).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                     for p in sorted(stage.rglob("*")) if p.is_file()]
        identity = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()[:16]
        (stage / "build-manifest.json").write_text(json.dumps({"buildId": identity, "files": inventory}, indent=2) + "\n")
        shutil.move(stage, output)
    return {"directory": str(output), "buildId": identity, "configHash": config_hash,
            "modelFingerprint": config["modelFingerprint"], "publicFiles": len(records),
            "bytes": sum(record["bytes"] for record in inventory)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.output), indent=2))
