"""Manifest-only streaming website and the existing training HTTP API."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

try:
    import training_coordinator
except ModuleNotFoundError as error:
    if error.name != "training_coordinator":
        raise
    # Local source checkout; deployed modules live together in /app/server.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import training_coordinator


CHUNK_BYTES = 64 * 1024
MAX_REJECTED_BODY_DRAIN = 1024 * 1024
REJECTED_BODY_DRAIN_SECONDS = 10.0
ISOLATION_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
}


def _digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def _file_path(root, relative):
    if (not isinstance(relative, str) or not relative
            or relative.startswith("/") or "\\" in relative
            or any(not part or part.startswith(".") for part in relative.split("/"))):
        raise ValueError("Invalid public file path")
    target = root.joinpath(*relative.split("/"))
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Public files cannot contain symlinks")
    if root not in target.resolve().parents:
        raise ValueError("Public file escapes public root")
    return target


def _open_file(root, record):
    target = _file_path(root, record["file"])
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != record["bytes"]:
            raise ValueError("Public file changed after manifest creation")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def load_manifest(public_root, manifest_path, *, config_hash=None, model_fingerprint=None):
    root = Path(public_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Public root must be a directory")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schemaVersion") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid public manifest schema")
    for key, expected in (("configHash", config_hash), ("modelFingerprint", model_fingerprint)):
        if not re.fullmatch(r"[a-f0-9]{64}", manifest.get(key, "")):
            raise ValueError("Invalid public manifest identity")
        if expected is not None and manifest[key] != expected:
            raise ValueError("Public manifest and coordinator identity differ")

    def validate_file(record):
        if not isinstance(record, dict) or not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")):
            raise ValueError("Invalid public file digest")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError("Invalid public file length")
        with _open_file(root, record) as incoming:
            if _digest(incoming) != record["sha256"]:
                raise ValueError("Public file hash differs from manifest: " + record["file"])

    for route, record in manifest["files"].items():
        if not isinstance(route, str) or route != "/" + record.get("file", ""):
            raise ValueError("Public route must match its explicit file path")
        if any(character in route for character in ("%", "?", "#", "\r", "\n")):
            raise ValueError("Invalid public route")
        if not isinstance(record.get("contentType"), str) or not re.fullmatch(
                r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+(?:; charset=[Uu][Tt][Ff]-8)?", record["contentType"]):
            raise ValueError("Invalid public content type")
        validate_file(record)
        if "gzip" in record:
            compressed = record["gzip"]
            if not isinstance(compressed, dict) or compressed.get("file") != record["file"] + ".gz":
                raise ValueError("Gzip must designate the explicit sibling file")
            validate_file(compressed)
    return root, manifest


def _accepts_gzip(header):
    encodings = {}
    for item in header.lower().split(","):
        parts = [part.strip() for part in item.split(";")]
        if not parts[0]:
            continue
        quality = 1.0
        for parameter in parts[1:]:
            if parameter.startswith("q="):
                try:
                    quality = float(parameter[2:])
                except ValueError:
                    quality = 0.0
        encodings[parts[0]] = quality if 0 <= quality <= 1 else 0.0
    return encodings.get("gzip", encodings.get("*", 0)) > 0


def allowed_origins(value=""):
    """Explicit origins only; incoming Host/X-Forwarded-Host never expand CORS."""
    origins = list(training_coordinator.DEFAULT_ORIGINS)
    for origin in re.split(r"[,\s]+", value.strip()):
        if origin and origin not in origins:
            origins.append(training_coordinator.validate_origin(origin))
    return tuple(origins)


def make_server(coordinator, public_root, manifest_path, *, host="127.0.0.1", port=8080,
                origins=training_coordinator.DEFAULT_ORIGINS, max_body_bytes=262144, ready=None):
    root, manifest = load_manifest(public_root, manifest_path,
                                  config_hash=coordinator.config_hash,
                                  model_fingerprint=coordinator.model_fingerprint)
    server = training_coordinator.make_server(coordinator, host=host, port=port,
                                             allowed_origins=origins, max_body_bytes=max_body_bytes)
    # A static bundle can be healthy while its imported job history is unusable.
    # Cloud Run must validate that history before promoting the new revision.
    if ready is None:
        ready = lambda: bool(coordinator.status())
    original_handler = server.RequestHandlerClass

    class Handler(original_handler):
        def request_body(self):
            try:
                return super().request_body()
            except training_coordinator.APIError as error:
                if error.status == 413 and error.code == "body_too_large":
                    # The original handler has validated Content-Length but
                    # rejected its size before reading any body. Closing with
                    # unread upload bytes can reset the proxy connection and
                    # replace the intended 413 with a gateway 502.
                    self._drain_rejected_body()
                raise

        def _drain_rejected_body(self):
            remaining = min(int(self.headers["Content-Length"]), MAX_REJECTED_BODY_DRAIN)
            previous_timeout = self.connection.gettimeout()
            deadline = time.monotonic() + REJECTED_BODY_DRAIN_SECONDS
            try:
                while remaining:
                    seconds = deadline - time.monotonic()
                    if seconds <= 0:
                        break
                    self.connection.settimeout(seconds)
                    # read1 performs at most one socket read. Buffered read(n)
                    # can keep waiting indefinitely while a peer trickles data.
                    chunk = self.rfile.read1(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except (TimeoutError, ConnectionError, OSError):
                pass
            finally:
                self.connection.settimeout(previous_timeout)

        def end_headers(self):
            sent = {line.split(b":", 1)[0].lower() for line in getattr(self, "_headers_buffer", [])}
            for name, value in ISOLATION_HEADERS.items():
                if name.lower().encode() not in sent:
                    self.send_header(name, value)
            super().end_headers()

        def _small_response(self, status, value, *, head=False):
            data = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head:
                self.wfile.write(data)

        def _path(self):
            try:
                parsed = urlsplit(self.path)
                if parsed.scheme or parsed.netloc or parsed.fragment:
                    return None
                path = unquote(parsed.path, errors="strict")
            except (ValueError, UnicodeError):
                return None
            if (not path.startswith("/") or "\\" in path
                    or any(ord(character) < 32 for character in path)
                    or any(part in (".", "..") for part in path.split("/"))):
                return None
            return path

        def _serve_public(self, head=False):
            path = self._path()
            if path is None:
                self._small_response(404, {"error": "not_found"}, head=head)
                return
            if path == "/healthz":
                try:
                    healthy = True if ready is None else bool(ready())
                except Exception:
                    healthy = False
                self._small_response(200 if healthy else 503,
                                     {"ready": healthy, "configHash": manifest["configHash"]}, head=head)
                return
            record = manifest["files"].get("/index.html" if path == "/" else path)
            if record is None:
                self._small_response(404, {"error": "not_found"}, head=head)
                return
            compressed = "gzip" in record and _accepts_gzip(self.headers.get("Accept-Encoding", ""))
            representation = record["gzip"] if compressed else record
            etag = '"' + representation["sha256"] + '"'
            candidates = [item.strip().removeprefix("W/") for item in self.headers.get("If-None-Match", "").split(",")]
            unchanged = "*" in candidates or etag in candidates
            try:
                incoming = _open_file(root, representation)
            except (OSError, ValueError):
                self._small_response(503, {"error": "asset_unavailable"}, head=head)
                return
            with incoming:
                self.send_response(304 if unchanged else 200)
                self.send_header("Content-Type", record["contentType"])
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
                if "gzip" in record:
                    self.send_header("Vary", "Accept-Encoding")
                if compressed:
                    self.send_header("Content-Encoding", "gzip")
                if not unchanged:
                    if head:
                        self.send_header("Content-Length", str(representation["bytes"]))
                    else:
                        # Cloud Run's HTTP/1 response limit excludes streamed
                        # responses. Never buffer the 207 MiB graph into a body.
                        self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if head or unchanged:
                    return
                try:
                    for chunk in iter(lambda: incoming.read(CHUNK_BYTES), b""):
                        self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    self.close_connection = True

        def do_GET(self):
            if urlsplit(self.path).path.startswith("/api/training/"):
                return super().do_GET()
            self._serve_public()

        def do_HEAD(self):
            if urlsplit(self.path).path.startswith("/api/training/"):
                return self._small_response(405, {"error": "method_not_allowed"}, head=True)
            self._serve_public(head=True)

    server.RequestHandlerClass = Handler
    return server


def main():
    from firestore_coordinator import FirestoreCoordinator

    app_root = Path(__file__).resolve().parent.parent
    config_path = os.environ.get("TRAINING_CONFIG", str(app_root / "config.json"))
    config, config_hash = training_coordinator.read_config(config_path)
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
    run_id = os.environ["TRAINING_RUN_ID"]
    lease_timeout_seconds = int(os.environ.get("TRAINING_LEASE_TIMEOUT_SECONDS", "180"))
    coordinator = FirestoreCoordinator(config, config_hash, project=project, database="(default)",
                                       run_id=run_id, initialize=False, lease_timeout_seconds=lease_timeout_seconds)
    server = None
    try:
        server = make_server(coordinator,
                             os.environ.get("PUBLIC_ROOT", str(app_root / "public")),
                             os.environ.get("PUBLIC_MANIFEST", str(app_root / "public-manifest.json")),
                             host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                             origins=allowed_origins(os.environ.get("ALLOWED_ORIGINS", "")),
                             max_body_bytes=262144)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        print(json.dumps({"ready": True, "port": server.server_port,
                          "configHash": config_hash, "runId": run_id}), flush=True)
        server.serve_forever()
    finally:
        if server is not None:
            server.server_close()
        coordinator.close()


if __name__ == "__main__":
    main()
