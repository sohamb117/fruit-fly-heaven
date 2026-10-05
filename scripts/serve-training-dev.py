"""Explicit loopback training pool: original contributor UI, native workers, durable SQLite."""
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler
import io
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import training_coordinator

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = [
    ("/body-engine/", ROOT / "packages/flybody-runtime/node_modules/@mujoco/mujoco"),
    ("/banc-engine/", ROOT / "packages/banc-runtime"),
    ("/banc-data/", ROOT / "data/prepared/banc888"),
    ("/body-model/", ROOT / "models"),
]
DEVELOPMENT_META = '<meta name="heaven-training-development" content="same-origin">'
MODEL_OVERRIDE_PATHS = frozenset({"/body-model/flybody-mujoco.xml", "/body-model/flybody-mujoco.json"})
RUNTIME_OVERRIDE_PATHS = frozenset(
    ["/" + name + ".js" for name in (
        "banc-antenna", "banc-ground-sense", "banc-haltere", "banc-leg-proprioception", "banc-proboscis",
        "banc-sensory-current", "banc-taste", "banc-tegula", "banc/embodiment", "body-world", "color-vision",
        "flight-scene-profile", "motor-decoder", "sensory-encoder", "virtual-haltere")]
    + ["/flybody-" + name + ".js" for name in (
        "contact-environment", "habitat-collision", "leg-actuation", "motor-excitation", "mouth-pose", "physics",
        "stance", "wing-event-excitation", "wing-load", "wing-pose", "wings", "world")]
    + ["/training/" + name + ".js" for name in (
        "airborne-reset-contract", "airborne-reset", "brain-resources", "compact-vision", "config-schema", "environment",
        "episode", "flight-objective", "flight-observation", "flight-parameters", "flight-telemetry", "maintained-flight-objective",
        "retinal-sensor", "sensory-feedback", "worker", "sensorimotor-parameters", "recovery-objective",
        "sequential-environment", "decoder-calibration", "flight-teacher")]
    + ["/banc-engine/src/" + name + ".js" for name in ("cell-models", "index", "model", "motor-events", "wasm", "webgpu")]
    + ["/banc-engine/dist/core.js", "/body-engine/mujoco.js", "/vendor/three.core.js", "/vendor/three.module.js",
       "/body-model/banc-leg-proprioception-v1.json", "/body-model/flight-teacher-calibration-v1.json"])


def validate_asset_overrides(config_bytes, asset_overrides):
    """Snapshot pinned model/runtime bytes; never override UI or API routes."""
    if asset_overrides is None:
        return {}
    if (not isinstance(asset_overrides, dict) or not MODEL_OVERRIDE_PATHS.issubset(asset_overrides)
            or not set(asset_overrides).issubset(MODEL_OVERRIDE_PATHS | RUNTIME_OVERRIDE_PATHS)):
        raise ValueError("Development bundle requires model XML/metadata and supported pinned runtime assets")
    if any(not isinstance(value, (str, bytes)) for value in asset_overrides.values()):
        raise ValueError("Development model overrides must contain text or bytes")
    overrides = {url: value.encode("utf-8") if isinstance(value, str) else bytes(value)
                 for url, value in asset_overrides.items()}
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("Development bundle configuration must be an object")
    assets = config.get("assets")
    if not isinstance(assets, dict) or not assets or any(
            not isinstance(url, str) or not url or "\n" in url or
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for url, digest in assets.items()):
        raise ValueError("Development bundle requires a pinned asset manifest")
    # Current canonical manifests use this order. Require it instead of
    # rewriting configText: the coordinator hashes the exact stored order.
    if list(assets) != sorted(assets):
        raise ValueError("Development bundle asset manifest must use canonical sorted order")
    fingerprint = hashlib.sha256("".join(f"{url}:{assets[url]}\n" for url in sorted(assets)).encode()).hexdigest()
    if config.get("modelFingerprint") != fingerprint:
        raise ValueError("Development bundle modelFingerprint does not match its asset manifest")
    for url, content in overrides.items():
        if assets.get(url) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Development bundle asset checksum mismatch: {url}")
    metadata = json.loads(overrides["/body-model/flybody-mujoco.json"])
    xml_hash = hashlib.sha256(overrides["/body-model/flybody-mujoco.xml"]).hexdigest()
    if not isinstance(metadata, dict) or metadata.get("xml_sha256") != xml_hash:
        raise ValueError("Development bundle metadata does not describe the supplied XML")
    return overrides


def load_bundle(bundle_path):
    """Read a source-pinned diagnostic bundle once, exclusively from reports."""
    path = Path(bundle_path).resolve()
    if not path.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Development bundle must be inside this checkout's reports directory")
    bundle = json.loads(path.read_bytes())
    if not isinstance(bundle, dict) or type(bundle.get("schemaVersion")) is not int or bundle["schemaVersion"] != 1:
        raise ValueError("Unsupported development bundle schemaVersion")
    if bundle.get("kind") != "flight-development-bundle":
        raise ValueError("Unsupported development bundle kind")
    if not isinstance(bundle.get("configText"), str):
        raise ValueError("Development bundle must contain exact configText")
    if not isinstance(bundle.get("assets"), dict) or any(not isinstance(value, str) for value in bundle["assets"].values()):
        raise ValueError("Development bundle assets must be serialized text")
    config_bytes = bundle["configText"].encode("utf-8")
    overrides = validate_asset_overrides(config_bytes, bundle["assets"])
    config = json.loads(config_bytes)
    if "configHash" in bundle and bundle["configHash"] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("Development bundle configHash does not match configText")
    if "modelFingerprint" in bundle and bundle["modelFingerprint"] != config["modelFingerprint"]:
        raise ValueError("Development bundle modelFingerprint differs from configText")
    return config_bytes, overrides


def make_server(coordinator, config_bytes, *, port=7845, asset_overrides=None, brain_sample=None):
    if hashlib.sha256(config_bytes).hexdigest() != coordinator.config_hash:
        raise ValueError("Served configuration and coordinator differ")
    overrides = validate_asset_overrides(config_bytes, asset_overrides)
    if brain_sample is not None:
        sample = json.loads(brain_sample)
        if sample.get("modelFingerprint") != coordinator.model_fingerprint or sample.get("dataset") != "BANC v888":
            raise ValueError("Brain display sample belongs to a different model")
        overrides["/training/brain-sample.json"] = bytes(brain_sample)
    server = training_coordinator.make_server(coordinator, host="127.0.0.1", port=port, allowed_origins=())
    api_handler = server.RequestHandlerClass
    original_html = (ROOT / "web/train.html").read_text()
    if original_html.count("</head>") != 1:
        raise ValueError("Expected one training page head")
    training_html = original_html.replace("</head>", DEVELOPMENT_META + "\n</head>").encode()

    class Handler(api_handler, SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

        def origin(self):
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}:
                raise training_coordinator.APIError(403, "host_denied", "Loopback Host required")
            origin = self.headers.get("Origin")
            if origin is not None and origin != "http://" + host:
                raise training_coordinator.APIError(403, "origin_denied", "Same-origin development request required")
            return origin

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("X-Content-Type-Options", "nosniff")
            # MuJoCo's pinned embind loader requires new Function. External
            # network connections, including production uploads, are blocked.
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; frame-ancestors 'none'")
            super().end_headers()

        def translate_path(self, value):
            path = unquote(urlsplit(value).path)
            base, relative = ROOT / "web", path.lstrip("/")
            for prefix, candidate in PREFIXES:
                if path.startswith(prefix):
                    base, relative = candidate, path[len(prefix):]
                    break
            target = (base / relative).resolve()
            return str(target if target.is_relative_to(base.resolve()) else ROOT / "web/__missing__")

        def list_directory(self, path):
            self.send_error(403, "Directory listing disabled")

        def send_head(self):
            try:
                self.origin()
            except training_coordinator.APIError as error:
                self.send_error(error.status, error.code)
                return None
            path = urlsplit(self.path).path
            content = training_html if path in ("/", "/train.html") else config_bytes if path == "/training/config.json" else overrides.get(path)
            if content is None:
                return super().send_head()
            self.send_response(200)
            content_type = "text/html; charset=utf-8" if path in ("/", "/train.html") else mimetypes.guess_type(path)[0] or "application/octet-stream"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            return io.BytesIO(content)

        def do_GET(self):
            if urlsplit(self.path).path.startswith("/api/training/"):
                return super().do_GET()
            return SimpleHTTPRequestHandler.do_GET(self)

        def do_HEAD(self):
            if urlsplit(self.path).path.startswith("/api/training/"):
                self.send_error(405)
            else:
                SimpleHTTPRequestHandler.do_HEAD(self)

    server.RequestHandlerClass = Handler
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7845)
    parser.add_argument("--database", type=Path, default=ROOT / "reports/training-flight-active/coordinator.sqlite3")
    parser.add_argument("--bundle", type=Path, help="Source-pinned reduced-model bundle inside reports; default uses canonical assets")
    args = parser.parse_args()
    database = args.database.resolve()
    if not database.is_relative_to((ROOT / "reports").resolve()):
        parser.error("Development database must be inside this checkout's reports directory")
    try:
        config_bytes, asset_overrides = load_bundle(args.bundle) if args.bundle else ((ROOT / "web/training/config.json").read_bytes(), None)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    database.parent.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    if config.get("trainingSequence"):
        from sequential_training import SequentialTrainingCoordinator
        coordinator = SequentialTrainingCoordinator(database, config, config_hash)
    else:
        coordinator = training_coordinator.TrainingCoordinator(database, config, config_hash)
    server = None
    try:
        sample_path = args.bundle.resolve().parent / "brain-sample.json" if args.bundle else None
        brain_sample = sample_path.read_bytes() if sample_path and sample_path.is_file() else None
        server = make_server(coordinator, config_bytes, port=args.port, asset_overrides=asset_overrides, brain_sample=brain_sample)
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}/train.html", "database": str(database),
                          "configHash": config_hash, "modelFingerprint": coordinator.model_fingerprint,
                          "stage": coordinator.stage, "durationSeconds": coordinator.duration,
                          "mode": "isolated-development-coordinator", "bundle": str(args.bundle.resolve()) if args.bundle else None}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        coordinator.close()


if __name__ == "__main__":
    main()
