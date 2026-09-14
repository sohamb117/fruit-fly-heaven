"""Loopback-only, development native-worker check. Does not contact a coordinator."""
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = [
    ("/body-engine/", ROOT / "packages/flybody-runtime/node_modules/@mujoco/mujoco"),
    ("/banc-engine/", ROOT / "packages/banc-runtime"),
    ("/banc-data/", ROOT / "data/prepared/banc888"),
    ("/body-model/", ROOT / "models"),
    ("/engine/", ROOT / "packages/fly-brain-wasm/dist"),
    ("/vision-engine/", ROOT / "packages/fly-vision-wasm/dist"),
    ("/color-engine/", ROOT / "packages/fly-color-wasm/dist"),
    ("/view-engine/", ROOT / "packages/brain-view-wasm/dist"),
    ("/anatomy/", ROOT / "data/anatomy"),
    ("/connectome/", ROOT / "data/prepared"),
]
MAX_REPORT_BYTES = 8 * 1024 * 1024


def read_plan(path):
    """Reject a stale or malformed operator plan before opening the server."""
    raw = path.read_bytes()
    if len(raw) > 256 * 1024:
        raise ValueError("Diagnostic plan exceeds 256 KiB")
    plan = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    config_raw = (ROOT / "web/training/config.json").read_bytes()
    config = json.loads(config_raw)
    if (not isinstance(plan, dict) or plan.get("schemaVersion") != 1
            or plan.get("kind") != "motor-interface-causal-sweep"):
        raise ValueError("Unsupported diagnostic plan")
    if (plan.get("configHash") != hashlib.sha256(config_raw).hexdigest()
            or plan.get("modelFingerprint") != config["modelFingerprint"]):
        raise ValueError("Diagnostic plan does not match the current configuration/model")
    if plan.get("parameterNames") != [p["name"] for p in config["parameters"]]:
        raise ValueError("Diagnostic parameter names differ from the current contract")
    jobs, names = plan.get("jobs"), set()
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 32:
        raise ValueError("Diagnostic plan requires 1 to 32 named jobs")
    stages = {stage["id"]: stage["durationSeconds"] for stage in config["stages"]}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Expected a diagnostic job object")
        name = job.get("name")
        if not isinstance(name, str) or not name or len(name) > 100 or name in names:
            raise ValueError("Diagnostic job names must be nonempty and unique")
        names.add(name)
        seed, values = job.get("seed"), job.get("parameters")
        if type(seed) is not int or not 0 <= seed <= 0xffffffff:
            raise ValueError("Diagnostic seed must be a uint32")
        if job.get("stage") not in stages or job.get("durationSeconds") != stages[job["stage"]]:
            raise ValueError("Diagnostic jobs must use a complete configured stage horizon")
        if not isinstance(values, list) or len(values) != len(config["parameters"]):
            raise ValueError("Diagnostic parameter vector length mismatch")
        for value, parameter in zip(values, config["parameters"]):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not parameter["min"] <= value <= parameter["max"]):
                raise ValueError("Diagnostic parameter outside configured bounds: " + parameter["name"])
    return {"plan": plan, "sha256": hashlib.sha256(raw).hexdigest()}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def local_host(self):
        return self.headers.get("Host") in {
            f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

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

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("X-Content-Type-Options", "nosniff")
        # The pinned MuJoCo embind loader uses new Function. Connections remain
        # confined to this local development origin, including worker fetches.
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; frame-ancestors 'none'")
        super().end_headers()

    def send_json(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.local_host():
            self.send_error(403, "Loopback Host required")
        elif urlsplit(self.path).path == "/__check_result":
            with self.server.report_lock:
                value = json.loads(self.server.report_file.read_text()) if self.server.report_file.exists() else {"phase": "not_started"}
            self.send_json(value)
        elif urlsplit(self.path).path == "/__check_plan":
            self.send_json(self.server.check_plan)
        else:
            super().do_GET()

    def do_POST(self):
        if (not self.local_host() or self.headers.get("Origin") != "http://" + self.headers.get("Host", "")
                or self.client_address[0] != "127.0.0.1"):
            self.send_error(403, "Same-origin loopback request required")
            return
        if urlsplit(self.path).path != "/__check_result":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REPORT_BYTES or self.headers.get_content_type() != "application/json":
                self.send_error(413, "Expected a bounded JSON report")
                return
            value = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(value, dict) or value.get("kind") != "safari-native-worker-check":
                raise ValueError("Expected diagnostic report")
            encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Invalid diagnostic report")
            return
        with self.server.report_lock:
            temporary = self.server.report_file.with_suffix(".tmp")
            temporary.write_text(encoded)
            temporary.replace(self.server.report_file)
        self.send_json({"saved": True})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7844)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Pinned JSON operator plan; omitted runs the original single baseline check")
    args = parser.parse_args()
    directory = args.report_dir.resolve()
    if not directory.is_relative_to((ROOT / "reports").resolve()):
        parser.error("--report-dir must be inside this checkout's reports directory")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        check_plan = read_plan(args.plan.resolve()) if args.plan else {"plan": None}
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.report_file = directory / "result.json"
    server.report_lock = threading.Lock()
    server.check_plan = check_plan
    print(f"Safari native check: http://127.0.0.1:{server.server_port}/test/training-safari-check.html", flush=True)
    print(f"Report: {server.report_file}", flush=True)
    if check_plan["plan"]:
        print(f"Pinned causal diagnostic: {len(check_plan['plan']['jobs'])} sequential jobs; no optimizer or coordinator upload", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
