"""HTTP/SQLite integration tests; no Firestore SDK or biological scores needed."""
import gzip
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cloudrun_server", Path(__file__).with_name("server.py"))
server_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server_module)
coordinator_module = server_module.training_coordinator


def fixture():
    return {
        "schemaVersion": 1, "environmentVersion": "banc-flybody-rl-v1",
        "modelFingerprint": hashlib.sha256(("fixture.wasm:" + "b" * 64 + "\n").encode()).hexdigest(),
        "assets": {"fixture.wasm": "b" * 64},
        "algorithm": "antithetic-evolution-strategies", "dtMs": .5, "bodyBlockMs": 2,
        "parameters": [{"name": "gain_log", "min": -2, "max": 2, "initial": 0}],
        "optimizer": {"populationPairs": 1, "sigma": .25, "learningRate": .035, "maximumUpdate": .15, "seed": 888},
        "objective": {"min": -10, "max": 10, "direction": "maximize"},
        "stage": "posture", "durationSeconds": 1,
        "stages": [{"id": "posture", "durationSeconds": 1}],
        "contribution": {"leaseSeconds": 10, "maxRequestBytes": 1024},
    }


class CloudRunServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.public = cls.root / "public"
        cls.public.mkdir()
        cls.coordinator = coordinator_module.TrainingCoordinator(cls.root / "private.sqlite3", fixture())
        cls.manifest = {"schemaVersion": 1, "configHash": cls.coordinator.config_hash,
                        "modelFingerprint": cls.coordinator.model_fingerprint, "files": {}}
        cls.page = b"<!doctype html><title>Training fixture</title>" * 20
        cls.add_file("index.html", b'<meta http-equiv="refresh" content="0;url=/train.html">', "text/html; charset=utf-8")
        cls.add_file("train.html", cls.page, "text/html; charset=utf-8", compressed=True)
        # Exceeds Cloud Run's ordinary HTTP/1 32 MiB response cap.
        cls.large_bytes = 33 * 1024 * 1024 + 17
        cls.large_sha = hashlib.sha256()
        large = cls.public / "graph.bin"
        block = bytes(range(256)) * 256
        with large.open("wb") as outgoing:
            remaining = cls.large_bytes
            while remaining:
                value = block[:min(len(block), remaining)]
                outgoing.write(value)
                cls.large_sha.update(value)
                remaining -= len(value)
        cls.manifest["files"]["/graph.bin"] = {"file": "graph.bin", "sha256": cls.large_sha.hexdigest(),
                                               "bytes": cls.large_bytes, "contentType": "application/octet-stream"}
        # Unlisted public-root files and neighboring private files are not routes.
        (cls.public / "server.py").write_text("not public")
        (cls.root / "secret.txt").write_text("private secret")
        cls.manifest_path = cls.root / "public-manifest.json"
        cls.manifest_path.write_text(json.dumps(cls.manifest))
        cls.healthy = True
        cls.server = server_module.make_server(cls.coordinator, cls.public, cls.manifest_path, port=0,
                                               origins=server_module.allowed_origins("https://training.example.org"),
                                               max_body_bytes=1024, ready=lambda: cls.healthy)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def add_file(cls, name, data, content_type, compressed=False):
        (cls.public / name).write_bytes(data)
        record = {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "contentType": content_type}
        if compressed:
            zipped = gzip.compress(data, mtime=0)
            (cls.public / (name + ".gz")).write_bytes(zipped)
            record["gzip"] = {"file": name + ".gz", "sha256": hashlib.sha256(zipped).hexdigest(), "bytes": len(zipped)}
        cls.manifest["files"]["/" + name] = record

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.coordinator.close()
        cls.temporary.cleanup()

    def request(self, path, method="GET", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def assert_isolated(self, headers):
        for name, value in server_module.ISOLATION_HEADERS.items():
            self.assertEqual(headers.get(name), value)

    def test_get_head_and_representation_etags(self):
        status, headers, body = self.request("/train.html")
        self.assertEqual((status, body), (200, self.page))
        self.assertEqual(headers["Transfer-Encoding"], "chunked")
        self.assertNotIn("Content-Length", headers)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assert_isolated(headers)
        etag = headers["ETag"]
        status, headers, body = self.request("/train.html", "HEAD")
        self.assertEqual((status, body), (200, b""))
        self.assertEqual(int(headers["Content-Length"]), len(self.page))
        self.assertNotIn("Transfer-Encoding", headers)
        self.assertEqual(headers["ETag"], etag)
        status, headers, body = self.request("/train.html", headers={"If-None-Match": '"unrelated", W/' + etag})
        self.assertEqual((status, body), (304, b""))
        self.assertEqual(headers["ETag"], etag)
        self.assert_isolated(headers)
        status, headers, body = self.request("/train.html", headers={"Accept-Encoding": "br, gzip;q=0.8"})
        self.assertEqual(status, 200)
        self.assertEqual(gzip.decompress(body), self.page)
        self.assertEqual(headers["Content-Encoding"], "gzip")
        self.assertEqual(headers["Vary"], "Accept-Encoding")
        self.assertNotEqual(headers["ETag"], etag)
        status, headers, body = self.request("/train.html", headers={"Accept-Encoding": "gzip;q=0, *;q=1"})
        self.assertEqual(body, self.page)
        self.assertNotIn("Content-Encoding", headers)

    def test_large_file_streams_beyond_32_mib(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request("GET", "/graph.bin")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Transfer-Encoding"), "chunked")
            self.assertIsNone(response.getheader("Content-Length"))
            self.assertTrue(response.chunked)
            digest = hashlib.sha256()
            count = 0
            for chunk in iter(lambda: response.read(64 * 1024), b""):
                digest.update(chunk)
                count += len(chunk)
            self.assertEqual(count, self.large_bytes)
            self.assertEqual(digest.hexdigest(), self.large_sha.hexdigest())
        finally:
            connection.close()

    def test_private_paths_traversal_and_unlisted_files_are_denied(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/%2e%2e%2fsecret.txt",
                     "/..%5csecret.txt", "/server.py", "/private.sqlite3", "/public-manifest.json",
                     "/config.json", "/.env", "/scripts/training_coordinator.py", "/train.html.gz", "/public/"):
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 404)
                self.assertNotIn(b"private secret", body)
                self.assert_isolated(headers)
        self.assertIn(b"/train.html", self.request("/")[2])

    def test_api_origin_and_body_validation_is_reused(self):
        status, headers, body = self.request("/api/training/status", headers={"Origin": "https://training.example.org"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["configHash"], self.coordinator.config_hash)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://training.example.org")
        self.assert_isolated(headers)
        status, headers, body = self.request("/api/training/status", headers={"Origin": "https://evil.example", "Host": "evil.example", "X-Forwarded-Host": "evil.example"})
        self.assertEqual((status, json.loads(body)["error"]), (403, "origin_denied"))
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assert_isolated(headers)
        status, headers, body = self.request("/api/training/lease", "POST", b"x" * 1025, {"Content-Type": "application/json"})
        self.assertEqual((status, json.loads(body)["error"]), (413, "body_too_large"))
        self.assert_isolated(headers)
        identity = {"contributorId": "server-test", "configHash": self.coordinator.config_hash,
                    "modelFingerprint": self.coordinator.model_fingerprint}
        status, headers, body = self.request("/api/training/lease", "POST", json.dumps(identity),
                                             {"Content-Type": "application/json", "Origin": "http://127.0.0.1:7842"})
        self.assertEqual(status, 200)
        self.assertIsInstance(json.loads(body)["job"], dict)
        status, headers, body = self.request("/api/training/result", "OPTIONS", headers={
            "Origin": "http://localhost:7843", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        self.assertEqual((status, body), (204, b""))
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:7843")
        self.assert_isolated(headers)

    def test_segmented_oversized_upload_receives_complete_413(self):
        server = server_module.make_server(self.coordinator, self.public, self.manifest_path,
                                           port=0, max_body_bytes=262144)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = b"x" * 262145
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.sendall(("POST /api/training/lease HTTP/1.1\r\n"
                                    "Host: localhost\r\nContent-Type: application/json\r\n"
                                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode())
                # Headers arrive before the body, as with a proxy forwarding an
                # upload. Closing with unread bytes used to reset this stream.
                time.sleep(.025)
                for offset in range(0, len(body), 8192):
                    connection.sendall(body[offset:offset + 8192])
                    time.sleep(.003)
                response = http.client.HTTPResponse(connection)
                response.begin()
                try:
                    self.assertEqual(response.status, 413)
                    self.assertEqual(json.loads(response.read())["error"], "body_too_large")
                    self.assertEqual(response.getheader("Cross-Origin-Embedder-Policy"), "require-corp")
                finally:
                    response.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_rejected_body_drain_stops_at_one_mib(self):
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=2) as connection:
            connection.sendall(("POST /api/training/lease HTTP/1.1\r\n"
                                "Host: localhost\r\nContent-Type: application/json\r\n"
                                "Content-Length: 2097152\r\nConnection: close\r\n\r\n").encode())
            # Send only the drain limit, not the declared two MiB. The response
            # must arrive without waiting for the remaining rejected bytes.
            connection.sendall(b"x" * (1024 * 1024))
            response = http.client.HTTPResponse(connection)
            response.begin()
            try:
                self.assertEqual(response.status, 413)
                self.assertEqual(json.loads(response.read())["error"], "body_too_large")
            finally:
                response.close()

    def test_rejected_body_drain_has_absolute_deadline(self):
        with patch.object(server_module, "REJECTED_BODY_DRAIN_SECONDS", .05):
            with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=2) as connection:
                connection.sendall(("POST /api/training/lease HTTP/1.1\r\n"
                                    "Host: localhost\r\nContent-Type: application/json\r\n"
                                    "Content-Length: 262145\r\nConnection: close\r\n\r\n").encode())
                # A client that declares a body but stops sending cannot hold
                # the rejection path open forever.
                started = time.monotonic()
                response = http.client.HTTPResponse(connection)
                response.begin()
                try:
                    self.assertEqual(response.status, 413)
                    self.assertEqual(json.loads(response.read())["error"], "body_too_large")
                    self.assertLess(time.monotonic() - started, 1)
                finally:
                    response.close()

    def test_default_health_rejects_unusable_coordinator_history(self):
        previous = self.server
        self.server = server_module.make_server(self.coordinator, self.public, self.manifest_path, port=0)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        try:
            failure = coordinator_module.APIError(503, "invalid_history", "Guarded assignment differs")
            with patch.object(self.coordinator, "status", side_effect=failure):
                status, _, body = self.request("/healthz")
                self.assertEqual(status, 503)
                self.assertFalse(json.loads(body)["ready"])
            self.assertEqual(self.request("/healthz")[0], 200)
        finally:
            self.server.shutdown()
            self.server.server_close()
            thread.join()
            self.server = previous

    def test_health_reflects_readiness(self):
        status, headers, body = self.request("/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ready"])
        self.assert_isolated(headers)
        type(self).healthy = False
        try:
            self.assertEqual(self.request("/healthz")[0], 503)
        finally:
            type(self).healthy = True

    def test_manifest_mismatch_and_symlinks_fail_closed(self):
        bad = json.loads(json.dumps(self.manifest))
        bad["files"]["/train.html"]["sha256"] = "0" * 64
        bad_path = self.root / "bad-manifest.json"
        bad_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            server_module.load_manifest(self.public, bad_path)
        bad = json.loads(json.dumps(self.manifest))
        (self.public / "escape.html").symlink_to(self.root / "secret.txt")
        bad["files"]["/escape.html"] = {**bad["files"]["/train.html"], "file": "escape.html"}
        bad_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "symlinks"):
            server_module.load_manifest(self.public, bad_path)
        with self.assertRaisesRegex(ValueError, "identity differ"):
            server_module.load_manifest(self.public, self.manifest_path, config_hash="c" * 64)


if __name__ == "__main__":
    unittest.main()
