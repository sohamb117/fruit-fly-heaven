"""Start only the packaged coordinator with explicit public and local origins."""
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

root = Path(__file__).resolve().parents[1]
origin = os.environ["PUBLIC_ORIGIN"]
url = urlsplit(origin)
if (url.scheme != "https" or not url.hostname or url.path or url.query
        or url.fragment or url.username or url.password or url.port is not None):
    raise SystemExit("PUBLIC_ORIGIN must be an HTTPS origin on port 443")

origins = [origin] + [f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (7842, 7843)]
argv = [sys.executable, "-u", str(root / "server/training_coordinator.py"),
        "--config", str(root / "config/config.json"),
        "--database", "/var/lib/fly-training/coordinator.sqlite3",
        "--host", "127.0.0.1", "--port", "7850", "--max-request-bytes", "262144"]
for value in origins:
    argv.extend(["--allow-origin", value])
os.execv(sys.executable, argv)
