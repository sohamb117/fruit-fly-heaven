"""Qualify an existing pod, copy a checksummed archive, and independently stop it.

This command never creates or starts a pod. Source, dependencies, public inputs,
CPU preparation caches and the remote frozen plan must already be present.
"""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_guard import supervise  # noqa: E402


def relative_run_path(value):
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:1] != ("runs",)
        or len(path.parts) < 2
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
    ):
        raise argparse.ArgumentTypeError("Use a relative path under runs/")
    return path.as_posix()


def worker(a):
    destination = Path(a.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    options = [
        "-i",
        str(Path(a.ssh_key).expanduser()),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
    ]
    target = f"root@{a.host}"
    archive = f"{a.run}.tar.gz"
    q = shlex.quote
    command = f"""set -eu
test "$RUNPOD_POD_ID" = {q(a.pod_id)}
cd /workspace/experiments/connectome_body
export UV_CACHE_DIR=/workspace/.uv-cache MUJOCO_GL=disable MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.mplconfig
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CONNECTOME_PREPARATION_REQUIRE_CACHE=1
mkdir -p {q(a.run)}
code=0
uv run --no-sync python -u scripts/qualify_learning_core.py --plan {q(a.plan + "/plan.json")} --output {q(a.run)} --max-seconds {a.training_seconds} --case-max-seconds 180 {"--resume" if a.resume else ""} >> {q(a.run + "/qualification.log")} 2>&1 || code=$?
printf '%s\\n' "$code" > {q(a.run + "/launcher-exit-code")}
tar -czf {q(archive)} {q(a.run)} {q(a.plan + "/plan.json")}
sha256sum {q(archive)} > {q(archive + ".sha256")}
"""
    subprocess.run(["ssh", *options, "-p", str(a.port), target, command], check=True)
    for name in (archive, archive + ".sha256"):
        subprocess.run(
            [
                "scp",
                *options,
                "-P",
                str(a.port),
                f"{target}:/workspace/experiments/connectome_body/{name}",
                str(destination / PurePosixPath(name).name),
            ],
            check=True,
        )
    local_archive = destination / PurePosixPath(archive).name
    expected = local_archive.with_suffix(local_archive.suffix + ".sha256").read_text().split()[0]
    with local_archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Downloaded checkpoint archive checksum mismatch")
    # Read only this known JSON member; never extract arbitrary archive paths.
    with tarfile.open(local_archive) as bundle:
        code = int(bundle.extractfile(f"{a.run}/launcher-exit-code").read())
        result = json.load(bundle.extractfile(f"{a.run}/qualification.json"))
        plan = json.load(bundle.extractfile(f"{a.plan}/plan.json"))
    if result["plan_fingerprint"] != plan["fingerprint"] or result["source"] != plan["source"]:
        raise ValueError("Qualification result belongs to another plan or source")
    (destination / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] and code == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ssh-key", required=True)
    p.add_argument("--plan", type=relative_run_path, required=True)
    p.add_argument("--run", type=relative_run_path, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--training-seconds", type=int, default=1200)
    p.add_argument("--hard-seconds", type=int, default=1500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not 0 < a.training_seconds < a.hard_seconds or not 1 <= a.port <= 65535:
        p.error("Require positive training time below the hard deadline and a valid SSH port")
    if a.worker:
        return worker(a)
    return supervise(
        a.pod_id,
        a.hard_seconds,
        Path(a.output) / "guard",
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
    )


if __name__ == "__main__":
    sys.exit(main())
