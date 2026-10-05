"""Periodically retrieve and verify a running pod's batched experiment artifacts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from runpod_short_test import checkpoint_summary, read_json, write_json


def backup(args):
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ssh = [
        "ssh",
        "-i",
        str(args.key),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    subprocess.run(
        [
            "rsync",
            "-az",
            "--exclude=*.tmp",
            "--exclude=.*",
            "-e",
            shlex.join(ssh),
            f"root@{args.host}:{args.remote.rstrip('/')}/",
            str(destination) + "/",
        ],
        check=True,
        timeout=45,
    )
    checkpoint = destination / "training/latest.pt"
    summary = checkpoint_summary(checkpoint) if checkpoint.exists() else None
    if (
        summary
        and summary["identity"] != read_json(checkpoint.parent / "manifest.json")["identity"]
    ):
        raise ValueError("Downloaded resume checkpoint does not match its manifest")
    report = {
        "time_unix": time.time(),
        "status": "copied_and_verified",
        "pod_id": args.pod,
        "checkpoint": summary,
        "run_status": read_json(destination / "status.json"),
    }
    write_json(destination.parent / "backup-status.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while time.time() < args.deadline_unix:
        try:
            report = backup(args)
            print(
                json.dumps(
                    {
                        "backup": report["status"],
                        "time_unix": report["time_unix"],
                        "updates": (report["checkpoint"] or {}).get("updates"),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            if args.once:
                raise
            print(json.dumps({"backup": "retry_needed", "error": repr(error)}), flush=True)
        if args.once:
            break
        time.sleep(min(45, max(0, args.deadline_unix - time.time())))


if __name__ == "__main__":
    main()
