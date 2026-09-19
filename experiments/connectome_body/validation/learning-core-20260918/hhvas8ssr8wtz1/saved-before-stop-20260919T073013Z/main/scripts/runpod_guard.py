"""Run a command with a detached RunPod stop watchdog; never create/start/delete pods.

The guard runs on the supervising host, independently of the worker and this chat.
Use an always-on host: suspending that host also suspends its watchdog. Credentials
stay on that host and are removed from the worker environment. Persistent pod
storage survives a stop; a stalled artifact download never extends the deadline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as out:
        json.dump(value, out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def credentials():
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    config = Path.home() / ".runpod/config.toml"
    if not key and config.exists():
        data = tomllib.loads(config.read_text())
        key = (data.get("apikey") or data.get("apiKey") or "").strip()
    if not key:
        raise RuntimeError("Configure RUNPOD_API_KEY or runpodctl locally before any paid launch")
    return key


class RunpodClient:
    def __init__(self, pod_id, key):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
            raise ValueError("Invalid pod ID")
        self.pod_id, self.key = pod_id, key

    def request(self, action=None):
        url = f"https://api.runpod.io/v2/pods/{self.pod_id}"
        data = None
        if action is not None:
            if action != "stop":
                raise ValueError("This guard only permits stop")
            url += "/action"
            data = json.dumps({"action": "stop"}).encode()
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                pod = json.load(response)
        except urllib.error.HTTPError as exc:
            # Response bodies may contain account details. Never log them or the key.
            raise RuntimeError(f"RunPod HTTP status {exc.code}") from None
        if pod.get("id") != self.pod_id:
            raise ValueError("RunPod returned a different pod")
        return {
            k: pod.get(k)
            for k in ("id", "status", "runtime", "actions", "locked", "createdAt", "startedAt")
        }

    def get(self):
        return self.request()

    def stop(self):
        return self.request("stop")


def target_identity(pod):
    return {k: pod[k] for k in ("id", "createdAt", "startedAt")}


def check_target(pod):
    if pod["status"] != "RUNNING" or pod.get("locked") or "stop" not in pod.get("actions", []):
        raise RuntimeError("Guard requires an existing running, unlocked, stoppable pod")
    if not pod.get("createdAt") or not pod.get("startedAt"):
        raise RuntimeError("Cannot identify this pod boot safely")


def watchdog(
    client,
    request,
    directory,
    *,
    poll=1.0,
    clock=time.time,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    directory = Path(directory)
    pod = client.get()
    check_target(pod)
    if target_identity(pod) != request["target"]:
        raise RuntimeError("Pod boot changed before the watchdog was armed")
    start = monotonic()
    remaining = max(0, request["deadline"] - clock())
    record = {
        "nonce": request["nonce"],
        "target": request["target"],
        "deadline": request["deadline"],
        "state": "armed",
        "guard_pid": os.getpid(),
        "heartbeat": clock(),
    }
    write_json(directory / "guard.json", record)
    while True:
        if clock() >= request["deadline"] or monotonic() - start >= remaining:
            reason = "hard_deadline"
            break
        try:
            heartbeat = json.loads((directory / "heartbeat.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            heartbeat = {"time": request["armed_at"], "nonce": request["nonce"]}
        if heartbeat.get("nonce") != request["nonce"]:
            reason = "supervisor_identity_changed"
            break
        if heartbeat.get("done"):
            reason = "worker_finished"
            break
        if clock() - heartbeat["time"] > request["heartbeat_timeout"]:
            reason = "supervisor_lost"
            break
        record["heartbeat"] = clock()
        write_json(directory / "guard.json", record)
        sleep(poll)
    record.update(state="stopping", reason=reason)
    write_json(directory / "guard.json", record)
    return stop_until_released(
        client, request, directory, record, poll=poll, clock=clock, sleep=sleep
    )


def stop_until_released(
    client, request, directory, record, *, poll=1, clock=time.time, sleep=time.sleep
):
    attempts = 0
    # Stay detached and keep retrying transient control-plane failures. Success
    # means a verified provider state, never merely a successful POST response.
    while True:
        attempts += 1
        try:
            pod = client.get()
            if {k: pod[k] for k in ("id", "createdAt")} != {
                k: request["target"][k] for k in ("id", "createdAt")
            }:
                record.update(state="target_changed", attempts=attempts)
                write_json(directory / "guard.json", record)
                return 2  # Never stop a later boot used by another job.
            if pod["status"] == "EXITED" and pod["runtime"] is None:
                record.update(state="stopped", verified_at=clock(), attempts=attempts)
                write_json(directory / "guard.json", record)
                return 0
            if target_identity(pod) != request["target"]:
                record.update(state="target_changed", attempts=attempts)
                write_json(directory / "guard.json", record)
                return 2
            if "stop" in pod.get("actions", []):
                client.stop()
        except Exception as exc:
            # Persist only the exception class; no credential-bearing request data.
            record["last_error_type"] = type(exc).__name__
        record.update(attempts=attempts, heartbeat=clock())
        write_json(directory / "guard.json", record)
        sleep(min(10, poll * max(1, attempts)))


def terminate_worker(worker):
    if worker is not None and worker.poll() is None:
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()


def supervise(pod_id, seconds, directory, command):
    if not math.isfinite(seconds) or seconds <= 0 or not command:
        raise ValueError("Positive finite deadline and a worker command are required")
    key = credentials()  # Fail before any worker or SSH command if auth is missing.
    client = RunpodClient(pod_id, key)
    pod = client.get()
    check_target(pod)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    now = time.time()
    request = {
        "target": target_identity(pod),
        "armed_at": now,
        "deadline": now + seconds,
        "heartbeat_timeout": 15,
        "nonce": uuid.uuid4().hex,
    }
    write_json(directory / "request.json", request)
    env = dict(os.environ, RUNPOD_API_KEY=key)
    with (directory / "watchdog.log").open("a") as log:
        guard = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "watch", "--output", str(directory)],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    worker = None
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    original = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for _ in range(100):
            if (directory / "guard.json").exists():
                armed = json.loads((directory / "guard.json").read_text())
                if armed["state"] == "armed" and armed["nonce"] == request["nonce"]:
                    break
            if guard.poll() is not None or interrupted:
                raise RuntimeError("Watchdog failed before worker launch")
            time.sleep(0.1)
        else:
            raise RuntimeError("Watchdog never acknowledged its deadline; worker not launched")
        worker_env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
        worker_env["CONNECTOME_PREPARATION_REQUIRE_CACHE"] = "1"
        worker = subprocess.Popen(command, env=worker_env, start_new_session=True)
        while worker.poll() is None:
            write_json(
                directory / "heartbeat.json", {"nonce": request["nonce"], "time": time.time()}
            )
            if interrupted or time.time() >= request["deadline"] or guard.poll() is not None:
                break
            time.sleep(0.5)
    finally:
        terminate_worker(worker)
        write_json(
            directory / "heartbeat.json",
            {"nonce": request["nonce"], "time": time.time(), "done": True},
        )
        for sig, handler in original.items():
            signal.signal(sig, handler)
        try:
            guard.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Keep the independent guard alive to retry; never cancel it on timeout.
            print(
                "Stop verification pending; detached watchdog continues retrying", file=sys.stderr
            )
        state = (
            json.loads((directory / "guard.json").read_text())
            if (directory / "guard.json").exists()
            else {}
        )
        if (
            state.get("state") != "stopped"
            and guard.poll() is not None
            and state.get("state") != "target_changed"
        ):
            state = {
                "nonce": request["nonce"],
                "target": request["target"],
                "state": "stopping",
                "reason": "watchdog_lost",
            }
            stop_until_released(client, request, directory, state)
        if state.get("state") != "stopped":
            raise RuntimeError("Pod shutdown is not yet verified; inspect guard.json")
    return worker.returncode if worker is not None else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--pod-id", required=True)
    run.add_argument("--seconds", type=float, required=True)
    run.add_argument("--output", required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    watch = sub.add_parser("watch")
    watch.add_argument("--output", required=True)
    a = p.parse_args()
    if a.mode == "watch":
        directory = Path(a.output)
        request = json.loads((directory / "request.json").read_text())
        return watchdog(RunpodClient(request["target"]["id"], credentials()), request, directory)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    return supervise(a.pod_id, a.seconds, a.output, command)


if __name__ == "__main__":
    sys.exit(main())
