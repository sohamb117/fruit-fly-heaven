import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import qualification_ssh_job as ssh_job
from scripts import qualify_learning_core as qualification
from scripts import runpod_guard as guard


class Clock:
    def __init__(self):
        self.now = 100.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Client:
    def __init__(self, failures=0):
        self.pod = {
            "id": "fixture",
            "createdAt": "created",
            "startedAt": "boot",
            "status": "RUNNING",
            "runtime": {},
            "actions": ["stop"],
            "locked": False,
        }
        self.stops, self.reads, self.failures = 0, 0, failures

    def get(self):
        self.reads += 1
        return dict(self.pod)

    def stop(self):
        self.stops += 1
        if self.stops <= self.failures:
            raise OSError("control-plane unavailable")
        self.pod.update(status="EXITED", runtime=None, startedAt=None)
        return self.pod


@pytest.mark.parametrize("reason", ["hard_deadline", "worker_finished", "supervisor_lost"])
def test_detached_watchdog_stops_and_verifies_provider(tmp_path, reason):
    clock, client = Clock(), Client(failures=2)
    request = {
        "target": guard.target_identity(client.get()),
        "deadline": 105,
        "armed_at": 100,
        "heartbeat_timeout": 2 if reason == "supervisor_lost" else 20,
        "nonce": "nonce",
    }
    if reason == "worker_finished":
        guard.write_json(tmp_path / "heartbeat.json", {"nonce": "nonce", "time": 100, "done": True})
    assert (
        guard.watchdog(
            client, request, tmp_path, clock=clock.time, monotonic=clock.time, sleep=clock.sleep
        )
        == 0
    )
    record = json.loads((tmp_path / "guard.json").read_text())
    assert record["state"] == "stopped" and record["reason"] == reason
    assert client.stops == 3 and client.reads >= 5


def test_watchdog_never_stops_a_new_boot(tmp_path):
    client, clock = Client(), Clock()
    request = {"target": guard.target_identity(client.get())}
    client.pod["startedAt"] = "another-boot"
    result = guard.stop_until_released(
        client, request, tmp_path, {}, clock=clock.time, sleep=clock.sleep
    )
    assert result == 2 and client.stops == 0


def test_missing_auth_prevents_any_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Worker was launched"))
    with pytest.raises(RuntimeError, match="Configure RUNPOD_API_KEY"):
        guard.supervise("fixture", 10, tmp_path / "run", ["worker"])
    assert not (tmp_path / "run").exists()


def test_guard_can_run_in_an_independent_process(tmp_path):
    # Real detached OS process; fake provider, no network or billable resources.
    code = """
import json, pathlib, time
from scripts.runpod_guard import watchdog, target_identity
class Client:
    def __init__(self):
        self.pod = dict(id="fixture", createdAt="created", startedAt="boot", status="RUNNING", runtime={}, actions=["stop"])
    def get(self): return self.pod.copy()
    def stop(self): self.pod.update(status="EXITED", runtime=None)
c=Client(); now=time.time()
r=dict(target=target_identity(c.get()), deadline=now+.15, armed_at=now, heartbeat_timeout=10, nonce="fixture")
raise SystemExit(watchdog(c,r,pathlib.Path(__import__("sys").argv[1]),poll=.02))
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], start_new_session=True)
    assert child.wait(timeout=5) == 0
    record = json.loads((tmp_path / "guard.json").read_text())
    assert record["state"] == "stopped" and record["reason"] == "hard_deadline"


def test_qualification_prioritizes_all_cross_body_pairs():
    rows = []
    for source in ("banc", "malecns", "celegans"):
        for task in ("hover", "worm_locomotion"):
            for variant in ("real", "degree_rewired", "gru", "rnn", "adapter_only"):
                for plasticity in ("adapters", "joint"):
                    rows.append(
                        dict(
                            id=str(len(rows)),
                            source=source,
                            task=task,
                            variant=variant,
                            plasticity=plasticity,
                            seed=0,
                            regime="bc_ppo",
                            experiment="cross_body_core",
                        )
                    )
    cases = qualification.cases({"conditions": rows})
    assert len(cases) == len({r["id"] for r in cases}) == 18
    assert {(r["source"], r["task"]) for r in cases[:6]} == {
        (source, task)
        for source in ("banc", "malecns", "celegans")
        for task in ("hover", "worm_locomotion")
    }
    assert {r["variant"] for r in cases[6:9]} == {"gru", "rnn", "adapter_only"}


def test_api_client_uses_verified_stop_endpoint_without_logging_keys(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"id":"fixture","status":"EXITED","runtime":null}'

    def open_request(request, timeout):
        requests.append(request)
        assert timeout == 5
        return Response()

    monkeypatch.setattr(guard.urllib.request, "urlopen", open_request)
    c = guard.RunpodClient("fixture", "fake-secret")
    assert c.stop()["status"] == "EXITED"
    assert requests[0].full_url == "https://api.runpod.io/v2/pods/fixture/action"
    assert json.loads(requests[0].data) == {"action": "stop"}
    with pytest.raises(ValueError, match="only permits stop"):
        c.request("terminate")


def test_supervisor_stops_pod_when_guard_dies_before_launch(tmp_path, monkeypatch):
    client = Client()
    monkeypatch.setattr(guard, "credentials", lambda: "fake")
    monkeypatch.setattr(guard, "RunpodClient", lambda *args: client)
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)

    class DeadGuard:
        def poll(self):
            return 1

        def wait(self, timeout):
            return 1

    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        assert command[2] == "watch"
        return DeadGuard()

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="Watchdog failed"):
        guard.supervise("fixture", 10, tmp_path / "run", ["worker"])
    assert len(calls) == 1 and client.stops == 1
    assert json.loads((tmp_path / "run/guard.json").read_text())["state"] == "stopped"


def test_case_timeout_does_not_prevent_other_body_checks(tmp_path, monkeypatch):
    from connectome_body.compatibility import learning_study

    monkeypatch.setenv("CONNECTOME_PREPARATION_REQUIRE_CACHE", "0")
    monkeypatch.setenv("CONNECTOME_PREPARATION_LOG", "0")

    plan = {"fingerprint": "plan", "source": "source"}
    monkeypatch.setattr(learning_study, "read_plan", lambda _: plan)
    monkeypatch.setattr(qualification, "cases", lambda _: [{"id": "first"}, {"id": "second"}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--plan",
            "fixture",
            "--output",
            str(tmp_path),
            "--max-seconds",
            "10",
            "--case-max-seconds",
            "1",
        ],
    )
    monkeypatch.setattr(qualification.os, "killpg", lambda *args: None)
    launched = []

    class Worker:
        pid = 123

        def __init__(self, index):
            self.index, self.calls, self.returncode = index, 0, 0

        def wait(self, timeout):
            self.calls += 1
            if self.index == 0 and self.calls == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 0

    def popen(command, **kwargs):
        index = int(command[-1])
        launched.append(index)
        if index == 1:
            (tmp_path / "case-01/qualification-case.json").write_text(json.dumps({"passed": True}))
        return Worker(index)

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(SystemExit) as exc:
        qualification.main()
    assert exc.value.code == 1 and launched == [0, 1]
    result = json.loads((tmp_path / "qualification.json").read_text())
    assert result["cases"][0]["reason"] == "case_deadline"
    assert result["cases"][1]["passed"] and not result["passed"]


def test_guard_acknowledgement_precedes_worker_and_key_is_not_forwarded(tmp_path, monkeypatch):
    client, calls = Client(), []
    directory = tmp_path / "run"
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-secret")
    monkeypatch.setattr(guard, "RunpodClient", lambda *args: client)

    class Guard:
        def poll(self):
            return None

        def wait(self, timeout):
            client.stop()
            guard.write_json(directory / "guard.json", {"state": "stopped"})
            return 0

    class Worker:
        returncode = 0

        def poll(self):
            return 0

    def popen(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            request = json.loads((directory / "request.json").read_text())
            assert kwargs["start_new_session"]
            guard.write_json(
                directory / "guard.json", {"state": "armed", "nonce": request["nonce"]}
            )
            return Guard()
        assert "RUNPOD_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["CONNECTOME_PREPARATION_REQUIRE_CACHE"] == "1"
        assert kwargs["start_new_session"]
        return Worker()

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert guard.supervise("fixture", 10, directory, ["worker"]) == 0
    assert len(calls) == 2 and client.stops == 1


@pytest.mark.parametrize(
    "exit_code,passed,corrupt",
    [(0, True, False), (1, True, False), (1, False, False), (0, True, True)],
)
def test_ssh_job_preserves_artifacts_and_rejects_stale_success(
    tmp_path, monkeypatch, exit_code, passed, corrupt
):
    remote = tmp_path / "remote"
    remote.mkdir()
    archive = remote / "qualification.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        members = {
            "runs/qualification/qualification.json": json.dumps(
                {"passed": passed, "plan_fingerprint": "plan", "source": "source"}
            ),
            "runs/qualification/launcher-exit-code": str(exit_code),
            "runs/plan/plan.json": json.dumps({"fingerprint": "plan", "source": "source"}),
        }
        for name, value in members.items():
            data = value.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    checksum = "bad" if corrupt else hashlib.sha256(archive.read_bytes()).hexdigest()
    (remote / "qualification.tar.gz.sha256").write_text(checksum + "  runs/qualification.tar.gz\n")

    def run(command, check):
        if command[0] == "ssh":
            assert 'test "$RUNPOD_POD_ID" = fixture' in command[-1]
            assert "|| code=$?" in command[-1]
        else:
            assert command[0] == "scp"
            shutil.copyfile(remote / Path(command[-1]).name, command[-1])

    monkeypatch.setattr(subprocess, "run", run)
    args = Namespace(
        output=tmp_path / "out",
        ssh_key="fixture-key",
        host="example.test",
        port=22,
        pod_id="fixture",
        plan="runs/plan",
        run="runs/qualification",
        training_seconds=10,
        resume=False,
    )
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            ssh_job.worker(args)
    else:
        assert ssh_job.worker(args) == (0 if passed and exit_code == 0 else 1)
    assert (tmp_path / "out/qualification.tar.gz").exists()


@pytest.mark.parametrize("path", ["/tmp/run", "runs", "runs/../../secrets", "runs/x;echo"])
def test_archive_scope_is_one_run_under_runs(path):
    with pytest.raises(Exception, match="relative path"):
        ssh_job.relative_run_path(path)
