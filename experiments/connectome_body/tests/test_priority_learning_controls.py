"""Lifecycle checks for the bounded priority window, using harmless subprocesses."""

import json
import time

import pytest

from scripts import priority_learning_controls as supervisor


@pytest.mark.parametrize("returncode", [0, 7])
def test_restores_admissions_after_child_completion_or_failure(tmp_path, monkeypatch, returncode):
    root = tmp_path
    scheduler = root / "runs/main-parallel-20260918"
    scheduler.mkdir(parents=True)
    deadline = time.time() + 180
    (scheduler / "request.json").write_text(
        json.dumps(
            {
                "deadline": deadline,
                "checkpoint_deadline": deadline - 10,
            }
        )
    )
    (scheduler / "control.json").write_text(json.dumps({"max_workers": 6, "ready_seeds": [0, 1]}))
    package = root / "connectome_body/compatibility"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "learning_study.py").write_text(f"raise SystemExit({returncode})\n")
    monkeypatch.setattr(supervisor, "ROOT", root)
    supervisor.main()
    output = root / "runs/priority-controls-20260918"
    completed = json.loads((output / "complete.json").read_text())
    request = json.loads((output / "request.json").read_text())
    assert request["hard_deadline"] < deadline
    assert completed["admission_limit"] == 6
    assert all(job["returncode"] == returncode for job in completed["jobs"])
    assert json.loads((scheduler / "control.json").read_text()) == {
        "max_workers": 6,
        "ready_seeds": [0, 1],
    }
    with pytest.raises(FileExistsError):
        supervisor.main()
