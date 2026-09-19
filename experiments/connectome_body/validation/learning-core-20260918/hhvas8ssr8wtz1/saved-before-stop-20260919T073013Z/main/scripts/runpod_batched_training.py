"""Bounded, resumable batch continuation with separate hover validation selection.

This is an exploratory follow-up, not a cell in the frozen batch-four matrix.
The caller must back up artifacts and stop the paid pod at its resource deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runpod_short_test import (  # noqa: E402
    checkpoint_summary,
    preflight,
    read_json,
    stop_child,
    write_json,
)


def training_block(config, output, stop_after):
    from connectome_body.adaptation.imitation import AdapterSpec, Optimization, train_offline

    stage = output / "training"
    return train_offline(
        AdapterSpec.from_dict(config["adapter"]),
        Optimization(**config["offline"]),
        config["training_caches"],
        config["data"]["validation"],
        stage,
        resume=(stage / "latest.pt").exists(),
        initial_checkpoint=config["initial_checkpoint"],
        stop_after=stop_after,
    )


def control_improves(candidate, incumbent, tolerance=1e-6):
    """Lexicographic success then score; ties retain the incumbent."""
    keys = ("success", "score", "numerical_failure")
    if not all(math.isfinite(candidate[k]) for k in keys):
        return False
    if candidate["numerical_failure"] > 0:
        return False
    if incumbent is None:
        return True
    for key in ("success", "score"):
        difference = candidate[key] - incumbent[key]
        if abs(difference) > tolerance:
            return difference > 0
    return False


def actor_hash(saved):
    digest = hashlib.sha256()
    for name, tensor in sorted(saved["actor"].items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_candidates(config, output, initial=False):
    import torch

    from connectome_body.adaptation.configuration import hover_config_from_dict
    from connectome_body.adaptation.evaluation import evaluate_controller
    from connectome_body.adaptation.hover import FlyBodyInterface
    from connectome_body.adaptation.imitation import load_actor
    from connectome_body.util import digest_file

    torch.set_num_threads(1)
    history_path = output / "control-validation.json"
    history = read_json(history_path, {"evaluations": [], "selected": None})
    if initial:
        source = Path(config["initial_checkpoint"])
        candidates = [("initial", source.parent, source.name)]
    else:
        stage = output / "training"
        saved = torch.load(stage / "latest.pt", map_location="cpu", weights_only=False)
        update = saved["updates"]
        candidates = []
        (stage / "snapshots").mkdir(exist_ok=True)
        for label, name in (("latest", "latest.pt"), ("teacher_mse_best", "best.pt")):
            relative = f"snapshots/update-{update:06d}-{label}.pt"
            destination = stage / relative
            if not destination.exists():
                shutil.copy2(stage / name, destination)
            elif digest_file(destination) != digest_file(stage / name):
                raise ValueError("An immutable evaluation snapshot changed")
            candidates.append((label, stage, relative))
    body = FlyBodyInterface(hover_config_from_dict(config["body"]))
    try:
        if body.fingerprint != read_json(output / "preflight.json")["body_fingerprint"]:
            raise ValueError("Validation body differs from the preflight body")
        for label, run, checkpoint in candidates:
            path = run / checkpoint
            saved = torch.load(path, map_location="cpu", weights_only=False)
            weights_hash = actor_hash(saved)
            existing = next(
                (r for r in history["evaluations"] if r["actor_sha256"] == weights_hash), None
            )
            if existing is not None:
                continue
            actor, _, _ = load_actor(run, checkpoint)
            started = time.monotonic()
            metrics = evaluate_controller(
                body, seeds=config["control_validation_seeds"], split="validation", actor=actor
            )
            row = {
                "label": label,
                "run": str(run),
                "checkpoint": checkpoint,
                "sha256": digest_file(path),
                "actor_sha256": weights_hash,
                "updates": saved["updates"],
                "phase": "parent" if initial else "batch64_replay",
                "wall_seconds": time.monotonic() - started,
                "metrics": metrics,
                "selection_data": "fixed_development_validation_seeds_not_test",
            }
            history["evaluations"].append(row)
            incumbent = history["selected"]
            if control_improves(metrics, incumbent["metrics"] if incumbent else None):
                # Each export is self-contained and immutable; the selection JSON is atomic.
                export = output / "selected" / weights_hash
                export.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run / "manifest.json", export / "manifest.json")
                shutil.copy2(path, export / "best.pt")
                row["export"] = str(export.relative_to(output))
                history["selected"] = row
            write_json(history_path, history)
            print(
                json.dumps(
                    {
                        "evaluation": label,
                        "updates": row["updates"],
                        "success": metrics["success"],
                        "score": metrics["score"],
                        "survival": metrics["survival"],
                    }
                ),
                flush=True,
            )
            del actor
    finally:
        body.close()


def audit(output):
    from connectome_body.util import digest_file

    stage = output / "training"
    result = {"time_unix": time.time(), "checkpoints": []}
    if (stage / "latest.pt").exists():
        summary = checkpoint_summary(stage / "latest.pt")
        if summary["identity"] != read_json(stage / "manifest.json")["identity"]:
            raise ValueError("Resume checkpoint identity differs from its manifest")
        result["checkpoints"].append(summary)
    selected = read_json(output / "control-validation.json", {}).get("selected")
    if selected:
        exported = output / selected["export"] / "best.pt"
        if digest_file(exported) != selected["sha256"]:
            raise ValueError("Selected controller export differs from its source")
        result["selected"] = selected
    result["files"] = {
        str(p.relative_to(output)): digest_file(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
        and p.name not in ("audit.json", "status.json", "events.json")
        and not p.name.endswith((".tmp", ".log"))
        and not p.name.startswith(".")
    }
    write_json(output / "audit.json", result)


def supervise(args, config, output):
    from connectome_body.adaptation.imitation import source_identity
    from connectome_body.adaptation.trajectories import TrajectoryCache
    from connectome_body.util import digest_file, digest_json

    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": config,
        "source_identity": source_identity(),
        "harness_sha256": digest_file(__file__),
        "parent_checkpoint_sha256": digest_file(config["initial_checkpoint"]),
        "training_caches": [
            {"fingerprint": c.fingerprint, "interactions": c.manifest["interactions"]}
            for c in map(TrajectoryCache, config["training_caches"])
        ],
        "continuation": "parent_actor_weights_only_new_Adam_new_counters",
        "resumption": "exact_optimizer_rng_counters_within_this_branch",
        "evidence": "exploratory_single_seed_offline_replay_and_hover_validation",
        "new_training_environment_interactions": 0,
    }
    identity["identity"] = digest_json(identity)
    previous = read_json(output / "experiment.json")
    if previous is not None and previous != identity:
        raise ValueError("Batched experiment identity changed")
    write_json(output / "experiment.json", identity)
    events = read_json(output / "events.json", [])
    child = None

    def event(phase, **details):
        row = {"phase": phase, "time_unix": time.time(), **details}
        events.append(row)
        write_json(output / "events.json", events)
        write_json(output / "status.json", row)
        print(json.dumps(row), flush=True)

    def phase(name, mode, deadline, *extra):
        nonlocal child
        command = [
            sys.executable,
            __file__,
            mode,
            "--config",
            str(args.config),
            "--output",
            str(output),
            *extra,
        ]
        with (output / f"{name}.log").open("a") as log:
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            event(name, pid=child.pid, deadline_unix=deadline)
            while child.poll() is None and time.time() < deadline:
                time.sleep(1)
            if child.poll() is None:
                stop_child(child)
                raise TimeoutError(f"{name} reached its time limit")
            if child.returncode != 0:
                raise RuntimeError(f"{name} exited with {child.returncode}; see {name}.log")

    outcome = "complete"
    # A final validation/audit window ends before the external resource stop deadline.
    training_deadline = args.deadline_unix - 180
    try:
        if not (output / "preflight.json").exists():
            phase("preflight", "preflight", min(time.time() + 180, training_deadline))
        if not (output / "control-validation.json").exists():
            phase("initial-control", "evaluate", training_deadline, "--initial")
        while time.time() < training_deadline:
            saved = read_json(output / "training/losses.json", [])
            current = saved[-1]["updates"] if saved else 0
            if current >= config["offline"]["updates"]:
                break
            boundary = (
                25
                if current < 25
                else (current // config["control_eval_every"] + 1) * config["control_eval_every"]
            )
            boundary = min(boundary, config["offline"]["updates"])
            phase(
                f"train-to-{boundary:06d}",
                "train",
                training_deadline,
                "--stop-after",
                str(boundary),
            )
            if boundary == 25:
                audit(output)  # Verify a resumable checkpoint before the longer first block.
            else:
                phase(f"control-{boundary:06d}", "evaluate", training_deadline)
                audit(output)
        else:
            outcome = "budget_exhausted"
    except TimeoutError as error:
        outcome = "budget_exhausted"
        event("training_deadline", detail=str(error))
    except Exception as error:
        event("failed", error=repr(error))
        raise
    finally:
        stop_child(child)
    if (output / "training/latest.pt").exists() and time.time() < args.deadline_unix - 20:
        try:
            phase("final-control", "evaluate", args.deadline_unix - 20)
        except TimeoutError:
            event("final_control_timeout", detail="Saved resume checkpoint retained")
    audit(output)
    event(outcome, audit="audit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "preflight", "train", "evaluate", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--initial", action="store_true")
    args = parser.parse_args()
    config, output = read_json(args.config), args.output.resolve()
    if args.mode == "supervise":
        if args.deadline_unix is None or args.deadline_unix <= time.time() + 240:
            parser.error("A future deadline with >=240 seconds remaining is required")
        supervise(args, config, output)
    elif args.mode == "preflight":
        preflight(
            config,
            output,
            config["expected_gpu"],
            config["expected_torch"],
            Path(config["body_reference"]),
        )
    elif args.mode == "train":
        print(json.dumps(training_block(config, output, args.stop_after)), flush=True)
    elif args.mode == "evaluate":
        evaluate_candidates(config, output, initial=args.initial)
    else:
        audit(output)


if __name__ == "__main__":
    main()
