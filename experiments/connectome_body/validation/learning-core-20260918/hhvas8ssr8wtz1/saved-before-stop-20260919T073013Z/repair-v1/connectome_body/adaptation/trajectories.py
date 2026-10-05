"""Immutable, hashed, complete trajectory datasets shared by all substrates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..util import atomic_json, digest_file, digest_json


class TrajectoryCache:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        claimed = self.manifest["fingerprint"]
        if digest_json({k: v for k, v in self.manifest.items() if k != "fingerprint"}) != claimed:
            raise ValueError("Trajectory manifest changed")
        if self.manifest["schema"] != "connectome-trajectories-v1":
            raise ValueError("Unknown trajectory cache schema")
        if verify:
            for row in self.manifest["trajectories"]:
                if digest_file(self.path / row["file"]) != row["sha256"]:
                    raise ValueError("Trajectory bytes changed")

    @property
    def fingerprint(self):
        return self.manifest["fingerprint"]

    def load(self, index):
        with np.load(
            self.path / self.manifest["trajectories"][index]["file"], allow_pickle=False
        ) as f:
            return {k: f[k] for k in f.files}

    def __len__(self):
        return len(self.manifest["trajectories"])


def write_cache(path, trajectories, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Trajectory caches are immutable: {path}")
    path.mkdir(parents=True)
    rows, dimensions = [], None
    for idx, trajectory in enumerate(trajectories):
        obs = np.asarray(trajectory["observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        steps = len(actions)
        if obs.ndim != 2 or actions.ndim != 2 or len(obs) != steps + 1 or steps < 1:
            raise ValueError("Cache requires T+1 observations and T teacher actions per episode")
        shape = [obs.shape[-1], actions.shape[-1]]
        if dimensions is not None and dimensions != shape:
            raise ValueError("Trajectory schemas differ within a cache")
        dimensions = shape
        arrays = {"observations": obs, "actions": actions}
        for name in ("terminated", "truncated"):
            arrays[name] = np.asarray(trajectory.get(name, np.zeros(steps)), dtype=bool)
            if arrays[name].shape != (steps,):
                raise ValueError("Bad terminal array")
        if np.any(arrays["terminated"][:-1] | arrays["truncated"][:-1]):
            raise ValueError("An episode boundary occurs inside a stored trajectory")
        arrays["mask"] = np.asarray(trajectory.get("mask", np.ones(steps)), dtype=np.float32)
        if (
            arrays["mask"].shape != (steps,)
            or not np.isin(arrays["mask"], [0, 1]).all()
            or not arrays["mask"].sum()
        ):
            raise ValueError("Each trajectory needs at least one labeled action")
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Nonfinite trajectory data")
        for name in ("executed_actions", "rewards"):
            if name in trajectory:
                arrays[name] = np.asarray(trajectory[name], dtype=np.float32)
                expected = actions.shape if name == "executed_actions" else (steps,)
                if arrays[name].shape != expected or not np.isfinite(arrays[name]).all():
                    raise ValueError(f"Invalid {name}")
        filename = f"trajectory-{idx:06d}.npz"
        np.savez_compressed(path / filename, **arrays)
        rows.append(
            {
                "file": filename,
                "sha256": digest_file(path / filename),
                "steps": steps,
                "scenario": trajectory.get("scenario", {}),
                "metrics": trajectory.get("metrics", {}),
            }
        )
    if not rows:
        raise ValueError("Empty trajectory cache")
    manifest = {
        "schema": "connectome-trajectories-v1",
        "metadata": metadata,
        "observation_dim": dimensions[0],
        "action_dim": dimensions[1],
        "trajectories": rows,
        "interactions": sum(r["steps"] for r in rows),
    }
    manifest["fingerprint"] = digest_json(manifest)
    atomic_json(path / "manifest.json", manifest)
    return TrajectoryCache(path)


def assert_paired_schema(caches):
    reference = caches[0].manifest
    for cache in caches[1:]:
        for key in ("observation_dim", "action_dim"):
            if cache.manifest[key] != reference[key]:
                raise ValueError("Caches use different observation/action dimensions")
        for key in ("body_fingerprint", "teacher_fingerprint", "task"):
            if cache.manifest["metadata"].get(key) != reference["metadata"].get(key):
                raise ValueError(f"Caches disagree on {key}")


def assert_disjoint_scenarios(train, validation):
    if train.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation use the same cache")
    train_ids = {digest_json(r["scenario"]) for r in train.manifest["trajectories"]}
    val_ids = {digest_json(r["scenario"]) for r in validation.manifest["trajectories"]}
    if train_ids & val_ids:
        raise ValueError("Training and validation scenario overlap")
