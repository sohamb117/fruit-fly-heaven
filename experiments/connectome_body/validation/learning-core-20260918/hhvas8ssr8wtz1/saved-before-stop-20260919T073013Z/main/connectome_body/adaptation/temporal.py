"""Balanced temporal problems with a strict memory-free chance baseline."""

from __future__ import annotations

import numpy as np

from ..util import seed_for
from .trajectories import write_cache


def temporal_cache(path, task="memory", episodes=128, length=24, seed=0, split="train"):
    if task not in ("memory", "delayed_xor", "oscillator") or episodes % 4 or length < 8:
        raise ValueError("Use memory/delayed_xor/oscillator, a multiple of four episodes, and T>=8")
    rng = np.random.default_rng(seed_for(seed, f"temporal-{task}-{split}"))
    combinations = np.tile(np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]), (episodes // 4, 1))
    rng.shuffle(combinations)
    trajectories = []
    for index, (first, second) in enumerate(combinations):
        obs = np.zeros((length + 1, 3), dtype=np.float32)
        obs[0, 0] = first
        # Distinct input times make the final query independent of instantaneous inputs.
        if task == "delayed_xor":
            obs[length // 3, 0] = second
        elif task == "oscillator":
            obs[0, 1] = second
        obs[-5:-1, 2] = 1
        target = first if task == "memory" else -first * second
        actions = np.full((length, 1), target, dtype=np.float32)
        if task == "oscillator":
            t = np.arange(length)
            actions[:, 0] = np.sin(t * (0.15 + 0.05 * second) + (0 if first > 0 else np.pi))
            obs[:, 2] = 1
        mask = np.zeros(length, dtype=np.float32)
        mask[-4:] = 1
        trajectories.append(
            {
                "observations": obs,
                "actions": actions,
                "mask": mask,
                "truncated": np.arange(length) == length - 1,
                "scenario": {"task": task, "split": split, "seed": seed, "episode": index},
            }
        )
    return write_cache(
        path,
        trajectories,
        {
            "task": task,
            "split": split,
            "seed": seed,
            "evidence": "synthetic_temporal_sanity",
            "teacher_fingerprint": f"analytic-{task}-v1",
            "body_fingerprint": "temporal-fixture-v1",
            "chance_accuracy": 0.5 if task != "oscillator" else None,
            "design": "Balanced labels; every final query observation identical across target classes",
        },
    )
