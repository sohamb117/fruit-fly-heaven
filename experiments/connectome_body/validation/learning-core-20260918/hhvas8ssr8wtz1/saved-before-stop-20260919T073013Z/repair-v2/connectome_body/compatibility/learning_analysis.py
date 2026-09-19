"""Seed-paired regime x substrate contrasts, and observed equal-compute comparisons."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..util import atomic_json, digest_json
from .analysis import _mean_interval, curve_metrics


def paired_interactions(rows, outcome="test_score"):
    lookup = {
        (r["task"], r["regime"], r["variant"], r["seed"]): r
        for r in rows
        if r["experiment"] == "training_regime"
    }
    results = []
    for task in sorted({r["task"] for r in rows}):
        for control in ("degree_rewired", "gru"):
            for regime in ("bc", "ppo", "bc_ppo"):
                values = []
                seeds = []
                for seed in range(5):
                    a = lookup.get((task, regime, "real", seed))
                    b = lookup.get((task, regime, control, seed))
                    if a and b and a.get(outcome) is not None and b.get(outcome) is not None:
                        values.append(a[outcome] - b[outcome])
                        seeds.append(seed)
                results.append(
                    {
                        "task": task,
                        "control": control,
                        "regime": regime,
                        "contrast": "real_minus_control",
                        "outcome": outcome,
                        **_mean_interval(values, seeds),
                    }
                )
            for regime in ("bc", "bc_ppo"):
                values = []
                seeds = []
                for seed in range(5):
                    cells = [
                        lookup.get((task, g, v, seed))
                        for g, v in [
                            (regime, "real"),
                            (regime, control),
                            ("ppo", "real"),
                            ("ppo", control),
                        ]
                    ]
                    if all(c and c.get(outcome) is not None for c in cells):
                        a, b, c, d = [r[outcome] for r in cells]
                        values.append((a - b) - (c - d))
                        seeds.append(seed)
                results.append(
                    {
                        "task": task,
                        "control": control,
                        "regime": regime,
                        "contrast": "gap_change_relative_to_ppo",
                        "outcome": outcome,
                        **_mean_interval(values, seeds),
                    }
                )
    return results


def score_at_compute(curve, seconds):
    """No interpolation or future scores; require measured coverage of the cutoff."""
    if not curve or max(r["compute_seconds"] for r in curve) < seconds:
        return None
    eligible = [r for r in curve if r["compute_seconds"] <= seconds]
    return max(eligible, key=lambda r: r["compute_seconds"])["mean_score"] if eligible else None


def report_learning(plan_path, output):
    from .learning_study import read_plan

    plan = read_plan(plan_path)
    root = Path(plan_path).resolve().parent / "runs"
    rows = []
    pending = []
    failures = []
    for cell in plan["conditions"]:
        if cell["regime"] == "temporal":
            continue
        path = root / cell["execution_id"]
        result_path = path / "result.json"
        if not result_path.exists():
            pending.append(cell["id"])
            if (path / "failure.json").exists() or (path / "execution_error.json").exists():
                failures.append(cell["id"])
            continue
        r = json.loads(result_path.read_text())
        manifest = json.loads((path / "manifest.json").read_text())
        if r.get("identity") != manifest["identity"] or r.get("manifest_sha256") != digest_json(
            manifest
        ):
            raise ValueError("Result/manifest seal mismatch")
        curve = json.loads((path / "learning_curve.json").read_text())
        row = {
            k: cell[k]
            for k in [
                "id",
                "experiment",
                "task",
                "source",
                "variant",
                "plasticity",
                "seed",
                "regime",
            ]
        }
        row.update(
            test_score=r["selected_checkpoint_evaluation"]["test"]["mean_score"],
            test_return=r["selected_checkpoint_evaluation"]["test"]["mean_return"],
            test_success=r["selected_checkpoint_evaluation"]["test"]["success_rate"],
            ood=r["selected_checkpoint_evaluation"]["ood"],
            test_episodes=r["selected_checkpoint_evaluation"]["test"]["episodes"],
        )
        compute = []
        if cell["regime"] == "bc":
            row.update(
                expert_action_validation_mse=r["expert_action_validation_mse"],
                unique_expert_samples=r["unique_expert_samples"],
                expert_exposures=r["expert_exposures"],
                teacher_environment_interactions=r["teacher_environment_interactions"],
                optimizer_steps=r["optimizer_steps"],
                rl_interactions=0,
                total_optimization_seconds=r["optimization_seconds"],
                wall_seconds=r["wall_seconds"],
            )
            compute = [
                {"compute_seconds": x["optimization_seconds"], "mean_score": x["mean_score"]}
                for x in curve
            ]
        else:
            learning = manifest["config"]["training"]
            row.update(
                curve_metrics(
                    curve,
                    learning["interactions"],
                    learning["success_threshold"],
                    learning["threshold_confirmations"],
                )
            )
            pre = r.get("pretraining") or {}
            bc_time = pre.get("bc_optimization_seconds", 0)
            row.update(
                rl_interactions=r["training_interactions"],
                unique_expert_samples=pre.get("expert_samples", 0),
                expert_exposures=pre.get("expert_exposures", 0),
                teacher_environment_interactions=pre.get("teacher_environment_interactions", 0),
                expert_action_validation_mse=pre.get("expert_action_validation_mse"),
                total_optimization_seconds=bc_time + r["training_wall_seconds"],
                optimizer_steps=pre.get("bc_optimizer_steps", 0) + r["optimizer_steps"],
                wall_seconds=pre.get("bc_wall_seconds", 0) + r["total_wall_seconds"],
                post_bc_closed_loop=pre.get("bc_closed_loop"),
                total_environment_data_cost=pre.get("teacher_environment_interactions", 0)
                + r["total_environment_interactions"],
            )
            if pre:
                bc_curve = json.loads(
                    (root / f"{cell['policy_id']}-bc" / "learning_curve.json").read_text()
                )
                compute.extend(
                    {"compute_seconds": x["optimization_seconds"], "mean_score": x["mean_score"]}
                    for x in bc_curve
                )
            compute.extend(
                {
                    "compute_seconds": bc_time + x["training_wall_seconds"],
                    "mean_score": x["mean_score"],
                }
                for x in curve
            )
        row["compute_curve"] = compute
        rows.append(row)
    equal_compute = {}
    for cutoff in plan["config"]["compute_comparison"]["cutoffs_seconds"]:
        matched = [
            {**r, "compute_score": score_at_compute(r["compute_curve"], cutoff)} for r in rows
        ]
        equal_compute[str(cutoff)] = paired_interactions(matched, "compute_score")
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["experiment"],
                row["task"],
                row["source"],
                row["variant"],
                row["plasticity"],
                row["regime"],
            )
        ].append(row)
    summaries = [
        {
            "cell": list(key),
            "mean_score": float(np.mean([r["test_score"] for r in values])),
            "score_variance": float(np.var([r["test_score"] for r in values], ddof=1))
            if len(values) > 1
            else None,
            "training_seeds": len(values),
            "unsuccessful_policy_fraction": float(
                np.mean([r["test_success"] < 0.8 for r in values])
            ),
        }
        for key, values in groups.items()
    ]
    result = {
        "schema": "learning-regime-report-v1",
        "plan_fingerprint": plan["fingerprint"],
        "rows": rows,
        "pending": pending,
        "execution_failures": failures,
        "summaries": summaries,
        "paired_contrasts": paired_interactions(rows),
        "equal_compute_contrasts": equal_compute,
        "limitations": [
            "Five paired seeds; intervals resample training seeds, not episodes.",
            "Compute curves use only prior measured validation scores and common observed coverage.",
            "Infrastructure failures remain explicit and are not fabricated zero-reward trials.",
            "Final return is budget-limited; asymptotic performance requires a plateau.",
        ],
    }
    atomic_json(Path(output) / "learning-report.json", result)
    return {
        "completed_cells": len(rows),
        "pending_cells": len(pending),
        "failed_cells": len(failures),
    }
