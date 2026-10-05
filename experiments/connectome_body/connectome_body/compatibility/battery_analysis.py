"""Missing-aware reports and source-group-held-out temporal-to-drone prediction."""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..util import atomic_json
from .analysis import _mean_interval
from .predictors import _pair_mse, _ridge_fit, _ridge_predict

ASSOCIATIONS = {
    "short_memory": ("attitude", "hover"),
    "long_memory": ("sensor_delay", "sensor_dropout"),
    "nonlinear_memory": ("trajectory", "motor_degradation", "motor_failure"),
    "frequency": ("trajectory", "gust", "turbulence"),
    "multi_timescale": ("trajectory", "waypoints"),
    "context_integration": ("setpoint", "color_navigation"),
    "mackey_glass": ("turbulence", "pursuit"),
    "change_point": ("gust", "motor_degradation", "motor_failure"),
}


def _mean(values):
    return float(np.mean(values)) if values else None


def capability_cells(rows):
    """Aggregate seeds/severities within cells, never treat them as substrates."""
    grouped = defaultdict(lambda: defaultdict(list))
    metadata = {}
    for row in rows:
        # Artificial controls are descriptive anchors, not extra biological
        # reconstructions. All null variants remain in their source group.
        if row["variant"] in ("adapter_only", "rnn", "gru"):
            continue
        key = (row["source"], row["variant"], row["budget"], row["plasticity"])
        grouped[key][(row["suite"], row["task"])].append(row)
        metadata[key] = row
    cells = []
    for key, tasks in grouped.items():
        diagnostic, drone = {}, {}
        for (suite, task), records in tasks.items():
            if suite == "drone":
                drone[task] = _mean([r["score"] for r in records])
            else:
                diagnostic[task] = _mean([r["score"] for r in records])
                if task == "linear_memory":
                    for capability in ("short_memory", "long_memory"):
                        diagnostic[capability] = _mean(
                            [
                                r["metrics"][capability]
                                for r in records
                                if r["metrics"].get(capability) is not None
                            ]
                        )
        nonlinear = [
            2 * diagnostic[t] - 1 if t == "temporal_xor" else diagnostic[t]
            for t in ("temporal_xor", "narma10")
            if t in diagnostic
        ]
        if len(nonlinear) == 2:
            diagnostic["nonlinear_memory"] = _mean(nonlinear)
        reference = metadata[key]
        cells.append(
            {
                "key": list(key),
                "group": reference["group"],
                "diagnostic": diagnostic,
                "drone": drone,
                "neurons": reference["neurons"],
                "edges": reference["edges"],
                "capacity": key[2],
                "plasticity": key[3],
            }
        )
    return cells


def held_out_prediction(cells, alphas=(0.01, 0.1, 1.0, 10.0)):
    cells = [
        c
        for c in cells
        if len(c["drone"]) >= 2 and all(c["diagnostic"].get(k) is not None for k in ASSOCIATIONS)
    ]
    groups = sorted({c["group"] for c in cells})
    if len(groups) < 4:
        return {
            "status": "unidentifiable",
            "independent_source_groups": len(groups),
            "reason": "Need at least four source groups with complete core diagnostics and multiple drone tasks",
        }
    tasks = sorted(set.intersection(*(set(c["drone"]) for c in cells)))
    if len(tasks) < 2:
        return {"status": "unidentifiable", "reason": "No common multi-task control panel"}
    regimes = sorted({c["plasticity"] for c in cells})
    base, enhanced, y, labels = [], [], [], []
    for cell in cells:
        overall = np.mean([cell["drone"][t] for t in tasks])
        diag = np.array([cell["diagnostic"][k] for k in ASSOCIATIONS])
        for task in tasks:
            values = [
                np.log1p(cell["neurons"]),
                np.log1p(cell["edges"]),
                np.log1p(cell["capacity"]),
            ]
            values += [float(cell["plasticity"] == r) for r in regimes]
            values += [float(task == t) for t in tasks]
            base.append(values)
            # Raw capabilities plus preregistered task interactions. Response is
            # task-relative performance, so generic across-task ability is removed.
            relevant = np.array([task in targets for targets in ASSOCIATIONS.values()])
            enhanced.append(values + diag.tolist() + (diag * relevant).tolist())
            y.append(cell["drone"][task] - overall)
            labels.append(cell["group"])
    y, labels = np.asarray(y), np.asarray(labels)
    results = {}
    for name, x in (
        ("size_capacity_task", np.asarray(base)),
        ("diagnostic_augmented", np.asarray(enhanced)),
    ):
        prediction = np.empty_like(y)
        folds = []
        for holdout in groups:
            train, test = labels != holdout, labels == holdout
            train_groups = sorted(set(labels[train]))
            errors = []
            for alpha in alphas:
                losses = []
                for inner in train_groups:
                    fit, valid = train & (labels != inner), train & (labels == inner)
                    model = _ridge_fit(x[fit], y[fit], labels[fit], alpha)
                    losses.append(
                        _pair_mse(y[valid], _ridge_predict(model, x[valid]), labels[valid])
                    )
                errors.append(float(np.mean(losses)))
            alpha = alphas[int(np.argmin(errors))]
            model = _ridge_fit(x[train], y[train], labels[train], alpha)
            prediction[test] = _ridge_predict(model, x[test])
            folds.append(
                {
                    "held_out": holdout,
                    "training_groups": train_groups,
                    "alpha": alpha,
                    "mse": float(np.mean((prediction[test] - y[test]) ** 2)),
                }
            )
        results[name] = {
            "group_balanced_mse": _pair_mse(y, prediction, labels),
            "folds": folds,
            "predictions": prediction.tolist(),
        }
    return {
        "status": "measured_exploratory",
        "source_groups": groups,
        "tasks": tasks,
        "response": "within-cell task score minus common-panel overall score",
        "actual": y.tolist(),
        "groups": labels.tolist(),
        "models": results,
        "mse_improvement": results["size_capacity_task"]["group_balanced_mse"]
        - results["diagnostic_augmented"]["group_balanced_mse"],
    }


def specificity(cells, seed=0):
    """Whole-source permutation tests for predeclared target/non-target contrasts."""
    results = []
    rng = np.random.default_rng(seed)
    for feature, targets in ASSOCIATIONS.items():
        eligible = [
            c
            for c in cells
            if c["diagnostic"].get(feature) is not None
            and any(t in c["drone"] for t in targets)
            and any(t not in targets for t in c["drone"])
        ]
        # Balance adapter capacities, regimes and null variants across sources.
        grouped = defaultdict(dict)
        for cell in eligible:
            stratum = (cell["key"][1], cell["capacity"], cell["plasticity"])
            grouped[cell["group"]][stratum] = cell
        common = set.intersection(*(set(g) for g in grouped.values())) if grouped else set()
        x, y, names = [], [], []
        for name, strata in sorted(grouped.items()):
            if not common:
                continue
            dx, contrasts = [], []
            for stratum in sorted(common):
                c = strata[stratum]
                dx.append(c["diagnostic"][feature])
                target = [v for t, v in c["drone"].items() if t in targets]
                other = [v for t, v in c["drone"].items() if t not in targets]
                contrasts.append(np.mean(target) - np.mean(other))
            x.append(np.mean(dx))
            y.append(np.mean(contrasts))
            names.append(name)
        if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            results.append(
                {"capability": feature, "status": "unidentifiable", "source_groups": names}
            )
            continue
        x, y = np.asarray(x), np.asarray(y)
        correlation = float(np.corrcoef(x, y)[0, 1])
        if len(x) <= 8:
            permutations = itertools.permutations(range(len(x)))
            stats = [abs(float(np.corrcoef(x[list(p)], y)[0, 1])) for p in permutations]
            pvalue = float(np.mean(np.asarray(stats) >= abs(correlation) - 1e-12))
        else:
            stats = [abs(float(np.corrcoef(rng.permutation(x), y)[0, 1])) for _ in range(999)]
            pvalue = (1 + sum(v >= abs(correlation) for v in stats)) / 1000
        results.append(
            {
                "capability": feature,
                "status": "measured_exploratory",
                "source_groups": names,
                "correlation": correlation,
                "p_two_sided": pvalue,
                "balanced_strata": len(common),
            }
        )
    measured = sorted([r for r in results if "p_two_sided" in r], key=lambda r: r["p_two_sided"])
    running = 0.0
    # Correct over all eight preregistered hypotheses, including unavailable ones.
    for i, row in enumerate(measured):
        running = max(running, min(1.0, (len(ASSOCIATIONS) - i) * row["p_two_sided"]))
        row["p_holm"] = running
    return results


def report_battery(plan_path, output):
    from .battery import execution_key, read_battery

    plan = read_battery(plan_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, inventory = [], []
    run_root = Path(plan_path).resolve().parent / "runs"
    for condition in plan["conditions"]:
        result_path = run_root / condition["execution_id"] / "result.json"
        item = {
            k: condition[k]
            for k in (
                "id",
                "suite",
                "task",
                "substrate",
                "variant",
                "plasticity",
                "budget",
                "seed",
                "status",
            )
        }
        item["measurement"] = "unmeasured"
        if result_path.exists():
            manifest = json.loads((result_path.parent / "manifest.json").read_text())
            result = json.loads(result_path.read_text())
            if (
                execution_key(manifest["config"]) != condition["execution_id"]
                or result["identity"] != manifest["identity"]
                or manifest["code_fingerprint"] != plan["code_fingerprint"]
            ):
                raise ValueError("Result does not belong to the frozen battery condition")
            if result.get("status") != "complete":
                raise ValueError("Nonterminal result cannot enter measured report")
            if condition["suite"] == "drone":
                evaluation = result["selected_checkpoint_evaluation"]["test"]
                metrics, score = evaluation, evaluation["mean_score"]
                threshold = result["experience_to_threshold"]
                exposures = result["training_interactions"]
                threshold_score = evaluation["success_rate"]
            else:
                metrics, score = result["test"], result["test"]["score"]
                threshold, exposures = (
                    result["exposures_to_threshold"],
                    result["optimization_exposures"],
                )
                threshold_score = score
            graph = plan["inputs"][condition["substrate"]]
            records.append(
                {
                    **item,
                    "measurement": "measured",
                    "source": condition["substrate"],
                    "group": graph["declaration"].get("source_group", condition["substrate"]),
                    "score": score,
                    "metrics": metrics,
                    "threshold": threshold,
                    "exposures": exposures,
                    "actual_trainable": result["parameters"]["total_trainable_parameters"],
                    "actual_adapter": result["parameters"]["adapter"]["allocated"],
                    "threshold_score": threshold_score,
                    "neurons": graph["neurons"],
                    "edges": graph["edges"],
                    "setting": condition["setting"],
                }
            )
            item["measurement"] = "measured"
        inventory.append(item)
    cells = capability_cells(records)
    report = {
        "schema": "substrate-battery-report-v1",
        "plan_fingerprint": plan["fingerprint"],
        "inventory": dict(Counter(r["measurement"] for r in inventory)),
        "conditions": inventory,
        "results": records,
        "capability_cells": cells,
        "held_out_prediction": held_out_prediction(cells),
        "specificity": specificity(cells),
        "adaptation_frontiers": frontiers(records),
        "paired_control_effects": paired_effects(records),
        "interpretation": "No scores imputed for missing runs. Training seeds/null realizations are not independent source substrates.",
    }
    atomic_json(output / "report.json", report)
    with (output / "inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]) if inventory else ["id"])
        writer.writeheader()
        writer.writerows(inventory)
    lines = [
        "# Drone and temporal substrate battery",
        "",
        f"Measured conditions: {len(records)} / {len(inventory)}.",
        "",
        "Missing runs are not assigned a score. Biological native-pair effects remain in the original three-body analysis.",
        "",
        f"Predictive analysis: {report['held_out_prediction']['status']}.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if records:
        _figures(records, output)
    return {
        "report": str(output / "report.json"),
        "measured": len(records),
        "conditions": len(inventory),
        "prediction": report["held_out_prediction"]["status"],
    }


def _figures(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for suite in ("drone", "temporal"):
        selected = [r for r in rows if r["suite"] == suite]
        if not selected:
            continue
        tasks = sorted({r["task"] for r in selected})
        labels = sorted(
            {(r["source"], r["variant"], r["budget"], r["plasticity"]) for r in selected}
        )
        matrix = np.full((len(labels), len(tasks)), np.nan)
        for i, label in enumerate(labels):
            for j, task in enumerate(tasks):
                values = [
                    r["score"]
                    for r in selected
                    if (r["source"], r["variant"], r["budget"], r["plasticity"]) == label
                    and r["task"] == task
                ]
                if values:
                    matrix[i, j] = np.mean(values)
        fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.5), max(4, len(labels) * 0.25)))
        plot = ax.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            vmin=-1 if suite == "temporal" else 0,
            vmax=1,
        )
        ax.set_xticks(range(len(tasks)), tasks, rotation=70, ha="right")
        ax.set_yticks(range(len(labels)), [" / ".join(map(str, label)) for label in labels])
        ax.set_title(f"{suite}: held-out scores (blank = unmeasured)")
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f"{suite}-heatmap.png", dpi=160)
        plt.close(fig)


def frontiers(rows, threshold=0.8):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["suite"],
            row["task"],
            row["source"],
            row["variant"],
            row["plasticity"],
            row["seed"],
            json.dumps(row["setting"], sort_keys=True),
        )
        grouped[key].append(row)
    records = []
    for key, group in sorted(grouped.items()):
        by_experience = defaultdict(list)
        for row in group:
            by_experience[row["exposures"]].append(row)
        for experience, points in sorted(by_experience.items()):
            successful = [r["actual_adapter"] for r in points if r["threshold_score"] >= threshold]
            records.append(
                {
                    "suite": key[0],
                    "task": key[1],
                    "source": key[2],
                    "variant": key[3],
                    "plasticity": key[4],
                    "seed": key[5],
                    "setting": json.loads(key[6]),
                    "experience": experience,
                    "experience_unit": "environment interactions"
                    if key[0] == "drone"
                    else "optimization input exposures",
                    "threshold": threshold,
                    "minimum_observed_successful_adapter": min(successful) if successful else None,
                    "right_censored": not successful,
                    "largest_observed_adapter": max(r["actual_adapter"] for r in points),
                    "grid_points": [
                        {
                            "requested_capacity": r["budget"],
                            "actual_adapter": r["actual_adapter"],
                            "total_trainable": r["actual_trainable"],
                            "test_score": r["threshold_score"],
                            "experience_to_threshold": r["threshold"],
                        }
                        for r in sorted(points, key=lambda r: r["actual_adapter"])
                    ],
                    "scope": "minimum over measured grid, no interpolation or imputation of missing capacities",
                }
            )
    return records


def paired_effects(rows):
    indexed = {}
    for row in rows:
        key = (
            row["suite"],
            row["task"],
            row["source"],
            row["budget"],
            row["plasticity"],
            row["seed"],
            json.dumps(row["setting"], sort_keys=True),
            row["variant"],
        )
        if key in indexed:
            raise ValueError("Duplicate seed/cell in paired effects")
        indexed[key] = row
    grouped = defaultdict(list)
    for key, real in indexed.items():
        if key[-1] != "real":
            continue
        for control in (
            "degree_rewired",
            "community_rewired",
            "random",
            "adapter_only",
            "rnn",
            "gru",
        ):
            null = indexed.get((*key[:-1], control))
            if null is not None:
                grouped[(*key[:5], key[6], control)].append((key[5], real["score"] - null["score"]))
    return [
        {
            "suite": key[0],
            "task": key[1],
            "source": key[2],
            "capacity": key[3],
            "plasticity": key[4],
            "setting": json.loads(key[5]),
            "contrast": f"real minus {key[6]}",
            **_mean_interval([x[1] for x in pairs], [x[0] for x in pairs]),
        }
        for key, pairs in sorted(grouped.items())
    ]
