"""Adaptation frontiers, paired effects, and identified crossed-body compatibility models.

The sampling unit is the training seed, never an evaluation episode or a null
graph draw. Unreached thresholds remain right-censored at the declared budget.
"""

from __future__ import annotations

import fcntl
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..train import threshold_experience
from ..util import atomic_json, digest_json


def _mean_interval(values, clusters, *, seed=0, replicates=2000):
    groups = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        if not np.isfinite(value):
            raise ValueError("Nonfinite outcome cannot enter statistical analysis")
        groups[str(cluster)].append(float(value))
    means = np.array([np.mean(items) for _, items in sorted(groups.items())])
    if not len(means):
        return {"mean": None, "ci95": None, "training_seeds": 0}
    interval = None
    if len(means) > 1:
        rng = np.random.default_rng(seed)
        samples = means[rng.integers(len(means), size=(replicates, len(means)))].mean(1)
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return {"mean": float(means.mean()), "ci95": interval, "training_seeds": len(means)}


def curve_metrics(curve, budget, threshold, confirmations):
    if not curve or budget <= 0:
        raise ValueError("Need an observed learning curve and positive experience budget")
    x = np.array([row["training_interactions"] for row in curve], dtype=float)
    if x[0] != 0 or np.any(np.diff(x) <= 0) or x[-1] != budget:
        raise ValueError("A complete curve must cover 0 through the exact budget in order")
    result = {}
    for field, label in (("mean_score", "score_auc"), ("success_rate", "success_auc")):
        values = np.array([row[field] for row in curve], dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Body scores and success rates must lie in [0,1]")
        result[label] = float(np.trapz(values, x) / budget)
    experience = threshold_experience(curve, threshold, confirmations, budget)
    result.update(
        experience=experience,
        restricted_experience=experience["first_crossing_step"] if experience["event"] else budget,
        final_validation_success=curve[-1]["success_rate"],
    )
    result["restricted_efficiency"] = 1 - result["restricted_experience"] / budget
    certification = next(
        (row for row in curve if row["training_interactions"] == experience["certification_step"]),
        None,
    )
    result["optimizer_steps_to_threshold"] = (
        certification.get("optimizer_steps") if certification else None
    )
    result["compute_seconds_to_threshold"] = (
        certification.get("training_wall_seconds") if certification else None
    )
    return result


def _live_lock(path):
    if not path.exists():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def collect_runs(root, *, include_smoke=False):
    root = Path(root)
    paths = (
        [root / "manifest.json"]
        if (root / "manifest.json").is_file()
        else sorted(root.rglob("manifest.json"))
    )
    rows, pending, excluded, seen = [], [], [], set()
    for path in paths:
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "compatibility-run-v1":
            continue
        config = manifest["config"]
        if not include_smoke and (
            config["purpose"] != "experiment" or manifest["evidence"] == "software_fixture_only"
        ):
            excluded.append(str(path.parent))
            continue
        identity = manifest["identity"]
        if identity in seen:
            continue  # Relocated exact resumes are not independent replicates.
        directory = path.parent
        completed, failure = directory / "result.json", directory / "failure.json"
        if _live_lock(directory / "run.lock"):
            pending.append({"run": str(directory), "reason": "live_training_lock"})
            continue
        failed = False
        if completed.exists():
            result = json.loads(completed.read_text())
            if result.get("identity") != identity or result.get("status") != "complete":
                raise ValueError(f"Result identity/status mismatch: {directory}")
            if result.get("manifest_sha256") and result["manifest_sha256"] != digest_json(manifest):
                raise ValueError(f"Run manifest changed after results were committed: {directory}")
            curve = json.loads((directory / "learning_curve.json").read_text())
        elif failure.exists() and json.loads(failure.read_text()).get("status") == "failed":
            record = json.loads(failure.read_text())
            if (
                record.get("identity") != identity
                or record.get("manifest_sha256") != digest_json(manifest)
                or record.get("exception") != "FloatingPointError"
                or record.get("fingerprint")
                != digest_json(
                    {key: value for key, value in record.items() if key != "fingerprint"}
                )
            ):
                raise ValueError(f"Numerical failure identity or seal mismatch: {directory}")
            failed = True
            # Intention-to-treat convention: a numerically unusable training
            # condition is assigned zero score, not silently removed. This is
            # explicit in every report and distinct from an unfinished run.
            result = {
                "parameters": manifest["parameters"],
                "optimizer_steps": record["optimizer_steps"],
                "training_wall_seconds": None,
            }
            curve = [
                {"training_interactions": n, "mean_score": 0.0, "success_rate": 0.0}
                for n in (0, config["training"]["interactions"])
            ]
        else:
            pending.append({"run": str(directory), "reason": "no_terminal_result"})
            continue
        seen.add(identity)
        controller, learning = config["controller"], config["training"]
        substrate, adapter = controller["substrate"], controller["adapter"]
        metadata, parameters = config.get("metadata", {}), result["parameters"]
        graph = manifest.get("topology") or {}
        label = metadata.get("connectome") or (
            "graph:" + graph["parent_graph"][:12] if graph else substrate["kind"]
        )
        row = {
            "identity": identity,
            "run": str(directory),
            "train_seed": config["train_seed"],
            "substrate_seed": substrate["seed"],
            "connectome": label,
            "native_body": metadata.get("native_body"),
            "body": config["body"]["name"],
            "task": config["body"]["task"],
            "kind": substrate["kind"],
            "topology": substrate["topology"],
            "initialization": substrate["initialization"],
            "sign_mode": substrate["dynamics"]["sign_mode"],
            "plasticity": substrate["plasticity"],
            "family": adapter["family"],
            "channels": adapter["channels"],
            "declared_budget": adapter["budget"],
            "adapter_parameters": parameters["adapter"]["allocated"],
            "adapter_trainable_parameters": sum(
                parameters[field]
                for field in ("encoder_trainable", "decoder_trainable", "port_trainable")
            ),
            "actor_parameters": parameters["total_trainable_parameters"],
            "width": parameters["adapter"]["width"],
            "rank": parameters["adapter"]["rank"],
            "encoder_depth": parameters["adapter"]["encoder_depth"],
            "decoder_depth": parameters["adapter"]["decoder_depth"],
            "experience_budget": learning["interactions"],
            "success_threshold": learning["success_threshold"],
            "training_failed": failed,
            "evidence": manifest["evidence"],
            "purpose": config["purpose"],
            "metadata": metadata,
            "optimizer_steps": result["optimizer_steps"],
            "training_wall_seconds": result["training_wall_seconds"],
            "observation_dim": manifest["observation_dim"],
            "action_dim": manifest["action_dim"],
            "state_dimension": parameters["state_dimension"],
            "graph_fingerprint": graph.get("parent_graph"),
            "body_fingerprint": manifest["body_fingerprint"],
            "perturbation": config.get("perturbation"),
            "initial_checkpoint": config.get("initial_checkpoint"),
            "curve": [{k: v for k, v in point.items() if k != "episodes"} for point in curve],
        }
        row.update(
            curve_metrics(
                curve,
                learning["interactions"],
                learning["success_threshold"],
                learning["threshold_confirmations"],
            )
        )
        if failed:
            row.update(
                final_score=0.0, success=0.0, selected_test_score=0.0, selected_test_success=0.0
            )
        else:
            final = result["final_checkpoint_test"]
            selected = result["selected_checkpoint_evaluation"]["test"]
            row.update(
                final_score=final["mean_score"],
                success=final["success_rate"],
                selected_test_score=selected["mean_score"],
                selected_test_success=selected["success_rate"],
            )
        rows.append(row)
    return {"rows": rows, "pending": pending, "excluded_smoke": excluded}


def paired_contrast(rows, left, right, *, keys, outcome="score_auc"):
    def select(filters):
        grouped = defaultdict(list)
        for row in rows:
            if all(row.get(key) == value for key, value in filters.items()):
                grouped[tuple(row[key] for key in keys)].append(row)
        return grouped

    a, b = select(left), select(right)
    pairs = []
    for key in sorted(a.keys() & b.keys(), key=str):
        if len({row["train_seed"] for row in [*a[key], *b[key]]}) != 1:
            raise ValueError("A paired contrast must retain the training-seed key")
        pairs.append(
            {
                "key": list(key),
                "train_seed": a[key][0]["train_seed"],
                "difference": float(
                    np.mean([r[outcome] for r in a[key]]) - np.mean([r[outcome] for r in b[key]])
                ),
                "actor_parameter_difference": float(
                    np.mean([r["actor_parameters"] for r in a[key]])
                    - np.mean([r["actor_parameters"] for r in b[key]])
                ),
            }
        )
    summary = _mean_interval([p["difference"] for p in pairs], [p["train_seed"] for p in pairs])
    seed_means = defaultdict(list)
    for item in pairs:
        seed_means[item["train_seed"]].append(item["difference"])
    values = np.array([np.mean(v) for v in seed_means.values()])
    effect = (
        float(values.mean() / values.std(ddof=1))
        if len(values) > 1 and values.std(ddof=1) > 0
        else None
    )
    return {
        "left": left,
        "right": right,
        "outcome": outcome,
        "pairs": pairs,
        "unmatched_left": len(a.keys() - b.keys()),
        "unmatched_right": len(b.keys() - a.keys()),
        "paired_effect_dz": effect,
        **summary,
    }


def capacity_frontiers(rows):
    keys = (
        "connectome",
        "kind",
        "topology",
        "initialization",
        "plasticity",
        "sign_mode",
        "body",
        "task",
        "family",
        "channels",
        "experience_budget",
        "train_seed",
        "substrate_seed",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        successful = [r for r in group if r["experience"]["event"] and not r["training_failed"]]
        minimum = min((r["adapter_parameters"] for r in successful), default=None)
        selected = [r for r in successful if r["adapter_parameters"] == minimum]
        complexity = {}
        for field in ("width", "rank", "encoder_depth", "decoder_depth", "actor_parameters"):
            complexity[field] = min(
                (r[field] for r in successful if r[field] is not None), default=None
            )
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "minimum_adapter_parameters": minimum,
                "capacity_censored": minimum is None,
                "largest_tested_adapter": max(r["adapter_parameters"] for r in group),
                "tested_capacities": sorted({r["adapter_parameters"] for r in group}),
                "minimum_successful_complexities": complexity,
                "selected_capacity_test_success": float(
                    np.mean([r["selected_test_success"] for r in selected])
                )
                if selected
                else None,
                "selection": "smallest tested capacity with a development-confirmed threshold crossing",
                "monotonicity_assumed": False,
            }
        )
    return result


def pareto_interfaces(rows):
    """Pareto filtering within one body/task and matched experience budget."""
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["connectome"],
                row["body"],
                row["task"],
                row["plasticity"],
                row["experience_budget"],
            )
        ].append(row)
    output = []
    for key, group in groups.items():
        options = defaultdict(list)
        for row in group:
            options[
                (
                    row["family"],
                    row["channels"],
                    row["adapter_parameters"],
                    row["topology"],
                    row["initialization"],
                )
            ].append(row)
        means = []
        for option, records in options.items():
            means.append(
                {
                    "family": option[0],
                    "channels": option[1],
                    "adapter_parameters": option[2],
                    "topology": option[3],
                    "initialization": option[4],
                    **{
                        metric: _mean_interval(
                            [r[metric] for r in records], [r["train_seed"] for r in records]
                        )["mean"]
                        for metric in (
                            "score_auc",
                            "success",
                            "final_score",
                            "restricted_experience",
                        )
                    },
                }
            )
        for candidate in means:

            def dominates(other):
                a = np.array(
                    [
                        -other["adapter_parameters"],
                        -other["restricted_experience"],
                        other["success"],
                        other["final_score"],
                    ]
                )
                b = np.array(
                    [
                        -candidate["adapter_parameters"],
                        -candidate["restricted_experience"],
                        candidate["success"],
                        candidate["final_score"],
                    ]
                )
                return np.all(a >= b) and np.any(a > b)

            candidate["pareto"] = not any(dominates(other) for other in means)
        output.append(
            {
                "connectome": key[0],
                "body": key[1],
                "task": key[2],
                "plasticity": key[3],
                "experience_budget": key[4],
                "interfaces": means,
                "interpretation": "exploratory empirical frontier; selection requires independent confirmation",
            }
        )
    return output


def _design(
    rows, covariates=("adapter_parameters", "actor_parameters", "experience_budget", "channels")
):
    columns, names = [np.ones(len(rows))], ["intercept"]
    for field in ("connectome", "body_task", "family", "plasticity"):
        # Body-task fixed effects control both levels without adding collinear
        # body dummies, including when task names are shared across bodies.
        values = [
            f"{row['body']}:{row['task']}" if field == "body_task" else str(row[field])
            for row in rows
        ]
        for level in sorted(set(values))[1:]:
            columns.append(np.array([value == level for value in values], dtype=float))
            names.append(f"{field}={level}")
    for field in covariates:
        values = np.log1p([row[field] for row in rows])
        # Keep a continuous covariate only if it adds information beyond fixed effects.
        if np.std(values) > 1e-12:
            candidate = np.column_stack([*columns, values])
            if np.linalg.matrix_rank(candidate) > len(columns):
                columns.append(values)
                names.append(f"log1p({field})")
    match = np.array([row["native_body"] == row["body"] for row in rows], dtype=float)
    columns.append(match)
    names.append("matched_pair")
    return np.column_stack(columns), names


def compatibility_regression(
    rows,
    *,
    outcome="score_auc",
    replicates=1000,
    seed=0,
    covariates=("adapter_parameters", "actor_parameters", "experience_budget", "channels"),
):
    rows = [row for row in rows if row["kind"] == "connectome" and row["topology"] == "real"]
    if not rows or any(row.get("native_body") is None for row in rows):
        return {
            "status": "unidentified",
            "reason": "Biological native-body labels and real-topology rows are required",
        }
    pairs = {(row["connectome"], row["body"]) for row in rows}
    connectomes, bodies = sorted({p[0] for p in pairs}), sorted({p[1] for p in pairs})
    missing = [(c, b) for c in connectomes for b in bodies if (c, b) not in pairs]
    if len(bodies) < 2 or len(connectomes) < 2 or missing:
        return {
            "status": "unidentified",
            "reason": "A complete crossed matrix is required",
            "missing_pairs": missing,
        }
    matrix, names = _design(rows, covariates)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        return {
            "status": "unidentified",
            "reason": "Matched-pair interaction is collinear with main effects",
        }
    response = np.array([row[outcome] for row in rows], dtype=float)
    if not np.isfinite(response).all():
        raise ValueError(
            "Regression requires finite outcomes; use budget-restricted efficiency for censored thresholds"
        )
    strata = [
        (
            r["connectome"],
            r["body"],
            r["train_seed"],
            r["family"],
            r["plasticity"],
            r["declared_budget"],
        )
        for r in rows
    ]
    counts = defaultdict(int)
    for value in strata:
        counts[value] += 1
    weights = np.array([1 / counts[value] for value in strata])

    def fit(indices):
        x, y, w = matrix[indices], response[indices], np.sqrt(weights[indices])
        if np.linalg.matrix_rank(x) != x.shape[1]:
            return None
        return np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]

    coefficients = fit(np.arange(len(rows)))
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row["train_seed"]].append(i)
    groups = list(clusters.values())
    rng, draws = np.random.default_rng(seed), []
    if len(groups) > 1:
        for _ in range(replicates):
            indices = np.concatenate(
                [groups[i] for i in rng.integers(len(groups), size=len(groups))]
            )
            estimate = fit(indices)
            if estimate is not None:
                draws.append(estimate[-1])
    interval = np.quantile(draws, [0.025, 0.975]).tolist() if draws else None
    additive = np.linalg.lstsq(
        matrix[:, :-1] * np.sqrt(weights)[:, None], response * np.sqrt(weights), rcond=None
    )[0]
    residuals = response - matrix[:, :-1] @ additive
    cell = defaultdict(list)
    for row, residual in zip(rows, residuals, strict=True):
        cell[(row["connectome"], row["body"])].append(float(residual))
    return {
        "status": "estimated",
        "outcome": outcome,
        "matched_pair_delta": float(coefficients[-1]),
        "ci95": interval,
        "training_seed_clusters": len(groups),
        "bootstrap_draws": len(draws),
        "coefficients": dict(zip(names, coefficients.tolist(), strict=True)),
        "residual_by_pair": [
            {"connectome": c, "body": b, "residual": float(np.mean(values))}
            for (c, b), values in sorted(cell.items())
        ],
        "weighting": "equal connectome-body-seed-family-plasticity-budget strata; tasks averaged within strata",
        "limitation": "inference is conditional on these connectomes and simulated bodies",
    }


def _figures(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    grouped = defaultdict(list)
    for row in rows:
        if row["kind"] in ("rnn", "gru"):
            # A common adapter size can wrap very different learned RNN sizes.
            # Their parameter/compute-matched contrasts are plotted separately.
            continue
        grouped[
            (
                row["body"],
                row["task"],
                row["family"],
                row["plasticity"],
                row["channels"],
                row["experience_budget"],
            )
        ].append(row)
    for group, records in sorted(grouped.items()):
        series = defaultdict(lambda: defaultdict(list))
        for row in records:
            label = f"{row['connectome']} / {row['topology']} / {row['initialization']}"
            if row["kind"] != "connectome":
                label = row["kind"]
            series[label][row["adapter_parameters"]].append(row)
        fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        for label, points in sorted(series.items()):
            x, y, low, high = [], [], [], []
            for capacity, subset in sorted(points.items()):
                summary = _mean_interval(
                    [r["selected_test_score"] for r in subset], [r["train_seed"] for r in subset]
                )
                x.append(capacity)
                y.append(summary["mean"])
                lo, hi = summary["ci95"] or [summary["mean"], summary["mean"]]
                low.append(lo)
                high.append(hi)
            axis.plot(x, y, marker="o", label=label)
            axis.fill_between(x, low, high, alpha=0.12)
        axis.set(
            xscale="log",
            ylim=(0, 1),
            xlabel="Actual allocated adapter parameters",
            ylabel="Held-out normalized performance",
            title=f"{group[0]}: {group[1]} · {group[2]} · {group[3]}",
        )
        axis.legend(fontsize=7, loc="best")
        stem = "capacity-" + digest_json(group)[:12]
        if any(
            r.get("purpose") == "smoke" or r.get("evidence") == "software_fixture_only"
            for r in records
        ):
            fig.suptitle("SOFTWARE CHECK — NOT RESEARCH EVIDENCE", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=180)
        plt.close(fig)
        figures.append({"group": list(group), "png": f"{stem}.png", "pdf": f"{stem}.pdf"})
    return figures


def analyze(root, output, *, include_smoke=False, figures=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collection = collect_runs(root, include_smoke=include_smoke)
    rows = collection["rows"]
    if not rows:
        raise ValueError(
            "No terminal paper-study runs; synthetic and smoke runs are excluded by default"
        )
    contrasts = []
    keys = (
        "connectome",
        "body",
        "task",
        "family",
        "declared_budget",
        "channels",
        "plasticity",
        "initialization",
        "sign_mode",
        "experience_budget",
        "train_seed",
        "substrate_seed",
    )
    for variant in ("degree_rewired", "community_rewired", "direction_shuffled", "random"):
        contrasts.append(
            paired_contrast(
                rows,
                {"kind": "connectome", "topology": "real"},
                {"kind": "connectome", "topology": variant},
                keys=keys,
            )
        )
    report = {
        "schema": "compatibility-analysis-v1",
        "terminal_runs": len(rows),
        "training_failures": sum(r["training_failed"] for r in rows),
        "failure_policy": "unusable failed training receives zero score and no threshold event; unfinished runs are pending",
        "pending": collection["pending"],
        "excluded_smoke": collection["excluded_smoke"],
        "capacity_frontiers": capacity_frontiers(rows),
        "pareto_interfaces": pareto_interfaces(rows),
        "topology_contrasts": contrasts,
        "compatibility": {
            metric: compatibility_regression(rows, outcome=metric)
            for metric in ("score_auc", "restricted_efficiency", "success", "final_score")
        },
        "figures": _figures(rows, output) if figures else [],
    }
    atomic_json(output / "runs.json", rows)
    atomic_json(output / "analysis.json", report)
    lines = [
        "# Connectome–body compatibility study",
        "",
        f"{len(rows)} terminal runs; {report['training_failures']} training failures; {len(collection['pending'])} unfinished runs.",
        "",
        "Confidence intervals resample training seeds. Episodes and repeated null draws are not independent training replicates.",
        "Unreached experience thresholds remain censored; capacity estimates are minima over tested values, with no monotonicity assumption.",
        "",
        "| Outcome | Matched-pair effect | 95% interval | Status |",
        "|---|---:|---|---|",
    ]
    for outcome, estimate in report["compatibility"].items():
        lines.append(
            f"| {outcome} | {estimate.get('matched_pair_delta', '—')} | {estimate.get('ci95')} | {estimate['status']} |"
        )
    if include_smoke:
        lines += [
            "",
            "This report explicitly includes software/smoke checks; it is not a scientific result.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
