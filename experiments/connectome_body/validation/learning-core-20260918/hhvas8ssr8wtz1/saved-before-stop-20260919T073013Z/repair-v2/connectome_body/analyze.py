"""Seed-clustered summaries, paired contrasts, and censored learning curves."""

from __future__ import annotations

import collections
import csv
import json
import os
from pathlib import Path

import numpy as np

from .policy import BASELINES
from .util import atomic_json, digest_json


def cluster_interval(values, clusters, seed=0, samples=2000):
    """Equal weight per training seed; port/null draws are not independent seeds."""
    grouped = collections.defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    means = np.array([np.mean(v) for _, v in sorted(grouped.items())])
    result = {
        "mean": float(means.mean()),
        "training_seeds": len(means),
        "runs": len(values),
        "ci95": None,
    }
    if len(means) >= 2:
        rng = np.random.default_rng(seed)
        draws = rng.choice(means, size=(samples, len(means)), replace=True).mean(1)
        result["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def survival(times, events, cap, weights=None):
    """Kaplan-Meier with events before censoring at a tied time; RMST to cap."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    if len(times) == 0 or cap <= 0 or np.any(times < 0) or np.any(times > cap):
        raise ValueError("Invalid censoring data")
    weights = np.ones(len(times)) if weights is None else np.asarray(weights, float)
    if weights.shape != times.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid survival weights")
    alive, last, area = 1.0, 0.0, 0.0
    curve = [{"interactions": 0.0, "not_yet_successful": 1.0}]
    for t in sorted(np.unique(times)):
        area += alive * (t - last)
        at_risk = float(weights[times >= t].sum())
        failures = float(weights[(times == t) & events].sum())
        if at_risk:
            alive *= 1 - failures / at_risk
        curve.append({"interactions": float(t), "not_yet_successful": float(alive)})
        last = t
    area += alive * (cap - last)
    return {
        "curve": curve,
        "restricted_mean_interactions": float(area),
        "reached_fraction": float(np.average(events, weights=weights)),
        "cap": cap,
        "events": int(events.sum()),
        "censored": int((~events).sum()),
    }


def learning_auc(curve, cap):
    x = np.array([r["training_interactions"] for r in curve], float)
    y = np.array([r["success_rate"] for r in curve], float)
    if x[0] != 0 or x[-1] != cap or np.any(np.diff(x) <= 0):
        raise ValueError("Completed learning curve must span [0, budget] monotonically")
    return float(np.trapz(y, x) / cap)


def load_results(root: Path, include_fixtures=False):
    rows, missing = [], []
    identities = set()
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "connectome-body-run-v1":
            continue
        result_path = manifest_path.parent / "result.json"
        if not result_path.exists():
            missing.append(str(manifest_path.parent))
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete" or result["identity"] != manifest["identity"]:
            raise ValueError(f"Invalid completed result: {result_path}")
        if result["evidence"] != "native_flybody_experiment" and not include_fixtures:
            continue
        if manifest["identity"] in identities:
            continue  # Copied/resumed copies of the same run are not replication.
        identities.add(manifest["identity"])
        config = manifest["config"]
        curve = json.loads((manifest_path.parent / "learning_curve.json").read_text())
        if manifest["graph"]:
            source = manifest["graph"]["provenance"]
            dataset = f"{source['dataset']} / {source.get('release', 'unversioned')} / {manifest['graph']['fingerprint'][:8]}"
        else:
            dataset = "baseline"
        protocol = digest_json(
            {
                "body": manifest["body_fingerprint"],
                "ppo": config["ppo"],
                "dynamics": config["dynamics"],
                "versions": manifest["versions"],
                "python": manifest["python"],
                "device": config["device"],
                "threads": config["threads"],
                "code": manifest["code_fingerprint"],
                "evidence": result["evidence"],
            }
        )
        row = {
            "path": str(manifest_path.parent),
            "protocol": protocol,
            "dataset": dataset,
            "task": config["body"]["task"],
            "substrate": config["substrate"],
            "train_seed": config["train_seed"],
            "substrate_seed": config["substrate_seed"],
            "budget": config["adapter_budget"],
            "success_threshold": config["ppo"]["success_threshold"],
            "actor_parameters": result["actor_parameters"],
            "training_interactions": result["training_interactions"],
            "evaluation_interactions": result["evaluation_interactions"],
            "total_environment_interactions": result["total_environment_interactions"],
            "critic_parameters": manifest["critic_parameters"],
            "actor_state_size": manifest["actor_state_size"],
            "evidence": result["evidence"],
            "test_success": result["evaluation"]["test"]["success_rate"],
            "test_score": result["evaluation"]["test"]["mean_score"],
            "ood_success": result["evaluation"]["ood"]["success_rate"],
            "generalization_gap": result["generalization_gap"],
            "fall_rate": result["evaluation"]["test"]["fall_rate"],
            "numerical_failure_rate": result["evaluation"]["test"]["numerical_failure_rate"],
            "learning_auc": learning_auc(curve, result["training_interactions"]),
            "training_wall_seconds": result["training_wall_seconds"],
            "experience": result["experience_to_threshold"],
            "curve": curve,
            "causal": result["causal_interventions"],
        }
        rows.append(row)
    return rows, missing


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        key = (row["protocol"], row["task"], row["dataset"], row["substrate"], row["budget"])
        groups[key].append(row)
    summary = []
    metrics = (
        "test_success",
        "test_score",
        "ood_success",
        "generalization_gap",
        "fall_rate",
        "numerical_failure_rate",
        "learning_auc",
        "training_wall_seconds",
    )
    for key, members in sorted(groups.items()):
        protocol, task, dataset, substrate, budget = key
        seeds = [r["train_seed"] for r in members]
        item = {
            "protocol": protocol,
            "task": task,
            "dataset": dataset,
            "substrate": substrate,
            "budget": budget,
            "success_threshold": members[0]["success_threshold"],
            "actor_parameters": float(np.mean([r["actor_parameters"] for r in members])),
            "evidence": members[0]["evidence"],
            **{m: cluster_interval([r[m] for r in members], seeds) for m in metrics},
        }
        events = [r["experience"]["event"] for r in members]
        times = [
            r["experience"]["certification_step"]
            if r["experience"]["event"]
            else r["experience"]["censor_step"]
            for r in members
        ]
        counts = collections.Counter(seeds)
        weights = np.array([1 / counts[seed] for seed in seeds])
        item["experience"] = survival(
            times, events, members[0]["training_interactions"], weights=weights
        )
        # Resample whole training-seed clusters, keeping all port/null draws.
        unique = sorted(set(seeds))
        if len(unique) >= 2:
            rng = np.random.default_rng(0)
            estimates = []
            for _ in range(1000):
                selected = rng.choice(unique, len(unique))
                indices = [i for seed in selected for i, value in enumerate(seeds) if value == seed]
                estimates.append(
                    survival(
                        np.array(times)[indices],
                        np.array(events)[indices],
                        members[0]["training_interactions"],
                        weights=weights[indices],
                    )["restricted_mean_interactions"]
                )
            item["experience"]["rmst_ci95"] = np.quantile(estimates, [0.025, 0.975]).tolist()
        else:
            item["experience"]["rmst_ci95"] = None
        item["causal_score_drop"] = {}
        for intervention in members[0]["causal"]:
            values = [r["test_score"] - r["causal"][intervention]["mean_score"] for r in members]
            item["causal_score_drop"][intervention] = cluster_interval(values, seeds)
        summary.append(item)
    return summary


def paired_contrasts(rows):
    result = []
    real_groups = collections.defaultdict(list)
    for row in rows:
        if row["substrate"] == "real":
            real_groups[(row["protocol"], row["task"], row["dataset"], row["budget"])].append(row)
    for (protocol, task, dataset, budget), real in sorted(real_groups.items()):
        for comparison in ("degree_shuffled", "matched_random", *BASELINES):
            candidates = [
                r
                for r in rows
                if r["protocol"] == protocol
                and r["task"] == task
                and r["budget"] == budget
                and r["substrate"] == comparison
                and (comparison in BASELINES or r["dataset"] == dataset)
            ]
            by_pair = {
                (r["train_seed"], 0 if comparison in BASELINES else r["substrate_seed"]): r
                for r in candidates
            }
            metrics = (
                "test_success",
                "test_score",
                "ood_success",
                "learning_auc",
                "capped_experience",
            )
            differences = {metric: [] for metric in metrics}
            seeds, used = [], 0
            for r in real:
                key = (r["train_seed"], 0 if comparison in BASELINES else r["substrate_seed"])
                other = by_pair.get(key)
                if other is not None:
                    for metric in metrics:
                        differences[metric].append(_metric(r, metric) - _metric(other, metric))
                    seeds.append(r["train_seed"])
                    used += 1
            if used:
                result.append(
                    {
                        "protocol": protocol,
                        "task": task,
                        "dataset": dataset,
                        "budget": budget,
                        "comparison": f"real-minus-{comparison}",
                        "paired_real_draws": used,
                        "unpaired_real_draws": len(real) - used,
                        **cluster_interval(differences["test_success"], seeds),
                        "metrics": {
                            metric: cluster_interval(values, seeds)
                            for metric, values in differences.items()
                        },
                    }
                )
    return result


def _metric(row, name):
    if name != "capped_experience":
        return row[name]
    experience = row["experience"]
    return experience["certification_step"] if experience["event"] else experience["censor_step"]


def cross_connectome_contrasts(rows):
    """Pair scenario/training seeds, averaging all port/null draws within seed."""
    from itertools import combinations

    result = []
    groups = collections.defaultdict(list)
    for row in rows:
        if row["substrate"] == "real":
            groups[row["protocol"], row["task"], row["budget"]].append(row)
    for (protocol, task, budget), members in sorted(groups.items()):
        for first, second in combinations(sorted({r["dataset"] for r in members}), 2):
            metrics = {}
            for metric in ("test_success", "ood_success", "learning_auc", "capped_experience"):
                by_seed = {}
                for dataset in (first, second):
                    values = collections.defaultdict(list)
                    for row in members:
                        if row["dataset"] == dataset:
                            values[row["train_seed"]].append(_metric(row, metric))
                    by_seed[dataset] = {seed: float(np.mean(v)) for seed, v in values.items()}
                common = sorted(by_seed[first].keys() & by_seed[second].keys())
                if common:
                    metrics[metric] = cluster_interval(
                        [by_seed[first][seed] - by_seed[second][seed] for seed in common], common
                    )
            if metrics:
                result.append(
                    {
                        "protocol": protocol,
                        "task": task,
                        "budget": budget,
                        "first": first,
                        "second": second,
                        "direction": "first minus second",
                        "metrics": metrics,
                        "interpretation": "Cross-connectome association on FlyBody; capped experience includes unsuccessful runs",
                    }
                )
    return result


def capacity_frontier(summary, required_run_fraction=0.8):
    """Descriptive grid frontier, not an optimization over arbitrary adapters."""
    groups = collections.defaultdict(list)
    for item in summary:
        key = (item["protocol"], item["task"], item["dataset"], item["substrate"])
        groups[key].append(item)
    result = []
    for key, members in sorted(groups.items()):
        members.sort(key=lambda r: r["actor_parameters"])
        qualified = [
            r
            for r in members
            if r["experience"]["reached_fraction"] >= required_run_fraction
            and r["test_success"]["mean"] >= r["success_threshold"]
        ]
        result.append(
            {
                "protocol": key[0],
                "task": key[1],
                "dataset": key[2],
                "substrate": key[3],
                "smallest_observed_successful_actor": qualified[0]["actor_parameters"]
                if qualified
                else None,
                "largest_tested_actor": members[-1]["actor_parameters"],
                "no_success_in_tested_grid": not qualified,
                "required_certified_run_fraction": required_run_fraction,
                "required_mean_test_success": members[0]["success_threshold"],
                "interpretation": "Empirical smallest successful tested size; not a proven global minimum or monotonicity claim",
            }
        )
    return result


def analyze(root: Path, output: Path, include_fixtures=False):
    rows, missing = load_results(root, include_fixtures)
    if not rows:
        raise ValueError(
            "No completed eligible experiments; validation requires --include-validation"
        )
    # Different dynamics/optimizers/source revisions get separate figures and
    # contrasts. Never silently pool sensitivity studies into a primary result.
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "connectome-body-analysis-v1",
        "completed_runs": len(rows),
        "incomplete_runs": missing,
        "groups": summary,
        "paired_contrasts": paired_contrasts(rows),
        "capacity_frontier": capacity_frontier(summary),
        "cross_connectome_contrasts": cross_connectome_contrasts(rows),
        "uncertainty": "95% percentile bootstrap over training-seed clusters; port/null draws stay together; unadjusted exploratory intervals",
        "interpretation": "Different specimens, species, coverage, graph size, and assumed dynamics can confound cross-connectome ranks; FlyBody alone does not identify a morphology interaction",
    }
    # Missing planned runs are also visible when they never started.
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        report["planned_runs"] = plan["planned_runs"]
        report["not_completed_run_ids"] = [
            r["id"] for r in plan["runs"] if not (root / "runs" / r["id"] / "result.json").exists()
        ]
    atomic_json(output / "summary.json", report)
    columns = [
        "dataset",
        "task",
        "substrate",
        "budget",
        "actor_parameters",
        "train_seed",
        "substrate_seed",
        "test_success",
        "test_score",
        "ood_success",
        "generalization_gap",
        "fall_rate",
        "learning_auc",
        "training_interactions",
        "evaluation_interactions",
        "total_environment_interactions",
        "critic_parameters",
        "actor_state_size",
        "training_wall_seconds",
        "evidence",
        "protocol",
        "path",
    ]
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _plots(rows, summary, output)
    text = [
        "# Connectome–FlyBody experiment report",
        "",
        f"Completed runs: {len(rows)}. Incomplete started runs: {len(missing)}.",
        "",
        "Software fixtures are marked explicitly. Confidence intervals resample training seeds, not episodes.",
        "",
        "| Task | Dataset | Substrate | Actual actor parameters | Test success | Reached threshold |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in summary:
        text.append(
            f"| {item['task']} | {item['dataset']} | {item['substrate']} | {item['actor_parameters']:.0f} | "
            f"{item['test_success']['mean']:.3f} | {item['experience']['reached_fraction']:.3f} |"
        )
    text += [
        "",
        report["uncertainty"],
        "",
        report["interpretation"],
        "",
        "See summary.json for paired effects, censoring, causal interventions, and protocol groups.",
    ]
    (output / "report.md").write_text("\n".join(text) + "\n")
    return report


def _plots(rows, summary, output):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    import matplotlib.pyplot as plt

    protocols = sorted(set(r["protocol"] for r in rows))
    for protocol in protocols:
        points = [r for r in summary if r["protocol"] == protocol]
        raw = [r for r in rows if r["protocol"] == protocol]
        task = raw[0]["task"]
        label = "VALIDATION ONLY — " if raw[0]["evidence"] != "native_flybody_experiment" else ""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
        for dataset, substrate in sorted(set((r["dataset"], r["substrate"]) for r in points)):
            selected = sorted(
                [r for r in points if (r["dataset"], r["substrate"]) == (dataset, substrate)],
                key=lambda r: r["actor_parameters"],
            )
            x = [r["actor_parameters"] for r in selected]
            name = _plot_label(dataset, substrate)
            for ax, metric in zip(axes[:2], ("test_success", "ood_success"), strict=True):
                y = np.array([r[metric]["mean"] for r in selected])
                ax.plot(x, y, "o-", label=name)
                intervals = [r[metric]["ci95"] for r in selected]
                if all(i is not None for i in intervals):
                    intervals = np.array(intervals)
                    ax.fill_between(x, intervals[:, 0], intervals[:, 1], alpha=0.12)
                ax.set(
                    xscale="log",
                    xlabel="Trainable actor parameters",
                    ylabel="Success rate",
                    ylim=(-0.03, 1.03),
                )
            axes[2].plot(
                x,
                [r["experience"]["restricted_mean_interactions"] for r in selected],
                "o-",
                label=name,
            )
        axes[0].set_title(f"{label}{task}: held-out scenarios")
        axes[1].set_title("Harder held-out scenarios")
        axes[2].set(
            xscale="log",
            xlabel="Trainable actor parameters",
            ylabel="Restricted mean interactions",
            title="Experience to certified threshold (censored)",
            ylim=(0, raw[0]["training_interactions"] * 1.03),
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
        prefix = f"{task}-{protocol[:8]}"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"capacity-{prefix}.{extension}", dpi=180)
        plt.close(fig)
        # Separate capacities instead of overlaying dozens of otherwise
        # indistinguishable learning curves in one crowded panel.
        for budget in sorted({r["budget"] for r in raw}):
            selected_raw = [r for r in raw if r["budget"] == budget]
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            for dataset, substrate in sorted(
                {(r["dataset"], r["substrate"]) for r in selected_raw}
            ):
                runs = [
                    r
                    for r in selected_raw
                    if (r["dataset"], r["substrate"]) == (dataset, substrate)
                ]
                steps = sorted({c["training_interactions"] for r in runs for c in r["curve"]})
                means = []
                for step in steps:
                    values, seeds = [], []
                    for run in runs:
                        value = next(
                            (
                                c["success_rate"]
                                for c in run["curve"]
                                if c["training_interactions"] == step
                            ),
                            None,
                        )
                        if value is not None:
                            values.append(value)
                            seeds.append(run["train_seed"])
                    means.append(cluster_interval(values, seeds, samples=100)["mean"])
                ax.plot(steps, means, label=_plot_label(dataset, substrate))
            ax.set(
                xlabel="Training environment interactions",
                ylabel="Validation success rate",
                ylim=(-0.03, 1.03),
                title=f"{label}{task}: learning curves, actor ceiling {budget:,}",
            )
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"learning-{prefix}-p{budget}.{extension}", dpi=180)
            plt.close(fig)


def _plot_label(dataset, substrate):
    if " / " in dataset:
        parts = dataset.split(" / ")
        dataset = f"{parts[0]} [{parts[-1]}]"
    return f"{dataset} / {substrate}"
