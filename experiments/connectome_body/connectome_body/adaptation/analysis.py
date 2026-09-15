"""Adaptation frontiers, paired topology advantage, and a prespecified MVP decision."""

from __future__ import annotations

import csv
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..body import PROJECT
from ..util import atomic_json, digest_file, digest_json


def interval(values, seed=0, samples=2000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite, nonempty seed measurements required")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(samples, len(values)))].mean(-1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(lo), float(hi)],
        "training_seeds": len(values),
        "between_seed_variation_estimable": len(values) > 1,
    }


def protocol_key(manifest):
    config = manifest["config"]
    adapter = config["adapter"]
    return digest_json(
        {
            "source": manifest["source_identity"],
            "body": manifest["body_fingerprint"],
            "teacher": manifest["teacher_fingerprint"],
            "data": manifest["data"],
            "offline": config["offline"],
            "dagger": config["dagger_optimization"],
            "online_budgets": config["online_budgets"],
            "beta": config.get("dagger_beta", 0),
            "device": adapter["device"],
            "threads": adapter["threads"],
            "channels": adapter["channels"],
            "support": adapter["support"],
            "rate": adapter["rate"],
            "evaluation_counts": [config["validation_episodes"], config["test_episodes"]],
            "success_threshold": config.get("success_threshold", 0.8),
            "evidence": manifest["evidence"],
        }
    )[:16]


def adaptation_records(root, include_validation=False):
    records, seen = [], set()
    root = Path(root)
    if (root / "plan.json").exists():
        plan = json.loads((root / "plan.json").read_text())
        paths = sorted({Path(row["output"]) / "result.json" for row in plan["runs"]})
    else:
        paths = sorted(root.rglob("result.json"))
    for path in paths:
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if "test_frontier" not in result or result.get("status") != "complete":
            continue
        if result["evidence"] != "native_hover_adaptation_experiment" and not include_validation:
            continue
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if (
            digest_json({k: v for k, v in manifest.items() if k != "identity"})
            != manifest["identity"]
        ):
            raise ValueError("Experiment manifest changed")
        if result["identity"] != manifest["identity"]:
            raise ValueError("Result identity differs from manifest")
        if result["identity"] in seen:
            continue
        seen.add(result["identity"])
        config = manifest["config"]
        dataset = config.get("dataset", "baseline" if result["graph"] is None else "unnamed")
        if result["graph"] is not None:
            dataset += "/" + result["graph"]["graph_fingerprint"][:8]
        records.append(
            {
                "path": str(path.parent),
                "protocol": protocol_key(manifest),
                "dataset": dataset,
                "variant": config["adapter"]["variant"],
                "budget": config["adapter"]["budget"],
                "seed": config["adapter"]["seed"],
                "actual_parameters": result["parameters"]["actual"],
                "result": result,
                "source_identity": manifest["source_identity"],
                "result_sha256": digest_file(path),
                "threshold": config.get("success_threshold", 0.8),
            }
        )
    return records


def mvp_decision(records, margin=0.05, required_seeds=(0, 1, 2), budgets=(5000, 80000)):
    primary = [r for r in records if r["dataset"].startswith("banc/") or r["dataset"] == "baseline"]
    if len({r["protocol"] for r in primary}) != 1:
        return {"go": False, "reason": "Need one complete, compatible MVP protocol"}
    if len({r["dataset"] for r in primary if r["dataset"].startswith("banc/")}) != 1:
        return {"go": False, "reason": "Do not pool distinct BANC releases for the MVP gate"}
    keys = [(r["variant"], r["budget"], r["seed"]) for r in primary]
    if len(keys) != len(set(keys)):
        return {"go": False, "reason": "Duplicate attempts for an MVP seed/cell"}
    index = {(r["variant"], r["budget"], r["seed"]): r for r in primary}
    variants = ("real", "degree_shuffled", "matched_random", "adapter_only")
    missing = [
        (v, p, s)
        for v in variants
        for p in budgets
        for s in required_seeds
        if (v, p, s) not in index
    ]
    if missing:
        return {"go": False, "reason": "MVP matrix incomplete", "missing": missing}
    if any(r["result"]["evidence"] != "native_hover_adaptation_experiment" for r in primary):
        return {"go": False, "reason": "Engineering/smoke evidence cannot open the scientific gate"}
    low = min(budgets)
    real = [index["real", low, s]["result"] for s in required_seeds]
    useful = np.mean([r["test"]["success"] for r in real]) >= 0.8
    controls = {}
    for variant in variants[1:]:
        other = [index[variant, low, s]["result"] for s in required_seeds]
        differences = np.array(
            [a["test"]["score"] - b["test"]["score"] for a, b in zip(real, other)]
        )
        cost_real = np.mean([r["experience"]["online_interactions"] for r in real])
        cost_other = np.mean([r["experience"]["online_interactions"] for r in other])
        separation = float(differences.mean()) >= margin and int(np.sum(differences > 0)) >= 2
        efficiency = (
            cost_other > 0
            and cost_real <= 0.8 * cost_other
            and all(r["experience"]["event"] for r in real)
        )
        controls[variant] = {
            "score_effect": interval(differences),
            "positive_pairs": int(np.sum(differences > 0)),
            "score_separation": bool(separation),
            "experience_saving": bool(efficiency),
        }
    go = useful and all(v["score_separation"] or v["experience_saving"] for v in controls.values())
    return {
        "go": bool(go),
        "reason": "MVP advancement rule met"
        if go
        else "MVP did not meet the prespecified advancement rule",
        "low_capacity": low,
        "real_success_at_least_0_8": bool(useful),
        "controls": controls,
        "rule": "Complete 24-cell MVP; low-capacity BANC success >=0.8; versus each control, score gain >=0.05 with >=2 positive pairs, or >=20% capped online-experience saving with all real seeds crossing",
        "interpretation": "Compute allocation decision; exploratory, not a confirmatory topology claim",
    }


def analyze(root, output, include_validation=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = adaptation_records(root, include_validation)
    rows, grouped = [], defaultdict(list)
    for record in records:
        result = record["result"]
        for stage in result["test_frontier"]:
            row = {
                key: record[key]
                for key in ("protocol", "dataset", "variant", "budget", "seed", "actual_parameters")
            }
            row.update(
                online_interactions=stage["online_interactions"],
                total_training_experience=stage["total_training_experience"],
                test_success=stage["test"]["success"],
                test_score=stage["test"]["score"],
            )
            rows.append(row)
            grouped[
                record["protocol"],
                record["dataset"],
                record["variant"],
                record["budget"],
                stage["online_interactions"],
            ].append(row)
    fields = [
        "protocol",
        "dataset",
        "variant",
        "budget",
        "seed",
        "actual_parameters",
        "online_interactions",
        "total_training_experience",
        "test_success",
        "test_score",
    ]
    with (output / "frontier.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = []
    for key, values in sorted(grouped.items()):
        if len({r["seed"] for r in values}) != len(values):
            raise ValueError(
                "Multiple distinct runs for the same seed/cell; do not count extra attempts as independent seeds"
            )
        groups.append(
            dict(zip(("protocol", "dataset", "variant", "budget", "online_interactions"), key))
            | {
                "parameters": values[0]["actual_parameters"],
                "success": interval([r["test_success"] for r in values]),
                "score": interval([r["test_score"] for r in values]),
            }
        )
    paired, affinity, compensation, lesions = [], [], [], []
    protocols = sorted({r["protocol"] for r in records})
    for protocol in protocols:
        entries = [r for r in records if r["protocol"] == protocol]
        index = {(r["dataset"], r["variant"], r["budget"], r["seed"]): r for r in entries}
        datasets = sorted({r["dataset"] for r in entries if r["variant"] == "real"})
        budgets = sorted({r["budget"] for r in entries})
        for dataset, budget, control in itertools.product(
            datasets,
            budgets,
            ("degree_shuffled", "matched_random", "adapter_only", "trainable_gru"),
        ):
            other_dataset = "baseline" if control in ("adapter_only", "trainable_gru") else dataset
            seeds = sorted(
                {
                    r["seed"]
                    for r in entries
                    if r["dataset"] == dataset and r["variant"] == "real" and r["budget"] == budget
                }
            )
            effects = []
            for seed in seeds:
                a, b = (
                    index.get((dataset, "real", budget, seed)),
                    index.get((other_dataset, control, budget, seed)),
                )
                if a and b:
                    effects.append(a["result"]["test"]["score"] - b["result"]["test"]["score"])
            if effects:
                paired.append(
                    {
                        "protocol": protocol,
                        "dataset": dataset,
                        "budget": budget,
                        "control": control,
                        "score_effect": interval(effects),
                    }
                )
        for first, second in itertools.combinations(datasets, 2):
            for budget in budgets:
                gaps, excess = [], []
                seeds = sorted({r["seed"] for r in entries})
                for seed in seeds:
                    a, b = (
                        index.get((first, "real", budget, seed)),
                        index.get((second, "real", budget, seed)),
                    )
                    ar, br = (
                        index.get((first, "degree_shuffled", budget, seed)),
                        index.get((second, "degree_shuffled", budget, seed)),
                    )
                    if a and b:
                        gap = a["result"]["test"]["score"] - b["result"]["test"]["score"]
                        gaps.append(gap)
                        if ar and br:
                            excess.append(
                                gap - ar["result"]["test"]["score"] + br["result"]["test"]["score"]
                            )
                if gaps:
                    effect = interval(gaps)
                    compensation.append(
                        {
                            "protocol": protocol,
                            "first": first,
                            "second": second,
                            "budget": budget,
                            "score_gap": effect,
                            "within_0_05_equivalence_band": bool(
                                effect["ci95"][0] >= -0.05 and effect["ci95"][1] <= 0.05
                            ),
                        }
                    )
                if excess:
                    affinity.append(
                        {
                            "protocol": protocol,
                            "first": first,
                            "second": second,
                            "budget": budget,
                            "difference_in_biological_advantage": interval(excess),
                        }
                    )
        for dataset, variant, budget, name in sorted(
            {
                (r["dataset"], r["variant"], r["budget"], name)
                for r in entries
                for name in r["result"]["interventions"]
            }
        ):
            subset = [
                r
                for r in entries
                if (r["dataset"], r["variant"], r["budget"]) == (dataset, variant, budget)
                and name in r["result"]["interventions"]
            ]
            drops = [
                r["result"]["test"]["score"] - r["result"]["interventions"][name]["score"]
                for r in subset
            ]
            lesions.append(
                {
                    "protocol": protocol,
                    "dataset": dataset,
                    "variant": variant,
                    "budget": budget,
                    "intervention": name,
                    "paired_score_drop": interval(drops),
                }
            )
    capacities = []
    for protocol, dataset, variant, experience in sorted(
        {(g["protocol"], g["dataset"], g["variant"], g["online_interactions"]) for g in groups}
    ):
        candidates = [
            g
            for g in groups
            if (g["protocol"], g["dataset"], g["variant"], g["online_interactions"])
            == (protocol, dataset, variant, experience)
        ]
        threshold = next(r["threshold"] for r in records if r["protocol"] == protocol)
        successful = [g["parameters"] for g in candidates if g["success"]["mean"] >= threshold]
        capacities.append(
            {
                "protocol": protocol,
                "dataset": dataset,
                "variant": variant,
                "online_interactions": experience,
                "P_tau": min(successful) if successful else None,
                "threshold": threshold,
                "right_censored": not successful,
                "largest_tested_capacity": max(g["parameters"] for g in candidates),
                "definition": "smallest tested actual capacity reaching mean held-out success; descriptive, no interpolation",
            }
        )
    experience = []
    for key in sorted({(r["protocol"], r["dataset"], r["variant"], r["budget"]) for r in records}):
        subset = [
            r for r in records if (r["protocol"], r["dataset"], r["variant"], r["budget"]) == key
        ]
        censored_costs = [r["result"]["experience"]["online_interactions"] for r in subset]
        curve_areas = []
        for r in subset:
            curve = r["result"]["test_frontier"]
            x = [v["online_interactions"] for v in curve]
            y = [v["test"]["success"] for v in curve]
            curve_areas.append(float(np.trapz(y, x) / (x[-1] - x[0])) if x[-1] > x[0] else y[0])
        experience.append(
            dict(zip(("protocol", "dataset", "variant", "budget"), key))
            | {
                "restricted_mean_online_experience": interval(censored_costs),
                "censored_seeds": sum(not r["result"]["experience"]["event"] for r in subset),
                "normalized_heldout_learning_auc": interval(curve_areas),
                "interpretation": "Common administrative censor cap; failed learners retained at cap; conditional on shared offline data",
            }
        )
    planned, coverage = None, None
    if (Path(root) / "plan.json").exists():
        plan = json.loads((Path(root) / "plan.json").read_text())
        planned = plan["planned_runs"]
        coverage = {"not_started": [], "incomplete": [], "completed": []}
        for row in plan["runs"]:
            path = Path(row["output"])
            state = (
                "completed"
                if (path / "result.json").exists()
                else "incomplete"
                if (path / "manifest.json").exists()
                else "not_started"
            )
            coverage[state].append(row["name"])
    summary = {
        "completed_runs": len(records),
        "planned_runs": planned,
        "groups": groups,
        "paired_topology_and_controls": paired,
        "adaptation_affinity": affinity,
        "capacity_compensation": compensation,
        "capacity_frontier": capacities,
        "experience_frontier": experience,
        "acute_interventions": lesions,
        "coverage": coverage,
        "analysis_source_sha256": digest_file(__file__),
        "run_sources": [
            {key: r[key] for key in ("path", "result_sha256", "source_identity")} for r in records
        ],
        "statistical_unit": "paired training seed, not episode; 95% seed bootstrap; exploratory and unadjusted",
        "equivalence": "An interval containing zero alone does not establish compensation/equivalence",
    }
    atomic_json(output / "summary.json", summary)
    decision = mvp_decision(records)
    decision.update(
        schema="connectome-mvp-decision-v1",
        run_sources=summary["run_sources"],
        source_identities=sorted({r["source_identity"] for r in records}),
    )
    atomic_json(output / "decision.json", decision)
    report = [
        "# Hover adaptation-frontier report",
        "",
        f"Completed runs: {len(records)}. Planned runs: {planned if planned is not None else 'not specified'}.",
        "",
        "Scientific runs and validation fixtures remain separate. Curves use held-out evaluations after all checkpoint selections were frozen.",
        "",
        "| Dataset | Condition | Budget | Online interactions | Test success | Training seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in groups:
        report.append(
            f"| {group['dataset']} | {group['variant']} | {group['budget']} | {group['online_interactions']} | {group['success']['mean']:.3f} | {group['success']['training_seeds']} |"
        )
    report += [
        "",
        f"MVP decision: {decision['reason']}.",
        "",
        "The shared teacher data and external teacher pretraining are additional prior information. Adaptation affinity on FlyBody does not identify a universal brain-body compatibility effect.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    if groups:
        _figures(
            groups,
            paired,
            compensation,
            affinity,
            lesions,
            output,
            validation=any(
                r["result"]["evidence"] != "native_hover_adaptation_experiment" for r in records
            ),
        )
    return summary


def _figures(groups, paired, compensation, affinity, lesions, output, validation=False):
    import matplotlib

    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    prefix = "Validation fixture: " if validation else ""

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", bbox_inches="tight", dpi=180)
        plt.close(fig)

    def effect_figure(entries, group_keys, measure, title, filename, band=False):
        if not entries:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in sorted({tuple(r[k] for k in group_keys) for r in entries}):
            values = sorted(
                [r for r in entries if tuple(r[k] for k in group_keys) == key],
                key=lambda r: r["budget"],
            )
            x = [r["budget"] for r in values]
            y = [r[measure]["mean"] for r in values]
            ax.plot(x, y, marker="o", label=" / ".join(key))
            ax.fill_between(
                x,
                [r[measure]["ci95"][0] for r in values],
                [r[measure]["ci95"][1] for r in values],
                alpha=0.15,
            )
        if band:
            ax.axhspan(-0.05, 0.05, color="gray", alpha=0.10, label="Prespecified ±0.05 band")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(
            xscale="log",
            xlabel="Trainable parameter ceiling",
            ylabel="Paired hover score difference",
            title=prefix + title,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
        save(fig, filename)

    for protocol in sorted({g["protocol"] for g in groups}):
        subset = [g for g in groups if g["protocol"] == protocol]
        last = max(g["online_interactions"] for g in subset)
        fig, ax = plt.subplots(figsize=(8, 5))
        for dataset, variant in sorted({(g["dataset"], g["variant"]) for g in subset}):
            values = sorted(
                [
                    g
                    for g in subset
                    if g["dataset"] == dataset
                    and g["variant"] == variant
                    and g["online_interactions"] == last
                ],
                key=lambda g: g["parameters"],
            )
            x, y = [v["parameters"] for v in values], [v["success"]["mean"] for v in values]
            ax.plot(x, y, marker="o", label=f"{dataset} / {variant}")
            ax.fill_between(
                x,
                [v["success"]["ci95"][0] for v in values],
                [v["success"]["ci95"][1] for v in values],
                alpha=0.15,
            )
        ax.set(
            xscale="log",
            ylim=(-0.03, 1.03),
            xlabel="Actual trainable adapter/controller parameters",
            ylabel="Held-out hover success",
            title=prefix + f"Hover adaptation frontier, N_online={last:,}",
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
        save(fig, f"capacity-{protocol}")
        for budget in sorted({g["budget"] for g in subset}):
            fig, ax = plt.subplots(figsize=(8, 5))
            for dataset, variant in sorted({(g["dataset"], g["variant"]) for g in subset}):
                values = sorted(
                    [
                        g
                        for g in subset
                        if g["dataset"] == dataset
                        and g["variant"] == variant
                        and g["budget"] == budget
                    ],
                    key=lambda g: g["online_interactions"],
                )
                ax.plot(
                    [v["online_interactions"] for v in values],
                    [v["success"]["mean"] for v in values],
                    marker="o",
                    label=f"{dataset} / {variant}",
                )
                ax.fill_between(
                    [v["online_interactions"] for v in values],
                    [v["success"]["ci95"][0] for v in values],
                    [v["success"]["ci95"][1] for v in values],
                    alpha=0.15,
                )
            ax.set(
                ylim=(-0.03, 1.03),
                xlabel="Additional DAgger environment interactions",
                ylabel="Held-out hover success",
                title=prefix + f"P ceiling={budget:,}; shared offline dataset held fixed",
            )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
            save(fig, f"experience-{protocol}-p{budget}")
        effect_figure(
            [r for r in paired if r["protocol"] == protocol],
            ("dataset", "control"),
            "score_effect",
            "Real topology versus matched controls",
            f"topology-advantage-{protocol}",
        )
        effect_figure(
            [r for r in compensation if r["protocol"] == protocol],
            ("first", "second"),
            "score_gap",
            "Substrate differences across adapter capacity",
            f"compensation-{protocol}",
            band=True,
        )
        effect_figure(
            [r for r in affinity if r["protocol"] == protocol],
            ("first", "second"),
            "difference_in_biological_advantage",
            "Difference in biological topology advantage",
            f"affinity-{protocol}",
        )
        for dataset in sorted({r["dataset"] for r in lesions if r["protocol"] == protocol}):
            effect_figure(
                [
                    r
                    for r in lesions
                    if r["protocol"] == protocol
                    and r["dataset"] == dataset
                    and r["variant"] in ("real", "adapter_only", "trainable_gru")
                ],
                ("variant", "intervention"),
                "paired_score_drop",
                f"Acute intervention: {dataset}",
                f"interventions-{protocol}-{dataset.replace('/', '-')}",
            )
