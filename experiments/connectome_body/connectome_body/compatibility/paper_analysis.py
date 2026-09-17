"""Experiment-specific estimands, paired controls, interaction tests and study evidence inventory."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..util import atomic_json, digest_json
from .analysis import (
    _figures,
    _mean_interval,
    capacity_frontiers,
    collect_runs,
    compatibility_regression,
    paired_contrast,
    pareto_interfaces,
)
from .runner import read_plan
from .training import source_identity

OUTCOMES = ("score_auc", "restricted_efficiency", "success", "final_score")
PAIR_KEYS = (
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


def interaction_test(rows, factors, *, outcome="score_auc", replicates=999, seed=0):
    """Highest-order interaction against all lower-order terms, with seed-cluster inference.

    Few seeds imply a coarse exact sign-flip distribution. This does not turn
    evaluation episodes or multiple graph nulls into independent experiments.
    """
    if len(factors) not in (2, 3):
        raise ValueError("Specify a two- or three-factor interaction")
    if not rows:
        return {"status": "unidentified", "reason": "No terminal observations"}
    categories = {factor: sorted({str(row[factor]) for row in rows}) for factor in factors}
    if any(len(levels) < 2 for levels in categories.values()):
        return {
            "status": "unidentified",
            "reason": "Every interaction factor needs at least two levels",
        }
    observed = {tuple(str(row[factor]) for factor in factors) for row in rows}
    missing = sorted(set(itertools.product(*categories.values())) - observed)
    if missing:
        return {
            "status": "unidentified",
            "reason": "Incomplete factorial cells",
            "missing_cells": [list(x) for x in missing],
        }
    y = np.array([row[outcome] for row in rows], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Interaction outcomes must be observed and finite")
    columns = [np.ones(len(rows))]
    # Block main effects not already included as interaction factors.
    for field in (
        "connectome",
        "body_task",
        "family",
        "plasticity",
        "initialization",
        "sign_mode",
        "topology",
    ):
        if field in factors or any(field not in row for row in rows):
            continue
        for level in sorted({str(row[field]) for row in rows})[1:]:
            columns.append(np.array([str(row[field]) == level for row in rows], float))
    for field in ("adapter_parameters", "actor_parameters", "experience_budget", "channels"):
        values = np.log1p([row[field] for row in rows])
        if np.std(values) > 1e-12:
            columns.append((values - values.mean()) / values.std())
    indicators = {
        factor: [
            np.array([str(row[factor]) == level for row in rows], float) for level in levels[1:]
        ]
        for factor, levels in categories.items()
    }
    highest = []
    for order in range(1, len(factors) + 1):
        destination = highest if order == len(factors) else columns
        for subset in itertools.combinations(factors, order):
            for parts in itertools.product(*(indicators[field] for field in subset)):
                destination.append(np.prod(parts, axis=0))
    counts = defaultdict(int)
    strata = [
        (
            *tuple(str(row[factor]) for factor in factors),
            row["train_seed"],
            row["declared_budget"],
            row["channels"],
        )
        for row in rows
    ]
    for key in strata:
        counts[key] += 1
    weights = np.sqrt(np.array([1 / counts[key] for key in strata], float))

    def basis(items):
        design = np.column_stack(items) * weights[:, None]
        u, singular, _ = np.linalg.svd(design, full_matrices=False)
        rank = int(np.sum(singular > max(design.shape) * np.finfo(float).eps * singular[0]))
        return u[:, :rank]

    reduced, full = basis(columns), basis([*columns, *highest])
    if full.shape[1] <= reduced.shape[1] or full.shape[1] >= len(rows):
        return {
            "status": "unidentified",
            "reason": "Interaction has no estimable additional rank or no residual replication",
        }
    response = weights * y
    fitted = reduced @ (reduced.T @ response)
    residual = response - fitted
    rss0 = float(residual @ residual)
    rss1 = max(0.0, float(response @ response - np.square(full.T @ response).sum()))
    observed_stat = max(0.0, rss0 - rss1)
    seeds = sorted({row["train_seed"] for row in rows})
    row_cluster = np.array([seeds.index(row["train_seed"]) for row in rows])
    pvalue, draws, exact = None, 0, False
    if len(seeds) > 1:
        if len(seeds) <= 10:
            signs = np.array(list(itertools.product((-1, 1), repeat=len(seeds))), dtype=float)
            exact = True
        else:
            signs = np.random.default_rng(seed).choice([-1.0, 1.0], (replicates, len(seeds)))
        statistics = []
        for block in np.array_split(signs, max(1, int(np.ceil(len(signs) / 64)))):
            simulated = fitted[:, None] + residual[:, None] * block[:, row_cluster].T
            statistics.extend(
                np.maximum(
                    0,
                    np.square(full.T @ simulated).sum(0) - np.square(reduced.T @ simulated).sum(0),
                )
            )
        draws = len(statistics)
        exceed = np.count_nonzero(
            np.array(statistics) >= observed_stat - max(1e-12, observed_stat * 1e-10)
        )
        pvalue = float(exceed / draws) if exact else float((exceed + 1) / (draws + 1))
    return {
        "status": "estimated",
        "factors": list(factors),
        "outcome": outcome,
        "reduced_rank": reduced.shape[1],
        "full_rank": full.shape[1],
        "rows": len(rows),
        "weighted_rss_reduction": observed_stat,
        "partial_eta_squared": observed_stat / rss0 if rss0 > 1e-15 else None,
        "seed_cluster_sign_flip_p": pvalue,
        "training_seeds": len(seeds),
        "sign_flip_draws": draws,
        "exact_sign_flip_enumeration": exact,
        "inference_note": "Rademacher wild-cluster test under lower-order null; few training seeds give coarse inference; no episode-level pseudoreplication",
    }


def _planned_matches(rows, conditions, experiment="3"):
    by_id = {row["condition_id"]: row for row in rows}
    grouped = defaultdict(list)
    missing = []
    for condition in conditions:
        if experiment not in condition["experiments"]:
            continue
        for match in condition["matches"]:
            a, b = by_id.get(match["reference"]), by_id.get(condition["id"])
            if a is None or b is None:
                missing.append([match["reference"], condition["id"]])
                continue
            key = (
                b["kind"],
                b["topology"],
                match["dimension"],
                a["connectome"],
                a["body"],
                a["task"],
                a["family"],
                a["plasticity"],
                a["declared_budget"],
            )
            grouped[key].append((a, b))
    records = []
    for key, pairs in sorted(grouped.items()):
        records.append(
            {
                "control_kind": key[0],
                "control_topology": key[1],
                "matching_dimension": key[2],
                "connectome": key[3],
                "body": key[4],
                "task": key[5],
                "family": key[6],
                "plasticity": key[7],
                "declared_budget": key[8],
                "adapter_parameters": float(np.mean([a["adapter_parameters"] for a, _ in pairs])),
                "outcomes": {
                    metric: _mean_interval(
                        [a[metric] - b[metric] for a, b in pairs],
                        [a["train_seed"] for a, b in pairs],
                    )
                    for metric in OUTCOMES
                },
                "actor_parameter_differences": [
                    a["actor_parameters"] - b["actor_parameters"] for a, b in pairs
                ],
                "state_dimension_differences": [
                    a["state_dimension"] - b["state_dimension"] for a, b in pairs
                ],
            }
        )
    return {"comparisons": records, "missing_pairs": missing}


def _post_training(plan, root, rows):
    by_id = {row["condition_id"]: row for row in rows}
    causal, robustness, transfers, predictions, pending = [], [], [], [], []
    conditions = {row["id"]: row for row in plan["conditions"]}
    for job in plan["jobs"]:
        directory = root / "jobs" / job["id"]
        path = directory / "result.json"
        if not path.exists():
            pending.append(job["id"])
            continue
        record = json.loads(path.read_text())
        expected = digest_json({"plan_fingerprint": plan["fingerprint"], "job": job})
        if (
            record.get("request_identity") != expected
            or record.get("status") != "complete"
            or record.get("fingerprint")
            != digest_json({key: value for key, value in record.items() if key != "fingerprint"})
        ):
            raise ValueError("Post-training evidence belongs to a different planned job")
        result = record["result"]
        if job["kind"] == "predict_pairs":
            predictions.append({"job": job["id"], **result})
            continue
        if job["reference"] not in by_id:
            pending.append(job["id"])
            continue
        source = by_id[job["reference"]]
        group = {
            key: source[key]
            for key in (
                "connectome",
                "body",
                "task",
                "family",
                "plasticity",
                "adapter_parameters",
                "train_seed",
                "topology",
                "kind",
            )
        }
        group.update(job=job["id"], reference=job["reference"])
        if job["kind"] in ("intervention", "robustness"):
            measured = result["evaluation"]
            group.update(
                mean_score=measured["mean_score"],
                success_rate=measured["success_rate"],
                score_loss=source["selected_test_score"] - measured["mean_score"],
                success_loss=source["selected_test_success"] - measured["success_rate"],
            )
            if job["kind"] == "intervention":
                group["intervention"] = job["parameters"].get("label", job["parameters"]["kind"])
                causal.append(group)
            else:
                group["perturbation"] = job["parameters"]["perturbation"]["name"]
                robustness.append(group)
        elif job["kind"] == "task_transfer":
            transfers.append(
                {
                    **group,
                    "mode": "zero_shot",
                    "target_task": job["parameters"]["body"]["task"],
                    "evaluation": result["evaluation"],
                    "source_training_interactions": result["source_training_interactions"],
                }
            )
        elif job["kind"] == "finetune":
            target_rows = collect_runs(
                directory / "training", include_smoke=plan["study"]["purpose"] == "smoke"
            )["rows"]
            if len(target_rows) != 1:
                raise ValueError("Completed fine-tuning job lacks one terminal target run")
            tuned = target_rows[0]
            scratch = []
            for condition in conditions.values():
                if (
                    condition["id"] not in by_id
                    or condition["config"]["body"] != job["parameters"]["body"]
                ):
                    continue
                if any(
                    match["reference"] == job["reference"]
                    and match["dimension"] == "target_task_experience"
                    for match in condition["matches"]
                ):
                    scratch.append(by_id[condition["id"]])
            transfers.append(
                {
                    **group,
                    "mode": "fine_tuned",
                    "target_task": tuned["task"],
                    "outcomes": {metric: tuned[metric] for metric in OUTCOMES},
                    "additional_experience": tuned["experience"],
                    "source_training_interactions": result["source_training_interactions"],
                    "restricted_total_experience": result["source_training_interactions"]
                    + tuned["restricted_experience"],
                    "scratch_controls": [
                        {
                            "condition": row["condition_id"],
                            "outcomes": {metric: row[metric] for metric in OUTCOMES},
                            "paired_difference": {
                                metric: tuned[metric] - row[metric] for metric in OUTCOMES
                            },
                        }
                        for row in scratch
                    ],
                }
            )
    return {
        "causal": causal,
        "robustness": robustness,
        "transfer": transfers,
        "predictions": predictions,
        "pending_jobs": pending,
    }


def summarize_evaluations(rows, factor):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(
            row[key]
            for key in (
                factor,
                "connectome",
                "body",
                "task",
                "family",
                "plasticity",
                "adapter_parameters",
                "topology",
            )
        )
        grouped[key].append(row)
    return [
        {
            "condition": list(key),
            **{
                metric: _mean_interval(
                    [row[metric] for row in subset], [row["train_seed"] for row in subset]
                )
                for metric in ("score_loss", "success_loss", "mean_score", "success_rate")
            },
        }
        for key, subset in sorted(grouped.items())
    ]


def topology_adjusted_compatibility(rows, variant="degree_rewired"):
    """Native-pair interaction after subtracting the same seed's topology null."""
    controls = defaultdict(list)
    for row in rows:
        if row["kind"] == "connectome" and row["topology"] == variant:
            controls[tuple(row[key] for key in PAIR_KEYS)].append(row)
    paired, missing = [], 0
    for row in rows:
        if row["kind"] != "connectome" or row["topology"] != "real":
            continue
        nulls = controls.get(tuple(row[key] for key in PAIR_KEYS))
        if not nulls:
            missing += 1
            continue
        value = dict(row)
        for outcome in OUTCOMES:
            value[f"topology_advantage_{outcome}"] = row[outcome] - float(
                np.mean([r[outcome] for r in nulls])
            )
        paired.append(value)
    return {
        "null": variant,
        "paired_rows": len(paired),
        "unmatched_real_rows": missing,
        "estimand": "Native-pair interaction in real-minus-matched-null performance; seed and task paired before regression",
        "outcomes": {
            metric: compatibility_regression(paired, outcome=f"topology_advantage_{metric}")
            for metric in OUTCOMES
        },
    }


def analyze_study(plan_path, output, *, figures=True):
    plan, root = read_plan(plan_path, verify_code=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collected = collect_runs(root / "runs", include_smoke=plan["study"]["purpose"] == "smoke")
    conditions = {item["id"]: item for item in plan["conditions"]}
    rows = []
    for row in collected["rows"]:
        identity = Path(row["run"]).name
        if identity not in conditions:
            raise ValueError("Unplanned result present in the study run directory")
        rows.append({**row, "condition_id": identity, "body_task": f"{row['body']}:{row['task']}"})

    def experiment(number):
        return [
            row for row in rows if str(number) in conditions[row["condition_id"]]["experiments"]
        ]

    biological = [
        row
        for row in experiment(6)
        if row["kind"] == "connectome"
        and row["topology"] == "real"
        and row["family"] != "anatomical"
    ]
    reports = {
        "1": {
            "generic_pareto_frontiers": pareto_interfaces(
                [row for row in experiment(1) if row["family"] != "anatomical"]
            ),
            "anatomical_oracle_frontiers": pareto_interfaces(
                [row for row in experiment(1) if row["family"] == "anatomical"]
            ),
        },
        "2": {
            "minimum_interface_frontiers": capacity_frontiers(
                experiment(1)
                + [
                    row
                    for row in experiment(2)
                    if row["condition_id"] not in {r["condition_id"] for r in experiment(1)}
                ]
            ),
            "architecture_connectome_body_interaction": interaction_test(
                biological, ("family", "connectome", "body_task")
            ),
        },
        "3": _planned_matches(rows, plan["conditions"]),
        "4": {
            "paired_topology_advantage": [
                paired_contrast(
                    experiment(4),
                    {"topology": "real"},
                    {"topology": topology},
                    keys=PAIR_KEYS,
                    outcome=outcome,
                )
                for topology in (
                    "community_rewired",
                    "degree_rewired",
                    "direction_shuffled",
                    "random",
                )
                for outcome in OUTCOMES
            ]
        },
        "5": {
            "initialization_plasticity_interaction": interaction_test(
                experiment(5), ("initialization", "plasticity")
            ),
            "sign_annotation_coverage": {
                key: value.get("summary", {}).get("annotated_sign_fraction")
                for key, value in plan["inputs"]["graphs"].items()
                if key in plan["study"]["graphs"]
            },
        },
        "6": {metric: compatibility_regression(biological, outcome=metric) for metric in OUTCOMES},
        "7": {
            "connectome_task_interactions": {
                body: interaction_test(
                    [row for row in experiment(6) if row["body"] == body], ("connectome", "task")
                )
                for body in plan["study"]["bodies"]
                if ":" not in body
            },
            "subgraph_comparisons": _planned_matches(rows, plan["conditions"], "7"),
        },
    }
    native = {row["connectome"]: row["native_body"] for row in biological}
    capacities = capacity_frontiers(biological)
    for row in capacities:
        restricted = row["minimum_adapter_parameters"] or row["largest_tested_adapter"]
        row.update(
            native_body=native[row["connectome"]],
            declared_budget=row["largest_tested_adapter"],
            restricted_capacity=restricted,
            restricted_capacity_efficiency=1 - restricted / row["largest_tested_adapter"],
        )
    reports["6"]["capacity_frontiers"] = capacities
    reports["6"]["restricted_capacity_interaction"] = compatibility_regression(
        capacities,
        outcome="restricted_capacity_efficiency",
        covariates=("experience_budget", "channels"),
    )
    reports["6"]["capacity_interpretation"] = (
        "Minimum confirmed tested capacity; failures right-censored at the largest tested value. The bounded efficiency model is not an uncensored estimate of P_tau."
    )
    reports["6"]["topology_adjusted_compatibility"] = {
        variant: topology_adjusted_compatibility(experiment(4), variant)
        for variant in ("degree_rewired", "community_rewired", "random")
    }
    # Task keys in the catalog include body:task; derive physical bodies once.
    reports["7"]["connectome_task_interactions"] = {
        body: interaction_test(
            [row for row in biological if row["body"] == body], ("connectome", "task")
        )
        for body in sorted({item["spec"]["name"] for item in plan["study"]["bodies"].values()})
    }
    post = _post_training(plan, root, rows)
    reports["8"] = {
        "paired_intervention_losses": summarize_evaluations(post["causal"], "intervention"),
        "observations": post["causal"],
    }
    reports["9"] = {
        "paired_robustness_losses": summarize_evaluations(post["robustness"], "perturbation"),
        "transfer": post["transfer"],
    }
    reports["10"] = {"held_out_pair_predictions": post["predictions"]}
    report = {
        "schema": "ten-experiment-paper-report-v1",
        "plan_fingerprint": plan["fingerprint"],
        "analysis_code_fingerprint": source_identity(),
        "purpose": plan["study"]["purpose"],
        "terminal_training_runs": len(rows),
        "planned_training_conditions": len(conditions),
        "coverage": plan["coverage"],
        "pending_runs": collected["pending"],
        "pending_jobs": post["pending_jobs"],
        "experiments": reports,
        "figures": _figures(experiment(3), output) if figures and experiment(3) else [],
        "interpretation": "Planned, missing, infeasible and measured evidence are separate. Ideal outcomes in the paper are hypotheses, not asserted results.",
    }
    if figures and rows:
        from .figures import paper_figures

        report["figures"].extend(
            paper_figures(
                experiment(4),
                biological,
                reports,
                output,
                smoke=plan["study"]["purpose"] == "smoke",
            )
        )
    atomic_json(output / "runs.json", rows)
    atomic_json(output / "paper-report.json", report)
    lines = [
        "# Ten-experiment compatibility study",
        "",
        f"Purpose: {plan['study']['purpose']}. {len(rows)} terminal training runs out of {len(conditions)} planned unique conditions.",
        "",
        "| Experiment | Training conditions | Runnable inputs | Post-training jobs |",
        "|---|---:|---:|---:|",
    ]
    for number, coverage in plan["coverage"].items():
        lines.append(
            f"| {number}: {coverage['title']} | {coverage['unique_training_conditions']} | {coverage['training_statuses'].get('ready', 0)} | {coverage['post_training_jobs']} |"
        )
    lines += [
        "",
        "Censored thresholds are not successful runs. Confidence intervals resample training seeds. A compiled condition is not a measurement.",
        "",
        "Machine-readable estimands, paired comparisons, missing cells, and causal/transfer results are in `paper-report.json`.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
