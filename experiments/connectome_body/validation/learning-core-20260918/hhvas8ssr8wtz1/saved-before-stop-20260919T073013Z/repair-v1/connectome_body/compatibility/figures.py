"""Standalone scientific figures from measured, paired paper-report records."""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..util import digest_json
from .analysis import paired_contrast


def _plot_curves(
    records, output, prefix, *, ylabel, xlabel="Actual allocated adapter parameters", smoke=False
):
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in records:
        groups[tuple(row["group"])].append(row)
    figures = []
    for group, rows in sorted(groups.items(), key=lambda x: str(x[0])):
        series = defaultdict(list)
        for row in rows:
            series[row["label"]].append(row)
        figure, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        axis.axhline(0, color="0.6", linewidth=0.8)
        for label, points in sorted(series.items()):
            points.sort(key=lambda item: item["x"])
            x, y, low, high = [], [], [], []
            for row in points:
                estimate = row["estimate"]
                if estimate["mean"] is None:
                    continue
                x.append(row["x"])
                y.append(estimate["mean"])
                lo, hi = estimate["ci95"] or [estimate["mean"], estimate["mean"]]
                low.append(lo)
                high.append(hi)
            axis.plot(x, y, marker="o", label=label)
            axis.fill_between(x, low, high, alpha=0.12)
        axis.set(xscale="log", xlabel=xlabel, ylabel=ylabel, title=" · ".join(map(str, group)))
        axis.legend(fontsize=7)
        stem = f"{prefix}-{digest_json(group)[:12]}"
        if smoke:
            figure.suptitle("SOFTWARE CHECK — NOT RESEARCH EVIDENCE", fontsize=9)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{stem}.{extension}", dpi=180)
        plt.close(figure)
        figures.append(
            {"kind": prefix, "group": list(group), "png": f"{stem}.png", "pdf": f"{stem}.pdf"}
        )
    return figures


def _paired_capacity_records(rows, *, cross_connectome=False):
    strata = defaultdict(list)
    for row in rows:
        if row["kind"] != "connectome" or row["family"] == "anatomical":
            continue
        group = tuple(
            row[k]
            for k in (
                "body",
                "task",
                "family",
                "plasticity",
                "channels",
                "experience_budget",
                "sign_mode",
                "initialization",
            )
        )
        strata[(group, row["adapter_parameters"])].append(row)
    result = []
    for (group, capacity), values in strata.items():
        if cross_connectome:
            comparisons = [
                ({"connectome": a}, {"connectome": b}, f"{a} minus {b}")
                for a, b in itertools.combinations(sorted({r["connectome"] for r in values}), 2)
            ]
        else:
            comparisons = [
                (
                    {"connectome": c, "topology": "real"},
                    {"connectome": c, "topology": "degree_rewired"},
                    c,
                )
                for c in sorted({r["connectome"] for r in values})
            ]
        for left, right, label in comparisons:
            estimate = paired_contrast(
                values,
                left,
                right,
                keys=("train_seed", "substrate_seed"),
                outcome="selected_test_score",
            )
            if estimate["mean"] is not None:
                result.append({"group": group, "x": capacity, "label": label, "estimate": estimate})
    return result


def _loss_figures(records, output, prefix, *, smoke=False):
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in records:
        factor, connectome, body, task, family, plasticity, capacity, topology = row["condition"]
        groups[(connectome, body, task, family, plasticity, capacity, topology)].append(
            (factor, row["score_loss"])
        )
    figures = []
    for group, points in sorted(groups.items(), key=lambda x: str(x[0])):
        points.sort(key=lambda x: x[0])
        fig, axis = plt.subplots(
            figsize=(7.5, max(3.5, len(points) * 0.34)), constrained_layout=True
        )
        for index, (label, estimate) in enumerate(points):
            mean = estimate["mean"]
            if mean is None:
                continue
            lo, hi = estimate["ci95"] or [mean, mean]
            axis.errorbar(
                mean,
                index,
                xerr=[[max(0, mean - lo)], [max(0, hi - mean)]],
                fmt="o",
                color="tab:blue",
                capsize=3,
            )
        axis.axvline(0, color="0.6", linewidth=0.8)
        axis.set(
            yticks=range(len(points)),
            yticklabels=[p[0] for p in points],
            xlabel="Paired loss in held-out score",
            title=" · ".join(map(str, group)),
        )
        stem = f"{prefix}-{digest_json(group)[:12]}"
        if smoke:
            fig.suptitle("SOFTWARE CHECK — NOT RESEARCH EVIDENCE", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=180)
        plt.close(fig)
        figures.append(
            {"kind": prefix, "group": list(group), "png": f"{stem}.png", "pdf": f"{stem}.pdf"}
        )
    return figures


def _residual_matrix(estimate, output, *, smoke=False):
    if estimate.get("status") != "estimated":
        return []
    import matplotlib.pyplot as plt

    records = estimate["residual_by_pair"]
    brains = sorted({r["connectome"] for r in records})
    bodies = sorted({r["body"] for r in records})
    values = np.full((len(brains), len(bodies)), np.nan)
    for row in records:
        values[brains.index(row["connectome"]), bodies.index(row["body"])] = row["residual"]
    limit = max(float(np.nanmax(np.abs(values))), 1e-6)
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = axis.imshow(
        np.ma.masked_invalid(values), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
    )
    axis.set(
        xticks=range(len(bodies)),
        xticklabels=bodies,
        yticks=range(len(brains)),
        yticklabels=brains,
        xlabel="Body",
        ylabel="Connectome",
        title="Residual learning-curve AUC",
    )
    for i, j in np.ndindex(values.shape):
        axis.text(
            j,
            i,
            "missing" if np.isnan(values[i, j]) else f"{values[i, j]:+.3f}",
            ha="center",
            va="center",
            fontsize=9,
        )
    figure.colorbar(image, ax=axis, label="Residual after main-effect adjustment")
    stem = "compatibility-residual-matrix"
    if smoke:
        figure.suptitle("SOFTWARE CHECK — NOT RESEARCH EVIDENCE", fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{stem}.{extension}", dpi=180)
    plt.close(figure)
    return [{"kind": "compatibility_residual_matrix", "png": stem + ".png", "pdf": stem + ".pdf"}]


def paper_figures(topology_rows, biological_rows, reports, output, *, smoke=False):
    import matplotlib

    matplotlib.use("Agg")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = _plot_curves(
        _paired_capacity_records(topology_rows),
        output,
        "topology-advantage",
        ylabel="Real minus degree-rewired held-out score",
        smoke=smoke,
    )
    figures += _plot_curves(
        _paired_capacity_records(biological_rows, cross_connectome=True),
        output,
        "substrate-gap",
        ylabel="Difference in held-out score",
        smoke=smoke,
    )
    controls = []
    for comparison in reports["3"]["comparisons"]:
        group = [
            comparison[k]
            for k in ("connectome", "body", "task", "family", "plasticity", "matching_dimension")
        ]
        controls.append(
            {
                "group": group,
                "x": comparison["adapter_parameters"],
                "label": comparison["control_kind"] + " / " + comparison["control_topology"],
                "estimate": comparison["outcomes"]["score_auc"],
            }
        )
    figures += _plot_curves(
        controls,
        output,
        "matched-controller-advantage",
        ylabel="Biological minus matched-control score AUC",
        smoke=smoke,
    )
    figures += _loss_figures(
        reports["8"]["paired_intervention_losses"], output, "causal-lesions", smoke=smoke
    )
    figures += _loss_figures(
        reports["9"]["paired_robustness_losses"], output, "robustness", smoke=smoke
    )
    figures += _residual_matrix(reports["6"]["score_auc"], output, smoke=smoke)
    return figures
