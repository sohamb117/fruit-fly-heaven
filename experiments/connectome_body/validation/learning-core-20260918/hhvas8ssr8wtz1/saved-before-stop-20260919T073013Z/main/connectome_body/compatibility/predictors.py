"""Scalable structural measurements and nested prediction on unseen body–connectome pairs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path

from ..util import atomic_json, digest_file, digest_json, seed_for


def _annotations(graph, path):
    if path is None:
        return {}, None
    data = json.loads(Path(path).read_text())
    if data.get("graph_fingerprint") != graph.fingerprint or not data.get("provenance"):
        raise ValueError("Structural anatomy annotations require graph identity and provenance")
    return data, digest_file(path)


def graph_features(
    graph, *, community_file=None, anatomy_file=None, landmarks=32, motif_samples=5000, seed=0
):
    if (
        type(landmarks) is not int
        or landmarks < 1
        or type(motif_samples) is not int
        or motif_samples < 1
    ):
        raise ValueError("Positive landmark and motif sample counts are required")
    n, m = graph.n, graph.m
    adjacency = sparse.csr_matrix((np.ones(m, dtype=bool), (graph.src, graph.dst)), shape=(n, n))
    outgoing = np.bincount(graph.src, minlength=n)
    incoming = np.bincount(graph.dst, minlength=n)
    weighted_out = np.bincount(graph.src, weights=graph.weight, minlength=n)
    weighted_in = np.bincount(graph.dst, weights=graph.weight, minlength=n)
    values = {
        "neurons": n,
        "edges": m,
        "sparsity": m / (n * max(1, n - 1)),
        "isolated_fraction": float(np.mean((outgoing + incoming) == 0)),
        "reciprocal_edge_fraction": float(adjacency.multiply(adjacency.T).nnz / max(1, m)),
        "annotated_sign_fraction": float(np.mean(graph.signs != 0)),
    }
    for label, degrees in (
        ("in_degree", incoming),
        ("out_degree", outgoing),
        ("in_strength", weighted_in),
        ("out_strength", weighted_out),
    ):
        values[f"{label}_mean"] = float(degrees.mean())
        values[f"{label}_coefficient_of_variation"] = float(
            degrees.std() / max(degrees.mean(), 1e-15)
        )
        for q in (0.5, 0.9, 0.99):
            values[f"{label}_q{int(q * 100)}"] = float(np.quantile(degrees, q))
    components, labels = connected_components(adjacency, directed=True, connection="strong")
    sizes = np.bincount(labels)
    values.update(
        strong_components=components,
        largest_strong_component_fraction=float(sizes.max() / n),
        flow_hierarchy=float(np.mean(labels[graph.src] != labels[graph.dst])),
    )
    rng = np.random.default_rng(seed_for(seed, "structural-predictors"))
    sources = rng.choice(n, min(n, landmarks), replace=False)
    distances = shortest_path(adjacency, directed=True, indices=sources, unweighted=True)
    off_diagonal = np.ones(distances.shape, dtype=bool)
    off_diagonal[np.arange(len(sources)), sources] = False
    reachable = np.isfinite(distances) & off_diagonal
    finite = distances[reachable]
    values.update(
        reachable_pair_fraction=float(reachable.sum() / max(1, off_diagonal.sum())),
        reachable_path_mean=float(finite.mean()) if finite.size else None,
        path_diameter_lower_bound=float(finite.max()) if finite.size else None,
    )
    # Connected-chain sampling avoids an almost-all-empty uniform-triad sample
    # in a 100k-neuron sparse graph. These are explicitly conditional motifs,
    # not estimates of an unrestricted triad census.
    valid_edges = np.flatnonzero(outgoing[graph.dst] > 0)
    draws = (
        rng.choice(valid_edges, motif_samples, replace=True)
        if len(valid_edges)
        else np.empty(0, int)
    )
    first, middle = graph.src[draws], graph.dst[draws]
    last = np.array(
        [
            adjacency.indices[adjacency.indptr[node] + rng.integers(outgoing[node])]
            for node in middle
        ],
        dtype=np.int64,
    )
    keep = first != last
    first, middle, last = first[keep], middle[keep], last[keep]
    bits = np.zeros(len(first), dtype=np.int64)
    for bit, (src, dst) in enumerate(
        ((middle, first), (last, middle), (first, last), (last, first))
    ):
        bits |= np.asarray(adjacency[src, dst]).ravel().astype(np.int64) << bit
    frequencies = np.bincount(bits, minlength=16) / max(1, len(bits))
    values.update(
        {f"chain_motif_{i:04b}": float(v) if len(bits) else None for i, v in enumerate(frequencies)}
    )
    values["chain_feedforward_closure"] = float(np.mean((bits & 4) != 0)) if len(bits) else None
    values["chain_cyclic_closure"] = float(np.mean((bits & 8) != 0)) if len(bits) else None
    values["directed_modularity"] = values["communities"] = None
    community_digest = None
    if community_file:
        record = json.loads(Path(community_file).read_text())
        if record.get("graph_fingerprint") != graph.fingerprint:
            raise ValueError("Community partition belongs to another graph")
        community = np.asarray(record["labels"])
        if community.shape != (n,) or community.dtype.kind not in "iu":
            raise ValueError("One integer community label per neuron is required")
        _, community = np.unique(community, return_inverse=True)
        total = float(graph.weight.sum(dtype=np.float64))
        internal = float(
            graph.weight[community[graph.src] == community[graph.dst]].sum(dtype=np.float64)
        )
        kout = np.bincount(community, weights=weighted_out)
        kin = np.bincount(community, weights=weighted_in)
        values["directed_modularity"] = internal / total - float(kout @ kin) / total**2
        values["communities"] = len(kout)
        community_digest = digest_file(community_file)
    anatomy, anatomy_digest = _annotations(graph, anatomy_file)
    lookup = {str(node): i for i, node in enumerate(graph.node_ids)}

    def ids(key):
        names = [str(node) for node in anatomy.get(key, [])]
        if len(set(names)) != len(names) or any(node not in lookup for node in names):
            raise ValueError(f"Malformed or unknown neuronal identities in {key}")
        return np.array([lookup[str(node)] for node in names], dtype=np.int64)

    sensory, motor = ids("sensory_ids"), ids("motor_ids")
    values.update(
        sensory_fraction=len(sensory) / n if len(sensory) else None,
        motor_fraction=len(motor) / n if len(motor) else None,
        sensory_motor_path_mean=None,
        sensory_motor_reachable_fraction=None,
    )
    if len(sensory) and len(motor):
        sampled = rng.choice(sensory, min(len(sensory), landmarks), replace=False)
        paths = shortest_path(adjacency, directed=True, indices=sampled, unweighted=True)[:, motor]
        valid = np.isfinite(paths)
        values["sensory_motor_reachable_fraction"] = float(valid.mean())
        values["sensory_motor_path_mean"] = float(paths[valid].mean()) if valid.any() else None
    pairs = anatomy.get("bilateral_pairs", [])
    mirror = np.full(n, -1, dtype=np.int64)
    for pair in pairs:
        if len(pair) != 2 or any(str(node) not in lookup for node in pair):
            raise ValueError("Invalid anatomical left/right pairing")
        a, b = (lookup[str(node)] for node in pair)
        if a == b or mirror[a] >= 0 or mirror[b] >= 0:
            raise ValueError("Bilateral pairing must be a disjoint involution")
        mirror[a], mirror[b] = b, a
    covered = (mirror[graph.src] >= 0) & (mirror[graph.dst] >= 0)
    values["bilateral_mapped_fraction"] = float(np.mean(mirror >= 0)) if pairs else None
    values["bilateral_edge_agreement"] = (
        float(np.asarray(adjacency[mirror[graph.src[covered]], mirror[graph.dst[covered]]]).mean())
        if covered.any()
        else None
    )
    record = {
        "schema": "connectome-structural-features-v1",
        "graph_fingerprint": graph.fingerprint,
        "features": values,
        "seed": seed,
        "landmarks": sources.tolist(),
        "motif_sampling": "uniform first edge then uniform successor, conditional on three distinct neurons",
        "motif_samples_requested": motif_samples,
        "motif_samples_retained": len(bits),
        "community_sha256": community_digest,
        "anatomy_sha256": anatomy_digest,
        "missing_features_are_unknown_not_zero": True,
    }
    record["fingerprint"] = digest_json(record)
    return record


def body_features(body):
    if hasattr(body, "structural_descriptors"):
        values = body.structural_descriptors()
    else:
        values = _mujoco_body_descriptors(body)
    config = getattr(body, "study_spec", getattr(body, "spec", None))
    if config is None:
        raise ValueError("Body descriptors require a declared study task")
    values.setdefault("task_target_speed_native_units", config.parameters.get("target_speed"))
    values.setdefault("task_trajectory_frequency_hz", config.parameters.get("trajectory_frequency"))
    result = {
        "schema": "body-structural-features-v1",
        "body_fingerprint": body.fingerprint,
        "body": config.name,
        "task": config.task,
        "features": values,
        "evidence": body.evidence,
        "units_note": "Fluid coefficients and speed remain native-unit descriptors; body fixed effects are included",
    }
    result["fingerprint"] = digest_json(result)
    return result


def _mujoco_body_descriptors(body):
    model = body.physics.model.ptr if hasattr(body, "physics") else body.model
    moving = model.body_mass > 0
    hinge = np.flatnonzero(model.jnt_type == body.mujoco.mjtJoint.mjJNT_HINGE)
    damping = np.asarray(model.dof_damping)
    reference_inertia = np.asarray(model.dof_M0)
    damped = (damping > 0) & (reference_inertia > 0)
    relaxation = reference_inertia[damped] / damping[damped]
    values = {
        "observation_dimension": body.obs_dim,
        "action_dimension": body.action_dim,
        "control_dt_seconds": body.control_dt,
        "articulated_hinges": len(hinge),
        "moving_bodies": int(moving.sum()),
        "generalized_velocities": model.nv,
        "actuator_state_dimension": model.na,
        "median_passive_damping_time_seconds": float(np.median(relaxation))
        if len(relaxation)
        else None,
        "fluid_density_native_units": float(model.opt.density),
        "fluid_viscosity_native_units": float(model.opt.viscosity),
    }
    return values


def _feature_matrix(rows, graph_records, body_records, *, structural):
    names, columns = [], []
    # Main-effect baseline explicitly controls body/task difficulty, connectome
    # identity/size, experience, and both learned interface and total capacity.
    for field in ("connectome", "body", "task", "family", "plasticity"):
        for level in sorted({str(r[field]) for r in rows})[1:]:
            names.append(f"{field}={level}")
            columns.append(np.array([str(row[field]) == level for row in rows], dtype=float))
    for field in (
        "adapter_parameters",
        "actor_parameters",
        "experience_budget",
        "channels",
        "state_dimension",
        "observation_dim",
        "action_dim",
    ):
        names.append(f"log1p({field})")
        columns.append(np.log1p([row[field] for row in rows]))
    if structural:
        gnames = sorted(set.union(*(set(record["features"]) for record in graph_records.values())))
        bnames = sorted(set.union(*(set(record["features"]) for record in body_records.values())))
        g = np.array(
            [
                [graph_records[row["connectome"]]["features"].get(name) for name in gnames]
                for row in rows
            ],
            dtype=float,
        )
        b = np.array(
            [
                [
                    body_records[f"{row['body']}:{row['task']}"]["features"].get(name)
                    for name in bnames
                ]
                for row in rows
            ],
            dtype=float,
        )
        # Signed log compression is fixed before outcome inspection. Missing
        # values are imputed within each training fold below, never globally.
        g, b = np.sign(g) * np.log1p(np.abs(g)), np.sign(b) * np.log1p(np.abs(b))
        for label, matrix, fields in (("graph", g, gnames), ("body", b, bnames)):
            for i, name in enumerate(fields):
                names.append(f"{label}:{name}")
                columns.append(matrix[:, i])
        for gi, name in enumerate(gnames):
            for bi, bname in enumerate(bnames):
                names.append(f"graph:{name}*body:{bname}")
                columns.append(g[:, gi] * b[:, bi])
    return np.column_stack(columns), names


def _ridge_fit(x, y, pairs, alpha):
    if not len(x) or not np.isfinite(y).all():
        raise ValueError("Predictor training requires observed finite responses")
    medians = np.array(
        [np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in x.T]
    )
    missing = ~np.isfinite(x)
    filled = np.where(missing, medians, x)
    filled = np.column_stack((filled, missing.astype(float)))
    counts = defaultdict(int)
    for pair in pairs:
        counts[pair] += 1
    weights = np.array([1 / counts[pair] for pair in pairs])
    weights /= weights.sum()
    mean = np.sum(filled * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((filled - mean) ** 2 * weights[:, None], axis=0))
    scale[scale < 1e-12] = 1.0
    center = float(weights @ y)
    design = (filled - mean) / scale * np.sqrt(weights[:, None])
    response = (y - center) * np.sqrt(weights)
    # Solve in the smaller space: graph x body feature interactions can exceed
    # the number of observed pairs. No optional ML framework is required.
    if design.shape[1] > design.shape[0]:
        beta = design.T @ np.linalg.solve(design @ design.T + alpha * np.eye(len(design)), response)
    else:
        beta = np.linalg.solve(
            design.T @ design + alpha * np.eye(design.shape[1]), design.T @ response
        )
    return {"medians": medians, "mean": mean, "scale": scale, "beta": beta, "center": center}


def _ridge_predict(model, x):
    missing = ~np.isfinite(x)
    filled = np.column_stack((np.where(missing, model["medians"], x), missing.astype(float)))
    return (filled - model["mean"]) / model["scale"] @ model["beta"] + model["center"]


def _pair_mse(actual, predicted, pairs):
    groups = defaultdict(list)
    for target, estimate, pair in zip(actual, predicted, pairs, strict=True):
        groups[pair].append((target - estimate) ** 2)
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _validate_records(rows, graph_records, body_records):
    for records, schema in (
        (graph_records, "connectome-structural-features-v1"),
        (body_records, "body-structural-features-v1"),
    ):
        for record in records.values():
            if record.get("schema") != schema or record.get("fingerprint") != digest_json(
                {key: value for key, value in record.items() if key != "fingerprint"}
            ):
                raise ValueError("Structural feature schema or fingerprint mismatch")
    for row in rows:
        graph = graph_records.get(row["connectome"])
        body = body_records.get(f"{row['body']}:{row['task']}")
        if graph is None or body is None:
            raise ValueError("Missing structural features for a declared pair")
        for key, record in (("graph_fingerprint", graph), ("body_fingerprint", body)):
            if row.get(key) is not None and row[key] != record.get(key):
                raise ValueError(f"Predictor features do not describe the trained {key}")
        if row.get("perturbation") or row.get("initial_checkpoint"):
            raise ValueError(
                "Primary compatibility prediction excludes transfer and perturbation runs"
            )


def predict_held_out_pairs(
    rows,
    graph_records,
    body_records,
    *,
    outcome="success",
    alphas=(0.001, 0.01, 0.1, 1.0, 10.0),
    seed=0,
):
    if outcome not in ("success", "final_score", "score_auc", "restricted_efficiency"):
        raise ValueError(
            "Use a bounded outcome; uncensored threshold times cannot replace failed runs"
        )
    if not alphas or any(not np.isfinite(value) or value <= 0 for value in alphas):
        raise ValueError("Ridge regularization candidates must be positive")
    if not rows or any(row["kind"] != "connectome" or row["topology"] != "real" for row in rows):
        raise ValueError("This compatibility predictor is defined on real biological pairings")
    pairs = [(row["connectome"], row["body"]) for row in rows]
    unique = sorted(set(pairs))
    if len(unique) < 4:
        raise ValueError(
            "At least four body-connectome pairs are needed for nested held-out-pair prediction"
        )
    _validate_records(rows, graph_records, body_records)
    y = np.array([row[outcome] for row in rows], dtype=float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Predictor outcomes must be bounded in [0,1]")
    # Both models receive the same outer AND inner folds. Generate them before
    # either fit, so model order or random draws cannot change the comparison.
    rng = np.random.default_rng(seed_for(seed, "pair-folds"))
    inner_folds = {}
    for held_pair in unique:
        training_pairs = [pair for pair in unique if pair != held_pair]
        shuffled = [training_pairs[i] for i in rng.permutation(len(training_pairs))]
        count = min(3, len(shuffled))
        inner_folds[held_pair] = [set(shuffled[k::count]) for k in range(count)]
    models, all_predictions = {}, {}
    for label, structural in (("main_effect_baseline", False), ("structural_interactions", True)):
        x, names = _feature_matrix(rows, graph_records, body_records, structural=structural)
        predictions = np.full(len(rows), np.nan)
        folds = []
        for held_pair in unique:
            train = np.array([pair != held_pair for pair in pairs])
            test = ~train
            training_pairs = [pair for pair in unique if pair != held_pair]
            inner = inner_folds[held_pair]
            scores = []
            for alpha in alphas:
                errors = []
                for validation in inner:
                    valid = train & np.array([pair in validation for pair in pairs])
                    fit = train & ~valid
                    fitted = _ridge_fit(
                        x[fit],
                        y[fit],
                        [pair for pair, ok in zip(pairs, fit, strict=True) if ok],
                        alpha,
                    )
                    prediction = np.clip(_ridge_predict(fitted, x[valid]), 0, 1)
                    errors.append(
                        _pair_mse(
                            y[valid],
                            prediction,
                            [pair for pair, ok in zip(pairs, valid, strict=True) if ok],
                        )
                    )
                scores.append(float(np.mean(errors)))
            alpha = float(alphas[int(np.argmin(scores))])
            fitted = _ridge_fit(
                x[train],
                y[train],
                [pair for pair, ok in zip(pairs, train, strict=True) if ok],
                alpha,
            )
            predictions[test] = np.clip(_ridge_predict(fitted, x[test]), 0, 1)
            folds.append(
                {
                    "test_pair": list(held_pair),
                    "train_pairs": [list(p) for p in training_pairs],
                    "inner_validation_pairs": [[list(p) for p in sorted(group)] for group in inner],
                    "selected_alpha": alpha,
                    "inner_mse": scores,
                    "training_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "training_imputation_medians": fitted["medians"].tolist(),
                }
            )
        models[label] = {
            "pair_balanced_mse": _pair_mse(y, predictions, pairs),
            "features": names,
            "folds": folds,
        }
        all_predictions[label] = predictions
    records = [
        {
            "pair": list(pair),
            "train_seed": row["train_seed"],
            "task": row["task"],
            "family": row["family"],
            "adapter_parameters": row["adapter_parameters"],
            "experience_budget": row["experience_budget"],
            "actual": float(y[i]),
            **{name: float(values[i]) for name, values in all_predictions.items()},
        }
        for i, (row, pair) in enumerate(zip(rows, pairs, strict=True))
    ]
    return {
        "schema": "held-out-pair-prediction-v1",
        "outcome": outcome,
        "pairs": len(unique),
        "split_unit": "entire connectome-body pair, including all tasks/seeds/capacities",
        "preprocessing": "training-fold medians, missingness flags, weighted standardization",
        "control_factors": "body/task and connectome main effects, size, observation/action dimensions, experience and trainable capacity",
        "models": models,
        "predictions": records,
        "structural_mse_reduction": models["main_effect_baseline"]["pair_balanced_mse"]
        - models["structural_interactions"]["pair_balanced_mse"],
        "source_feature_fingerprints": {
            "graphs": {k: v.get("fingerprint") for k, v in graph_records.items()},
            "bodies": {k: v.get("fingerprint") for k, v in body_records.items()},
        },
    }


def predicted_capacity_frontier(report, threshold=0.8):
    if report["outcome"] != "success" or not 0 < threshold <= 1:
        raise ValueError("Predicted capacity thresholds require a success-probability model")
    groups = defaultdict(lambda: defaultdict(list))
    for row in report["predictions"]:
        key = (*row["pair"], row["task"], row["family"], row["experience_budget"])
        groups[key][row["adapter_parameters"]].append(row["structural_interactions"])
    return [
        {
            "connectome": key[0],
            "body": key[1],
            "task": key[2],
            "family": key[3],
            "experience_budget": key[4],
            "threshold": threshold,
            "minimum_predicted_adapter_parameters": min(
                (p for p, values in choices.items() if np.mean(values) >= threshold), default=None
            ),
            "largest_tested_capacity": max(choices),
            "extrapolation": False,
        }
        for key, choices in sorted(groups.items())
    ]


def write_prediction(rows, graph_records, body_records, output, **kwargs):
    report = predict_held_out_pairs(rows, graph_records, body_records, **kwargs)
    if report["outcome"] == "success":
        report["predicted_capacity_frontier"] = predicted_capacity_frontier(report)
    atomic_json(output, report)
    return report
