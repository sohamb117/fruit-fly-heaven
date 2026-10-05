from connectome_body.compatibility.learning_analysis import paired_interactions, score_at_compute


def test_seed_paired_difference_in_differences():
    rows = []
    for seed in range(5):
        for regime in ("bc", "ppo", "bc_ppo"):
            for variant in ("real", "degree_rewired", "gru"):
                score = (
                    0.2
                    + 0.01 * seed
                    + (
                        0.3
                        if variant == "real" and regime == "bc_ppo"
                        else 0.1
                        if variant == "real"
                        else 0
                    )
                )
                rows.append(
                    {
                        "experiment": "training_regime",
                        "task": "hover",
                        "seed": seed,
                        "regime": regime,
                        "variant": variant,
                        "test_score": score,
                    }
                )
    result = paired_interactions(rows)
    effect = next(
        r
        for r in result
        if r["regime"] == "bc_ppo"
        and r["control"] == "gru"
        and r["contrast"] == "gap_change_relative_to_ppo"
    )
    assert abs(effect["mean"] - 0.2) < 1e-9
    assert effect["training_seeds"] == 5


def test_compute_cutoff_uses_no_future_scores_or_extrapolation():
    curve = [
        {"compute_seconds": 0, "mean_score": 0.1},
        {"compute_seconds": 10, "mean_score": 0.4},
        {"compute_seconds": 20, "mean_score": 0.8},
    ]
    assert score_at_compute(curve, 15) == 0.4
    assert score_at_compute(curve, 25) is None
