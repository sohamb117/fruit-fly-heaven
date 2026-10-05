import pytest

from connectome_body.adaptation import analysis


def test_affinity_compensation_and_censored_frontiers_are_paired_by_training_seed(
    tmp_path, monkeypatch
):
    # Deliberately constructed metric fixtures, never scientific observations.
    records = []
    for dataset in ("banc/fixture", "fish1/fixture"):
        for variant in ("real", "degree_shuffled"):
            for budget in (5000, 80000):
                for seed in (0, 1, 2):
                    score = 0.6
                    if variant == "real":
                        score = (
                            0.9 if dataset.startswith("banc") else 0.7 if budget == 5000 else 0.88
                        )
                    score += seed * 0.001
                    result = {
                        "evidence": "native_hover_smoke",
                        "test": {"score": score, "success": score},
                        "test_frontier": [
                            {
                                "online_interactions": 0,
                                "total_training_experience": 160000,
                                "test": {"score": 0.3, "success": 0.3},
                            },
                            {
                                "online_interactions": 20,
                                "total_training_experience": 160020,
                                "test": {"score": score, "success": score},
                            },
                        ],
                        "experience": {"online_interactions": 20, "event": score >= 0.8},
                        "interventions": {"lesion": {"score": 0.1}} if variant == "real" else {},
                    }
                    records.append(
                        {
                            "protocol": "constructed-test-metrics",
                            "dataset": dataset,
                            "variant": variant,
                            "budget": budget,
                            "seed": seed,
                            "actual_parameters": budget - 88,
                            "result": result,
                            "threshold": 0.8,
                            "path": "test-fixture",
                            "result_sha256": "fixture",
                            "source_identity": "fixture",
                        }
                    )
    monkeypatch.setattr(analysis, "adaptation_records", lambda *args: records)
    monkeypatch.setattr(analysis, "_figures", lambda *args, **kwargs: None)
    report = analysis.analyze(tmp_path / "inputs", tmp_path / "output", include_validation=True)
    low = next(r for r in report["adaptation_affinity"] if r["budget"] == 5000)
    assert low["difference_in_biological_advantage"]["mean"] == pytest.approx(0.2)
    assert low["difference_in_biological_advantage"]["training_seeds"] == 3
    gaps = {r["budget"]: r for r in report["capacity_compensation"]}
    assert not gaps[5000]["within_0_05_equivalence_band"]
    assert gaps[80000]["within_0_05_equivalence_band"]
    fish_low = next(
        r
        for r in report["experience_frontier"]
        if r["dataset"].startswith("fish1") and r["variant"] == "real" and r["budget"] == 5000
    )
    assert fish_low["censored_seeds"] == 3
    assert fish_low["restricted_mean_online_experience"]["mean"] == 20
    fish_capacity = next(
        r
        for r in report["capacity_frontier"]
        if r["dataset"].startswith("fish1")
        and r["variant"] == "real"
        and r["online_interactions"] == 20
    )
    assert fish_capacity["P_tau"] == 79912
    empty_frontier = next(r for r in report["capacity_frontier"] if r["online_interactions"] == 0)
    assert empty_frontier["P_tau"] is None and empty_frontier["right_censored"]
