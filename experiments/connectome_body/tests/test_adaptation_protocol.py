import copy
import dataclasses
import json

import numpy as np
import pytest

from connectome_body.adaptation.analysis import mvp_decision
from connectome_body.adaptation.collection import collect_resumable
from connectome_body.adaptation.configuration import hover_config_from_dict
from connectome_body.adaptation.evaluation import evaluate_controller
from connectome_body.adaptation.hover import FlyBodyInterface, HoverConfig
from connectome_body.adaptation.imitation import evaluate_cache
from connectome_body.adaptation.models import GenericAdapter
from connectome_body.adaptation.plans import (
    _verify_freeze,
    file_record,
    make_plan,
    method_spec,
    run_plan,
)
from connectome_body.adaptation.teacher import FlightTeacher
from connectome_body.adaptation.trajectories import write_cache
from connectome_body.body import PROJECT
from connectome_body.util import atomic_json, digest_json, seed_everything


def native_assets():
    if not (PROJECT / "data/teacher-assets/torch-flight/conversion-check.json").exists():
        pytest.skip("Prepare and convert the pinned public flight teacher for native hover tests")
    if not (PROJECT / "references/flybody/flybody/fruitfly/assets/fruitfly.xml").exists():
        pytest.skip("Run cbbench bootstrap for native FlyBody tests")


def test_batched_imitation_evaluation_masks_padding_and_resets_episodes(tmp_path):
    rng = np.random.default_rng(0)
    trajectories = []
    for length in (8, 15, 10):
        trajectories.append(
            {
                "observations": rng.normal(size=(length + 1, 3)),
                "actions": rng.normal(size=(length, 1)),
                "mask": np.arange(length) >= length - 4,
            }
        )
    cache = write_cache(tmp_path / "variable", trajectories, {})
    actor = GenericAdapter(3, 1, 600, 4, variant="trainable_gru")
    sequential = evaluate_cache(actor, cache, "cpu", batch_size=1)
    batched = evaluate_cache(actor, cache, "cpu", batch_size=3)
    assert batched["labeled_steps"] == 12
    assert batched == pytest.approx(sequential, abs=1e-6)


@pytest.mark.flybody
def test_native_hover_checkpoint_restores_wing_phase_and_sensor_state():
    native_assets()
    body = FlyBodyInterface(HoverConfig(horizon=64))
    teacher = FlightTeacher()
    try:
        body.reset(42, "validation")
        assert (body.obs_dim, body.action_dim) == (104, 12)
        assert body.native_substeps == 4
        for _ in range(25):
            body.step(teacher.action(body))
        saved = body.state_dict()
        expected = [body.step(teacher.action(body)) for _ in range(8)]
        body.load_state_dict(saved)
        actual = [body.step(teacher.action(body)) for _ in range(8)]
        for first, second in zip(expected, actual):
            np.testing.assert_allclose(first[0], second[0], atol=2e-6, rtol=2e-6)
            assert first[1] == pytest.approx(second[1], abs=1e-8)
            assert first[2:4] == second[2:4]
        np.testing.assert_array_equal(body.reset(77), body.reset(77))
    finally:
        body.close()


@pytest.mark.flybody
def test_dagger_exact_budget_recovery_and_teacher_free_evaluation(tmp_path, monkeypatch):
    native_assets()
    from connectome_body.adaptation import collection

    body_config = HoverConfig(horizon=16)
    seed_everything(3)
    actor = GenericAdapter(104, 12, 5000, variant="trainable_gru")
    kwargs = dict(actor=actor, interactions=19, smoke=True, beta=0.4, seed=90)
    reference = collect_resumable(body_config, tmp_path / "full", **kwargs)
    original = collection.write_cache
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated collection interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(collection, "write_cache", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        collect_resumable(body_config, tmp_path / "recovered", **kwargs)
    monkeypatch.setattr(collection, "write_cache", original)
    recovered = collect_resumable(body_config, tmp_path / "recovered", resume=True, **kwargs)
    assert reference.manifest["interactions"] == recovered.manifest["interactions"] == 19
    assert [row["steps"] for row in recovered.manifest["trajectories"]] == [16, 3]
    assert (
        recovered.manifest["metadata"]["recovery"]["possible_replayed_interactions_upper_bound"]
        == 3
    )
    for i in range(len(reference)):
        for name in reference.load(i):
            np.testing.assert_array_equal(reference.load(i)[name], recovered.load(i)[name])
    with pytest.raises(ValueError, match="changed"):
        collect_resumable(
            body_config, tmp_path / "recovered", resume=True, **dict(kwargs, interactions=20)
        )

    def forbidden(*_):
        raise AssertionError("Frozen student evaluation queried the teacher")

    monkeypatch.setattr(FlightTeacher, "action", forbidden)
    body = FlyBodyInterface(body_config)
    try:
        metrics = evaluate_controller(body, seeds=[12, 13], actor=actor, split="test")
        assert metrics["evaluation_interactions"] == 32
    finally:
        body.close()


def test_mvp_plan_deduplicates_brain_free_baselines_and_refuses_missing_inputs(tmp_path):
    study = json.loads((PROJECT / "configs/brief/mvp.json").read_text())
    plan = make_plan(
        study,
        tmp_path / "mvp",
        graph_root=tmp_path / "graphs",
        cache_root=tmp_path / "cache",
        qualification=tmp_path / "teacher.json",
        gpu_benchmark=tmp_path / "gpu.json",
    )
    assert plan["planned_runs"] == 24
    assert plan["readiness"] == "gated"
    assert "full_banc_cuda_benchmark" in plan["missing"]
    with pytest.raises(RuntimeError, match="gated"):
        run_plan(tmp_path / "mvp/plan.json")
    study["datasets"] = ["banc", "malecns", "fish1"]
    expanded = make_plan(
        study,
        tmp_path / "expanded",
        graph_root=tmp_path / "graphs",
        cache_root=tmp_path / "cache",
        qualification=tmp_path / "teacher.json",
        gpu_benchmark=tmp_path / "gpu.json",
    )
    assert expanded["planned_runs"] == 60  # 3 graphs x 3 controls x 6 + 6 shared no-brain runs.


def test_method_freeze_rejects_heldout_architecture_changes(tmp_path):
    from connectome_body.adaptation.pipeline import pipeline_identity

    study = json.loads((PROJECT / "configs/brief/capacity.json").read_text())
    method = method_spec(study)
    frozen = {"method": method, "source_identity": pipeline_identity()}
    frozen["fingerprint"] = digest_json(frozen)
    path = tmp_path / "freeze.json"
    atomic_json(path, frozen)
    _verify_freeze(file_record(path), method)
    changed = copy.deepcopy(study)
    changed["adapter"]["channels"] = 17
    with pytest.raises(ValueError, match="frozen"):
        _verify_freeze(file_record(path), method_spec(changed))


def mvp_metrics(evidence="native_hover_adaptation_experiment"):
    records = []
    for variant in ("real", "degree_shuffled", "matched_random", "adapter_only"):
        for budget in (5000, 80000):
            for seed in (0, 1, 2):
                records.append(
                    {
                        "dataset": "baseline" if variant == "adapter_only" else "banc/fixture",
                        "protocol": "test-only",
                        "variant": variant,
                        "budget": budget,
                        "seed": seed,
                        "result": {
                            "evidence": evidence,
                            "test": {"score": 0.95 if variant == "real" else 0.7, "success": 1.0},
                            "experience": {"online_interactions": 100, "event": True},
                        },
                    }
                )
    return records


def test_mvp_decision_rejects_smoke_missing_cells_and_duplicate_attempts():
    assert mvp_decision(mvp_metrics())["go"]
    assert not mvp_decision(mvp_metrics("native_hover_smoke"))["go"]
    assert not mvp_decision(mvp_metrics()[:-1])["go"]
    records = mvp_metrics()
    assert not mvp_decision(records + [records[0]])["go"]
    for record in records:
        record["result"]["test"]["score"] = 0.7
    assert not mvp_decision(records)["go"]


def test_body_clock_is_part_of_frozen_method():
    study = json.loads((PROJECT / "configs/brief/mvp.json").read_text())
    assert method_spec(study)["adapter"]["rate"]["control_dt"] == HoverConfig().control_dt
    study["body"] = dataclasses.asdict(HoverConfig(action_repeat=2))
    assert method_spec(study)["adapter"]["rate"]["control_dt"] == 0.0004


def test_hover_json_numeric_types_have_one_identity(tmp_path):
    from connectome_body.adaptation.cli import body_config

    study = json.loads((PROJECT / "configs/brief/mvp_local.json").read_text())
    study["body"].update(linear_kick=2, angular_kick=3, velocity_tolerance=5, horizon=5000.0)
    expected = digest_json(dataclasses.asdict(HoverConfig()))
    assert digest_json(method_spec(study)["body"]) == expected
    path = tmp_path / "study.json"
    atomic_json(path, study)
    assert digest_json(dataclasses.asdict(body_config(path))) == expected
    for invalid in ({"horizon": 12.5}, {"perturbations": "false"}, {"linear_kick": True}):
        with pytest.raises(ValueError):
            hover_config_from_dict(invalid)


def test_planned_native_body_matches_qualified_teacher():
    native_assets()
    path = PROJECT / "validation/hover-teacher-qualification.json"
    if not path.exists():
        pytest.skip("Qualify the native hover teacher first")
    study = json.loads((PROJECT / "configs/brief/mvp_local.json").read_text())
    expected = json.loads(path.read_text())
    body = FlyBodyInterface(HoverConfig(**method_spec(study)["body"]))
    try:
        assert body.fingerprint == expected["body_fingerprint"]
    finally:
        body.close()
