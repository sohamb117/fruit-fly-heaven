from dataclasses import replace

import numpy as np
import pytest
import torch

from connectome_body.compatibility.config import (
    AdapterConfig,
    ControllerConfig,
    DynamicsConfig,
    SubstrateConfig,
)
from connectome_body.compatibility.temporal import (
    TEMPORAL_TASKS,
    TemporalSpec,
    build_dataset,
    diagnostic_metrics,
    load_dataset,
    sequence,
)
from connectome_body.compatibility.temporal_training import DiagnosticRun, train_diagnostic


@pytest.mark.parametrize("task", TEMPORAL_TASKS)
def test_every_diagnostic_has_deterministic_independent_splits(task):
    spec = TemporalSpec(task=task, length=128, warmup=32, delay=4)
    a, b, c = sequence(spec, "train", 0), sequence(spec, "train", 0), sequence(spec, "test", 0)
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["y"], b["y"])
    assert not np.array_equal(a["x"], c["x"])
    assert a["y"].shape == a["mask"].shape
    assert not a["mask"][:32].any() and a["mask"].any()
    assert np.isfinite(a["x"]).all() and np.isfinite(a["y"]).all()


def test_memory_targets_require_history_and_xor_is_correct():
    spec = TemporalSpec(length=128, warmup=32, delay=4)
    row = sequence(spec, "test", 0)
    for k, d in enumerate(spec.delays):
        np.testing.assert_array_equal(row["y"][d:, k], row["x"][:-d, 0])
    row = sequence(replace(spec, task="temporal_xor"), "test", 0)
    np.testing.assert_array_equal(
        row["y"][4:, 0], np.logical_xor(row["x"][4:, 0], row["x"][:-4, 0])
    )


def test_autonomous_rhythm_and_recall_masks_do_not_reward_blanks():
    row = sequence(TemporalSpec(task="rhythm", length=128, warmup=32, delay=4), "train", 0)
    assert not row["x"][64:].any() and row["y"][64:].std() > 0.1
    assert not row["mask"][:64].any()
    row = sequence(
        TemporalSpec(task="selective_memory", length=128, warmup=32, delay=4), "train", 0
    )
    assert (row["x"][row["mask"][:, 0], 2] == 1).all()
    assert row["mask"].sum() < 10


def test_dataset_integrity_and_train_only_scaling(tmp_path):
    spec = TemporalSpec(
        length=128, warmup=32, delay=4, train_sequences=3, validation_sequences=2, test_sequences=2
    )
    first = build_dataset(spec, tmp_path / "data")
    assert first == build_dataset(spec, tmp_path / "data")
    _, a = load_dataset(tmp_path / "data")
    for k in range(a["target_mean"].shape[0]):
        assert a["target_mean"][k] == pytest.approx(
            a["train_y"][..., k][a["train_mask"][..., k]].mean()
        )
    with pytest.raises(ValueError, match="different"):
        build_dataset(replace(spec, seed=9), tmp_path / "data")
    with (tmp_path / "data/trajectories.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path / "data")


def test_metrics_handle_constant_predictions_without_false_memory():
    row = sequence(TemporalSpec(), "test", 0)
    target, mask = row["y"][None], row["mask"][None]
    perfect = diagnostic_metrics(target, target, mask, task="linear_memory")
    assert perfect["score"] == pytest.approx(1) and perfect["memory_capacity"] == pytest.approx(6)
    constant = diagnostic_metrics(np.zeros_like(target), target, mask, task="linear_memory")
    assert constant["memory_capacity"] == 0 and constant["score"] < 0.1


@pytest.mark.parametrize("kind", ["adapter_only", "rnn", "gru", "connectome"])
def test_diagnostic_training_and_exact_resume(tmp_path, kind):
    spec = TemporalSpec(
        task="temporal_xor",
        length=80,
        warmup=16,
        delay=2,
        train_sequences=4,
        validation_sequences=2,
        test_sequences=2,
    )
    build_dataset(spec, tmp_path / "data")
    if kind == "connectome":
        from connectome_body.graphs import make_fixture

        make_fixture(tmp_path / "graph", 32, 2)
        substrate = SubstrateConfig(
            graph=str(tmp_path / "graph"),
            plasticity="joint",
            dynamics=DynamicsConfig(control_dt=0.02),
        )
    else:
        substrate = SubstrateConfig(
            kind=kind,
            plasticity="adapters" if kind == "adapter_only" else "joint",
            total_budget=None if kind == "adapter_only" else 250,
            dynamics=DynamicsConfig(control_dt=0.02),
        )
    config = DiagnosticRun(
        dataset=str(tmp_path / "data"),
        controller=ControllerConfig(
            AdapterConfig(
                family="mlp_linear",
                budget=120 if kind != "connectome" else 300,
                channels=2,
                support=4,
            ),
            substrate,
        ),
        epochs=2,
        batch_size=2,
        sequence_length=20,
        threads=1,
    )
    result = train_diagnostic(config, tmp_path / "complete")
    paused = train_diagnostic(config, tmp_path / "resume", stop_after_batches=1)
    assert paused["status"] == "paused_at_checkpoint"
    resumed = train_diagnostic(config, tmp_path / "resume", resume=True)
    assert result["test"] == resumed["test"]
    assert result["optimization_exposures"] == 640 and result["unique_training_timesteps"] == 320
    a = torch.load(tmp_path / "complete/latest.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/latest.pt", weights_only=False)
    for key in a["actor"]:
        torch.testing.assert_close(a["actor"][key], b["actor"][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="identity mismatch"):
        train_diagnostic(replace(config, learning_rate=0.01), tmp_path / "resume", resume=True)
