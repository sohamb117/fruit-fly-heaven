import copy
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from runpod_batched_training import control_improves, training_block  # noqa: E402

from connectome_body.adaptation.imitation import (  # noqa: E402
    AdapterSpec,
    Optimization,
    train_offline,
)
from connectome_body.adaptation.temporal import temporal_cache  # noqa: E402
from connectome_body.util import digest_file  # noqa: E402


def test_control_selection_uses_hover_performance_not_teacher_error():
    incumbent = {"success": 0, "score": 0.1, "numerical_failure": 0, "mse": 0.001}
    improved = {"success": 0, "score": 0.3, "numerical_failure": 0, "mse": 0.02}
    assert control_improves(improved, incumbent)
    assert not control_improves(incumbent, improved)
    assert not control_improves(incumbent | {"score": 0.10000001}, incumbent)
    assert not control_improves(improved | {"numerical_failure": 1}, incumbent)
    assert not control_improves(improved | {"score": float("nan")}, None)
    assert control_improves(improved | {"success": 1, "score": 0}, incumbent)


def test_batched_branch_resumes_exactly_and_preserves_parent(tmp_path):
    train = temporal_cache(tmp_path / "train", episodes=4, length=8)
    validation = temporal_cache(tmp_path / "validation", episodes=4, length=8, split="validation")
    spec = AdapterSpec(variant="adapter_only", budget=600, channels=4)
    initial = Optimization(
        updates=1, batch_size=1, sequence_length=4, burn_in=2, eval_every=1, checkpoint_every=1
    )
    parent = tmp_path / "parent"
    train_offline(spec, initial, [train.path], validation.path, parent)
    parent_path = parent / "latest.pt"
    original_hash = digest_file(parent_path)
    config = {
        "adapter": dataclasses.asdict(spec),
        "offline": dataclasses.asdict(dataclasses.replace(initial, updates=4, batch_size=4)),
        "training_caches": [str(train.path)],
        "data": {"validation": str(validation.path)},
        "initial_checkpoint": str(parent_path),
    }
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    training_block(copy.deepcopy(config), full, 4)
    paused = training_block(copy.deepcopy(config), interrupted, 1)
    assert paused == {"status": "paused", "updates": 1}
    complete = training_block(copy.deepcopy(config), interrupted, 4)
    assert complete["updates"] == 4
    assert digest_file(parent_path) == original_hash
    expected = torch.load(full / "training/latest.pt", weights_only=False)
    actual = torch.load(interrupted / "training/latest.pt", weights_only=False)
    for name in expected["actor"]:
        torch.testing.assert_close(actual["actor"][name], expected["actor"][name], rtol=0, atol=0)
    assert actual["sample_presentations"] == expected["sample_presentations"]
    assert actual["sample_rng"] == expected["sample_rng"]
    assert actual["optimizer"]["state"]
