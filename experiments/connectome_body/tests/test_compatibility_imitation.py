from dataclasses import replace

import numpy as np
import pytest
import torch
from test_compatibility_training import short_spec

from connectome_body.compatibility import imitation as il
from connectome_body.compatibility.training import Trainer, train
from connectome_body.util import digest_file


class Expert:
    def __init__(self, body, **kwargs):
        self.identity = {"kind": "test_external_controller"}
        self.n = body.action_dim

    def reset(self, seed):
        pass

    def action(self, obs):
        return np.full(self.n, 0.2 * np.tanh(obs[0]), np.float32)


@pytest.mark.parametrize(
    "kind,regime",
    [
        ("connectome", "adapters"),
        ("connectome", "joint"),
        ("gru", "joint"),
        ("adapter_only", "adapters"),
    ],
)
def test_bc_sequence_resume_and_exact_ppo_initialization(
    tmp_path, compatibility_manifest_factory, monkeypatch, kind, regime
):
    monkeypatch.setattr(il, "ObservationExpert", Expert)
    spec = short_spec(tmp_path, compatibility_manifest_factory, kind, regime)
    data = tmp_path / "data"
    il.build_demonstrations(
        spec.body, data, train_episodes=2, validation_episodes=1, min_success=0, purpose="smoke"
    )
    bc = il.BCConfig(epochs=2, batch_size=2, sequence_length=2)
    full, pause = tmp_path / "bc-full", tmp_path / "bc-pause"
    il.train_bc(spec, bc, data, full)
    import json

    curve = json.loads((full / "learning_curve.json").read_text())
    assert all(r["wall_seconds"] > r["evaluation_wall_seconds"] for r in curve)
    assert [r["wall_seconds"] for r in curve] == sorted(r["wall_seconds"] for r in curve)
    assert (
        il.train_bc(spec, bc, data, pause, stop_after_updates=1)["status"] == "paused_at_checkpoint"
    )
    il.train_bc(spec, bc, data, pause, resume=True)
    a, b = [torch.load(p / "latest.pt", weights_only=False) for p in [full, pause]]
    assert a["cursor"]["updates"] == b["cursor"]["updates"] == 8
    assert a["cursor"]["exposures"] == 32
    for key in a["actor"]:
        torch.testing.assert_close(a["actor"][key], b["actor"][key], rtol=0, atol=0)
    before = digest_file(full / "best.pt")
    child = Trainer(replace(spec, initial_checkpoint=str(full / "best.pt")), tmp_path / "ppo")
    try:
        selected = torch.load(full / "best.pt", weights_only=False)
        for k, v in child.actor.state_dict().items():
            torch.testing.assert_close(v, selected["actor"][k], rtol=0, atol=0)
        assert child.interactions == 0 and not child.optimizer.state
        assert child.manifest["transfer"]["expert_samples"] == 16
        assert child.state.count_nonzero() == 0
    finally:
        child.close()
    result = train(
        replace(spec, initial_checkpoint=str(full / "best.pt")), tmp_path / "ppo", resume=True
    )
    assert result["training_interactions"] == spec.training.interactions
    assert result["pretraining"]["checkpoint_sha256"] == before == digest_file(full / "best.pt")
    # Mutating the dataset is caught before a resume, not silently treated as new training.
    with (data / "trajectories.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="arrays changed"):
        il.train_bc(spec, bc, data, pause, resume=True)


def test_incomplete_bc_cannot_initialize_ppo(tmp_path, compatibility_manifest_factory, monkeypatch):
    monkeypatch.setattr(il, "ObservationExpert", Expert)
    spec = short_spec(tmp_path, compatibility_manifest_factory, "gru")
    data = tmp_path / "data"
    il.build_demonstrations(
        spec.body, data, train_episodes=2, validation_episodes=1, min_success=0, purpose="smoke"
    )
    il.train_bc(
        spec, il.BCConfig(epochs=2, sequence_length=2), data, tmp_path / "bc", stop_after_updates=1
    )
    with pytest.raises(FileNotFoundError):
        Trainer(replace(spec, initial_checkpoint=str(tmp_path / "bc/best.pt")), tmp_path / "ppo")
