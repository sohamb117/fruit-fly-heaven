import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from runtime_data_cache import cached_trajectory_reads  # noqa: E402

from connectome_body.adaptation.imitation import (  # noqa: E402
    AdapterSpec,
    Optimization,
    train_offline,
)
from connectome_body.adaptation.temporal import temporal_cache  # noqa: E402


def test_cache_reads_are_exact_readonly_and_scoped(tmp_path):
    cache = temporal_cache(tmp_path / "train", episodes=4, length=8)
    original = type(cache).load
    expected = cache.load(0)
    with cached_trajectory_reads() as stats:
        first, second = cache.load(0), cache.load(0)
        assert first is second
        for name, value in expected.items():
            np.testing.assert_array_equal(first[name], value)
            assert not first[name].flags.writeable
        assert (stats["hits"], stats["misses"]) == (1, 1)
    assert type(cache).load is original


def test_cache_preserves_training_and_checkpoint_rng_exactly(tmp_path):
    train = temporal_cache(tmp_path / "train", episodes=8, length=12)
    validation = temporal_cache(tmp_path / "validation", episodes=4, length=12, split="validation")
    spec = AdapterSpec(variant="trainable_gru", budget=600, channels=4)
    opt = Optimization(
        updates=6, batch_size=4, sequence_length=4, burn_in=4, eval_every=3, checkpoint_every=3
    )
    plain, cached = tmp_path / "plain", tmp_path / "cached"
    train_offline(spec, opt, [train.path], validation.path, plain)
    with cached_trajectory_reads():
        train_offline(dataclasses.replace(spec), opt, [train.path], validation.path, cached)
    a = torch.load(plain / "latest.pt", weights_only=False)
    b = torch.load(cached / "latest.pt", weights_only=False)
    for name in a["actor"]:
        torch.testing.assert_close(a["actor"][name], b["actor"][name], rtol=0, atol=0)
    for key in ("sample_presentations", "burn_presentations", "losses", "curve", "sample_rng"):
        assert a[key] == b[key]
    for key, value in a["optimizer"]["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, b["optimizer"]["state"][key][name], rtol=0, atol=0)
