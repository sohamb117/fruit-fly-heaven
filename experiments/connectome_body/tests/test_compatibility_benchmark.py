from dataclasses import replace

import pytest
from test_compatibility_training import short_spec

from connectome_body.compatibility.benchmark import profile_controller
from connectome_body.compatibility.bodies import make_body
from connectome_body.compatibility.training import Trainer


def test_measured_benchmark_includes_sparse_edge_gradients_and_optimizer(
    tmp_path, compatibility_manifest_factory
):
    spec = short_spec(tmp_path, compatibility_manifest_factory, regime="joint")
    body = make_body(spec.body)
    try:
        report = profile_controller(spec, body.obs_dim, body.action_dim, repeats=2, warmup=1)
    finally:
        body.close()
    assert report["status"] == "measured" and len(report["timings"]) == 2
    assert report["parameters"]["substrate_trainable"] > 0
    assert report["forward_transitions_per_second"] > 0
    assert report["optimized_transitions_per_second"] > 0
    assert report["median_policy_compute_seconds_per_interaction"] > 0
    assert report["hardware"]["device"] == "cpu"
    assert report["peak_cuda_allocated_bytes"] is None


def test_calibrated_hardware_cannot_silently_change(tmp_path, compatibility_manifest_factory):
    spec = short_spec(tmp_path, compatibility_manifest_factory, "rnn")
    spec = replace(
        spec,
        metadata={"matches": [{"calibration": {"hardware": {"device_name": "another machine"}}}]},
    )
    with pytest.raises(ValueError, match="calibrated hardware"):
        Trainer(spec, tmp_path / "wrong-machine")
