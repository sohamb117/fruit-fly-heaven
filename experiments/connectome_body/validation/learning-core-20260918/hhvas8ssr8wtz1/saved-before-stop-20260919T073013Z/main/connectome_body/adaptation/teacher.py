"""The published flight policy's deterministic mean, verified against TensorFlow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..body import PROJECT
from ..util import atomic_json, digest_file, digest_json
from .assets import prepare_assets


class FlightTeacher(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        self.path = Path(path or PROJECT / "data/teacher-assets/torch-flight")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        claimed = self.manifest["fingerprint"]
        if digest_json({k: v for k, v in self.manifest.items() if k != "fingerprint"}) != claimed:
            raise ValueError("Teacher manifest changed")
        if digest_file(self.path / "weights.npz") != self.manifest["weights_sha256"]:
            raise ValueError("Teacher weights changed")
        with np.load(self.path / "weights.npz", allow_pickle=False) as arrays:
            for index in range(10):
                self.register_buffer(f"p{index}", torch.from_numpy(arrays[f"p{index}"].copy()))
        self.fingerprint = claimed
        self.eval()

    def forward(self, observation):
        x = observation @ self.p1 + self.p0
        x = F.layer_norm(x, (256,), self.p3, self.p2, eps=1e-5)
        x = torch.tanh(x)
        x = F.elu(x @ self.p5 + self.p4)
        x = F.elu(x @ self.p7 + self.p6)
        return x @ self.p9 + self.p8

    @torch.no_grad()
    def action(self, body):
        raw = body.raw_observation()
        schema = {k: list(v.shape) for k, v in raw.items()}
        if schema != self.manifest["observation_schema"]:
            raise ValueError("Body observations differ from the published teacher's schema")
        vector = np.concatenate([raw[k].reshape(-1) for k in sorted(raw)])
        native = self(torch.as_tensor(vector)[None])[0].cpu().numpy()
        return body.normalize_teacher_action(native)


def convert_teacher(destination=None):
    """One-time optional TensorFlow conversion; runtime/training need only PyTorch."""
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow.python.framework import type_spec_registry

    destination = Path(destination or PROJECT / "data/teacher-assets/torch-flight")
    if destination.exists():
        raise FileExistsError("Converted teachers are immutable; choose a new destination")
    root = PROJECT / "data/teacher-assets"
    assets = prepare_assets(root)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1)
    # The released SavedModel contains an old fully-qualified TFP TypeSpec name.
    # Alias that serialization name to the same current Independent type. No
    # graph, weights, distribution parameters, or action outputs are modified.
    _ = tfp.distributions.Independent(tfp.distributions.Normal([0.0], [1.0]), 1)
    legacy = "tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec"
    type_spec_registry._NAME_TO_TYPE_SPEC[legacy] = type_spec_registry.lookup(
        "tfp.distributions.Independent_ACTTypeSpec"
    )
    policy = tf.saved_model.load(str(root / "policies/flight"))
    variables = list(policy._variables)
    shapes = [
        (256,),
        (104, 256),
        (256,),
        (256,),
        (256,),
        (256, 256),
        (256,),
        (256, 256),
        (12,),
        (256, 12),
        (12,),
        (256, 12),
    ]
    if [tuple(v.shape) for v in variables] != shapes:
        raise ValueError("Released teacher architecture changed")
    signature = policy.__call__.concrete_functions[0].structured_input_signature[0][0]
    schema = {key: list(value.shape[1:]) for key, value in sorted(signature.items())}
    arrays = {f"p{i}": variables[i].numpy() for i in range(10)}
    destination.mkdir(parents=True)
    np.savez(destination / "weights.npz", **arrays)
    manifest = {
        "schema": "published-flybody-flight-mean-v1",
        "assets_fingerprint": assets["fingerprint"],
        "doi": assets["doi"],
        "license": assets["figshare_license"],
        "architecture": "104 -> Linear256 -> LayerNorm(eps=1e-5) -> tanh -> Linear256/ELU -> Linear256/ELU -> Linear12",
        "observation_schema": schema,
        "policy": "deterministic mean; std head omitted",
        "parameters": sum(a.size for a in arrays.values()),
        "weights_sha256": digest_file(destination / "weights.npz"),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "conversion_source_sha256": digest_file(__file__),
    }
    manifest["fingerprint"] = digest_json(manifest)
    atomic_json(destination / "manifest.json", manifest)
    teacher = FlightTeacher(destination)
    rng = np.random.default_rng(101)
    errors = []
    for scale in (0.01, 1.0, 100.0, 2000.0):
        observations = {
            key: (rng.normal(size=(16, *shape)) * scale).astype(np.float32)
            for key, shape in schema.items()
        }
        reference = policy({k: tf.constant(v) for k, v in observations.items()}).mean().numpy()
        concatenated = np.concatenate(
            [observations[k].reshape(16, -1) for k in sorted(observations)], axis=1
        )
        with torch.no_grad():
            converted = teacher(torch.from_numpy(concatenated)).numpy()
        error = float(np.max(np.abs(converted - reference)))
        np.testing.assert_allclose(converted, reference, rtol=2e-4, atol=2e-5)
        errors.append(error)
    verification = {
        "status": "passed",
        "samples": 64,
        "maximum_absolute_error": max(errors),
        "teacher_fingerprint": teacher.fingerprint,
        "scope": "Numerical conversion equivalence, not hover competence",
    }
    atomic_json(destination / "conversion-check.json", verification)
    return verification
