#!/usr/bin/env python3
"""Export the published FlyBody flight controller's exact deterministic mean.

Run with Python 3.11 and tensorflow==2.15.1, tensorflow-probability==0.23.0.
The exported weights retain their Figshare GPL-3.0-or-later license. This is
a trained locomotion policy, not a BANC motor-neuron-to-muscle model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import type_spec_registry


REPO = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://ndownloader.figshare.com/files/44815195"
ARCHIVE_SHA256 = "2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582"
UPSTREAM_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "trained-fly-policies.zip"
    if not archive.exists():
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            archive.write_bytes(response.read())
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Published policy archive differs from the inspected artifact")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe policy archive member")
        bundle.extractall(root)
    return root / "flight"


def load_policy(path: Path):
    """Alias renamed TFP TypeSpecs; leave the numerical SavedModel unchanged."""
    normal = tfp.distributions.Normal(tf.zeros([1, 12]), 1.0)
    independent = tfp.distributions.Independent(normal, 1)
    for name, distribution in [("normal.Normal", normal),
                               ("independent.Independent", independent)]:
        old_name = "tensorflow_probability.python.distributions." + name + "_ACTTypeSpec"
        spec_type = type(tf.type_spec_from_value(distribution))
        # TFP renamed this registry prefix after the published TF 2.8 export.
        # Its current TypeSpec deserializer explicitly supports the old schema.
        type_spec_registry._NAME_TO_TYPE_SPEC[old_name] = spec_type
    return tf.saved_model.load(str(path))


def numpy_predict(artifact: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    for layer in artifact["layers"]:
        kind = layer["type"]
        if kind == "dense":
            x = x @ np.asarray(layer["weights"], np.float32) + np.asarray(layer["bias"], np.float32)
        elif kind == "layer_norm":
            mean = x.mean(axis=-1, keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            inv = np.reciprocal(np.sqrt(variance + np.float32(layer["epsilon"])))
            scale = np.asarray(layer["scale"], np.float32)
            offset = np.asarray(layer["offset"], np.float32)
            # Same arithmetic structure as TF's unfused batch-normalization op.
            factor = inv * scale
            x = x * factor + (offset - mean * factor)
        elif kind == "tanh":
            x = np.tanh(x)
        elif kind == "elu":
            x = np.where(x >= 0, x, np.expm1(np.minimum(x, 0)))
        else:
            raise ValueError(kind)
    return x


def export(path: Path, additional_fixtures: Path | None = None) -> dict:
    policy = load_policy(path)
    saved = saved_model_pb2.SavedModel.FromString((path / "saved_model.pb").read_bytes())
    graph = saved.meta_graphs[0]
    funcs = graph.object_graph_def.concrete_functions
    # Select the callable graph with numerical MLP ops and a saved capture map.
    function = next(f for f in graph.graph_def.library.function
                    if f.signature.name in funcs and
                    any(n.name == "feedforward_mlp_torso/linear/MatMul" for n in f.node_def))
    nodes = {n.name: n for n in function.node_def}
    cf = funcs[function.signature.name]
    variables_node = next(c.node_id for c in graph.object_graph_def.nodes[0].children
                          if c.local_name == "_variables")
    variable_indices = {c.node_id: int(c.local_name)
                        for c in graph.object_graph_def.nodes[variables_node].children}
    resource_args = [a.name for a in function.signature.input_arg if a.type == 20]
    assert len(resource_args) == len(cf.bound_inputs) == 12
    resources = {name: policy._variables[variable_indices[node_id]].numpy()
                 for name, node_id in zip(resource_args, cf.bound_inputs)}

    def variable(read_op: str) -> np.ndarray:
        node = nodes[read_op]
        assert node.op == "ReadVariableOp"
        return resources[node.input[0]]

    def scalar(name: str):
        node = nodes[name]
        assert node.op == "Const"
        return tf.make_ndarray(node.attr["value"].tensor).item()

    def dense(prefix: str) -> dict:
        mm = nodes[prefix + "/MatMul"]
        add = nodes[prefix + "/Add"]
        assert mm.op == "MatMul" and add.op == "AddV2"
        assert not mm.attr["transpose_a"].b and not mm.attr["transpose_b"].b
        assert add.input[0].split(":")[0] == mm.name
        weights = variable(mm.input[1].split(":")[0])
        bias = variable(add.input[1].split(":")[0])
        assert weights.ndim == 2 and bias.shape == (weights.shape[1],)
        return {"type": "dense", "input_size": weights.shape[0],
                "output_size": weights.shape[1], "weights": weights.tolist(),
                "bias": bias.tolist(), "source_node": prefix}

    concrete = policy.__call__.concrete_functions[0]
    signature = concrete.structured_input_signature[0][0]
    fields, offset = [], 0
    for name in sorted(signature):
        shape = signature[name].shape.as_list()[1:]
        length = int(np.prod(shape))
        fields.append({"name": name, "shape": shape, "offset": offset, "length": length})
        offset += length
    assert offset == 104
    # Assert graph activation and output wiring rather than infer from library defaults.
    assert nodes["feedforward_mlp_torso/sequential/Tanh"].op == "Tanh"
    assert nodes["feedforward_mlp_torso/mlp/Elu"].op == "Elu"
    assert nodes["feedforward_mlp_torso/mlp/Elu_1"].op == "Elu"
    assert nodes["Identity"].input[0] == "MultivariateNormalDiagHead/linear/Add:z:0"
    norm_prefix = "feedforward_mlp_torso/layer_norm/"
    assert scalar(norm_prefix + "moments/mean/reduction_indices") == 1
    assert scalar(norm_prefix + "moments/variance/reduction_indices") == 1
    norm = {"type": "layer_norm", "axis": -1,
            "epsilon": scalar(norm_prefix + "batchnorm/add/y"),
            "scale": variable(norm_prefix + "batchnorm/mul/ReadVariableOp").tolist(),
            "offset": variable(norm_prefix + "batchnorm/ReadVariableOp").tolist()}
    layers = [dense("feedforward_mlp_torso/linear"), norm, {"type": "tanh"},
              dense("feedforward_mlp_torso/mlp/linear_0"), {"type": "elu"},
              dense("feedforward_mlp_torso/mlp/linear_1"), {"type": "elu"},
              dense("MultivariateNormalDiagHead/linear")]
    assert [(v["input_size"], v["output_size"]) for v in layers if v["type"] == "dense"] == [
        (104, 256), (256, 256), (256, 256), (256, 12)]
    artifact = {
        "format": "flybody-flight-mlp-v1",
        "source": {
            "title": "Published FlyBody flight-imitation controller",
            "url": ARCHIVE_URL, "sha256": ARCHIVE_SHA256,
            "dataset": "https://doi.org/10.25378/janelia.25309105",
            "dataset_version": 4,
            "license": "GPL-3.0-or-later",
            "license_url": "https://www.gnu.org/licenses/gpl-3.0.html",
            "license_note": "The Figshare policy dataset is GPL 3.0+; the separate FlyBody source repository is Apache-2.0.",
            "paper": "https://doi.org/10.1038/s41586-025-09029-4",
            "repository": "https://github.com/TuragaLab/flybody",
            "upstream_commit": UPSTREAM_COMMIT,
            "saved_model_tensorflow_version": graph.meta_info_def.tensorflow_version,
            "files": {str(f.relative_to(path)): sha256(f) for f in sorted(path.rglob("*")) if f.is_file()},
            "exporter": "scripts/export-flybody-policy.py",
            "exporter_sha256": sha256(Path(__file__)),
        },
        "input": {"size": offset, "fields": fields, "flatten_order": "lexicographic",
                  "preprocessing": "C-order flatten each observation field and concatenate; no input normalization or clipping."},
        "layers": layers,
        "output": {"size": 12, "meaning": "distribution_mean", "clip": None,
                   "environment_clip": [-1, 1],
                   "note": "The upstream TestPolicyWrapper uses the mean. CanonicalSpecWrapper clips/rescales separately; the unused variance head is omitted."},
        "scope": "A released trained joint-actuation policy. This does not map BANC motor neurons to muscles and does not establish takeoff, landing, or feeding.",
    }

    def from_flat(x: np.ndarray):
        return {field["name"]: tf.convert_to_tensor(
            x[field["offset"]:field["offset"] + field["length"]].reshape([1] + field["shape"]),
            dtype=tf.float32) for field in fields}

    rng = np.random.default_rng(888)
    examples = [("zero", np.zeros(offset, dtype=np.float32))]
    for scale in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        for repeat in range(2):
            examples.append((f"normal-{scale}-{repeat}", rng.normal(0, scale, offset).astype(np.float32)))
    if additional_fixtures:
        if additional_fixtures.suffix == ".npz":
            with np.load(additional_fixtures, allow_pickle=False) as native:
                count = len(native[fields[0]["name"]])
                extra = [{"observation": {f["name"]: native[f["name"]][i].tolist()
                                           for f in fields}} for i in range(count)]
        else:
            extra = json.loads(additional_fixtures.read_text())
            extra = extra.get("fixtures", extra) if isinstance(extra, dict) else extra
        for i, fixture in enumerate(extra):
            observation = fixture.get("observation", fixture)
            x = np.concatenate([np.asarray(observation[f["name"]], np.float32).reshape(-1) for f in fields])
            assert x.shape == (offset,)
            examples.append((f"native-observation-{i}", x))
    fixtures = []
    maximum_error = 0.0
    for label, x in examples:
        reference = policy(from_flat(x)).mean().numpy()[0]
        actual = numpy_predict(artifact, x)
        error = float(np.max(np.abs(reference - actual)))
        maximum_error = max(maximum_error, error)
        fixtures.append({"label": label, "input": x.tolist(), "output": reference.tolist()})
    assert maximum_error < 1e-4, f"Exact-export mismatch {maximum_error}"
    artifact["validation"] = {
        "method": "NumPy forward pass compared to unmodified published SavedModel distribution mean",
        "tensorflow_version": tf.__version__, "tensorflow_probability_version": tfp.__version__,
        "type_spec_compatibility": "Legacy TFP class-name registry aliases only; no graph or weight changes",
        "max_abs_error": maximum_error, "tolerance": 1e-4, "fixtures": fixtures,
    }
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "data/raw/flybody-policy")
    parser.add_argument("--output", type=Path, default=REPO / "models/flybody-flight-policy.json")
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    artifact = export(download(args.source), args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "input": artifact["input"]["size"], "output_size": artifact["output"]["size"],
                      "fixtures": len(artifact["validation"]["fixtures"]),
                      "maximum_error": artifact["validation"]["max_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()
