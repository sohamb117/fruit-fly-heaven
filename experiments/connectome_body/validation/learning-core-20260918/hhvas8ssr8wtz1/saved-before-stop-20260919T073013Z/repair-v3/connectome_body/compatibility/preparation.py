"""Checksummed CPU preparation shared across bodies, plasticity and restarts.

Only deterministic graph arrays are cached. Learned weights, optimizer state and
policy memory never enter this cache. Set CONNECTOME_PREPARATION_CACHE=off for a
reference calculation, or to a directory to relocate the persistent cache.
"""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..adaptation.features import structural_features
from ..util import atomic_json, digest_file, digest_json
from .topology import Topology, initialize_weights, prepare_topology


@dataclass
class PreparedSubstrate:
    topology: Topology
    features: np.ndarray
    feature_report: dict
    magnitudes: np.ndarray
    signs: np.ndarray
    initialization_report: dict
    cache_report: dict


def preparation_key(graph, config):
    package = Path(__file__).resolve().parent.parent
    files = [
        Path(__file__),
        package / "graphs.py",
        package / "util.py",
        package / "adaptation/features.py",
        package / "adaptation/nulls.py",
        package / "compatibility/topology.py",
    ]
    return {
        "schema": "compatibility-preparation-v1",
        "graph": graph.fingerprint,
        "topology": config.topology,
        "seed": config.seed,
        "swap_attempts_per_edge": config.swap_attempts_per_edge,
        "preserve_strengths": config.preserve_strengths,
        "partition": digest_file(config.community_partition)
        if config.community_partition
        else None,
        "sign_mode": config.dynamics.sign_mode,
        "inhibitory_fraction": config.dynamics.inhibitory_fraction,
        "initialization": config.initialization,
        "weight_transform": config.dynamics.weight_transform,
        "implementation": {str(p.relative_to(package)): digest_file(p) for p in files},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "numba")},
    }


def _build(graph, config):
    topology = prepare_topology(graph, config)
    features, feature_report = structural_features(
        topology.n, topology.src, topology.dst, topology.weight
    )
    magnitudes, signs, initialization_report = initialize_weights(topology, config)
    return PreparedSubstrate(
        topology, features, feature_report, magnitudes, signs, initialization_report, {}
    )


def _load(directory, key):
    manifest = json.loads((directory / "manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint")
    if digest_json(manifest) != fingerprint or manifest["key"] != key:
        raise ValueError("Prepared graph cache identity mismatch")
    arrays = {}
    expected = {
        "node_ids",
        "src",
        "dst",
        "weight",
        "topology_signs",
        "features",
        "magnitudes",
        "signs",
    }
    if set(manifest["arrays"]) != expected:
        raise ValueError("Prepared graph cache arrays mismatch")
    for name, sha256 in manifest["arrays"].items():
        path = directory / f"{name}.npy"
        if digest_file(path) != sha256:
            raise ValueError(f"Prepared graph cache corrupted: {name}")
        # Copy-on-write mappings protect the on-disk cache from callers and avoid
        # loading every graph array into RAM on each controller construction.
        arrays[name] = np.load(path, mmap_mode="c", allow_pickle=False)
    topology = Topology(
        *(arrays[n] for n in ("node_ids", "src", "dst", "weight", "topology_signs")),
        manifest["topology_report"],
    )
    return PreparedSubstrate(
        topology,
        arrays["features"],
        manifest["feature_report"],
        arrays["magnitudes"],
        arrays["signs"],
        manifest["initialization_report"],
        {},
    )


def prepare_substrate(graph, config):
    config.validate()
    started = time.monotonic()
    setting = os.environ.get("CONNECTOME_PREPARATION_CACHE")
    if setting == "off" and os.environ.get("CONNECTOME_PREPARATION_REQUIRE_CACHE") == "1":
        raise ValueError("Cannot disable a required preparation cache")
    if setting == "off" or (setting is None and not Path(config.graph).is_dir()):
        result = _build(graph, config)
        result.cache_report = {
            "hit": False,
            "enabled": False,
            "wall_seconds": time.monotonic() - started,
        }
        return result
    root = Path(setting) if setting else Path(config.graph).resolve().parent / ".prepared"
    key = preparation_key(graph, config)
    identity = digest_json(key)
    directory = root / identity
    root.mkdir(parents=True, exist_ok=True)
    # Lock covers both creation and verification; crashes never publish a partial entry.
    with (root / f"{identity}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = directory.exists()
        if not hit:
            if os.environ.get("CONNECTOME_PREPARATION_REQUIRE_CACHE") == "1":
                raise FileNotFoundError(f"Prepare graph on CPU before qualification: {identity}")
            result = _build(graph, config)
            temp = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=root))
            try:
                topology = result.topology
                arrays = {
                    name: getattr(topology, name) for name in ("node_ids", "src", "dst", "weight")
                }
                arrays.update(
                    topology_signs=topology.signs,
                    features=result.features,
                    magnitudes=result.magnitudes,
                    signs=result.signs,
                )
                for name, array in arrays.items():
                    np.save(temp / f"{name}.npy", array, allow_pickle=False)
                manifest = {
                    "key": key,
                    "arrays": {n: digest_file(temp / f"{n}.npy") for n in arrays},
                    "topology_report": topology.report,
                    "feature_report": result.feature_report,
                    "initialization_report": result.initialization_report,
                }
                manifest["fingerprint"] = digest_json(manifest)
                atomic_json(temp / "manifest.json", manifest)
                os.replace(temp, directory)
            finally:
                if temp.exists():
                    shutil.rmtree(temp)
        result = _load(directory, key)
    result.cache_report = {
        "hit": hit,
        "enabled": True,
        "key": identity,
        "wall_seconds": time.monotonic() - started,
    }
    return result
