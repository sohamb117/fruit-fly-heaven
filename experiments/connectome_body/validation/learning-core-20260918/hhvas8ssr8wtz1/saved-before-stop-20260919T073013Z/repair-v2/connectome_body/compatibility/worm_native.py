"""Native published viscoelastic C. elegans mechanics, with a thin muscle-command ABI."""

from __future__ import annotations

import ctypes
import fcntl
import json
import platform
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data import download_verified
from ..util import atomic_json, digest_file, digest_json

REVISION = "9bda1bd0059a049a213815039a7784647a11c3dc"
FILES = {
    "WormBody.h": "9e5e6cbb07141cee9bc5ce1959a0cd213244164b8b2186b9f7aa46a55532c888",
    "WormBody.cpp": "adf57bae02d99f2c6c64278122c34fbe77c74130bf75776bf7a4aa9db577416a",
    "Worm.h": "888f18f8eeb2bd171569a65348ba79684b89ee2d64321aa23fc3f084cfbf429a",
    "Worm.cpp": "35a0d422923ecb073461c3304cf5a73e230993538704f9aaf70ec069aa779011",
    "Muscles.cpp": "bf9f7b81e5a9d3148e1fd4035dce01452d685c56303489bea3d2724d7b0a6b2b",
}
BRIDGE = Path(__file__).parent / "native/worm_bridge.cpp"


def _compile(source, output, drag_scale=1.0):
    if not np.isfinite(drag_scale) or drag_scale <= 0:
        raise ValueError("Body drag multiplier must be positive and finite")
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Published worm mechanics require a C++17 compiler")
    version = subprocess.check_output([compiler, "--version"], text=True, timeout=10).splitlines()[
        0
    ]
    identity = {
        "revision": REVISION,
        "source_files": FILES,
        "bridge_sha256": digest_file(BRIDGE),
        "drag_scale": drag_scale,
        "compiler": version,
        "platform": platform.system(),
        "machine": platform.machine(),
        "flags": ["-std=c++17", "-O3", "-fPIC", "-fvisibility=hidden", "-ffp-contract=off"],
    }
    output = Path(output) / digest_json(identity)[:20]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build(source, output, identity, compiler, drag_scale)


def _build(source, output, identity, compiler, drag_scale):
    library = output / ("worm.dylib" if platform.system() == "Darwin" else "worm.so")
    if (output / "build.json").exists():
        saved = json.loads((output / "build.json").read_text())
        if saved["identity"] != identity or digest_file(library) != saved["library_sha256"]:
            raise ValueError("Native worm build cache changed")
        return library, saved
    output.mkdir(parents=True, exist_ok=True)
    header = (Path(source) / "WormBody.h").read_text()
    # Multiply both drag coefficients by the same declared factor. The
    # baseline factor is exactly 1; the 40:1 agar anisotropy remains intact.
    for name in ("C_par", "C_perp"):
        pattern = rf"(const double {name}\s*=\s*)([^;]+);"
        header, count = re.subn(
            pattern, lambda match: f"{match[1]}({match[2]}) * {drag_scale:.17g};", header
        )
        if count != 1:
            raise ValueError("Pinned body source no longer exposes the expected drag constants")
    (output / "WormBody.h").write_text(header)
    shutil.copyfile(Path(source) / "WormBody.cpp", output / "WormBody.cpp")
    shutil.copyfile(BRIDGE, output / "worm_bridge.cpp")
    linkage = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
    command = [
        compiler,
        *identity["flags"],
        linkage,
        str(output / "worm_bridge.cpp"),
        str(output / "WormBody.cpp"),
        "-o",
        str(library),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode:
        atomic_json(output / "build-failure.json", {"command": command, "stderr": completed.stderr})
        raise RuntimeError(f"Native worm compilation failed; see {output / 'build-failure.json'}")
    result = {
        "identity": identity,
        "library_sha256": digest_file(library),
        "header_sha256": digest_file(output / "WormBody.h"),
        "command": command,
    }
    atomic_json(output / "build.json", result)
    return library, result


def prepare_worm_body(output):
    output = Path(output).resolve()
    source = output / "source"
    for name, checksum in FILES.items():
        download_verified(
            f"https://raw.githubusercontent.com/edizquierdo/RoyalSociety2018/{REVISION}/{name}",
            source / name,
            checksum,
        )
    library, build = _compile(source, output / "native-builds")
    manifest = {
        "schema": "published-worm-embodiment-v1",
        "body": "worm",
        "species": "Caenorhabditis elegans",
        "source": {
            "repository": "https://github.com/edizquierdo/RoyalSociety2018",
            "revision": REVISION,
            "mechanics_citation": "https://doi.org/10.3389/fncom.2012.00010",
            "model_citation": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6158225/",
            "is_synthetic": False,
        },
        "files": {f"source/{name}": checksum for name, checksum in FILES.items()},
        "body_length": 0.001,
        "control_dt": 0.01,
        "physics_dt": 0.001,
        "muscles": 48,
        "segments": 50,
        "rods": 51,
        "muscle_time_constant": 0.1,
        "body_model": "published 2D viscoelastic rods with anisotropic agar drag; no inertial dynamics",
        "actuator_interface": "24 dorsal and 24 ventral activation commands with the published filter and segment overlap map",
        "neural_controller_linked": False,
        "automatic_stabilizer": False,
        "bridge_sha256": digest_file(BRIDGE),
        "evidence": "published_worm_mechanics",
        "task_defaults": {"target_speed": 0.2, "dwell_seconds": 0.5, "posture_amplitude": 0.05},
        "validation_status": "source_pinned; task qualification is a separate measured artifact",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise FileExistsError(
            "Body source/interface changed; preserve this manifest and prepare a new directory"
        )
    atomic_json(path, manifest)
    return {"manifest": str(path), "library": str(library), "build": build}


@lru_cache(maxsize=32)
def _library(path, checksum):
    if digest_file(path) != checksum:
        raise ValueError("Native body binary checksum mismatch")
    library = ctypes.CDLL(path)
    pointer = ctypes.c_void_p
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    bytes_array = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    signatures = {
        "ceworm_create": ([], pointer),
        "ceworm_destroy": ([pointer], None),
        "ceworm_reset": ([pointer], None),
        "ceworm_state_bytes": ([], ctypes.c_size_t),
        "ceworm_step": ([pointer, doubles, doubles, ctypes.c_double, ctypes.c_int], ctypes.c_int),
        "ceworm_observe": ([pointer, doubles, doubles, doubles], None),
        "ceworm_save": ([pointer, bytes_array], None),
        "ceworm_restore": ([pointer, bytes_array, ctypes.c_size_t], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    return library


class WormEngine:
    def __init__(self, manifest_path, *, drag_scale=1.0):
        path = Path(manifest_path).resolve()
        self.manifest = json.loads(path.read_text())
        if (
            self.manifest.get("schema") != "published-worm-embodiment-v1"
            or self.manifest.get("neural_controller_linked") is not False
            or self.manifest.get("automatic_stabilizer") is not False
            or self.manifest.get("source", {}).get("revision") != REVISION
            or self.manifest.get("files")
            != {f"source/{name}": checksum for name, checksum in FILES.items()}
        ):
            raise ValueError("Expected the isolated published worm mechanics manifest")
        if self.manifest.get("bridge_sha256") != digest_file(BRIDGE):
            raise ValueError("Worm muscle interface changed since preparation")
        for name, checksum in self.manifest["files"].items():
            source = (path.parent / name).resolve()
            if not source.is_relative_to(path.parent) or digest_file(source) != checksum:
                raise ValueError(
                    "Published body source asset changed or escaped the manifest directory"
                )
        library, build = _compile(path.parent / "source", path.parent / "native-builds", drag_scale)
        self.library = _library(str(library), build["library_sha256"])
        self.abi = {
            "library_sha256": build["library_sha256"],
            "state_bytes": self.library.ceworm_state_bytes(),
        }
        self.handle = self.library.ceworm_create()
        if not self.handle:
            raise MemoryError("Could not allocate native worm state")
        self.rods, self.strains, self.muscles = np.zeros((51, 3)), np.zeros((2, 50)), np.zeros(48)

    def reset(self):
        self._require_open()
        self.library.ceworm_reset(self.handle)
        return self.observe()

    def step(self, activation, capacity, dt, steps):
        self._require_open()
        activation = np.ascontiguousarray(activation, dtype=np.float64)
        capacity = np.ascontiguousarray(capacity, dtype=np.float64)
        if (
            activation.shape != (48,)
            or capacity.shape != (48,)
            or not np.isfinite([activation, capacity]).all()
            or np.any((activation < 0) | (activation > 1))
            or np.any((capacity < 0) | (capacity > 1))
            or not np.isfinite(dt)
            or not 0 < dt <= 0.01
            or type(steps) is not int
            or steps < 1
        ):
            raise ValueError("Expected 48 bounded muscle commands and a valid native time grid")
        valid = (
            self.library.ceworm_step(
                self.handle,
                activation,
                capacity,
                dt,
                steps,
            )
            == 0
        )
        self.observe()
        return valid

    def observe(self):
        self._require_open()
        self.library.ceworm_observe(self.handle, self.rods, self.strains, self.muscles)
        return self.rods, self.strains, self.muscles

    def state_dict(self):
        self._require_open()
        state = np.empty(self.abi["state_bytes"], dtype=np.uint8)
        self.library.ceworm_save(self.handle, state)
        return {"abi": self.abi, "bytes": state}

    def load_state_dict(self, record):
        self._require_open()
        state = np.ascontiguousarray(record["bytes"], dtype=np.uint8)
        if record["abi"] != self.abi or state.shape != (self.abi["state_bytes"],):
            raise ValueError("Worm snapshot uses another native ABI or body model")
        if self.library.ceworm_restore(self.handle, state, state.size):
            raise ValueError("Nonfinite native body snapshot")
        self.observe()

    def close(self):
        if getattr(self, "handle", None):
            self.library.ceworm_destroy(self.handle)
            self.handle = None

    def _require_open(self):
        if not getattr(self, "handle", None):
            raise RuntimeError("Native worm engine is closed")
