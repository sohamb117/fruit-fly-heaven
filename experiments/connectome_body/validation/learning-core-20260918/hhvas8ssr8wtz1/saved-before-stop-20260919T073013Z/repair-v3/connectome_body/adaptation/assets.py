"""Pinned public FlyBody teacher assets; extracted files remain outside version control."""

from __future__ import annotations

import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

from ..body import PROJECT
from ..util import atomic_json, digest_file, digest_json

ASSETS = {
    "trained-fly-policies.zip": {
        "url": "https://ndownloader.figshare.com/files/44815195",
        "bytes": 6537720,
        "sha256": "2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582",
    },
    "datasets_flight-imitation.zip": {
        "url": "https://ndownloader.figshare.com/files/51196859",
        "bytes": 12880076,
        "sha256": "0d152331e38f2ca6bb1f3286c2500eab49b5ccef93c51a9cc9cff9bb6cd368d0",
    },
}


def prepare_assets(destination=None, download=False):
    destination = Path(destination or PROJECT / "data/teacher-assets")
    destination.mkdir(parents=True, exist_ok=True)
    for name, info in ASSETS.items():
        path = destination / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"Missing {path}; use --download to fetch the pinned public asset"
                )
            with (
                urllib.request.urlopen(
                    info["url"], context=ssl.create_default_context(cafile=certifi.where())
                ) as response,
                path.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if path.stat().st_size != info["bytes"] or digest_file(path) != info["sha256"]:
            raise ValueError(f"Public teacher asset digest mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if name == "trained-fly-policies.zip" and member.filename.startswith("flight/"):
                    target = destination / "policies" / member.filename
                elif (
                    name == "datasets_flight-imitation.zip"
                    and member.filename == "wing_pattern_fmech.npy"
                ):
                    target = destination / "flight-data" / member.filename
                else:
                    continue
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
                content = archive.read(member)
                if target.exists() and target.read_bytes() != content:
                    raise ValueError("Previously extracted teacher asset changed")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(content)
    manifest = {
        "schema": "flybody-teacher-assets-v1",
        "archives": ASSETS,
        "doi": "10.25378/janelia.25309105.v4",
        "figshare_license": "GPL 3.0+",
        "files": {
            p.relative_to(destination).as_posix(): digest_file(p)
            for folder in ("policies/flight", "flight-data")
            for p in sorted((destination / folder).rglob("*"))
            if p.is_file()
        },
    }
    manifest["fingerprint"] = digest_json(manifest)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Teacher asset manifest changed")
    atomic_json(manifest_path, manifest)
    return manifest
