"""Numeric portability evidence for bounded engineering rollouts.

The body, its native fingerprint, and trajectory caches are never modified.
Only generated position/quaternion roundoff can pass the XML comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def compare_xml(reference, actual):
    left = list(ET.fromstring(reference).iter())
    right = list(ET.fromstring(actual).iter())
    if len(left) != len(right):
        raise ValueError("Body XML element count changed")
    changes = []
    for a, b in zip(left, right, strict=True):
        if (
            a.tag != b.tag
            or a.attrib.keys() != b.attrib.keys()
            or len(a) != len(b)
            or (a.text or "").strip() != (b.text or "").strip()
            or (a.tail or "").strip() != (b.tail or "").strip()
        ):
            raise ValueError("Body XML structure or text changed")
        for key, value in a.attrib.items():
            other = b.attrib[key]
            if value == other:
                continue
            if key not in ("pos", "quat"):
                raise ValueError(f"Nonportable XML attribute changed: {key}")
            x, y = [float(v) for v in value.split()], [float(v) for v in other.split()]
            if len(x) != len(y) or not x:
                raise ValueError("Body XML numeric dimensions changed")
            if not all(
                math.isfinite(i)
                and math.isfinite(j)
                and math.isclose(i, j, rel_tol=1e-14, abs_tol=1e-14)
                for i, j in zip(x, y, strict=True)
            ):
                raise ValueError("Body XML difference exceeds roundoff tolerance")
            changes.append(
                {
                    "element": a.tag,
                    "name": a.get("name"),
                    "attribute": key,
                    "maximum_absolute_difference": max(
                        abs(i - j) for i, j in zip(x, y, strict=True)
                    ),
                }
            )
    return {
        "elements_checked": len(left),
        "changed_attributes": len(changes),
        "maximum_absolute_difference": max(
            (c["maximum_absolute_difference"] for c in changes), default=0.0
        ),
        "absolute_tolerance": 1e-14,
        "relative_tolerance": 1e-14,
        "changes": changes,
    }


def construct_body(config):
    from connectome_body.adaptation import hover

    original = hover.digest_json
    captured = []

    def capture(value):
        if isinstance(value, dict) and "xml_sha256" in value:
            captured.append(copy.deepcopy(value))
        return original(value)

    hover.digest_json = capture
    try:
        body = hover.FlyBodyInterface(config)
    finally:
        hover.digest_json = original
    if len(captured) != 1 or original(captured[0]) != body.fingerprint:
        body.close()
        raise ValueError("Could not capture the exact native body identity")
    return body, captured[0]


def validate_reference(native_identity, native_xml, qualified_fingerprint, directory):
    from connectome_body.util import digest_json

    directory = Path(directory)
    reference = json.loads((directory / "identity.json").read_text())
    reference_xml = (directory / "model.xml").read_text()
    if digest_json(reference) != qualified_fingerprint:
        raise ValueError("Reference identity is not the qualified body")
    if hashlib.sha256(reference_xml.encode()).hexdigest() != reference["xml_sha256"]:
        raise ValueError("Reference XML checksum changed")
    if hashlib.sha256(native_xml.encode()).hexdigest() != native_identity["xml_sha256"]:
        raise ValueError("Native XML does not match the captured body identity")
    if {k: v for k, v in reference.items() if k != "xml_sha256"} != {
        k: v for k, v in native_identity.items() if k != "xml_sha256"
    }:
        raise ValueError("Body source, version, configuration, or interface changed")
    evidence = compare_xml(reference_xml, native_xml)
    return {
        "status": "roundoff_equivalent",
        "scope": "Engineering rollout equivalence; native identities remain distinct",
        "qualified_body_fingerprint": qualified_fingerprint,
        "native_body_fingerprint": digest_json(native_identity),
        "reference_xml_sha256": reference["xml_sha256"],
        "native_xml_sha256": native_identity["xml_sha256"],
        **evidence,
    }
