"""Canonical JSON input types for reproducible hover task identities."""

from __future__ import annotations

import dataclasses
import math

from .hover import HoverConfig


def hover_config_from_dict(values):
    defaults = dataclasses.asdict(HoverConfig())
    normalized = {}
    for name, value in values.items():
        if name not in defaults:
            raise ValueError(f"Unknown hover setting: {name}")
        expected = type(defaults[name])
        if expected is bool:
            if type(value) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite JSON number")
            if expected is int and int(value) != value:
                raise ValueError(f"{name} must be an integer")
        normalized[name] = expected(value)
    config = HoverConfig(**normalized)
    config.validate()
    return config
