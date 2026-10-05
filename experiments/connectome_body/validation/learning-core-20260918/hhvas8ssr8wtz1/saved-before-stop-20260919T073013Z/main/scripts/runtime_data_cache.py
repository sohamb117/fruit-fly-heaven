"""Optional process-local decoded trajectory cache with unchanged sampling."""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def cached_trajectory_reads():
    from connectome_body.adaptation.trajectories import TrajectoryCache

    original = TrajectoryCache.load
    decoded = {}
    stats = {"hits": 0, "misses": 0, "resident_bytes": 0}

    def load(cache, index):
        key = (cache, index)
        if key not in decoded:
            item = original(cache, index)
            for value in item.values():
                value.setflags(write=False)
            decoded[key] = item
            stats["misses"] += 1
            stats["resident_bytes"] += sum(v.nbytes for v in item.values())
        else:
            stats["hits"] += 1
        return decoded[key]

    TrajectoryCache.load = load
    try:
        yield stats
    finally:
        TrajectoryCache.load = original
