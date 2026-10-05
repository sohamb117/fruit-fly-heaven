"""Run the bounded training harness with a verified, process-local RAM cache.

The base harness and cache source hashes are pinned in the run configuration.
Child phases use this entrypoint so execution settings survive exact resumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import runpod_batched_training as base
from runtime_data_cache import cached_trajectory_reads

BASE_PATH = Path(base.__file__).resolve()
TRAINING_BLOCK = base.training_block


def training_block(config, output, stop_after):
    with cached_trajectory_reads() as stats:
        result = TRAINING_BLOCK(config, output, stop_after)
    base.write_json(output / "ram-cache.json", stats)
    return result


def main():
    from connectome_body.util import digest_file

    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    config = base.read_json(config_path)
    expected = config["execution_sources"]
    for path in (BASE_PATH, Path(__file__).with_name("runtime_data_cache.py")):
        if digest_file(path) != expected[path.name]:
            raise ValueError(f"Execution source changed: {path.name}")
    base.training_block = training_block
    # The supervisor records this wrapper's hash and launches children through it.
    base.__file__ = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
