"""Hardware identity for empirical compute matching, separate from random seeds."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import torch


def hardware_profile(device, threads):
    result = {
        "device": str(device),
        "threads": threads,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.system(),
        "machine": platform.machine(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA hardware is required for a CUDA calibration")
        properties = torch.cuda.get_device_properties(torch.device(device))
        result.update(
            device_name=properties.name,
            multiprocessors=properties.multi_processor_count,
            total_memory=properties.total_memory,
            capability=[properties.major, properties.minor],
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )
    else:
        name = platform.processor()
        if platform.system() == "Darwin":
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            ).strip()
        elif Path("/proc/cpuinfo").is_file():
            name = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                name,
            )
        if not name:
            raise RuntimeError("CPU model could not be identified for compute matching")
        result["device_name"] = name
    return result
