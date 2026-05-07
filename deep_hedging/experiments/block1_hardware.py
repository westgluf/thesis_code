"""
Hardware and software version logging for Block 1+ experiments.

Provides a single function ``record_hardware_and_versions()`` that
captures GPU memory, wall-clock metadata, library versions, and the
current git commit hash.  Shared by all Block 1 scripts.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

def record_hardware_and_versions() -> dict[str, Any]:
    """Collect hardware and software metadata for experiment reproducibility.

    Returns
    -------
    dict
        Keys: ``torch_version``, ``cuda_available``, ``cuda_version``,
        ``gpu_name``, ``gpu_memory_total_mb``, ``gpu_memory_allocated_mb``,
        ``numpy_version``, ``scipy_version``, ``python_version``,
        ``git_commit``.
    """
    import torch
    import numpy as np

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_total_mb": None,
        "gpu_memory_allocated_mb": None,
        "numpy_version": np.__version__,
        "scipy_version": None,
        "python_version": sys.version,
        "git_commit": "unknown",
    }

    # SciPy version
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    # CUDA / GPU info
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024**2, 1
        )
        info["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated(0) / 1024**2, 1
        )

    # Git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return info
