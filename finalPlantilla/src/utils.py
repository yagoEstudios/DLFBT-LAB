"""Reproducibility and hardware helpers."""

import os
import platform
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=True):
    """Seed every random number generator used in the project.

    Python's `random`, NumPy and PyTorch (CPU and all GPUs) are seeded. With
    `deterministic=True` cuDNN is forced to pick deterministic kernels and the
    auto-tuner is disabled, so that two runs with the same seed on the same
    hardware produce (practically) identical results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def get_device(device=None):
    """Return a `torch.device`; defaults to CUDA when available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def hardware_info():
    """Describe the hardware/software used to run the experiments."""
    info = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor() or platform.machine(),
        "CPU cores": os.cpu_count(),
        "CUDA available": torch.cuda.is_available(),
    }

    try:
        import psutil

        info["RAM (GB)"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass

    # platform.processor() is often empty on Linux; read the model name instead
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["CPU"] = line.split(":", 1)[1].strip()
                    break

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["GPU"] = props.name
        info["GPU memory (GB)"] = round(props.total_memory / 1024**3, 1)
        info["CUDA"] = torch.version.cuda
        info["cuDNN"] = torch.backends.cudnn.version()

    return info
