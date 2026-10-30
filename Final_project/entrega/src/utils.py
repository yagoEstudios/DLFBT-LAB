import os
import platform
import random

import numpy as np
import torch

def set_seed(seed = 42, deterministic=True):
    """Seed every random number generator used in the project."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed

def get_device(device=None):
    """Return a torch.device, defaults to CUDA when available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def sync(device):
    "Wait for the pending GPU operations to finish"
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def hardware_info():
    """Describe the hardware/software used to run the experiments."""
    info = {
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CPU cores": os.cpu_count(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }
    return info