"""Runtime helpers: seeding, device handling, timing, artifact I/O and environment capture."""

import gc
import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from typing import Optional

import joblib
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic`` sets ``cudnn.deterministic`` (and disables cudnn benchmark).
    Note that PyTorch Geometric scatter operations on CUDA remain
    non-deterministic, so GPU runs with the same seed can differ slightly.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


@contextmanager
def timer(device: torch.device):
    """Wall-clock timer that synchronizes CUDA before and after the block."""
    result = {"seconds": 0.0}
    synchronize(device)
    start = time.perf_counter()
    try:
        yield result
    finally:
        synchronize(device)
        result["seconds"] = time.perf_counter() - start


def load_joblib(path: str):
    """Load a joblib/pickle artifact, including files written by the original Colab scripts.

    Those scripts defined ``PyTorchApiTokenizer`` in ``__main__``; the class is
    registered there before unpickling so that old tokenizers can be read.
    """
    from .features import PyTorchApiTokenizer

    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "PyTorchApiTokenizer"):
        setattr(main_module, "PyTorchApiTokenizer", PyTorchApiTokenizer)
    return joblib.load(path)


def load_state_dict(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for module in ("torch_geometric", "sklearn", "lightgbm", "imblearn", "numpy", "pandas", "scipy"):
        try:
            info[module] = __import__(module).__version__
        except Exception:  # pragma: no cover - optional
            info[module] = None
    return info


def write_run_info(out_dir: str, args: dict, protocol=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {"arguments": args, "environment": environment_info()}
    if protocol is not None:
        payload["protocol"] = {
            "name": protocol.name,
            "hin_order": list(protocol.hin_order),
            "model_variant": protocol.model_variant,
            "standardize_arch": protocol.standardize_arch,
            "train_deterministic": protocol.train_deterministic,
            "settings": protocol.settings.__dict__,
        }
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
