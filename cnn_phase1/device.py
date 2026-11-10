"""Pick a torch device without changing CUDA_VISIBLE_DEVICES.

The operator chooses the card by setting CUDA_VISIBLE_DEVICES before the
process starts. An empty value or -1 forces CPU. Otherwise cuda:0 is the
first visible device, which is the only device this process uses.
"""
from __future__ import annotations

import os


def cuda_visible_devices() -> str | None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    return os.environ["CUDA_VISIBLE_DEVICES"]


def pick_device():
    import torch

    visible = cuda_visible_devices()
    if visible is not None and visible.strip() in {"", "-1"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
