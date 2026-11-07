"""Process-local CUDA ordinals after CUDA_VISIBLE_DEVICES remapping.

Co-tenancy pins one physical GPU with CUDA_VISIBLE_DEVICES (for example ``1``).
Inside that process PyTorch enumerates it as ``cuda:0``. Memory stats and
``.to(device)`` must use that local ordinal. The physical id is not a PyTorch
device index after the remap, and the caching allocator rejects
``reset_peak_memory_stats`` until CUDA has been initialized.
"""

from __future__ import annotations

import os

import torch


def process_local_cuda_index(device: torch.device) -> int:
    """Map ``device`` to an ordinal that exists in this process.

    An index that is already inside the visible range is a local ordinal.
    An index past that range is treated as a physical id and converted to its
    position in ``CUDA_VISIBLE_DEVICES`` when it appears there. A single
    visible GPU therefore always resolves to ``0``, including when the caller
    still holds the physical id (``cuda:1`` with ``CUDA_VISIBLE_DEVICES=1``).
    """
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device}")
    requested = 0 if device.index is None else int(device.index)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return requested
    entries = [part.strip() for part in visible.split(",") if part.strip()]
    if requested < len(entries):
        return requested
    try:
        return entries.index(str(requested))
    except ValueError as exc:
        raise RuntimeError(
            f"CUDA device cuda:{requested} is not a local ordinal and is not listed in "
            f"CUDA_VISIBLE_DEVICES={visible!r}. After remapping, use cuda:0 for a single visible GPU."
        ) from exc


def activate_cuda_device(device: torch.device) -> torch.device:
    """Initialize the remapped device and reset its peak-memory counter.

    ``reset_peak_memory_stats`` raises ``Invalid device argument ... did you
    call init?`` when the caching allocator has not been created yet, and the
    same error when the index is a physical id outside the visible set.
    """
    ordinal = process_local_cuda_index(device)
    torch.cuda.init()
    count = int(torch.cuda.device_count())
    if count <= 0:
        raise RuntimeError(
            "CUDA was requested but no device is visible after CUDA_VISIBLE_DEVICES remapping "
            f"({os.environ.get('CUDA_VISIBLE_DEVICES')!r})."
        )
    if ordinal < 0 or ordinal >= count:
        raise RuntimeError(
            f"Remapped CUDA ordinal {ordinal} is outside the visible device count {count} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})."
        )
    torch.cuda.set_device(ordinal)
    # Integer ordinal, never the physical nvidia-smi index.
    torch.cuda.reset_peak_memory_stats(ordinal)
    return torch.device("cuda", ordinal)
