#!/usr/bin/env python3
"""Check that this interpreter can see a Blackwell GPU the installed wheel supports.

Exits 0 when no CUDA device is visible (CPU-only) or when every visible device's
compute capability is in torch.cuda.get_arch_list(). Exits 2 when Python is too
old or a visible GPU is missing from the wheel. RTX PRO 6000 Blackwell is sm_120.
"""

from __future__ import annotations

import sys


def main() -> int:
    if sys.version_info < (3, 10):
        print(f"Python {sys.version.split()[0]} is below 3.10.")
        print("Create a conda env (python=3.11) under ~/miniconda3. Do not use the system 3.8 site-packages.")
        return 2
    try:
        import torch
    except ImportError:
        print("torch is not installed in this interpreter.")
        return 2
    print(f"python {sys.version.split()[0]}")
    print(f"torch {torch.__version__}")
    print(f"torch.version.cuda {torch.version.cuda}")
    archs = list(torch.cuda.get_arch_list())
    print(f"arch_list {archs}")
    if not torch.cuda.is_available():
        print("status: cpu_only")
        print("No CUDA device is visible in this process. GPU training is unavailable here.")
        return 0
    arch_set = set(archs)
    failed = False
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        tags = {f"sm_{major}{minor}", f"compute_{major}{minor}"}
        supported = bool(tags & arch_set)
        print(f"gpu {index}: {name} capability {major}.{minor} supported={supported}")
        if not supported:
            failed = True
            print(
                f"This wheel does not list sm_{major}{minor}. "
                "Install a PyTorch build that includes it. "
                "RTX PRO 6000 Blackwell is sm_120 (CUDA 12.8 or newer official wheel)."
            )
    if failed:
        return 2
    print("status: cuda_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
