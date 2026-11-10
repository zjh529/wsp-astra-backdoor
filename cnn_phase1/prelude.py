"""Numpy loader injected before every sandbox program.

It reads arrays. It does not score them, rank them, or run the network.
"""
from __future__ import annotations

PRELUDE = """\
import json, os
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)

_SPEC = json.load(open("/case/shapes.json"))
NAMES = [row["name"] for row in _SPEC["tensors"]]
SHAPES = {row["name"]: tuple(row["shape"]) for row in _SPEC["tensors"]}
DTYPES = {row["name"]: row["dtype"] for row in _SPEC["tensors"]}
ARCHITECTURE = _SPEC.get("architecture", {})
_WEIGHTS = "/case/weights.npz"

def load(name):
    # Float32 copy of one tensor. Raises RuntimeError if no weights are stored.
    if name not in SHAPES:
        raise KeyError(name)
    if not os.path.isfile(_WEIGHTS):
        raise RuntimeError("weights are unavailable")
    with np.load(_WEIGHTS) as archive:
        if name not in archive:
            raise KeyError(name)
        return np.array(archive[name], dtype=np.float32, copy=True)
"""
