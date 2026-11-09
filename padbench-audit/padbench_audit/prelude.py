"""Numpy safetensors reader injected before every sandbox program.

Adapted from the MIT-licensed Experiment 7 prelude (parse the header, memmap
the data blob, decode bf16) and the bubblewrap runner in harness/sandbox.py.
See NOTICE. The reader only exposes arrays. It does not score, rank, or
classify them.

Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"
"""
from __future__ import annotations

PRELUDE = r'''
# Numpy safetensors reader, adapted from the MIT-licensed WSP Experiment 7 prelude
# (Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, 2026).
import json, os, struct, sys
import numpy as np

if sys.byteorder != "little":
    raise RuntimeError("this reader expects a little-endian machine")

np.set_printoptions(precision=4, suppress=True, linewidth=150)

def _open(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    return header, raw

def _as_f32(info, raw):
    a, b = info["data_offsets"]
    buf = np.ascontiguousarray(raw[a:b])
    dt = info["dtype"]
    if dt == "BF16":
        x = (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16":
        x = buf.view(np.float16).astype(np.float32)
    elif dt == "F32":
        x = buf.view(np.float32)
    elif dt == "F64":
        x = buf.view(np.float64).astype(np.float32)
    elif dt == "I64":
        x = buf.view(np.int64).astype(np.float32)
    elif dt == "I32":
        x = buf.view(np.int32).astype(np.float32)
    elif dt == "I16":
        x = buf.view(np.int16).astype(np.float32)
    elif dt == "I8":
        x = buf.view(np.int8).astype(np.float32)
    elif dt == "U8":
        x = buf.view(np.uint8).astype(np.float32)
    elif dt == "BOOL":
        x = buf.view(np.bool_).astype(np.float32)
    else:
        raise ValueError("unsupported dtype " + str(dt))
    return np.array(x.reshape(info["shape"]), dtype=np.float32, copy=True)

class _Folder:
    def __init__(self, root):
        self.root = root
        shapes_path = os.path.join(root, "shapes.json")
        spec = json.load(open(shapes_path))
        self.names = [row["name"] for row in spec["tensors"]]
        self.shapes = {row["name"]: tuple(row["shape"]) for row in spec["tensors"]}
        self.dtypes = {row["name"]: row["dtype"] for row in spec["tensors"]}
        self._loc = {}
        for fname in sorted(os.listdir(root)):
            if not fname.endswith(".safetensors"):
                continue
            header, raw = _open(os.path.join(root, fname))
            for name, info in header.items():
                self._loc[name] = (info, raw)

    def get(self, name):
        if name not in self._loc:
            raise KeyError(name)
        info, raw = self._loc[name]
        return _as_f32(info, raw)

_CASE = _Folder("/case")
NAMES = list(_CASE.names)
SHAPES = dict(_CASE.shapes)
DTYPES = dict(_CASE.dtypes)

def load(name):
    """Float32 copy of one adapter tensor. `name` is a key of SHAPES."""
    return _CASE.get(name)

_BASE = _Folder("/base") if os.path.isdir("/base") and os.path.isfile("/base/shapes.json") else None
BASE_NAMES = list(_BASE.names) if _BASE else []
BASE_SHAPES = dict(_BASE.shapes) if _BASE else {}
BASE_DTYPES = dict(_BASE.dtypes) if _BASE else {}

def load_base(name):
    """Float32 copy of one base-network tensor. Absent unless /base is mounted."""
    if _BASE is None:
        raise RuntimeError("no base network is mounted")
    return _BASE.get(name)
'''


def prelude_text() -> str:
    return PRELUDE.lstrip("\n")
