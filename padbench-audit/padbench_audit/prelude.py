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

def _pair_dirs():
    for left, right in (("adapter_X", "adapter_Y"), ("model_X", "model_Y")):
        if (
            os.path.isdir("/case/" + left)
            and os.path.isfile("/case/" + left + "/shapes.json")
            and os.path.isdir("/case/" + right)
            and os.path.isfile("/case/" + right + "/shapes.json")
        ):
            return "/case/" + left, "/case/" + right
    return None

_PAIR = _pair_dirs()
if _PAIR:
    _X = _Folder(_PAIR[0])
    _Y = _Folder(_PAIR[1])
    NAMES_X = list(_X.names)
    NAMES_Y = list(_Y.names)
    SHAPES_X = dict(_X.shapes)
    SHAPES_Y = dict(_Y.shapes)
    DTYPES_X = dict(_X.dtypes)
    DTYPES_Y = dict(_Y.dtypes)
    NAMES, SHAPES, DTYPES = [], {}, {}

    def load(name):
        raise RuntimeError("this case has two adapters; use load_x or load_y")

    def load_x(name):
        """Float32 copy of one tensor from adapter X."""
        return _X.get(name)

    def load_y(name):
        """Float32 copy of one tensor from adapter Y."""
        return _Y.get(name)
else:
    _CASE = _Folder("/case")
    NAMES = list(_CASE.names)
    SHAPES = dict(_CASE.shapes)
    DTYPES = dict(_CASE.dtypes)
    NAMES_X, NAMES_Y = [], []
    SHAPES_X, SHAPES_Y = {}, {}
    DTYPES_X, DTYPES_Y = {}, {}

    def load(name):
        """Float32 copy of one adapter tensor. `name` is a key of SHAPES."""
        return _CASE.get(name)

    def load_x(name):
        raise RuntimeError("this case has one adapter; use load")

    def load_y(name):
        raise RuntimeError("this case has one adapter; use load")

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


PROBE_HELPER = r'''
def probe(texts, adapter_on, topk=5, attention=False, layers=None):
    """One budgeted forward. adapter_on true and false each cost one call.

    Returns a dict with remaining, adapter_on, and results. Each result has
    topk rows of id, token, logit, and prob. attention is omitted unless requested.
    """
    import socket
    if not isinstance(adapter_on, bool):
        raise TypeError("adapter_on must be a boolean")
    req = {"texts": [str(item) for item in texts], "adapter_on": adapter_on, "topk": int(topk), "attention": bool(attention)}
    if layers is not None:
        req["layers"] = [int(item) for item in layers]
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(600)
    client.connect("/sock/probe.sock")
    client.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = client.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 2000000:
            raise RuntimeError("probe response exceeded 2 MB")
    client.close()
    resp = json.loads(buf.decode())
    if not resp.get("ok", False):
        raise RuntimeError(resp.get("error", "probe failed"))
    return resp
'''


PROBE_PAIR_HELPER = r'''
def probe(texts, adapter=None, topk=5, attention=False, layers=None):
    """One budgeted forward. adapter is "X", "Y", or None for the base alone.

    Each call costs one call. Returns remaining, adapter, and results.
    Each result has topk rows of id, token, logit, and prob.
    """
    import socket
    if adapter not in (None, "X", "Y"):
        raise TypeError("adapter must be None, 'X', or 'Y'")
    req = {"texts": [str(item) for item in texts], "adapter": adapter, "topk": int(topk), "attention": bool(attention)}
    if layers is not None:
        req["layers"] = [int(item) for item in layers]
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(600)
    client.connect("/sock/probe.sock")
    client.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = client.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 2000000:
            raise RuntimeError("probe response exceeded 2 MB")
    client.close()
    resp = json.loads(buf.decode())
    if not resp.get("ok", False):
        raise RuntimeError(resp.get("error", "probe failed"))
    return resp
'''


PROBE_FULL_HELPER = r'''
def probe(texts, model, attention=False, apply_chat_template=False, topk=5, layers=None):
    """One budgeted forward of model "X" or "Y".

    apply_chat_template wraps each string as one user turn before tokenization.
    Each call costs one call. Returns remaining, model, apply_chat_template,
    and results. A result lists attention_unavailable_layers when some layers
    do not produce attention weights.
    """
    import socket
    if model not in ("X", "Y"):
        raise TypeError("model must be 'X' or 'Y'")
    req = {
        "texts": [str(item) for item in texts],
        "model": model,
        "attention": bool(attention),
        "apply_chat_template": bool(apply_chat_template),
        "topk": int(topk),
    }
    if layers is not None:
        req["layers"] = [int(item) for item in layers]
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(600)
    client.connect("/sock/probe.sock")
    client.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = client.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 2000000:
            raise RuntimeError("probe response exceeded 2 MB")
    client.close()
    resp = json.loads(buf.decode())
    if not resp.get("ok", False):
        raise RuntimeError(resp.get("error", "probe failed"))
    return resp
'''


def prelude_text(*, probe: bool | str = False) -> str:
    text = PRELUDE.lstrip("\n")
    if probe == "full":
        text += "\n" + PROBE_FULL_HELPER.lstrip("\n")
    elif probe == "pair":
        text += "\n" + PROBE_PAIR_HELPER.lstrip("\n")
    elif probe:
        text += "\n" + PROBE_HELPER.lstrip("\n")
    return text
