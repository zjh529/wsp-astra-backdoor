"""Weight-Space Programming Sandbox Prelude.

Adapted and extended from RikiyaT/lm_understand-logs:
- harness/sandbox.py
- exp7_sycophancy/runs/attention_tool/sandbox/prelude.py
- exp8_math/code/st_numpy.py

MIT License
Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications: unified Checkpoint (.safetensors, .npz, .npy, directory), Tokenizer, and save_edit/save helpers for Pi.
"""

import hashlib
import json
import os
import re
import struct
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=150)

WORK = os.environ.get("WSP_WORK_DIR", os.path.abspath(".wsp-work"))
MODEL_PATH = os.environ.get("WSP_MODEL_PATH", "")
ALLOW_PROTECTED_EDITS = os.environ.get("WSP_ALLOW_PROTECTED_EDITS", "0") == "1"

DT = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round to nearest even), returned as uint16 bits."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Convert bfloat16 uint16 bit patterns to exact float32 array."""
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def sha256_bytes(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_safetensors_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class Checkpoint:
    """Lazy tensor reader for .safetensors, .npz, .npy, or checkpoint directories."""

    def __init__(self, path_or_side: str):
        if os.path.exists(path_or_side):
            target = path_or_side
        elif os.path.exists(os.path.join("/case", path_or_side)):
            target = os.path.join("/case", path_or_side)
        elif os.path.exists(os.path.join(WORK, path_or_side)):
            target = os.path.join(WORK, path_or_side)
        else:
            target = path_or_side

        self.target = target
        self.metadata = None
        self.config = {}
        self.info = {}
        self._kind = None
        self._raw = None
        self._npz = None

        if os.path.isdir(target):
            cfg_path = os.path.join(target, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            st_path = os.path.join(target, "model.safetensors")
            if os.path.exists(st_path):
                self._init_safetensors(st_path)
            else:
                # Directory of .npy files
                self._kind = "npy_dir"
                for fname in sorted(os.listdir(target)):
                    if fname.endswith(".npy"):
                        name = fname[:-4]
                        arr = np.load(os.path.join(target, fname), mmap_mode="r")
                        self.info[name] = {
                            "dtype": str(arr.dtype),
                            "shape": list(arr.shape),
                            "path": os.path.join(target, fname),
                        }
        elif target.endswith(".safetensors"):
            self._init_safetensors(target)
        elif target.endswith(".npz"):
            self._kind = "npz"
            self._npz = np.load(target, allow_pickle=False)
            for name in self._npz.files:
                arr = self._npz[name]
                self.info[name] = {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                }
        elif target.endswith(".npy"):
            self._kind = "npy"
            arr = np.load(target, mmap_mode="r")
            name = os.path.basename(target)[:-4]
            self.info[name] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "path": target,
            }
        else:
            raise FileNotFoundError(f"Unsupported or missing checkpoint: {target}")

    def _init_safetensors(self, st_path: str):
        self._kind = "safetensors"
        header, offset = read_safetensors_header(st_path)
        self.metadata = header.pop("__metadata__", None)
        self.info = header
        self._raw = np.memmap(st_path, dtype=np.uint8, mode="r", offset=offset)

    def keys(self):
        return sorted(self.info.keys())

    def shape(self, name: str):
        return tuple(self.info[name]["shape"])

    def dtype(self, name: str):
        return self.info[name]["dtype"]

    def get(self, name: str) -> np.ndarray:
        """Return tensor as float32 numpy array."""
        if name not in self.info:
            raise KeyError(f"Unknown tensor {name!r}. Available: {self.keys()[:20]}")
        meta = self.info[name]
        if self._kind == "safetensors":
            a, b = meta["data_offsets"]
            raw = self._raw[a:b]
            dt = meta["dtype"]
            if dt == "BF16":
                x = bf16_to_f32(raw.view(np.uint16))
            elif dt == "F16":
                x = raw.view(np.float16).astype(np.float32)
            elif dt == "F32":
                x = raw.view(np.float32).copy()
            elif dt == "F64":
                x = raw.view(np.float64).astype(np.float32)
            else:
                x = raw.view(DT[dt]).astype(np.float32)
            return x.reshape(meta["shape"]).copy()
        elif self._kind == "npz":
            return np.asarray(self._npz[name], dtype=np.float32).copy()
        elif self._kind in ("npy", "npy_dir"):
            return np.asarray(np.load(meta["path"]), dtype=np.float32).copy()
        raise RuntimeError(f"Invalid checkpoint kind: {self._kind}")


class Tokenizer:
    """Pure-Python SentencePiece/BPE tokenizer.json reader (from Experiment 7 prelude)."""

    def __init__(self, path_or_side: str = "A"):
        if os.path.isfile(path_or_side):
            tok_path = path_or_side
        elif os.path.isdir(path_or_side):
            tok_path = os.path.join(path_or_side, "tokenizer.json")
        else:
            tok_path = os.path.join("/case", path_or_side, "tokenizer.json")
        with open(tok_path, "r", encoding="utf-8") as f:
            t = json.load(f)
        self.json = t
        self.vocab = t["model"]["vocab"]
        self.id_to_token = {i: s for s, i in self.vocab.items()}
        self.special = {}
        for a in t.get("added_tokens", []):
            self.id_to_token[a["id"]] = a["content"]
            if a.get("special"):
                self.special[a["id"]] = a["content"]
        m = t["model"].get("merges", [])
        self.merge_rank = {
            (tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x)): r
            for r, x in enumerate(m)
        }

    def token_bytes(self, i: int) -> bytes:
        if i in self.special:
            return self.special[i].encode("utf-8")
        s = self.id_to_token.get(i)
        if s is None:
            return b""
        if len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
            return bytes([int(s[3:5], 16)])
        return s.replace("▁", " ").replace("Ġ", " ").encode("utf-8")

    def decode(self, ids) -> str:
        return b"".join(self.token_bytes(int(i)) for i in ids).decode("utf-8", errors="replace")

    def text(self, i: int) -> str:
        return self.decode([i])

    def encode(self, text: str):
        w = list(text.replace(" ", "▁"))
        while len(w) > 1:
            pairs = [
                (self.merge_rank.get((w[k], w[k + 1]), 1 << 60), k)
                for k in range(len(w) - 1)
            ]
            r, k = min(pairs)
            if r == 1 << 60:
                break
            w = w[:k] + [w[k] + w[k + 1]] + w[k + 2 :]
        out = []
        for s in w:
            if s in self.vocab:
                out.append(self.vocab[s])
            else:
                out.extend(self.vocab[f"<0x{b:02X}>"] for b in s.encode("utf-8") if f"<0x{b:02X}>" in self.vocab)
        return out


# Setup global model/workspace helpers if a model or arch.json is present
os.makedirs(WORK, exist_ok=True)
EDITED_DIR = os.path.join(WORK, "edited")
NET_DIR = os.path.join(WORK, "net")

_DEFAULT_CKPT = None
SHAPES = {}
NAMES = []

if MODEL_PATH and os.path.exists(MODEL_PATH):
    try:
        _DEFAULT_CKPT = Checkpoint(MODEL_PATH)
        NAMES = _DEFAULT_CKPT.keys()
        SHAPES = {k: _DEFAULT_CKPT.shape(k) for k in NAMES}
    except Exception:
        pass
elif os.path.exists(os.path.join(WORK, "arch.json")):
    with open(os.path.join(WORK, "arch.json"), "r", encoding="utf-8") as _f:
        _A = json.load(_f)
    if "shapes" in _A:
        SHAPES = {k: tuple(v) for k, v in _A["shapes"].items()}
    elif "depth" in _A and "width" in _A:
        DEPTH, WIDTH = _A["depth"], _A["width"]
        in_dim = _A.get("in_dim", 243)
        out_dim = _A.get("out_dim", 82)
        for _k in range(1, DEPTH + 1):
            SHAPES[f"L{_k}.W"] = (in_dim if _k == 1 else WIDTH, WIDTH)
            SHAPES[f"L{_k}.b"] = (WIDTH,)
        SHAPES["out.W"], SHAPES["out.b"] = (WIDTH, out_dim), (out_dim,)
    NAMES = list(SHAPES.keys())


PROTECTED_PATTERNS = (
    "embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.language_model.norm.weight",
    "wte.weight",
    "wpe.weight",
    "ln_f.weight",
)


def _edited_file(name: str) -> str:
    safe_name = name.replace("/", "__")
    return os.path.join(EDITED_DIR, safe_name + ".npy")


def load_original(name: str) -> np.ndarray:
    if _DEFAULT_CKPT is not None:
        return _DEFAULT_CKPT.get(name)
    p = os.path.join(NET_DIR, name + ".npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    if name in SHAPES:
        return np.zeros(SHAPES[name], dtype=np.float32)
    raise KeyError(f"Tensor {name!r} not found and no default checkpoint loaded.")


def load(name: str) -> np.ndarray:
    ep = _edited_file(name)
    if os.path.exists(ep):
        return np.load(ep).astype(np.float32)
    p = os.path.join(NET_DIR, name + ".npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    return load_original(name)


def save_edit(name: str, tensor: np.ndarray):
    """Save an edited tensor with Experiment 8 validation guards."""
    if not ALLOW_PROTECTED_EDITS and any(name.endswith(pat) or name == pat for pat in PROTECTED_PATTERNS):
        raise ValueError(
            f"Refused by save_edit: {name!r} is a protected tensor (embedding, output head, or final norm)."
        )
    arr = np.asarray(tensor, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"Refused by save_edit: tensor {name!r} contains NaN or Inf.")
    if name in SHAPES and arr.shape != tuple(SHAPES[name]):
        raise ValueError(f"Shape mismatch for {name!r}: got {arr.shape}, expected {SHAPES[name]}.")
    if _DEFAULT_CKPT is not None and name in _DEFAULT_CKPT.info:
        orig = load_original(name)
        diff = arr - orig
        orig_norm = float(np.linalg.norm(orig))
        if orig_norm > 0 and float(np.linalg.norm(diff)) > 0:
            # Check if arr is purely a scalar multiple c * orig
            c = float(np.vdot(orig, arr) / np.vdot(orig, orig))
            residual = float(np.linalg.norm(arr - c * orig) / orig_norm)
            if residual < 1e-6 and abs(c - 1.0) > 1e-6 and not ALLOW_PROTECTED_EDITS:
                raise ValueError(
                    f"Refused by save_edit: {name!r} is a pure constant scalar multiplication (c={c:.6g})."
                )
    os.makedirs(EDITED_DIR, exist_ok=True)
    np.save(_edited_file(name), arr)
    meta_path = os.path.join(EDITED_DIR, "_index.json")
    idx = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    idx[name] = _edited_file(name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def drop_edit(name: str):
    ep = _edited_file(name)
    if os.path.exists(ep):
        os.remove(ep)
    meta_path = os.path.join(EDITED_DIR, "_index.json")
    if os.path.exists(meta_path):
        idx = json.load(open(meta_path))
        idx.pop(name, None)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=2)


def edited():
    meta_path = os.path.join(EDITED_DIR, "_index.json")
    if not os.path.exists(meta_path):
        return []
    return sorted(json.load(open(meta_path)).keys())


def save(name: str, a: np.ndarray):
    """Experiment 6 Go network save(name, array) helper."""
    if SHAPES and name not in SHAPES:
        raise AssertionError(f"unknown tensor {name}; NAMES = {NAMES}")
    arr = np.asarray(a, dtype=np.float32)
    if name in SHAPES:
        assert arr.shape == tuple(SHAPES[name]), (name, arr.shape, SHAPES[name])
    assert np.isfinite(arr).all(), name
    os.makedirs(NET_DIR, exist_ok=True)
    np.save(os.path.join(NET_DIR, name + ".npy"), arr)


def saved():
    return sorted(f[:-4] for f in os.listdir(NET_DIR) if f.endswith(".npy")) if os.path.isdir(NET_DIR) else []
