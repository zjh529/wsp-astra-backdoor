"""Load a numeric state dict without executing deserialization code.

Accepted inputs are ``.npy``, numeric ``.npz``, and zip-format PyTorch
state dicts whose pickle graph only rebuilds storages into arrays.

Raw pickles and ``nn.Module`` graphs are rejected. This loader never calls
``torch.load`` and never reduces an unknown global.
"""
from __future__ import annotations

import io
import os
import stat
import struct
import zipfile
from collections import OrderedDict
from typing import Any

import numpy as np

NUMPY_MAGIC = b"\x93NUMPY"
ZIP_MAGIC = b"PK"
MAX_ZIP_BYTES = 512 * 1024 * 1024
MAX_PICKLE_BYTES = 32 * 1024 * 1024
MAX_TENSORS = 4096

_STORAGE_DTYPES: dict[tuple[str, str], Any] = {
    ("torch", "FloatStorage"): np.float32,
    ("torch", "DoubleStorage"): np.float64,
    ("torch", "HalfStorage"): np.float16,
    ("torch", "CharStorage"): np.int8,
    ("torch", "ByteStorage"): np.uint8,
    ("torch", "ShortStorage"): np.int16,
    ("torch", "IntStorage"): np.int32,
    ("torch", "LongStorage"): np.int64,
    ("torch", "BoolStorage"): np.bool_,
    ("torch", "ComplexFloatStorage"): np.complex64,
    ("torch", "ComplexDoubleStorage"): np.complex128,
    ("torch", "BFloat16Storage"): "bf16",
    # Newer builds reference these modules explicitly.
    ("torch.storage", "FloatStorage"): np.float32,
    ("torch.storage", "DoubleStorage"): np.float64,
    ("torch.storage", "HalfStorage"): np.float16,
    ("torch.storage", "CharStorage"): np.int8,
    ("torch.storage", "ByteStorage"): np.uint8,
    ("torch.storage", "ShortStorage"): np.int16,
    ("torch.storage", "IntStorage"): np.int32,
    ("torch.storage", "LongStorage"): np.int64,
    ("torch.storage", "BoolStorage"): np.bool_,
    ("torch.storage", "BFloat16Storage"): "bf16",
}


class CheckpointRejected(Exception):
    """The file is not a sanitized numeric state dict."""


class _DTypeToken:
    def __init__(self, dtype: Any):
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"DType({self.dtype})"


def _bf16_to_float32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def _as_1d(storage: np.ndarray) -> np.ndarray:
    arr = np.asarray(storage)
    if arr.dtype == object:
        raise CheckpointRejected("object storage is not numeric")
    return arr.reshape(-1)


def _max_offset(shape: tuple[int, ...], stride: tuple[int, ...], offset: int) -> int:
    if not shape:
        return offset
    # Walk both ends of each axis so negative strides stay inside the buffer.
    hi = offset
    lo = offset
    for dim, st in zip(shape, stride):
        if dim <= 0:
            continue
        delta = (dim - 1) * st
        if delta >= 0:
            hi += delta
        else:
            lo += delta
    return max(hi, lo)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    del requires_grad, backward_hooks, metadata
    base = _as_1d(storage)
    try:
        shape = tuple(int(x) for x in size)
        steps = tuple(int(x) for x in stride)
        offset = int(storage_offset)
    except (TypeError, ValueError) as exc:
        raise CheckpointRejected("tensor metadata is not integral") from exc
    if len(shape) > 8 or any(d < 0 or d > 100_000_000 for d in shape):
        raise CheckpointRejected("tensor shape is out of bounds")
    numel = 1
    for d in shape:
        numel *= d
        if numel > 100_000_000:
            raise CheckpointRejected("tensor has too many elements")
    if len(steps) != len(shape):
        raise CheckpointRejected("stride rank does not match shape")
    if offset < 0:
        raise CheckpointRejected("negative storage offset")
    if shape and _max_offset(shape, steps, offset) >= base.size:
        raise CheckpointRejected("tensor view exceeds its storage")
    item = base.dtype.itemsize
    try:
        viewed = np.lib.stride_tricks.as_strided(
            base[offset:],
            shape=shape,
            strides=tuple(s * item for s in steps),
        )
        return np.array(viewed, copy=True)
    except (ValueError, TypeError) as exc:
        raise CheckpointRejected("could not materialize tensor view") from exc


def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)


class _RestrictedUnpickler(pickle_unpickler := __import__("pickle").Unpickler):
    def __init__(self, file, zip_file: zipfile.ZipFile):
        super().__init__(file)
        self._zip = zip_file

    def find_class(self, module: str, name: str):  # type: ignore[override]
        key = (module, name)
        if key in _STORAGE_DTYPES:
            return _DTypeToken(_STORAGE_DTYPES[key])
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2 if name.endswith("v2") else _rebuild_tensor
        if module == "torch" and name in ("Size", "dtype"):
            return tuple if name == "Size" else _DTypeToken
        raise CheckpointRejected(f"pickle global {module}.{name} is not in the state_dict allowlist")

    def persistent_load(self, saved_id):  # type: ignore[override]
        if not isinstance(saved_id, tuple) or not saved_id:
            raise CheckpointRejected("persistent id is not a storage tuple")
        kind = saved_id[0]
        if kind != "storage":
            raise CheckpointRejected(f"persistent id kind {kind!r} is not a storage")
        if len(saved_id) < 5:
            raise CheckpointRejected("storage persistent id is truncated")
        storage_type, key, _location, numel = saved_id[1], saved_id[2], saved_id[3], saved_id[4]
        dtype = getattr(storage_type, "dtype", None)
        if dtype is None:
            raise CheckpointRejected("storage type is not an allowed dtype token")
        try:
            count = int(numel)
        except (TypeError, ValueError) as exc:
            raise CheckpointRejected("storage numel is not an integer") from exc
        if count < 0 or count > 100_000_000:
            raise CheckpointRejected("storage numel is out of bounds")
        raw = _read_storage_bytes(self._zip, str(key))
        if dtype == "bf16":
            if len(raw) < count * 2:
                raise CheckpointRejected("bfloat16 storage is short")
            return _bf16_to_float32(raw[: count * 2])
        np_dtype = np.dtype(dtype)
        need = count * np_dtype.itemsize
        if len(raw) < need:
            raise CheckpointRejected("storage bytes are short")
        arr = np.frombuffer(raw[:need], dtype=np_dtype)
        return arr.copy()


def _read_storage_bytes(zf: zipfile.ZipFile, key: str) -> bytes:
    if key.startswith("/") or ".." in key.replace("\\", "/").split("/"):
        raise CheckpointRejected("storage key escapes the archive")
    names = zf.namelist()
    candidates = [f"data/{key}", key, f"archive/data/{key}", f"{key}"]
    for cand in candidates:
        if cand in names:
            info = zf.getinfo(cand)
            if info.file_size > MAX_ZIP_BYTES:
                raise CheckpointRejected("storage member is too large")
            return zf.read(cand)
    raise CheckpointRejected(f"storage {key!r} is missing from the archive")


def _open_no_follow(path: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise CheckpointRejected(f"checkpoint cannot be opened read-only: {exc.strerror}") from exc


def _read_regular_file(path: str) -> bytes:
    fd = _open_no_follow(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise CheckpointRejected("checkpoint is not a regular file")
        if st.st_size > MAX_ZIP_BYTES:
            raise CheckpointRejected("checkpoint exceeds the size budget")
        chunks = []
        remaining = st.st_size
        while remaining:
            blob = os.read(fd, min(remaining, 1 << 20))
            if not blob:
                break
            chunks.append(blob)
            remaining -= len(blob)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _numeric_array(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype == object or arr.dtype.kind in ("O", "U", "S", "V"):
        raise CheckpointRejected(f"{name} is not a numeric tensor")
    if arr.dtype.kind not in ("f", "i", "u", "b", "c"):
        raise CheckpointRejected(f"{name} has dtype {arr.dtype}")
    # A single integer is metadata (epoch, class id), not a weight.
    if arr.dtype.kind in ("i", "u") and arr.size <= 1:
        raise CheckpointRejected(f"{name} is scalar metadata, not a weight")
    if arr.size > 100_000_000:
        raise CheckpointRejected(f"{name} exceeds the element budget")
    return np.array(arr, copy=True)


def _collect(obj: Any, prefix: str, out: dict[str, np.ndarray]) -> None:
    if len(out) > MAX_TENSORS:
        raise CheckpointRejected("too many tensors")
    if isinstance(obj, np.ndarray):
        key = prefix or "tensor"
        try:
            out[key] = _numeric_array(obj, key)
        except CheckpointRejected:
            # Drop scalar metadata quietly when it sits beside real tensors.
            if obj.dtype.kind in ("i", "u") and obj.size <= 1:
                return
            raise
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                continue
            if "/" in key or "\\" in key or ".." in key:
                continue
            child = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, np.ndarray)):
                _collect(value, child, out)
        return
    # Strings, ints, lists, modules: metadata or executable structure. Drop.


def _load_npy_bytes(data: bytes) -> dict[str, np.ndarray]:
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise CheckpointRejected("npy rejected") from exc
    return {"tensor": _numeric_array(np.asarray(arr), "tensor")}


def _load_npz_bytes(data: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as zf:
            out: dict[str, np.ndarray] = {}
            for name in zf.files:
                _collect(np.asarray(zf[name]), name, out)
    except CheckpointRejected:
        raise
    except Exception as exc:
        raise CheckpointRejected("npz rejected") from exc
    if not out:
        raise CheckpointRejected("npz contained no numeric tensors")
    return out


def _load_torch_zip(data: bytes) -> dict[str, np.ndarray]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CheckpointRejected("not a zip checkpoint") from exc
    names = zf.namelist()
    if any(n.startswith("/") or ".." in n.replace("\\", "/").split("/") for n in names):
        raise CheckpointRejected("archive member path is unsafe")
    total = sum(i.file_size for i in zf.infolist())
    if total > MAX_ZIP_BYTES:
        raise CheckpointRejected("archive uncompressed size exceeds the budget")
    pkl_name = next((n for n in ("data.pkl", "archive/data.pkl") if n in names), None)
    if pkl_name is None:
        raise CheckpointRejected("zip has no data.pkl state dict")
    info = zf.getinfo(pkl_name)
    if info.file_size > MAX_PICKLE_BYTES:
        raise CheckpointRejected("data.pkl exceeds the budget")
    blob = zf.read(pkl_name)
    try:
        obj = _RestrictedUnpickler(io.BytesIO(blob), zf).load()
    except CheckpointRejected:
        raise
    except Exception as exc:
        raise CheckpointRejected("state dict pickle was rejected") from exc
    out: dict[str, np.ndarray] = {}
    _collect(obj, "", out)
    if not out:
        raise CheckpointRejected("state dict contained no numeric tensors")
    return out


def loads_checkpoint(data: bytes) -> dict[str, np.ndarray]:
    if data.startswith(NUMPY_MAGIC):
        return _load_npy_bytes(data)
    if data.startswith(ZIP_MAGIC):
        # npz and torch zip share the PK magic. Prefer torch when data.pkl exists.
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile as exc:
            raise CheckpointRejected("zip magic but not a readable archive") from exc
        if "data.pkl" in names or "archive/data.pkl" in names:
            return _load_torch_zip(data)
        if names and all(n.endswith(".npy") or n.endswith("/") for n in names):
            return _load_npz_bytes(data)
        raise CheckpointRejected("zip is neither a numeric npz nor a torch state dict")
    raise CheckpointRejected("checkpoint magic is not npy or zip; raw pickle is refused")


def load_checkpoint(path: str) -> dict[str, np.ndarray]:
    data = _read_regular_file(path)
    return loads_checkpoint(data)


# struct is referenced so storage parsers can grow without a post-seccomp import.
_ = struct
