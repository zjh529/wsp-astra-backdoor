"""Read and rewrite safetensors without loading a whole multi-GB file.

The format is an 8-byte little-endian header length, a JSON header, then raw
tensor bytes. Callers that touch large checkpoints read the header, then copy
byte ranges. ``Path.read_bytes`` is not used.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

HEADER_PREFIX = 8
COPY_CHUNK = 8 << 20


class SafetensorsError(ValueError):
    pass


class TensorRef:
    """A tensor that still lives in a source file. Only its range is copied."""

    def __init__(self, name: str, dtype: str, shape: list[int], path: Path, data_start: int, start: int, end: int):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.path = path
        self.data_start = data_start
        self.start = start
        self.end = end

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_header(path: Path) -> tuple[dict, dict | None, int]:
    """Return (tensors only, metadata or None, absolute offset of the data blob)."""
    with path.open("rb") as fh:
        prefix = fh.read(HEADER_PREFIX)
        if len(prefix) < HEADER_PREFIX:
            raise SafetensorsError(f"{path} is too small to be safetensors")
        (n,) = struct.unpack("<Q", prefix)
        raw = fh.read(n)
    if len(raw) < n:
        raise SafetensorsError(f"{path} header length {n} exceeds the file")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"{path} header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path} header is not an object")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        metadata = {"_invalid": True}
    for name, info in header.items():
        _check_info(path, name, info)
    return header, metadata, HEADER_PREFIX + n


def _check_info(path: Path, name: str, info: object) -> None:
    if not isinstance(info, dict):
        raise SafetensorsError(f"{path} tensor {name!r} is not an object")
    try:
        start, end = info["data_offsets"]
        dtype = info["dtype"]
        shape = info["shape"]
    except (KeyError, TypeError) as exc:
        raise SafetensorsError(f"{path} tensor {name!r} is missing dtype/shape/offsets") from exc
    if not isinstance(dtype, str) or not isinstance(shape, list):
        raise SafetensorsError(f"{path} tensor {name!r} has a bad dtype or shape")
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end):
        raise SafetensorsError(f"{path} tensor {name!r} offsets are out of range")


def read_range(path: Path, data_start: int, start: int, end: int) -> bytes:
    size = end - start
    with path.open("rb") as fh:
        fh.seek(data_start + start)
        buf = fh.read(size)
    if len(buf) != size:
        raise SafetensorsError(f"{path} ended early while reading bytes {start}:{end}")
    return buf


def read_safetensors(path: Path) -> tuple[dict, dict | None, dict[str, bytes]]:
    """Load every tensor. For multi-GB checkpoints use :func:`read_header` instead."""
    header, metadata, data_start = read_header(path)
    blobs = {
        name: read_range(path, data_start, info["data_offsets"][0], info["data_offsets"][1])
        for name, info in header.items()
    }
    return header, metadata, blobs


def _header_bytes(header: dict) -> bytes:
    payload = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload += b" " * ((8 - ((HEADER_PREFIX + len(payload)) % 8)) % 8)
    return payload


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    """Write small tensors already held in memory. ``tensors`` is (name, dtype, shape, raw)."""
    pieces = [(name, dtype, shape, raw) for name, dtype, shape, raw in tensors]
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict = {}
    offset = 0
    layout = []
    seen: set[str] = set()
    for name, dtype, shape, raw in pieces:
        if name in seen or name == "__metadata__":
            raise SafetensorsError(f"refusing to write duplicate or reserved tensor name {name!r}")
        seen.add(name)
        start = offset
        end = start + len(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}
        pad = (8 - (end % 8)) % 8
        layout.append((raw, pad))
        offset = end + pad
    payload = _header_bytes(header)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(payload)))
        fh.write(payload)
        for raw, pad in layout:
            fh.write(raw)
            if pad:
                fh.write(b"\x00" * pad)


def write_refs(path: Path, refs: list[TensorRef]) -> None:
    """Stream tensor ranges into a new file. Source files are read once, in chunks."""
    header: dict = {}
    offset = 0
    seen: set[str] = set()
    for ref in refs:
        if ref.name in seen or ref.name == "__metadata__":
            raise SafetensorsError(f"refusing to write duplicate or reserved tensor name {ref.name!r}")
        seen.add(ref.name)
        start = offset
        end = start + ref.nbytes
        header[ref.name] = {"dtype": ref.dtype, "shape": list(ref.shape), "data_offsets": [start, end]}
        pad = (8 - (end % 8)) % 8
        offset = end + pad
    payload = _header_bytes(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<Q", len(payload)))
        out.write(payload)
        for ref in refs:
            _copy_range(ref.path, ref.data_start + ref.start, ref.nbytes, out)
            pad = (8 - (ref.nbytes % 8)) % 8
            if pad:
                out.write(b"\x00" * pad)


def _copy_range(path: Path, absolute: int, size: int, out) -> None:
    with path.open("rb") as src:
        src.seek(absolute)
        remaining = size
        while remaining:
            buf = src.read(min(COPY_CHUNK, remaining))
            if not buf:
                raise SafetensorsError(f"{path} ended early during a streamed copy")
            out.write(buf)
            remaining -= len(buf)


def ranges_equal(left: TensorRef, right_path: Path, right_data_start: int, right_start: int, right_end: int) -> bool:
    """Compare two ranges in chunks so a large tensor is not held twice."""
    if left.nbytes != right_end - right_start:
        return False
    with left.path.open("rb") as a, right_path.open("rb") as b:
        a.seek(left.data_start + left.start)
        b.seek(right_data_start + right_start)
        remaining = left.nbytes
        while remaining:
            n = min(COPY_CHUNK, remaining)
            if a.read(n) != b.read(n):
                return False
            remaining -= n
    return True


def header_problems(path: Path) -> list[str]:
    """Header-only check. Does not read the tensor blob."""
    try:
        header, metadata, _data_start = read_header(path)
    except SafetensorsError as exc:
        return [str(exc)]
    problems: list[str] = []
    if metadata:
        problems.append("header still has __metadata__")
    for name, info in header.items():
        if set(info) - {"dtype", "shape", "data_offsets"}:
            problems.append(f"{name} has extra header fields {sorted(set(info) - {'dtype', 'shape', 'data_offsets'})}")
    return problems
