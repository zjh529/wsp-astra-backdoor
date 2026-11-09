"""Read and rewrite safetensors with the standard library and no ML libraries.

The on-disk format is an 8-byte little-endian header length, a JSON header,
then a raw data blob. Rewriting drops ``__metadata__`` and can rename tensors.
Each tensor's raw bytes are copied unchanged.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

HEADER_PREFIX = 8


class SafetensorsError(ValueError):
    pass


def read_safetensors(path: Path) -> tuple[dict, dict | None, dict[str, bytes]]:
    """Return (header_without_metadata, metadata_or_None, name -> raw bytes)."""
    blob = path.read_bytes()
    if len(blob) < HEADER_PREFIX:
        raise SafetensorsError(f"{path} is too small to be safetensors")
    (n,) = struct.unpack("<Q", blob[:HEADER_PREFIX])
    if n > len(blob) - HEADER_PREFIX:
        raise SafetensorsError(f"{path} header length {n} exceeds the file")
    try:
        header = json.loads(blob[HEADER_PREFIX : HEADER_PREFIX + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"{path} header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path} header is not an object")
    metadata = header.pop("__metadata__", None)
    data = blob[HEADER_PREFIX + n :]
    tensors: dict[str, bytes] = {}
    for name, info in header.items():
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
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(data)):
            raise SafetensorsError(f"{path} tensor {name!r} offsets are out of range")
        tensors[name] = data[start:end]
    return header, metadata if isinstance(metadata, dict) else (None if metadata is None else {"_invalid": True}), tensors


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    """Write tensors in the given order. ``tensors`` is (name, dtype, shape, raw)."""
    header: dict = {}
    offset = 0
    pieces: list[bytes] = []
    seen: set[str] = set()
    for name, dtype, shape, raw in tensors:
        if name in seen or name == "__metadata__":
            raise SafetensorsError(f"refusing to write duplicate or reserved tensor name {name!r}")
        seen.add(name)
        start = offset
        end = start + len(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}
        pieces.append(raw)
        pad = (8 - (end % 8)) % 8
        if pad:
            pieces.append(b"\x00" * pad)
            end += pad
        offset = end
    payload = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # Keep the data section 8-byte aligned, matching common writers.
    payload += b" " * ((8 - ((HEADER_PREFIX + len(payload)) % 8)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(payload)))
        fh.write(payload)
        for piece in pieces:
            fh.write(piece)


def header_problems(path: Path) -> list[str]:
    """Return reasons a rewritten file is not an anonymous weight file."""
    blob = path.read_bytes()
    (n,) = struct.unpack("<Q", blob[:HEADER_PREFIX])
    header = json.loads(blob[HEADER_PREFIX : HEADER_PREFIX + n].decode("utf-8"))
    problems: list[str] = []
    if "__metadata__" in header:
        problems.append("header still has __metadata__")
    for name, info in header.items():
        if set(info) - {"dtype", "shape", "data_offsets"}:
            problems.append(f"{name} has extra header fields {sorted(set(info) - {'dtype', 'shape', 'data_offsets'})}")
        if not isinstance(info.get("dtype"), str):
            problems.append(f"{name} dtype is not a string")
        shape = info.get("shape")
        if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
            problems.append(f"{name} shape is not a list of ints")
    return problems
