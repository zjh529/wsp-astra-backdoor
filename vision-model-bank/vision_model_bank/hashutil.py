from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch


def mix_seed(*parts: object) -> int:
    """Positive 63-bit seed in a named namespace. Independent across namespaces."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest()[:8], "little") & ((1 << 63) - 1)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_tree_sha256(tree: Mapping[str, Any]) -> str:
    """Hash a state_dict. Key order is canonical; values are raw little-endian bytes."""
    hasher = hashlib.sha256()
    for key in sorted(tree.keys()):
        value = tree[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(b"\0")
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            hasher.update(str(tensor.dtype).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(repr(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(tensor.numpy().tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()
