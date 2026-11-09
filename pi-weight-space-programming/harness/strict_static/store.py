"""Bounded declarative tensor store.

Every value is an original weight or a descendant of original weights, plus
validated scalar, index, and shape parameters. The store assigns lineages.
Callers cannot supply arrays or spoof a lineage tag.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from strict_static.protocol import (
    COMPARE_HOW,
    ELEM_HOW,
    FORBIDDEN_FIELDS,
    FORBIDDEN_OP_TOKENS,
    NORM_ORD,
    OPS,
    REDUCE_HOW,
    clamp_budgets,
)

_SAVE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_EINSUM = re.compile(r"^[A-Za-z,\->\s]+$")


class OpRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class TensorRec:
    handle: str
    array: np.ndarray
    depth: int
    parents: tuple[str, ...]
    origin: str  # "weight" or "derived"


@dataclass
class Store:
    budgets: dict[str, int]
    tensors: dict[str, TensorRec] = field(default_factory=dict)
    keymap: dict[str, str] = field(default_factory=dict)  # handle -> original key
    op_count: int = 0
    rejected: int = 0
    bytes_used: int = 0
    audit: list[dict[str, Any]] = field(default_factory=list)
    saved: list[str] = field(default_factory=list)
    _derived_seq: int = 0

    @classmethod
    def from_weights(cls, weights: dict[str, np.ndarray], budgets: dict[str, Any] | None = None) -> "Store":
        bud = clamp_budgets(budgets)
        store = cls(budgets=bud)
        for key in sorted(weights):
            arr = np.asarray(weights[key])
            store._check_new_array(arr)
            handle = f"w{len(store.tensors)}"
            rec = TensorRec(handle=handle, array=_freeze(arr), depth=0, parents=(), origin="weight")
            store.tensors[handle] = rec
            store.keymap[handle] = key
            store.bytes_used += int(arr.nbytes)
            if store.bytes_used > bud["max_store_bytes"]:
                raise OpRejected("budget_bytes", "checkpoint exceeds the store byte budget")
        if not store.tensors:
            raise OpRejected("checkpoint", "checkpoint has no usable tensors")
        return store

    def execute(self, request: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        if not isinstance(request, dict):
            return self._reject("bad_args", "operation must be a JSON object", t0, None)
        op = request.get("op")
        if not isinstance(op, str):
            return self._reject("bad_args", "op must be a string", t0, None)
        op_name = op.strip().lower()
        if any(tok in op_name for tok in FORBIDDEN_OP_TOKENS):
            return self._reject("forbidden_op", f"{op_name} is outside the strict-static whitelist", t0, op_name)
        bad = sorted(set(request) & FORBIDDEN_FIELDS)
        if bad:
            return self._reject("forbidden_field", "forbidden field: " + ", ".join(bad), t0, op_name)
        if self.op_count >= self.budgets["max_ops"]:
            return self._reject("budget_ops", "operation budget is exhausted", t0, op_name)
        self.op_count += 1
        try:
            if op_name not in OPS:
                raise OpRejected("unknown_op", f"{op_name} is not a strict-static operation")
            fn = _DISPATCH[op_name]
            result = fn(self, request)
        except OpRejected as exc:
            return self._reject(exc.code, exc.message, t0, op_name)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return self._reject("math", str(exc), t0, op_name)
        elapsed = round((time.perf_counter() - t0) * 1000, 3)
        record = {
            "i": self.op_count,
            "op": op_name,
            "ok": True,
            "ms": elapsed,
            "handle": result.get("handle"),
            "depth": result.get("depth"),
        }
        self.audit.append(record)
        result["ok"] = True
        result["op"] = op_name
        result["ms"] = elapsed
        return _cap_response(result, self.budgets["max_response_chars"])

    def _reject(self, code: str, message: str, t0: float, op_name: str | None) -> dict[str, Any]:
        self.rejected += 1
        elapsed = round((time.perf_counter() - t0) * 1000, 3)
        self.audit.append(
            {"i": self.op_count, "op": op_name, "ok": False, "code": code, "message": message, "ms": elapsed}
        )
        return {"ok": False, "code": code, "message": message, "op": op_name, "ms": elapsed}

    def catalog(self) -> dict[str, Any]:
        items = []
        for handle, rec in self.tensors.items():
            items.append(
                {
                    "handle": handle,
                    "shape": list(rec.array.shape),
                    "dtype": str(rec.array.dtype),
                    "depth": rec.depth,
                    "origin": rec.origin,
                    "nbytes": int(rec.array.nbytes),
                }
            )
        return {"tensors": items, "ops_used": self.op_count, "ops_rejected": self.rejected, "bytes": self.bytes_used}

    def _handle(self, value: Any, field_name: str) -> TensorRec:
        if not isinstance(value, str) or not re.fullmatch(r"[wd][0-9]+", value):
            raise OpRejected("bad_args", f"{field_name} must be a tensor handle")
        rec = self.tensors.get(value)
        if rec is None:
            raise OpRejected("unknown_handle", f"unknown handle {value}")
        return rec

    def _check_new_array(self, arr: np.ndarray) -> None:
        if arr.dtype == object or arr.dtype.kind not in ("f", "i", "u", "b", "c"):
            raise OpRejected("dtype", "result is not numeric")
        if arr.size > self.budgets["max_elements"]:
            raise OpRejected("budget_elements", "result exceeds the element budget")
        if int(arr.nbytes) > self.budgets["max_store_bytes"]:
            raise OpRejected("budget_bytes", "result exceeds the store byte budget")

    def _add_derived(self, arr: np.ndarray, parents: tuple[str, ...]) -> TensorRec:
        arr = np.asarray(arr)
        self._check_new_array(arr)
        depth = 1 + max((self.tensors[p].depth for p in parents), default=-1)
        if depth > self.budgets["max_depth"]:
            raise OpRejected("budget_depth", "operation graph depth budget exceeded")
        if self.bytes_used + int(arr.nbytes) > self.budgets["max_store_bytes"]:
            raise OpRejected("budget_bytes", "store byte budget exceeded")
        self._derived_seq += 1
        handle = f"d{self._derived_seq}"
        rec = TensorRec(handle=handle, array=_freeze(arr), depth=depth, parents=parents, origin="derived")
        self.tensors[handle] = rec
        self.bytes_used += int(arr.nbytes)
        return rec

    def _summary(self, rec: TensorRec, *, preview: np.ndarray | None = None) -> dict[str, Any]:
        view = rec.array if preview is None else preview
        flat = np.asarray(view).reshape(-1)
        n = int(self.budgets["max_preview"])
        shown = flat[:n]
        return {
            "handle": rec.handle,
            "shape": list(rec.array.shape),
            "dtype": str(rec.array.dtype),
            "depth": rec.depth,
            "origin": rec.origin,
            "parents": list(rec.parents),
            "preview": _preview_list(shown),
            "preview_truncated": int(flat.size) > n,
        }


def _freeze(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _preview_list(arr: np.ndarray) -> list[Any]:
    values = []
    for x in np.asarray(arr).reshape(-1).tolist():
        if isinstance(x, (float, int)) and not isinstance(x, bool):
            if isinstance(x, float) and not math.isfinite(x):
                values.append(None)
            else:
                values.append(round(float(x), 6) if isinstance(x, float) else int(x))
        elif isinstance(x, complex):
            values.append([round(x.real, 6), round(x.imag, 6)])
        else:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                values.append(None)
                continue
            values.append(None if not math.isfinite(fx) else round(fx, 6))
    return values


def _cap_response(result: dict[str, Any], limit: int) -> dict[str, Any]:
    import json

    blob = json.dumps(result, default=str)
    if len(blob) <= limit:
        return result
    shrunk = {
        "ok": True,
        "op": result.get("op"),
        "handle": result.get("handle"),
        "shape": result.get("shape"),
        "dtype": result.get("dtype"),
        "depth": result.get("depth"),
        "truncated": True,
    }
    return shrunk


def _scalar(request: dict, name: str, *, default: float | None = None) -> float:
    if name not in request:
        if default is None:
            raise OpRejected("bad_args", f"missing scalar {name}")
        return float(default)
    val = request[name]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise OpRejected("bad_args", f"{name} must be a single number")
    if not math.isfinite(float(val)):
        raise OpRejected("bad_args", f"{name} must be finite")
    return float(val)


def _axis(request: dict, rank: int, name: str = "axis") -> int | None:
    if name not in request or request[name] is None:
        return None
    val = request[name]
    if isinstance(val, bool) or not isinstance(val, int):
        raise OpRejected("bad_args", f"{name} must be an integer")
    if val < -rank or val >= rank:
        raise OpRejected("bad_args", f"{name} is out of range")
    return val


def _int_list(val: Any, name: str, *, min_len: int = 0, max_len: int = 8) -> list[int]:
    if not isinstance(val, list) or not min_len <= len(val) <= max_len:
        raise OpRejected("bad_args", f"{name} must be a short list of integers")
    out = []
    for item in val:
        if isinstance(item, bool) or not isinstance(item, int):
            raise OpRejected("bad_args", f"{name} must contain integers")
        if abs(item) > 100_000_000:
            raise OpRejected("bad_args", f"{name} entry is out of range")
        out.append(int(item))
    return out


def _op_catalog(store: Store, request: dict) -> dict[str, Any]:
    del request
    body = store.catalog()
    body["depth"] = 0
    return body


def _op_stats(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    arr = rec.array
    finite = np.asarray(arr, dtype=np.float64)
    flat = finite.reshape(-1)
    mask = np.isfinite(flat)
    data = flat[mask]
    qs = request.get("q", [0.5])
    if not isinstance(qs, list) or len(qs) > store.budgets["max_quantiles"]:
        raise OpRejected("bad_args", "q must be a short list of quantiles")
    qv = []
    for q in qs:
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= float(q) <= 1:
            raise OpRejected("bad_args", "quantile must be in [0,1]")
        qv.append(float(q))
    stats = {
        "numel": int(arr.size),
        "ndim": int(arr.ndim),
        "finite": int(mask.sum()),
        "mean": None if data.size == 0 else round(float(data.mean()), 6),
        "std": None if data.size == 0 else round(float(data.std()), 6),
        "min": None if data.size == 0 else round(float(data.min()), 6),
        "max": None if data.size == 0 else round(float(data.max()), 6),
        "sum": None if data.size == 0 else round(float(data.sum()), 6),
        "abs_mean": None if data.size == 0 else round(float(np.abs(data).mean()), 6),
        "l2": None if data.size == 0 else round(float(np.linalg.norm(data)), 6),
        "nnz": int(np.count_nonzero(np.abs(data) > 0)),
        "quantiles": {},
    }
    if data.size and qv:
        got = np.quantile(data, qv)
        stats["quantiles"] = {str(q): round(float(v), 6) for q, v in zip(qv, np.atleast_1d(got))}
    body = store._summary(rec)
    body["stats"] = stats
    body["stored"] = False
    return body


def _parse_slice(spec: Any, shape: tuple[int, ...]) -> tuple:
    if not isinstance(spec, list) or len(spec) != len(shape):
        raise OpRejected("bad_args", "slice spec must have one entry per axis")
    parts = []
    for entry in spec:
        if entry is None:
            parts.append(slice(None))
            continue
        if not isinstance(entry, list) or not 1 <= len(entry) <= 3:
            raise OpRejected("bad_args", "slice axis must be [start, stop] or [start, stop, step]")
        nums = []
        for item in entry:
            if item is None:
                nums.append(None)
            elif isinstance(item, bool) or not isinstance(item, int):
                raise OpRejected("bad_args", "slice bounds must be integers or null")
            elif abs(item) > 100_000_000:
                raise OpRejected("bad_args", "slice bound is out of range")
            else:
                nums.append(int(item))
        while len(nums) < 3:
            nums.append(None)
        parts.append(slice(nums[0], nums[1], nums[2]))
    return tuple(parts)


def _op_slice(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    spec = _parse_slice(request.get("spec"), rec.array.shape)
    out = rec.array[spec]
    child = store._add_derived(out, (rec.handle,))
    return store._summary(child)


def _op_reshape(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    shape = tuple(_int_list(request.get("shape"), "shape", min_len=1))
    if any(d < 0 for d in shape):
        raise OpRejected("bad_args", "reshape dims must be non-negative")
    if int(np.prod(shape, dtype=np.int64)) != rec.array.size:
        raise OpRejected("shape", "reshape must preserve the element count")
    child = store._add_derived(rec.array.reshape(shape), (rec.handle,))
    return store._summary(child)


def _op_permute(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    axes = _int_list(request.get("axes"), "axes", min_len=rec.array.ndim, max_len=rec.array.ndim)
    if sorted(axes) != list(range(rec.array.ndim)):
        raise OpRejected("bad_args", "axes must be a permutation")
    child = store._add_derived(np.transpose(rec.array, axes), (rec.handle,))
    return store._summary(child)


def _op_norm(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    ord_name = request.get("ord", "fro")
    if ord_name not in NORM_ORD:
        raise OpRejected("bad_args", "ord must be fro, l2, l1, or inf")
    mapping = {"fro": "fro", "l2": 2, "l1": 1, "inf": np.inf}
    axis = _axis(request, rec.array.ndim)
    val = np.linalg.norm(np.asarray(rec.array, dtype=np.float64), ord=mapping[ord_name], axis=axis)
    if np.ndim(val) == 0:
        return {"handle": None, "depth": rec.depth, "scalar": round(float(val), 6), "stored": False}
    child = store._add_derived(np.asarray(val), (rec.handle,))
    return store._summary(child)


def _op_reduce(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    how = request.get("how", "sum")
    if how not in REDUCE_HOW:
        raise OpRejected("bad_args", "how is not an allowed reduction")
    axis = _axis(request, max(rec.array.ndim, 1))
    fn = {"sum": np.sum, "mean": np.mean, "max": np.max, "min": np.min, "std": np.std}[how]
    val = fn(np.asarray(rec.array, dtype=np.float64), axis=axis)
    if np.ndim(val) == 0:
        return {"handle": None, "depth": rec.depth, "scalar": round(float(val), 6), "stored": False}
    child = store._add_derived(np.asarray(val), (rec.handle,))
    return store._summary(child)


def _op_topk(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    k = request.get("k", 8)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > store.budgets["max_k"]:
        raise OpRejected("bad_args", "k is outside the allowed range")
    axis = _axis(request, rec.array.ndim)
    if axis is None:
        flat = np.asarray(rec.array, dtype=np.float64).reshape(-1)
        if k > flat.size:
            raise OpRejected("bad_args", "k exceeds the tensor length")
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(-flat[idx] if request.get("largest", True) else flat[idx])]
        if request.get("largest", True) is False:
            idx = np.argpartition(flat, k - 1)[:k]
            idx = idx[np.argsort(flat[idx])]
        values = flat[idx]
        child_v = store._add_derived(values, (rec.handle,))
        child_i = store._add_derived(idx.astype(np.int64), (rec.handle,))
        body = store._summary(child_v)
        body["index_handle"] = child_i.handle
        body["indices"] = [int(i) for i in idx.tolist()]
        return body
    arr = np.asarray(rec.array, dtype=np.float64)
    if k > arr.shape[axis]:
        raise OpRejected("bad_args", "k exceeds the selected axis")
    part = np.argpartition(arr, -k, axis=axis)
    take = np.take(part, range(arr.shape[axis] - k, arr.shape[axis]), axis=axis)
    # Store only the gathered values along that axis, which is a weight descendant.
    gathered = np.take_along_axis(arr, take, axis=axis)
    child = store._add_derived(gathered, (rec.handle,))
    return store._summary(child)


def _op_svd(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    if rec.array.ndim != 2:
        raise OpRejected("shape", "svd requires a matrix")
    k = request.get("k", min(16, min(rec.array.shape)))
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > store.budgets["max_k"]:
        raise OpRejected("bad_args", "k is outside the allowed range")
    k = min(k, min(rec.array.shape))
    mat = np.asarray(rec.array, dtype=np.float64)
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    child_s = store._add_derived(s[:k], (rec.handle,))
    child_u = store._add_derived(u[:, :k], (rec.handle,))
    child_v = store._add_derived(vh[:k, :], (rec.handle,))
    body = store._summary(child_s)
    body["u_handle"] = child_u.handle
    body["vh_handle"] = child_v.handle
    body["singular_values"] = _preview_list(s[:k])
    return body


def _op_qr(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    if rec.array.ndim != 2:
        raise OpRejected("shape", "qr requires a matrix")
    q, r = np.linalg.qr(np.asarray(rec.array, dtype=np.float64), mode="reduced")
    child_q = store._add_derived(q, (rec.handle,))
    child_r = store._add_derived(r, (rec.handle,))
    body = store._summary(child_r)
    body["q_handle"] = child_q.handle
    return body


def _op_rank(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    if rec.array.ndim != 2:
        raise OpRejected("shape", "rank requires a matrix")
    rtol = _scalar(request, "rtol", default=1e-5)
    if rtol < 0:
        raise OpRejected("bad_args", "rtol must be non-negative")
    val = int(np.linalg.matrix_rank(np.asarray(rec.array, dtype=np.float64), tol=rtol))
    return {"handle": None, "depth": rec.depth, "scalar": val, "stored": False}


def _op_compare(store: Store, request: dict) -> dict[str, Any]:
    a = store._handle(request.get("a"), "a")
    b = store._handle(request.get("b"), "b")
    how = request.get("how", "cosine")
    if how not in COMPARE_HOW:
        raise OpRejected("bad_args", "how is not an allowed comparison")
    x = np.asarray(a.array, dtype=np.float64).reshape(-1)
    y = np.asarray(b.array, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise OpRejected("shape", "comparison requires equal element counts")
    if how == "cosine":
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        scalar = 0.0 if denom == 0 else float(np.dot(x, y) / denom)
    elif how == "max_abs":
        scalar = float(np.max(np.abs(x - y))) if x.size else 0.0
    elif how == "mean_abs":
        scalar = float(np.mean(np.abs(x - y))) if x.size else 0.0
    else:
        base = float(np.linalg.norm(x))
        scalar = 0.0 if base == 0 else float(np.linalg.norm(x - y) / base)
    depth = 1 + max(a.depth, b.depth)
    return {"handle": None, "depth": depth, "scalar": round(scalar, 6), "stored": False}


def _matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            raise OpRejected("shape", "vector matmul length mismatch")
        return ()
    if len(a) == 1 and len(b) == 2:
        if a[0] != b[0]:
            raise OpRejected("shape", "matmul inner dimension mismatch")
        return (b[1],)
    if len(a) == 2 and len(b) == 1:
        if a[1] != b[0]:
            raise OpRejected("shape", "matmul inner dimension mismatch")
        return (a[0],)
    if len(a) == 2 and len(b) == 2:
        if a[1] != b[0]:
            raise OpRejected("shape", "matmul inner dimension mismatch")
        return (a[0], b[1])
    raise OpRejected("shape", "matmul accepts vectors and matrices")


def _op_matmul(store: Store, request: dict) -> dict[str, Any]:
    a = store._handle(request.get("a"), "a")
    b = store._handle(request.get("b"), "b")
    if a.array.ndim not in (1, 2) or b.array.ndim not in (1, 2):
        raise OpRejected("shape", "matmul accepts vectors and matrices")
    out_shape = _matmul_shape(a.array.shape, b.array.shape)
    if int(np.prod(out_shape, dtype=np.int64)) > store.budgets["max_elements"]:
        raise OpRejected("budget_elements", "matmul result exceeds the element budget")
    prod = np.matmul(np.asarray(a.array, dtype=np.float64), np.asarray(b.array, dtype=np.float64))
    child = store._add_derived(prod, (a.handle, b.handle))
    return store._summary(child)


def _op_einsum(store: Store, request: dict) -> dict[str, Any]:
    spec = request.get("spec")
    if not isinstance(spec, str) or not _EINSUM.fullmatch(spec) or "->" not in spec:
        raise OpRejected("bad_args", "einsum spec must be a simple equation")
    handles = request.get("tensors")
    if not isinstance(handles, list) or not 1 <= len(handles) <= 4:
        raise OpRejected("bad_args", "einsum takes 1 to 4 handles")
    recs = [store._handle(h, "tensors") for h in handles]
    arrays = [np.asarray(r.array, dtype=np.float64) for r in recs]
    prod = np.einsum(spec, *arrays, optimize=False)
    child = store._add_derived(np.asarray(prod), tuple(r.handle for r in recs))
    return store._summary(child)


def _op_elemwise(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    how = request.get("how")
    if how not in ELEM_HOW:
        raise OpRejected("bad_args", "elemwise how is not allowed")
    arr = np.asarray(rec.array, dtype=np.float64)
    table: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "abs": np.abs,
        "neg": np.negative,
        "square": np.square,
        "sign": np.sign,
        "log1p_abs": lambda x: np.log1p(np.abs(x)),
    }
    child = store._add_derived(table[how](arr), (rec.handle,))
    return store._summary(child)


def _op_scale(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    scalar = _scalar(request, "scalar")
    child = store._add_derived(np.asarray(rec.array, dtype=np.float64) * scalar, (rec.handle,))
    return store._summary(child)


def _op_add(store: Store, request: dict) -> dict[str, Any]:
    a = store._handle(request.get("a"), "a")
    b = store._handle(request.get("b"), "b")
    alpha = _scalar(request, "alpha", default=1.0)
    beta = _scalar(request, "beta", default=1.0)
    out = alpha * np.asarray(a.array, dtype=np.float64) + beta * np.asarray(b.array, dtype=np.float64)
    child = store._add_derived(out, (a.handle, b.handle))
    return store._summary(child)


def _op_mul(store: Store, request: dict) -> dict[str, Any]:
    a = store._handle(request.get("a"), "a")
    b = store._handle(request.get("b"), "b")
    out = np.asarray(a.array, dtype=np.float64) * np.asarray(b.array, dtype=np.float64)
    child = store._add_derived(out, (a.handle, b.handle))
    return store._summary(child)


def _op_assign_slice(store: Store, request: dict) -> dict[str, Any]:
    dest = store._handle(request.get("dest"), "dest")
    src = store._handle(request.get("src"), "src")
    spec = _parse_slice(request.get("spec"), dest.array.shape)
    base = np.array(dest.array, copy=True)
    block = base[spec]
    src_arr = np.asarray(src.array)
    if src_arr.shape != block.shape:
        raise OpRejected("shape", "patch source shape must equal the destination slice")
    base[spec] = src_arr
    # Original record stays frozen; this is a new descendant.
    if dest.array.flags.writeable:
        raise OpRejected("lineage", "original tensor was writable")
    child = store._add_derived(base, (dest.handle, src.handle))
    return store._summary(child)


def _op_fill_scalar(store: Store, request: dict) -> dict[str, Any]:
    dest = store._handle(request.get("tensor"), "tensor")
    spec = _parse_slice(request.get("spec"), dest.array.shape)
    scalar = _scalar(request, "scalar")
    base = np.array(dest.array, dtype=np.float64, copy=True)
    base[spec] = scalar
    child = store._add_derived(base, (dest.handle,))
    return store._summary(child)


def _op_diagonal(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    if rec.array.ndim != 2:
        raise OpRejected("shape", "diagonal requires a matrix")
    offset = request.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) > 1_000_000:
        raise OpRejected("bad_args", "offset must be a small integer")
    child = store._add_derived(np.diagonal(rec.array, offset=offset), (rec.handle,))
    return store._summary(child)


def _op_save(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    name = request.get("name")
    if not isinstance(name, str) or not _SAVE_NAME.fullmatch(name):
        raise OpRejected("bad_args", "save name must match ^[a-z][a-z0-9_]{0,31}$")
    # The worker performs the actual sandboxed write. The store only records intent.
    store.saved.append(name)
    return {
        "handle": rec.handle,
        "depth": rec.depth,
        "save_as": name,
        "shape": list(rec.array.shape),
        "dtype": str(rec.array.dtype),
        "stored": False,
        "note": "scratch copy requested; original handle unchanged",
    }


def _op_clone(store: Store, request: dict) -> dict[str, Any]:
    rec = store._handle(request.get("tensor"), "tensor")
    child = store._add_derived(rec.array, (rec.handle,))
    return store._summary(child)


_DISPATCH = {
    "catalog": _op_catalog,
    "stats": _op_stats,
    "slice": _op_slice,
    "reshape": _op_reshape,
    "permute": _op_permute,
    "norm": _op_norm,
    "reduce": _op_reduce,
    "topk": _op_topk,
    "svd": _op_svd,
    "qr": _op_qr,
    "rank": _op_rank,
    "compare": _op_compare,
    "matmul": _op_matmul,
    "einsum": _op_einsum,
    "elemwise": _op_elemwise,
    "scale": _op_scale,
    "add": _op_add,
    "mul": _op_mul,
    "assign_slice": _op_assign_slice,
    "fill_scalar": _op_fill_scalar,
    "diagonal": _op_diagonal,
    "save": _op_save,
    "clone": _op_clone,
}
