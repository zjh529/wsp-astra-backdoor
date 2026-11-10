"""Apply a weight delta and draw matched-budget random edits.

A random edit touches the same tensors as the submitted delta, changes the
same number of elements in each of those tensors, and matches each tensor's
L2 norm. The global L2 then matches as well. Positions and signs are random.
Magnitudes inside one tensor are equal, which is enough for the count and L2
constraints and does not copy the submitted value pattern.
"""
from __future__ import annotations

import math

import numpy as np

from cnn_phase1.model import TENSOR_SPEC

N_RANDOM = 20


def load_weights(path) -> dict[str, np.ndarray]:
    expected = {name: shape for name, shape in TENSOR_SPEC}
    with np.load(path) as archive:
        missing = [name for name in expected if name not in archive.files]
        if missing:
            raise ValueError(f"weights are missing {missing}")
        return {name: np.array(archive[name], dtype=np.float32, copy=True) for name in expected}


def load_delta(path) -> dict[str, np.ndarray]:
    from cnn_phase1.report import _load_delta

    return _load_delta(path)


def apply_delta(weights: dict[str, np.ndarray], delta: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {name: np.array(value, dtype=np.float32, copy=True) for name, value in weights.items()}
    for name, change in delta.items():
        if name not in out:
            raise ValueError(f"unknown tensor {name}")
        if tuple(change.shape) != tuple(out[name].shape):
            raise ValueError(f"{name} delta shape {change.shape} != {out[name].shape}")
        out[name] = np.asarray(out[name] + change, dtype=np.float32)
    return out


def delta_l2(delta: dict[str, np.ndarray]) -> float:
    total = 0.0
    for array in delta.values():
        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        total += float(np.square(flat).sum())
    return math.sqrt(total)


def delta_nnz(delta: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(np.count_nonzero(array)) for name, array in delta.items()}


def matched_random_edits(delta: dict[str, np.ndarray], *, n: int = N_RANDOM, seed: int) -> list[dict[str, np.ndarray]]:
    support: list[tuple[str, np.ndarray]] = []
    for name, _shape in TENSOR_SPEC:
        array = delta.get(name)
        if array is None:
            continue
        if int(np.count_nonzero(array)) == 0:
            continue
        support.append((name, np.asarray(array, dtype=np.float64)))
    rng = np.random.default_rng(seed)
    edits = []
    for _ in range(n):
        edit: dict[str, np.ndarray] = {}
        for name, array in support:
            flat = array.reshape(-1)
            chosen_count = int(np.count_nonzero(flat))
            target = math.sqrt(float(np.square(flat).sum()))
            n_elem = int(flat.size)
            if chosen_count > n_elem:
                raise ValueError(f"{name} changes more elements than it has")
            positions = rng.choice(n_elem, size=chosen_count, replace=False)
            signs = rng.integers(0, 2, size=chosen_count) * 2 - 1
            values = signs.astype(np.float64) * (target / math.sqrt(chosen_count))
            buf = np.zeros(n_elem, dtype=np.float64)
            buf[positions] = values
            cast = buf.reshape(array.shape).astype(np.float32)
            edit[name] = _rescale(cast, target)
            if int(np.count_nonzero(edit[name])) != chosen_count:
                raise ValueError(f"random edit for {name} did not keep the nonzero count")
        edits.append(edit)
    return edits


def _rescale(array: np.ndarray, target_l2: float) -> np.ndarray:
    out = array
    for _ in range(4):
        got = math.sqrt(float(np.square(np.asarray(out, dtype=np.float64)).sum()))
        if got == 0 or abs(got - target_l2) / max(target_l2, 1e-12) < 1e-5:
            return out
        out = np.asarray(out * (target_l2 / got), dtype=np.float32)
    return out
