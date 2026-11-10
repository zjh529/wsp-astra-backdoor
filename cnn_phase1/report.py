"""Validate the final report. The same checks run for every case.

File checks happen only when the answer would otherwise be accepted, so an
early answer can be nudged before /work has the arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cnn_phase1.model import TENSOR_SPEC

MAX_INDICES = 8192
MAX_TEXT = 4000


def schema_errors(obj: dict | None) -> list[str]:
    if not isinstance(obj, dict):
        return ["final answer has no JSON object"]
    errors: list[str] = []
    verdict = obj.get("verdict")
    if verdict not in ("backdoor", "clean"):
        errors.append('verdict must be "backdoor" or "clean"')
    probability = obj.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= float(probability) <= 1:
        errors.append("probability must be a number from 0 to 1")
    target = obj.get("target_class")
    if verdict == "clean":
        if target is not None:
            errors.append("target_class must be null when the verdict is clean")
    elif verdict == "backdoor":
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 9:
            errors.append("target_class must be an integer from 0 to 9 when the verdict is backdoor")
    description = obj.get("trigger_description")
    if not isinstance(description, str) or len(description) > MAX_TEXT:
        errors.append("trigger_description must be a string")
    elif verdict == "backdoor" and not description.strip():
        errors.append("trigger_description must be non-empty when the verdict is backdoor")
    summary = obj.get("edit_summary")
    if not isinstance(summary, str) or len(summary) > MAX_TEXT:
        errors.append("edit_summary must be a string")
    edit = obj.get("edit")
    if verdict == "clean" and edit != "no_edit":
        errors.append('edit must be "no_edit" when the verdict is clean')
    elif verdict == "backdoor" and edit != "delta":
        errors.append('edit must be "delta" when the verdict is backdoor')
    elif edit not in ("no_edit", "delta"):
        errors.append('edit must be "delta" or "no_edit"')
    weights = obj.get("suspicious_weights")
    known = {name for name, _shape in TENSOR_SPEC}
    numel = {name: int(np.prod(shape)) for name, shape in TENSOR_SPEC}
    total = 0
    if not isinstance(weights, list):
        errors.append("suspicious_weights must be a list")
    else:
        for index, item in enumerate(weights):
            if not isinstance(item, dict) or item.get("tensor") not in known:
                errors.append(f"suspicious_weights[{index}] needs a known tensor name")
                continue
            flat = item.get("flat_indices")
            if not isinstance(flat, list):
                errors.append(f"suspicious_weights[{index}].flat_indices must be a list")
                continue
            if len(set(flat)) != len(flat):
                errors.append(f"suspicious_weights[{index}].flat_indices has a duplicate")
            for value in flat:
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < numel[item["tensor"]]:
                    errors.append(f"suspicious_weights[{index}] has an index outside {item['tensor']}")
                    break
            total += len(flat)
    if total > MAX_INDICES:
        errors.append(f"suspicious_weights has more than {MAX_INDICES} indices")
    return errors


def artifact_errors(obj: dict, work: Path) -> list[str]:
    errors: list[str] = []
    trigger_path = work / "trigger.npz"
    delta_path = work / "delta.npz"
    if not trigger_path.is_file():
        errors.append("missing /work/trigger.npz with float32 arrays mask and pattern, each shape (28, 28)")
    else:
        try:
            mask, pattern = _load_trigger(trigger_path)
        except ValueError as exc:
            errors.append(str(exc))
            mask = pattern = None
        if mask is not None and obj.get("verdict") == "clean":
            if np.any(mask != 0) or np.any(pattern != 0):
                errors.append("a clean verdict requires mask and pattern to be all zeros")
    if obj.get("edit") == "no_edit" and delta_path.exists():
        errors.append('edit is "no_edit" but /work/delta.npz exists; remove it')
    if obj.get("edit") == "delta":
        if not delta_path.is_file():
            errors.append("missing /work/delta.npz")
        else:
            try:
                _load_delta(delta_path)
            except ValueError as exc:
                errors.append(str(exc))
    return errors


def _load_trigger(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if set(archive.files) != {"mask", "pattern"}:
            raise ValueError("trigger.npz must contain exactly the arrays mask and pattern")
        mask = np.array(archive["mask"], dtype=np.float32, copy=True)
        pattern = np.array(archive["pattern"], dtype=np.float32, copy=True)
    for name, array in (("mask", mask), ("pattern", pattern)):
        if array.shape != (28, 28):
            raise ValueError(f"trigger {name} must have shape (28, 28)")
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError(f"trigger {name} must be finite and in [0, 1]")
    return mask, pattern


def _load_delta(path: Path) -> dict[str, np.ndarray]:
    expected = {name: shape for name, shape in TENSOR_SPEC}
    with np.load(path) as archive:
        unknown = [name for name in archive.files if name not in expected]
        if unknown:
            raise ValueError(f"delta.npz has unknown tensors {unknown}")
        out = {}
        for name in archive.files:
            array = np.array(archive[name], dtype=np.float32, copy=True)
            if tuple(array.shape) != expected[name]:
                raise ValueError(f"delta {name} has shape {array.shape}, expected {expected[name]}")
            if not np.isfinite(array).all():
                raise ValueError(f"delta {name} has a non-finite value")
            out[name] = array
    return out
