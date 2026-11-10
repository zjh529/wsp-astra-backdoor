"""Trigger overlap and the fc1 columns whose receptive field meets it.

Two stride-2 pools and two kernel-3 padded convolutions. A 7×7 cell after
the second pool depends on input coordinates from pooled_span(index).
fc1 is flattened as channel-major (32, 7, 7), which is PyTorch's contiguous
view of that map.
"""
from __future__ import annotations

import numpy as np


def pooled_span(index: int, limit: int = 28) -> tuple[int, int]:
    pool1_lo = 2 * index - 1
    pool1_hi = 2 * index + 2
    lo = 2 * pool1_lo - 1
    hi = 2 * pool1_hi + 2
    return max(0, lo), min(limit - 1, hi)


def overlapping_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells = []
    binary = np.asarray(mask) > 0
    for row in range(7):
        row_lo, row_hi = pooled_span(row)
        for col in range(7):
            col_lo, col_hi = pooled_span(col)
            if np.any(binary[row_lo : row_hi + 1, col_lo : col_hi + 1]):
                cells.append((row, col))
    return cells


def fc1_input_columns(mask: np.ndarray, channels: int = 32) -> set[int]:
    columns = set()
    for row, col in overlapping_cells(mask):
        for channel in range(channels):
            columns.add(channel * 49 + row * 7 + col)
    return columns


def iou(predicted: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> float:
    pred = np.asarray(predicted) >= threshold
    gt = np.asarray(truth) >= threshold
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return inter / union


def weight_scores(suspicious: list[dict], gt_mask: np.ndarray) -> dict:
    columns = fc1_input_columns(gt_mask)
    reported: set[int] = set()
    for item in suspicious:
        if item.get("tensor") != "fc1.weight":
            continue
        for index in item.get("flat_indices") or []:
            reported.add(int(index) % 1568)
    if not reported:
        precision = None
    else:
        precision = len(reported & columns) / len(reported)
    recall = (len(reported & columns) / len(columns)) if columns else None
    return {
        "precision": precision,
        "recall": recall,
        "n_reported_columns": len(reported),
        "n_gt_columns": len(columns),
    }
