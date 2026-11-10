"""Clean accuracy and attack success. Used only after predictions are frozen."""
from __future__ import annotations

import copy

import numpy as np
import torch

from cnn_phase1.model import TinyCNN


def load_checkpoint(path, device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = TinyCNN()
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def paste(images: torch.Tensor, mask: np.ndarray, pattern: np.ndarray) -> torch.Tensor:
    mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).view(1, 1, 28, 28)
    pattern_t = torch.from_numpy(np.asarray(pattern, dtype=np.float32)).view(1, 1, 28, 28)
    return ((1.0 - mask_t) * images + mask_t * pattern_t).clamp(0, 1)


def predict(model: TinyCNN, images: torch.Tensor, device, batch: int = 256) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch):
            outs.append(model(images[start : start + batch].to(device)).argmax(1).cpu())
    return torch.cat(outs) if outs else torch.zeros(0, dtype=torch.long)


def accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return float((pred == labels).to(dtype=torch.float64).mean())


def attack_success(pred: torch.Tensor, labels: torch.Tensor, target: int) -> float:
    keep = labels != int(target)
    if int(keep.sum()) == 0:
        return float("nan")
    return float((pred[keep] == int(target)).to(dtype=torch.float64).mean())


def with_delta(model: TinyCNN, delta: dict[str, np.ndarray], device) -> TinyCNN:
    edited = copy.deepcopy(model)
    params = dict(edited.named_parameters())
    with torch.no_grad():
        for name, array in delta.items():
            if name not in params:
                raise ValueError(f"unknown parameter {name}")
            change = torch.from_numpy(np.ascontiguousarray(array)).to(params[name].device)
            if tuple(change.shape) != tuple(params[name].shape):
                raise ValueError(f"{name} delta shape {tuple(change.shape)} != {tuple(params[name].shape)}")
            params[name].add_(change)
    edited.eval()
    return edited


def score_model(model, images, labels, mask, pattern, target: int, device) -> tuple[float, float]:
    clean_pred = predict(model, images, device)
    triggered = paste(images, mask, pattern)
    trig_pred = predict(model, triggered, device)
    return accuracy(clean_pred, labels), attack_success(trig_pred, labels, target)
